"""Orphan-citation adoption: the closed rule and the full-leg remedy.

2026-08-27, pilot ordinary leg 2: eight out-of-band review-hold appends --
built through ``terminal_review_hold_candidate`` under a context whose run
never joined any generation, and inheriting their superseded heads' retired
``crawler-run`` citation -- landed inside the open generation's window, and
``derive_partition_proof`` refused the whole leg at admission ("position is
not provenance").  The tripwire is correct and stays untouched; the campaign
was still unresumable, because the sweep never re-visits already-held heads
and admission refuses before any close could supersede them.

These tests pin the lawful remedy:

* the hold candidate builder now REQUIRES the appending run's citation, so a
  production sweep hold can never inherit a stale citation again;
* ``orphan_hold_adoption_refusal`` is the closed, replay-derived eligibility
  rule -- a review-held failure terminal byte-equal to the production
  containment marking over sealed pre-window history adopts, everything else
  refuses typed;
* ``orphan_adoption_drift`` refuses a re-append that changed anything beyond
  parentage and the re-citation; and
* end to end through real driver legs: the incident shape is adopted by the
  next leg under its own joined run (census-disclosed, leg completes), while
  a tampered out-of-band row keeps refusing at the partition proof with its
  refusal disclosed on the same census.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.checkpoint import canonical_records_root
from menagerie.crawler.driver import CommandAuthorLane
from menagerie.crawler.driver_contracts import DriverConfig, DriverPaths, IntakeSnapshot
from menagerie.crawler.grant_parks import (
    append_grant_park_row,
    build_grant_park_row,
    grant_park_path,
    grant_park_released,
    load_grant_park_rows,
)
from menagerie.crawler.partition_proof import PartitionProofError
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import (
    CanonicalReducer,
    GrantPendingSupersessionError,
    OrphanAdoptionRefusal,
)
from menagerie.crawler.terminal_review import (
    ORPHAN_CITATION_ADOPTION_DISPOSITION,
    ORPHAN_CITATION_ADOPTION_MEMBER_DISPOSITION,
    orphan_adoption_drift,
    orphan_hold_adoption_refusal,
    terminal_review_hold_candidate,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    _BLOCKED_SOURCE_PAYLOAD,
    _driver,
    _install_executor_lane,
    _paths,
    _snapshot,
    _test_authority_context,
)

# ---------------------------------------------------------------------------
# Pure halves: the closed adoption rule
# ---------------------------------------------------------------------------


def _parent(stable_id: str = "m1") -> dict[str, Any]:
    """One settled failure-terminal ledger head in the pre-era citation shape."""

    return {
        "stable_id": stable_id,
        "record_seq": 536,
        "record_revision": f"sha256:parent-{stable_id}",
        "parent_revision": None,
        "status": {
            "kind": "failed",
            "code": "failed:environment",
            "stage": "environment",
            "attempt_ids": [f"sha256:attempt-{stable_id}"],
            "supersedes_revision": None,
            "human_review": {
                "required": False,
                "reason": None,
                "queue": None,
                "requested_at": None,
            },
        },
        "provenance": {
            "author_model": "claude-fable-5-1",
            "checker_model": "gpt-5.6-sol",
            "producer_run_id": "crawler-run",
            "machine_id": "legacy-machine",
        },
        "dependency_vector": {
            # Snapshot/item pins: provenance, never rotation -- adoption
            # adjudicates them byte-exact (Opus C1).
            "intake_snapshot_id": f"intake-{stable_id}",
            "intake_snapshot_sha256": f"sha256:intake-{stable_id}",
            "intake_item_sha256": f"sha256:item-{stable_id}",
            "runner_identity": "sha256:runner-old",
            "terminal_rule_identity": "sha256:terminal-old",
        },
        "budget": {"explicit_grants": []},
        "notes": "",
    }


def _orphan(parent: Mapping[str, Any]) -> dict[str, Any]:
    """The incident row: the containment hold inheriting the parent citation."""

    row = terminal_review_hold_candidate(
        parent,
        requested_at="2026-08-27T10:30:00Z",
        # The incident mechanism, reproduced deliberately: the out-of-band
        # append carried the PARENT's citation, naming a run that never
        # joined the open generation.
        producer_run_id=str(parent["provenance"]["producer_run_id"]),
        machine_id=str(parent["provenance"]["machine_id"]),
    )
    # Ledger assignment, plus the reducer-derived vector re-derivation the
    # real append performed: the rotatable identity axes moved between the
    # two appends, while the snapshot/item pins stayed verbatim (exactly the
    # real pilot rows' head-vs-parent diff shape).
    row["record_seq"] = 567
    row["record_revision"] = "sha256:orphan-m1"
    vector = dict(parent["dependency_vector"])
    vector.update(
        {
            "runner_identity": "sha256:runner-new",
            "terminal_rule_identity": "sha256:terminal-new",
        }
    )
    row["dependency_vector"] = vector
    return row


@pytest.mark.smoke
def test_the_incident_shape_is_adoptable() -> None:
    parent = _parent()
    head = _orphan(parent)
    assert (
        orphan_hold_adoption_refusal(head, parent, parent_row_index=535, window_floor=566)
        is None
    )


@pytest.mark.smoke
def test_an_in_window_parent_refuses_the_forged_two_row_chain() -> None:
    # A forged parent appended alongside the hold sits INSIDE the window, so
    # adoption can never launder content through a fabricated ancestry.
    parent = _parent()
    head = _orphan(parent)
    reason = orphan_hold_adoption_refusal(head, parent, parent_row_index=570, window_floor=566)
    assert reason is not None and "sealed pre-generation history" in reason


@pytest.mark.smoke
def test_content_beyond_the_containment_marking_refuses_by_path() -> None:
    parent = _parent()
    head = _orphan(parent)
    head["notes"] = "laundered claim"
    reason = orphan_hold_adoption_refusal(head, parent, parent_row_index=535, window_floor=566)
    assert reason is not None and "notes" in reason


@pytest.mark.smoke
def test_a_recited_provenance_refuses() -> None:
    # An out-of-band row that re-stamped its own citation differs from its
    # parent outside the containment marking; only the LEG re-cites.
    parent = _parent()
    head = _orphan(parent)
    head["provenance"]["producer_run_id"] = "run-forged"
    reason = orphan_hold_adoption_refusal(head, parent, parent_row_index=535, window_floor=566)
    assert reason is not None and "provenance.producer_run_id" in reason


@pytest.mark.smoke
def test_an_unheld_head_belongs_to_the_sweep_not_adoption() -> None:
    parent = _parent()
    head = _orphan(parent)
    head["status"]["human_review"]["required"] = False
    reason = orphan_hold_adoption_refusal(head, parent, parent_row_index=535, window_floor=566)
    assert reason is not None and "not review-held" in reason


@pytest.mark.smoke
def test_a_malformed_hold_block_refuses() -> None:
    parent = _parent()
    for field, value, fragment in (
        ("reason", "operator says so", "reason"),
        ("queue", "some-other-queue", "queue"),
        ("requested_at", None, "requested_at"),
    ):
        head = _orphan(parent)
        head["status"]["human_review"][field] = value
        reason = orphan_hold_adoption_refusal(
            head, parent, parent_row_index=535, window_floor=566
        )
        assert reason is not None and fragment in reason


@pytest.mark.smoke
def test_a_non_failed_kind_refuses() -> None:
    parent = _parent()
    head = _orphan(parent)
    head["status"]["kind"] = "runs"
    reason = orphan_hold_adoption_refusal(head, parent, parent_row_index=535, window_floor=566)
    assert reason is not None and "'failed'" in reason


@pytest.mark.smoke
def test_unbound_parentage_refuses() -> None:
    parent = _parent()
    head = _orphan(parent)
    head["parent_revision"] = "sha256:someone-else"
    reason = orphan_hold_adoption_refusal(head, parent, parent_row_index=535, window_floor=566)
    assert reason is not None and "parent_revision" in reason
    head = _orphan(parent)
    head["status"]["supersedes_revision"] = "sha256:someone-else"
    reason = orphan_hold_adoption_refusal(head, parent, parent_row_index=535, window_floor=566)
    assert reason is not None and "supersedes_revision" in reason


def _prepared(head: Mapping[str, Any]) -> dict[str, Any]:
    """The adopting leg's re-append after ``prepare_model`` re-derivation."""

    prepared = {
        key: deepcopy(value)
        for key, value in head.items()
        if key not in {"record_seq", "record_revision"}
    }
    prepared["parent_revision"] = head["record_revision"]
    prepared["status"]["supersedes_revision"] = head["record_revision"]
    prepared["provenance"]["producer_run_id"] = "run-adopting"
    prepared["provenance"]["machine_id"] = "adopting-machine"
    # prepare_model re-derives the vector from live context: the rotatable
    # identity axes may move; the pins re-derive to the leg's ACTIVE values,
    # which for an adoptable member already equal the recorded ones.
    prepared["dependency_vector"] = dict(head["dependency_vector"])
    prepared["dependency_vector"].update(
        {
            "runner_identity": "sha256:runner-newer",
            "terminal_rule_identity": "sha256:terminal-newer",
        }
    )
    return prepared


@pytest.mark.smoke
def test_adoption_drift_allows_exactly_recitation_and_vector_rederivation() -> None:
    parent = _parent()
    head = _orphan(parent)
    prepared = _prepared(head)
    assert orphan_adoption_drift(prepared, head) is None
    prepared["status"]["code"] = "failed:import"
    drift = orphan_adoption_drift(prepared, head)
    assert drift is not None and "status.code" in drift


@pytest.mark.smoke
def test_a_snapshot_or_item_pin_move_refuses_adoption() -> None:
    # Opus C1 (repro_vector.py shape): the out-of-band append ran with a
    # DIFFERENT active intake, so its live prepare_model stamped NEW
    # snapshot/item pins while the sealed parent carries the OLD ones. Three
    # authorities read those pins (the proof's recorded_under refusal, the
    # register door's recorded-history proof, the close's exact-pin
    # staleness); the blanket dependency_vector exclusion let adoption
    # re-stamp them with zero adjudication. The enumerated rotatable-axis
    # law refuses by pin path instead.
    parent = _parent()
    head = _orphan(parent)
    head["dependency_vector"].update(
        {
            "intake_snapshot_id": "snapshot-NEW",
            "intake_snapshot_sha256": "sha256:intake-NEW",
            "intake_item_sha256": "sha256:item-NEW",
        }
    )
    reason = orphan_hold_adoption_refusal(head, parent, parent_row_index=535, window_floor=566)
    assert reason is not None and "dependency_vector.intake_snapshot_id" in reason


@pytest.mark.smoke
def test_adoption_drift_refuses_a_pin_move_after_prepare() -> None:
    # The drift half of Opus C1: prepare_model stamps the adopting leg's
    # ACTIVE pins, so an orphan row recorded under a different snapshot
    # refuses typed instead of silently moving its recorded snapshot,
    # un-staling its item pin, and rebinding the close's stale-excluded
    # cohort. Old work keeps old provenance or stays refused.
    parent = _parent()
    head = _orphan(parent)
    prepared = _prepared(head)
    prepared["dependency_vector"].update(
        {
            "intake_snapshot_id": "intake-current",
            "intake_item_sha256": "sha256:item-current",
        }
    )
    drift = orphan_adoption_drift(prepared, head)
    assert drift is not None and "dependency_vector.intake_snapshot_id" in drift


@pytest.mark.smoke
def test_an_evidence_binding_move_refuses_adoption() -> None:
    # Evidence bindings (attempt ids, artifact authority, recipe and
    # representative revisions) are per-work provenance, not rotation: a
    # move refuses on both halves of the rule.
    parent = _parent()
    head = _orphan(parent)
    head["dependency_vector"]["accepted_attempt_ids"] = ["sha256:forged-attempt"]
    reason = orphan_hold_adoption_refusal(head, parent, parent_row_index=535, window_floor=566)
    assert reason is not None and "accepted_attempt_ids" in reason
    head = _orphan(parent)
    prepared = _prepared(head)
    prepared["dependency_vector"]["artifact_transaction_id"] = "sha256:other-txn"
    drift = orphan_adoption_drift(prepared, head)
    assert drift is not None and "artifact_transaction_id" in drift


# ---------------------------------------------------------------------------
# End to end: real driver legs over a records tree with an orphan citation
# ---------------------------------------------------------------------------


def _leg(tmp_path: Path, snapshot: IntakeSnapshot, run_id: str, **kwargs: Any):
    """One fixture driver re-cited to its own run identity.

    The shared fixture pins ``run_id="run-test"`` for every leg, which is the
    PRE-ERA world (one shared constant, every citation trivially joined).
    Reproducing the incident requires the D-22/DX-23 era: each invocation
    carries its own minted identity, so a citation inherited across
    generations is orphaned.
    """

    import dataclasses

    driver = _driver(tmp_path, snapshot, **kwargs)
    driver.config = dataclasses.replace(driver.config, run_id=run_id)
    return driver


def _open_generation_with_settled_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[IntakeSnapshot, str, DriverPaths]:
    """Leg 1 settles a failure terminal; leg 2 opens a generation and pauses.

    Reproduces the pilot's standing state: a settled review-free failed head
    below the open generation's floor, with the generation still OPEN when
    the out-of-band append lands (so the appended row sits inside its
    window and inherits a citation from a run that never joined it).
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    _install_executor_lane(monkeypatch, _BLOCKED_SOURCE_PAYLOAD)
    first = _leg(
        tmp_path, snapshot, "run-leg1", author=CommandAuthorLane(("fake-author",))
    ).run()
    assert first.status == "complete"
    settled = [
        record
        for record in scan_jsonl(paths.ledgers.models)
        if record["stable_id"] == stable_id
    ]
    assert settled[-1]["status"]["kind"] == "failed"
    paused = _leg(tmp_path, snapshot, "run-leg2-paused", review_at=1).run()
    assert paused.status == "paused:review-checkpoint"
    return snapshot, stable_id, paths


def _append_out_of_band_hold(
    snapshot: IntakeSnapshot,
    paths: DriverPaths,
    *,
    tamper: bool = False,
) -> Mapping[str, Any]:
    """The incident script: a reducer append outside any joined run."""

    context = _test_authority_context(snapshot, DriverConfig())
    with CanonicalReducer(paths.ledgers, context) as reducer:
        head = dict(reducer.raw_head_view[snapshot.items[0].stable_id])
        candidate = terminal_review_hold_candidate(
            head,
            requested_at="2026-08-27T10:30:00Z",
            producer_run_id=str(head["provenance"]["producer_run_id"]),
            machine_id=str(head["provenance"]["machine_id"]),
        )
        if tamper:
            candidate["notes"] = "laundered content beyond the marking"
        return reducer.append_model(reducer.prepare_model(candidate)).record


def _adoption_census_events(paths: DriverPaths) -> list[Mapping[str, Any]]:
    return [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition")
        == ORPHAN_CITATION_ADOPTION_DISPOSITION
    ]


def _member_events(paths: DriverPaths) -> list[Mapping[str, Any]]:
    return [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition")
        == ORPHAN_CITATION_ADOPTION_MEMBER_DISPOSITION
    ]


def test_the_next_leg_adopts_the_incident_row_under_its_own_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CC D-24 justified (essential cohesion): one linear e2e narrative
    # (settle -> orphan -> adopt -> replay) whose assertions cross-check the
    # census, member disclosure, and ledger rows the SAME legs produced;
    # splitting it would sever those checks from their producing story.
    snapshot, stable_id, paths = _open_generation_with_settled_head(tmp_path, monkeypatch)
    orphan = _append_out_of_band_hold(snapshot, paths)
    assert orphan["provenance"]["producer_run_id"] == "run-leg1"
    resumed = _leg(tmp_path, snapshot, "run-leg3-adopting")
    result = resumed.run()
    # The whole partition is settled terminals, so the leg completes through
    # the terminal-partition family; the point is that it COMPLETES (the
    # admission and close proofs both passed) instead of refusing.
    assert result.status.startswith("terminal-partition-complete")

    censuses = _adoption_census_events(paths)
    assert len(censuses) == 1
    details = censuses[0]["details"]
    assert details["eligible"] == 1
    assert details["adopted"]["count"] == 1
    assert details["unadoptable"]["count"] == 0
    assert details["beyond_bound"]["count"] == 0
    member = details["adopted"]["members"][0]
    assert member["stable_id"] == stable_id
    assert member["cited_run_id"] == orphan["provenance"]["producer_run_id"]
    # The crash-durable per-member disclosure landed too (before the append;
    # the crash ordering itself is pinned in the dying-leg test below).
    member_disclosures = _member_events(paths)
    assert len(member_disclosures) == 1
    assert member_disclosures[0]["details"]["stable_id"] == stable_id
    assert member_disclosures[0]["details"]["orphan_revision"] == orphan["record_revision"]

    revisions = [
        record
        for record in scan_jsonl(paths.ledgers.models)
        if record["stable_id"] == stable_id
    ]
    adopted = revisions[-1]
    # Head-continuity: the adoption supersedes the orphan row verbatim under
    # the adopting leg's own joined run; the orphan row stays in history.
    assert adopted["parent_revision"] == orphan["record_revision"]
    assert adopted["status"]["supersedes_revision"] == orphan["record_revision"]
    assert adopted["provenance"]["producer_run_id"] == resumed.config.run_id
    assert adopted["status"]["code"] == orphan["status"]["code"]
    assert adopted["status"]["human_review"]["required"] is True
    assert adopted["record_revision"] == member["adopted_revision"]
    assert any(
        record["record_revision"] == orphan["record_revision"] for record in revisions
    )

    # The adopted row must also survive SEMANTIC REPLAY: every later leg
    # re-opens the reducer over the full ledger, and a replay refusal of the
    # adoption supersession (the review-hold grant tripwire re-derives on
    # replay too) would re-brick the campaign one leg later. A fourth leg
    # replays everything, finds no orphans (the head now cites a joined run
    # of ITS generation only if re-orphaned -- it is below the next floor),
    # appends no second census, and completes.
    follow_up = _leg(tmp_path, snapshot, "run-leg4-replaying")
    assert follow_up.run().status.startswith("terminal-partition-complete")
    assert len(_adoption_census_events(paths)) == 1


def test_a_tampered_out_of_band_row_keeps_refusing_and_is_disclosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, stable_id, paths = _open_generation_with_settled_head(tmp_path, monkeypatch)
    _append_out_of_band_hold(snapshot, paths, tamper=True)
    with pytest.raises(PartitionProofError) as refusal:
        _leg(tmp_path, snapshot, "run-leg3-refusing").run()
    assert "position is not provenance" in str(refusal.value)

    censuses = _adoption_census_events(paths)
    assert len(censuses) == 1
    details = censuses[0]["details"]
    assert details["adopted"]["count"] == 0
    assert details["unadoptable"]["count"] == 1
    member = details["unadoptable"]["members"][0]
    assert member["stable_id"] == stable_id
    assert "notes" in member["reason"]


def test_a_citation_unchanged_verbatim_reappend_keeps_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Opus C2a (repro_carve.py shape): a verbatim re-append whose citation
    # does NOT move is a no-op supersession of operator property. Admitting
    # it grant-free would advance the held head's revision, and
    # grant_park_released reads any revision advance as "the granted
    # supersession landed" -- a live grant-pending park would be silently
    # consumed. The reducer carve requires the citation to actually MOVE.
    snapshot, stable_id, paths = _open_generation_with_settled_head(tmp_path, monkeypatch)
    _append_out_of_band_hold(snapshot, paths)
    context = _test_authority_context(snapshot, DriverConfig())
    with CanonicalReducer(paths.ledgers, context) as reducer:
        held = dict(reducer.raw_head_view[stable_id])
        assert held["status"]["human_review"]["required"] is True
        candidate = {
            key: deepcopy(value)
            for key, value in held.items()
            if key not in {"record_seq", "record_revision", "ledger_seq", "payload_sha256"}
        }
        candidate["parent_revision"] = held["record_revision"]
        candidate["status"]["supersedes_revision"] = held["record_revision"]
        with pytest.raises(GrantPendingSupersessionError):
            reducer.append_model(reducer.prepare_model(candidate))


def test_a_live_park_named_head_refuses_adoption_and_is_disclosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Opus C2b, the pilot's live-park shape (m3353 `author`, m8590
    # `accuracy-gate` x2): LIVE grant-pending park rows name review-held
    # heads. When such a head is also orphan-cited, adoption must refuse --
    # its re-append would advance the head revision and grant_park_released
    # would read every park CONSUMED, dropping the pending operator decision
    # from the live park census.
    snapshot, stable_id, paths = _open_generation_with_settled_head(tmp_path, monkeypatch)
    orphan = _append_out_of_band_hold(snapshot, paths)
    park_path = grant_park_path(canonical_records_root(paths.ledgers.models))
    for work_id, stage in (("work-review-a", "author"), ("work-review-b", "accuracy-gate")):
        append_grant_park_row(
            park_path,
            build_grant_park_row(
                stable_id=stable_id,
                work_id=work_id,
                name=stable_id,
                campaign_id="campaign-park-fixture",
                run_id="run-parking",
                machine_id="machine-parking",
                created_at="2026-08-27T11:00:00Z",
                blocked_parent_revision=str(orphan["record_revision"]),
                blocked_stage=stage,
                pending_model={
                    "stable_id": stable_id,
                    "status": {"code": str(orphan["status"]["code"]), "reason_code": None},
                },
            ),
        )
    with pytest.raises(PartitionProofError):
        _leg(tmp_path, snapshot, "run-leg3-parked").run()

    censuses = _adoption_census_events(paths)
    assert len(censuses) == 1
    details = censuses[0]["details"]
    assert details["adopted"]["count"] == 0
    assert details["unadoptable"]["count"] == 1
    member = details["unadoptable"]["members"][0]
    assert member["stable_id"] == stable_id
    assert "live park" in member["reason"]
    assert _member_events(paths) == []
    # Both parks stayed LIVE (the head revision never advanced), and the
    # models ledger holds no adoption row.
    head = [
        record
        for record in scan_jsonl(paths.ledgers.models)
        if record["stable_id"] == stable_id
    ][-1]
    assert head["record_revision"] == orphan["record_revision"]
    live = [
        row
        for row in load_grant_park_rows([park_path])
        if not grant_park_released(row, head)
    ]
    assert len(live) == 2


def test_a_moved_citation_reappend_of_a_live_parked_head_refuses_at_the_reducer_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sol C2 round-3 KILL counterexample, verbatim: a moved citation is not
    # evidence that the closed adoption rule ran. Before this fix, a DIRECT
    # CanonicalReducer caller (the incident's out-of-band actor class) could
    # take a review-held head named by a LIVE grant-pending park, re-append
    # it verbatim under a different already-joined citation, and advance the
    # head grant-free -- grant_park_released then read the park CONSUMED and
    # the next leg never entered adoption for that member. The carve now
    # re-derives park liveness from the park ledger itself at the append
    # door, so the refusal binds every reducer caller, not just the driver.
    snapshot, stable_id, paths = _open_generation_with_settled_head(tmp_path, monkeypatch)
    held = _append_out_of_band_hold(snapshot, paths)
    park_path = grant_park_path(canonical_records_root(paths.ledgers.models))
    park = append_grant_park_row(
        park_path,
        build_grant_park_row(
            stable_id=stable_id,
            work_id="work-live-park",
            name=stable_id,
            campaign_id="campaign-fixture",
            run_id="run-parking",
            machine_id="machine-parking",
            created_at="2026-07-14T11:59:00Z",
            blocked_parent_revision=str(held["record_revision"]),
            blocked_stage=str(held["status"]["stage"]),
            pending_model={
                "stable_id": stable_id,
                "status": {"code": held["status"]["code"], "reason_code": None},
            },
        ),
    )
    context = _test_authority_context(snapshot, DriverConfig())
    with CanonicalReducer(paths.ledgers, context) as reducer:
        head = dict(reducer.raw_head_view[stable_id])
        candidate = terminal_review_hold_candidate(
            head,
            requested_at=str(head["status"]["human_review"]["requested_at"]),
            # This run already joined the open generation in the fixture.
            producer_run_id="run-leg2-paused",
            machine_id="machine-one-shot",
        )
        with pytest.raises(GrantPendingSupersessionError):
            reducer.append_model(reducer.prepare_model(candidate))
    # The head never advanced, so the park stays LIVE: the pending operator
    # decision survives for the driver fence to disclose on the next leg.
    final = [
        record
        for record in scan_jsonl(paths.ledgers.models)
        if record["stable_id"] == stable_id
    ][-1]
    assert final["record_revision"] == held["record_revision"]
    assert not grant_park_released(park, final)
    with pytest.raises(PartitionProofError):
        _leg(tmp_path, snapshot, "run-leg3-after-refusal").run()
    details = _adoption_census_events(paths)[-1]["details"]
    assert details["adopted"]["count"] == 0
    assert details["unadoptable"]["count"] == 1
    assert "live park" in details["unadoptable"]["members"][0]["reason"]


def test_members_beyond_the_per_leg_bound_are_disclosed_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Opus C3: the adoption burst is bounded per leg and the beyond-bound
    # remainder is disclosed, never silently skipped; the next leg adopts it
    # (monotonic progress under the bound).
    import menagerie.crawler.driver as driver_module

    snapshot, stable_id, paths = _open_generation_with_settled_head(tmp_path, monkeypatch)
    _append_out_of_band_hold(snapshot, paths)
    monkeypatch.setattr(driver_module, "ORPHAN_ADOPTION_PER_LEG", 0)
    with pytest.raises(PartitionProofError):
        _leg(tmp_path, snapshot, "run-leg3-bounded").run()
    details = _adoption_census_events(paths)[-1]["details"]
    assert details["bound"] == 0
    assert details["adopted"]["count"] == 0
    assert details["unadoptable"]["count"] == 0
    assert details["beyond_bound"]["count"] == 1
    assert details["beyond_bound"]["members"][0]["stable_id"] == stable_id
    assert _member_events(paths) == []

    monkeypatch.setattr(driver_module, "ORPHAN_ADOPTION_PER_LEG", 128)
    result = _leg(tmp_path, snapshot, "run-leg4-adopting").run()
    assert result.status.startswith("terminal-partition-complete")
    assert _adoption_census_events(paths)[-1]["details"]["adopted"]["count"] == 1


def test_a_leg_dying_before_the_append_never_loses_the_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Opus C5: the per-member census event is appended BEFORE the adoption
    # re-append it discloses. Kill the leg exactly between the two writes:
    # the disclosure is already durable, the models ledger is untouched, no
    # summary census fires -- and the next leg retries the still-orphaned
    # member and adopts it (a duplicate member event is coherent).
    from menagerie.crawler.driver_contracts import DriverPaused

    snapshot, stable_id, paths = _open_generation_with_settled_head(tmp_path, monkeypatch)
    orphan = _append_out_of_band_hold(snapshot, paths)
    original = CanonicalReducer.append_model

    def dying_append(self: CanonicalReducer, model: Mapping[str, Any]) -> Any:
        provenance = model.get("provenance")
        if (
            isinstance(provenance, Mapping)
            and provenance.get("producer_run_id") == "run-leg3-dying"
        ):
            raise DriverPaused("simulated death between disclosure and append")
        return original(self, model)

    monkeypatch.setattr(CanonicalReducer, "append_model", dying_append)
    try:
        outcome: Any = _leg(tmp_path, snapshot, "run-leg3-dying").run().status
    except Exception as exc:  # noqa: BLE001 -- either surface is a dead leg
        outcome = exc
    assert not str(outcome).startswith("terminal-partition-complete")

    # The disclosure survived the death; the adoption itself never landed.
    disclosures = _member_events(paths)
    assert len(disclosures) == 1
    assert disclosures[0]["details"]["stable_id"] == stable_id
    assert _adoption_census_events(paths) == []
    head = [
        record
        for record in scan_jsonl(paths.ledgers.models)
        if record["stable_id"] == stable_id
    ][-1]
    assert head["record_revision"] == orphan["record_revision"]

    monkeypatch.setattr(CanonicalReducer, "append_model", original)
    result = _leg(tmp_path, snapshot, "run-leg4-retrying").run()
    assert result.status.startswith("terminal-partition-complete")
    assert len(_member_events(paths)) == 2
    assert _adoption_census_events(paths)[-1]["details"]["adopted"]["count"] == 1


# ---------------------------------------------------------------------------
# P5-SOL-05: the write door itself. Direct ``append_model`` can never discharge
# the moved-citation carve; only ``append_orphan_adoption`` may, and it
# re-derives the ENTIRE closed rule from ledgers first. Reducer-level fixture
# (no driver legs): a settled failed head below an open generation's floor,
# the incident hold appended in-window with an inherited (never-joined)
# citation, and two runs durably joined to the open generation.
# ---------------------------------------------------------------------------

_REQ_AT = "2026-08-27T10:30:00Z"


def _reducer_orphan_state(tmp_path: Path) -> tuple[Any, Any, Mapping[str, Any]]:
    """Build the incident standing state directly at the reducer level."""

    from menagerie.crawler.leg_generation import open_leg_generation
    from menagerie.crawler.models import LedgerPaths
    from menagerie.crawler.tests.conftest import (
        bind_terminal_attempts,
        make_authority_context,
        make_failed_attempt,
        make_model,
    )

    records = tmp_path / "records"
    ledgers = LedgerPaths(
        models=records / "models" / "current-shard.jsonl",
        attempts=records / "attempts" / "local.jsonl",
        gates=records / "gates" / "current-shard.jsonl",
    )
    context = make_authority_context(["m_example"])
    attempt = make_failed_attempt()
    model = bind_terminal_attempts(make_model(status_code="failed:source"), [attempt])
    with CanonicalReducer(ledgers, context) as reducer:
        reducer.append_attempt(attempt)
        reducer.append_model(reducer.prepare_model(model))
    # Open one generation; both calls join their run to the SAME open epoch
    # (the second continues it), so the floor seals the settled row below.
    for run_id in ("run-adopting", "run-joined-sibling"):
        open_leg_generation(
            records,
            campaign_id="c1-mech",
            intake_snapshot_id=context.active_intake_snapshot_id,
            intake_snapshot_sha256=context.active_intake_snapshot_sha256,
            run_id=run_id,
            opened_at="2026-08-27T10:00:00Z",
        )
    # The incident script: an in-window hold inheriting the settled head's
    # citation ("run-test" never joined the generation). The settled head is
    # review-free, so this append never meets the grant tripwire.
    with CanonicalReducer(ledgers, context) as reducer:
        head = dict(reducer.raw_head_view["m_example"])
        candidate = terminal_review_hold_candidate(
            head,
            requested_at=_REQ_AT,
            producer_run_id=str(head["provenance"]["producer_run_id"]),
            machine_id=str(head["provenance"]["machine_id"]),
        )
        orphan = reducer.append_model(reducer.prepare_model(candidate)).record
    assert orphan["provenance"]["producer_run_id"] == "run-test"
    return ledgers, context, orphan


def _adoption_candidate(
    reducer: CanonicalReducer,
    head: Mapping[str, Any],
    *,
    producer_run_id: str = "run-adopting",
) -> dict[str, Any]:
    """One prepared moved-citation adoption re-append over ``head``."""

    candidate = terminal_review_hold_candidate(
        head,
        requested_at=str(head["status"]["human_review"]["requested_at"]),
        producer_run_id=producer_run_id,
        machine_id="machine-adopting",
    )
    return reducer.prepare_model(candidate)


def test_direct_append_of_a_drift_clean_moved_citation_hold_refuses_grantless(
    tmp_path: Path,
) -> None:
    # Sol's P5-SOL-05 probe, park-free: a moved citation plus drift-cleanliness
    # is NOT evidence that the closed adoption rule ran, so a DIRECT
    # ``append_model`` caller (the 2026-08-27 out-of-band actor class) meets
    # the grant tripwire even for the otherwise-lawful shape. Before this fix
    # the carve admitted the append grant-free with no live park present.
    ledgers, context, orphan = _reducer_orphan_state(tmp_path)
    with CanonicalReducer(ledgers, context) as reducer:
        prepared = _adoption_candidate(reducer, dict(reducer.raw_head_view["m_example"]))
        with pytest.raises(GrantPendingSupersessionError):
            reducer.append_model(dict(prepared))
        head = reducer.raw_head_view["m_example"]
        assert head["record_revision"] == orphan["record_revision"]


def test_the_adoption_api_appends_the_lawful_case_and_it_replays(
    tmp_path: Path,
) -> None:
    # The one live path through the carve: the write door re-derives the full
    # closed rule from ledgers, appends, and the persisted adopted row keeps
    # replaying at its recorded basis (era honesty) in a fresh reducer.
    ledgers, context, orphan = _reducer_orphan_state(tmp_path)
    with CanonicalReducer(ledgers, context) as reducer:
        prepared = _adoption_candidate(reducer, dict(reducer.raw_head_view["m_example"]))
        adopted = reducer.append_orphan_adoption(prepared).record
    assert adopted["parent_revision"] == orphan["record_revision"]
    assert adopted["status"]["supersedes_revision"] == orphan["record_revision"]
    assert adopted["provenance"]["producer_run_id"] == "run-adopting"
    assert adopted["status"]["code"] == orphan["status"]["code"]
    assert adopted["status"]["human_review"]["required"] is True
    # Era honesty: reducer construction semantically replays the whole ledger,
    # including the adopted supersession over its review-held parent.
    with CanonicalReducer(ledgers, context) as replayed:
        assert (
            replayed.raw_head_view["m_example"]["record_revision"]
            == adopted["record_revision"]
        )


def test_the_adoption_api_refuses_an_unjoined_appending_run_typed(
    tmp_path: Path,
) -> None:
    ledgers, context, _orphan = _reducer_orphan_state(tmp_path)
    with CanonicalReducer(ledgers, context) as reducer:
        prepared = _adoption_candidate(
            reducer,
            dict(reducer.raw_head_view["m_example"]),
            producer_run_id="run-never-joined",
        )
        with pytest.raises(OrphanAdoptionRefusal, match="joined 0 open"):
            reducer.append_orphan_adoption(prepared)


def test_the_adoption_api_refuses_a_citation_that_is_not_orphaned(
    tmp_path: Path,
) -> None:
    # The head's own citation names a JOINED run: nothing is orphaned, so the
    # API has nothing to adopt -- laundering a fresh supersession through the
    # adoption door is refused typed.
    ledgers, context, _orphan = _reducer_orphan_state(tmp_path)
    # Re-cite the held head to a joined run THROUGH THE API (the lawful move),
    # so the current head's citation now names a joined run.
    with CanonicalReducer(ledgers, context) as reducer:
        prepared = _adoption_candidate(
            reducer,
            dict(reducer.raw_head_view["m_example"]),
            producer_run_id="run-joined-sibling",
        )
        reducer.append_orphan_adoption(prepared)
        probe = _adoption_candidate(reducer, dict(reducer.raw_head_view["m_example"]))
        with pytest.raises(OrphanAdoptionRefusal, match="nothing is orphaned"):
            reducer.append_orphan_adoption(probe)


def test_the_adoption_api_refuses_an_in_window_parent_typed(tmp_path: Path) -> None:
    # Sol's probe shape exactly: the head's PARENT sits inside the open window
    # (a second incident-era hold over the first), so adoption would launder
    # unprovenanced two-row content. The API refuses on the closed rule, and
    # the direct append refuses grantless as always.
    ledgers, context, _orphan = _reducer_orphan_state(tmp_path)
    with CanonicalReducer(ledgers, context) as reducer:
        held = dict(reducer.raw_head_view["m_example"])
        hold2 = terminal_review_hold_candidate(
            held,
            requested_at=str(held["status"]["human_review"]["requested_at"]),
            producer_run_id="run-out-of-band-2",
            machine_id="machine-oob",
        )
        # Plant the incident-era second hold by simulating the pre-fix write
        # door (the flag is the module-private token the API sets); persisted
        # history of this shape must keep replaying, which reducer
        # construction below re-proves.
        reducer._orphan_adoption_admission = True
        try:
            second = reducer.append_model(reducer.prepare_model(hold2)).record
        finally:
            reducer._orphan_adoption_admission = False
    with CanonicalReducer(ledgers, context) as reducer:
        assert reducer.raw_head_view["m_example"]["record_revision"] == second[
            "record_revision"
        ]
        prepared = _adoption_candidate(reducer, dict(reducer.raw_head_view["m_example"]))
        with pytest.raises(GrantPendingSupersessionError):
            reducer.append_model(dict(prepared))
        with pytest.raises(OrphanAdoptionRefusal, match="sealed pre-generation history"):
            reducer.append_orphan_adoption(prepared)


def test_the_adoption_api_refuses_a_non_unique_parent_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A parent revision resolving to more than one row in the cited source
    # ledger is ambiguous ancestry; the API refuses before any lookup is
    # trusted. The doctored read is injected at the API's own ledger reader.
    import menagerie.crawler.reducer as reducer_module

    ledgers, context, _orphan = _reducer_orphan_state(tmp_path)
    real_scan = reducer_module.scan_jsonl

    def doubled_parent(path: Path, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        rows = list(real_scan(path, *args, **kwargs))
        if rows and path.name == "current-shard.jsonl" and "models" in str(path):
            rows.append(deepcopy(rows[0]))
        return rows

    with CanonicalReducer(ledgers, context) as reducer:
        prepared = _adoption_candidate(reducer, dict(reducer.raw_head_view["m_example"]))
        monkeypatch.setattr(reducer_module, "scan_jsonl", doubled_parent)
        with pytest.raises(OrphanAdoptionRefusal, match="resolves to 2 rows"):
            reducer.append_orphan_adoption(prepared)


def test_the_adoption_api_refuses_a_live_park_typed(tmp_path: Path) -> None:
    ledgers, context, orphan = _reducer_orphan_state(tmp_path)
    park_path = grant_park_path(canonical_records_root(ledgers.models))
    append_grant_park_row(
        park_path,
        build_grant_park_row(
            stable_id="m_example",
            work_id="work-live-park",
            name="m_example",
            campaign_id="campaign-fixture",
            run_id="run-parking",
            machine_id="machine-parking",
            created_at="2026-08-27T11:00:00Z",
            blocked_parent_revision=str(orphan["record_revision"]),
            blocked_stage=str(orphan["status"]["stage"]),
            pending_model={
                "stable_id": "m_example",
                "status": {"code": str(orphan["status"]["code"]), "reason_code": None},
            },
        ),
    )
    with CanonicalReducer(ledgers, context) as reducer:
        prepared = _adoption_candidate(reducer, dict(reducer.raw_head_view["m_example"]))
        with pytest.raises(OrphanAdoptionRefusal, match="live park"):
            reducer.append_orphan_adoption(prepared)
        assert (
            reducer.raw_head_view["m_example"]["record_revision"]
            == orphan["record_revision"]
        )
