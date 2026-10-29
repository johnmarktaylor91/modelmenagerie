"""C3-B: live parks stay disclosed at close, and grant voids never lie.

Two defects the C3-B valve/park/grant rehearsal triggered on current main.

**Park disclosure (the livelock half).** ``driver._run_locked``'s completion arm
derives ``review_holds`` / ``refusal_parks`` -- and therefore the whole
``complete`` vs ``terminal-partition-complete-*`` decision -- from the RUN-SCOPED
deferral stash. The stash holds exactly ONE row per model, and on a resume the
row is whichever durable valve row ``_still_valid_deferral_row`` returns FIRST:
capacity, then coverage, then rebuild, then disk floor, then the grant park, then
the refusal park. A model carrying BOTH an ordinary valve row and a live park row
for the same work generation therefore closes ``complete-with-deferrals`` while
an operator decision is still owed -- exactly the ``complete*``-with-a-live-park
state W3 and V2 both forbid. Park liveness is now derived from the durable
ledgers (fail-closed, additive: it can only ever move a close AWAY from
``complete``), so which row wins the spend short-circuit cannot change what the
close discloses.

**Void semantics (the double-spend half).** ``build_canonical_requeue_void``
accepted a void for an ALREADY-CONSUMED grant, and the driver's consumption arm
never consults voids when deciding which generation is ``active``. The append
therefore succeeded, printed a receipt, and changed nothing: the granted work
generation stayed bound and its attempts allowance stayed spent. The tool's own
``--void`` help already promised "an existing unconsumed grant"; issuance now
refuses typed, names the consumption, and names the real lever.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from menagerie.crawler import driver as driver_module
from menagerie.crawler import driver_admission
from menagerie.crawler.checker_lane import (
    CHECKER_LANE_DEFERRAL_DISPOSITION,
    append_checker_lane_deferral_row,
    build_checker_lane_deferral_row,
    checker_lane_deferral_path,
)
from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    append_canonical_requeue_void,
    build_canonical_requeue_grant,
    canonical_intake_snapshot_roots,
    build_canonical_requeue_void,
    canonical_operational_ledger_path,
    canonical_requeue_grants_path,
)
from menagerie.crawler.constants import (
    DEFERRED_TERMINAL_GRANT_STAGE,
    FAILURE_REASON_CODES,
    OperationalEventKind,
)
from menagerie.crawler.disk_floor import (
    DISK_FLOOR_DEFERRAL_DISPOSITION,
    append_disk_floor_deferral_row,
    build_disk_floor_deferral_row,
    disk_floor_deferral_path,
)
from menagerie.crawler.driver import _live_park_rows
from menagerie.crawler.driver_admission import (
    _deferral_row_fact_still_holds,
    _grant_stage_binds_terminal,
    _persisted_deferral_rows,
    _still_valid_deferral_row,
)
from menagerie.crawler.environment_coverage import (
    COVERAGE_DEFERRAL_DISPOSITION,
    coverage_deferral_path,
)
from menagerie.crawler.environment_rebuild import (
    REBUILD_DEFERRAL_DISPOSITION,
    append_rebuild_deferral_row,
    build_rebuild_deferral_row,
    rebuild_deferral_path,
)
from menagerie.crawler.grant_parks import (
    AWARD_PARK_DISPOSITION,
    GRANT_PENDING_DISPOSITION,
    append_grant_park_row,
    build_grant_park_row,
    grant_park_path,
)
from menagerie.crawler.host_capacity import (
    CAPACITY_DEFERRAL_DISPOSITION,
    HOST_MEMORY_ENV_VAR,
    capacity_deferral_path,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_DISPOSITION,
    append_refusal_park_row,
    build_refusal_park_row,
    refusal_park_path,
)
from menagerie.crawler.scheduler_parks import (
    SCHEDULER_QUARANTINE_DISPOSITION,
    append_scheduler_quarantine_row,
    build_scheduler_quarantine_row,
    scheduler_quarantine_path,
)
from menagerie.crawler.status import (
    MACHINE_DEFERRAL_DISPOSITIONS,
    machine_deferral_rejection,
)
from menagerie.crawler.tests.test_partition_deferral_accounting import (
    _worker_observed_rss_row,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    _driver,
    _paths,
    _snapshot,
    _write_jsonl,
)

REVISION = "sha256:" + "a" * 64
NEWER_REVISION = "sha256:" + "b" * 64

#: The five ordinary valve dispositions that can co-reside with a park row for
#: the same work generation. Every one of them precedes both park ledgers in
#: `_persisted_deferral_rows`' fixed order, so every one of them can win the
#: reclaim and mask the park.
ORDINARY_DISPOSITIONS = (
    CAPACITY_DEFERRAL_DISPOSITION,
    COVERAGE_DEFERRAL_DISPOSITION,
    REBUILD_DEFERRAL_DISPOSITION,
    DISK_FLOOR_DEFERRAL_DISPOSITION,
    CHECKER_LANE_DEFERRAL_DISPOSITION,
)


def _grant_park(stable_id: str = "m_held", work_id: str = "work-m_held") -> dict:
    """Build one valid grant-pending park row blocked on REVISION."""

    return build_grant_park_row(
        stable_id=stable_id,
        work_id=work_id,
        name="Held",
        campaign_id="c3b-mech",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-12T00:00:00Z",
        blocked_parent_revision=REVISION,
        blocked_stage="accuracy-gate",
        pending_model={
            "stable_id": stable_id,
            "status": {"code": "runs", "reason_code": None, "stage": None},
        },
    )


def _refusal_park(
    stable_id: str = "m_refused",
    work_id: str = "work-m_refused",
    *,
    blocked_parent_revision: str | None = None,
    campaign_id: str = "c3b-mech",
    run_id: str = "run-1",
) -> dict:
    """Build one valid terminal-refusal park row."""

    return build_refusal_park_row(
        stable_id=stable_id,
        work_id=work_id,
        name="Refused",
        campaign_id=campaign_id,
        run_id=run_id,
        machine_id="machine-1",
        created_at="2026-08-12T00:00:00Z",
        refusal_class="intended-and-fallback-refused",
        record_error_type="menagerie.crawler.reducer.ReductionError",
        record_error="terminal proof is unavailable",
        fallback_error_type="menagerie.crawler.reducer.ReductionError",
        fallback_error="minimal rung refused too",
        intended_status_code="failed:import",
        intended_reason_code="import-error",
        blocked_parent_revision=blocked_parent_revision,
        candidate_sha256=None,
    )


def _ordinary_row(disposition: str, stable_id: str, work_id: str) -> dict[str, Any]:
    """Build one minimally valid ordinary valve row for a disposition.

    Only the fields the disclosure derivation reads (identity, disposition,
    remedy, digest) matter here: the point of the pin is that NO ordinary row,
    whatever its family, can hide a live park.
    """

    if disposition == CAPACITY_DEFERRAL_DISPOSITION:
        row = _worker_observed_rss_row(stable_id)
        row.pop("row_sha256")
        row["work_id"] = work_id
        row["row_sha256"] = stable_hash(row)
        return row
    row = {
        "schema_version": f"menagerie.crawler.mech-{disposition}.v1",
        "stable_id": stable_id,
        "work_id": work_id,
        "name": "Held",
        "campaign_id": "c3b-mech",
        "run_id": "run-1",
        "machine_id": "machine-1",
        "created_at": "2026-08-12T00:00:00Z",
        "disposition": disposition,
        "recheck_hint": "mechanical co-resident row for the disclosure pin",
    }
    row["row_sha256"] = stable_hash(row)
    return row


# ---------------------------------------------------------------------------
# park disclosure -- the close can never read complete* behind a live park


@pytest.mark.smoke
def test_a_live_grant_park_is_disclosed_when_another_valve_row_wins() -> None:
    """An ordinary valve row for the same work generation cannot hide the park."""

    park = _grant_park()
    ordinary = _ordinary_row(CAPACITY_DEFERRAL_DISPOSITION, "m_held", "work-m_held")
    live = _live_park_rows(
        {"m_held": (ordinary, park)},
        frozenset({"m_held"}),
        {},
    )
    assert set(live[GRANT_PENDING_DISPOSITION]) == {"m_held"}
    assert live[GRANT_PENDING_DISPOSITION]["m_held"]["row_sha256"] == park["row_sha256"]
    assert live[REFUSAL_PARK_DISPOSITION] == {}


@pytest.mark.smoke
def test_a_live_refusal_park_is_disclosed_when_another_valve_row_wins() -> None:
    """The refusal park is the same class and gets the same guarantee."""

    park = _refusal_park()
    ordinary = _ordinary_row(CHECKER_LANE_DEFERRAL_DISPOSITION, "m_refused", "work-m_refused")
    live = _live_park_rows(
        {"m_refused": (ordinary, park)},
        frozenset({"m_refused"}),
        {},
    )
    assert set(live[REFUSAL_PARK_DISPOSITION]) == {"m_refused"}
    assert live[GRANT_PENDING_DISPOSITION] == {}


@pytest.mark.smoke
@pytest.mark.parametrize("ordinary_disposition", ORDINARY_DISPOSITIONS)
@pytest.mark.parametrize("park_disposition", (GRANT_PENDING_DISPOSITION, REFUSAL_PARK_DISPOSITION))
def test_every_disposition_pair_still_discloses_the_live_park(
    ordinary_disposition: str, park_disposition: str
) -> None:
    """The livelock hunt: no (ordinary valve, park) pair loses the park.

    Ten pairs, both row orders. Each is a state a resume can reach, because a
    park fires at award time on a work generation an earlier leg may already
    have parked for a host/environment/checker reason.
    """

    stable_id = "m_pair"
    work_id = "work-m_pair"
    park = (
        _grant_park(stable_id, work_id)
        if park_disposition == GRANT_PENDING_DISPOSITION
        else _refusal_park(stable_id, work_id)
    )
    ordinary = _ordinary_row(ordinary_disposition, stable_id, work_id)
    for rows in ((ordinary, park), (park, ordinary)):
        live = _live_park_rows({stable_id: rows}, frozenset({stable_id}), {})
        assert set(live[park_disposition]) == {stable_id}, (
            f"{ordinary_disposition} hid a live {park_disposition} in order"
            f" {[row['disposition'] for row in rows]}"
        )


@pytest.mark.smoke
def test_a_released_park_is_never_disclosed() -> None:
    """Only OWED decisions force the distinct close; a consumed park is silent."""

    park = _grant_park()
    refusal = _refusal_park(blocked_parent_revision=REVISION)
    granted = {"stable_id": "m_held", "record_revision": NEWER_REVISION}
    redriven = {"stable_id": "m_refused", "record_revision": NEWER_REVISION}
    live = _live_park_rows(
        {"m_held": (park,), "m_refused": (refusal,)},
        frozenset({"m_held", "m_refused"}),
        {"m_held": granted, "m_refused": redriven},
    )
    assert live[GRANT_PENDING_DISPOSITION] == {}
    assert live[REFUSAL_PARK_DISPOSITION] == {}
    # ...while the SAME blocked revision still governing keeps the park live.
    still_blocked = _live_park_rows(
        {"m_held": (park,)},
        frozenset({"m_held"}),
        {"m_held": {"stable_id": "m_held", "record_revision": REVISION}},
    )
    assert set(still_blocked[GRANT_PENDING_DISPOSITION]) == {"m_held"}


@pytest.mark.smoke
def test_out_of_scope_and_tampered_park_rows_are_never_disclosed() -> None:
    """Scoping and fail-closed validation both hold for the disclosure read."""

    from menagerie.crawler.grant_parks import SELECTION_EXCLUSION_DISPOSITION

    empty = {
        GRANT_PENDING_DISPOSITION: {},
        AWARD_PARK_DISPOSITION: {},
        SELECTION_EXCLUSION_DISPOSITION: {},
        REFUSAL_PARK_DISPOSITION: {},
    }
    park = _grant_park()
    assert _live_park_rows({"m_held": (park,)}, frozenset({"m_other"}), {}) == empty
    tampered = dict(park)
    tampered["pending"] = {**park["pending"], "blocked_stage": "import"}
    assert _live_park_rows({"m_held": (tampered,)}, frozenset({"m_held"}), {}) == empty


@pytest.mark.smoke
def test_the_completion_arm_derives_park_liveness_from_the_durable_ledgers() -> None:
    """Source-binding: the close cannot go back to trusting the run-scoped stash.

    Deleting the ledger-derived read reinstates the exact masking this module
    pins, so the binding is pinned in the anti-substitution style.
    """

    source = inspect.getsource(driver_module.CrawlerDriver._run_locked)
    # D-17 moved the ledger-derived read into the ONE shared derivation the
    # CLI boundary gate consumes too; the close must call it, and it must
    # still be the durable read this pin was born to protect.
    assert "durable_readiness_accounting(" in source
    shared = inspect.getsource(driver_module.durable_readiness_accounting)
    assert "_live_park_rows(" in shared
    assert "_persisted_deferral_rows(driver)" in shared
    # The state blocks must be built from the disclosure rows, never from a
    # stash lookup that can silently drop a park id it has no row for.
    assert "scoped_deferrals[stable_id]\n" not in source.split("review_holds")[-1]


@pytest.mark.heavy
def test_a_reclaimed_valve_row_never_closes_complete_behind_a_live_park(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Driver-level: the masked close, end to end, on the real completion arm.

    Model A carries a capacity row whose refusal still holds (so the resume
    short-circuit reclaims it and the stash holds the CAPACITY row) plus a live
    parent-less refusal park. Before the fix the close read
    ``complete-with-deferrals`` with no ``refusal_parks`` block at all.
    """

    monkeypatch.setenv(HOST_MEMORY_ENV_VAR, str(8 * 2**30))
    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    held_id = snapshot.items[0].stable_id
    records_root = paths.ledgers.models.parent.parent
    work_id = f"work-{held_id}"
    # The withheld member's rows must be honest ledger provenance under
    # close-v2: the proof's provenance term drops foreign-campaign rows, its
    # withholding arm cites only rows minted inside a leg window by that
    # generation's own runs, and the driver's later leg carries a headless
    # member only from a resident ancestor close naming those exact rows. So
    # a prior fixture generation of this same rung mints the rows in-window
    # and closes honestly; the driver leg then reclaims and carries them.
    from menagerie.crawler.tests.close_object_support import (
        open_fixture_leg,
        record_fixture_close,
    )

    open_fixture_leg(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id=snapshot.snapshot_id,
        intake_snapshot_sha256=snapshot.snapshot_sha256,
    )
    capacity_path = capacity_deferral_path(records_root)
    capacity_path.parent.mkdir(parents=True, exist_ok=True)
    capacity_row = _ordinary_row(CAPACITY_DEFERRAL_DISPOSITION, held_id, work_id)
    capacity_row.pop("row_sha256")
    capacity_row.update(campaign_id="c1-mech", run_id="pytest-close")
    capacity_row["row_sha256"] = stable_hash(capacity_row)
    capacity_path.write_bytes(canonical_json_bytes(capacity_row) + b"\n")
    append_refusal_park_row(
        refusal_park_path(records_root),
        _refusal_park(held_id, work_id, campaign_id="c1-mech", run_id="pytest-close"),
    )
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id=snapshot.snapshot_id,
        intake_snapshot_sha256=snapshot.snapshot_sha256,
        member_ids=[held_id],
        completion_status="terminal-partition-complete-with-deferrals-with-refusal-parks",
        # This fixture's rows are dated 2026-08-12; the close that counts
        # them must not PREDATE them (D-49 refuses a post-dated counted row).
        closed_at="2026-08-12T00:00:00Z",
    )

    result = _driver(tmp_path, snapshot, paths_override=paths).run()

    assert "complete" in result.status
    assert result.status.endswith("-with-refusal-parks"), result.status
    state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    assert state["status"] == result.status
    parked = state["refusal_parks"]
    assert parked["count"] == 1
    assert parked["pending"][0]["stable_id"] == held_id
    # The capacity claim is still reported honestly beside the park -- the model
    # really is withheld for BOTH reasons.
    assert held_id in state["machine_deferrals"]["by_disposition"][CAPACITY_DEFERRAL_DISPOSITION]


# ---------------------------------------------------------------------------
# no review-held head may be unreachable by every expressible grant


def _review_held_head(status_code: str, stage: str | None) -> dict[str, Any]:
    """Return a review-held lineage head in the exact ledger shape."""

    return {
        "stable_id": "m_held",
        "record_revision": REVISION,
        "parent_revision": None,
        "record_seq": 1,
        "status": {
            "kind": status_code.split(":", 1)[0],
            "code": status_code,
            "stage": stage,
            "human_review": {
                "queue": "crawler-human-review",
                "reason": None,
                "requested_at": "2026-08-12T00:00:00Z",
                "required": True,
            },
        },
        "budget": {"explicit_grants": []},
    }


@pytest.mark.smoke
@pytest.mark.parametrize(
    "status_code",
    ("deferred:needs-cuda", "skipped:not-a-neural-net", "runs"),
)
def test_the_sentinel_binds_every_null_stage_review_held_head(status_code: str) -> None:
    """No null-stage review-held head may be unreachable by every grant.

    ``driver_models`` derives ``stage = code.split(":",1)[1] if kind == "failed"
    else None``, so a null stage means kind ``deferred``, ``skipped``, or ``runs``
    -- and ALL THREE are minted review-held by the ``_terminalize`` containment
    ladder, which re-appends the ORIGINAL status code with ``human_review=True``
    (that is exactly how the live pilot's m4256 got a review-held
    ``deferred:needs-cuda`` head). Keyed on ``kind == "deferred"``, the other two
    classes had NO expressible grant at all: ordinary stages mismatch null, and
    the sentinel refused the kind. That is the stuck state the axis forbids.
    """

    head = _review_held_head(status_code, None)
    assert head["status"]["stage"] is None
    assert _grant_stage_binds_terminal({"stage": DEFERRED_TERMINAL_GRANT_STAGE}, head)
    # ...and no ordinary failure stage may launder into a null-stage head.
    for stage in ("accuracy-gate", "import", "runner", "author", "resource"):
        assert not _grant_stage_binds_terminal({"stage": stage}, head)


@pytest.mark.smoke
def test_the_sentinel_still_never_binds_a_staged_head() -> None:
    """The narrowness that matters is unchanged in both directions."""

    for stage in sorted(FAILURE_REASON_CODES):
        head = _review_held_head(f"failed:{stage}", stage)
        assert not _grant_stage_binds_terminal({"stage": DEFERRED_TERMINAL_GRANT_STAGE}, head)
        assert _grant_stage_binds_terminal({"stage": stage}, head)


@pytest.mark.smoke
def test_a_null_stage_park_hint_names_a_grant_the_tool_accepts(tmp_path: Path) -> None:
    """The park's own remedy must be an executable command, not `--stage 'none'`.

    ``_park_grant_pending_supersession`` recorded the literal string ``"none"``
    for a blocked head with no stage, and ``build_grant_park_row`` prints
    ``requeue <id> --stage <blocked_stage>`` into the operator-facing
    ``recheck_hint``. ``"none"`` is in neither ``FAILURE_REASON_CODES`` nor the
    sentinel, so the hint sent the operator to a command that could only refuse.
    """

    source = inspect.getsource(driver_admission._park_grant_pending_supersession)
    assert "DEFERRED_TERMINAL_GRANT_STAGE" in source
    assert 'else "none"' not in source

    row = build_grant_park_row(
        stable_id="m_held",
        work_id="work-m_held",
        name="Held",
        campaign_id="c3b-mech",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-12T00:00:00Z",
        blocked_parent_revision=REVISION,
        blocked_stage=DEFERRED_TERMINAL_GRANT_STAGE,
        pending_model={
            "stable_id": "m_held",
            "status": {"code": "runs", "reason_code": None, "stage": None},
        },
    )
    assert f"--stage {DEFERRED_TERMINAL_GRANT_STAGE!r}" in row["recheck_hint"]
    # The named stage is one the issuance path actually accepts.
    records = tmp_path / "records"
    snapshot = _snapshot_dir(tmp_path, records / "intake", ("Held",))
    path = records / "operational" / "requeue-grants.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    grant = build_canonical_requeue_grant(
        path,
        stable_id=snapshot.items[0].stable_id,
        stage=row["pending"]["blocked_stage"],
        reason="release the park the hint described",
        attempts=1,
        granted_by="operator",
    )
    assert grant["stage"] == DEFERRED_TERMINAL_GRANT_STAGE


# ---------------------------------------------------------------------------
# the durable valve read must cover EVERY accounted disposition


def _seed_every_valve_ledger(records_root: Path) -> dict[str, str]:
    """Write one real, validated row per accounted valve ledger.

    Returns the disposition -> stable ID map the read is expected to reproduce.
    Coverage and capacity use the committed rung-10 pilot fixtures (real rows
    from the leg that produced them); the rest use their production builders.
    """

    fixtures = Path(__file__).resolve().parent
    seeded: dict[str, str] = {}

    coverage_row = json.loads(
        (fixtures / "rung10_environment_coverage_deferrals.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    coverage_path = coverage_deferral_path(records_root)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_bytes(canonical_json_bytes(coverage_row) + b"\n")
    seeded[COVERAGE_DEFERRAL_DISPOSITION] = str(coverage_row["stable_id"])

    capacity_row = json.loads(
        (fixtures / "rung10_host_capacity_deferrals.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    capacity_deferral_path(records_root).write_bytes(canonical_json_bytes(capacity_row) + b"\n")
    seeded[CAPACITY_DEFERRAL_DISPOSITION] = str(capacity_row["stable_id"])

    append_rebuild_deferral_row(
        rebuild_deferral_path(records_root),
        build_rebuild_deferral_row(
            stable_id="m_rebuild",
            work_id="work-m_rebuild",
            name="Rebuild",
            campaign_id="c3b-mech",
            run_id="run-1",
            machine_id="machine-1",
            created_at="2026-08-12T00:00:00Z",
            quarantine_event_id="environment-cleanup-quarantined-1",
            intent="graph",
            target="osx-arm64",
            artifact_identity="sha256:" + "c" * 64,
            artifact_state={"lock": {"state": "present", "sha256": "sha256:" + "d" * 64}},
            env_generation=None,
            failure_type="EnvironmentBuildError",
        ),
    )
    seeded[REBUILD_DEFERRAL_DISPOSITION] = "m_rebuild"

    append_disk_floor_deferral_row(
        disk_floor_deferral_path(records_root),
        build_disk_floor_deferral_row(
            stable_id="m_disk",
            work_id="work-m_disk",
            name="Disk",
            campaign_id="c3b-mech",
            run_id="run-1",
            machine_id="machine-1",
            created_at="2026-08-12T00:00:00Z",
            intent="core",
            target="osx-arm64",
            env_root="/nonexistent/env-store",
            floor_bytes=32 * 2**30,
            free_bytes=2 * 2**30,
        ),
    )
    seeded[DISK_FLOOR_DEFERRAL_DISPOSITION] = "m_disk"

    append_checker_lane_deferral_row(
        checker_lane_deferral_path(records_root),
        build_checker_lane_deferral_row(
            stable_id="m_checker",
            work_id="work-m_checker",
            name="Checker",
            campaign_id="c3b-mech",
            run_id="run-1",
            machine_id="machine-1",
            created_at="2026-08-12T00:00:00Z",
            gate_kind="metadata_batch",
            failure_class="permanent-contract-rejection",
            failure_evidence="checker returned no schema-valid verdict for the member",
            attempted_batch_sizes=[8, 4, 2, 1],
        ),
    )
    seeded[CHECKER_LANE_DEFERRAL_DISPOSITION] = "m_checker"

    append_grant_park_row(grant_park_path(records_root), _grant_park())
    seeded[GRANT_PENDING_DISPOSITION] = "m_held"
    # AW-1: the award-supersession park shares the grant-park ledger and must
    # never be write-only to the durable read either.
    append_grant_park_row(
        grant_park_path(records_root),
        build_grant_park_row(
            stable_id="m_award",
            work_id="work-m_award",
            name="Award",
            campaign_id="c3b-mech",
            run_id="run-1",
            machine_id="machine-1",
            created_at="2026-08-12T00:00:00Z",
            blocked_parent_revision=REVISION,
            blocked_stage="deferred-terminal",
            pending_model={
                "stable_id": "m_award",
                "status": {"code": "failed:source", "reason_code": None, "stage": "source"},
            },
            disposition=AWARD_PARK_DISPOSITION,
        ),
    )
    seeded[AWARD_PARK_DISPOSITION] = "m_award"
    # R2-1: the selection-exclusion park shares the same ledger and joined the
    # closed disposition set with its durable read in the same reviewed change.
    from menagerie.crawler.grant_parks import (
        SELECTION_EXCLUSION_DISPOSITION,
        build_selection_exclusion_park_row,
    )

    append_grant_park_row(
        grant_park_path(records_root),
        build_selection_exclusion_park_row(
            stable_id="m_excluded",
            name="Excluded",
            campaign_id="c3b-mech",
            machine_id="machine-1",
            created_at="2026-08-12T00:00:00Z",
            held_revision=REVISION,
            held_status_code="failed:author",
            decision="PD-7-slop-family-routes",
            reason="frozen decision exclusion",
        ),
    )
    seeded[SELECTION_EXCLUSION_DISPOSITION] = "m_excluded"
    append_refusal_park_row(refusal_park_path(records_root), _refusal_park())
    seeded[REFUSAL_PARK_DISPOSITION] = "m_refused"
    # BL-8 (a)/(b): the scheduler-fault quarantine joined the closed
    # disposition set in the same reviewed change that gave it this durable
    # read, so it can never be a write-only ninth valve.
    append_scheduler_quarantine_row(
        scheduler_quarantine_path(records_root),
        build_scheduler_quarantine_row(
            stable_id="m_sched",
            work_id="work-m_sched",
            name="Sched",
            campaign_id="c3b-mech",
            run_id="run-1",
            machine_id="machine-1",
            created_at="2026-08-12T00:00:00Z",
            site="scheduler-round",
            reason_code="member-fault",
            fault_evidence="synthetic member fault escaped the seam's containment arms",
            exception_type="RuntimeError",
        ),
    )
    seeded[SCHEDULER_QUARANTINE_DISPOSITION] = "m_sched"
    # W1: the quota-deferral valve joined the closed disposition set in the
    # same reviewed change that gave it this durable read; the PD-3 recidivist
    # park shares its ledger and must not be write-only either.
    from menagerie.crawler.quota_deferral import (
        QUOTA_DEFERRAL_DISPOSITION,
        QUOTA_DEFERRAL_MAX_GRANTS,
        STORM_RECIDIVIST_DISPOSITION,
        mint_quota_deferral,
        quota_deferral_path,
    )

    def _storm(stable_id: str, run_id: str) -> None:
        mint_quota_deferral(
            quota_deferral_path(records_root),
            stable_id=stable_id,
            work_id=f"work-{stable_id}",
            campaign_id="c3b-mech",
            run_id=run_id,
            machine_id="machine-1",
            created_at="2026-08-12T00:00:00Z",
            evidence={"witness": "author-sessions-exhaustion-classifier"},
        )

    # The recidivist member's own earlier valve rows read back too, so it
    # seeds FIRST and the pure valve member's row lands last in file order.
    for ordinal in range(QUOTA_DEFERRAL_MAX_GRANTS + 1):
        _storm("m_recidivist", f"run-{ordinal}")
    seeded[STORM_RECIDIVIST_DISPOSITION] = "m_recidivist"
    _storm("m_storm", "run-1")
    seeded[QUOTA_DEFERRAL_DISPOSITION] = "m_storm"
    return seeded


@pytest.mark.smoke
def test_every_accounted_valve_disposition_has_a_durable_reader(tmp_path: Path) -> None:
    """The checker-lane ledger was WRITE-ONLY to the driver's durable read.

    ``_persisted_deferral_rows`` read five valve ledgers plus the two parks and
    silently omitted ``intake-extensions/checker-lane-deferrals.jsonl`` -- even
    though the disposition is in the closed ``MACHINE_DEFERRAL_DISPOSITIONS`` set,
    ``crawl status --verify-partition`` loads it, and the module's own contract
    says the partition "dereferences these rows exactly like the admission-valve
    rows". Two consequences: a superseded checker-lane row could never appear in
    the close's ``superseded_rows`` (the exact rung-10 disclosure gap), and
    ``merge_host_shards`` -- which derives shard OWNERSHIP from this same read --
    saw a checker-lane-parked model as owned by nobody and refused the merge.

    Pinned as a TOTALITY over the closed disposition set so a seventh valve
    cannot be added write-only again.
    """

    records_root = tmp_path / "records"
    (records_root / "models").mkdir(parents=True, exist_ok=True)
    seeded = _seed_every_valve_ledger(records_root)
    assert set(seeded) == set(MACHINE_DEFERRAL_DISPOSITIONS)

    driver = SimpleNamespace(
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=records_root / "models" / "current-shard.jsonl")
        )
    )
    governing = _persisted_deferral_rows(driver)
    read_back = {
        str(row["disposition"]): stable_id for stable_id, rows in governing.items() for row in rows
    }
    assert read_back == seeded
    # Every row read back also satisfies the partition's fail-closed contract.
    for stable_id, rows in governing.items():
        for row in rows:
            assert machine_deferral_rejection(stable_id, row) is None


@pytest.mark.smoke
def test_a_checker_lane_row_still_never_reclaims_a_model(tmp_path: Path) -> None:
    """Reading the ledger must not start short-circuiting author spend.

    The checker-lane fact ("the lane failed to deliver a verdict") can only be
    re-tested by asking the lane again, so it has no still-holds arm and the model
    is authored normally next leg. Including the ledger in the durable read is a
    DISCLOSURE and OWNERSHIP change only.
    """

    records_root = tmp_path / "records"
    (records_root / "models").mkdir(parents=True, exist_ok=True)
    _seed_every_valve_ledger(records_root)
    driver = SimpleNamespace(
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=records_root / "models" / "current-shard.jsonl")
        ),
        config=SimpleNamespace(machine_id="machine-1"),
        registry=SimpleNamespace(intents={}),
    )
    row = next(
        row
        for rows in _persisted_deferral_rows(driver).values()
        for row in rows
        if row["disposition"] == CHECKER_LANE_DEFERRAL_DISPOSITION
    )
    item = SimpleNamespace(
        stable_id="m_checker",
        active_work_id="work-m_checker",
        route=SimpleNamespace(intent="core"),
    )
    assert _deferral_row_fact_still_holds(driver, item, row) is False
    assert _still_valid_deferral_row(driver, item) is None


# ---------------------------------------------------------------------------
# grant issuance -- a staging LABEL directory is not a corrupt snapshot


def _snapshot_dir(tmp_path: Path, output_root: Path, names: tuple[str, ...]) -> Any:
    """Create one real intake snapshot under ``output_root``."""

    master = tmp_path / f"master-{output_root.name}-{names[0]}.jsonl"
    deferred = tmp_path / f"deferred-{output_root.name}-{names[0]}.jsonl"
    _write_jsonl(master, [{"name": name, "zoo": "fixtures", "variant": "base"} for name in names])
    _write_jsonl(deferred, [])
    return create_intake_snapshot(master, deferred, output_root)


@pytest.mark.smoke
def test_a_staging_label_directory_never_kills_grant_issuance(tmp_path: Path) -> None:
    """The LIVE pilot blocker: one label directory refused EVERY operator grant.

    ``records/intake`` grew ``intake_head50/intake-<digest>/`` on 2026-08-11 (the
    ladder staging layout). ``build_canonical_requeue_grant`` handed every
    immediate child of ``records/intake`` to ``load_intake_snapshot``, so the
    LABEL directory -- which has no ``manifest.json`` of its own -- raised
    ``canonical intake snapshot is invalid`` and the campaign's only re-drive
    lever was dead for every model at once. The nested snapshot's members must be
    found, not fatal.
    """

    records = tmp_path / "records"
    flat = _snapshot_dir(tmp_path, records / "intake", ("Flat",))
    nested = _snapshot_dir(tmp_path, records / "intake" / "intake_head50", ("Nested",))
    assert flat.root.parent == records / "intake"
    assert nested.root.parent == records / "intake" / "intake_head50"
    path = records / "operational" / "requeue-grants.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    for stable_id in (flat.items[0].stable_id, nested.items[0].stable_id):
        grant = build_canonical_requeue_grant(
            path,
            stable_id=stable_id,
            stage="accuracy-gate",
            reason="operator approved re-adjudication",
            attempts=1,
            granted_by="operator",
        )
        append_canonical_requeue_grant(path, grant)
        assert grant["stable_id"] == stable_id
    # The membership check it guards is still armed for a genuine outsider.
    with pytest.raises(ValueError, match="absent from canonical intake snapshots"):
        build_canonical_requeue_grant(
            path,
            stable_id="m_outsider",
            stage="accuracy-gate",
            reason="operator approved re-adjudication",
            attempts=1,
            granted_by="operator",
        )


@pytest.mark.smoke
def test_a_genuinely_corrupt_snapshot_still_refuses_issuance(tmp_path: Path) -> None:
    """Integrity unchanged: a directory that IS a snapshot must still load."""

    records = tmp_path / "records"
    snapshot = _snapshot_dir(tmp_path, records / "intake", ("Flat",))
    (snapshot.root / "manifest.json").write_text("{not json", encoding="utf-8")
    path = records / "operational" / "requeue-grants.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="canonical intake snapshot is invalid"):
        build_canonical_requeue_grant(
            path,
            stable_id=snapshot.items[0].stable_id,
            stage="accuracy-gate",
            reason="operator approved re-adjudication",
            attempts=1,
            granted_by="operator",
        )


@pytest.mark.smoke
def test_snapshot_discovery_stops_below_the_declared_depth(tmp_path: Path) -> None:
    """Descent is bounded: a deeper layout is a change worth noticing, not absorbed."""

    records = tmp_path / "records"
    _snapshot_dir(tmp_path, records / "intake" / "a" / "b", ("Deep",))
    assert canonical_intake_snapshot_roots(records / "intake") == ()
    assert canonical_intake_snapshot_roots(records / "intake" / "missing") == ()


# ---------------------------------------------------------------------------
# grant void semantics -- a void that changes nothing must not be accepted


def _grants_ledger(tmp_path: Path) -> Path:
    """Return a canonical grant ledger path under a records/operational root."""

    path = tmp_path / "records" / "operational" / "requeue-grants.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _issue(path: Path, stable_id: str = "m_grant") -> Mapping[str, Any]:
    """Issue and durably append one canonical requeue grant."""

    grant = build_canonical_requeue_grant(
        path,
        stable_id=stable_id,
        stage="accuracy-gate",
        reason="operator approved re-adjudication",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(path, grant)
    return grant


def _record_consumption(path: Path, grant: Mapping[str, Any]) -> None:
    """Append the exact consumption event shape the driver writes."""

    events = path.parent / "events.jsonl"
    with events.open("ab") as handle:
        handle.write(
            canonical_json_bytes(
                {
                    "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
                    "details": {
                        "grant_id": str(grant["grant_id"]),
                        "stable_id": str(grant["stable_id"]),
                        "new_work_generation": 1,
                    },
                }
            )
            + b"\n"
        )


@pytest.mark.smoke
def test_voiding_a_consumed_grant_refuses_and_names_the_consumption(
    tmp_path: Path,
) -> None:
    """A void that the driver would ignore is refused, not printed as a receipt.

    The consumption already bound a work generation and already spent its
    attempts allowance; ``_consume_requeue_grants`` reads voids only when
    choosing among UNCONSUMED grants, so accepting this append told the operator
    a revocation happened when nothing did.
    """

    path = _grants_ledger(tmp_path)
    grant = _issue(path)
    _record_consumption(path, grant)

    with pytest.raises(ValueError) as excinfo:
        build_canonical_requeue_void(
            path,
            void_of=str(grant["grant_id"]),
            reason="issued against the wrong stage",
            voided_by="operator",
        )
    message = str(excinfo.value)
    assert str(grant["grant_id"]) in message
    assert "already consumed" in message
    # Nothing was appended: the ledger still holds exactly the one grant.
    assert len([row for row in scan_jsonl(path, validate=False)]) == 1


@pytest.mark.smoke
def test_voiding_an_unconsumed_grant_still_appends(tmp_path: Path) -> None:
    """Control: the sanctioned void-before-reissue lane is untouched."""

    path = _grants_ledger(tmp_path)
    grant = _issue(path)
    void = build_canonical_requeue_void(
        path,
        void_of=str(grant["grant_id"]),
        reason="wrong stage; re-issuing",
        voided_by="operator",
    )
    assert append_canonical_requeue_void(path, void) is True
    assert void["void_of"] == str(grant["grant_id"])
    # ...and a sibling model's consumption never blocks this model's void.
    other = _issue(path, stable_id="m_other")
    _record_consumption(path, other)
    reissued = build_canonical_requeue_grant(
        path,
        stable_id="m_grant",
        stage="accuracy-gate",
        reason="operator approved re-adjudication",
        attempts=1,
        granted_by="operator",
    )
    assert reissued["grant_id"] != str(grant["grant_id"])


@pytest.mark.smoke
def test_the_requeue_tool_help_matches_the_enforced_void_contract() -> None:
    """The tool promised "unconsumed"; the promise is now enforced upstream."""

    from menagerie.crawler.tools import requeue as requeue_tool

    action = next(item for item in requeue_tool.build_parser()._actions if item.dest == "void_of")
    assert "unconsumed" in str(action.help)
    assert "already consumed" in inspect.getsource(build_canonical_requeue_void)


@pytest.mark.smoke
def test_a_voided_then_consumed_grant_cannot_be_double_voided(tmp_path: Path) -> None:
    """Void-then-consume is a corruption shape, not a second voidable grant.

    A grant voided BEFORE consumption is never newly consumed, so a consumption
    event for it can only come from a ledger that already disagrees with itself.
    Re-voiding must stay idempotent rather than mint a second void row.
    """

    path = _grants_ledger(tmp_path)
    grant = _issue(path)
    void = build_canonical_requeue_void(
        path,
        void_of=str(grant["grant_id"]),
        reason="withdrawn",
        voided_by="operator",
    )
    append_canonical_requeue_void(path, void)
    assert append_canonical_requeue_void(path, void) is False
    _record_consumption(path, grant)
    # The already-voided grant refuses a SECOND void for the consumption reason
    # rather than silently appending a duplicate decision.
    with pytest.raises(ValueError):
        build_canonical_requeue_void(
            path,
            void_of=str(grant["grant_id"]),
            reason="withdrawn again",
            voided_by="operator",
        )


@pytest.mark.smoke
def test_consumption_lookup_reads_the_canonical_events_beside_the_ledger(
    tmp_path: Path,
) -> None:
    """The void's consumption evidence is the same ledger issuance already reads.

    ``_outstanding_logical_twin`` (T1-13b) and the void refusal must consult ONE
    consumption source, or the two operator levers disagree about whether a
    grant is spent.
    """

    path = _grants_ledger(tmp_path)
    assert (
        canonical_requeue_grants_path(tmp_path / "records" / "models" / "current-shard.jsonl")
        == path
    )
    assert canonical_operational_ledger_path(
        tmp_path / "records" / "models" / "current-shard.jsonl"
    ) == (path.parent / "events.jsonl")
    grant = _issue(path)
    _record_consumption(path, grant)
    # Issuance sees the consumption (a consumed twin re-issues legally)...
    twin = build_canonical_requeue_grant(
        path,
        stable_id="m_grant",
        stage="accuracy-gate",
        reason="operator approved re-adjudication",
        attempts=1,
        granted_by="operator",
    )
    assert twin["grant_id"] != str(grant["grant_id"])
    # ...and the void sees the SAME consumption.
    with pytest.raises(ValueError, match="already consumed"):
        build_canonical_requeue_void(
            path,
            void_of=str(grant["grant_id"]),
            reason="withdrawn",
            voided_by="operator",
        )
