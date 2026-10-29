"""Partition close PARKS grant-pending review-held supersessions.

Rung-12 census, JOB 4: legs 3 and 6 (42h of compute) died at the FINAL partition
assertion because a re-driven model's fresh terminal could not supersede its
review-held predecessor without a new explicit grant. The tripwire is CORRECT --
review-held records are operator property and silent supersession is the harm --
but the loud failure at close cost one operator grant round-trip plus one full
leg relaunch each time, and bred the 2026-08-10 workaround of PRE-GRANTING all
38 review-held models before launch. The adjudicated fix: a supersession blocked
ONLY by the missing grant parks as a durable, typed grant-pending row that keeps
the review-held terminal governing, closes the partition as
``terminal-partition-complete-with-review-holds``, and lets the NEXT leg consume
the grant and publish normally. No auto-granting; any supersession blocked for
any OTHER reason still fails loudly.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    canonical_operational_ledger_path,
    canonical_requeue_grants_path,
)
from menagerie.crawler.driver_admission import (
    _deferral_row_fact_still_holds,
    _persisted_deferral_rows,
)
from menagerie.crawler.grant_parks import (
    GRANT_PARK_EVENT_DISPOSITION,
    GRANT_PENDING_DISPOSITION,
    GrantParkError,
    append_grant_park_row,
    append_grant_park_void,
    build_grant_park_row,
    build_grant_park_void,
    grant_park_path,
    grant_park_released,
    load_grant_park_rows,
    validate_grant_park_row,
)
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import (
    GrantPendingSupersessionError,
    ReductionError,
)
from menagerie.crawler.status import (
    MACHINE_DEFERRAL_DISPOSITIONS,
    machine_deferral_rejection,
    partition_report,
)
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_failed_attempt,
    make_model,
)
from menagerie.crawler.tests.test_reducer import CanonicalReducer
from menagerie.crawler.tests.test_slice_f_driver import (
    ChangedInputAuthor,
    CheckerScript,
    FakeForward,
    RepairingIdentityAuthor,
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
    _write_jsonl,
)

#: Per-test tiering (T5-7 tier repair): the multi-second cells below moved to
#: heavy/slow so the smoke tier stays a sub-minute gate; everything else keeps
#: its smoke mark. All cells still run in the full crawler suite.

REVISION = "sha256:" + "a" * 64
OTHER_REVISION = "sha256:" + "b" * 64


def _park_row(**overrides: object) -> dict:
    """Build one valid grant-pending park row with optional field overrides."""

    values: dict = {
        "stable_id": "m_reviewed",
        "work_id": "work-m_reviewed",
        "name": "Reviewed",
        "campaign_id": "campaign-1",
        "run_id": "run-1",
        "machine_id": "machine-1",
        "created_at": "2026-08-11T00:00:00Z",
        "blocked_parent_revision": REVISION,
        "blocked_stage": "accuracy-gate",
        "pending_model": {
            "stable_id": "m_reviewed",
            "status": {"code": "runs", "reason_code": None, "stage": None},
        },
    }
    values.update(overrides)
    return build_grant_park_row(**values)  # type: ignore[arg-type]


@pytest.mark.smoke
def test_grant_park_row_roundtrips_and_names_the_remedy() -> None:
    """A built row validates, is partition-claimable, and names the grant lane."""

    row = _park_row()
    assert row["disposition"] == GRANT_PENDING_DISPOSITION
    assert GRANT_PENDING_DISPOSITION in MACHINE_DEFERRAL_DISPOSITIONS
    assert validate_grant_park_row(row) == row
    pending = row["pending"]
    assert pending["blocked_parent_revision"] == REVISION
    assert pending["blocked_stage"] == "accuracy-gate"
    assert pending["pending_status_code"] == "runs"
    assert pending["pending_model"]["stable_id"] == "m_reviewed"
    hint = row["recheck_hint"]
    assert "grant" in hint
    assert "review-held" in hint
    assert "accuracy-gate" in hint
    # No auto-granting: the remedy is an operator decision, never the machine's.
    assert "operator" in hint
    # The partition's fail-closed claim validation accepts the persisted row.
    assert machine_deferral_rejection("m_reviewed", row) is None
    assert machine_deferral_rejection("m_other", row) is not None


@pytest.mark.smoke
def test_grant_park_row_rejects_tampering_and_malformed_pending() -> None:
    """Digest, identity, and pending-append validation stay fail-closed."""

    row = _park_row()
    tampered = dict(row)
    tampered["name"] = "Renamed"
    with pytest.raises(GrantParkError, match="digest"):
        validate_grant_park_row(tampered)
    stripped = {key: value for key, value in row.items() if key != "pending"}
    with pytest.raises(GrantParkError, match="missing"):
        validate_grant_park_row(stripped)
    with pytest.raises(GrantParkError, match="stable_id"):
        _park_row(pending_model={"stable_id": "m_other", "status": {"code": "runs"}})
    with pytest.raises(GrantParkError, match="blocked_parent_revision"):
        _park_row(blocked_parent_revision="")
    with pytest.raises(GrantParkError, match="status"):
        _park_row(pending_model={"stable_id": "m_reviewed"})
    # A pending model that names a DIFFERENT parent is not this park's fact.
    with pytest.raises(GrantParkError, match="parent"):
        _park_row(
            pending_model={
                "stable_id": "m_reviewed",
                "parent_revision": OTHER_REVISION,
                "status": {"code": "runs"},
            }
        )


@pytest.mark.smoke
def test_grant_park_append_is_idempotent_and_void_reopens(tmp_path: Path) -> None:
    """The same blocked fact replays as a no-op; a void releases and reopens."""

    ledger = grant_park_path(tmp_path)
    first = append_grant_park_row(ledger, _park_row())
    # A resume re-derives the same blocked append with a fresh timestamp and
    # run id: the SAME fact, so a no-op returning the first persisted row.
    replay = append_grant_park_row(
        ledger, _park_row(created_at="2026-08-11T01:00:00Z", run_id="run-2")
    )
    assert replay == first
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert [row["row_sha256"] for row in load_grant_park_rows([ledger])] == [first["row_sha256"]]

    # An operator void releases the park: the row stops governing entirely.
    void = build_grant_park_void(
        void_of=str(first["row_sha256"]),
        reason="stale pending append; re-derive under the fresh generation",
        voided_by="operator",
        created_at="2026-08-11T02:00:00Z",
    )
    append_grant_park_void(ledger, void)
    assert load_grant_park_rows([ledger]) == ()
    # Voiding is idempotent and never rewrites history.
    append_grant_park_void(ledger, void)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2

    # A fresh park after the void is a NEW fact: the voided row cannot swallow it.
    reopened = append_grant_park_row(
        ledger, _park_row(created_at="2026-08-11T03:00:00Z", run_id="run-3")
    )
    assert reopened["row_sha256"] != first["row_sha256"]
    assert [row["row_sha256"] for row in load_grant_park_rows([ledger])] == [reopened["row_sha256"]]
    # A void must reference a persisted row.
    with pytest.raises(GrantParkError, match="unknown"):
        append_grant_park_void(
            ledger,
            build_grant_park_void(
                void_of="sha256:" + "d" * 64,
                reason="nothing to void",
                voided_by="operator",
                created_at="2026-08-11T04:00:00Z",
            ),
        )


def _ledger_paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths for reducer-level tests."""

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _review_held_first_terminal() -> tuple[dict, dict]:
    """Return one decisive failed attempt and its review-held terminal model."""

    failed_attempt = make_failed_attempt()
    first = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    first["status"]["human_review"] = {
        "required": True,
        "reason": None,
        "queue": "crawler-human-review",
        "requested_at": first["created_at"],
    }
    return failed_attempt, first


def _superseding_candidate(persisted: dict) -> dict:
    """Return a fully valid superseding revision of one persisted record."""

    candidate = deepcopy(persisted)
    candidate.pop("record_seq", None)
    candidate.pop("record_revision", None)
    candidate["parent_revision"] = persisted["record_revision"]
    candidate["status"]["supersedes_revision"] = persisted["record_revision"]
    candidate["notes"] = "fresh re-drive under the current dependency tree"
    return candidate


@pytest.mark.smoke
def test_reducer_raises_typed_error_only_when_grant_is_the_sole_blocker(
    tmp_path: Path,
) -> None:
    """The typed park signal fires ONLY for otherwise-valid grantless supersessions."""

    paths = _ledger_paths(tmp_path)
    failed_attempt, first = _review_held_first_terminal()
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
        persisted = reducer.append_model(first).record

        # Otherwise valid, blocked ONLY by the missing grant: typed and parkable.
        candidate = _superseding_candidate(persisted)
        with pytest.raises(GrantPendingSupersessionError) as blocked:
            reducer.append_model(candidate)
        assert isinstance(blocked.value, ReductionError)
        assert blocked.value.stable_id == "m_example"
        assert blocked.value.blocked_parent_revision == persisted["record_revision"]
        assert blocked.value.blocked_stage == "source"
        assert "requires a new explicit grant" in str(blocked.value)
        # The refusal is a refusal: the review-held terminal still governs.
        assert [record["record_revision"] for record in scan_jsonl(paths.models)] == [
            persisted["record_revision"]
        ]

        # Tripwire: a supersession broken BEFORE the grant rule (bad lineage)
        # fails loudly with the plain refusal, never the parkable signal.
        bad_lineage = _superseding_candidate(persisted)
        bad_lineage["status"]["supersedes_revision"] = "sha256:" + "f" * 64
        with pytest.raises(ReductionError, match="exactly match parent_revision") as broken:
            reducer.append_model(bad_lineage)
        assert not isinstance(broken.value, GrantPendingSupersessionError)

        # Tripwire: a defect in ANY later validation (unknown terminal status)
        # also outranks the park signal -- the park may never mask a bad append.
        bad_status = _superseding_candidate(persisted)
        bad_status["status"]["code"] = "failed:not-a-real-stage"
        with pytest.raises(ReductionError) as invalid:
            reducer.append_model(bad_status)
        assert not isinstance(invalid.value, GrantPendingSupersessionError)


@pytest.mark.smoke
def test_reducer_grant_blocked_pre_check_matches_the_append_rule(
    tmp_path: Path,
) -> None:
    """The publication pre-check mirrors the append tripwire exactly."""

    paths = _ledger_paths(tmp_path)
    failed_attempt, first = _review_held_first_terminal()
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
        persisted = reducer.append_model(first).record
        candidate = _superseding_candidate(persisted)
        blocked = reducer.grant_blocked_supersession(candidate)
        assert blocked is not None
        assert blocked["record_revision"] == persisted["record_revision"]
        # A new explicit grant unblocks the pre-check.
        granted = deepcopy(candidate)
        granted["budget"]["explicit_grants"] = [
            *granted["budget"].get("explicit_grants", []),
            "sha256:" + "c" * 64,
        ]
        assert reducer.grant_blocked_supersession(granted) is None

    # A previous terminal WITHOUT human review never pre-checks as blocked.
    other = _ledger_paths(tmp_path / "unreviewed")
    unreviewed_attempt = make_failed_attempt()
    unreviewed = bind_terminal_attempts(
        make_model(status_code="failed:source"), [unreviewed_attempt]
    )
    with CanonicalReducer(other, ["m_example"]) as reducer:
        reducer.append_attempt(unreviewed_attempt)
        persisted = reducer.append_model(unreviewed).record
        assert reducer.grant_blocked_supersession(_superseding_candidate(persisted)) is None


@pytest.mark.smoke
def test_partition_accounts_grant_pending_rows_by_revision() -> None:
    """A live park claims an absent model, coexists with its exact review-held
    terminal, and reads as consumed once a granted supersession lands."""

    row = _park_row()

    # Claiming: the review-held terminal is dependency-stale (absent from the
    # projection), so the park row is the model's only truthful accounting.
    claiming = partition_report(["m_reviewed"], [], machine_deferrals={"m_reviewed": row})
    assert claiming.valid
    assert "m_reviewed" in claiming.machine_deferred.get(GRANT_PENDING_DISPOSITION, ())
    assert "m_reviewed" in claiming.grant_pending_ids

    # Coexisting: the review-held terminal is still projection-current. The
    # terminal satisfies the partition and the park stays LIVE -- it is a
    # pending supersession, never a stale row a present record superseded.
    held = {
        "stable_id": "m_reviewed",
        "record_revision": REVISION,
        "status": {"code": "failed:accuracy-gate"},
    }
    coexisting = partition_report(["m_reviewed"], [held], machine_deferrals={"m_reviewed": row})
    assert coexisting.valid
    assert "m_reviewed" not in coexisting.superseded_deferral_ids
    assert "m_reviewed" in coexisting.grant_pending_ids

    # Consumed: a granted supersession moved the head revision; the park row is
    # released and disclosed exactly as any other superseded valve row.
    superseding = {
        "stable_id": "m_reviewed",
        "record_revision": OTHER_REVISION,
        "status": {"code": "runs"},
    }
    consumed = partition_report(
        ["m_reviewed"], [superseding], machine_deferrals={"m_reviewed": row}
    )
    assert consumed.valid
    assert "m_reviewed" in consumed.superseded_deferral_ids
    assert "m_reviewed" not in consumed.grant_pending_ids
    assert grant_park_released(row, superseding) is True
    assert grant_park_released(row, held) is False
    assert grant_park_released(row, None) is False


@pytest.mark.smoke
def test_grant_park_fact_still_holds_on_the_same_review_held_head(
    tmp_path: Path,
) -> None:
    """The re-check keys on the exact review-held ledger head, fail-open."""

    models = tmp_path / "records" / "models.jsonl"
    models.parent.mkdir(parents=True)
    head: dict[str, Any] = {
        "stable_id": "m_reviewed",
        "record_revision": REVISION,
        "record_seq": 1,
        "parent_revision": None,
        "status": {
            "code": "failed:accuracy-gate",
            "supersedes_revision": None,
            "human_review": {"required": True},
        },
    }
    models.write_text(json.dumps(head) + "\n", encoding="utf-8")
    driver = SimpleNamespace(
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=models),
        ),
        config=SimpleNamespace(machine_id="machine-1"),
    )
    item = SimpleNamespace(route=SimpleNamespace(intent="core"))
    row = _park_row()
    assert _deferral_row_fact_still_holds(driver, item, row) is True  # type: ignore[arg-type]

    # A superseded head (the grant lane published) releases the fact.
    moved = dict(head)
    moved.update(record_revision=OTHER_REVISION, record_seq=2, parent_revision=REVISION)
    moved["status"] = {
        **head["status"],
        "supersedes_revision": REVISION,
        "human_review": {"required": False},
    }
    models.write_text(json.dumps(head) + "\n" + json.dumps(moved) + "\n", encoding="utf-8")
    assert _deferral_row_fact_still_holds(driver, item, row) is False  # type: ignore[arg-type]

    # An unreadable ledger is "unknown", never "still provably blocked".
    models.unlink()
    assert _deferral_row_fact_still_holds(driver, item, row) is False  # type: ignore[arg-type]


@pytest.mark.smoke
def test_grant_parks_tool_lists_and_voids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator CLI lists live parks and appends a reasoned void."""

    from menagerie.crawler.tools.grant_parks import main as parks_main

    records_root = tmp_path / "records"
    row = append_grant_park_row(grant_park_path(records_root), _park_row())
    base = ["--repo-root", str(tmp_path), "--records-root", str(records_root)]

    assert parks_main(base) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [item["row_sha256"] for item in listed["review_holds"]] == [row["row_sha256"]]

    # A void is an operator decision: it refuses to land without a reason.
    assert parks_main(["--void", str(row["row_sha256"]), *base]) == 1
    assert (
        parks_main(["--void", str(row["row_sha256"]), "--reason", "operator rejects", *base]) == 0
    )
    capsys.readouterr()
    assert load_grant_park_rows([grant_park_path(records_root)]) == ()


def _rewrite_intake_with_changed_sources(tmp_path: Path, count: int) -> IntakeSnapshot:
    """Rewrite the synthetic master roster with changed immutable intake bytes."""

    _write_jsonl(
        tmp_path / "master.jsonl",
        [
            {
                "name": f"Example{index}",
                "zoo": "fixtures",
                "variant": "base",
                "source_url": f"https://example.com/changed/{index}",
            }
            for index in range(count)
        ],
    )
    return create_intake_snapshot(
        tmp_path / "master.jsonl",
        tmp_path / "deferred.jsonl",
        tmp_path / "intake",
    )


@pytest.mark.heavy
def test_three_legs_park_then_grant_then_consume(tmp_path: Path) -> None:
    """Leg 2 parks instead of dying; the grant releases; leg 3 consumes and closes.

    This is the exact rung-12 leg-3/leg-6 trajectory: leg 1 mints review-held
    terminals, changed intake makes them dependency-stale, and the re-drive's
    fresh awards may not supersede operator property without a grant.

    COMPOSE (AW x WS-1): ordinary mode never re-drives a settled head, stale
    or not (clause 4 / V3) -- leg 2's re-drive is now the operator's NAMED
    ``--redrive-targets`` decision. Admission and supersession stay separate
    authorities: the named re-drive admits the work, and the close STILL parks
    the supersession of operator property pending a grant. The grants then
    release the parks at the SELECTION layer (the pending operator decision
    has been made) and leg 3 consumes through the one grant lane.
    """

    from dataclasses import replace

    from menagerie.crawler.tests.campaign_binding_support import (
        adopt_successor_snapshot,
        bind_snapshot_at_records_root,
    )

    count = 2
    snapshot = _snapshot(tmp_path, count=count)
    paths = _paths(tmp_path, snapshot)
    records_root = paths.ledgers.models.parent.parent
    bind_snapshot_at_records_root(records_root, snapshot)

    # LEG 1: exhausted metadata repairs terminalize every model review-held.
    first = _driver(
        tmp_path,
        snapshot,
        author=RepairingIdentityAuthor(),
        checker=ScriptedChecker(
            script=CheckerScript(metadata_verdict="inaccurate", lineage_repairs=True)
        ),
    ).run()
    assert first.status == "terminal-partition-complete"
    leg1_models = scan_jsonl(paths.ledgers.models)
    heads = {record["stable_id"]: record for record in leg1_models}
    assert len(heads) == count
    assert all(record["status"]["human_review"]["required"] for record in heads.values())

    # LEG 2: changed intake bytes stale every leg-1 record; the operator names
    # the re-drive (--redrive-targets) and the completed fresh work's award is
    # blocked ONLY by the missing grant.
    changed_snapshot = _rewrite_intake_with_changed_sources(tmp_path, count)
    # Chain the changed rung to leg 1's registered snapshot exactly as
    # `stage_rung` does in production: the close-v2 partition proof carries a
    # parked member's unchanged head across legs only along the registered
    # lineage (BL-5), so an unregistered changed snapshot would read leg-1
    # evidence as foreign at the idempotent-resume close below.
    adopt_successor_snapshot(records_root, changed_snapshot, label="changed-intake-redrive")
    second_driver = _driver(
        tmp_path,
        changed_snapshot,
        author=ChangedInputAuthor(),
        forward=FakeForward(),
    )
    second_driver.config = replace(
        second_driver.config,
        redrive_targets=tuple(sorted(heads)),
        # D-54: a named re-drive is an operator act with a blast radius; its
        # rationale is required and recorded on the durable leg config event.
        redrive_reason="grant-parks leg 2: re-drive the heads staled by changed intake bytes",
    )
    second = second_driver.run()

    # THE fix: the close PARKS instead of failing the run, with a distinct,
    # never-"complete" status.
    assert second.status == "terminal-partition-complete-with-review-holds"
    # No new model revision minted: the review-held terminals still govern.
    assert len(scan_jsonl(paths.ledgers.models)) == len(leg1_models)
    park_rows = load_grant_park_rows([grant_park_path(records_root)])
    assert {row["stable_id"] for row in park_rows} == set(heads)
    for row in park_rows:
        assert row["disposition"] == GRANT_PENDING_DISPOSITION
        assert (
            row["pending"]["blocked_parent_revision"]
            == (heads[str(row["stable_id"])]["record_revision"])
        )
        assert row["pending"]["blocked_stage"] == "accuracy-gate"
        assert row["pending"]["pending_status_code"] == "runs"
        assert machine_deferral_rejection(str(row["stable_id"]), row) is None
    # The durable rows are what the next leg reads back.
    persisted = _persisted_deferral_rows(SimpleNamespace(paths=paths))
    assert set(persisted) == set(heads)

    # The park is loud where operators look: one operational event per model
    # and a driver-state block naming every pending grant.
    events = scan_jsonl(canonical_operational_ledger_path(paths.ledgers.models))
    parked_events = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == GRANT_PARK_EVENT_DISPOSITION
    ]
    assert {event["details"]["stable_id"] for event in parked_events} == set(heads)
    assert not [event for event in events if event.get("event_kind") == "terminal-unrecordable"]
    state = json.loads(Path(paths.driver_state).read_text(encoding="utf-8"))
    assert state["status"] == "terminal-partition-complete-with-review-holds"
    assert {entry["stable_id"] for entry in state["review_holds"]["pending"]} == set(heads)
    for entry in state["review_holds"]["pending"]:
        assert entry["stage"] == "accuracy-gate"
        assert entry["blocked_parent_revision"] == (heads[entry["stable_id"]]["record_revision"])

    # Grant consumption stays the ONLY publish path: nothing was authorized
    # for publication while the grant was pending.
    artifact_events = scan_jsonl(paths.ledgers.artifacts)
    assert not [
        event
        for event in artifact_events
        if event.get("event_kind") in {"publication-authorized", "terminal-authorized"}
    ]

    # A resume against the still-ungranted ledger is idempotent: same parks,
    # no duplicate rows, still no new terminals.
    resumed = _driver(
        tmp_path,
        changed_snapshot,
        author=ChangedInputAuthor(),
        forward=FakeForward(),
    ).run()
    assert resumed.status == "terminal-partition-complete-with-review-holds"
    assert len(scan_jsonl(paths.ledgers.models)) == len(leg1_models)
    assert (
        len(
            [
                line
                for line in grant_park_path(records_root).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        == count
    )

    # LEG 3: the operator grants each pending supersession; the next close
    # consumes the grants through the EXISTING lane and publishes.
    grants_path = canonical_requeue_grants_path(paths.ledgers.models)
    grant_ids = {}
    for stable_id in heads:
        payload = build_canonical_requeue_grant(
            grants_path,
            stable_id=stable_id,
            stage="accuracy-gate",
            reason="reviewed: supersession approved after park",
            attempts=1,
            granted_by="operator",
            target_intent=None,
        )
        append_canonical_requeue_grant(grants_path, payload)
        grant_ids[stable_id] = str(payload["grant_id"])

    third = _driver(
        tmp_path,
        changed_snapshot,
        author=ChangedInputAuthor(),
        forward=FakeForward(),
    ).run()
    assert third.status == "complete"
    final = {
        record["stable_id"]: record
        for record in scan_jsonl(paths.ledgers.models)
        if record["stable_id"] in heads
    }
    for stable_id, head in heads.items():
        published = final[stable_id]
        assert published["status"]["code"] == "runs"
        assert published["parent_revision"] == head["record_revision"]
        assert grant_ids[stable_id] in published["budget"]["explicit_grants"]
    # The parks are consumed: every row is released by its granted supersession
    # and the clean close clears the review-holds block.
    for row in load_grant_park_rows([grant_park_path(records_root)]):
        assert grant_park_released(row, final[str(row["stable_id"])]) is True
    state = json.loads(Path(paths.driver_state).read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert "review_holds" not in state


@pytest.mark.smoke
def test_null_stage_review_held_park_records_a_grantable_stage(
    tmp_path: Path,
) -> None:
    """A stage-null review-held supersession parks with the GRANTABLE sentinel.

    The T2-05 review-held deferral class has ``status.stage is None``, so its
    blocked supersession used to park as ``blocked_stage="none"`` -- and the
    row's own recheck hint then instructed ``requeue --stage 'none'``, a
    command the grant builder REFUSES (``"none"`` is neither a failure stage
    nor the sentinel). The park's remedy must round-trip: the stage the row
    names is the stage an operator grant may actually carry, and that grant
    must bind the recorded head.
    """

    from types import SimpleNamespace

    from menagerie.crawler.constants import (
        DEFERRED_TERMINAL_GRANT_STAGE,
        OPERATIONAL_EVENT_SCHEMA_VERSION,
    )
    from menagerie.crawler.driver_admission import (
        _grant_stage_binds_terminal,
        _park_grant_pending_supersession,
    )
    from menagerie.crawler.recordio import JsonlLedger
    from menagerie.crawler.reducer import (
        GrantPendingSupersessionError,
        default_ledger_paths,
    )

    records_root = tmp_path / "records"
    ledgers = default_ledger_paths(records_root)
    ledgers.models.parent.mkdir(parents=True, exist_ok=True)
    driver = SimpleNamespace(
        paths=SimpleNamespace(ledgers=ledgers),
        dependencies=SimpleNamespace(clock=lambda: "2026-08-12T00:00:00Z"),
        config=SimpleNamespace(campaign_id="campaign-1", run_id="run-1", machine_id="machine-1"),
    )
    item = SimpleNamespace(
        stable_id="m_deferred",
        active_work_id="work-m_deferred",
        intake=SimpleNamespace(name="Deferred"),
    )
    blocked = GrantPendingSupersessionError(
        stable_id="m_deferred",
        blocked_parent_revision=REVISION,
        blocked_stage=None,
    )
    pending_model = {
        "stable_id": "m_deferred",
        "parent_revision": REVISION,
        "status": {"code": "runs", "reason_code": None, "stage": None},
    }
    with JsonlLedger(
        tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION
    ) as operational:
        row = _park_grant_pending_supersession(driver, item, pending_model, blocked, operational)

    recorded_stage = row["pending"]["blocked_stage"]
    assert recorded_stage == DEFERRED_TERMINAL_GRANT_STAGE
    assert DEFERRED_TERMINAL_GRANT_STAGE in row["recheck_hint"]

    # The remedy round-trips: the stage the row names is issuable...
    grants_path = canonical_requeue_grants_path(ledgers.models)
    grants_path.parent.mkdir(parents=True, exist_ok=True)
    grant = build_canonical_requeue_grant(
        grants_path,
        stable_id="m_deferred",
        stage=recorded_stage,
        reason="operator releases the parked supersession",
        attempts=1,
        granted_by="operator",
    )
    # ...and the issued grant BINDS the null-stage review-held head it parks.
    head = {
        "record_revision": REVISION,
        "status": {
            "kind": "deferred",
            "code": "deferred:needs-cuda",
            "stage": None,
            "human_review": {"required": True},
        },
    }
    assert _grant_stage_binds_terminal(grant, head) is True
