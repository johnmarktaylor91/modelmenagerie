"""Torn valve-ledger tails recover like every other ledger (T1-07 / T5-1).

Every machine-deferral valve (disk-floor, host-capacity, environment-coverage,
environment-rebuild, checker-lane, grant-pending parks) and the operational
requeue-grants ledger is an append-only JSONL file written under SIGKILL,
power loss, and -- for the disk-floor valve, by design -- ENOSPC. A crash
mid-append leaves a torn, non-newline-terminated final line. The strict
whole-file decode every valve performs on BOTH read and append must not turn
that torn tail into a permanent brick: the recovery contract is recordio's
evidence-then-truncate (the torn bytes are hashed into a durable recovery
ledger BEFORE truncation), after which appends and reads proceed and the
parked claim is re-derivable. A COMPLETE malformed line is hard corruption
and stays a typed refusal: only a torn final line ever recovers, and never
when the retained prefix itself fails the valve's own validation.
"""

from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import pytest

from menagerie.crawler import checker_lane, grant_parks
from menagerie.crawler.checker_lane import (
    CheckerLaneDeferralError,
    append_checker_lane_deferral_row,
    build_checker_lane_deferral_row,
    checker_lane_deferral_path,
)
from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    append_canonical_requeue_void,
    build_canonical_requeue_grant,
    build_canonical_requeue_void,
)
from menagerie.crawler.cli import EXIT_OK, main as cli_main
from menagerie.crawler.disk_floor import (
    DiskFloorDeferralError,
    append_disk_floor_deferral_row,
    build_disk_floor_deferral_row,
    disk_floor_deferral_path,
    load_disk_floor_deferral_rows,
)
from menagerie.crawler.environment_coverage import (
    CoverageDeferralError,
    append_coverage_deferral_row,
    build_coverage_deferral_row,
    assess_environment_coverage,
    coverage_deferral_path,
    load_coverage_deferral_rows,
)
from menagerie.crawler.environment_rebuild import (
    RebuildDeferralError,
    append_rebuild_deferral_row,
    build_rebuild_deferral_row,
    load_rebuild_deferral_rows,
    rebuild_deferral_path,
)
from menagerie.crawler.envs import load_environment_registry
from menagerie.crawler.grant_parks import (
    GrantParkError,
    append_grant_park_row,
    append_grant_park_void,
    build_grant_park_row,
    build_grant_park_void,
    grant_park_path,
    load_grant_park_rows,
)
from menagerie.crawler.quota_deferral import (
    QuotaDeferralError,
    load_quota_deferral_rows,
    mint_quota_deferral,
    quota_deferral_path,
)
from menagerie.crawler.release_vocabulary import (
    ReleaseVocabularyError,
    append_blocked_prerequisite_record,
    blocked_prerequisite_record_path,
    blocked_prerequisite_records,
    build_blocked_prerequisite_record,
)
from menagerie.crawler.host_capacity import (
    CapacityDeferralError,
    append_capacity_deferral_row,
    assess_model_capacity,
    build_capacity_deferral_row,
    capacity_deferral_path,
    load_capacity_deferral_rows,
)
from menagerie.crawler.recordio import LedgerCorruptionError, scan_jsonl
from menagerie.crawler.refusal_parks import (
    RefusalParkError,
    append_refusal_park_row,
    build_refusal_park_row,
    load_refusal_park_rows,
    refusal_park_path,
)
from menagerie.crawler.scheduler_parks import (
    SchedulerQuarantineError,
    append_scheduler_quarantine_row,
    build_scheduler_quarantine_row,
    load_scheduler_quarantine_rows,
    scheduler_quarantine_path,
)
from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign
from menagerie.crawler.tests.test_environment_coverage import CORE_PACKAGES
from menagerie.crawler.tests.test_host_capacity import MIXTRAL_RECIPE, SMALL_HOST
from menagerie.crawler.tests.test_slice_f_driver import _snapshot

pytestmark = pytest.mark.smoke

REVISION = "sha256:" + "a" * 64

_REBUILD_STATE = {
    "osx-arm64.lock": {"state": "present", "sha256": "sha256:" + "aa" * 32},
    "osx-arm64.resolved.json": {"state": "present", "sha256": "sha256:" + "bb" * 32},
}

_IDENTITY_DEFAULTS = {
    "campaign_id": "campaign-1",
    "run_id": "run-1",
    "machine_id": "machine-1",
    "created_at": "2026-08-11T00:00:00Z",
}


def _identity(index: int) -> dict[str, Any]:
    """Return the varying identity fields for the ``index``-th test row."""

    return {
        "stable_id": f"m_{index:04d}",
        "work_id": f"work-m_{index:04d}",
        "name": f"Example{index}",
        **_IDENTITY_DEFAULTS,
    }


def _disk_floor_row(index: int) -> dict[str, Any]:
    """Build one valid disk-floor deferral row."""

    return build_disk_floor_deferral_row(
        **_identity(index),
        intent="core",
        target="osx-arm64",
        env_root="/tmp/envs",
        floor_bytes=30 * 1024**3,
        free_bytes=5 * 1024**3,
    )


def _capacity_row(index: int) -> dict[str, Any]:
    """Build one valid host-capacity deferral row from the real refusal."""

    return build_capacity_deferral_row(
        **_identity(index),
        assessment=assess_model_capacity(MIXTRAL_RECIPE, host=SMALL_HOST),
    )


def _coverage_row(index: int) -> dict[str, Any]:
    """Build one valid environment-coverage deferral row."""

    implementation = {
        "recipe_type": "declarative-library",
        "library_recipe": {"distribution": "not-in-any-intent", "version": "1.0.0"},
    }
    return build_coverage_deferral_row(
        **_identity(index),
        assessment=assess_environment_coverage(
            implementation,
            routed_intent="core",
            routed_packages=list(CORE_PACKAGES),
            registry=load_environment_registry(),
        ),
    )


def _rebuild_row(index: int) -> dict[str, Any]:
    """Build one valid environment-rebuild deferral row."""

    return build_rebuild_deferral_row(
        **_identity(index),
        quarantine_event_id="environment-cleanup-abc123",
        intent="core",
        target="osx-arm64",
        artifact_identity="sha256:" + "11" * 32,
        artifact_state=_REBUILD_STATE,
        env_generation="sha256:" + "22" * 32,
        failure_type="menagerie.crawler.env_lifecycle.DiskRecoveryError",
    )


def _checker_lane_row(index: int) -> dict[str, Any]:
    """Build one valid checker-lane deferral row."""

    return build_checker_lane_deferral_row(
        **_identity(index),
        gate_kind="metadata_batch",
        failure_class="permanent-contract-rejection",
        failure_evidence="checker exited 64 at batch floor one",
        attempted_batch_sizes=(4, 2, 1),
    )


def _grant_park_row(index: int) -> dict[str, Any]:
    """Build one valid grant-pending park row."""

    identity = _identity(index)
    return build_grant_park_row(
        **identity,
        blocked_parent_revision=REVISION,
        blocked_stage="accuracy-gate",
        pending_model={
            "stable_id": identity["stable_id"],
            "status": {"code": "runs", "reason_code": None, "stage": None},
        },
    )


def _refusal_park_row(index: int) -> dict[str, Any]:
    """Build one valid terminal-refusal park row."""

    return build_refusal_park_row(
        **_identity(index),
        refusal_class="ReductionError",
        record_error_type="ReductionError",
        record_error="terminal cap not reached",
        fallback_error_type="ReductionError",
        fallback_error="terminal cap not reached",
        intended_status_code="failed:forward",
        intended_reason_code=None,
        blocked_parent_revision=None,
        candidate_sha256=None,
    )


def _scheduler_quarantine_row(index: int) -> dict[str, Any]:
    """Build one valid scheduler-fault quarantine row."""

    return build_scheduler_quarantine_row(
        **_identity(index),
        site="scheduler-round",
        reason_code="member-fault",
        fault_evidence="synthetic member fault escaped the seam's containment arms",
        exception_type="RuntimeError",
    )


def _load_checker_lane_rows(paths: Sequence[Path]) -> tuple[Mapping[str, Any], ...]:
    """Route through the checker-lane loader every other valve already has."""

    loader = getattr(checker_lane, "load_checker_lane_deferral_rows", None)
    assert loader is not None, (
        "checker_lane must expose load_checker_lane_deferral_rows so status and "
        "resume reads share one validated, tail-recovering read path"
    )
    return loader(paths)


@dataclass(frozen=True)
class ValveSpec:
    """One valve ledger's complete torn-tail test surface."""

    key: str
    module_file: str
    path_for: Callable[[Path], Path]
    build_row: Callable[[int], dict[str, Any]]
    append: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]
    load: Callable[[Sequence[Path]], tuple[Mapping[str, Any], ...]]
    error: type[Exception]


def _quota_deferral_mint_kwargs(index: int) -> dict[str, Any]:
    """Build one quota-deferral mint argument set (the family's one chokepoint)."""

    identity = _identity(index)
    identity.pop("name")
    return {
        **identity,
        "evidence": {"witness": "author-sessions-exhaustion-classifier"},
    }


def _mint_quota_deferral_adapter(path: Path, row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Adapt the mint chokepoint to the suite's append(path, row) shape."""

    return mint_quota_deferral(path, **row)


def _load_quota_deferral_paths(paths: Sequence[Path]) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for path in paths for row in load_quota_deferral_rows(path))


def _blocked_prerequisite_row(index: int) -> dict[str, Any]:
    """Build one valid blocked-prerequisite seam record."""

    identity = _identity(index)
    identity.pop("name")
    return build_blocked_prerequisite_record(
        **identity,
        stage="environment",
        prerequisite_tokens=("runtime-dependency:jsonschema",),
        gate_id=f"gate-{index:04d}",
        author_result_sha256="sha256:" + f"{index:x}".rjust(64, "0"),
    )


def _load_blocked_prerequisite_paths(paths: Sequence[Path]) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for path in paths for row in blocked_prerequisite_records(path))


VALVES: tuple[ValveSpec, ...] = (
    ValveSpec(
        "disk-floor",
        "disk_floor.py",
        disk_floor_deferral_path,
        _disk_floor_row,
        append_disk_floor_deferral_row,
        load_disk_floor_deferral_rows,
        DiskFloorDeferralError,
    ),
    ValveSpec(
        "host-capacity",
        "host_capacity.py",
        capacity_deferral_path,
        _capacity_row,
        append_capacity_deferral_row,
        load_capacity_deferral_rows,
        CapacityDeferralError,
    ),
    ValveSpec(
        "environment-coverage",
        "environment_coverage.py",
        coverage_deferral_path,
        _coverage_row,
        append_coverage_deferral_row,
        load_coverage_deferral_rows,
        CoverageDeferralError,
    ),
    ValveSpec(
        "environment-rebuild",
        "environment_rebuild.py",
        rebuild_deferral_path,
        _rebuild_row,
        append_rebuild_deferral_row,
        load_rebuild_deferral_rows,
        RebuildDeferralError,
    ),
    ValveSpec(
        "checker-lane",
        "checker_lane.py",
        checker_lane_deferral_path,
        _checker_lane_row,
        append_checker_lane_deferral_row,
        _load_checker_lane_rows,
        CheckerLaneDeferralError,
    ),
    ValveSpec(
        "grant-pending",
        "grant_parks.py",
        grant_park_path,
        _grant_park_row,
        append_grant_park_row,
        load_grant_park_rows,
        GrantParkError,
    ),
    ValveSpec(
        "terminal-refusal",
        "refusal_parks.py",
        refusal_park_path,
        _refusal_park_row,
        append_refusal_park_row,
        load_refusal_park_rows,
        RefusalParkError,
    ),
    # BL-8 (a)/(b): the scheduler-fault quarantine valve registered in the
    # same reviewed change that introduced its ledger.
    ValveSpec(
        "scheduler-quarantine",
        "scheduler_parks.py",
        scheduler_quarantine_path,
        _scheduler_quarantine_row,
        append_scheduler_quarantine_row,
        load_scheduler_quarantine_rows,
        SchedulerQuarantineError,
    ),
    ValveSpec(
        "quota-deferral",
        "quota_deferral.py",
        quota_deferral_path,
        _quota_deferral_mint_kwargs,
        _mint_quota_deferral_adapter,
        _load_quota_deferral_paths,
        QuotaDeferralError,
    ),
    ValveSpec(
        "blocked-prerequisite",
        "release_vocabulary.py",
        blocked_prerequisite_record_path,
        _blocked_prerequisite_row,
        append_blocked_prerequisite_record,
        _load_blocked_prerequisite_paths,
        ReleaseVocabularyError,
    ),
)

_VALVE_IDS = [spec.key for spec in VALVES]


def _tear_final_line(path: Path) -> int:
    """Truncate the ledger mid final line, exactly as a crash mid-append does.

    Returns
    -------
    int
        Byte offset where the retained complete prefix ends.
    """

    data = path.read_bytes()
    assert data.endswith(b"\n")
    good_end = data.rindex(b"\n", 0, len(data) - 1) + 1 if data.count(b"\n") > 1 else 0
    torn_line = data[good_end:]
    torn = data[: good_end + len(torn_line) // 2]
    assert not torn.endswith(b"\n")
    path.write_bytes(torn)
    return good_end


def _evidence_path(ledger: Path) -> Path:
    """Return recordio's default recovery-evidence ledger for one valve file."""

    return ledger.with_suffix(f"{ledger.suffix}.recovery.jsonl")


@pytest.mark.parametrize("spec", VALVES, ids=_VALVE_IDS)
def test_a_torn_valve_append_recovers_like_every_other_ledger(
    spec: ValveSpec, tmp_path: Path
) -> None:
    """A crash mid-append never bricks future appends on any valve ledger.

    The torn final line is evidenced (offset + tail hash, fsynced) and then
    truncated; the earlier complete row survives; the interrupted append is
    re-derivable and lands.
    """

    ledger = spec.path_for(tmp_path)
    first = spec.append(ledger, spec.build_row(1))
    second_row = spec.build_row(2)
    spec.append(ledger, second_row)
    good_end = _tear_final_line(ledger)

    recovered = spec.append(ledger, second_row)
    assert recovered["stable_id"] == second_row["stable_id"]

    loaded = spec.load([ledger])
    assert {row["stable_id"] for row in loaded} == {
        first["stable_id"],
        second_row["stable_id"],
    }

    evidence = _evidence_path(ledger)
    assert evidence.is_file(), "tail recovery must leave durable evidence"
    records = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert records[0]["byte_offset"] == good_end
    assert records[0]["byte_count"] > 0
    assert records[0]["tail_sha256"].startswith("sha256:")


@pytest.mark.parametrize("spec", VALVES, ids=_VALVE_IDS)
def test_a_torn_valve_load_never_kills_a_status_read(spec: ValveSpec, tmp_path: Path) -> None:
    """Reading a torn valve ledger recovers with evidence instead of dying.

    ``crawler status`` and launch preflight read these ledgers through the
    ``load_*`` functions; a torn tail must yield every complete row plus
    durable evidence of the truncated fragment, never an exception and never
    a silent drop.
    """

    ledger = spec.path_for(tmp_path)
    first = spec.append(ledger, spec.build_row(1))
    spec.append(ledger, spec.build_row(2))
    _tear_final_line(ledger)

    loaded = spec.load([ledger])
    assert {row["stable_id"] for row in loaded} == {first["stable_id"]}
    assert _evidence_path(ledger).is_file(), "a dropped torn row must be evidenced, never silent"

    # The interrupted parking is re-derivable afterwards: nothing was bricked.
    reparked = spec.append(ledger, spec.build_row(2))
    assert {row["stable_id"] for row in spec.load([ledger])} == {
        first["stable_id"],
        reparked["stable_id"],
    }


@pytest.mark.parametrize("spec", VALVES, ids=_VALVE_IDS)
def test_newline_terminated_garbage_is_still_refused(spec: ValveSpec, tmp_path: Path) -> None:
    """A COMPLETE malformed line stays hard, typed corruption on read and append."""

    ledger = spec.path_for(tmp_path)
    spec.append(ledger, spec.build_row(1))
    with ledger.open("ab") as handle:
        handle.write(b"not-json\n")
    frozen = ledger.read_bytes()

    with pytest.raises(spec.error):
        spec.append(ledger, spec.build_row(2))
    with pytest.raises(spec.error):
        spec.load([ledger])
    assert ledger.read_bytes() == frozen, "hard corruption must never be rewritten"
    assert not _evidence_path(ledger).exists()


@pytest.mark.parametrize("spec", VALVES, ids=_VALVE_IDS)
def test_a_torn_tail_behind_hard_corruption_is_not_truncated(
    spec: ValveSpec, tmp_path: Path
) -> None:
    """Recovery refuses to touch a ledger whose retained prefix is corrupt.

    Evidence-then-truncate may only ever remove the torn FINAL line; if any
    preceding complete line is malformed the ledger is hard corruption and
    must be preserved byte-for-byte for the operator.
    """

    ledger = spec.path_for(tmp_path)
    spec.append(ledger, spec.build_row(1))
    with ledger.open("ab") as handle:
        handle.write(b"not-json\n")
        handle.write(b'{"torn":')
    frozen = ledger.read_bytes()

    with pytest.raises(spec.error):
        spec.append(ledger, spec.build_row(2))
    assert ledger.read_bytes() == frozen
    assert not _evidence_path(ledger).exists()


def test_torn_park_ledger_earlier_row_still_claims_on_resume(tmp_path: Path) -> None:
    """The lenient driver resume reader keeps the earlier park's claim.

    T1-07's silent-drop arm: the torn line itself is recovered WITH evidence
    by the valve read path, and the rows before it keep claiming their models
    through ``_persisted_deferral_rows`` exactly as before the crash.
    """

    from menagerie.crawler.driver_admission import _persisted_deferral_rows

    records_root = tmp_path / "records"
    ledger = grant_park_path(records_root)
    first = append_grant_park_row(ledger, _grant_park_row(1))
    append_grant_park_row(ledger, _grant_park_row(2))
    _tear_final_line(ledger)

    driver = SimpleNamespace(
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=records_root / "models" / "current-shard.jsonl")
        )
    )
    persisted = _persisted_deferral_rows(driver)
    assert first["stable_id"] in persisted

    # The strict valve read recovers rather than silently dropping: evidence.
    loaded = load_grant_park_rows([ledger])
    assert {row["stable_id"] for row in loaded} == {first["stable_id"]}
    assert _evidence_path(ledger).is_file()


def test_torn_grant_park_void_append_recovers(tmp_path: Path) -> None:
    """The park VOID append path recovers a torn tail exactly like row appends."""

    ledger = grant_park_path(tmp_path)
    first = append_grant_park_row(ledger, _grant_park_row(1))
    append_grant_park_row(ledger, _grant_park_row(2))
    _tear_final_line(ledger)

    void = build_grant_park_void(
        void_of=str(first["row_sha256"]),
        reason="operator released the park",
        voided_by="operator",
        created_at="2026-08-11T02:00:00Z",
    )
    append_grant_park_void(ledger, void)
    assert load_grant_park_rows([ledger]) == ()
    assert _evidence_path(ledger).is_file()


def test_requeue_grants_ledger_recovers_a_torn_tail(tmp_path: Path) -> None:
    """The operational requeue-grants ledger recovers on build and append.

    Both the generation-allocating build and the idempotent append strictly
    re-read the whole ledger; a torn final line (the requeue tool killed
    mid-append) must not brick the operator's grant lane.
    """

    ledger = tmp_path / "records" / "operational" / "requeue-grants.jsonl"
    ledger.parent.mkdir(parents=True)
    first = build_canonical_requeue_grant(
        ledger,
        stable_id="m_0001",
        stage="author",
        reason="first",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(ledger, first)
    second = build_canonical_requeue_grant(
        ledger,
        stable_id="m_0002",
        stage="author",
        reason="second",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(ledger, second)
    good_end = _tear_final_line(ledger)

    rebuilt = build_canonical_requeue_grant(
        ledger,
        stable_id="m_0002",
        stage="author",
        reason="second",
        attempts=1,
        granted_by="operator",
    )
    # W31: created_at is a fresh occurrence stamp outside the grant identity;
    # everything identity-bearing must restore the exact torn generation.
    assert {k: v for k, v in rebuilt.items() if k != "created_at"} == {
        k: v for k, v in second.items() if k != "created_at"
    }, "recovery must restore the exact torn generation"
    assert isinstance(rebuilt["created_at"], str) and rebuilt["created_at"]
    append_canonical_requeue_grant(ledger, rebuilt)

    rows = scan_jsonl(ledger, validate=False)
    assert [row["stable_id"] for row in rows] == ["m_0001", "m_0002"]

    void = build_canonical_requeue_void(
        ledger, void_of=str(first["grant_id"]), reason="stale", voided_by="operator"
    )
    append_canonical_requeue_void(ledger, void)

    evidence = _evidence_path(ledger)
    assert evidence.is_file()
    records = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert records[0]["byte_offset"] == good_end


def test_requeue_grants_complete_malformed_line_is_still_refused(
    tmp_path: Path,
) -> None:
    """Newline-terminated garbage in the grants ledger stays hard corruption."""

    ledger = tmp_path / "records" / "operational" / "requeue-grants.jsonl"
    ledger.parent.mkdir(parents=True)
    grant = build_canonical_requeue_grant(
        ledger,
        stable_id="m_0001",
        stage="author",
        reason="first",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(ledger, grant)
    with ledger.open("ab") as handle:
        handle.write(b"not-json\n")
    frozen = ledger.read_bytes()

    with pytest.raises(LedgerCorruptionError):
        build_canonical_requeue_grant(
            ledger,
            stable_id="m_0002",
            stage="author",
            reason="second",
            attempts=1,
            granted_by="operator",
        )
    assert ledger.read_bytes() == frozen
    assert not _evidence_path(ledger).exists()


def test_crawler_status_survives_torn_park_and_checker_lane_ledgers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plain ``crawler status`` is the T1-17 mitigant: it must not die torn.

    A torn grant-pending park ledger (read unconditionally at status time)
    and a torn checker-lane ledger (read for the partition view) both
    recover with evidence; the command exits OK and prints its payload.
    """

    snapshot = _snapshot(tmp_path)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    records_root = tmp_path / "records"

    park_ledger = grant_park_path(records_root)
    append_grant_park_row(park_ledger, _grant_park_row(1))
    append_grant_park_row(park_ledger, _grant_park_row(2))
    _tear_final_line(park_ledger)

    lane_ledger = checker_lane_deferral_path(records_root)
    append_checker_lane_deferral_row(lane_ledger, _checker_lane_row(3))
    append_checker_lane_deferral_row(lane_ledger, _checker_lane_row(4))
    _tear_final_line(lane_ledger)

    exit_code = cli_main(
        [
            "--repo-root",
            str(tmp_path),
            "status",
            "--intake",
            str(snapshot.root),
            "--records-root",
            str(records_root),
            "--verify-partition",
        ]
    )
    assert exit_code == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "funnel" in payload
    holds = {row["stable_id"] for row in payload.get("review_holds", ())}
    assert "m_0001" in holds, "the surviving park row must still surface"


def test_sigkill_between_park_append_and_fsync_resumes(tmp_path: Path) -> None:
    """A parker killed after write but before fsync resumes idempotently.

    The kill window T5-4 cares about at the valve boundary: the line may be
    complete-but-unfsynced (this test) or torn (the truncation tests above).
    Either way the next leg's re-derived parking lands without conflict and
    without a duplicate row.

    The kill is injected at the DURABILITY BARRIER, which since r2fix is
    ``identity.durable_fsync`` (``F_FULLFSYNC`` on Darwin) rather than
    ``os.fsync``. Patching the barrier through the writer's own module binding
    keeps the fault exactly where it was -- after the write and flush, before
    the bytes are durable -- instead of letting the injection quietly become a
    no-op the moment the write path stopped calling ``os.fsync``.
    """

    ledger = grant_park_path(tmp_path)
    append_grant_park_row(ledger, _grant_park_row(1))
    row = _grant_park_row(2)

    pid = os.fork()
    if pid == 0:  # pragma: no cover - child process dies by SIGKILL
        try:
            grant_parks.durable_fsync = lambda fd: os.kill(  # type: ignore[assignment]
                os.getpid(), signal.SIGKILL
            )
            append_grant_park_row(ledger, row)
        finally:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL

    resumed = append_grant_park_row(ledger, row)
    assert resumed["row_sha256"] == row["row_sha256"]
    loaded = load_grant_park_rows([ledger])
    assert {item["stable_id"] for item in loaded} == {"m_0001", "m_0002"}
    persisted = ledger.read_text(encoding="utf-8").splitlines()
    assert (
        sum(1 for line in persisted if json.loads(line)["row_sha256"] == row["row_sha256"]) == 1
    ), "the crash-window row must persist exactly once"


def test_valve_membership_is_one_closed_inventory() -> None:
    """Every hand-rolled valve ledger decoder routes through tail recovery.

    The valve fingerprint is the strict whole-file ``splitlines`` decode under
    an exclusive ``flock``. This inventory is CLOSED: a new valve module that
    clones the pattern without routing its reads through recordio's
    evidence-then-truncate recovery fails here by set inequality, and an
    existing valve that drops the recovery call fails the reference check.
    """

    import menagerie.crawler as crawler_package

    crawler_root = Path(crawler_package.__file__).parent
    decode_idiom = 'decode("utf-8").splitlines()'
    expected = {spec.module_file for spec in VALVES}

    hand_rolled: set[str] = set()
    for source_file in sorted(crawler_root.rglob("*.py")):
        relative = source_file.relative_to(crawler_root)
        if relative.parts[0] == "tests":
            continue
        source = source_file.read_text(encoding="utf-8")
        if decode_idiom in source and "fcntl.flock" in source:
            hand_rolled.add(relative.as_posix())

    # checkpoint.py is a fingerprint false-positive, not a valve: its flock
    # guards the W31 requeue-grant append (read-check-write race closure),
    # every requeue read routes through recordio's evidence-then-truncate
    # recovery (asserted below; torn-tail behavior covered by
    # test_requeue_grants_ledger_recovers_a_torn_tail), and its only
    # splitlines decodes are checkpoint-staging scans of committed file
    # bytes, never a live valve ledger read.
    # env_lifecycle.py's environment-turn-effort ledger is a mirror ledger, not
    # a valve family (no build/append/load row API for this suite to drive);
    # its reads and appends route through recordio's evidence-then-truncate
    # recovery (asserted below) and its torn-tail behavior is covered by
    # test_environment_turn_torn_tail.py (D-60).
    non_valve_recovered = {"checkpoint.py", "env_lifecycle.py"}
    for module_file in sorted(non_valve_recovered):
        source = (crawler_root / module_file).read_text(encoding="utf-8")
        assert "recover_torn_tail" in source, (
            f"{module_file} lost its recordio torn-tail recovery routing; it "
            "may no longer claim the non-valve carve-out"
        )
    hand_rolled -= non_valve_recovered

    assert hand_rolled == expected, (
        "valve ledger inventory changed: every module that hand-rolls a "
        "flocked splitlines JSONL decode must be registered in VALVES with "
        "torn-tail recovery coverage"
    )

    for module_file in sorted(expected):
        source = (crawler_root / module_file).read_text(encoding="utf-8")
        assert "read_with_tail_recovery" in source, (
            f"{module_file} must route its ledger reads through recordio.read_with_tail_recovery"
        )

    # The requeue-grants ledger lives in checkpoint.py over scan_jsonl: its
    # build/append readers carry the same recovery obligation.
    checkpoint_source = (crawler_root / "checkpoint.py").read_text(encoding="utf-8")
    assert "recover_torn_tail" in checkpoint_source, (
        "checkpoint.py requeue-grant readers must recover a torn final line"
    )
