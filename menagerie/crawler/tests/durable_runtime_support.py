"""RUNTIME evidence that each named durable writer really issues a data barrier.

Why this exists at all
----------------------
``tests/test_durable_fsync_census.py`` is an AST census, and an AST census is a
guarantee about SPELLINGS. It has now been patched twice for spellings it could not
see (PSLRE4 D-4 the aliased ``from os import fsync as _flush`` import; PSLRE5 P5-D-3
the module-granular duty a sibling writer discharged), and PSLRE6 D6-4 found the next
one: a money-row append can reach the page-cache-only syscall through a COMPUTED
attribute -- ``getattr(os, "fs" + "ync")(fd)`` -- which every arm of an attribute /
import scan is structurally blind to. Adding a third and fourth AST pattern is the
flail: the pattern space is unbounded, and each new arm only closes the one spelling
someone thought of.

So this module changes the KIND of evidence instead of extending the pattern list.
It patches the two barrier PRIMITIVES (``os.fsync`` and ``fcntl.fcntl``), drives each
named writer's REAL append against a real file, and asserts what actually happened at
the syscall boundary:

* a DATA barrier was issued on the ledger's own data inode -- not on its parent
  directory, which is a different syscall making a different guarantee (D6-1);
* at the moment of that barrier the file already held its final bytes, so the barrier
  followed the last write rather than preceding it;
* the named writer function was on the call stack when it happened, so a sibling
  writer in the same module cannot discharge another writer's duty (P5-D-3); and
* on Darwin -- the only host the round21 release registry supports -- the barrier was
  ``F_FULLFSYNC`` and NOT ``os.fsync``, which is exactly the D6-4 escape: however the
  call is spelled, a page-cache-only flush of the ledger inode is now a failure.

Because the patch is on the ``os`` module attribute rather than on any name in any
production module, it is reached by ``os.fsync(fd)``, by ``getattr(os, "fsync")(fd)``,
and by every other runtime attribute lookup, computed or not. The one spelling it does
NOT see is an alias bound at IMPORT time (``from os import fsync as _flush``), because
that binding captured the original function object before the patch existed -- and
that spelling is precisely what the AST census's import arm already forbids
structurally. The two arms are complementary by construction, and
``test_durable_fsync_runtime.py`` asserts that complementarity rather than assuming it.
"""

from __future__ import annotations

import fcntl
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Iterator, Mapping, Optional

from menagerie.crawler.identity import DARWIN_F_FULLFSYNC

#: The real primitives, captured once at import so a spy can call through them.
_REAL_FSYNC = os.fsync
_REAL_FCNTL = fcntl.fcntl
_REAL_FSTAT = os.fstat

#: ``os.fsync`` on Darwin returns once the bytes leave the kernel page cache; the
#: device may still lose them. ``F_FULLFSYNC`` is the only power-loss barrier there.
PAGE_CACHE_ONLY = "os.fsync"
DEVICE_BARRIER = "fcntl.F_FULLFSYNC"


@dataclass(frozen=True)
class BarrierEvent:
    """One observed barrier syscall, with the file it named and who issued it."""

    kind: str
    device: int
    inode: int
    is_directory: bool
    size: int
    stack: tuple[tuple[str, str], ...]

    def issued_by(self, module_name: str, function: str) -> bool:
        """Return whether one named writer function was on the issuing stack."""

        return (module_name, function) in self.stack

    def names(self, target: os.stat_result) -> bool:
        """Return whether this barrier named one exact file, by inode identity."""

        return (self.device, self.inode) == (target.st_dev, target.st_ino)


def _stack() -> tuple[tuple[str, str], ...]:
    """Return the ``(module file name, function)`` chain above the spy, innermost first."""

    frames: list[tuple[str, str]] = []
    frame: Optional[FrameType] = sys._getframe(2)  # noqa: SLF001 -- stack attribution
    while frame is not None:
        frames.append((Path(frame.f_code.co_filename).name, frame.f_code.co_name))
        frame = frame.f_back
    return tuple(frames)


class BarrierRecorder:
    """Record every barrier syscall issued while it is installed.

    The spies CALL THROUGH to the real primitives, so the writers under observation
    are genuinely durable while being observed: this is instrumentation, never a
    stub that would let a broken writer look fine.
    """

    def __init__(self) -> None:
        """Start with an empty barrier log."""

        self.events: list[BarrierEvent] = []

    def _record(self, kind: str, descriptor: int) -> None:
        """Append one barrier event, resolving the descriptor to a file identity."""

        info = _REAL_FSTAT(descriptor)
        self.events.append(
            BarrierEvent(
                kind=kind,
                device=info.st_dev,
                inode=info.st_ino,
                is_directory=stat.S_ISDIR(info.st_mode),
                size=info.st_size,
                stack=_stack(),
            )
        )

    def install(self, monkeypatch: Any) -> None:
        """Patch the barrier primitives on the ``os`` and ``fcntl`` MODULE objects.

        Patching the module attribute (rather than any importer's name) is what makes
        a computed attribute lookup visible: ``getattr(os, "fsync")`` resolves through
        the same slot the spy occupies.
        """

        def spy_fsync(descriptor: int) -> None:
            """Record and perform one page-cache flush."""

            self._record(PAGE_CACHE_ONLY, descriptor)
            _REAL_FSYNC(descriptor)

        def spy_fcntl(descriptor: int, command: int, *rest: Any) -> Any:
            """Record a device barrier, and perform every ``fcntl`` unchanged."""

            if command == DARWIN_F_FULLFSYNC:
                self._record(DEVICE_BARRIER, descriptor)
            return _REAL_FCNTL(descriptor, command, *rest)

        monkeypatch.setattr(os, "fsync", spy_fsync)
        monkeypatch.setattr(fcntl, "fcntl", spy_fcntl)

    # -- queries ----------------------------------------------------------

    def data_barriers(self, target: os.stat_result) -> tuple[BarrierEvent, ...]:
        """Return every barrier issued on one exact file's data inode."""

        return tuple(
            event
            for event in self.events
            if event.names(target) and not event.is_directory
        )

    def directory_barriers(self) -> tuple[BarrierEvent, ...]:
        """Return every barrier issued on a directory rather than a file."""

        return tuple(event for event in self.events if event.is_directory)


def durable_runtime_findings(
    recorder: BarrierRecorder,
    *,
    ledger: Path,
    module_name: str,
    function: str,
) -> list[str]:
    """Return why one writer's REAL append was not power-loss durable, if it was not.

    Three findings, each a distinct failure the AST census cannot see:

    * no data barrier reached the ledger's own inode -- a writer whose only barrier
      was ``fsync_directory`` satisfied the AST census's undifferentiated barrier
      vocabulary while leaving the ROW itself in a volatile cache (D6-1);
    * the named writer was not on the stack of any such barrier -- a sibling writer
      in the same module issued it (P5-D-3, measured rather than parsed);
    * on Darwin the barrier was a page-cache-only ``os.fsync`` however it was spelled,
      including through a computed attribute (D6-4).

    Parameters
    ----------
    recorder:
        Recorder installed for the duration of the driver's run.
    ledger:
        The ledger file whose bytes the writer acknowledged.
    module_name, function:
        The census's name for this writer.

    Returns
    -------
    list[str]
        Findings, empty when the writer is durable.
    """

    if not ledger.is_file():
        return [f"{function} wrote no {ledger.name}: the runtime driver proved nothing"]
    target = ledger.stat()
    barriers = recorder.data_barriers(target)
    if not barriers:
        return [
            f"{module_name}:{function} issued NO data barrier on {ledger.name}'s own "
            "inode: a directory barrier does not make the row's bytes durable, so an "
            "acknowledged row can still vanish"
        ]
    findings: list[str] = []
    mine = [event for event in barriers if event.issued_by(module_name, function)]
    if not mine:
        return [
            f"{module_name}:{function} was not on the stack of any data barrier for "
            f"{ledger.name}: some OTHER writer's barrier is answering for its duty"
        ]
    if not any(event.size == target.st_size for event in mine):
        findings.append(
            f"{module_name}:{function} issued its last data barrier at "
            f"{max(event.size for event in mine)} bytes while {ledger.name} ends at "
            f"{target.st_size}: the trailing write was never flushed"
        )
    if sys.platform == "darwin":
        page_cache = [event for event in mine if event.kind == PAGE_CACHE_ONLY]
        if page_cache or any(event.kind != DEVICE_BARRIER for event in mine):
            findings.append(
                f"{module_name}:{function} flushed {ledger.name} with "
                f"{sorted({event.kind for event in mine})} rather than F_FULLFSYNC: on "
                "Darwin that leaves the bytes in the device's volatile cache, and it "
                "is reachable through any spelling of the raw syscall"
            )
    return findings


#: ``(tmp_path) -> the ledger file whose durability the driver just exercised``.
RuntimeDriver = Callable[[Path], Path]


def _records(root: Path) -> Path:
    """Return a fresh canonical records root below one temporary directory."""

    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    return records


def _identity(**overrides: Any) -> Mapping[str, Any]:
    """Return the shared work-identity block every valve and park row carries."""

    values: dict[str, Any] = {
        "stable_id": "m_runtime",
        "work_id": "work-m_runtime",
        "name": "Runtime",
        "campaign_id": "c1-mech",
        "run_id": "run-1",
        "machine_id": "machine-1",
        "created_at": "2026-08-19T00:00:00Z",
    }
    values.update(overrides)
    return values


def drive_fact_ledger(root: Path) -> Path:
    """Append one real model record through the canonical JSONL fact writer."""

    from menagerie.crawler.constants import MODEL_SCHEMA_VERSION_V3
    from menagerie.crawler.recordio import JsonlLedger
    from menagerie.crawler.tests.conftest import make_model

    path = _records(root) / "models.jsonl"
    ledger = JsonlLedger(path, MODEL_SCHEMA_VERSION_V3)
    try:
        ledger.append(make_model())
    finally:
        ledger.close()
    return path


def _authority(root: Path) -> Any:
    """Return one temporary paid-session authority over a fresh money ledger."""

    from menagerie.crawler.paid_session_lifecycle import PaidSessionAuthority

    return PaidSessionAuthority(
        _records(root) / "operational" / "spend.jsonl",
        runtime_root=root / ".crawl-local",
        clock=lambda: "2026-08-19T12:00:00Z",
    )


def _reserve(authority: Any) -> None:
    """Take one real money reservation, which appends a money row and its anchor."""

    from menagerie.crawler.paid_session_accounting import BudgetPolicy

    authority.reserve(
        scope={
            "campaign_id": "c1-mech",
            "generation_id": "generation-open",
            "lane": "checker",
            "dispatch_kind": "metadata_batch",
            "work_generations": (("m_runtime", "work-m_runtime"),),
            "gate_kind": "metadata_batch",
            "checker_identity": None,
            "run_id": "run-1",
            "machine_id": "machine-1",
        },
        policy=BudgetPolicy(
            quota_sessions=100,
            quota_wall_seconds=100_000.0,
            runaway_sessions=100,
            runaway_wall_seconds=100_000.0,
            reservation_wall_seconds=100.0,
        ),
    )


def drive_money_row(root: Path) -> Path:
    """Append one REAL money row -- the D6-4 target -- and name its ledger."""

    authority = _authority(root)
    _reserve(authority)
    return authority.path


def drive_money_anchor(root: Path) -> Path:
    """Take one real reservation and name the external extent ANCHOR object."""

    authority = _authority(root)
    _reserve(authority)
    return authority.path.with_name(f"{authority.path.name}.anchor.json")


def drive_capacity_valve(root: Path) -> Path:
    """Append one real host-capacity machine-deferral row."""

    import json

    from menagerie.crawler.host_capacity import (
        append_capacity_deferral_row,
        capacity_deferral_path,
    )

    # The committed rung-10 pilot row: a REAL row from the leg that produced it, which
    # the writer re-validates on the way in (the sibling disclosure census uses the
    # same fixture rather than reconstructing a refusing assessment by hand).
    fixture = Path(__file__).resolve().parent / "rung10_host_capacity_deferrals.jsonl"
    row = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])
    records = _records(root)
    path = capacity_deferral_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_capacity_deferral_row(path, row)
    return path


def drive_coverage_valve(root: Path) -> Path:
    """Append one real environment-coverage machine-deferral row."""

    import json

    from menagerie.crawler.environment_coverage import (
        append_coverage_deferral_row,
        coverage_deferral_path,
    )

    fixture = Path(__file__).resolve().parent / "rung10_environment_coverage_deferrals.jsonl"
    row = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])
    records = _records(root)
    path = coverage_deferral_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_coverage_deferral_row(path, row)
    return path


def drive_checker_lane_valve(root: Path) -> Path:
    """Append one real checker-lane machine-deferral row."""

    from menagerie.crawler.checker_lane import (
        append_checker_lane_deferral_row,
        build_checker_lane_deferral_row,
        checker_lane_deferral_path,
    )

    records = _records(root)
    path = checker_lane_deferral_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_checker_lane_deferral_row(
        path,
        build_checker_lane_deferral_row(
            **_identity(),
            gate_kind="metadata_batch",
            failure_class="permanent-contract-rejection",
            failure_evidence="checker returned no schema-valid verdict for the member",
            attempted_batch_sizes=[8, 4, 2, 1],
        ),
    )
    return path


def drive_rebuild_valve(root: Path) -> Path:
    """Append one real environment-rebuild machine-deferral row."""

    from menagerie.crawler.environment_rebuild import (
        append_rebuild_deferral_row,
        build_rebuild_deferral_row,
        rebuild_deferral_path,
    )

    records = _records(root)
    path = rebuild_deferral_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_rebuild_deferral_row(
        path,
        build_rebuild_deferral_row(
            **_identity(),
            quarantine_event_id="environment-cleanup-quarantined-1",
            intent="graph",
            target="osx-arm64",
            artifact_identity="sha256:" + "c" * 64,
            artifact_state={"lock": {"state": "present", "sha256": "sha256:" + "d" * 64}},
            env_generation=None,
            failure_type="EnvironmentBuildError",
        ),
    )
    return path


def drive_disk_floor_valve(root: Path) -> Path:
    """Append one real disk-floor machine-deferral row."""

    from menagerie.crawler.disk_floor import (
        append_disk_floor_deferral_row,
        build_disk_floor_deferral_row,
        disk_floor_deferral_path,
    )

    records = _records(root)
    path = disk_floor_deferral_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_disk_floor_deferral_row(
        path,
        build_disk_floor_deferral_row(
            **_identity(),
            intent="core",
            target="osx-arm64",
            env_root="/nonexistent/env-store",
            floor_bytes=32 * 2**30,
            free_bytes=2 * 2**30,
        ),
    )
    return path


def _quarantine_row() -> Mapping[str, Any]:
    """Return one valid scheduler-quarantine park row."""

    from menagerie.crawler.scheduler_parks import build_scheduler_quarantine_row

    return build_scheduler_quarantine_row(
        **_identity(),
        site="scheduler-round",
        reason_code="member-fault",
        fault_evidence="the scheduler raised while firing the lane",
        exception_type="RuntimeError",
    )


def drive_scheduler_quarantine_row(root: Path) -> Path:
    """Append one real scheduler-quarantine park row."""

    from menagerie.crawler.scheduler_parks import (
        append_scheduler_quarantine_row,
        scheduler_quarantine_path,
    )

    records = _records(root)
    path = scheduler_quarantine_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_scheduler_quarantine_row(path, _quarantine_row())
    return path


def drive_scheduler_quarantine_void(root: Path) -> Path:
    """Append one real scheduler-quarantine VOID over a live park row."""

    from menagerie.crawler.scheduler_parks import (
        append_scheduler_quarantine_row,
        append_scheduler_quarantine_void,
        build_scheduler_quarantine_void,
        scheduler_quarantine_path,
    )

    records = _records(root)
    path = scheduler_quarantine_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = append_scheduler_quarantine_row(path, _quarantine_row())
    append_scheduler_quarantine_void(
        path,
        build_scheduler_quarantine_void(
            void_of=str(row["row_sha256"]),
            reason="the fault was a machine condition and cleared",
            voided_by="operator",
            created_at="2026-08-19T01:00:00Z",
        ),
    )
    return path


def _grant_park_row() -> Mapping[str, Any]:
    """Return one valid grant-pending park row."""

    from menagerie.crawler.grant_parks import build_grant_park_row

    return build_grant_park_row(
        **_identity(),
        blocked_parent_revision="sha256:" + "a" * 64,
        blocked_stage="accuracy-gate",
        pending_model={
            "stable_id": "m_runtime",
            "status": {"code": "runs", "reason_code": None, "stage": None},
        },
    )


def drive_grant_park_row(root: Path) -> Path:
    """Append one real grant-pending park row."""

    from menagerie.crawler.grant_parks import append_grant_park_row, grant_park_path

    records = _records(root)
    path = grant_park_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_grant_park_row(path, _grant_park_row())
    return path


def drive_grant_park_void(root: Path) -> Path:
    """Append one real grant-park VOID over a live park row."""

    from menagerie.crawler.grant_parks import (
        append_grant_park_row,
        append_grant_park_void,
        build_grant_park_void,
        grant_park_path,
    )

    records = _records(root)
    path = grant_park_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = append_grant_park_row(path, _grant_park_row())
    append_grant_park_void(
        path,
        build_grant_park_void(
            void_of=str(row["row_sha256"]),
            reason="the blocking parent revision was superseded",
            voided_by="operator",
            created_at="2026-08-19T01:00:00Z",
        ),
    )
    return path


def _refusal_park_row() -> Mapping[str, Any]:
    """Return one valid terminal-refusal park row."""

    from menagerie.crawler.refusal_parks import build_refusal_park_row

    return build_refusal_park_row(
        **_identity(),
        refusal_class="intended-and-fallback-refused",
        record_error_type="menagerie.crawler.reducer.ReductionError",
        record_error="terminal proof is unavailable",
        fallback_error_type="menagerie.crawler.reducer.ReductionError",
        fallback_error="minimal rung refused too",
        intended_status_code="failed:import",
        intended_reason_code="import-error",
        blocked_parent_revision=None,
        candidate_sha256=None,
    )


def drive_refusal_park_row(root: Path) -> Path:
    """Append one real terminal-refusal park row."""

    from menagerie.crawler.refusal_parks import append_refusal_park_row, refusal_park_path

    records = _records(root)
    path = refusal_park_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_refusal_park_row(path, _refusal_park_row())
    return path


def drive_refusal_park_void(root: Path) -> Path:
    """Append one real refusal-park VOID over a live park row."""

    from menagerie.crawler.refusal_parks import (
        append_refusal_park_row,
        append_refusal_park_void,
        build_refusal_park_void,
        refusal_park_path,
    )

    records = _records(root)
    path = refusal_park_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = append_refusal_park_row(path, _refusal_park_row())
    append_refusal_park_void(
        path,
        build_refusal_park_void(
            void_of=str(row["row_sha256"]),
            reason="the terminal was recorded on the retry",
            voided_by="operator",
            created_at="2026-08-19T01:00:00Z",
        ),
    )
    return path


def drive_leg_open(root: Path) -> Path:
    """Open one real leg generation, which writes the durable leg-open epoch."""

    from menagerie.crawler.leg_generation import leg_events_path, open_leg_generation

    records = _records(root)
    open_leg_generation(
        records,
        campaign_id="c1-mech",
        intake_snapshot_id="rung-0001",
        intake_snapshot_sha256="sha256:" + "e" * 64,
        run_id="run-1",
        opened_at="2026-08-19T00:00:00Z",
    )
    return leg_events_path(records)


def drive_leg_event(root: Path) -> Path:
    """Seal one real leg PLAN event through the shared leg-event writer."""

    from menagerie.crawler.leg_generation import (
        leg_events_path,
        open_leg_generation,
        record_leg_plan,
    )

    records = _records(root)
    generation = open_leg_generation(
        records,
        campaign_id="c1-mech",
        intake_snapshot_id="rung-0001",
        intake_snapshot_sha256="sha256:" + "e" * 64,
        run_id="run-1",
        opened_at="2026-08-19T00:00:00Z",
    )
    record_leg_plan(
        records,
        campaign_id="c1-mech",
        generation_id=generation.generation_id,
        run_id="run-1",
        admitted_ids=["m_runtime"],
        plan_payload={"selected": ["m_runtime"]},
        created_at="2026-08-19T00:01:00Z",
    )
    return leg_events_path(records)


def drive_close_object(root: Path) -> Path:
    """Append one real historical close object to the close meta ledger."""

    from menagerie.crawler.close_object import (
        append_close_object,
        close_object_path,
        predecessor_close_sha256,
        verifier_identity,
    )
    from menagerie.crawler.identity import stamp_row_digest

    records = _records(root)
    body: dict[str, Any] = {
        "schema_version": "menagerie.crawler.leg-close.v1",
        "campaign_id": "c1-mech",
        "intake_snapshot_id": "rung-0001",
        "intake_snapshot_sha256": "sha256:" + "e" * 64,
        "phase": None,
        "run_id": "run-1",
        "machine_id": "machine-1",
        "closed_at": "2026-08-19T00:02:00Z",
        "completion_status": "complete",
        "predecessor_close_sha256": predecessor_close_sha256(records, "c1-mech"),
        "ledgers": {
            name: {"rows": 0, "ledger_sha256": stamp_row_digest({})["row_sha256"], "tail_sha256": None}
            for name in ("models", "attempts", "gates", "artifacts")
        },
        "partition": {
            "intake": 0,
            "valid": True,
            "missing": [],
            "extra": [],
            "duplicates": [],
            "invalid": [],
            "buckets": {},
            "machine_deferred": {},
            "grant_pending": [],
            "refusal_parked": [],
        },
        "currency": {"stale_by_axis": {}},
        "verifier_identity": verifier_identity({}),
        "authority_identities": {},
    }
    append_close_object(
        close_object_path(records), stamp_row_digest(body, field="close_sha256")
    )
    return close_object_path(records)


def drive_rung_registration(root: Path) -> Path:
    """Append one real rung registration to the registry meta ledger."""

    from menagerie.crawler.rung_staging import (
        append_rung_registration,
        build_rung_registration,
        rung_registry_path,
    )

    records = _records(root)
    path = rung_registry_path(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_rung_registration(
        path,
        build_rung_registration(
            campaign_id="c1-mech",
            snapshot_id="intake-" + "a" * 20,
            snapshot_sha256="sha256:" + "e" * 64,
            derived_from_snapshot_id="intake-" + "0" * 20,
            member_count=1,
            label="runtime-fixture",
            registered_by="pytest",
            created_at="2026-08-19T00:00:00Z",
        ),
    )
    return path


def _grant_ledger(root: Path) -> Path:
    """Return the canonical operational grant ledger below one records root."""

    from menagerie.crawler.checkpoint import canonical_operational_ledger_path

    records = _records(root)
    path = canonical_operational_ledger_path(records / "models.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def drive_requeue_grant(root: Path) -> Path:
    """Append one real operator requeue grant to the operational ledger."""

    from menagerie.crawler.checkpoint import (
        append_canonical_requeue_grant,
        build_canonical_requeue_grant,
    )

    path = _grant_ledger(root)
    append_canonical_requeue_grant(
        path,
        build_canonical_requeue_grant(
            path,
            stable_id="m_runtime",
            stage="accuracy-gate",
            reason="release the runtime fixture",
            attempts=1,
            granted_by="operator",
        ),
    )
    return path


def drive_requeue_void(root: Path) -> Path:
    """Append one real operator VOID over a live unconsumed requeue grant."""

    from menagerie.crawler.checkpoint import (
        append_canonical_requeue_grant,
        append_canonical_requeue_void,
        build_canonical_requeue_grant,
        build_canonical_requeue_void,
    )

    path = _grant_ledger(root)
    grant = build_canonical_requeue_grant(
        path,
        stable_id="m_runtime",
        stage="accuracy-gate",
        reason="release the runtime fixture",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(path, grant)
    append_canonical_requeue_void(
        path,
        build_canonical_requeue_void(
            path,
            void_of=str(grant["grant_id"]),
            reason="the grant was superseded",
            voided_by="operator",
            created_at="2026-08-19T01:00:00Z",
        ),
    )
    return path


#: EVERY named durable writer, keyed exactly as the AST census names it, mapped to a
#: driver that performs its REAL append. ``test_durable_fsync_runtime.py`` asserts
#: this covers the census's writer universe by SET EQUALITY, so a new ledger cannot
#: be added with an AST-only duty.
RUNTIME_DRIVERS: dict[tuple[str, str], RuntimeDriver] = {
    ("recordio.py", "append"): drive_fact_ledger,
    ("paid_session_durability.py", "append"): drive_money_row,
    ("paid_session_durability.py", "_write_anchor"): drive_money_anchor,
    ("host_capacity.py", "append_capacity_deferral_row"): drive_capacity_valve,
    ("environment_coverage.py", "append_coverage_deferral_row"): drive_coverage_valve,
    ("checker_lane.py", "append_checker_lane_deferral_row"): drive_checker_lane_valve,
    ("environment_rebuild.py", "append_rebuild_deferral_row"): drive_rebuild_valve,
    ("disk_floor.py", "append_disk_floor_deferral_row"): drive_disk_floor_valve,
    ("scheduler_parks.py", "append_scheduler_quarantine_row"): drive_scheduler_quarantine_row,
    ("scheduler_parks.py", "append_scheduler_quarantine_void"): drive_scheduler_quarantine_void,
    ("grant_parks.py", "append_grant_park_row"): drive_grant_park_row,
    ("grant_parks.py", "append_grant_park_void"): drive_grant_park_void,
    ("refusal_parks.py", "append_refusal_park_row"): drive_refusal_park_row,
    ("refusal_parks.py", "append_refusal_park_void"): drive_refusal_park_void,
    ("leg_generation.py", "open_leg_generation"): drive_leg_open,
    ("leg_generation.py", "_append_leg_event"): drive_leg_event,
    ("close_object.py", "append_close_object"): drive_close_object,
    ("rung_staging.py", "append_rung_registration"): drive_rung_registration,
    ("checkpoint.py", "append_canonical_requeue_grant"): drive_requeue_grant,
    ("checkpoint.py", "append_canonical_requeue_void"): drive_requeue_void,
}


def driven_rows(ledger: Path) -> Iterator[dict[str, Any]]:
    """Yield every decoded row one driver actually persisted.

    The same drivers that prove durability at runtime also supply the ONLY real
    writer output the content-address transparency arms have (PSLRE6 D6-6): before
    this, both transparency tests ran on hand-built rows, so "every family's own
    stamp is the body derivation" was asserted about rows no writer produced.
    """

    import json

    if not ledger.is_file():
        return
    for line in ledger.read_bytes().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            yield value
