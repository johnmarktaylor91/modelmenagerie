"""Sequential target-solved environment creation, probing, use, and teardown."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from menagerie.crawler.authority import EnvironmentAuthorityCache
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    FailureStage,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.envs import (
    DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES,
    LEGACY_UNBOUND_OVERLAY_MANIFESTS,
    EnvironmentIntent,
    IntentProbes,
)
from menagerie.crawler.effort import EffortTracker, EffortUsage, StageCap
from menagerie.crawler.identity import (
    canonical_json_bytes,
    compute_env_generation,
    durable_fsync,
    fsync_directory,
    hash_bytes,
    utc_now,
)
from menagerie.crawler.models import TailRecoveryEvidence
from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.package_namespace import (
    PackageNamespaceError,
    assert_inventory_provisions,
)
from menagerie.crawler.recordio import (
    JsonlLedger,
    operational_events_path,
    recover_torn_tail,
)

LOGGER = logging.getLogger(__name__)


class EnvironmentLifecycleError(RuntimeError):
    """Base class for typed lifecycle failures."""


class EnvironmentSolveError(EnvironmentLifecycleError):
    """Raised when an on-target intent solve fails."""


class CommittedLockFamilyError(EnvironmentSolveError):
    """Raised when no proven committed lock family can serve as the build input.

    Committed locks are INPUTS, not outputs: a launch builds FROM the
    committed, provenance-bound family and never solves in place (T3-T1
    3-lab; the pre-fix lane solved fresh or stale-cache-first and overwrote
    the proven family before create/probe). Solving is a rotation-only
    operation -- run ``tools/rotate_setup_artifacts.py`` to prove and publish
    a new family. Subclasses the solve arm so the driver transition table
    keeps its typed environment routing rather than the build-failed
    catch-all.
    """


class EnvironmentProbeError(EnvironmentLifecycleError):
    """Raised when an exact environment probe fails."""


class EnvironmentOverlayError(EnvironmentLifecycleError):
    """Raised when a declared pip overlay cannot be installed exactly."""


class EnvironmentBusyError(EnvironmentLifecycleError):
    """Raised when a second environment is requested before teardown."""


class EnvironmentBackendUnavailableError(EnvironmentLifecycleError):
    """Raised when the environment tooling wrapper reports transient infrastructure.

    The operator protocol's sysexits split (75 = retryable infrastructure,
    64 = permanent rejection) must survive the subprocess boundary: folding
    every nonzero wrapper exit into one integration error erased the typing
    and let the driver's old catch-all terminalize a transient outage
    (T3-T2 facet a). This class carries exit 75: a HOST/infrastructure
    condition -- the wrapper, store, or network is unavailable RIGHT NOW --
    so the driver parks affected work instead of minting model terminals.
    """


class EnvironmentBackendContractError(EnvironmentLifecycleError):
    """Raised when the environment tooling wrapper permanently rejects an action.

    Exit 64: the wrapper adjudicated the request itself as invalid for this
    intent's setup (a bad lock, an unusable registration). That is a fact
    about the SETUP, never about any model, so the driver parks the intent's
    work until the setup artifacts change or an operator clears it.
    """


class DiskRecoveryError(EnvironmentLifecycleError):
    """Raised on an unmet disk floor or actual environment residue after teardown.

    The pre-build floor refusal raises the :class:`DiskFloorError` subclass so
    the driver can park it as a machine deferral; residue raises this base.

    Deliberately NOT raised for a mere free-space delta: ambient disk
    consumption during a leg (ledgers, diagnostics, unrelated processes) moves
    the ``statvfs`` measurement without any cleanup failure, and treating that
    delta as incomplete cleanup quarantined a healthy environment generation in
    rung 11. The residue authority is the recorded environment prefix itself --
    the artifacts this lifecycle materialized -- never a df delta.
    """


class DiskFloorError(DiskRecoveryError):
    """Raised when the host free-disk floor refuses to START an environment build.

    This is a MACHINE condition, never a fact about any model: no solve,
    build, or probe ran for anything the environment would have served. The
    driver parks affected work identities in the durable disk-floor
    machine-deferral ledger instead of terminalizing them (rung-12 census:
    six models were terminalized ``failed:resource/disk-floor`` from host
    disk state, burying honest negatives under an infrastructure artifact).
    Teardown residue keeps raising the base :class:`DiskRecoveryError`.
    """

    def __init__(
        self,
        message: str,
        *,
        floor_bytes: int,
        free_bytes: int,
        env_root: str,
        condition: str = "floor",
        destination_role: str = "env-root",
    ) -> None:
        """Carry the refused floor, the measurement, the probed root, and its role.

        ``condition`` distinguishes the two refusal shapes: ``"floor"`` is a
        pre-write measurement below the configured minimum; ``"enospc"`` is a
        write that ALREADY failed with ``ENOSPC`` (the failed write is the
        refusal evidence, so the accompanying measurement may legitimately sit
        at or above the floor -- quota, or another writer freeing space).
        ``destination_role`` names WHICH write destination was starved
        (``env-root``, ``backend-state``, ``tmp``): the T3-T3 live hole was a
        floor that watched the SSD at env_root while every backend write
        landed on the 23-GiB internal disk.
        """

        super().__init__(message)
        self.floor_bytes = floor_bytes
        self.free_bytes = free_bytes
        self.env_root = env_root
        self.condition = condition
        self.destination_role = destination_role


class EnvironmentCleanupError(EnvironmentLifecycleError):
    """Raised when removal of a completed environment fails."""


class EnvironmentExactnessError(EnvironmentLifecycleError):
    """Raised when materialized environment bytes disagree with declared facts."""


@dataclass(frozen=True)
class ArtifactReceipt:
    """One solver artifact whose digest was verified from materialized bytes."""

    url: str
    sha256: str


@dataclass(frozen=True)
class SolveResult:
    """Exact bytes and measured effort returned by an on-target solver."""

    lock_bytes: bytes
    resolved_export_bytes: bytes
    elapsed_seconds: float
    artifact_bytes: int
    artifact_receipts: tuple[ArtifactReceipt, ...] = ()


@dataclass(frozen=True)
class ProbeResult:
    """Result of one declared environment probe."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class LifecycleResult:
    """Completed sequential lifecycle metrics and generated artifacts."""

    intent: str
    target: str
    export_sha256: str
    probe_results: tuple[ProbeResult, ...]
    disk_before: int
    disk_after_create: int
    disk_after_teardown: int
    disk_recovery_checked: bool
    #: Free-space bytes that moved during the lifecycle for reasons OTHER than
    #: environment residue (the prefix was verified gone). Ambient consumption
    #: is reported, never treated as incomplete cleanup.
    ambient_disk_delta_bytes: int = 0


@dataclass(frozen=True)
class DiskSpaceObservation:
    """One measured filesystem destination governed by the disk authority."""

    role: str
    destination: Path
    probe: Path
    free_bytes: int | None
    error: str | None = None


@dataclass(frozen=True)
class DiskSpaceSnapshot:
    """One complete observation from the crawler's disk-space authority."""

    destinations: tuple[DiskSpaceObservation, ...]
    authority: DiskSpaceObservation | None


class EnvironmentBackend(Protocol):
    """Backend boundary implemented by conda tooling or a test fake.

    A backend MAY additionally expose
    ``overlay(prefix, manifest_path) -> Mapping[str, Any]`` installing one
    committed pip-overlay manifest strictly offline into the created prefix and
    returning the typed receipt the lifecycle validates against the manifest.
    An intent that DECLARES an overlay refuses to run against a backend without
    the method: a declared overlay never degrades to a conda-only runtime.

    A backend MAY additionally expose ``owned_paths(prefix) -> Sequence[Path]``
    enumerating EVERY generation-owned filesystem root it materialized for the
    prefix -- the prefix itself plus any dedicated scratch, package cache, or
    registration state. The lifecycle captures that receipt after ``create``
    and absence-checks every entry after ``remove``: a backend that reports a
    successful removal while leaving a dedicated root behind is real residue,
    not ambient disk consumption (rung-11 review finding 4). A backend without
    the method is checked on the prefix alone.
    """

    def solve(self, environment_file: Path, target: str) -> SolveResult:
        """Solve direct dependencies on the actual target machine."""

        ...

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        """Create an immutable environment and return its installed inventory bytes."""

        ...

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        """Run all declared imports, export checks, and build canaries."""

        ...

    def remove(self, prefix: Path) -> None:
        """Remove only the named environment and its dedicated scratch/cache."""

        ...


DiskFree = Callable[[Path], int]
UseEnvironment = Callable[[Path, tuple[ProbeResult, ...]], None]
IntentEffort = Callable[[EnvironmentIntent], EffortTracker]


def _backend_owned_paths(backend: EnvironmentBackend, prefix: Path) -> tuple[Path, ...]:
    """Return the residue-check inventory for one materialized environment.

    Parameters
    ----------
    backend:
        Active environment backend.
    prefix:
        Materialized environment prefix.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Every generation-owned root the backend enumerates (always including
        the prefix), or the prefix alone for a backend without the receipt.
    """

    enumerate_owned = getattr(backend, "owned_paths", None)
    if enumerate_owned is None:
        return (prefix,)
    owned = tuple(Path(value) for value in enumerate_owned(prefix))
    return owned if prefix in owned else (prefix, *owned)


#: Closed record identity for one durable environment-turn effort outcome row.
ENVIRONMENT_TURN_EFFORT_RECORD_VERSION = "menagerie.crawler.environment-turn-effort.v1"

#: Closed outcome vocabulary for environment-turn effort rows. ``attempt`` is
#: appended when a turn's admission charge lands, ``success`` when the turn's
#: environment materialized and probed against its committed receipt, and
#: ``cleared`` when the driver honors an operator clearance receipt for an
#: effort-exhaustion park (the row NAMES the receipt's event id).
_ENVIRONMENT_TURN_OUTCOMES = frozenset({"attempt", "success", "cleared"})


def environment_turn_effort_path(records_root: Path) -> Path:
    """Return the durable append-only environment-turn effort ledger path.

    Parameters
    ----------
    records_root:
        Campaign canonical records root.

    Returns
    -------
    pathlib.Path
        Operational ledger recording every environment-turn effort outcome.
    """

    return Path(records_root) / "operational" / "environment-turn-effort.jsonl"


#: Closed campaign-health disposition for one evidenced env-turn ledger
#: torn-tail recovery (D-60). Per-occurrence identity: the torn bytes are
#: truncated under the data-inode flock, so one exact torn tail can only
#: ever be recovered -- and disclosed -- once.
ENVIRONMENT_TURN_TAIL_RECOVERY_DISPOSITION = "environment-turn-effort-tail-recovered"


def _decode_environment_turn_payload(
    payload: bytes, ledger_path: Path
) -> list[dict[str, Any]]:
    """Strictly decode complete turn-effort rows from whole-ledger bytes.

    A malformed line is hard corruption and refuses loudly -- silently
    skipping it could erase recorded attempts and re-permit the full-prefix
    rebuild churn this ledger exists to stop (BL-9b). This decoder is also
    the ``prefix_check`` handed to torn-tail recovery, so the bytes recovery
    may retain are exactly the bytes this reader accepts.
    """

    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.split(b"\n")):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EnvironmentLifecycleError(
                f"environment-turn effort ledger has a corrupt interior row at "
                f"{ledger_path}:{index + 1}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EnvironmentLifecycleError(
                f"environment-turn effort ledger row {index + 1} at {ledger_path} "
                "is not an object"
            )
        rows.append(value)
    return rows


def _disclose_environment_turn_tail_recovery(
    ledger_path: Path, evidence: TailRecoveryEvidence
) -> None:
    """Append the campaign-health disclosure for one evidenced tail recovery.

    Never a silent skip (D-60): the removed bytes are already durably
    evidenced in the recovery ledger, and this event surfaces the crash
    signature on the canonical operational ledger -- the sibling
    ``operational/events.jsonl`` under the same records root. An append
    failure propagates; a recovery that cannot be disclosed must not pass
    silently.
    """

    records_root = ledger_path.parent.parent
    events_path = operational_events_path(records_root)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
        "status": OperationalEventStatus.DEGRADED.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": "environment-turn-effort-recovery",
        "machine_id": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "details": {
            "disposition": ENVIRONMENT_TURN_TAIL_RECOVERY_DISPOSITION,
            "ledger_path": str(ledger_path),
            "byte_offset": evidence.byte_offset,
            "byte_count": evidence.byte_count,
            "tail_sha256": evidence.tail_sha256,
            "recovered_at": evidence.recovered_at,
            "evidence_path": str(evidence.evidence_path),
        },
    }
    # Bounded wait mirrors the driver's own courtesy bound (T5-5): a closing
    # sibling writer is waited out; contention that outlives it stays typed.
    with JsonlLedger(
        events_path, OPERATIONAL_EVENT_SCHEMA_VERSION, lock_timeout=10.0
    ) as ledger:
        ledger.append(mint_operational_event(event))


def _recover_environment_turn_tail(
    ledger_path: Path, *, data_lock_held: bool = False
) -> TailRecoveryEvidence | None:
    """Evidence, truncate, and disclose one torn turn-effort tail.

    Routes through :func:`menagerie.crawler.recordio.recover_torn_tail`: the
    removed bytes land in the durable recovery evidence ledger BEFORE the
    truncation, a complete malformed retained row refuses with this module's
    own typed error and nothing is truncated, and a live writer's data-inode
    flock refuses recovery typed rather than truncating an in-flight append.
    """

    evidence = recover_torn_tail(
        ledger_path,
        prefix_check=lambda retained: _decode_environment_turn_payload(
            retained.read_bytes(), ledger_path
        ),
        data_lock_held=data_lock_held,
    )
    if evidence is not None:
        _disclose_environment_turn_tail_recovery(ledger_path, evidence)
    return evidence


def recover_environment_turn_effort_ledger(
    records_root: Path,
) -> TailRecoveryEvidence | None:
    """Heal a crash-torn turn-effort ledger tail at startup, evidenced.

    Called at lane build, BEFORE the driver opens its long-lived operational
    ledger handle: the effort ledger's only writer is the DriverLock-excluded
    driver itself, so a torn tail can only exist at process start, and healing
    here lets the disclosure append to an uncontended events ledger.

    Parameters
    ----------
    records_root:
        Campaign canonical records root.

    Returns
    -------
    TailRecoveryEvidence | None
        Evidence for a recovered tail, or ``None`` for a complete ledger.
    """

    ledger_path = environment_turn_effort_path(records_root)
    try:
        content = ledger_path.read_bytes()
    except FileNotFoundError:
        return None
    if not content or content.endswith(b"\n"):
        return None
    return _recover_environment_turn_tail(ledger_path)


def _read_environment_turn_rows(ledger_path: Path) -> list[dict[str, Any]]:
    """Read the turn-effort ledger, recovering only an evidenced torn tail.

    A crash mid-append leaves at most one non-newline-terminated tail; it is
    evidenced and truncated through recordio's recovery -- durable evidence
    row plus a campaign-health operational event -- NEVER silently dropped
    (D-60: a silent drop un-counted one env-turn attempt per crash, and the
    exposure compounds under a crash loop). A malformed INTERIOR line is hard
    corruption and refuses loudly (BL-9b).
    """

    try:
        content = ledger_path.read_bytes()
    except FileNotFoundError:
        return []
    if content and not content.endswith(b"\n"):
        _recover_environment_turn_tail(ledger_path)
        content = ledger_path.read_bytes()
    return _decode_environment_turn_payload(content, ledger_path)


def _append_environment_turn_row(ledger_path: Path, row: Mapping[str, Any]) -> None:
    """Durably append one turn-effort outcome row before control flow proceeds.

    Valve discipline (D-60): the append holds an exclusive flock on the
    ledger's data inode -- the same single-writer exclusion the sibling
    hand-rolled valve ledgers hold, and the flock torn-tail recovery probes
    non-blocking before truncating -- and a torn tail left by a prior crash
    is evidenced and truncated BEFORE the new row lands, so an outcome row
    can never weld onto a torn fragment as one malformed complete line.
    """

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with ledger_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = handle.read()
        if existing and not existing.endswith(b"\n"):
            _recover_environment_turn_tail(ledger_path, data_lock_held=True)
        handle.seek(0, os.SEEK_END)
        handle.write(payload)
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(ledger_path.parent)


def replay_environment_turn_attempts(
    ledger_path: Path, key: tuple[str, str, str | None]
) -> int:
    """Rederive one intent-generation's live attempt count from the ledger.

    Replay-derived, never producer-flagged: the count is recomputed on every
    resume exclusively from the append-only outcome rows -- attempts since the
    most recent ``success`` or ``cleared`` outcome for the exact
    ``(intent, target, generation)`` tuple.

    Parameters
    ----------
    ledger_path:
        Durable turn-effort ledger.
    key:
        Exact ``(intent, target, generation)`` scope.

    Returns
    -------
    int
        Attempts accumulated since the scope last succeeded or was cleared.
    """

    intent_name, target, generation = key
    attempts = 0
    for row in _read_environment_turn_rows(ledger_path):
        if (
            row.get("intent") != intent_name
            or row.get("target") != target
            or row.get("generation") != generation
        ):
            continue
        outcome = row.get("outcome")
        if outcome == "attempt":
            attempts += 1
        elif outcome in ("success", "cleared"):
            attempts = 0
    return attempts


class _DurableEnvironmentEffortTracker(EffortTracker):
    """Effort tracker that mirrors environment-turn outcomes into a ledger.

    Ordering is deliberate on ``consume``: the in-memory cap check runs FIRST
    (a refused admission runs no build and appends nothing), and the durable
    ``attempt`` row lands BEFORE the caller proceeds to the backend, so a
    crash mid-build still counts the attempt on resume (BL-9b). An append
    failure raises before any build spends disk or wall clock.
    """

    def __init__(
        self,
        caps: Mapping[str, StageCap],
        *,
        ledger_path: Path,
        key: tuple[str, str, str | None],
    ) -> None:
        super().__init__(caps)
        self._ledger_path = ledger_path
        self._key = key

    def _append_outcome(self, outcome: str, **extra: Any) -> None:
        intent_name, target, generation = self._key
        _append_environment_turn_row(
            self._ledger_path,
            {
                "record_version": ENVIRONMENT_TURN_EFFORT_RECORD_VERSION,
                "intent": intent_name,
                "target": target,
                "generation": generation,
                "outcome": outcome,
                **extra,
            },
        )

    def consume(
        self,
        stage: str | FailureStage,
        *,
        attempts: int = 0,
        seconds: float = 0.0,
        bytes_used: int = 0,
    ) -> EffortUsage:
        usage = super().consume(stage, attempts=attempts, seconds=seconds, bytes_used=bytes_used)
        if attempts > 0:
            self._append_outcome("attempt")
        return usage

    def record_success(self, stage: str | FailureStage) -> None:
        super().record_success(stage)
        self._append_outcome("success")

    def reset_after_clearance(self, release_event_id: str) -> None:
        """Restore the budget on an operator clearance receipt (BL-9a/c).

        The receipt is the operator's evidence-bearing statement that the
        setup was repaired; leaving the exhausted count in place would re-park
        the intent on its very next admission, making the clearance a no-op.
        ``release_event_id`` names the cleared PARK event, binding the reset
        row to authenticated evidence and keying its replay idempotency.
        """

        super().record_success("environment")
        self._append_outcome("cleared", release_event_id=release_event_id)

    def clearance_already_applied(self, release_event_id: str) -> bool:
        """Return whether a reset row for this park event already exists."""

        intent_name, target, generation = self._key
        for row in _read_environment_turn_rows(self._ledger_path):
            if (
                row.get("intent") == intent_name
                and row.get("target") == target
                and row.get("generation") == generation
                and row.get("outcome") == "cleared"
                and row.get("release_event_id") == release_event_id
            ):
                return True
        return False


class PerIntentGenerationEffort:
    """Own an independent environment effort tracker per intent-generation.

    With ``ledger_path`` configured, every turn outcome (attempt, success,
    clearance) is mirrored into a durable append-only ledger and each
    tracker's live attempt count is rederived from it on first access, so
    env-turn admission survives a supervisor/wake restart instead of
    re-permitting full-prefix env-build churn (BL-9b). Without a ledger the
    provider keeps its historical in-memory behavior (tests).
    """

    def __init__(self, cap: StageCap, *, ledger_path: Path | None = None) -> None:
        """Configure the common cap applied independently to every scope.

        Parameters
        ----------
        cap:
            Environment cap cloned into each lazily observed intent-generation.
        ledger_path:
            Optional durable turn-effort ledger for restart-safe admission.
        """

        self._cap = cap
        self._ledger_path = None if ledger_path is None else Path(ledger_path)
        self._trackers: dict[tuple[str, str, str | None], EffortTracker] = {}

    def __call__(self, intent: EnvironmentIntent) -> EffortTracker:
        """Return the tracker for one exact intent-generation.

        Parameters
        ----------
        intent:
            Immutable environment intent and target generation.

        Returns
        -------
        EffortTracker
            Independent tracker for the intent, target, and generation tuple.
        """

        key = (intent.name, intent.lock.target, intent.generation)
        if key not in self._trackers:
            if self._ledger_path is None:
                self._trackers[key] = EffortTracker({"environment": self._cap})
            else:
                tracker = _DurableEnvironmentEffortTracker(
                    {"environment": self._cap},
                    ledger_path=self._ledger_path,
                    key=key,
                )
                replayed = replay_environment_turn_attempts(self._ledger_path, key)
                if replayed:
                    tracker.restore_usage("environment", EffortUsage(attempts=replayed))
                self._trackers[key] = tracker
        return self._trackers[key]

    def reset_after_clearance(self, intent: EnvironmentIntent, release_event_id: str) -> bool:
        """Restore one intent-generation's budget for an operator clearance.

        Idempotent per cleared park event: a reset that already landed in the
        durable ledger (replay-derived, never a flag) is not applied twice, so
        a genuine post-clearance failed attempt is never silently erased by a
        later leg re-observing the same old receipt.

        Parameters
        ----------
        intent:
            Cleared environment intent.
        release_event_id:
            Content-addressed id of the cleared park event.

        Returns
        -------
        bool
            Whether anything was reset.
        """

        tracker = self(intent)
        already_applied = getattr(tracker, "clearance_already_applied", None)
        if already_applied is not None and already_applied(release_event_id):
            return False
        if tracker.usage("environment") == EffortUsage():
            return False
        reset = getattr(tracker, "reset_after_clearance", None)
        if reset is not None:
            reset(release_event_id)
        else:
            tracker.record_success("environment")
        return True


def disk_free_bytes(path: Path) -> int:
    """Return available filesystem bytes at a path.

    Bytes, never blocks: ``f_bavail`` (blocks available to unprivileged
    writers) times ``f_frsize`` (the fragment size those counts are in), the
    same arithmetic ``shutil.disk_usage().free`` uses. A blocks-vs-bytes slip
    here mis-sets every configured threshold by the ~4096x block factor
    (T5-8 sibling finding), so the equivalence is pinned by test.

    Parameters
    ----------
    path:
        Existing path on the environment filesystem.

    Returns
    -------
    int
        Available bytes.
    """

    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def _nearest_existing(path: Path) -> Path:
    """Return the closest existing ancestor a filesystem probe can measure."""

    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


class DiskSpaceAuthority:
    """Measure all real write destinations through one conservative authority.

    The authoritative observation is the successfully measured destination
    with the least free space. This keeps the floor honest when the environment
    store, operator state, and temporary staging land on different filesystems,
    while giving lifecycle and launch preflight one shared selection rule.
    """

    def __init__(
        self,
        destinations: Sequence[tuple[str, Path]],
        *,
        disk_free: DiskFree = disk_free_bytes,
    ) -> None:
        """Bind the actual write destinations and free-byte measurement.

        Parameters
        ----------
        destinations:
            Ordered ``(role, path)`` pairs receiving crawler writes.
        disk_free:
            Free-byte measurement, injectable for deterministic tests.
        """

        if not destinations:
            raise ValueError("disk authority requires at least one write destination")
        self._destinations = tuple((role, Path(path)) for role, path in destinations)
        self._disk_free = disk_free

    @property
    def destinations(self) -> tuple[tuple[str, Path], ...]:
        """Return the bound write destinations in authority order."""

        return self._destinations

    def snapshot(self) -> DiskSpaceSnapshot:
        """Measure each filesystem once and select the lowest-free authority.

        Returns
        -------
        DiskSpaceSnapshot
            Deduplicated destination observations and the conservative
            lowest-free successful measurement. Measurement errors remain in
            the snapshot so callers can fail closed with their own typed error.
        """

        observations: list[DiskSpaceObservation] = []
        probed: set[Path] = set()
        for role, destination in self._destinations:
            probe = _nearest_existing(destination)
            if probe in probed:
                continue
            probed.add(probe)
            try:
                free = self._disk_free(probe)
            except OSError as exc:
                observations.append(DiskSpaceObservation(role, destination, probe, None, str(exc)))
            else:
                observations.append(DiskSpaceObservation(role, destination, probe, int(free)))
        successful = tuple(row for row in observations if row.free_bytes is not None)
        authority = (
            min(successful, key=lambda row: int(row.free_bytes or 0)) if successful else None
        )
        return DiskSpaceSnapshot(tuple(observations), authority)


class SequentialEnvironmentLifecycle:
    """Own exactly one active environment from target solve through teardown."""

    def __init__(
        self,
        backend: EnvironmentBackend,
        effort: EffortTracker | IntentEffort,
        *,
        env_root: Path,
        disk_free: DiskFree = disk_free_bytes,
        minimum_free_bytes: int = DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES,
        recovery_tolerance_bytes: int = 1024**2,
        tmp_root: Path | None = None,
    ) -> None:
        """Configure the backend, effort caps, roots, and disk invariants."""

        self._backend = backend
        self._effort_provider = effort
        self._effort_for_intent = effort if callable(effort) else lambda _intent: effort
        self._env_root = env_root
        self._disk_free = disk_free
        self._minimum_free_bytes = minimum_free_bytes
        self._recovery_tolerance_bytes = recovery_tolerance_bytes
        self._tmp_root = Path(tempfile.gettempdir()) if tmp_root is None else Path(tmp_root)
        self._active: Path | None = None
        self._authority_cache: EnvironmentAuthorityCache | None = None
        # The busy guard is a real mutex, not a check-then-act (T5-5): the env
        # lane is a host singleton (T3-T8) and a second concurrent `run` must
        # deterministically observe `EnvironmentBusyError`, never a race
        # through the validation window before the prefix claim lands.
        self._activation_mutex = threading.Lock()
        self._activating = False

    @property
    def tmp_root(self) -> Path:
        """Return the temporary write root bound into the disk authority."""

        return self._tmp_root

    def reset_effort_after_clearance(
        self, intent: EnvironmentIntent, release_event_id: str
    ) -> bool:
        """Restore one intent's turn budget for an operator clearance receipt.

        BL-9c: an effort-exhaustion park is released by an explicit clearance
        receipt, and the very next admission would re-raise (and re-park) off
        the durably exhausted count unless the budget is restored with it.
        Delegates to the effort provider when it supports clearance resets; a
        plain per-lifecycle tracker (tests) resets in memory.

        Parameters
        ----------
        intent:
            Cleared environment intent.
        release_event_id:
            Content-addressed id of the cleared park event.

        Returns
        -------
        bool
            Whether any accumulated budget was actually reset.
        """

        reset = getattr(self._effort_provider, "reset_after_clearance", None)
        if reset is not None:
            return bool(reset(intent, release_event_id))
        tracker = self._effort_for_intent(intent)
        if tracker.usage("environment") == EffortUsage():
            return False
        tracker.record_success("environment")
        return True

    @property
    def environment_authority_cache(self) -> EnvironmentAuthorityCache:
        """Return the cache owned by the currently active immutable prefix.

        Returns
        -------
        EnvironmentAuthorityCache
            Sole collector shared by scheduling, currentness, recovery, and spawn checks.

        Raises
        ------
        EnvironmentBusyError
            If no prefix is active or its cache has already been invalidated.
        """

        if self._active is None or self._authority_cache is None:
            raise EnvironmentBusyError("no active environment authority cache exists")
        return self._authority_cache

    def active_authority_cache(self, prefix: Path) -> EnvironmentAuthorityCache:
        """Return the lifecycle cache only for its exact active prefix.

        Parameters
        ----------
        prefix:
            Materialized prefix entering the lifecycle's use callback.

        Returns
        -------
        EnvironmentAuthorityCache
            Active lifecycle-owned collector.

        Raises
        ------
        EnvironmentBusyError
            If the requested prefix is not the active lifecycle prefix.
        """

        if self._active is None or prefix.resolve(strict=True) != self._active.resolve(strict=True):
            raise EnvironmentBusyError(
                "environment authority cache requested for an inactive prefix"
            )
        return self.environment_authority_cache

    def _disk_space_authority(self) -> DiskSpaceAuthority:
        """Return the one authority over every actual lane write destination.

        T3-T3 (3-lab): the floor used to statvfs env_root alone -- a symlink
        to the 931-GiB SSD -- while the artifact CAS, package cache,
        wheelhouse, and TMPDIR all landed on the ~23-GiB internal disk, so a
        real ENOSPC never tripped the floor. A backend MAY expose
        ``write_destinations() -> Sequence[Path]`` enumerating its dedicated
        state roots; the temporary directory is always probed because
        conda-lock and staging write there.

        Returns
        -------
        DiskSpaceAuthority
            Shared authority over the environment root, every backend state
            root, and the configured temporary staging root.
        """

        roots: list[tuple[str, Path]] = [("env-root", self._env_root)]
        enumerate_destinations = getattr(self._backend, "write_destinations", None)
        if enumerate_destinations is not None:
            for value in enumerate_destinations():
                roots.append(("backend-state", Path(value)))
        roots.append(("tmp", self._tmp_root))
        return DiskSpaceAuthority(roots, disk_free=self._disk_free)

    def _disk_space_snapshot(self) -> DiskSpaceSnapshot:
        """Measure the authority and fail closed if any destination is unreadable.

        Returns
        -------
        DiskSpaceSnapshot
            Complete authority observation with a lowest-free destination.

        Raises
        ------
        OSError
            If any actual write destination cannot be measured.
        """

        snapshot = self._disk_space_authority().snapshot()
        failures = tuple(row for row in snapshot.destinations if row.error is not None)
        if failures:
            listed = "; ".join(f"{row.role} at {row.probe}: {row.error}" for row in failures)
            raise OSError(f"disk authority could not measure write destination(s): {listed}")
        if snapshot.authority is None:
            raise OSError("disk authority produced no filesystem measurement")
        return snapshot

    def _assert_disk_floor(self) -> None:
        """Refuse typed when ANY write destination sits below the disk floor.

        Each destination's nearest existing ancestor is measured once (a
        destination that resolves to an already-probed path is not measured
        twice). A zero floor has nothing to enforce and probes nothing.

        Raises
        ------
        DiskFloorError
            Naming the starved destination, its role, and the measurement.
        """

        if self._minimum_free_bytes <= 0:
            return
        authority = self._disk_space_snapshot().authority
        if authority is None or authority.free_bytes is None:
            raise OSError("disk authority produced no free-byte measurement")
        if authority.free_bytes < self._minimum_free_bytes:
            raise DiskFloorError(
                f"less than the required {self._minimum_free_bytes} bytes of "
                f"free disk is available at {authority.probe} "
                f"({authority.free_bytes} bytes free; authoritative write "
                f"destination role: {authority.role})",
                floor_bytes=self._minimum_free_bytes,
                free_bytes=authority.free_bytes,
                env_root=str(authority.probe),
                condition="floor",
                destination_role=authority.role,
            )

    def _floor_after_enospc(self, exc: OSError) -> DiskFloorError:
        """Type one observed ENOSPC write failure as the disk-floor condition.

        The failed write itself is the refusal evidence, so the accompanying
        measurement is reported verbatim even when it sits at or above the
        floor (quota exhaustion, or another writer freeing space between the
        failure and the probe). The reported destination is the one measuring
        the LEAST free bytes -- the best available guess at which filesystem
        filled.

        Parameters
        ----------
        exc:
            The observed ``ENOSPC`` failure.

        Returns
        -------
        DiskFloorError
            Typed ``condition="enospc"`` refusal for the driver's floor park.
        """

        snapshot = self._disk_space_authority().snapshot()
        authority = snapshot.authority
        starved_role = authority.role if authority is not None else "env-root"
        starved_probe = (
            authority.probe if authority is not None else _nearest_existing(self._env_root)
        )
        minimum = authority.free_bytes if authority is not None else None
        return DiskFloorError(
            f"environment write failed with ENOSPC ({exc}); least free write "
            f"destination is {starved_probe} at {minimum if minimum is not None else 0} "
            "bytes",
            floor_bytes=max(self._minimum_free_bytes, 1),
            free_bytes=minimum if minimum is not None else 0,
            env_root=str(starved_probe),
            condition="enospc",
            destination_role=starved_role,
        )

    def run(
        self,
        intent: EnvironmentIntent,
        *,
        use: UseEnvironment,
    ) -> LifecycleResult:
        """Create, probe, use, teardown, and verify one committed environment.

        The committed, provenance-bound lock family is the authoritative BUILD
        INPUT: the prefix is created offline from ``intent.lock.lock_path``
        and validated byte-for-byte against the committed resolved export.
        The lane never solves and never rewrites the committed family --
        solving is a rotation-only operation
        (``tools/rotate_setup_artifacts.py``). Runtime probes compare against
        the committed receipt byte-for-byte; drift refuses and requires a new
        rotation instead of rewriting setup authority.

        Parameters
        ----------
        intent:
            Loaded intent whose committed lock family is the build input.
        use:
            Driver callback that processes assigned models sequentially.

        Returns
        -------
        LifecycleResult
            Completed lifecycle evidence.

        Raises
        ------
        EnvironmentBusyError
            If another environment remains active.
        CommittedLockFamilyError
            If the intent has no proven committed family to build from
            (``unlocked``, ``incomplete``, ``hash-mismatch``, ``superseded``,
            or mid-publication ``publishing``).
        EnvironmentLifecycleError
            If create, probe, or disk invariants fail. Teardown residue (the
            prefix still present after a claimed removal) is
            :class:`DiskRecoveryError`; an ambient free-space delta with the
            prefix verified gone is reported, never raised.
        EffortCapExceeded
            If build work exceeds the configured environment cap.
        """

        with self._activation_mutex:
            if self._activating or self._active is not None:
                raise EnvironmentBusyError(
                    f"environment already active at {self._active or '<activating>'}"
                )
            self._activating = True
        try:
            return self._run_activated(intent, use=use)
        finally:
            with self._activation_mutex:
                self._activating = False

    def _run_activated(
        self,
        intent: EnvironmentIntent,
        *,
        use: UseEnvironment,
    ) -> LifecycleResult:
        """Build, probe, drive, and tear down under an exclusive activation claim.

        Parameters
        ----------
        intent:
            Environment intent whose committed lock family builds the prefix.
        use:
            Callback receiving the sealed prefix and probe results.

        Returns
        -------
        LifecycleResult
            Disk accounting for the completed turn.
        """

        self._env_root.mkdir(parents=True, exist_ok=True)
        self._assert_disk_floor()
        disk_before = self._disk_free(self._env_root)
        family_status = intent.lock.status
        if family_status != "locked":
            raise CommittedLockFamilyError(
                f"intent {intent.name!r} has no proven committed lock family to "
                f"build from (status: {family_status}); the committed family is "
                "the build input and solving is rotation-only -- run "
                "tools/rotate_setup_artifacts.py to prove and publish one"
            )
        lock_bytes = intent.lock.lock_bytes or b""
        export_bytes = intent.lock.export_bytes or b""
        parse_exact_lock(lock_bytes)
        declared_inventory = parse_resolved_export(export_bytes)
        _require_lock_inventory_match(lock_bytes, declared_inventory)
        export_hash = hash_bytes(export_bytes)
        effort = self._effort_for_intent(intent)
        effort.consume("environment", attempts=1)
        # The committed-family flow's post-solve point: the build input is
        # validated and nothing is written yet. Disk that collapsed between
        # admission and here (the T3-T3 window) refuses typed BEFORE the
        # backend writes a single byte.
        self._assert_disk_floor()

        prefix = self._env_root / f"{intent.name}-{export_hash[7:19]}"
        self._active = prefix
        disk_after_create = disk_before
        probe_results: tuple[ProbeResult, ...] = ()
        owned_paths: tuple[Path, ...] = (prefix,)
        try:
            try:
                installed_inventory = self._backend.create(intent.lock.lock_path, prefix)
                if installed_inventory != declared_inventory:
                    raise EnvironmentExactnessError(
                        "created-prefix package inventory differs from the declared resolved export"
                    )
                self._apply_pip_overlay(intent, prefix)
                # The residue receipt of every generation-owned root is the
                # authority applied after teardown; it is captured AFTER the
                # overlay step so a generation-local wheelhouse joins the
                # inventory, and while the backend's dedicated state provably
                # exists.
                owned_paths = _backend_owned_paths(self._backend, prefix)
                disk_after_create = self._disk_free(self._env_root)
                raw_probe_results = tuple(self._backend.probe(prefix, intent.probes))
                probe_results = validate_probe_receipts(intent.probes, raw_probe_results)
                receipt_path = intent.lock.lock_path.with_name(f"{intent.lock.target}.probes.json")
                observed_receipt = canonical_probe_receipt_bytes(probe_results)
                try:
                    committed_receipt = receipt_path.read_bytes()
                except OSError as exc:
                    raise EnvironmentProbeError(
                        f"committed probe receipt is unavailable at {receipt_path}; "
                        "runtime cannot author setup authority -- rotate the intent"
                    ) from exc
                if committed_receipt != observed_receipt:
                    raise EnvironmentProbeError(
                        f"runtime probes differ from committed receipt {receipt_path}; "
                        "runtime cannot rewrite a sealed setup generation -- rotate "
                        "the intent"
                    )
                # BL-9a: the environment turn demonstrably succeeded -- the
                # prefix materialized offline from the committed family and
                # every probe matched the sealed receipt -- so the turn's
                # admission charge is returned. The cap bounds consecutive
                # GENUINE build/probe retries, never normal scheduling; before
                # this reset, ordinary successful turns exhausted the
                # attempts=2 cap and parked whole intents behind a human-only
                # release (six pilot firings). Later phases (use, teardown)
                # own their failures through the host-condition/cleanup arms
                # and never spend intent-setup retry budget.
                effort.record_success("environment")
            except OSError as exc:
                # The documented real mid-create death (T5-10): conda dies with
                # ENOSPC leaving a partial prefix. Disk exhaustion DURING the
                # build is the same machine condition the pre-write floor
                # refuses, so it carries the same type -- the driver parks it
                # behind the disk-floor valve instead of terminalizing models
                # -- and the teardown below removes the partial state, so a
                # later attempt never mistakes it for "built".
                if exc.errno == errno.ENOSPC:
                    raise self._floor_after_enospc(exc) from exc
                raise
            self._authority_cache = EnvironmentAuthorityCache()
            use(prefix, probe_results)
        finally:
            try:
                try:
                    self._backend.remove(prefix)
                except Exception as exc:
                    raise EnvironmentCleanupError(
                        f"environment removal failed for {intent.name}: {exc}"
                    ) from exc
            finally:
                if self._authority_cache is not None:
                    self._authority_cache.invalidate()
                    self._authority_cache = None
                self._active = None
        # Residue authority is the recorded artifact inventory -- the prefix
        # plus every generation-owned root the backend enumerated at create
        # time -- never a df delta. Free space moving for unrelated reasons
        # (ledger appends, diagnostics, other processes) is AMBIENT
        # consumption: reported and logged, but not a cleanup failure. Rung 11
        # quarantined a healthy generation, and fanned that quarantine out
        # over 21 models, from exactly such an ambient delta; the follow-up
        # review then proved prefix-only checking blind to a falsely
        # successful removal that leaves the dedicated package cache or
        # registration behind (finding 4).
        residue = tuple(path for path in owned_paths if path.exists() or path.is_symlink())
        if residue:
            listed = ", ".join(str(path) for path in residue)
            raise DiskRecoveryError(f"teardown left environment residue at {listed}")
        try:
            disk_after_teardown = self._disk_free(self._env_root)
        except Exception as exc:
            raise EnvironmentCleanupError(
                f"post-removal disk recovery measurement failed for {intent.name}: {exc}"
            ) from exc
        ambient_disk_delta = max(0, disk_before - disk_after_teardown)
        if ambient_disk_delta > self._recovery_tolerance_bytes:
            LOGGER.warning(
                "environment teardown for %s removed every generation-owned root, "
                "but ambient disk consumption moved %d free bytes during the "
                "lifecycle (before=%d, after=%d); this is not cleanup residue, "
                "proceeding",
                intent.name,
                ambient_disk_delta,
                disk_before,
                disk_after_teardown,
            )
        return LifecycleResult(
            intent=intent.name,
            target=intent.lock.target,
            export_sha256=export_hash,
            probe_results=probe_results,
            disk_before=disk_before,
            disk_after_create=disk_after_create,
            disk_after_teardown=disk_after_teardown,
            disk_recovery_checked=True,
            ambient_disk_delta_bytes=ambient_disk_delta,
        )

    def _apply_pip_overlay(self, intent: EnvironmentIntent, prefix: Path) -> None:
        """Install one intent's declared pip overlay into the created prefix.

        The overlay runs strictly BETWEEN creation and probing, so every
        declared probe -- including the overlay libraries' own import canaries
        -- exercises the fully overlaid runtime, and the probe receipt (a
        committed setup-artifact identity member) attests it.

        Parameters
        ----------
        intent:
            Loaded environment intent, possibly declaring a pip overlay.
        prefix:
            Created environment prefix.

        Raises
        ------
        EnvironmentOverlayError
            If the overlay is declared but unpinned, superseded, or invalid;
            if the backend cannot perform overlays; or if the backend receipt
            does not exactly match the committed manifest. All fail CLOSED: an
            overlay intent never silently runs without its overlay.
        """

        overlay = getattr(intent, "pip_overlay", None)
        if overlay is None:
            return
        status = overlay.status
        if status != "pinned":
            raise EnvironmentOverlayError(
                f"pip overlay for {intent.name!r} is {status}; rotate the intent to "
                "compile and commit its pinned manifest before running"
            )
        binding = getattr(overlay, "export_binding", None)
        if binding is None:
            # EN-8: an unbound manifest pairs with ANY conda solve -- refuse,
            # except the two exact committed legacy manifests (self-retiring
            # enumeration; see envs.LEGACY_UNBOUND_OVERLAY_MANIFESTS).
            if (
                hash_bytes(overlay.manifest_bytes or b"")
                not in LEGACY_UNBOUND_OVERLAY_MANIFESTS
            ):
                raise EnvironmentOverlayError(
                    f"pip overlay for {intent.name!r} carries no conda-resolved-export "
                    "binding; an unbound manifest could pair with any solve -- rotate "
                    "the intent so the manifest is compiled and bound against its own "
                    "conda export"
                )
        elif binding != hash_bytes(intent.lock.export_bytes or b""):
            raise EnvironmentOverlayError(
                f"pip overlay for {intent.name!r} was compiled against a different conda "
                "resolved export than the committed family; rotate the intent so the "
                "conda solve and its pip closure are proven together"
            )
        apply_overlay = getattr(self._backend, "overlay", None)
        if apply_overlay is None:
            raise EnvironmentOverlayError(
                f"intent {intent.name!r} declares a pip overlay but the environment "
                "backend has no overlay action; a declared overlay never degrades "
                "to a conda-only runtime"
            )
        receipt = apply_overlay(prefix, overlay.manifest_path)
        if not isinstance(receipt, Mapping):
            raise EnvironmentOverlayError(
                f"environment backend returned a malformed overlay receipt for {intent.name!r}"
            )
        manifest_bytes = overlay.manifest_bytes or b""
        if receipt.get("manifest_sha256") != hash_bytes(manifest_bytes):
            raise EnvironmentOverlayError(
                f"overlay receipt for {intent.name!r} attests a different manifest "
                "than the committed one"
            )
        expected_rows = [dict(row) for row in overlay.rows()]
        observed_rows = receipt.get("packages")
        if observed_rows != expected_rows:
            raise EnvironmentOverlayError(
                f"overlay receipt for {intent.name!r} installed rows differ from "
                "the committed manifest"
            )


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOCK_LINE_PATTERN = re.compile(r"(?P<url>\S+?)(?:#sha256=|#)(?P<digest>[0-9a-f]{64})\Z")
_PACKAGE_FIELDS = ("name", "version", "build", "url", "sha256")


def parse_exact_lock(content: bytes) -> tuple[ArtifactReceipt, ...]:
    """Parse an explicit lock and require one canonical SHA-256 per artifact.

    Parameters
    ----------
    content:
        Exact lock bytes produced by the target solver.

    Returns
    -------
    tuple[ArtifactReceipt, ...]
        Ordered URL and digest receipts from the lock.

    Raises
    ------
    EnvironmentExactnessError
        If the lock is malformed, empty, unhashed, or contains duplicates.
    """

    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EnvironmentExactnessError("environment lock is not UTF-8") from exc
    receipts: list[ArtifactReceipt] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith(("#", "@EXPLICIT")):
            continue
        match = _LOCK_LINE_PATTERN.fullmatch(value)
        if match is None:
            raise EnvironmentExactnessError(
                f"environment lock entry lacks a canonical SHA-256 digest: {value!r}"
            )
        receipts.append(ArtifactReceipt(match.group("url"), f"sha256:{match.group('digest')}"))
    if not receipts:
        raise EnvironmentExactnessError("environment lock has no artifact entries")
    if len({receipt.url for receipt in receipts}) != len(receipts):
        raise EnvironmentExactnessError("environment lock contains duplicate artifact URLs")
    return tuple(receipts)


def parse_resolved_export(content: bytes) -> bytes:
    """Parse and canonicalize the declared installed-package export.

    Parameters
    ----------
    content:
        Declared resolved-export bytes.

    Returns
    -------
    bytes
        Canonical installed inventory bytes suitable for byte comparison.

    Raises
    ------
    EnvironmentExactnessError
        If the export is not the closed exact package-inventory format.
    """

    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentExactnessError("resolved export is not valid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {"packages"}:
        raise EnvironmentExactnessError(
            "resolved export must contain exactly one packages inventory"
        )
    return _canonical_package_inventory(value.get("packages"), "resolved export")


def installed_package_inventory_bytes(prefix: Path) -> bytes:
    """Derive canonical package evidence only from immutable ``conda-meta`` files.

    Parameters
    ----------
    prefix:
        Created environment prefix.

    Returns
    -------
    bytes
        Canonical package inventory derived from installed metadata.

    Raises
    ------
    EnvironmentExactnessError
        If installed metadata is absent, malformed, lacks exact artifact hashes,
        or contradicts a declared Python-distribution provision.
    """

    metadata_paths = sorted((prefix / "conda-meta").glob("*.json"))
    if not metadata_paths:
        raise EnvironmentExactnessError(f"installed conda metadata is missing below {prefix}")
    rows: list[Mapping[str, Any]] = []
    for path in metadata_paths:
        try:
            value = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EnvironmentExactnessError(
                f"installed package metadata is invalid: {path}"
            ) from exc
        if not isinstance(value, Mapping):
            raise EnvironmentExactnessError(f"installed package metadata must be an object: {path}")
        _require_declared_provisions(value, path)
        rows.append(value)
    return _canonical_package_inventory(rows, "installed conda metadata")


def _require_declared_provisions(record: Mapping[str, Any], path: Path) -> None:
    """Re-prove the machine-owned provision registry against installed evidence.

    The R1 artifact digest is resolved PRE-gate from a static inventory whose
    package names live in the packaging system's namespace, while the recipe
    names a Python distribution. The registry that bridges the two spellings is
    a cached derivation, so this is where the cache is checked against the only
    authority on it: the file list the package actually installed. A registry
    entry that has gone stale stops environment creation here rather than
    mis-selecting a digest later.

    Parameters
    ----------
    record:
        One installed-package metadata record.
    path:
        Metadata path used in diagnostics.

    Raises
    ------
    EnvironmentExactnessError
        If the record contradicts a declared provision.
    """

    name = record.get("name")
    if not isinstance(name, str) or not name:
        return
    files = record.get("files")
    listed = (
        tuple(item for item in files if isinstance(item, str))
        if isinstance(files, Sequence) and not isinstance(files, (str, bytes))
        else None
    )
    try:
        assert_inventory_provisions(name, listed)
    except PackageNamespaceError as exc:
        raise EnvironmentExactnessError(f"{exc} ({path})") from exc


def _canonical_package_inventory(value: Any, label: str) -> bytes:
    """Return canonical bytes for a closed exact package inventory.

    Parameters
    ----------
    value:
        Candidate package-row sequence.
    label:
        Error-context label.

    Returns
    -------
    bytes
        Canonical newline-terminated JSON inventory.

    Raises
    ------
    EnvironmentExactnessError
        If rows are absent, ambiguous, or lack immutable artifact fields.
    """

    if not isinstance(value, list) or not value:
        raise EnvironmentExactnessError(f"{label} has no package rows")
    rows: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise EnvironmentExactnessError(f"{label} contains a non-object package row")
        row: dict[str, str] = {}
        for field in _PACKAGE_FIELDS:
            item = raw.get(field)
            if not isinstance(item, str) or not item:
                raise EnvironmentExactnessError(f"{label} package row lacks nonempty {field}")
            row[field] = item
        if re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            row["sha256"] = f"sha256:{row['sha256']}"
        if _HASH_PATTERN.fullmatch(row["sha256"]) is None:
            raise EnvironmentExactnessError(
                f"{label} package row has a noncanonical SHA-256 digest"
            )
        rows.append(row)
    rows.sort(key=lambda row: tuple(row[field] for field in _PACKAGE_FIELDS))
    identities = [tuple(row[field] for field in _PACKAGE_FIELDS) for row in rows]
    if len(identities) != len(set(identities)):
        raise EnvironmentExactnessError(f"{label} contains duplicate package rows")
    return canonical_json_bytes({"packages": rows}) + b"\n"


def _require_lock_inventory_match(lock_bytes: bytes, package_bytes: bytes) -> None:
    """Require lock artifacts and installed/exported packages to be one-to-one.

    Parameters
    ----------
    lock_bytes, package_bytes:
        Exact target lock and canonical package-inventory bytes.

    Raises
    ------
    EnvironmentExactnessError
        If either side names an artifact URL/digest absent from the other.
    """

    lock_receipts = {(receipt.url, receipt.sha256) for receipt in parse_exact_lock(lock_bytes)}
    try:
        value = json.loads(package_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentExactnessError("canonical package inventory is invalid") from exc
    packages = value.get("packages") if isinstance(value, Mapping) else None
    if not isinstance(packages, list):
        raise EnvironmentExactnessError("canonical package inventory has no package rows")
    package_receipts = {
        (str(package.get("url")), str(package.get("sha256")))
        for package in packages
        if isinstance(package, Mapping)
    }
    if lock_receipts != package_receipts or len(package_receipts) != len(packages):
        raise EnvironmentExactnessError(
            "target lock artifacts differ from the created-prefix package inventory"
        )


def expected_probe_names(probes: IntentProbes) -> tuple[str, ...]:
    """Return the exact ordered receipt names for one declared probe contract.

    Parameters
    ----------
    probes:
        Declared import, export, and source-build probes.

    Returns
    -------
    tuple[str, ...]
        Closed ordered receipt-name set.
    """

    return (
        *(f"import:{name}" for name in probes.imports),
        *(f"export:{check.module}:{check.attribute}" for check in probes.export_checks),
        *(f"source-build:{build.name}" for build in probes.source_build),
    )


def validate_probe_receipts(
    probes: IntentProbes, receipts: Sequence[ProbeResult]
) -> tuple[ProbeResult, ...]:
    """Require one successful named receipt for every declared probe and no others.

    Parameters
    ----------
    probes, receipts:
        Declared probe contract and backend observations.

    Returns
    -------
    tuple[ProbeResult, ...]
        Receipts reordered to the canonical declaration order.

    Raises
    ------
    EnvironmentProbeError
        If the declared set is empty, duplicated, missing, extra, or failed.
    """

    expected = expected_probe_names(probes)
    if not expected or len(set(expected)) != len(expected):
        raise EnvironmentProbeError("environment probe contract is empty or ambiguous")
    by_name: dict[str, ProbeResult] = {}
    for receipt in receipts:
        if receipt.name in by_name:
            raise EnvironmentProbeError(f"duplicate environment probe receipt: {receipt.name}")
        by_name[receipt.name] = receipt
    if set(by_name) != set(expected):
        missing = sorted(set(expected) - set(by_name))
        extra = sorted(set(by_name) - set(expected))
        raise EnvironmentProbeError(
            f"environment probe receipts do not match intent: missing={missing}, extra={extra}"
        )
    failed = [name for name in expected if not by_name[name].passed]
    if failed:
        raise EnvironmentProbeError(f"environment probes failed: {', '.join(failed)}")
    return tuple(by_name[name] for name in expected)


def canonical_probe_receipt_bytes(receipts: Sequence[ProbeResult]) -> bytes:
    """Serialize canonical successful probe receipts for durable recomputation.

    Parameters
    ----------
    receipts:
        Already validated ordered receipts.

    Returns
    -------
    bytes
        Canonical newline-terminated receipt artifact.
    """

    return (
        canonical_json_bytes(
            {
                "probes": [
                    {"name": receipt.name, "passed": receipt.passed, "detail": receipt.detail}
                    for receipt in receipts
                ]
            }
        )
        + b"\n"
    )


def parse_probe_receipt_bytes(probes: IntentProbes, content: bytes) -> tuple[ProbeResult, ...]:
    """Parse a durable receipt artifact against its declared probe contract.

    Parameters
    ----------
    probes, content:
        Declared probes and committed canonical receipt bytes.

    Returns
    -------
    tuple[ProbeResult, ...]
        Exact validated receipts in declaration order.

    Raises
    ------
    EnvironmentProbeError
        If the artifact is malformed, noncanonical, or does not match the contract.
    """

    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentProbeError("environment probe receipt artifact is invalid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {"probes"}:
        raise EnvironmentProbeError("environment probe receipt artifact has an invalid schema")
    raw_receipts = value.get("probes")
    if not isinstance(raw_receipts, list):
        raise EnvironmentProbeError("environment probe receipt list is missing")
    receipts: list[ProbeResult] = []
    for raw in raw_receipts:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"name", "passed", "detail"}
            or not isinstance(raw.get("name"), str)
            or not isinstance(raw.get("passed"), bool)
            or not isinstance(raw.get("detail"), str)
        ):
            raise EnvironmentProbeError("environment probe receipt row is malformed")
        receipts.append(ProbeResult(str(raw["name"]), bool(raw["passed"]), str(raw["detail"])))
    validated = validate_probe_receipts(probes, receipts)
    if content != canonical_probe_receipt_bytes(validated):
        raise EnvironmentProbeError("environment probe receipt artifact is not canonical")
    return validated


def materialized_environment_generation(
    intent: EnvironmentIntent,
    *,
    lock_bytes: bytes,
    export_bytes: bytes,
    package_bytes: bytes,
    python_version: str,
    compiler_identity: str,
    sdk_identity: str,
    probe_results: Sequence[ProbeResult],
) -> str:
    """Recompute one generation solely from exact artifacts and canonical receipts.

    Parameters
    ----------
    intent:
        Registry-derived intent and probe contract.
    lock_bytes, export_bytes, package_bytes:
        Exact lock, declared export, and created-prefix package inventory bytes.
    python_version, compiler_identity, sdk_identity:
        Observed interpreter and toolchain facts.
    probe_results:
        Exact one-to-one declared probe receipts.

    Returns
    -------
    str
        Canonical environment generation.

    Raises
    ------
    EnvironmentExactnessError
        If committed artifacts do not form one exact environment.
    """

    parse_exact_lock(lock_bytes)
    if parse_resolved_export(export_bytes) != package_bytes:
        raise EnvironmentExactnessError(
            "created-prefix package inventory differs from the declared resolved export"
        )
    _require_lock_inventory_match(lock_bytes, package_bytes)
    # T1-16: a declared pip overlay is part of the materialized runtime, so its
    # pinned manifest joins the generation -- an overlay-only rotation must
    # change the materialized identity, or old runs keep reading "current".
    # Intents without a declared overlay hash the exact payload they always
    # have, so every existing generation is unchanged.
    overlay = getattr(intent, "pip_overlay", None)
    overlay_manifest_sha256: str | None = None
    if overlay is not None:
        if overlay.status != "pinned":
            raise EnvironmentExactnessError(
                f"pip overlay for {intent.name!r} is {overlay.status}; a "
                "materialized overlay runtime requires the pinned manifest"
            )
        binding = getattr(overlay, "export_binding", None)
        if binding is None:
            # EN-8: absent bindings refuse (see envs.LEGACY_UNBOUND_OVERLAY_MANIFESTS).
            if (
                hash_bytes(overlay.manifest_bytes or b"")
                not in LEGACY_UNBOUND_OVERLAY_MANIFESTS
            ):
                raise EnvironmentExactnessError(
                    f"pip overlay for {intent.name!r} carries no conda-resolved-export "
                    "binding; an unbound manifest proves no materialized identity"
                )
        elif binding != hash_bytes(export_bytes):
            raise EnvironmentExactnessError(
                f"pip overlay for {intent.name!r} binds a different conda resolved "
                "export than the materialized one"
            )
        overlay_manifest_sha256 = hash_bytes(overlay.manifest_bytes or b"")
    receipts = validate_probe_receipts(intent.probes, probe_results)
    probe_intent = {
        "imports": list(intent.probes.imports),
        "export_checks": [vars(value) for value in intent.probes.export_checks],
        "source_build": [vars(value) for value in intent.probes.source_build],
    }
    observed_probes = [
        {"name": result.name, "passed": result.passed, "detail": result.detail}
        for result in receipts
    ]
    platform_facts = {
        "target": intent.lock.target,
        "python": python_version,
        "compiler": compiler_identity,
        "sdk": sdk_identity,
        "packages_manifest_sha256": hash_bytes(package_bytes),
    }
    if overlay_manifest_sha256 is not None:
        platform_facts["pip_overlay_manifest_sha256"] = overlay_manifest_sha256
    return compute_env_generation(
        {
            "name": intent.name,
            "framework": intent.framework,
            "target": intent.lock.target,
            "channels": list(intent.channels),
            "dependencies": list(intent.dependencies),
            "probe_intent": probe_intent,
        },
        hash_bytes(lock_bytes),
        hash_bytes(export_bytes),
        platform_facts,
        observed_probes,
    )


def _atomic_write(path: Path, content: bytes) -> None:
    """Atomically replace one rotation-owned setup artifact and fsync it.

    Runtime lifecycle paths must never call this helper; the rotation
    publisher imports it to write staging and flat publication members.
    """

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        durable_fsync(handle.fileno())
    os.replace(temporary, path)
