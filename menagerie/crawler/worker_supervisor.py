"""Parent-owned argv-only subprocess isolation and resource observations."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import re
import resource
import secrets
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, Mapping, Optional, Sequence

from menagerie.crawler.authority import (
    EnvironmentVerificationToken,
    ExecutionReadManifestV2,
    ExecutionReadManifestV3,
    environment_read_capability,
    exact_read_capability,
    WorkerLease as WorkerLease,
    completion_line_for_raw_award_receipt,
    derive_parent_attestation,
)
from menagerie.crawler.constants import (
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    STDIO_TAIL_MAX_CHARS,
    FailureStage,
)
from menagerie.crawler.execution_lock import (
    DEFAULT_EXECUTION_LOCK_TIMEOUT_SECONDS,
    acquire_global_execution_flock,
    release_global_execution_flock,
)
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    durable_fsync,
    hash_bytes,
    stable_hash,
    utc_now,
)
from menagerie.crawler.policy import (
    _PARENT_ALLOWED_READ_PATHS_ENV,
    _PARENT_STANDARD_INPUT_ASSET_ENV,
    _MACOS_FILE_READ_DENY_RULE,
    _MODEL_DATA_SUFFIXES,
    HostTransportLibraryCapability,
    SandboxUnavailableError,
    _SYSTEM_DATA_READ_ROOTS,
    _SYSTEM_TIMEZONE_READ_ROOTS,
    _allowed_exact_or_derived_file,
    _linux_host_transport_library_capability,
    _linux_runtime_code_roots,
    _runtime_code_path_allowed,
    _runtime_import_metadata_path_allowed,
    _runtime_model_data_path,
    _runtime_native_code_path_allowed,
    _runtime_package_data_paths,
    _runtime_static_path_allowed,
    build_safe_environment,
    detect_os_sandbox,
    generate_macos_sandbox_profile,
    verify_execution_read_manifest,
    wrap_with_os_sandbox,
)

_DENIED_ERRNOS = ("EACCES", "ENETDOWN", "ENETUNREACH", "EPERM", "EROFS")
_WRITE_OPEN_FLAGS = ("O_APPEND", "O_CREAT", "O_RDWR", "O_TMPFILE", "O_TRUNC", "O_WRONLY")
_WRITE_SYSCALLS = frozenset(
    {
        "chmod",
        "chown",
        "creat",
        "fchmodat",
        "fchownat",
        "link",
        "linkat",
        "mkdir",
        "mkdirat",
        "mknod",
        "mknodat",
        "open",
        "openat",
        "openat2",
        "rename",
        "renameat",
        "renameat2",
        "rmdir",
        "symlink",
        "symlinkat",
        "truncate",
        "unlink",
        "unlinkat",
        "utime",
        "utimensat",
        "utimes",
    }
)
_SYSCALL_PATTERN = re.compile(r"(?:^|\s)([a-z][a-z0-9_]*)\(")
_QUOTED_PATH_PATTERN = re.compile(r'"((?:[^"\\]|\\.)*)"')
_OPEN_RESULT_PATTERN = re.compile(r"\)\s+=\s+(-?\d+)(?:<[^>]*>)?(?:\s|$)")
_RESUMED_TRACE_PATTERN = re.compile(
    r"^(?P<leader>\s*(?:(?:\[pid\s+)?(?P<pid>\d+)\]?\s+)?)"
    r"<\.\.\. (?P<syscall>[a-z][a-z0-9_]*) resumed>(?P<suffix>.*)$"
)
_TRACE_PID_PATTERN = re.compile(r"^\s*(?:\[pid\s+)?(?P<pid>\d+)\]?\s+")
_NON_FILE_DESCRIPTOR_PREFIXES = ("anon_inode:", "memfd:", "pipe:", "socket:")
_SPECIAL_READ_ROOTS = (Path("/dev"), Path("/proc"), Path("/sys"))
_SYSTEM_READ_FILES = frozenset(
    {
        Path("/etc/group"),
        Path("/etc/hosts"),
        Path("/etc/ld.so.cache"),
        Path("/etc/locale.alias"),
        Path("/etc/localtime"),
        Path("/etc/nsswitch.conf"),
        Path("/etc/passwd"),
        Path("/etc/resolv.conf"),
        # Linux platform-detection probes ubiquitous in packaging code; on
        # macOS the open fails ENOENT and the ATTEMPT must still be benign
        # (m10759's undisabled probe died here). The macOS /etc symlink
        # resolution is named alongside, mirroring the localtime treatment.
        Path("/etc/os-release"),
        Path("/private/etc/os-release"),
        Path("/usr/lib/os-release"),
        Path("/usr/share/locale/locale.alias"),
        # platform.mac_ver() reads the OS version record during ordinary imports
        # (huggingface_hub walks it while timm/transformers load). The Seatbelt
        # profile already allows every /System read as benign OS metadata; naming
        # the exact file here keeps the in-process and parent classifiers from
        # drifting stricter than that OS-sandbox authority.
        Path("/System/Library/CoreServices/SystemVersion.plist"),
    }
)
_TERMINAL_TRACE_PATTERN = re.compile(r"\+\+\+ (?:exited with|killed by) .+ \+\+\+$")
_MACOS_AUDIT_COMPLETION_MARKER = "MENAGERIE_MACOS_SANDBOX_AUDIT_COMPLETE_V1"
_MACOS_AUDIT_SENTINEL_PREFIX = ".menagerie-seatbelt-audit-"
_MACOS_AUDIT_DELIVERY_TIMEOUT_SECONDS = 30.0
_DARWIN_LIBOMP_REGISTRATION_PREFIX = "__KMP_REGISTERED_LIB_"
_AT_FDCWD = -2
_RENAME_SWAP = 0x00000002
_PARENT_COMPLETION_CHALLENGE_ENV = "MENAGERIE_PARENT_COMPLETION_CHALLENGE"
_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V1 "
_WORKER_COMPLETION_V2_PREFIX = "MENAGERIE_WORKER_COMPLETION_V2 "
_WORKER_COMPLETION_V3_PREFIX = "MENAGERIE_WORKER_COMPLETION_V3 "
#: Dedicated v3 completion channel: an atomic file in the receipt directory,
#: so nothing a model prints to stdout can touch award validity (BL-12(a)).
WORKER_COMPLETION_FILE_NAME = "completion.json"
WORKER_COMPLETION_FILE_VERSION = "menagerie.crawler.worker-completion-file.v1"
WORKER_COMPLETION_PROOF_VERSION = "menagerie.crawler.worker-completion-proof.v1"
#: Bounded window scanned for legacy stdout completion lines: the parent never
#: decodes a whole model-owned stream again (BL-12(a)/(h)).
_COMPLETION_WINDOW_BYTES = 64 * 1024
#: Chunk size for streaming stream-file hashes (BL-12(h)).
_STREAM_HASH_CHUNK_BYTES = 1024 * 1024
#: Bounded grace for reaping a signalled worker root before the parent records
#: a typed host fault instead of blocking forever (BL-12(b), C-19).
_WORKER_REAP_GRACE_SECONDS = 30.0
#: Bounded wait for post-reap process-group survivors to die (BL-12(c)).
_POST_REAP_SURVIVOR_GRACE_SECONDS = 2.0
#: Poll-loop iterations between process-group RSS sweeps (~0.5 s at the 10 ms
#: poll cadence): group enumeration costs one ``ps`` spawn on macOS.
_GROUP_RSS_SAMPLE_STRIDE = 50

#: DX-8/RE-8 soft-watermark parameters (JMT-adopted defaults, B3-XQ-4 / §6
#: C.12). The parent kills at the watermark BELOW the hard cap so the 10 ms
#: sampler can never lose the race to a fast allocator blowing straight past
#: the cap into host OOM (the suspected 8/23 host-crash mechanism): the kill
#: threshold is 85% of the enforced limit, with the headroom floored at
#: 256 MiB so small configured caps keep a real safety margin.
RSS_SOFT_WATERMARK_RATIO = 0.85
RSS_WATERMARK_HEADROOM_FLOOR_BYTES = 256 * 1024**2
#: Above this fraction of the kill threshold the parent sweeps the WHOLE
#: process group every iteration instead of every ``_GROUP_RSS_SAMPLE_STRIDE``:
#: near the cap, a child spawned between strided sweeps is exactly the
#: overshoot the watermark exists to bound.
RSS_GROUP_SAMPLE_PRESSURE_RATIO = 0.60


def effective_rss_kill_threshold_bytes(rss_limit_bytes: int) -> int:
    """Return the soft-watermark kill threshold for one enforced RSS cap.

    Parameters
    ----------
    rss_limit_bytes:
        The enforced hard cap (``<= 0`` disables RSS containment entirely).

    Returns
    -------
    int
        The parent-enforced kill threshold: 85% of the cap, keeping at least
        256 MiB of headroom below it where the cap is large enough to afford
        that floor (never more than half the cap on tiny configured limits).
    """

    if rss_limit_bytes <= 0:
        return 0
    headroom = max(
        int(rss_limit_bytes * (1.0 - RSS_SOFT_WATERMARK_RATIO)),
        min(RSS_WATERMARK_HEADROOM_FLOOR_BYTES, rss_limit_bytes // 2),
    )
    return max(1, rss_limit_bytes - headroom)


class WorkerUnreapableError(RuntimeError):
    """A signalled worker root could not be reaped within the bounded grace.

    A worker in uninterruptible sleep (Metal/MPS driver hang, D-state I/O)
    ignores SIGKILL until the kernel releases it. The pre-fix supervisor blocked
    on ``process.wait()`` forever -- the silent-stall class the sprint's I2
    invariant forbids. This typed HOST fault converts the hang into a bounded
    halt with evidence; it must never settle a model's head.

    Parameters
    ----------
    pid, pgid:
        Root worker PID and its bound process group.
    detail:
        Bounded parent-owned diagnosis including the kill outcome.
    """

    def __init__(self, pid: int, pgid: int, detail: str) -> None:
        super().__init__(
            f"worker root pid={pid} pgid={pgid} unreapable within bounded grace: {detail}"
        )
        self.pid = pid
        self.pgid = pgid
        self.detail = detail
_REQUEST_SHA256_ENV = "MENAGERIE_WORKER_REQUEST_SHA256"
_READ_MANIFEST_ID_ENV = "MENAGERIE_EXECUTION_READ_MANIFEST_ID"
_WORKER_LOCK_FD_ENV = "MENAGERIE_WORKER_LOCK_FD"
_LIFECYCLE_FD_ENV = "MENAGERIE_WORKER_LIFECYCLE_FD"
_GLOBAL_LOCK_FD_ENV = "MENAGERIE_WORKER_GLOBAL_LOCK_FD"
_WORKER_RESULT_VERSION = "menagerie.crawler.worker-result.v3"
_WORKER_DIAGNOSTIC_VERSION = "menagerie.crawler.worker-receipt.v1"
_RAW_AWARD_RECEIPT_VERSION = "menagerie.crawler.raw-award-receipt.v3"
_PARENT_ATTESTATION_VERSION = "menagerie.crawler.parent-attestation.v2"
#: DX-11/RE-12: v3 binds the process-group teardown facts into the attestation,
#: so a survivor-forged award is structurally impossible -- the parent mints
#: only v3 now; v2 remains readable for pre-rotation history.
_PARENT_ATTESTATION_VERSION_V3 = "menagerie.crawler.parent-attestation.v3"
_SHUTDOWN_CHILD_DURABILITY_EVENT_REGISTRY: Mapping[str, tuple[str, ...]] = {
    "worker-lease-started": ("child_pid", "child_start_token", "child_pgid"),
}
_WORKER_RESULT_KEYS = frozenset(
    {
        "result_version",
        "raw_award_receipt",
        "raw_award_receipt_sha256",
        "diagnostic",
        "result_sha256",
    }
)
_WORKER_DIAGNOSTIC_KEYS = frozenset(
    {
        "receipt_version",
        "stable_id",
        "source_identity",
        "recipe_revision",
        "observed_recipe_revision",
        "observed_adapter_sha256",
        "observed_code_manifest_sha256",
        "observed_input_asset_sha256",
        "execution_identity",
        "seed",
        "input_seed",
        "mode",
        "device",
        "framework",
        "awards_runs",
        "constructor_started",
        "constructor_completed",
        "input_completed",
        "per_mode",
        "declared_meaningful_modes",
        "detected_meaningful_modes",
        "meaningful_modes",
        "train_eval_divergence",
        "divergence_evidence",
        "policy_observation",
        "error",
    }
)
_RAW_AWARD_RECEIPT_KEYS = frozenset(
    {
        "receipt_version",
        "request_nonce",
        "request_sha256",
        "stable_id",
        "work_id",
        "execution_identity",
        "recipe_revision",
        "code_manifest_identity",
        "input_identity",
        "requested_mode",
        "observation",
    }
)
_PARENT_ATTESTATION_KEYS = frozenset(
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
_PARENT_ATTESTATION_KEYS_V3 = _PARENT_ATTESTATION_KEYS | {
    "group_teardown_outcome",
    "group_survivor_pids",
}

#: DX-11/RE-12: the teardown outcomes that may attest a success. ``survivors``,
#: ``unverifiable``, ``foreign-group``, and an absent outcome all make an award
#: impossible -- a surviving grandchild both outlives supervision and inherits
#: execution flocks, so nothing it touched is attestable.
_AWARDABLE_TEARDOWN_OUTCOMES = frozenset({"clean", "reaped"})

# Structural authority registry consumed by the Round-17 AST inventory. Every
# ``subprocess.Popen`` edge in this module must have exactly one reviewed role.
_SUBPROCESS_SPAWN_REGISTRY = {
    "_emit_macos_audit_sentinel": "audit-sentinel:no-model-work",
    "_start_macos_denial_audit": "audit-collector:no-model-work",
    "run_isolated_subprocess": "model-worker:inherited-live-lease",
}


@dataclass(frozen=True)
class SupervisorObservation:
    """Facts observed exclusively by the parent process.

    Parameters
    ----------
    argv, cwd:
        Exact non-shell invocation.
    exit_code, signal_number:
        Process result observed by the parent.
    wall_seconds, cpu_seconds, peak_rss_bytes:
        Parent-measured resource facts.
    timed_out, rss_exceeded:
        Resource enforcement outcomes.
    stdout/stderr fields:
        Hashes, sizes, bounded tails, and local paths.
    failed_read_probe_paths:
        Undeclared read-only opens that failed before returning a descriptor.
    success_attestation_sha256, attested_receipt_sha256:
        Parent-owned success attestation and the exact child receipt it witnessed.
    """

    argv: tuple[str, ...]
    cwd: str
    exit_code: Optional[int]
    signal_number: Optional[int]
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    timed_out: bool
    rss_exceeded: bool
    stdout_sha256: str
    stdout_bytes: int
    stdout_tail: str
    stderr_sha256: str
    stderr_bytes: int
    stderr_tail: str
    stdout_path: str
    stderr_path: str
    failed_read_probe_paths: tuple[str, ...] = ()
    success_attestation_sha256: Optional[str] = None
    attested_receipt_sha256: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    shutdown_requested: bool = False
    # BL-12(c)/(d): every process-group teardown result is bound and recorded,
    # and the memory fact names the mechanism that measured it.
    group_teardown_outcome: Optional[str] = None
    group_survivor_pids: tuple[int, ...] = ()
    peak_rss_basis: str = "root-process"
    # DX-8/RE-8: the parent-enforced soft-watermark kill threshold and how far
    # past it the observed peak actually got before the kill took effect. The
    # overshoot quantifies sampling lag per attempt (the suspected 8/23
    # host-crash mechanism is a fast allocator outrunning the 10 ms sampler).
    rss_kill_threshold_bytes: int = 0
    rss_overshoot_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible parent observation.

        Returns
        -------
        dict[str, Any]
            Complete supervisor facts.
        """

        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "signal": self.signal_number,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "timed_out": self.timed_out,
            "rss_exceeded": self.rss_exceeded,
            "stdout_sha256": self.stdout_sha256,
            "stdout_bytes": self.stdout_bytes,
            "stdout_tail": self.stdout_tail,
            "stderr_sha256": self.stderr_sha256,
            "stderr_bytes": self.stderr_bytes,
            "stderr_tail": self.stderr_tail,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "failed_read_probe_paths": list(self.failed_read_probe_paths),
            "success_attestation_sha256": self.success_attestation_sha256,
            "attested_receipt_sha256": self.attested_receipt_sha256,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "shutdown_requested": self.shutdown_requested,
            "group_teardown_outcome": self.group_teardown_outcome,
            "group_survivor_pids": list(self.group_survivor_pids),
            "peak_rss_basis": self.peak_rss_basis,
            "rss_kill_threshold_bytes": self.rss_kill_threshold_bytes,
            "rss_overshoot_bytes": self.rss_overshoot_bytes,
        }


@dataclass(frozen=True)
class SupervisedResult:
    """Parent observation plus an optional verified atomic worker receipt.

    Parameters
    ----------
    observation:
        Facts measured by the supervisor.
    worker_receipt:
        Parsed receipt only when a complete valid atomic file exists.
    receipt_error:
        Parent diagnosis when the receipt is absent or invalid.
    success_attestation_sha256:
        Parent-owned proof of a normally completed worker success path.
    """

    observation: SupervisorObservation
    worker_receipt: Optional[dict[str, Any]]
    receipt_error: Optional[str]
    success_attestation_sha256: Optional[str] = None
    raw_award_receipt: Optional[dict[str, Any]] = None
    raw_award_receipt_sha256: Optional[str] = None
    parent_attestation: Optional[dict[str, Any]] = None
    unattested_partial: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class VerifiedWorkerResult:
    """One closed live v3 worker result projected for semantic consumers.

    Parameters
    ----------
    result_sha256:
        Recomputed digest of the closed outer v3 wrapper.
    diagnostic:
        Closed nested worker-receipt.v1 diagnostic. It never grants award authority.
    raw_award_receipt, raw_award_receipt_sha256, raw_observation:
        Authenticated success-only award receipt, its digest, and its observation.
    parent_attestation:
        Parent-owned process and completion attestation, when available.
    """

    result_sha256: str
    diagnostic: dict[str, Any]
    raw_award_receipt: Optional[dict[str, Any]]
    raw_award_receipt_sha256: Optional[str]
    raw_observation: Optional[dict[str, Any]]
    parent_attestation: Optional[dict[str, Any]]


def _load_worker_result_value(
    value: Mapping[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Validate and copy one closed worker-result.v3 outer envelope.

    Parameters
    ----------
    value:
        Parsed child result mapping.

    Returns
    -------
    tuple[dict[str, Any] | None, str | None]
        Closed copied wrapper or a parent-owned protocol error.
    """

    if set(value) != _WORKER_RESULT_KEYS:
        return None, "invalid-receipt:worker-result-envelope"
    if value.get("result_version") != _WORKER_RESULT_VERSION:
        return None, "invalid-receipt:worker-result-version"
    claimed_result_hash = value.get("result_sha256")
    result_payload = {key: item for key, item in value.items() if key != "result_sha256"}
    if not isinstance(claimed_result_hash, str) or claimed_result_hash != stable_hash(
        result_payload
    ):
        return None, "invalid-receipt:result-hash-mismatch"
    diagnostic = value.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        return None, "invalid-receipt:missing-diagnostic"
    if set(diagnostic) != _WORKER_DIAGNOSTIC_KEYS:
        return None, "invalid-receipt:diagnostic-envelope"
    if diagnostic.get("receipt_version") != _WORKER_DIAGNOSTIC_VERSION:
        return None, "invalid-receipt:diagnostic-version"
    raw_receipt = value.get("raw_award_receipt")
    raw_digest = value.get("raw_award_receipt_sha256")
    if (raw_receipt is None) != (raw_digest is None):
        return None, "invalid-receipt:partial-raw-award-receipt"
    if raw_receipt is not None:
        if not isinstance(raw_receipt, Mapping) or set(raw_receipt) != _RAW_AWARD_RECEIPT_KEYS:
            return None, "invalid-receipt:raw-award-envelope"
        if raw_receipt.get("receipt_version") != _RAW_AWARD_RECEIPT_VERSION:
            return None, "invalid-receipt:raw-award-version"
        if not isinstance(raw_digest, str) or raw_digest != stable_hash(raw_receipt):
            return None, "invalid-receipt:raw-award-hash-mismatch"
    return deepcopy(dict(value)), None


def _parent_attestation_error(
    result: SupervisedResult,
    parent_attestation: Mapping[str, Any],
    raw_receipt: Optional[Mapping[str, Any]],
    raw_digest: Optional[str],
) -> Optional[str]:
    """Validate one parent attestation against exact supervised process facts.

    Parameters
    ----------
    result:
        Parent-observed supervised result.
    parent_attestation:
        Claimed parent attestation mapping.
    raw_receipt, raw_digest:
        Optional authenticated raw award receipt and digest.

    Returns
    -------
    str | None
        Protocol error, or ``None`` when the attestation is internally consistent.
    """

    version = parent_attestation.get("attestation_version")
    if version == _PARENT_ATTESTATION_VERSION_V3:
        expected_keys = _PARENT_ATTESTATION_KEYS_V3
    elif version == _PARENT_ATTESTATION_VERSION:
        expected_keys = _PARENT_ATTESTATION_KEYS
    else:
        return "invalid-receipt:parent-attestation-version"
    if set(parent_attestation) != expected_keys:
        return "invalid-receipt:parent-attestation-envelope"
    payload = {
        key: value for key, value in parent_attestation.items() if key != "attestation_sha256"
    }
    attestation_sha256 = parent_attestation.get("attestation_sha256")
    if not isinstance(attestation_sha256, str) or attestation_sha256 != stable_hash(payload):
        return "invalid-receipt:parent-attestation-hash"
    observation = result.observation
    expected_observation = {
        "exit_code": observation.exit_code,
        "signal": observation.signal_number,
        "timed_out": observation.timed_out,
        "rss_exceeded": observation.rss_exceeded,
        "peak_rss_bytes": observation.peak_rss_bytes,
        "stdout_sha256": observation.stdout_sha256,
        "stderr_sha256": observation.stderr_sha256,
        "started_at": observation.started_at,
        "finished_at": observation.finished_at,
    }
    if version == _PARENT_ATTESTATION_VERSION_V3:
        expected_observation["group_teardown_outcome"] = observation.group_teardown_outcome
        expected_observation["group_survivor_pids"] = list(observation.group_survivor_pids)
    if any(parent_attestation.get(key) != value for key, value in expected_observation.items()):
        return "invalid-receipt:parent-attestation-observation"
    if raw_receipt is not None:
        completion_line_sha256 = hash_bytes(
            (completion_line_for_raw_award_receipt(raw_receipt) + "\n").encode("utf-8")
        )
        if (
            parent_attestation.get("request_nonce") != raw_receipt.get("request_nonce")
            or parent_attestation.get("request_sha256") != raw_receipt.get("request_sha256")
            or parent_attestation.get("completion_line_sha256") != completion_line_sha256
            or parent_attestation.get("named_raw_award_receipt_sha256") != raw_digest
            or result.success_attestation_sha256 != attestation_sha256
        ):
            return "invalid-receipt:parent-attestation-binding"
    return None


def verify_supervised_worker_result(
    result: SupervisedResult,
    *,
    expected_stable_id: str,
    expected_work_id: str,
    expected_source_identity: str,
    expected_recipe_revision: str,
    expected_execution_identity: str,
    expected_code_manifest_identity: Optional[str],
    requested_mode: Optional[str],
) -> tuple[Optional[VerifiedWorkerResult], Optional[str]]:
    """Project one live supervised result after v3 and association verification.

    Parameters
    ----------
    result:
        Live supervisor result. Flat v1 receipts are never accepted here.
    expected_stable_id, expected_work_id, expected_source_identity:
        Driver-owned proposal and work associations.
    expected_recipe_revision, expected_execution_identity:
        Driver-owned execution associations.
    expected_code_manifest_identity:
        Driver-owned recursive code-manifest identity.
    requested_mode:
        Explicit subprocess mode, or ``None`` for an all-modes diagnostic.

    Returns
    -------
    tuple[VerifiedWorkerResult | None, str | None]
        Verified typed projection or a closed protocol error.
    """

    if result.receipt_error not in {None, "missing-or-mismatched-v3-attestation"}:
        return None, result.receipt_error
    outer = result.worker_receipt
    if not isinstance(outer, Mapping):
        return None, "missing-receipt"
    loaded, load_error = _load_worker_result_value(outer)
    if loaded is None:
        return None, load_error
    diagnostic = loaded["diagnostic"]
    if not isinstance(diagnostic, dict):
        return None, "invalid-receipt:missing-diagnostic"
    if (
        diagnostic.get("stable_id") != expected_stable_id
        or diagnostic.get("source_identity") != expected_source_identity
        or diagnostic.get("recipe_revision") != expected_recipe_revision
        or diagnostic.get("execution_identity") != expected_execution_identity
    ):
        return None, "invalid-receipt:identity"

    raw_value = loaded.get("raw_award_receipt")
    raw_digest_value = loaded.get("raw_award_receipt_sha256")
    raw_receipt = dict(raw_value) if isinstance(raw_value, Mapping) else None
    raw_digest = str(raw_digest_value) if isinstance(raw_digest_value, str) else None
    raw_observation: Optional[dict[str, Any]] = None
    if raw_receipt is not None:
        if (
            raw_receipt.get("stable_id") != expected_stable_id
            or raw_receipt.get("work_id") != expected_work_id
            or raw_receipt.get("recipe_revision") != expected_recipe_revision
            or raw_receipt.get("execution_identity") != expected_execution_identity
            or raw_receipt.get("code_manifest_identity") != expected_code_manifest_identity
            or raw_receipt.get("requested_mode") != requested_mode
        ):
            return None, "invalid-receipt:raw-award-identity"
        observation_value = raw_receipt.get("observation")
        if not isinstance(observation_value, Mapping):
            return None, "invalid-receipt:raw-award-observation"
        raw_observation = deepcopy(dict(observation_value))
        per_mode = diagnostic.get("per_mode")
        diagnostic_mode = (
            per_mode.get(requested_mode)
            if isinstance(per_mode, Mapping) and requested_mode is not None
            else None
        )
        if not isinstance(diagnostic_mode, Mapping):
            return None, "invalid-receipt:raw-award-mode"
        shared_fields = set(raw_observation) & set(diagnostic_mode)
        if any(raw_observation.get(key) != diagnostic_mode.get(key) for key in shared_fields):
            return None, "invalid-receipt:raw-diagnostic-mismatch"
        if (
            raw_observation.get("observed_recipe_revision")
            != diagnostic.get("observed_recipe_revision")
            or raw_observation.get("observed_adapter_sha256")
            != diagnostic.get("observed_adapter_sha256")
            or raw_observation.get("observed_code_manifest_sha256")
            != diagnostic.get("observed_code_manifest_sha256")
            or raw_observation.get("observed_input_asset_sha256")
            != diagnostic.get("observed_input_asset_sha256")
        ):
            return None, "invalid-receipt:raw-diagnostic-identity"
        if result.raw_award_receipt != raw_receipt or result.raw_award_receipt_sha256 != raw_digest:
            return None, "invalid-receipt:supervisor-raw-award-mismatch"
    elif result.raw_award_receipt is not None or result.raw_award_receipt_sha256 is not None:
        return None, "invalid-receipt:unexpected-supervisor-raw-award"

    parent = result.parent_attestation
    parent_copy = deepcopy(parent) if isinstance(parent, Mapping) else None
    if parent_copy is not None:
        parent_error = _parent_attestation_error(result, parent_copy, raw_receipt, raw_digest)
        if parent_error is not None:
            return None, parent_error
    if raw_receipt is not None and parent_copy is None:
        return None, "missing-parent-success-attestation"
    return (
        VerifiedWorkerResult(
            result_sha256=str(loaded["result_sha256"]),
            diagnostic=deepcopy(diagnostic),
            raw_award_receipt=raw_receipt,
            raw_award_receipt_sha256=raw_digest,
            raw_observation=raw_observation,
            parent_attestation=parent_copy,
        ),
        None,
    )


def worker_result_outer_for_diagnostics(result: SupervisedResult) -> Optional[dict[str, Any]]:
    """Copy the opaque outer worker result for shutdown-only diagnostics.

    Parameters
    ----------
    result:
        Interrupted supervised result.

    Returns
    -------
    dict[str, Any] | None
        Opaque outer bytes as parsed by the supervisor, without semantic projection.
    """

    value = result.worker_receipt
    return deepcopy(value) if isinstance(value, dict) else None


@dataclass(frozen=True)
class SandboxDenialObservation:
    """OS-boundary policy denials observed by the supervisor.

    Parameters
    ----------
    network_attempted:
        Whether a network syscall reached the offline boundary.
    socket_targets:
        Sanitized syscall targets observed by the broker.
    write_outside_scratch_attempted:
        Whether a write syscall outside the allowed roots was denied.
    write_paths:
        Sanitized denied write paths.
    checkpoint_or_weight_read_attempted, checkpoint_paths:
        Undeclared model-data reads observed at the kernel boundary.
    failed_read_probe_paths:
        Undeclared read-only opens that returned a negative file descriptor. These
        retain diagnostic value but do not poison a receipt because no bytes were read.
    telemetry_failure:
        Fail-closed broker-integrity diagnostic, if telemetry was not trustworthy.
    """

    network_attempted: bool = False
    socket_targets: tuple[str, ...] = ()
    write_outside_scratch_attempted: bool = False
    write_paths: tuple[str, ...] = ()
    checkpoint_or_weight_read_attempted: bool = False
    checkpoint_paths: tuple[str, ...] = ()
    failed_read_probe_paths: tuple[str, ...] = ()
    telemetry_failure: Optional[str] = None

    @property
    def poisoned(self) -> bool:
        """Return whether the denial invalidates an otherwise successful receipt.

        Returns
        -------
        bool
            True for any observed denied network or outside-write operation.
        """

        return (
            self.network_attempted
            or self.write_outside_scratch_attempted
            or self.checkpoint_or_weight_read_attempted
            or self.telemetry_failure is not None
        )


@dataclass
class WorkerLeaseHandle:
    """Parent-side descriptors for one frozen ``WorkerLease`` contract.

    Parameters
    ----------
    lease:
        Exact durable lease metadata.
    lock_path, record_path:
        Campaign-local kernel lock and fsynced local metadata paths.
    global_lock_path, global_lock_fd:
        Cross-campaign execution lock and its inherited descriptor. ``None`` is
        reserved for direct unit-level lease callers; production workers always
        carry both.
    lock_fd:
        Open locked descriptor transferred to the child at spawn.
    lifecycle_read_fd, lifecycle_write_fd:
        Parent-death pipe. The child watches the read end; the parent owns the
        write end until supervision ends.
    """

    lease: WorkerLease
    lock_path: Path
    record_path: Path
    global_lock_path: Optional[Path]
    global_lock_fd: Optional[int]
    lock_fd: Optional[int]
    lifecycle_read_fd: Optional[int]
    lifecycle_write_fd: Optional[int]


@dataclass(frozen=True)
class WorkerLeaseRecovery:
    """Bounded startup reconciliation result for one durable worker lease."""

    state: str
    lease: Optional[WorkerLease]
    detail: str
    lock_held: bool
    reaped: bool


def _worker_lease_mapping(lease: WorkerLease) -> dict[str, Any]:
    """Serialize the frozen worker lease without inventing another contract.

    Parameters
    ----------
    lease:
        Frozen lease metadata.

    Returns
    -------
    dict[str, Any]
        JSON-compatible exact field mapping.
    """

    return {
        "lease_id": lease.lease_id,
        "nonce": lease.nonce,
        "run_id": lease.run_id,
        "stable_id": lease.stable_id,
        "work_id": lease.work_id,
        "request_identity": lease.request_identity,
        "execution_identity": lease.execution_identity,
        "boot_id": lease.boot_id,
        "driver_pid": lease.driver_pid,
        "driver_start_token": lease.driver_start_token,
        "child_pid": lease.child_pid,
        "child_start_token": lease.child_start_token,
        "child_pgid": lease.child_pgid,
        "receipt_path": str(lease.receipt_path),
        "opened_at": lease.opened_at,
        "deadline_at": lease.deadline_at,
    }


def _worker_lease_from_mapping(value: Mapping[str, Any]) -> WorkerLease:
    """Parse exact durable metadata into the frozen worker lease type.

    Parameters
    ----------
    value:
        Decoded lease record.

    Returns
    -------
    WorkerLease
        Validated frozen contract.

    Raises
    ------
    ValueError
        If fields are missing, extra, or have invalid primitive types.
    """

    expected = {
        "lease_id",
        "nonce",
        "run_id",
        "stable_id",
        "work_id",
        "request_identity",
        "execution_identity",
        "boot_id",
        "driver_pid",
        "driver_start_token",
        "child_pid",
        "child_start_token",
        "child_pgid",
        "receipt_path",
        "opened_at",
        "deadline_at",
    }
    if set(value) != expected:
        raise ValueError("worker lease record fields do not match the frozen contract")
    string_fields = (
        "lease_id",
        "nonce",
        "run_id",
        "stable_id",
        "work_id",
        "request_identity",
        "execution_identity",
        "boot_id",
        "driver_start_token",
        "receipt_path",
        "opened_at",
        "deadline_at",
    )
    if any(not isinstance(value[field], str) or not value[field] for field in string_fields):
        raise ValueError("worker lease string fields must be non-empty")
    try:
        opened_at = datetime.fromisoformat(str(value["opened_at"]).replace("Z", "+00:00"))
        deadline_at = datetime.fromisoformat(str(value["deadline_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("worker lease timestamps must be RFC 3339") from exc
    if opened_at.tzinfo is None or deadline_at.tzinfo is None or deadline_at <= opened_at:
        raise ValueError("worker lease deadline must be after its UTC open time")
    if not isinstance(value["driver_pid"], int) or value["driver_pid"] <= 0:
        raise ValueError("worker lease driver_pid must be positive")
    for field_name in ("child_pid", "child_pgid"):
        item = value[field_name]
        if item is not None and (not isinstance(item, int) or item <= 0):
            raise ValueError(f"worker lease {field_name} must be null or positive")
    child_token = value["child_start_token"]
    if child_token is not None and (not isinstance(child_token, str) or not child_token):
        raise ValueError("worker lease child_start_token must be null or non-empty")
    return WorkerLease(
        lease_id=str(value["lease_id"]),
        nonce=str(value["nonce"]),
        run_id=str(value["run_id"]),
        stable_id=str(value["stable_id"]),
        work_id=str(value["work_id"]),
        request_identity=str(value["request_identity"]),
        execution_identity=str(value["execution_identity"]),
        boot_id=str(value["boot_id"]),
        driver_pid=int(value["driver_pid"]),
        driver_start_token=str(value["driver_start_token"]),
        child_pid=value["child_pid"],
        child_start_token=child_token,
        child_pgid=value["child_pgid"],
        receipt_path=Path(str(value["receipt_path"])),
        opened_at=str(value["opened_at"]),
        deadline_at=str(value["deadline_at"]),
    )


def _atomic_write_worker_lease(path: Path, lease: WorkerLease) -> None:
    """Atomically fsync one exact worker lease record.

    Parameters
    ----------
    path:
        Gitignored local lease record.
    lease:
        Frozen metadata to persist.
    """

    atomic_replace_bytes(path, canonical_json_bytes(_worker_lease_mapping(lease)) + b"\n")


def current_boot_id() -> str:
    """Return a stable identifier for the current host boot generation.

    This is the crawler's ONE boot-identity chokepoint. Both sources are true
    per-boot generation markers minted by the kernel at boot: the Linux boot
    UUID and the Darwin boot-session UUID. Neither moves under an NTP clock
    step, so paid custody bound to a boot can never be released as stale-boot
    by a wall-clock adjustment (clock-derived boot-time records are forbidden
    here and everywhere else in the crawler).

    Returns
    -------
    str
        Linux kernel boot UUID or the Darwin ``kern.bootsessionuuid``.

    Raises
    ------
    RuntimeError
        If the platform exposes no verifiable boot identity.
    """

    linux_boot = Path("/proc/sys/kernel/random/boot_id")
    try:
        value = linux_boot.read_text(encoding="ascii").strip()
    except OSError:
        value = ""
    if value:
        return value
    try:
        completed = subprocess.run(
            ("sysctl", "-n", "kern.bootsessionuuid"),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("cannot establish the current boot identity") from None
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError("cannot establish the current boot identity")
    return value


_PROC_PIDTBSDINFO = 3


class _ProcBsdInfo(ctypes.Structure):
    """``struct proc_bsdinfo`` from ``sys/proc_info.h`` (PROC_PIDTBSDINFO flavor).

    Declared in full rather than as a byte window so the start-time offset comes
    from the compiler's own field layout. ``proc_pidinfo`` reports how many bytes
    it wrote, and :func:`_darwin_process_start_token` refuses anything other than
    the exact declared size, so a future struct change degrades to ``None`` (and
    then to the ``ps`` fallback) instead of reading a wrong offset.
    """

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def _darwin_process_start_token(pid: int) -> Optional[str]:
    """Return one macOS process-start token derived without spawning a child.

    The kernel publishes a process's start time through ``proc_pidinfo``, the
    same libproc entry point :func:`_macos_rss` already uses for the worker's
    live memory. Reading it directly is what lets process identity work under
    Seatbelt process-info confinement, where exec'ing ``ps`` raises EPERM and
    took the entire supervised forward lane with it.

    Microsecond resolution also makes this token STRICTLY finer-grained than the
    one-second ``ps -o lstart=`` string it precedes.

    Parameters
    ----------
    pid:
        Process identifier to inspect.

    Returns
    -------
    str | None
        Start-time token, or ``None`` when this host cannot answer for the
        process -- including a process owned by another user, which libproc
        refuses and the ``ps`` fallback still answers.
    """

    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        proc_pidinfo.restype = ctypes.c_int
        info = _ProcBsdInfo()
        written = proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info))
    except (AttributeError, OSError, ValueError):
        return None
    # A short write, a mismatched identity, or a zero start time is not a
    # process this token can speak for.
    if written != ctypes.sizeof(info) or info.pbi_pid != pid:
        return None
    if info.pbi_start_tvsec == 0:
        return None
    return f"darwin-start-time:{info.pbi_start_tvsec}.{info.pbi_start_tvusec:06d}"


def _start_tokens_comparably_differ(bound: str, observed: str) -> bool:
    """Return whether two start tokens prove the PID now names a foreign process.

    Tokens carry their probe scheme as a prefix (``darwin-start-time:``,
    ``linux-start-ticks:``, ``ps-lstart:``). Only SAME-scheme values are
    comparable: an exited-but-unreaped child on macOS fails the libproc probe
    and answers through the coarser ``ps`` fallback, so a cross-scheme pair is
    an artifact of the probe path, never evidence of PID reuse -- and while the
    child is unreaped the kernel still holds its PID for this exact process.
    Misreading that pair as foreign made every drained-group teardown a
    recorded cleanup hazard (full-corpus catch, 2026-08-12).

    Parameters
    ----------
    bound:
        Token bound at spawn.
    observed:
        Token observed at teardown.

    Returns
    -------
    bool
        True only for same-scheme tokens with different values.
    """

    bound_scheme, _, _ = bound.partition(":")
    observed_scheme, _, _ = observed.partition(":")
    if bound_scheme != observed_scheme:
        return False
    return bound != observed


def process_start_token(pid: int) -> Optional[str]:
    """Return an OS process-start token resistant to PID reuse.

    Parameters
    ----------
    pid:
        Process identifier to inspect.

    Returns
    -------
    str | None
        Kernel start ticks on Linux, the libproc start time on macOS, a
        normalized ``ps`` start time when neither is available, or ``None`` when
        the process cannot be verified.
    """

    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        stat_text = ""
    if stat_text:
        closing = stat_text.rfind(")")
        fields = stat_text[closing + 2 :].split() if closing >= 0 else []
        if len(fields) > 19:
            return f"linux-start-ticks:{fields[19]}"
    darwin_token = _darwin_process_start_token(pid)
    if darwin_token is not None:
        return darwin_token
    try:
        completed = subprocess.run(
            ("ps", "-o", "lstart=", "-p", str(pid)),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = " ".join(completed.stdout.split())
    return None if not value else f"ps-lstart:{value}"


def open_worker_lease(
    lock_path: Path,
    record_path: Path,
    lease: WorkerLease,
    *,
    global_lock_path: Optional[Path] = None,
    on_lock_acquired: Optional[Callable[[WorkerLease], None]] = None,
    shutdown_event: Optional[threading.Event] = None,
    global_lock_timeout_seconds: float = DEFAULT_EXECUTION_LOCK_TIMEOUT_SECONDS,
) -> WorkerLeaseHandle:
    """Acquire the single-execution kernel lock and persist its exact lease.

    Parameters
    ----------
    lock_path, record_path:
        Worker kernel-lock and durable local metadata paths.
    lease:
        Frozen pre-spawn lease with child identity fields unset.
    global_lock_path:
        Host-global execution flock acquired before the campaign-local lease.
        Production callers must provide the shared path.
    on_lock_acquired:
        Single-writer callback invoked while the kernel lock is held but before
        the lease record is fsynced. The driver uses this exact boundary to append
        ``worker-lease-opened`` in the frozen lock order.
    shutdown_event:
        Driver-owned shutdown signal consulted while waiting for the host-global
        execution flock, so an operator SIGTERM interrupts the wait.
    global_lock_timeout_seconds:
        Bounded host-global acquire deadline (sprint invariant I2).

    Returns
    -------
    WorkerLeaseHandle
        Parent descriptors ready for explicit child inheritance.

    Raises
    ------
    RuntimeError
        If another execution process owns the worker lock.
    ValueError
        If the lease is inconsistent with this boot or pre-spawn state.
    """

    if (
        lease.child_pid is not None
        or lease.child_start_token is not None
        or lease.child_pgid is not None
    ):
        raise ValueError("new worker lease must not contain child identity")
    if lease.boot_id != current_boot_id():
        raise ValueError("new worker lease boot identity is stale")
    global_descriptor = (
        acquire_global_execution_flock(
            global_lock_path,
            timeout_seconds=global_lock_timeout_seconds,
            shutdown_event=shutdown_event,
            holder_metadata={
                "lease_id": lease.lease_id,
                "run_id": lease.run_id,
                "stable_id": lease.stable_id,
                "work_id": lease.work_id,
                "driver_pid": lease.driver_pid,
                "driver_start_token": lease.driver_start_token,
                "boot_id": lease.boot_id,
                "opened_at": lease.opened_at,
            },
        )
        if global_lock_path is not None
        else None
    )
    descriptor: Optional[int] = None
    read_fd: Optional[int] = None
    write_fd: Optional[int] = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("worker lock is already held") from exc
        if on_lock_acquired is not None:
            on_lock_acquired(lease)
        read_fd, write_fd = os.pipe()
        _atomic_write_worker_lease(record_path, lease)
    except BaseException:
        if read_fd is not None and write_fd is not None:
            os.close(read_fd)
            os.close(write_fd)
        if descriptor is not None:
            os.close(descriptor)
        release_global_execution_flock(global_descriptor)
        raise
    assert descriptor is not None
    assert read_fd is not None
    assert write_fd is not None
    return WorkerLeaseHandle(
        lease=lease,
        lock_path=lock_path,
        record_path=record_path,
        global_lock_path=global_lock_path,
        global_lock_fd=global_descriptor,
        lock_fd=descriptor,
        lifecycle_read_fd=read_fd,
        lifecycle_write_fd=write_fd,
    )


def clear_worker_lease(handle: WorkerLeaseHandle) -> None:
    """Release parent descriptors and durably clear a closed lease record.

    Parameters
    ----------
    handle:
        Lease handle whose child has exited and whose operational closure event has
        already been appended by the single writer.
    """

    for attribute in (
        "lifecycle_read_fd",
        "lifecycle_write_fd",
        "lock_fd",
        "global_lock_fd",
    ):
        descriptor = getattr(handle, attribute)
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
            setattr(handle, attribute, None)
    handle.record_path.unlink(missing_ok=True)
    descriptor = os.open(handle.record_path.parent, os.O_RDONLY)
    try:
        durable_fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_worker_lease(path: Path) -> WorkerLease:
    """Read one exact local lease record.

    Parameters
    ----------
    path:
        Durable metadata path.

    Returns
    -------
    WorkerLease
        Parsed frozen lease.
    """

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("worker lease record must be an object")
    return _worker_lease_from_mapping(value)


def _deadline_timestamp(value: str) -> float:
    """Parse one lease deadline as a UTC epoch timestamp.

    Parameters
    ----------
    value:
        RFC 3339 timestamp.

    Returns
    -------
    float
        POSIX timestamp.
    """

    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def reconcile_worker_lease(
    lock_path: Path,
    record_path: Path,
    *,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.05,
) -> WorkerLeaseRecovery:
    """Reconcile a prior child-held lease before any new work admission.

    Parameters
    ----------
    lock_path, record_path:
        Worker kernel lock and durable metadata paths.
    timeout_seconds:
        Bounded recovery wall time. This never extends the recorded lease deadline.
    poll_seconds:
        Poll cadence while waiting for a verified child or lock release.

    Returns
    -------
    WorkerLeaseRecovery
        Closed reconciliation state for operational-event and attempt assembly.
    """

    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("worker lease recovery bounds must be positive")
    lease: Optional[WorkerLease]
    try:
        lease = _read_worker_lease(record_path) if record_path.exists() else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        lease = None
        parse_error = f"corrupt-worker-lease:{type(exc).__name__}"
    else:
        parse_error = ""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_free = False
        else:
            lock_free = True
        if lock_free:
            if lease is None:
                state = "none" if not parse_error else "failed-closed"
                return WorkerLeaseRecovery(
                    state, None, parse_error or "no-open-lease", False, False
                )
            if lease.boot_id != current_boot_id():
                state = "stale-boot"
            elif lease.child_pid is None:
                state = "never-started"
            elif lease.receipt_path.exists():
                state = "completed-before-recovery"
            else:
                state = "released-without-receipt"
            return WorkerLeaseRecovery(state, lease, state, False, False)
        if lease is None:
            return WorkerLeaseRecovery(
                "failed-closed",
                None,
                parse_error or "held-lock-without-verifiable-lease",
                True,
                False,
            )
        if lease.boot_id != current_boot_id():
            return WorkerLeaseRecovery(
                "failed-closed", lease, "held-lock-from-stale-boot", True, False
            )
        if lease.child_pid is None or lease.child_start_token is None or lease.child_pgid is None:
            return WorkerLeaseRecovery(
                "failed-closed", lease, "held-lock-without-child-identity", True, False
            )
        if process_start_token(lease.child_pid) != lease.child_start_token:
            return WorkerLeaseRecovery(
                "failed-closed", lease, "child-start-token-mismatch", True, False
            )
        try:
            observed_pgid = os.getpgid(lease.child_pid)
        except ProcessLookupError:
            observed_pgid = None
        if observed_pgid != lease.child_pgid or lease.child_pgid != lease.child_pid:
            return WorkerLeaseRecovery(
                "failed-closed", lease, "child-process-group-mismatch", True, False
            )
        recovery_deadline = min(
            time.monotonic() + timeout_seconds,
            time.monotonic() + max(0.0, _deadline_timestamp(lease.deadline_at) - time.time()),
        )
        while time.monotonic() < recovery_deadline:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(min(poll_seconds, max(0.0, recovery_deadline - time.monotonic())))
                continue
            return WorkerLeaseRecovery(
                "completed-before-recovery", lease, "verified-child-exited", False, False
            )
        if time.time() < _deadline_timestamp(lease.deadline_at):
            return WorkerLeaseRecovery(
                "active", lease, "verified-child-still-within-lease-deadline", True, False
            )
        if process_start_token(lease.child_pid) != lease.child_start_token:
            return WorkerLeaseRecovery(
                "failed-closed", lease, "child-identity-changed-before-reap", True, False
            )
        try:
            observed_pgid = os.getpgid(lease.child_pid)
        except ProcessLookupError:
            observed_pgid = None
        if observed_pgid != lease.child_pgid:
            return WorkerLeaseRecovery(
                "failed-closed", lease, "child-group-changed-before-reap", True, False
            )
        # Identity was just re-verified above, so this signal targets a group this
        # recovery provably owns. A denied signal against a drained group is not a
        # recovery failure; a genuinely surviving group still fails closed below,
        # because the kernel lock will not release.
        signal_verified_process_group(lease.child_pid, lease.child_pgid, lease.child_start_token)
        release_deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < release_deadline:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(poll_seconds)
                continue
            return WorkerLeaseRecovery("reaped", lease, "verified-child-reaped", False, True)
        return WorkerLeaseRecovery(
            "failed-closed", lease, "verified-child-lock-did-not-release", True, False
        )
    finally:
        os.close(descriptor)


# Every signal number an installed shutdown handler has actually observed, in
# delivery order. A shutdown handler is the ONLY route by which an ambient
# SIGTERM/SIGINT can be converted into ordinary program state -- the event is
# set, the run reports ``interrupted:shutdown``, and nothing downstream can tell
# that outcome apart from one the code chose. Recording the raw signal number is
# what makes the difference recoverable after the fact.
_OBSERVED_SHUTDOWN_SIGNALS: list[int] = []
# The ledger exists to answer "was a signal delivered, and which one", so a
# signal storm has nothing further to say. Bounding it keeps a pathological
# sender from growing this list without limit inside a signal handler.
_OBSERVED_SHUTDOWN_SIGNAL_LIMIT = 64


def observed_shutdown_signals() -> tuple[int, ...]:
    """Return the signals delivered to installed shutdown handlers in this process.

    Only deliveries that happened while :func:`shutdown_signal_handlers` had its
    handlers installed are visible here; outside that window the interpreter's own
    disposition applies and no shutdown event is set either.

    Returns
    -------
    tuple[int, ...]
        Observed signal numbers in delivery order, bounded by
        :data:`_OBSERVED_SHUTDOWN_SIGNAL_LIMIT`.
    """

    return tuple(_OBSERVED_SHUTDOWN_SIGNALS)


def clear_observed_shutdown_signals() -> None:
    """Empty the observed-signal ledger so a later window starts from nothing."""

    _OBSERVED_SHUTDOWN_SIGNALS.clear()


@contextmanager
def shutdown_signal_handlers(shutdown_event: threading.Event) -> Iterator[None]:
    """Install SIGTERM/SIGINT handlers that only set a shutdown event.

    Parameters
    ----------
    shutdown_event:
        Driver-owned event threaded into worker supervision.

    Yields
    ------
    None
        Control while the handlers are installed. Prior handlers are restored.
    """

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("shutdown signal handlers require the main thread")

    def request_shutdown(signum: int, frame: Any) -> None:
        """Record the delivered signal, then set the shutdown event.

        Parameters
        ----------
        signum, frame:
            Standard Python signal-handler arguments.
        """

        del frame
        # A bounded list append is the only signal-unsafe-free way to keep the
        # provenance of the shutdown. Without it the delivered signal leaves no
        # trace anywhere and an ambient signal is indistinguishable from a
        # deliberate one by the time anybody looks.
        if len(_OBSERVED_SHUTDOWN_SIGNALS) < _OBSERVED_SHUTDOWN_SIGNAL_LIMIT:
            _OBSERVED_SHUTDOWN_SIGNALS.append(signum)
        shutdown_event.set()

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        for signum in previous:
            signal.signal(signum, request_shutdown)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _child_limit(
    rss_limit_bytes: int,
    lease_record_path: Optional[Path] = None,
    lease: Optional[WorkerLease] = None,
    create_darwin_libomp_blocker: bool = False,
) -> None:
    """Apply limits, parent-death defense, and trusted lease bootstrap.

    Parameters
    ----------
    rss_limit_bytes:
        Requested memory cap in bytes.
    lease_record_path, lease:
        Optional exact durable lease filled with child identity before model import.
    create_darwin_libomp_blocker:
        Whether to create the exact PID-bound LLVM OpenMP registration blocker before
        entering Seatbelt.
    """

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0:  # PR_SET_PDEATHSIG
            os._exit(126)
        if os.getppid() == 1:
            os._exit(126)
    if create_darwin_libomp_blocker:
        child_pid = os.getpid()
        blocker = _darwin_libomp_registration_blocker(child_pid)
        blocker_error = _claim_darwin_libomp_registration_blocker(
            blocker,
            reclaim_stale_regular_file_for_pid=child_pid,
        )
        if blocker_error is not None:
            _write_child_bootstrap_error(blocker_error)
            os._exit(126)
    if lease_record_path is not None and lease is not None:
        child_pid = os.getpid()
        child_pgid = os.getpgrp()
        if child_pgid != child_pid:
            os._exit(126)
        child_token = process_start_token(child_pid)
        if child_token is None:
            os._exit(126)
        active_lease = WorkerLease(
            lease_id=lease.lease_id,
            nonce=lease.nonce,
            run_id=lease.run_id,
            stable_id=lease.stable_id,
            work_id=lease.work_id,
            request_identity=lease.request_identity,
            execution_identity=lease.execution_identity,
            boot_id=lease.boot_id,
            driver_pid=lease.driver_pid,
            driver_start_token=lease.driver_start_token,
            child_pid=child_pid,
            child_start_token=child_token,
            child_pgid=child_pgid,
            receipt_path=lease.receipt_path,
            opened_at=lease.opened_at,
            deadline_at=lease.deadline_at,
        )
        try:
            _atomic_write_worker_lease(lease_record_path, active_lease)
        except Exception:
            os._exit(126)
    if rss_limit_bytes > 0:
        # Darwin aliases RLIMIT_AS, RLIMIT_DATA, and RLIMIT_RSS onto one limit its kernel
        # never implemented, and refuses every attempt to lower it with EINVAL, so the
        # child-side cap simply does not exist there. Memory containment on macOS is carried
        # by the parent RSS sampler in run_isolated_subprocess, which polls _macos_rss and
        # kills the process group on breach. Refusing the spawn instead would not add any
        # containment. Every other platform keeps the original fail-closed behavior.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (rss_limit_bytes, rss_limit_bytes))
            if sys.platform == "darwin" and hasattr(resource, "RLIMIT_DATA"):
                resource.setrlimit(resource.RLIMIT_DATA, (rss_limit_bytes, rss_limit_bytes))
        except (OSError, ValueError):
            if sys.platform != "darwin":
                raise
        try:
            # Per-file write ceiling (BL-12(h), opus W-13): a verbose model
            # could previously write tens of GB of stdout inside its timeout
            # while staying under its own RSS cap -- disk is the crawl's
            # oxygen. A breach delivers SIGXFSZ, an honest, attributable
            # resource failure, instead of exhausting the volume.
            resource.setrlimit(resource.RLIMIT_FSIZE, (rss_limit_bytes, rss_limit_bytes))
        except (OSError, ValueError):
            if sys.platform != "darwin":
                raise


def _darwin_libomp_registration_blocker(pid: int) -> Path:
    """Return the exact Darwin LLVM OpenMP temporary registration path.

    Parameters
    ----------
    pid:
        Sandboxed worker process identifier retained across ``sandbox-exec``.

    Returns
    -------
    pathlib.Path
        PID-specific path LLVM OpenMP otherwise creates in the host-global temporary root.
    """

    return Path("/private/tmp") / f"{_DARWIN_LIBOMP_REGISTRATION_PREFIX}{pid}"


def _claim_darwin_libomp_registration_blocker(
    path: Path,
    *,
    reclaim_stale_regular_file_for_pid: Optional[int] = None,
) -> Optional[str]:
    """Atomically claim one Darwin LLVM OpenMP registration blocker.

    Parameters
    ----------
    path:
        Exact PID-bound path to occupy before entering Seatbelt.
    reclaim_stale_regular_file_for_pid:
        The just-forked child's own PID when a pre-existing regular file at its
        exact registration path may be reclaimed. ``None`` preserves the
        ordinary fail-closed claim behavior.

    Returns
    -------
    str | None
        ``None`` only for the process whose atomic ``mkdir`` created the blocker
        or whose just-forked own-PID stale regular file was atomically reclaimed.
        Otherwise, a path-specific fail-closed diagnostic distinguishing a foreign
        registration file from a supervisor blocker directory.
    """

    try:
        path.mkdir(mode=0o500)
    except FileExistsError:
        try:
            status = path.lstat()
        except FileNotFoundError:
            return (
                "Darwin libomp registration blocker claim raced with removal at "
                f"{path}; refusing to start without continuous protection"
            )
        except OSError as exc:
            return (
                f"Darwin libomp registration blocker at {path} could not be inspected "
                f"after EEXIST: {exc}"
            )
        if stat.S_ISREG(status.st_mode):
            if reclaim_stale_regular_file_for_pid is not None:
                reclaim_error = _reclaim_stale_darwin_libomp_registration_file(
                    path,
                    own_pid=reclaim_stale_regular_file_for_pid,
                )
                if reclaim_error is None:
                    _write_child_bootstrap_notice(
                        "reclaimed provably stale LLVM OpenMP registration file "
                        f"at {path} for this just-forked worker PID"
                    )
                    return None
                return reclaim_error
            return (
                "pre-existing foreign LLVM OpenMP registration file blocks worker startup "
                f"at {path}; this PID-keyed entry may be stale after abnormal termination"
            )
        if stat.S_ISDIR(status.st_mode):
            return (
                "existing supervisor LLVM OpenMP blocker directory refused a duplicate claim "
                f"at {path}; another claimant owns it or its ownership is unverifiable"
            )
        return (
            "invalid pre-existing LLVM OpenMP registration entry blocks worker startup "
            f"at {path}; mode={stat.S_IFMT(status.st_mode):#o}"
        )
    except OSError as exc:
        return f"Darwin libomp registration blocker could not be created at {path}: {exc}"
    return None


def _reclaim_stale_darwin_libomp_registration_file(path: Path, *, own_pid: int) -> Optional[str]:
    """Atomically replace one provably stale own-PID libomp file with a blocker.

    Parameters
    ----------
    path:
        Exact registration path whose creation claim observed an existing regular file.
    own_pid:
        PID of the child immediately after ``fork`` and before any exec or libomp load.

    Returns
    -------
    str | None
        ``None`` after the stale file was removed and an owned ``0500`` directory
        continuously occupied ``path``. Otherwise a fail-closed diagnostic.
    """

    expected_path = _darwin_libomp_registration_blocker(own_pid)
    if path != expected_path:
        return (
            "refusing to reclaim LLVM OpenMP registration entry outside the just-forked "
            f"worker's own PID path: {path}"
        )
    try:
        stale_status = path.lstat()
    except OSError as exc:
        return (
            "Darwin libomp registration blocker could not be re-inspected for stale-file "
            f"reclaim at {path}: {exc}"
        )
    if not stat.S_ISREG(stale_status.st_mode):
        return (
            "LLVM OpenMP registration entry changed before stale-file reclaim at "
            f"{path}; refusing to replace a non-file entry"
        )

    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".menagerie-libomp-reclaim-{own_pid}-", dir=path.parent)
        )
        staging.chmod(0o500)
        staging_status = staging.lstat()
    except OSError as exc:
        return f"could not prepare atomic stale libomp blocker reclaim at {path}: {exc}"
    if (
        not stat.S_ISDIR(staging_status.st_mode)
        or stat.S_IMODE(staging_status.st_mode) != 0o500
        or staging_status.st_uid != os.getuid()
    ):
        try:
            staging.rmdir()
        except OSError:
            pass
        return f"prepared stale libomp blocker has invalid ownership or mode at {staging}"

    swap_error = _darwin_atomic_swap(staging, path)
    if swap_error is not None:
        try:
            staging.rmdir()
        except OSError:
            pass
        return (
            f"could not atomically reclaim stale libomp registration file at {path}: {swap_error}"
        )

    try:
        displaced_status = staging.lstat()
        claimed_status = path.lstat()
    except OSError as exc:
        return (
            "atomic stale libomp blocker reclaim could not be verified at "
            f"{path}: {exc}; refusing worker bootstrap"
        )
    stale_identity = (stale_status.st_dev, stale_status.st_ino)
    staging_identity = (staging_status.st_dev, staging_status.st_ino)
    if (
        not stat.S_ISREG(displaced_status.st_mode)
        or (displaced_status.st_dev, displaced_status.st_ino) != stale_identity
        or not stat.S_ISDIR(claimed_status.st_mode)
        or stat.S_IMODE(claimed_status.st_mode) != 0o500
        or claimed_status.st_uid != os.getuid()
        or (claimed_status.st_dev, claimed_status.st_ino) != staging_identity
    ):
        restore_error = _darwin_atomic_swap(staging, path)
        if restore_error is None:
            try:
                staging.rmdir()
            except OSError:
                pass
        return (
            "atomic stale libomp blocker reclaim changed an unverified entry at "
            f"{path}; restored the prior entry={restore_error is None} and refusing worker bootstrap"
        )
    try:
        staging.unlink()
    except OSError as exc:
        return f"reclaimed stale libomp registration file could not be removed at {staging}: {exc}"
    return None


def _darwin_atomic_swap(source: Path, destination: Path) -> Optional[str]:
    """Atomically exchange two Darwin filesystem entries without a missing-path window.

    Parameters
    ----------
    source, destination:
        Existing entries on the same Darwin volume to exchange.

    Returns
    -------
    str | None
        ``None`` on a successful ``RENAME_SWAP``; otherwise the operating-system error.
    """

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameatx_np = libc.renameatx_np
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
    except (AttributeError, OSError) as exc:
        return f"renameatx_np is unavailable: {exc}"
    if (
        renameatx_np(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_SWAP,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        return str(OSError(error_number, os.strerror(error_number)))
    return None


def _write_child_bootstrap_error(message: str) -> None:
    """Write one bounded child-bootstrap diagnostic to the supervised stderr.

    Parameters
    ----------
    message:
        Specific fail-closed reason to preserve before ``os._exit``.
    """

    payload = f"menagerie worker bootstrap failed: {message}\n".encode(
        "utf-8", errors="backslashreplace"
    )
    try:
        os.write(2, payload[:4096])
    except OSError:
        pass


def _write_child_bootstrap_notice(message: str) -> None:
    """Write one bounded successful child-bootstrap diagnostic to supervised stderr.

    Parameters
    ----------
    message:
        Specific benign bootstrap event to preserve for later supervision diagnostics.
    """

    payload = f"menagerie worker bootstrap: {message}\n".encode("utf-8", errors="backslashreplace")
    try:
        os.write(2, payload[:4096])
    except OSError:
        pass


def _verify_darwin_libomp_registration_blocker(path: Path) -> tuple[int, int]:
    """Verify and return the trusted pre-sandbox OpenMP blocker identity.

    Parameters
    ----------
    path:
        Exact PID-bound blocker created by the child bootstrap.

    Returns
    -------
    tuple[int, int]
        Device and inode pair used to verify cleanup.

    Raises
    ------
    SandboxUnavailableError
        If the blocker is absent, replaced, writable, or owned by another user.
    """

    try:
        status = path.lstat()
    except OSError as exc:
        raise SandboxUnavailableError(
            f"Darwin libomp registration blocker is missing at {path}"
        ) from exc
    if stat.S_ISREG(status.st_mode):
        raise SandboxUnavailableError(
            "pre-existing foreign LLVM OpenMP registration file prevented worker bootstrap "
            f"at {path}; this PID-keyed entry may be stale after abnormal termination"
        )
    if not stat.S_ISDIR(status.st_mode):
        raise SandboxUnavailableError(
            f"Darwin libomp registration blocker at {path} has invalid "
            f"mode {stat.S_IFMT(status.st_mode):#o}"
        )
    if stat.S_IMODE(status.st_mode) != 0o500 or status.st_uid != os.getuid():
        raise SandboxUnavailableError(
            f"Darwin libomp registration blocker directory is invalid at {path}"
        )
    return status.st_dev, status.st_ino


def _remove_darwin_libomp_registration_blocker(
    path: Path,
    expected_identity: tuple[int, int],
) -> bool:
    """Remove one unchanged empty OpenMP blocker after sandbox exit.

    Parameters
    ----------
    path:
        Exact PID-bound blocker path.
    expected_identity:
        Parent-recorded device and inode pair.

    Returns
    -------
    bool
        True only when the unchanged empty directory was removed.
    """

    try:
        status = path.lstat()
        if not stat.S_ISDIR(status.st_mode) or (status.st_dev, status.st_ino) != expected_identity:
            return False
        path.rmdir()
    except OSError:
        return False
    return True


def _linux_rss(pid: int) -> int:
    """Read current Linux resident bytes for a child process.

    Parameters
    ----------
    pid:
        Child process ID.

    Returns
    -------
    int
        Resident bytes, or zero when unavailable.
    """

    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _rusage_v2_memory_bytes(buffer: bytes) -> int:
    """Return the enforceable memory reading from one ``rusage_info_v2`` image.

    Parameters
    ----------
    buffer:
        Raw ``rusage_info_v2`` bytes as filled by ``proc_pid_rusage``.

    Returns
    -------
    int
        ``max(ri_resident_size, ri_phys_footprint)`` in bytes.

    Notes
    -----
    Darwin has no kernel memory rlimit, so the parent sampler is the ONLY
    memory containment for macOS workers. ``ri_resident_size`` alone is not an
    enforceable metric there: under system memory pressure the kernel
    compresses and swaps a runaway worker's pages, holding resident size below
    any cap sized to physical RAM while the true footprint keeps growing until
    the OS memorystatus (jetsam) killer delivers an unattributable SIGKILL
    first. ``ri_phys_footprint`` counts dirty+compressed+IOKit memory -- the
    same metric jetsam enforces -- so sampling it makes the configured cap bind
    before the OS kill and the breach classify as ``resource``/``rss-cap``
    instead of an anonymous ``worker terminated by signal 9`` native crash
    (rung-2 m5915 Mixtral incident). The max keeps the reading monotonic with
    the historical resident-only sampler.
    """

    # rusage_info_v2: UUID (16 bytes), six uint64 counters, then
    # ri_resident_size at offset 64 and ri_phys_footprint at offset 72.
    resident = int.from_bytes(buffer[64:72], "little")
    footprint = int.from_bytes(buffer[72:80], "little")
    return max(resident, footprint)


def _macos_rss(pid: int) -> int:
    """Read live enforceable memory bytes for one macOS child via ``proc_pid_rusage``.

    Parameters
    ----------
    pid:
        Sandboxed worker process ID.

    Returns
    -------
    int
        ``max(resident, phys_footprint)`` bytes for the live worker, or zero
        when unavailable.
    """

    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pid_rusage = libproc.proc_pid_rusage
        proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        proc_pid_rusage.restype = ctypes.c_int
        buffer = (ctypes.c_ubyte * 256)()
        if proc_pid_rusage(pid, 2, ctypes.byref(buffer)) != 0:
            return 0
        return _rusage_v2_memory_bytes(bytes(buffer))
    except (AttributeError, OSError, ValueError):
        return 0


def _child_rss(pid: int, *, platform_name: Optional[str] = None) -> int:
    """Read live child RSS using the current host's supported mechanism.

    Parameters
    ----------
    pid:
        Root child process ID.
    platform_name:
        Optional platform override used by parser/dispatch tests.

    Returns
    -------
    int
        Live resident bytes, or zero when the platform sampler is unavailable.
    """

    selected = sys.platform if platform_name is None else platform_name
    if selected == "darwin":
        return _macos_rss(pid)
    if selected.startswith("linux"):
        return _linux_rss(pid)
    return 0


def _process_group_rss(pgid: int, root_pid: int) -> tuple[int, str]:
    """Sum live resident bytes over one worker's whole process group.

    Root-only sampling let multi-process models bypass the memory cap by
    architecture (BL-12(d), fable W-F6 / TB-06): loky executors and
    constructor multiprocessing pools split their footprint across children
    the sampler never saw, and on the Linux lane the sampled root is
    ``strace``, so ``rss_exceeded`` could never fire at all. Summing over the
    bound group closes both doors with one mechanism.

    Parameters
    ----------
    pgid:
        Process-group ID bound at spawn.
    root_pid:
        Root child PID used when group enumeration is unavailable.

    Returns
    -------
    tuple[int, str]
        ``(resident_bytes, basis)`` where basis is ``"process-group-sum"``
        when membership was proven and ``"root-process"`` otherwise.
    """

    members = _live_process_group_members(pgid)
    if members is None:
        return (_child_rss(root_pid), "root-process")
    return (sum(_child_rss(member) for member in members), "process-group-sum")


def _rusage_peak_rss_floor_bytes(
    usage_before: resource.struct_rusage,
    usage_after: resource.struct_rusage,
    *,
    platform_name: Optional[str] = None,
) -> int:
    """Return a correctly scaled rusage floor only for the first reaped child.

    Parameters
    ----------
    usage_before, usage_after:
        Process-lifetime ``RUSAGE_CHILDREN`` snapshots.
    platform_name:
        Optional platform override used by unit tests.

    Returns
    -------
    int
        A byte floor for the first child only. Later calls return zero because
        ``ru_maxrss`` cannot be attributed to an individual reaped child.
    """

    if int(usage_before.ru_maxrss) != 0:
        return 0
    scale = 1 if (sys.platform if platform_name is None else platform_name) == "darwin" else 1024
    return max(0, int(usage_after.ru_maxrss)) * scale


PROCESS_GROUP_SIGNALLED = "signalled"
PROCESS_GROUP_ALREADY_EXITED = "already-exited"
PROCESS_GROUP_SKIPPED_REAPED_PID = "skipped-reaped-pid"
PROCESS_GROUP_SKIPPED_FOREIGN_PID = "skipped-foreign-pid"
PROCESS_GROUP_PERMISSION_DENIED = "permission-denied"
BENIGN_PROCESS_GROUP_TEARDOWNS = frozenset({PROCESS_GROUP_SIGNALLED, PROCESS_GROUP_ALREADY_EXITED})


@dataclass(frozen=True)
class ProcessGroupHandle:
    """Spawn-time identity of one child's own isolated process group.

    Parameters
    ----------
    process:
        Root child launched with ``start_new_session=True``.
    pid, pgid:
        Root child PID and the process-group ID it leads. ``setsid`` in the child
        makes these equal, so the group is knowable the moment the spawn returns.
    isolated:
        Whether the child was observed leading its own group. A child that shares
        a foreign group is never a legal ``killpg`` target.
    start_token:
        PID-reuse-resistant OS process-start token sampled while the child was
        still unreaped, or ``None`` when the host probe was unavailable.
    """

    process: subprocess.Popen[Any]
    pid: int
    pgid: int
    isolated: bool
    start_token: Optional[str]


@dataclass(frozen=True)
class ProcessGroupTeardown:
    """Outcome of one force-teardown attempt against an isolated group.

    Parameters
    ----------
    outcome:
        One frozen outcome name. Only :data:`BENIGN_PROCESS_GROUP_TEARDOWNS`
        members mean the group is provably gone or was provably signalled.
    pid, pgid:
        Targeted root PID and process-group ID.
    detail:
        Bounded parent-owned diagnosis.
    survivors:
        Non-zombie group members observed after a denied signal.
    """

    outcome: str
    pid: int
    pgid: int
    detail: str
    survivors: tuple[int, ...] = ()

    @property
    def benign(self) -> bool:
        """Whether teardown reached its goal without an unresolved hazard.

        Returns
        -------
        bool
            True only when the group was signalled or is provably gone.
        """

        return self.outcome in BENIGN_PROCESS_GROUP_TEARDOWNS


def capture_process_group(process: subprocess.Popen[Any]) -> ProcessGroupHandle:
    """Bind a child's process-group identity while the child is definitely alive.

    Called immediately after the spawn returns, so the PID is still held by this
    exact child and cannot yet have been recycled. Binding here -- rather than
    re-deriving a target from ``process.pid`` at teardown time -- is what closes
    the PID-reuse race at its source.

    Parameters
    ----------
    process:
        Root child launched with ``start_new_session=True``.

    Returns
    -------
    ProcessGroupHandle
        Immutable spawn-time group identity used by every later teardown.
    """

    pid = process.pid
    try:
        observed_pgid: Optional[int] = os.getpgid(pid)
    except OSError:
        # Darwin refuses ``getpgid`` for a zombie, so a child that exited during
        # spawn is unverifiable this way. ``setsid`` still ran before ``exec``,
        # so the group ID is exactly the PID.
        observed_pgid = None
    return ProcessGroupHandle(
        process=process,
        pid=pid,
        pgid=pid,
        isolated=observed_pgid is None or observed_pgid == pid,
        start_token=process_start_token(pid),
    )


def _live_process_group_members(pgid: int) -> Optional[tuple[int, ...]]:
    """Return the non-zombie members of one process group.

    Parameters
    ----------
    pgid:
        Process-group ID to enumerate.

    Returns
    -------
    tuple[int, ...] | None
        Sorted live member PIDs, or ``None`` when the host probe failed and
        group membership therefore cannot be proven either way.
    """

    members: list[int] = []
    if sys.platform.startswith("linux"):
        try:
            entries = sorted(Path("/proc").iterdir())
        except OSError:
            return None
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat_text = (entry / "stat").read_text(encoding="ascii")
            except OSError:
                # The process exited mid-scan, which is exactly "not a survivor".
                continue
            closing = stat_text.rfind(")")
            fields = stat_text[closing + 2 :].split() if closing >= 0 else []
            if len(fields) < 3:
                continue
            try:
                member_pgid = int(fields[2])
            except ValueError:
                continue
            if member_pgid == pgid and fields[0] != "Z":
                members.append(int(entry.name))
        return tuple(sorted(members))
    try:
        completed = subprocess.run(
            ("ps", "-A", "-o", "pid=,pgid=,state="),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in completed.stdout.splitlines():
        columns = line.split(maxsplit=2)
        if len(columns) < 3:
            continue
        try:
            member_pid = int(columns[0])
            member_pgid = int(columns[1])
        except ValueError:
            continue
        if member_pgid == pgid and not columns[2].startswith("Z"):
            members.append(member_pid)
    return tuple(sorted(members))


def _record_process_group_teardown(record: ProcessGroupTeardown) -> ProcessGroupTeardown:
    """Emit a bounded supervisor diagnostic for a non-benign teardown.

    Parameters
    ----------
    record:
        Teardown outcome to record.

    Returns
    -------
    ProcessGroupTeardown
        The same record, so callers can both record and return in one step.
    """

    if record.benign:
        return record
    payload = (
        "menagerie supervisor process-group teardown "
        f"{record.outcome}: pid={record.pid} pgid={record.pgid} "
        f"detail={record.detail} survivors={list(record.survivors)}\n"
    ).encode("utf-8", errors="backslashreplace")
    try:
        os.write(2, payload[:4096])
    except OSError:
        pass
    return record


def _kill_process_group(
    group: ProcessGroupHandle, sig: int = signal.SIGKILL
) -> ProcessGroupTeardown:
    """Terminate a complete isolated process group we can still prove we own.

    The parent decides to force-terminate from a liveness read that is already
    stale by the time the signal is delivered, so the group can drain underneath
    the call. Two distinct hazards follow, and both are resolved here rather than
    swallowed:

    * Darwin's ``killpg`` skips zombies when it counts signallable members and
      reports ``EPERM`` -- not ``ESRCH`` -- when none remain. A group whose last
      member is this parent's own unreaped zombie is therefore *already torn
      down*, yet raises ``PermissionError``.
    * Once the root child has been reaped its PID is free for the kernel to hand
      to an unrelated process. Signalling it would SIGKILL a stranger's group on
      any host where the parent happened to have permission.

    Ownership is established before any signal by two independent facts: the
    child has not been reaped by this parent (so the kernel still holds its PID),
    and its :func:`process_start_token` still matches the token bound at spawn.

    Parameters
    ----------
    group:
        Spawn-time group identity from :func:`capture_process_group`.
    sig:
        Signal to deliver to the verified group. Ownership is what this routine
        establishes, and that proof is signal-independent, so a caller running a
        graceful SIGTERM-then-SIGKILL sequence uses the same verification for
        both phases instead of hand-rolling one of them.

    Returns
    -------
    ProcessGroupTeardown
        Structured outcome. ``benign`` is true only when the group was signalled
        or is provably gone; every other outcome is a recorded cleanup hazard.
    """

    if not group.isolated:
        return _record_process_group_teardown(
            ProcessGroupTeardown(
                PROCESS_GROUP_SKIPPED_FOREIGN_PID,
                group.pid,
                group.pgid,
                "child-does-not-lead-its-own-process-group",
            )
        )
    # ``returncode`` is set only by this parent's own reap. While it is unset the
    # kernel still holds the PID for this exact child, so the PID cannot have been
    # recycled and the group ID bound at spawn is still ours.
    if group.process.returncode is not None:
        return _record_process_group_teardown(
            ProcessGroupTeardown(
                PROCESS_GROUP_SKIPPED_REAPED_PID,
                group.pid,
                group.pgid,
                "root-child-already-reaped-so-pid-may-be-recycled",
            )
        )
    observed_token = process_start_token(group.pid)
    if (
        group.start_token is not None
        and observed_token is not None
        and _start_tokens_comparably_differ(group.start_token, observed_token)
    ):
        return _record_process_group_teardown(
            ProcessGroupTeardown(
                PROCESS_GROUP_SKIPPED_FOREIGN_PID,
                group.pid,
                group.pgid,
                "start-token-changed-while-child-was-unreaped",
            )
        )
    return signal_verified_process_group(group.pid, group.pgid, group.start_token, sig=sig)


def signal_verified_process_group(
    pid: int,
    pgid: int,
    expected_start_token: Optional[str],
    sig: int = signal.SIGKILL,
) -> ProcessGroupTeardown:
    """Signal one already-identity-verified process group and classify the result.

    Callers must have proven, immediately before, that ``pid`` still names the
    process whose group they intend to tear down. This function only classifies
    what the kernel reports back.

    Parameters
    ----------
    pid, pgid:
        Verified root PID and the process-group ID to signal.
    expected_start_token:
        Start token bound when ownership was established, or ``None`` when the
        host probe was unavailable.
    sig:
        Signal to deliver. Every outcome classified below -- ``ESRCH`` for a gone
        group, and Darwin's zombie-counting ``EPERM`` -- is reported by the kernel
        per delivery attempt, not per signal, so the classification holds for any
        signal a caller chooses.

    Returns
    -------
    ProcessGroupTeardown
        Structured outcome; ``benign`` is true only when the group was signalled
        or is provably gone.
    """

    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return ProcessGroupTeardown(
            PROCESS_GROUP_ALREADY_EXITED,
            pid,
            pgid,
            "no-such-process-group",
        )
    except PermissionError:
        survivors = _live_process_group_members(pgid)
        if survivors == ():
            # Every member is an exited or zombie process, so the force teardown
            # this call was asked to perform has already happened.
            return ProcessGroupTeardown(
                PROCESS_GROUP_ALREADY_EXITED,
                pid,
                pgid,
                "process-group-drained-before-signal",
            )
        recheck_token = process_start_token(pid)
        if (
            expected_start_token is not None
            and recheck_token is not None
            and recheck_token != expected_start_token
        ):
            return _record_process_group_teardown(
                ProcessGroupTeardown(
                    PROCESS_GROUP_SKIPPED_FOREIGN_PID,
                    pid,
                    pgid,
                    "pid-recycled-into-a-foreign-process-group",
                    () if survivors is None else survivors,
                )
            )
        return _record_process_group_teardown(
            ProcessGroupTeardown(
                PROCESS_GROUP_PERMISSION_DENIED,
                pid,
                pgid,
                (
                    "membership-unprovable-after-denied-signal"
                    if survivors is None
                    else "live-owned-group-members-refused-the-signal"
                ),
                () if survivors is None else survivors,
            )
        )
    return ProcessGroupTeardown(
        PROCESS_GROUP_SIGNALLED,
        pid,
        pgid,
        # Names the signal actually delivered: a graceful SIGTERM phase and a
        # forced SIGKILL phase must never read identically in a diagnostic.
        f"signal-{sig}-delivered-to-process-group",
    )


def _reap_worker_root(
    process: subprocess.Popen[Any],
    group: ProcessGroupHandle,
    *,
    grace_seconds: float = _WORKER_REAP_GRACE_SECONDS,
) -> Optional[ProcessGroupTeardown]:
    """Reap the worker root under an explicit deadline with a kill escalation.

    Every pre-fix supervision path ended in an unbounded ``process.wait()``
    (BL-12(b), fable W-F5): one D-state worker hung the unattended run forever,
    indistinguishable from a long forward. The reap is now wait-with-deadline,
    then a verified group SIGKILL, then one final bounded wait; expiry raises
    the typed :class:`WorkerUnreapableError` host fault.

    Parameters
    ----------
    process:
        Root worker child.
    group:
        Spawn-time bound group identity.
    grace_seconds:
        Deadline for each bounded wait phase.

    Returns
    -------
    ProcessGroupTeardown | None
        The escalation kill outcome when one was needed, else ``None``.

    Raises
    ------
    WorkerUnreapableError
        If the root cannot be reaped after the escalation deadline.
    """

    try:
        process.wait(timeout=grace_seconds)
        return None
    except subprocess.TimeoutExpired:
        pass
    escalation = _kill_process_group(group, sig=signal.SIGKILL)
    try:
        process.wait(timeout=grace_seconds)
        return escalation
    except subprocess.TimeoutExpired:
        raise WorkerUnreapableError(
            group.pid,
            group.pgid,
            f"SIGKILL escalation outcome {escalation.outcome!r}; "
            f"root still unreaped after 2x{grace_seconds:.0f}s",
        ) from None


def _teardown_group_after_reap(
    pgid: int,
    *,
    grace_seconds: float = _POST_REAP_SURVIVOR_GRACE_SECONDS,
) -> tuple[str, tuple[int, ...]]:
    """Kill and drain process-group survivors after the root has been reaped.

    The normal-exit path used to end supervision with no group teardown at all
    (BL-12(c), fable W-F2): joblib/loky resident executors and constructor
    multiprocessing pools survived as live orphans holding the inherited
    execution flocks, minting a false terminal for every subsequent model.

    Signalling the group POST-reap is safe exactly when the group still has
    members: POSIX forbids reusing a process-group ID while any process keeps
    it as its group, so a non-empty membership proves the ID still names this
    worker's group and cannot have been recycled.

    Parameters
    ----------
    pgid:
        Process-group ID bound at spawn.
    grace_seconds:
        Bounded drain deadline after the survivor SIGKILL.

    Returns
    -------
    tuple[str, tuple[int, ...]]
        ``(outcome, survivors)``: ``"clean"`` with no members, ``"reaped"``
        after a successful drain, ``"unverifiable"`` when membership cannot be
        proven, or ``"survivors"`` with the PIDs that outlived the drain.
    """

    members = _live_process_group_members(pgid)
    if members is None:
        return ("unverifiable", ())
    if members == ():
        return ("clean", ())
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return ("clean", ())
    except PermissionError:
        # Darwin reports EPERM when only zombies remain; re-enumerate below.
        pass
    deadline = time.monotonic() + grace_seconds
    survivors = members
    while time.monotonic() < deadline:
        observed = _live_process_group_members(pgid)
        if observed is None:
            return ("unverifiable", ())
        if observed == ():
            return ("reaped", ())
        survivors = observed
        time.sleep(0.05)
    return ("survivors", tuple(survivors))


def _tail(data: bytes) -> str:
    """Decode a bounded stdout/stderr tail.

    Parameters
    ----------
    data:
        Complete captured stream bytes.

    Returns
    -------
    str
        Replacement-decoded bounded tail.
    """

    return data.decode("utf-8", errors="replace")[-STDIO_TAIL_MAX_CHARS:]


def _hash_file_streaming(path: Path) -> tuple[str, int]:
    """Hash one stream file by bounded chunks without materializing it.

    The pre-fix supervisor read whole model-owned streams into driver memory
    (``read_bytes`` twice, a full replacement-decode for a 1,500-char tail):
    one chatty worker could OOM the single long-lived driver process
    (BL-12(h), fable W-F4).

    Parameters
    ----------
    path:
        Captured stream file.

    Returns
    -------
    tuple[str, int]
        ``(sha256, total_bytes)`` computed with constant memory.
    """

    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_STREAM_HASH_CHUNK_BYTES):
                digest.update(chunk)
                total += len(chunk)
    except OSError:
        return hash_bytes(b""), 0
    return f"sha256:{digest.hexdigest()}", total


def _final_window_bytes(path: Path, window_bytes: int = _COMPLETION_WINDOW_BYTES) -> bytes:
    """Read at most the final ``window_bytes`` of one stream file.

    Parameters
    ----------
    path:
        Captured stream file.
    window_bytes:
        Bounded window size.

    Returns
    -------
    bytes
        Final window, or empty bytes when unreadable.
    """

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - window_bytes))
            return handle.read(window_bytes)
    except OSError:
        return b""


def _tail_file(path: Path) -> str:
    """Decode a bounded tail directly from a stream file.

    Parameters
    ----------
    path:
        Captured stream file.

    Returns
    -------
    str
        Replacement-decoded bounded tail read with constant memory.
    """

    # Four bytes per character upper-bounds UTF-8, so this window always
    # contains at least STDIO_TAIL_MAX_CHARS decoded characters.
    return _tail(_final_window_bytes(path, STDIO_TAIL_MAX_CHARS * 4))


def _verified_worker_completion(
    stdout: bytes,
    challenge: str,
    *,
    request_nonce: Optional[str] = None,
    request_sha256: Optional[str] = None,
) -> Optional[dict[str, str]]:
    """Verify the final normal-completion line against a parent-only challenge.

    Parameters
    ----------
    stdout:
        Exact worker stdout bytes observed by the parent.
    challenge:
        Fresh challenge removed from the worker environment before model code runs.
    request_nonce, request_sha256:
        Exact v3 request association. Supplying both requires the v2 completion
        protocol; omitting both retains only legacy-untrusted diagnostics.

    Returns
    -------
    dict[str, str] | None
        Bound child receipt digest and proof, or ``None`` when normal completion was
        not witnessed.

    Notes
    -----
    The scan is deliberately displacement- and poison-tolerant (BL-12(a),
    fable W-F1): the completion is located by its LAST prefix occurrence in
    the supplied window rather than by a whole-stream strict decode with a
    last-line rule, so buffered model output flushed after the completion, a
    glued partial line, or one invalid byte elsewhere in the stream can no
    longer destroy the attestation of a successful run. The challenge proof
    -- not stream geometry -- is the authority.
    """

    v3 = request_nonce is not None or request_sha256 is not None
    prefix = _WORKER_COMPLETION_V3_PREFIX if v3 else _WORKER_COMPLETION_PREFIX
    prefix_bytes = prefix.encode("ascii")
    index = stdout.rfind(prefix_bytes)
    if index < 0:
        return None
    line_bytes = stdout[index:].split(b"\n", 1)[0]
    try:
        line = line_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    try:
        value = json.loads(line[len(prefix) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    receipt_key = "raw_award_receipt_sha256" if v3 else "receipt_sha256"
    receipt_sha256 = value.get(receipt_key)
    if not isinstance(receipt_sha256, str):
        return None
    if v3:
        if (
            set(value) != {"raw_award_receipt_sha256", "request_nonce", "request_sha256"}
            or value.get("request_nonce") != request_nonce
            or value.get("request_sha256") != request_sha256
            or json.dumps(value, sort_keys=True, separators=(",", ":")) != line[len(prefix) :]
        ):
            return None
        return {receipt_key: receipt_sha256, "completion_line": line}
    proof = value.get("proof")
    if not isinstance(proof, str):
        return None
    proof_payload: dict[str, Any] = {
        "version": "menagerie.crawler.worker-completion.v1",
        "challenge": challenge,
        receipt_key: receipt_sha256,
    }
    expected = stable_hash(proof_payload)
    if proof != expected:
        return None
    return {
        receipt_key: receipt_sha256,
        "completion_line": line,
    }


def worker_completion_challenge_proof(challenge: str, completion_line: str) -> str:
    """Derive the parent-secret proof binding one v3 completion line.

    The challenge is popped from the environment as ``worker.main()``'s first
    statement -- before any model import -- so this proof can only be computed
    by code that ran before model code. It is what makes the completion an
    attestation rather than a claim (BL-12(a), opus W-6: the v3 protocol
    generated the secret and then never verified it).

    Parameters
    ----------
    challenge:
        Fresh parent-minted secret for this exact spawn.
    completion_line:
        Exact v3 completion line without its trailing newline.

    Returns
    -------
    str
        Domain-separated proof digest.
    """

    return stable_hash(
        {
            "version": WORKER_COMPLETION_PROOF_VERSION,
            "challenge": challenge,
            "completion_line": completion_line,
        }
    )


def _verified_worker_completion_file(
    completion_path: Path,
    challenge: str,
    *,
    request_nonce: str,
    request_sha256: str,
) -> Optional[dict[str, str]]:
    """Verify the v3 completion through its dedicated atomic file channel.

    Attestation no longer rides on model-owned stdout (BL-12(a)): the worker
    writes ``completion.json`` next to the receipt, bound to the parent's
    completion challenge. Model code cannot forge the record -- the challenge
    was popped before any model import -- and nothing a model prints can
    displace it.

    Parameters
    ----------
    completion_path:
        Exact ``completion.json`` path inside the receipt directory.
    challenge:
        Fresh parent secret minted for this exact spawn.
    request_nonce, request_sha256:
        Exact v3 request association.

    Returns
    -------
    dict[str, str] | None
        Bound receipt digest and completion line, or ``None`` when normal
        completion was not witnessed or the proof fails.
    """

    try:
        raw = completion_path.read_bytes()
    except OSError:
        return None
    if len(raw) > _COMPLETION_WINDOW_BYTES:
        return None
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "completion_line",
        "challenge_proof",
    }:
        return None
    if value.get("version") != WORKER_COMPLETION_FILE_VERSION:
        return None
    completion_line = value.get("completion_line")
    claimed_proof = value.get("challenge_proof")
    if not isinstance(completion_line, str) or not isinstance(claimed_proof, str):
        return None
    if claimed_proof != worker_completion_challenge_proof(challenge, completion_line):
        return None
    verified = _verified_worker_completion(
        completion_line.encode("utf-8"),
        challenge,
        request_nonce=request_nonce,
        request_sha256=request_sha256,
    )
    if verified is None:
        return None
    if verified.get("completion_line") != completion_line:
        return None
    return verified


def build_parent_attestation(
    *,
    request_nonce: str,
    request_sha256: str,
    completion: Optional[Mapping[str, str]],
    observation: SupervisorObservation,
) -> dict[str, Any]:
    """Build the frozen v2 parent attestation from parent-observed facts.

    Parameters
    ----------
    request_nonce, request_sha256:
        Exact launched request association.
    completion:
        Challenge-verified completion payload, or ``None`` when normal completion
        was not witnessed.
    observation:
        Complete parent-owned process observation.

    Returns
    -------
    dict[str, Any]
        Closed parent attestation with its canonical self-hash.
    """

    completion_line = None if completion is None else completion.get("completion_line")
    raw_digest = None if completion is None else completion.get("raw_award_receipt_sha256")
    payload: dict[str, Any] = {
        "attestation_version": _PARENT_ATTESTATION_VERSION_V3,
        "request_nonce": request_nonce,
        "request_sha256": request_sha256,
        "completion_line_sha256": (
            None if completion_line is None else hash_bytes(completion_line.encode("utf-8"))
        ),
        "named_raw_award_receipt_sha256": raw_digest,
        "exit_code": observation.exit_code,
        "signal": observation.signal_number,
        "timed_out": observation.timed_out,
        "rss_exceeded": observation.rss_exceeded,
        "peak_rss_bytes": observation.peak_rss_bytes,
        "stdout_sha256": observation.stdout_sha256,
        "stderr_sha256": observation.stderr_sha256,
        "started_at": observation.started_at,
        "finished_at": observation.finished_at,
        # DX-11/RE-12: the teardown facts ride the attestation, so no later
        # reader can bless a run whose process group outlived supervision.
        "group_teardown_outcome": observation.group_teardown_outcome,
        "group_survivor_pids": list(observation.group_survivor_pids),
    }
    payload["attestation_sha256"] = stable_hash(payload)
    return payload


def parent_success_attestation_sha256(completion_line: str, observation: Mapping[str, Any]) -> str:
    """Hash one parent-owned success attestation from exact process facts.

    Parameters
    ----------
    completion_line:
        Challenge-verified final worker completion line.
    observation:
        Exact persisted parent process facts.

    Returns
    -------
    str
        Domain-separated parent attestation digest.
    """

    return stable_hash(
        {
            "version": "menagerie.crawler.parent-success-attestation.v1",
            "completion_line": completion_line,
            "exit_code": observation.get("exit_code"),
            "signal": observation.get("signal"),
            "wall_seconds": observation.get("wall_seconds"),
            "cpu_seconds": observation.get("cpu_seconds"),
            "peak_rss_bytes": observation.get("peak_rss_bytes"),
            "stdout_sha256": observation.get("stdout_sha256"),
            "stderr_sha256": observation.get("stderr_sha256"),
        }
    )


def _rusage_seconds(usage: resource.struct_rusage) -> float:
    """Return combined user and system CPU seconds.

    Parameters
    ----------
    usage:
        Child resource usage snapshot.

    Returns
    -------
    float
        CPU seconds.
    """

    return float(usage.ru_utime + usage.ru_stime)


def _linux_audited_argv(
    sandboxed_argv: Sequence[str], audit_executable: str, audit_path: Path
) -> tuple[str, ...]:
    """Wrap a Linux sandbox command in a parent-owned syscall broker.

    Parameters
    ----------
    sandboxed_argv:
        Complete bubblewrap or unshare command.
    audit_executable:
        Absolute path to ``strace``.
    audit_path:
        Parent-owned broker output path outside every child-writable bind.

    Returns
    -------
    tuple[str, ...]
        Broker command supervising the complete sandboxed process tree.
    """

    return (
        audit_executable,
        "-f",
        "-q",
        "-yy",
        "-e",
        (
            "trace=%network,%file,%process,mmap,read,pread64,readv,preadv,preadv2,"
            "readlinkat,name_to_handle_at,open_by_handle_at"
        ),
        "-s",
        "4096",
        "-o",
        str(audit_path),
        "--",
        *sandboxed_argv,
    )


def _parent_owned_audit_path(
    scratch_root: Path,
    write_roots: Sequence[Path],
    *,
    filename: str = "sandbox-syscalls.log",
) -> tuple[Path, tuple[int, int]]:
    """Create immutable-identity telemetry storage outside child-writable roots.

    Parameters
    ----------
    scratch_root:
        Supervisor scratch directory used to choose a nearby parent-owned sibling.
    write_roots:
        Roots that the OS sandbox exposes writable to the child.
    filename:
        Fixed telemetry filename inside the parent-owned directory.

    Returns
    -------
    tuple[pathlib.Path, tuple[int, int]]
        Audit path and its parent-recorded device/inode identity.

    Raises
    ------
    SandboxUnavailableError
        If no parent-owned location can be established.
    """

    roots = tuple(root.resolve() for root in write_roots)
    base = scratch_root.resolve().parent
    for _attempt in range(8):
        directory = base / f".menagerie-parent-audit-{os.getpid()}-{time.time_ns()}"
        if not any(directory == root or root in directory.parents for root in roots):
            try:
                directory.mkdir(mode=0o700)
                path = directory / filename
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(descriptor)
                os.link(path, directory / f"{filename}.anchor")
                status = path.stat()
                return path, (status.st_dev, status.st_ino)
            except OSError as exc:
                raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value) from exc
        base = base.parent
    raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)


def _request_allowed_read_paths(
    argv: Sequence[str],
    manifest: Optional[ExecutionReadManifestV2 | ExecutionReadManifestV3] = None,
    *,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> tuple[Path, ...]:
    """Return parent bootstrap files plus one compiled read capability.

    Parameters
    ----------
    argv:
        Original unsandboxed command vector.
    manifest:
        Frozen trusted capability compiled outside author-controlled request data.
    verification_token:
        Optional cache-created spawn proof shared by every v3 projection consumer.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Parent-verified paths allowed as bootstrap, implementation, standard input,
        or exact runtime files.
    """

    allowed: list[Path] = []
    if argv:
        executable = Path(argv[0])
        if executable.is_file():
            allowed.append(executable)
            # A virtual-environment interpreter reads its own pyvenv.cfg during startup to
            # find the base prefix owning the standard library. The marker carries no code
            # suffix, so the environment suffix grants can never cover it, and site aborts the
            # worker when the read is denied. Name the one exact file, only when it exists.
            marker = executable.parent.parent / "pyvenv.cfg"
            if executable.parent.name == "bin" and marker.is_file():
                allowed.append(marker)
    try:
        request_index = argv.index("--request") + 1
        request_path = Path(argv[request_index]).resolve()
    except (ValueError, IndexError):
        return tuple(dict.fromkeys(allowed))
    if request_path.is_file():
        allowed.append(request_path)
    if manifest is None:
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            request = None
        recipe = request.get("recipe") if isinstance(request, Mapping) else None
        if isinstance(recipe, Mapping):
            legacy_paths = [recipe.get("path")]
            members = recipe.get("code_manifest")
            if isinstance(members, list):
                legacy_paths.extend(
                    member.get("path") for member in members if isinstance(member, Mapping)
                )
            for path_value in legacy_paths:
                if isinstance(path_value, str) and path_value:
                    path = Path(path_value).resolve()
                    if path.is_file():
                        allowed.append(path)
    if manifest is not None:
        verify_execution_read_manifest(
            manifest,
            verification_token=verification_token,
        )
        if isinstance(manifest, ExecutionReadManifestV3):
            capability = environment_read_capability(
                manifest,
                verification_token=verification_token,
            )
            allowed.extend(capability.exact_member_paths)
            allowed.append(capability.environment_prefix)
            allowed.extend(capability.startup_pth_paths)
        elif isinstance(manifest, ExecutionReadManifestV2):
            allowed.extend(exact_read_capability(manifest).member_paths)
        if manifest.standard_input_asset is not None:
            allowed.append(manifest.standard_input_asset[0])
        allowed.extend(path for path, kind in manifest.runtime_support if kind == "runtime-file")
    return tuple(
        dict.fromkeys(
            path.resolve()
            for path in allowed
            if path.is_absolute() and (path.is_file() or path.is_dir())
        )
    )


def _runtime_read_roots(argv: Sequence[str], cwd: Path) -> tuple[Path, ...]:
    """Return environment/source roots limited to runtime-code reads on macOS.

    Parameters
    ----------
    argv:
        Original worker command vector.
    cwd:
        Read-only source working directory.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Python environment and source roots.
    """

    roots = [cwd.resolve()]
    if argv:
        # A virtual environment splits its runtime in two: site-packages lives under the venv
        # prefix reached through the launcher path, while libpython and the standard library
        # live under the base prefix the launcher symlink resolves to. Taking only the resolved
        # path drops site-packages, and taking only the unresolved path drops the standard
        # library. Both spellings collapse to one root for a non-venv interpreter.
        for candidate in (Path(argv[0]), Path(argv[0]).resolve()):
            roots.append(
                candidate.parent.parent if candidate.parent.name == "bin" else candidate.parent
            )
    return tuple(dict.fromkeys(root.resolve() for root in roots))


def _macos_runtime_read_capabilities(
    execution_read_manifest: Optional[ExecutionReadManifestV2 | ExecutionReadManifestV3],
    discovered_roots: Sequence[Path],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return the macOS profile's runtime read roots and package-data read paths.

    Parameters
    ----------
    execution_read_manifest:
        Shipped worker read capability, or ``None`` for the discovery-only path.
    discovered_roots:
        Roots :func:`_runtime_read_roots` derived from the worker argv and its
        read-only source working directory.

    Returns
    -------
    tuple[tuple[pathlib.Path, ...], tuple[pathlib.Path, ...]]
        Roots handed to :func:`generate_macos_sandbox_profile` as
        ``runtime_read_roots``, and the extra package-data paths granted exactly.

    A v3 manifest carries the whole runtime grant on its sealed environment
    prefix, so no discovered root may contribute a code-suffix grant. The
    discovered roots must still be handed to the profile generator, because the
    worker's ``python -m`` source working directory is a ``sys.path`` entry and
    an ordinary ``importlib.metadata`` scan reads the confined metadata names
    inside any ``.dist-info``/``.egg-info`` it finds there. The Linux classifier
    (:func:`_runtime_import_metadata_path_allowed`) accepts exactly those names
    unconditionally, so dropping the roots here makes the Seatbelt authority
    stricter than the Linux one and SIGKILLs the worker before its constructor
    runs. The generator emits only the confined metadata clauses for a
    non-prefix root, never a code or payload grant.
    """

    roots = tuple(discovered_roots)
    if execution_read_manifest is None:
        return roots, _runtime_package_data_paths(roots)
    if isinstance(execution_read_manifest, ExecutionReadManifestV3):
        return (execution_read_manifest.environment_authority.prefix, *roots), ()
    # A v2 manifest names every readable file exactly and grants no root at all.
    return (), ()


def _syscall_name(line: str) -> Optional[str]:
    """Return the traced syscall name from one broker line.

    Parameters
    ----------
    line:
        One text line emitted by the syscall broker.

    Returns
    -------
    str | None
        Syscall name, excluding strace process prefixes.
    """

    match = _SYSCALL_PATTERN.search(line)
    return None if match is None else match.group(1)


def _network_target(line: str) -> str:
    """Return a bounded network target description from a traced syscall.

    Parameters
    ----------
    line:
        Network syscall trace line.

    Returns
    -------
    str
        Sanitized address-family and endpoint excerpt.
    """

    family = "AF_INET6" if "AF_INET6" in line else "AF_INET"
    address_match = re.search(r'(?:inet_addr|inet_pton\([^,]+,)\("?([^"),]+)', line)
    port_match = re.search(r"sin6?_port=htons\((\d+)\)", line)
    address = address_match.group(1) if address_match is not None else "unknown"
    port = port_match.group(1) if port_match is not None else "unknown"
    return f"{family}:{address}:{port}"[:500]


def _decoded_trace_paths(line: str) -> tuple[str, ...]:
    """Decode quoted filesystem paths from one syscall trace line.

    Parameters
    ----------
    line:
        File syscall trace line.

    Returns
    -------
    tuple[str, ...]
        Best-effort decoded quoted values in call order.
    """

    values: list[str] = []
    for match in _QUOTED_PATH_PATTERN.finditer(line):
        encoded = match.group(1)
        try:
            values.append(bytes(encoded, "utf-8").decode("unicode_escape"))
        except UnicodeDecodeError:
            values.append(encoded)
    return tuple(values)


def _sandbox_temporary_alias_path(candidate: Path, write_roots: Sequence[Path]) -> Path:
    """Map Linux private temporary mount aliases back to the worker scratch.

    Parameters
    ----------
    candidate:
        Absolute path observed inside the bubblewrap namespace.
    write_roots:
        Writable roots with the worker scratch first.

    Returns
    -------
    pathlib.Path
        Host-side scratch path for private ``/tmp`` and ``/dev/shm`` aliases.
    """

    if not write_roots:
        return candidate
    for sandbox_alias in (Path("/tmp"), Path("/dev/shm")):
        if candidate.parent == sandbox_alias and re.fullmatch(
            r"__KMP_REGISTERED_LIB_\d+_\d+", candidate.name
        ):
            return write_roots[0] / "tmp" / candidate.relative_to(sandbox_alias)
    return candidate


def _outside_allowed_roots(path_text: str, cwd: Path, write_roots: Sequence[Path]) -> bool:
    """Return whether a traced write path is outside every allowed root.

    Parameters
    ----------
    path_text:
        Traced absolute or working-directory-relative path.
    cwd:
        Worker working directory.
    write_roots:
        Sole writable roots granted to the OS sandbox.

    Returns
    -------
    bool
        True when the path is not beneath an allowed root.
    """

    path = Path(path_text)
    if path.is_absolute():
        path = _sandbox_temporary_alias_path(path, write_roots)
    candidate = (path if path.is_absolute() else cwd / path).resolve()
    roots = tuple(root.resolve() for root in write_roots)
    return not any(candidate == root or root in candidate.parents for root in roots)


def _telemetry_failure_observation(detail: str) -> SandboxDenialObservation:
    """Return a fail-closed policy observation for invalid broker telemetry.

    Parameters
    ----------
    detail:
        Bounded parent-owned integrity diagnosis.

    Returns
    -------
    SandboxDenialObservation
        Poisoning checkpoint-read observation.
    """

    # Telemetry integrity failures keep poisoning the receipt (no award on
    # unverifiable telemetry), but they no longer wear the ``checkpoint-read``
    # label: a fact about the HOST's log delivery must never read as a model
    # policy violation (BL-12(e), fable W-F3 lane b -- 11 of the pilot's 13
    # failed:policy heads were mislabeled by the old collapsed vocabulary).
    return SandboxDenialObservation(
        telemetry_failure=detail,
    )


def _resolved_trace_path(path_text: str, cwd: Path) -> Path:
    """Resolve one traced path conservatively against the worker directory.

    Parameters
    ----------
    path_text:
        Absolute or tracee-relative path.
    cwd:
        Worker working directory.

    Returns
    -------
    pathlib.Path
        Normalized candidate path.
    """

    path = Path(path_text)
    return (path if path.is_absolute() else cwd / path).resolve()


def _read_path_is_allowed(
    path_text: str,
    cwd: Path,
    write_roots: Sequence[Path],
    allowed_read_paths: Sequence[Path],
    runtime_code_roots: Sequence[Path],
    *,
    host_transport_capability: Optional[HostTransportLibraryCapability] = None,
    directory_only: bool = False,
    standard_input_asset: Optional[Path] = None,
) -> bool:
    """Return whether a kernel-level read belongs to the closed runtime allowlist.

    Parameters
    ----------
    path_text:
        Path decoded from one successful read-only open.
    cwd:
        Worker working directory.
    write_roots:
        Fresh scratch/result roots whose contents are parent-authorized runtime state.
    allowed_read_paths:
        Exact source/input paths derived from the immutable worker request.
    runtime_code_roots:
        Environment and verified-source roots limited to runtime-code reads.
    host_transport_capability:
        Exact interpreter ELF members mounted for this worker launch.
    directory_only:
        Whether the traced open explicitly required a directory descriptor.
    standard_input_asset:
        Exact trusted standard asset. It is the only model-data-shaped path that
        may bypass the deny-first classifier.

    Returns
    -------
    bool
        True only for declared inputs, source/runtime code, or OS runtime support.
    """

    raw_path = Path(path_text)
    lexical_candidate = (raw_path if raw_path.is_absolute() else cwd / raw_path).absolute()
    candidate = _resolved_trace_path(path_text, cwd)
    if raw_path.is_absolute():
        candidate = _sandbox_temporary_alias_path(candidate, write_roots).resolve()
    roots = tuple(root.resolve() for root in write_roots)
    allowed = tuple(path.resolve() for path in allowed_read_paths)
    runtime_roots = tuple(root.resolve() for root in runtime_code_roots)
    if lexical_candidate in _SYSTEM_READ_FILES:
        return True
    if path_text in {"self", "self/fd", "self/mountinfo"} or re.fullmatch(
        r"\d+/(?:fd|ns)(?:/.*)?", path_text
    ):
        return True
    if any(candidate == root or root in candidate.parents for root in _SPECIAL_READ_ROOTS):
        return True
    if any(
        candidate == root
        or root in candidate.parents
        or lexical_candidate == root
        or root in lexical_candidate.parents
        for root in _SYSTEM_TIMEZONE_READ_ROOTS
    ):
        return True
    # Host OS data trees by prefix + category (B1, 2026-08-12): /usr/share and
    # /etc carry only OS payloads no worker can write, never model checkpoints.
    # Resolved spelling ONLY -- a lexical match would let a "/usr/share/../.."
    # dot-dot escape launder an arbitrary read through the system prefix; the
    # macOS /etc symlink is covered by the /private/etc root after resolution.
    if any(candidate == root or root in candidate.parents for root in _SYSTEM_DATA_READ_ROOTS):
        return True
    if any(candidate == root or root in candidate.parents for root in roots):
        return True
    if _runtime_import_metadata_path_allowed(candidate):
        return True
    if directory_only and any(
        candidate == root or root in candidate.parents for root in runtime_roots
    ):
        return True
    if candidate.is_relative_to(Path("/usr/lib/locale")) and (
        candidate.name == "locale-archive" or candidate.name.startswith("LC_")
    ):
        return True
    try:
        if candidate.is_dir():
            return True
    except OSError:
        pass
    if _runtime_native_code_path_allowed(candidate, runtime_roots):
        return True
    if _system_transport_library_path_allowed(candidate, host_transport_capability):
        return True
    if _runtime_code_path_allowed(candidate, runtime_code_roots):
        return True
    standard_asset = None if standard_input_asset is None else standard_input_asset.resolve()
    if _allowed_exact_or_derived_file(candidate, allowed):
        return True
    if _runtime_model_data_path(candidate) and candidate != standard_asset:
        startup_pth = (
            candidate.suffix.lower() == ".pth"
            and candidate in allowed
            and _runtime_static_path_allowed(candidate)
        )
        if not startup_pth:
            return False
    if any(candidate == path or path in candidate.parents for path in allowed):
        return True
    if candidate.name == "gconv-modules.cache" and "gconv" in candidate.parts:
        return True
    return False


def _system_transport_library_path_allowed(
    path: Path,
    capability: Optional[HostTransportLibraryCapability] = None,
) -> bool:
    """Return whether a path belongs to the exact host transport capability.

    Parameters
    ----------
    path:
        Resolved candidate path observed by parent syscall telemetry.
    capability:
        Closed exact capability derived for the selected worker interpreter.

    Returns
    -------
    bool
        True only for an exact declared transport member.
    """

    return capability is not None and capability.allows(path)


def _trace_process_id(line: str) -> str:
    """Return the strace process identifier or a single-process sentinel.

    Parameters
    ----------
    line:
        One syscall telemetry line.

    Returns
    -------
    str
        Stable key used to pair unfinished and resumed syscall records.
    """

    match = _TRACE_PID_PATTERN.match(line)
    return match.group("pid") if match is not None else "<single-process>"


def _complete_trace_records(lines: Sequence[str]) -> Optional[tuple[str, ...]]:
    """Join interleaved strace unfinished/resumed records without losing results.

    Parameters
    ----------
    lines:
        Integrity-checked raw strace lines.

    Returns
    -------
    tuple[str, ...] | None
        Complete records, or ``None`` when continuation telemetry is inconsistent.
    """

    pending: dict[tuple[str, str], str] = {}
    completed: list[str] = []
    for line in lines:
        if "<unfinished ...>" in line:
            syscall = _syscall_name(line)
            if syscall is None:
                return None
            key = (_trace_process_id(line), syscall)
            if key in pending:
                return None
            pending[key] = line.replace("<unfinished ...>", "", 1).rstrip()
            continue
        resumed = _RESUMED_TRACE_PATTERN.match(line)
        if resumed is not None:
            key = (_trace_process_id(line), resumed.group("syscall"))
            prefix = pending.pop(key, None)
            if prefix is None:
                return None
            completed.append(f"{prefix}{resumed.group('suffix')}")
            continue
        completed.append(line)
    terminated_processes = {
        _trace_process_id(line) for line in lines if _TERMINAL_TRACE_PATTERN.search(line)
    }
    benign_terminal_pending = {"read", "wait4", "exit_group"}
    if pending and not all(
        process_id in terminated_processes and syscall in benign_terminal_pending
        for process_id, syscall in pending
    ):
        return None
    return tuple(completed)


def _read_only_open_result(line: str) -> Optional[int]:
    """Return the file descriptor result from one complete read-only open.

    Parameters
    ----------
    line:
        Complete ``open``, ``openat``, or ``openat2`` trace record.

    Returns
    -------
    int | None
        Nonnegative descriptor for success, negative result for failure, or ``None``
        when the supposedly complete result is unparsable.
    """

    matches = _OPEN_RESULT_PATTERN.findall(line)
    return int(matches[-1]) if matches else None


def _trace_line_is_well_formed(line: str) -> bool:
    """Return whether one nonempty strace line has a recognized complete form.

    Parameters
    ----------
    line:
        One parent-owned telemetry line.

    Returns
    -------
    bool
        True for syscall, signal, continuation, or terminal records.
    """

    return bool(
        _syscall_name(line)
        or _TERMINAL_TRACE_PATTERN.search(line)
        or "<unfinished ...>" in line
        or "resumed>" in line
        or re.search(r"(?:^|\s)--- SIG[A-Z0-9]+ ", line)
    )


def _worker_trace_records(
    records: Sequence[str],
    capability: HostTransportLibraryCapability,
) -> Optional[tuple[str, ...]]:
    """Return records at and after the selected worker interpreter exec.

    Parent strace starts outside bubblewrap, so the sandbox launcher's own loader
    traffic precedes the model worker and is not part of worker read authority.

    Parameters
    ----------
    records:
        Complete chronological syscall records for the bubblewrap process tree.
    capability:
        Exact transport capability bound to the selected worker interpreter.

    Returns
    -------
    tuple[str, ...] | None
        Worker-phase records beginning with its exact exec, or ``None`` when the
        audited process tree never entered the declared interpreter.
    """

    for index, line in enumerate(records):
        if _syscall_name(line) != "execve":
            continue
        paths = _decoded_trace_paths(line)
        if paths and _resolved_trace_path(paths[0], Path("/")) == capability.interpreter:
            return tuple(records[index:])
    return None


def _parse_linux_denial_audit(
    audit_path: Path,
    cwd: Path,
    write_roots: Sequence[Path],
    *,
    expected_identity: Optional[tuple[int, int]] = None,
    allowed_read_paths: Sequence[Path] = (),
    runtime_code_roots: Sequence[Path] = (),
    host_transport_capability: Optional[HostTransportLibraryCapability] = None,
    standard_input_asset: Optional[Path] = None,
) -> SandboxDenialObservation:
    """Parse Linux syscall telemetry into closed worker policy observations.

    Parameters
    ----------
    audit_path:
        Broker output produced for one child process tree.
    cwd:
        Worker working directory used to resolve relative write targets.
    write_roots:
        Sole OS-sandbox writable roots.
    expected_identity:
        Parent-recorded device/inode pair for replacement detection.
    allowed_read_paths:
        Explicit source/input paths authorized for model execution.
    runtime_code_roots:
        Environment and verified-source roots limited to runtime-code reads.
    host_transport_capability:
        Exact interpreter ELF members mounted for this worker launch.
    standard_input_asset:
        Exact trusted standard asset allowed through deny-first model-data checks.

    Returns
    -------
    SandboxDenialObservation
        Deduplicated network and outside-write denials.
    """

    try:
        status = audit_path.stat()
        if not stat.S_ISREG(status.st_mode):
            return _telemetry_failure_observation("not-regular")
        if expected_identity is not None and (status.st_dev, status.st_ino) != expected_identity:
            return _telemetry_failure_observation("replaced")
        anchor = audit_path.with_name(f"{audit_path.name}.anchor")
        if expected_identity is not None:
            anchor_status = anchor.stat()
            if (anchor_status.st_dev, anchor_status.st_ino) != (status.st_dev, status.st_ino):
                return _telemetry_failure_observation("replaced")
        content = audit_path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return _telemetry_failure_observation("unparsable-encoding")
    except OSError:
        return _telemetry_failure_observation("missing")
    raw_lines = content.splitlines()
    if not raw_lines:
        return _telemetry_failure_observation("empty")
    if not _TERMINAL_TRACE_PATTERN.search(raw_lines[-1]):
        return _telemetry_failure_observation("truncated")
    if any(not _trace_line_is_well_formed(line) for line in raw_lines if line.strip()):
        return _telemetry_failure_observation("unparsable-record")
    completed_lines = _complete_trace_records(raw_lines)
    if completed_lines is None:
        return _telemetry_failure_observation("unparsable-continuation")
    if host_transport_capability is not None:
        worker_records = _worker_trace_records(completed_lines, host_transport_capability)
        if worker_records is None:
            return _telemetry_failure_observation("missing-worker-exec")
        completed_lines = worker_records
    socket_targets: list[str] = []
    write_paths: list[str] = []
    checkpoint_paths: list[str] = []
    failed_read_probe_paths: list[str] = []
    for line in completed_lines:
        syscall = _syscall_name(line)
        if syscall in {"connect", "send", "sendmsg", "sendmmsg", "sendto"} and (
            "AF_INET" in line or "AF_INET6" in line
        ):
            socket_targets.append(_network_target(line))
            continue
        if syscall in {"open", "openat", "openat2"} and not any(
            flag in line for flag in _WRITE_OPEN_FLAGS
        ):
            paths = _decoded_trace_paths(line)
            if paths:
                path_text = paths[0]
                allowed = _read_path_is_allowed(
                    path_text,
                    cwd,
                    write_roots,
                    allowed_read_paths,
                    runtime_code_roots,
                    host_transport_capability=host_transport_capability,
                    directory_only="O_DIRECTORY" in line,
                    standard_input_asset=standard_input_asset,
                )
                if allowed:
                    continue
                result = _read_only_open_result(line)
                if result is None:
                    return _telemetry_failure_observation("unparsable-open-result")
                if result < 0:
                    failed_read_probe_paths.append(path_text)
                    candidate = _resolved_trace_path(path_text, cwd)
                    loader_probe = (
                        "O_DIRECTORY" in line
                        or candidate.suffix.lower()
                        in {
                            ".a",
                            ".dylib",
                            ".pyd",
                            ".so",
                        }
                        or ".so." in candidate.name.lower()
                    )
                    if not loader_probe or candidate.suffix.lower() in _MODEL_DATA_SUFFIXES:
                        checkpoint_paths.append(path_text)
                else:
                    checkpoint_paths.append(path_text)
            continue
        if syscall in {"readlinkat", "name_to_handle_at"}:
            paths = _decoded_trace_paths(line)
            if paths and not _read_path_is_allowed(
                paths[0],
                cwd,
                write_roots,
                allowed_read_paths,
                runtime_code_roots,
                host_transport_capability=host_transport_capability,
                standard_input_asset=standard_input_asset,
            ):
                checkpoint_paths.append(paths[0])
            continue
        if syscall == "open_by_handle_at":
            checkpoint_paths.append("<open_by_handle_at:undeclared-file-handle>")
            continue
        if syscall in {"mmap", "read", "pread64", "readv", "preadv", "preadv2"}:
            descriptor_path = re.search(r"\b\d+<([^>]+)>", line)
            descriptor_path_text = descriptor_path.group(1) if descriptor_path is not None else None
            if (
                descriptor_path_text is not None
                and not descriptor_path_text.startswith(_NON_FILE_DESCRIPTOR_PREFIXES)
                and not _read_path_is_allowed(
                    descriptor_path_text,
                    cwd,
                    write_roots,
                    allowed_read_paths,
                    runtime_code_roots,
                    host_transport_capability=host_transport_capability,
                    standard_input_asset=standard_input_asset,
                )
            ):
                checkpoint_paths.append(descriptor_path_text)
            continue
        if syscall not in _WRITE_SYSCALLS or not any(
            f" {errno} " in line for errno in _DENIED_ERRNOS
        ):
            continue
        if syscall in {"open", "openat", "openat2"} and not any(
            flag in line for flag in _WRITE_OPEN_FLAGS
        ):
            continue
        paths = _decoded_trace_paths(line)
        outside_paths = [path for path in paths if _outside_allowed_roots(path, cwd, write_roots)]
        if outside_paths:
            write_paths.extend(outside_paths)
        elif not paths:
            write_paths.append(f"<{syscall}:denied-outside-sandbox>")
    unique_targets = tuple(dict.fromkeys(socket_targets))
    unique_paths = tuple(dict.fromkeys(write_paths))
    unique_checkpoints = tuple(dict.fromkeys(checkpoint_paths))
    unique_failed_probes = tuple(dict.fromkeys(failed_read_probe_paths))
    return SandboxDenialObservation(
        network_attempted=bool(unique_targets),
        socket_targets=unique_targets,
        write_outside_scratch_attempted=bool(unique_paths),
        write_paths=unique_paths,
        checkpoint_or_weight_read_attempted=bool(unique_checkpoints),
        checkpoint_paths=unique_checkpoints,
        failed_read_probe_paths=unique_failed_probes,
    )


def _macos_denial_message(line: str) -> Optional[str]:
    """Extract one Seatbelt denial message from an audited macOS record.

    Parameters
    ----------
    line:
        One parent-owned NDJSON or legacy textual audit record.

    Returns
    -------
    str | None
        Denial message, or ``None`` when the record is unrecognized.
    """

    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, Mapping):
            return None
        for key in ("eventMessage", "composedMessage", "message"):
            message = value.get(key)
            if isinstance(message, str) and "deny" in message.lower():
                return message
        return None
    return stripped if "deny" in stripped.lower() else None


def _macos_denial_process_ids(line: str) -> frozenset[int]:
    """Extract process and parent identifiers from one macOS log record.

    Parameters
    ----------
    line:
        Unified-log NDJSON or legacy Seatbelt text.

    Returns
    -------
    frozenset[int]
        Process identifiers explicitly associated with the record.
    """

    stripped = line.strip()
    identifiers: set[int] = set()
    if stripped.startswith("{"):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, Mapping):
            for key in ("processID", "parentProcessID", "pid", "ppid"):
                candidate = value.get(key)
                if isinstance(candidate, int):
                    identifiers.add(candidate)
    identifiers.update(int(value) for value in re.findall(r"\((\d+)\)\s+deny", line))
    return frozenset(identifiers)


def _macos_denial_process_path(line: str) -> Optional[Path]:
    """Extract the denied process image path from one unified-log record.

    Parameters
    ----------
    line:
        Unified-log NDJSON record.

    Returns
    -------
    pathlib.Path | None
        Resolved process image when the record carries one.
    """

    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    for key in ("processImagePath", "senderImagePath"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.startswith("/"):
            return Path(candidate).resolve()
    return None


def _macos_denial_excerpt(message: str) -> str:
    """Return a bounded denial summary while retaining its target path.

    Forced Seatbelt telemetry appends a multiline diagnostic report after the ordinary
    one-line denial. Keeping the tail would discard the leading denied path and make the
    parent observation unactionable.

    Parameters
    ----------
    message:
        Decoded unified-log event message.

    Returns
    -------
    str
        At most 500 characters from the end of the first denial line.
    """

    first_line = message.splitlines()[0] if message.splitlines() else message
    return first_line[-500:]


def _macos_file_read_target(message: str) -> str:
    """Return the exact denied read target when the Seatbelt line carries one.

    Parameters
    ----------
    message:
        Decoded unified-log event message.

    Returns
    -------
    str
        Denied path, or a bounded diagnostic excerpt for an unfamiliar record shape.
    """

    first_line = message.splitlines()[0] if message.splitlines() else message
    for separator in (" file-read-data ", " file read data "):
        _prefix, found, target = first_line.partition(separator)
        if found and target:
            return target
    return _macos_denial_excerpt(message)


def _macos_denial_audit(
    telemetry: bytes,
    *,
    expected_process_ids: Sequence[int] = (),
    expected_process_roots: Sequence[Path] = (),
    ignored_process_ids: Sequence[int] = (),
) -> SandboxDenialObservation:
    """Parse completion-marked parent-owned macOS Seatbelt telemetry.

    Parameters
    ----------
    telemetry:
        Complete bytes written by the parent-controlled unified-log collector.
    expected_process_ids:
        Sandboxed worker process-tree roots. Records outside this scope are noise.
    expected_process_roots:
        Parent-owned executable/source roots that scope a descendant whose denial record
        omitted its ancestry.
    ignored_process_ids:
        Parent-created audit sentinel processes that prove collector delivery only.

    Returns
    -------
    SandboxDenialObservation
        Parsed denials, with fail-closed poison for missing or malformed completion.
    """

    try:
        lines = telemetry.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        return _telemetry_failure_observation("unparsable-encoding")
    if not lines:
        return _telemetry_failure_observation("empty")
    completed = lines[-1] == _MACOS_AUDIT_COMPLETION_MARKER
    records = lines[:-1] if completed else lines

    network: list[str] = []
    writes: list[str] = []
    checkpoint_paths: list[str] = []
    expected_ids = set(expected_process_ids)
    ignored_ids = set(ignored_process_ids)
    process_roots = tuple(path.resolve() for path in expected_process_roots)
    unparseable = False
    unattributable_denial = False
    for line in records:
        record_ids = _macos_denial_process_ids(line)
        if ignored_ids.intersection(record_ids):
            continue
        if expected_ids or process_roots:
            process_path = _macos_denial_process_path(line)
            scoped_by_path = process_path is not None and any(
                process_path == root or root in process_path.parents for root in process_roots
            )
            if not expected_ids.intersection(record_ids) and not scoped_by_path:
                if process_path is None:
                    message = _macos_denial_message(line)
                    if message is None or any(
                        marker in message.lower()
                        for marker in (
                            "network",
                            "file-write",
                            "file write",
                            "file-read-data",
                            "file read data",
                        )
                    ):
                        unattributable_denial = True
                continue
            # Records may carry only the denied descendant PID. Runtime-root scoping
            # admits that first record; retaining its IDs grows the ancestry closure.
            expected_ids.update(record_ids)
        message = _macos_denial_message(line)
        if message is None:
            unparseable = True
            continue
        lowered = message.lower()
        recognized = False
        if "network" in lowered:
            network.append(_macos_denial_excerpt(message))
            recognized = True
        if "file-write" in lowered or "file write" in lowered:
            writes.append(_macos_denial_excerpt(message))
            recognized = True
        if "file-read-data" in lowered or "file read data" in lowered:
            checkpoint_paths.append(_macos_file_read_target(message))
            recognized = True
        # Seatbelt also emits unrelated denials such as mach-lookup. They are not
        # evidence of the network/file policy classes and must not poison this worker.
        del recognized
    observed = SandboxDenialObservation(
        network_attempted=bool(network),
        socket_targets=tuple(dict.fromkeys(network)),
        write_outside_scratch_attempted=bool(writes),
        write_paths=tuple(dict.fromkeys(writes)),
        checkpoint_or_weight_read_attempted=bool(checkpoint_paths),
        checkpoint_paths=tuple(dict.fromkeys(checkpoint_paths)),
    )
    failures: list[SandboxDenialObservation] = []
    if not completed:
        failures.append(_telemetry_failure_observation("truncated"))
    if unparseable:
        failures.append(_telemetry_failure_observation("unparsable-record"))
    if unattributable_denial:
        failures.append(_telemetry_failure_observation("unattributable-denial"))
    return _merge_denial_observations(observed, *failures)


def _parse_macos_denial_audit(
    audit_path: Path,
    *,
    expected_identity: Optional[tuple[int, int]] = None,
    expected_process_ids: Sequence[int] = (),
    expected_process_roots: Sequence[Path] = (),
    ignored_process_ids: Sequence[int] = (),
) -> SandboxDenialObservation:
    """Verify and parse one parent-owned macOS Seatbelt audit channel.

    Parameters
    ----------
    audit_path:
        Parent-controlled unified-log output path.
    expected_identity:
        Parent-recorded device/inode pair for replacement detection.
    expected_process_ids:
        Sandboxed process-tree root identifiers used to discard machine-wide noise.
    expected_process_roots:
        Parent-owned roots used to recognize descendant process images.
    ignored_process_ids:
        Parent-owned audit sentinel process identifiers.

    Returns
    -------
    SandboxDenialObservation
        Parsed denial or fail-closed telemetry-integrity observation.
    """

    try:
        status = audit_path.stat()
        if not stat.S_ISREG(status.st_mode):
            return _telemetry_failure_observation("not-regular")
        if expected_identity is not None and (status.st_dev, status.st_ino) != expected_identity:
            return _telemetry_failure_observation("replaced")
        anchor = audit_path.with_name(f"{audit_path.name}.anchor")
        anchor_status = anchor.stat()
        if (anchor_status.st_dev, anchor_status.st_ino) != (status.st_dev, status.st_ino):
            return _telemetry_failure_observation("replaced")
        telemetry = audit_path.read_bytes()
    except OSError:
        return _telemetry_failure_observation("missing")
    return _macos_denial_audit(
        telemetry,
        expected_process_ids=expected_process_ids,
        expected_process_roots=expected_process_roots,
        ignored_process_ids=ignored_process_ids,
    )


@dataclass
class _MacOSAuditChannel:
    """Parent-owned unified-log collector state for one sandboxed process tree."""

    path: Path
    expected_identity: tuple[int, int]
    process: subprocess.Popen[Any]
    process_group: ProcessGroupHandle
    handle: BinaryIO
    worker_pid: Optional[int] = None
    sandbox_executable: Optional[str] = None
    profile_path: Optional[Path] = None
    profile_text: Optional[str] = None
    worker_executable: Optional[str] = None
    sentinel_process_ids: list[int] = field(default_factory=list)


def _wait_for_macos_audit_collector(
    process: subprocess.Popen[Any],
    audit_path: Path,
) -> bool:
    """Wait until ``log stream`` has initialized its output channel.

    Parameters
    ----------
    process:
        Parent-owned unified-log collector.
    audit_path:
        Collector output file.

    Returns
    -------
    bool
        True only after the live collector has written its startup banner.
    """

    deadline = time.monotonic() + _MACOS_AUDIT_DELIVERY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            if audit_path.stat().st_size > 0:
                return True
        except OSError:
            return False
        time.sleep(0.05)
    return False


def _emit_macos_audit_sentinel(channel: _MacOSAuditChannel, phase: str) -> bool:
    """Emit and observe a parent-owned denied read through one Seatbelt collector.

    Parameters
    ----------
    channel:
        Live collector and exact sandbox launch authority.
    phase:
        ``startup`` or ``post-exit`` lifetime boundary.

    Returns
    -------
    bool
        True only when the collector receives the unique denial within the bounded
        audit-delivery timeout.
    """

    if (
        channel.sandbox_executable is None
        or channel.profile_path is None
        or channel.profile_text is None
        or channel.worker_executable is None
    ):
        return False
    # The sentinel must be a file that really exists. A lookup that fails with ENOENT is
    # answered by the filesystem before Seatbelt evaluates the profile, so a missing path
    # emits no denial at all and the collector could never observe one. The unique sentinel
    # therefore lives inside the parent-owned audit directory, which is created 0700 outside
    # every child write root and is named by no read grant, so opening it is denied by policy.
    sentinel_token = secrets.token_hex(16)
    sentinel_file = channel.path.parent / f"{_MACOS_AUDIT_SENTINEL_PREFIX}{sentinel_token}-{phase}"
    sentinel_profile = channel.path.parent / (
        f".menagerie-seatbelt-sentinel-{sentinel_token}-{phase}.sb"
    )
    sentinel_path = str(sentinel_file)
    try:
        descriptor = os.open(sentinel_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        sentinel_deny = (
            "(deny file-read-data (with send-signal SIGKILL) (with telemetry) "
            f'(with message "MENAGERIE_MACOS_AUDIT_SENTINEL_{sentinel_token}_{phase}"))'
        )
        if channel.profile_text.count(_MACOS_FILE_READ_DENY_RULE) != 1:
            return False
        sentinel_profile_text = channel.profile_text.replace(
            _MACOS_FILE_READ_DENY_RULE,
            sentinel_deny,
            1,
        )
        with sentinel_profile.open("x", encoding="utf-8") as profile_handle:
            profile_handle.write(sentinel_profile_text)
            if not sentinel_profile_text.endswith("\n"):
                profile_handle.write("\n")
    except OSError:
        return False
    try:
        sentinel = subprocess.Popen(
            (
                channel.sandbox_executable,
                "-f",
                str(sentinel_profile),
                channel.worker_executable,
                "-I",
                "-S",
                "-c",
                "import sys\nopen(sys.argv[1], 'rb')\n",
                sentinel_path,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )
        channel.sentinel_process_ids.append(sentinel.pid)
        # The bound interpreter must die from the profile's forced-report signal. A
        # readable or vanished sentinel exits differently and produces no exact-path
        # denial record, so both states fail the combined signal-and-collector proof.
        if sentinel.wait(timeout=5) != -signal.SIGKILL:
            return False
        deadline = time.monotonic() + _MACOS_AUDIT_DELIVERY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                collected = channel.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
            # The collector writes NDJSON, which escapes every path separator, so the marker
            # is only ever visible once a record is decoded. Searching the raw bytes silently
            # never matches and would report a healthy collector as dead.
            if any(
                sentinel_path in message
                for message in (_macos_denial_message(line) for line in collected.splitlines())
                if message is not None
            ):
                return True
            time.sleep(0.05)
        return False
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        try:
            sentinel_file.unlink()
        except OSError:
            pass
        try:
            sentinel_profile.unlink()
        except OSError:
            pass


def _start_macos_denial_audit(
    scratch_root: Path,
    write_roots: Sequence[Path],
    *,
    sandbox_executable: str,
    profile_path: Path,
    worker_executable: str,
) -> _MacOSAuditChannel:
    """Start the parent-controlled macOS Seatbelt denial collector.

    Parameters
    ----------
    scratch_root:
        Supervisor scratch root used to place a non-child-writable sibling channel.
    write_roots:
        Every root writable by the sandboxed child.
    sandbox_executable, profile_path, worker_executable:
        Exact launch authority used for parent-owned collector sentinel probes.

    Returns
    -------
    _MacOSAuditChannel
        Live parent-owned collector and immutable channel identity.

    Raises
    ------
    SandboxUnavailableError
        If the unified-log audit API cannot be started.
    """

    log_executable = shutil.which("log")
    if log_executable is None:
        raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
    path, identity = _parent_owned_audit_path(
        scratch_root,
        write_roots,
        filename="macos-seatbelt.ndjson",
    )
    handle = path.open("wb")
    process: Optional[subprocess.Popen[Any]] = None
    process_group: Optional[ProcessGroupHandle] = None
    try:
        profile_text = profile_path.read_text(encoding="utf-8")
        process = subprocess.Popen(
            (
                log_executable,
                "stream",
                "--style",
                "ndjson",
                "--predicate",
                'eventMessage CONTAINS[c] "deny"',
            ),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )
        # Bind the collector's group identity while it is definitely alive, before
        # anything can reap it and free its PID for reuse.
        process_group = capture_process_group(process)
        # A fixed sleep races collector initialization under load. Wait for the
        # parent-owned stream's startup banner before emitting the lifetime sentinel.
        if not _wait_for_macos_audit_collector(process, path):
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
        channel = _MacOSAuditChannel(
            path,
            identity,
            process,
            process_group,
            handle,
            sandbox_executable=sandbox_executable,
            profile_path=profile_path,
            profile_text=profile_text,
            worker_executable=worker_executable,
        )
        if not _emit_macos_audit_sentinel(channel, "startup"):
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
    except (OSError, SandboxUnavailableError):
        if process is not None and process.poll() is None:
            if process_group is None:
                process_group = capture_process_group(process)
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(process_group)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # The collector is a parent-owned ``log stream``; on a
                    # SIGKILL-immune wedge the raise below already routes this
                    # spawn to sandbox-unavailable. Never block forever (I2).
                    pass
        handle.close()
        raise
    return channel


def _finish_macos_denial_audit(channel: _MacOSAuditChannel) -> None:
    """Stop a macOS collector and append the parent completion marker if trustworthy.

    Parameters
    ----------
    channel:
        Live parent-owned collector state.
    """

    completed = False
    drained = False
    try:
        # A denial emitted only after the worker exits proves that log delivery spans
        # the child's entire lifetime. Absence is telemetry-incomplete and receives no
        # terminal marker, so parsing poisons closed.
        drained = _emit_macos_audit_sentinel(channel, "post-exit")
        channel.process.terminate()
        return_code = channel.process.wait(timeout=5)
        completed = drained and return_code in {0, -signal.SIGTERM}
    except (OSError, subprocess.TimeoutExpired):
        _kill_process_group(channel.process_group)
        try:
            channel.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            # ``completed`` stays False, so parsing poisons closed; a wedged
            # collector must never hang supervision (I2).
            pass
    finally:
        channel.handle.flush()
        durable_fsync(channel.handle.fileno())
        channel.handle.close()
    if not completed:
        return
    try:
        status = channel.path.stat()
        if (status.st_dev, status.st_ino) != channel.expected_identity:
            return
        with channel.path.open("ab") as handle:
            if status.st_size > 0:
                with channel.path.open("rb") as read_handle:
                    read_handle.seek(-1, os.SEEK_END)
                    if read_handle.read(1) != b"\n":
                        handle.write(b"\n")
            handle.write((_MACOS_AUDIT_COMPLETION_MARKER + "\n").encode("ascii"))
            handle.flush()
            durable_fsync(handle.fileno())
    except OSError:
        return


def _merge_denial_observations(
    *observations: SandboxDenialObservation,
) -> SandboxDenialObservation:
    """Merge OS denial channels without losing any observed target.

    Parameters
    ----------
    *observations:
        Denial observations for the same process tree.

    Returns
    -------
    SandboxDenialObservation
        Union of all flags and targets.
    """

    targets = tuple(
        dict.fromkeys(
            target for observation in observations for target in observation.socket_targets
        )
    )
    paths = tuple(
        dict.fromkeys(path for observation in observations for path in observation.write_paths)
    )
    checkpoint_paths = tuple(
        dict.fromkeys(path for observation in observations for path in observation.checkpoint_paths)
    )
    failed_read_probe_paths = tuple(
        dict.fromkeys(
            path for observation in observations for path in observation.failed_read_probe_paths
        )
    )
    telemetry_failures = tuple(
        dict.fromkeys(
            observation.telemetry_failure
            for observation in observations
            if observation.telemetry_failure is not None
        )
    )
    return SandboxDenialObservation(
        network_attempted=any(observation.network_attempted for observation in observations),
        socket_targets=targets,
        write_outside_scratch_attempted=any(
            observation.write_outside_scratch_attempted for observation in observations
        ),
        write_paths=paths,
        checkpoint_or_weight_read_attempted=any(
            observation.checkpoint_or_weight_read_attempted for observation in observations
        ),
        checkpoint_paths=checkpoint_paths,
        failed_read_probe_paths=failed_read_probe_paths,
        telemetry_failure=";".join(telemetry_failures) if telemetry_failures else None,
    )


def _atomic_rewrite_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically replace one supervisor-poisoned self-hashed receipt.

    Parameters
    ----------
    path:
        Existing worker receipt path.
    receipt:
        Receipt payload including its recomputed self hash.
    """

    temporary = path.with_name(f".{path.name}.{os.getpid()}.supervisor.tmp")
    data = canonical_json_bytes(receipt) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            durable_fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _sandbox_denial_reason_code(denial: SandboxDenialObservation) -> str:
    """Return the dominant closed reason for one parent-observed denial.

    Parameters
    ----------
    denial:
        Parsed and integrity-checked OS sandbox observation.

    Returns
    -------
    str
        ``network-attempt``, ``write-outside-scratch``, ``telemetry-integrity``,
        or ``checkpoint-read``.
    """

    if denial.network_attempted:
        return "network-attempt"
    if denial.write_outside_scratch_attempted:
        return "write-outside-scratch"
    if denial.checkpoint_or_weight_read_attempted:
        return "checkpoint-read"
    if denial.telemetry_failure is not None:
        return "telemetry-integrity"
    return "checkpoint-read"


def _write_parent_policy_failure_receipt(
    receipt_path: Path,
    request: Mapping[str, Any],
    denial: SandboxDenialObservation,
) -> None:
    """Write a non-awarding parent failure receipt for a force-killed macOS worker.

    Seatbelt terminates the worker before model code can catch an undeclared read, so no
    child receipt can exist. The parent already owns the audited denial and request
    association; this envelope carries only those failure facts and can never contain a raw
    award receipt.

    Parameters
    ----------
    receipt_path:
        Exact request-bound result path.
    request:
        Parent-authored worker request already verified against the execution manifest.
    denial:
        Completed parent-owned Seatbelt observation.
    """

    meaningful_modes_value = request.get("meaningful_modes")
    meaningful_modes = (
        [str(value) for value in meaningful_modes_value]
        if isinstance(meaningful_modes_value, list)
        else []
    )
    policy = {
        "network_attempted": denial.network_attempted,
        "socket_targets": list(denial.socket_targets),
        "checkpoint_or_weight_read_attempted": denial.checkpoint_or_weight_read_attempted,
        "checkpoint_paths": list(denial.checkpoint_paths),
        "write_outside_scratch_attempted": denial.write_outside_scratch_attempted,
        "write_paths": list(denial.write_paths),
        "credentials_present": False,
        "torchlens_import_attempted": False,
        "cache_read_attempted": False,
        "telemetry_failure": denial.telemetry_failure,
    }
    reason_code = _sandbox_denial_reason_code(denial)
    error = {
        "reason_code": reason_code,
        "exception_type": "menagerie.crawler.worker_supervisor.SandboxDenialObservation",
        "message": "Seatbelt terminated the worker after an audited policy denial",
        "traceback": None,
    }
    diagnostic = {
        "receipt_version": _WORKER_DIAGNOSTIC_VERSION,
        "stable_id": request.get("stable_id"),
        "source_identity": request.get("source_identity"),
        "recipe_revision": request.get("recipe_revision"),
        "observed_recipe_revision": None,
        "observed_adapter_sha256": None,
        "observed_code_manifest_sha256": None,
        "observed_input_asset_sha256": None,
        "execution_identity": request.get("execution_identity"),
        "seed": request.get("seed"),
        "input_seed": request.get("input_seed"),
        "mode": request.get("mode"),
        "device": request.get("device"),
        "framework": request.get("framework"),
        "awards_runs": False,
        "constructor_started": False,
        "constructor_completed": False,
        "input_completed": False,
        "per_mode": {},
        "declared_meaningful_modes": meaningful_modes,
        "detected_meaningful_modes": [],
        "meaningful_modes": meaningful_modes,
        "train_eval_divergence": None,
        "divergence_evidence": None,
        "policy_observation": policy,
        "error": error,
    }
    record_payload = {
        "result_version": _WORKER_RESULT_VERSION,
        "raw_award_receipt": None,
        "raw_award_receipt_sha256": None,
        "diagnostic": diagnostic,
    }
    _atomic_rewrite_receipt(
        receipt_path,
        {**record_payload, "result_sha256": stable_hash(record_payload)},
    )


def _poison_receipt_for_policy_failure(
    receipt_path: Path,
    reason_code: str,
    *,
    denial: Optional[SandboxDenialObservation] = None,
) -> bool:
    """Atomically poison one valid receipt for a parent-observed policy failure.

    Parameters
    ----------
    receipt_path:
        Atomic worker receipt to audit and, when necessary, replace.
    reason_code:
        Closed policy reason selected from observed flags.
    denial:
        Optional OS-boundary details merged into the worker policy observation.

    Returns
    -------
    bool
        True only when a valid worker receipt was poisoned.
    """

    if not receipt_path.is_file():
        return False
    try:
        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(loaded, dict):
        return False
    verified, load_error = _load_worker_result_value(loaded)
    if verified is None or load_error is not None:
        return False
    diagnostic = verified.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        return False
    payload = dict(diagnostic)
    policy_value = payload.get("policy_observation")
    if not isinstance(policy_value, Mapping):
        return False
    policy = dict(policy_value)
    if denial is not None:
        policy["network_attempted"] = bool(policy.get("network_attempted")) or (
            denial.network_attempted
        )
        policy["socket_targets"] = list(
            dict.fromkeys([*policy.get("socket_targets", []), *denial.socket_targets])
        )
        policy["write_outside_scratch_attempted"] = (
            bool(policy.get("write_outside_scratch_attempted"))
            or denial.write_outside_scratch_attempted
        )
        policy["write_paths"] = list(
            dict.fromkeys([*policy.get("write_paths", []), *denial.write_paths])
        )
        policy["checkpoint_or_weight_read_attempted"] = (
            bool(policy.get("checkpoint_or_weight_read_attempted"))
            or denial.checkpoint_or_weight_read_attempted
        )
        policy["checkpoint_paths"] = list(
            dict.fromkeys([*policy.get("checkpoint_paths", []), *denial.checkpoint_paths])
        )
        if denial.telemetry_failure is not None:
            policy["telemetry_failure"] = denial.telemetry_failure
    payload["policy_observation"] = policy
    error = {
        "reason_code": reason_code,
        "exception_type": ("menagerie.crawler.worker_supervisor.SandboxDenialObservation"),
        "message": (
            "parent-owned syscall telemetry observed a forbidden operation or failed integrity"
        ),
        "traceback": None,
    }
    payload["error"] = error
    per_mode_value = payload.get("per_mode")
    if isinstance(per_mode_value, Mapping):
        per_mode: dict[str, Any] = {}
        for mode, mode_value in per_mode_value.items():
            if isinstance(mode_value, Mapping):
                poisoned_mode = dict(mode_value)
                poisoned_mode["error"] = error
                per_mode[str(mode)] = poisoned_mode
            else:
                per_mode[str(mode)] = mode_value
        payload["per_mode"] = per_mode
    record_payload = {
        "result_version": _WORKER_RESULT_VERSION,
        "raw_award_receipt": None,
        "raw_award_receipt_sha256": None,
        "diagnostic": payload,
    }
    record = {**record_payload, "result_sha256": stable_hash(record_payload)}
    _atomic_rewrite_receipt(receipt_path, record)
    return True


def poison_receipt_for_sandbox_denial(receipt_path: Path, denial: SandboxDenialObservation) -> bool:
    """Poison an otherwise successful worker receipt after an OS denial.

    Parameters
    ----------
    receipt_path:
        Atomic worker receipt to audit and, when necessary, replace.
    denial:
        Parent-observed OS denial telemetry.

    Returns
    -------
    bool
        True only when a valid worker receipt was poisoned.
    """

    if not denial.poisoned:
        return False
    reason_code = _sandbox_denial_reason_code(denial)
    return _poison_receipt_for_policy_failure(receipt_path, reason_code, denial=denial)


def _caught_policy_reason(receipt: Mapping[str, Any]) -> Optional[str]:
    """Return the dominant dirty policy reason from one verified worker receipt.

    Parameters
    ----------
    receipt:
        Self-hash-verified child receipt.

    Returns
    -------
    str | None
        Closed policy reason, or ``None`` for a clean observation.
    """

    diagnostic = receipt.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        return None
    policy = diagnostic.get("policy_observation")
    if not isinstance(policy, Mapping):
        return None
    reasons = (
        ("network_attempted", "network-attempt"),
        ("checkpoint_or_weight_read_attempted", "checkpoint-read"),
        ("cache_read_attempted", "checkpoint-read"),
        ("write_outside_scratch_attempted", "write-outside-scratch"),
        ("credentials_present", "credentials-exposed"),
        ("torchlens_import_attempted", "torchlens-import"),
        # Host telemetry loss stays poisoning but carries its own honest
        # reason, never ``checkpoint-read`` (BL-12(e)).
        ("telemetry_failure", "telemetry-integrity"),
    )
    for field_name, reason_code in reasons:
        if policy.get(field_name):
            return reason_code
    return None


def _poison_receipts_in_roots(
    write_roots: Sequence[Path], denial: SandboxDenialObservation
) -> None:
    """Poison worker receipts found directly in explicit result roots.

    Parameters
    ----------
    write_roots:
        Explicit result directories granted to the child.
    denial:
        Parent-observed OS denial telemetry.
    """

    if not denial.poisoned:
        return
    for root in write_roots:
        try:
            candidates = tuple(root.glob("*.json"))
        except OSError:
            continue
        for candidate in candidates:
            poison_receipt_for_sandbox_denial(candidate, denial)


def run_isolated_subprocess(
    argv: Sequence[str],
    scratch_root: Path,
    *,
    timeout_seconds: float = DEFAULT_FORWARD_TIMEOUT_SECONDS,
    rss_limit_bytes: int = 12 * 1024**3,
    cwd: Optional[Path] = None,
    base_environment: Optional[Mapping[str, str]] = None,
    additional_write_roots: Sequence[Path] = (),
    worker_completion_challenge: Optional[str] = None,
    execution_read_manifest: Optional[ExecutionReadManifestV2 | ExecutionReadManifestV3] = None,
    shutdown_event: Optional[threading.Event] = None,
    worker_lease_handle: Optional[WorkerLeaseHandle] = None,
    request_nonce: Optional[str] = None,
    request_sha256: Optional[str] = None,
    on_lease_started: Optional[Callable[[WorkerLease], None]] = None,
    verification_token: Optional[EnvironmentVerificationToken] = None,
    policy_failure_receipt_path: Optional[Path] = None,
    policy_failure_request: Optional[Mapping[str, Any]] = None,
) -> SupervisorObservation:
    """Launch a fresh credential-scrubbed subprocess inside an OS sandbox.

    Parameters
    ----------
    argv:
        Exact executable argument vector.
    scratch_root:
        Fresh writable cache/log root.
    timeout_seconds:
        Parent-enforced wall timeout.
    rss_limit_bytes:
        Parent-observed RSS and child address-space cap.
    cwd:
        Read-only source working directory. Defaults to the current directory.
    base_environment:
        Optional environment filtered through the safe allowlist.
    additional_write_roots:
        Explicit result roots writable in addition to scratch.
    worker_completion_challenge:
        Fresh parent secret used only for the standard worker's normal-completion
        attestation. Generic isolated subprocesses leave this unset.
    execution_read_manifest:
        Frozen out-of-band read capability. Author-authored request paths never
        participate in sandbox grants.
    shutdown_event:
        Driver-owned signal event polled during supervision.
    worker_lease_handle:
        Child-inherited kernel lock and lifecycle descriptors for standard workers.
    request_nonce, request_sha256:
        Exact v3 request association required for the v2 completion protocol.
    on_lease_started:
        Single-writer callback invoked after trusted child bootstrap has fsynced
        PID/start-token/PGID and before model execution is polled.
    verification_token:
        Cache-created spawn proof shared by compiler, projection, renderer, and supervisor.
    policy_failure_receipt_path, policy_failure_request:
        Optional exact v3 result path and parent request used only to persist a non-awarding
        failure when forced Seatbelt termination precludes a child receipt.

    Returns
    -------
    SupervisorObservation
        Parent-only process and resource facts.

    Raises
    ------
    SandboxUnavailableError
        If no complete OS sandbox is working on this host.
    """

    if not argv or any(not isinstance(value, str) or "\x00" in value for value in argv):
        raise ValueError("argv must contain non-empty NUL-free strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if (request_nonce is None) != (request_sha256 is None):
        raise ValueError("request nonce and digest must be supplied together")
    if (policy_failure_receipt_path is None) != (policy_failure_request is None):
        raise ValueError("policy failure receipt path and request must be supplied together")
    if execution_read_manifest is not None:
        verify_execution_read_manifest(
            execution_read_manifest,
            verification_token=verification_token,
        )
    scratch_root.mkdir(parents=True, exist_ok=True)
    write_roots = (scratch_root.resolve(), *(path.resolve() for path in additional_write_roots))
    for root in write_roots:
        root.mkdir(parents=True, exist_ok=True)
    stdout_path = scratch_root / "stdout.log"
    stderr_path = scratch_root / "stderr.log"
    safe_environment = build_safe_environment(scratch_root, base_environment=base_environment)
    if isinstance(execution_read_manifest, ExecutionReadManifestV3):
        openssl_config = (
            execution_read_manifest.environment_authority.prefix / "ssl" / "openssl.cnf"
        )
        if openssl_config.is_file():
            safe_environment["OPENSSL_CONF"] = str(openssl_config)
    if worker_completion_challenge is not None:
        safe_environment[_PARENT_COMPLETION_CHALLENGE_ENV] = worker_completion_challenge
    if request_sha256 is not None:
        safe_environment[_REQUEST_SHA256_ENV] = request_sha256
    if execution_read_manifest is not None:
        safe_environment[_READ_MANIFEST_ID_ENV] = execution_read_manifest.manifest_id
        if execution_read_manifest.standard_input_asset is not None:
            safe_environment[_PARENT_STANDARD_INPUT_ASSET_ENV] = str(
                execution_read_manifest.standard_input_asset[0]
            )
    inherited_fds: tuple[int, ...] = ()
    if worker_lease_handle is not None:
        if worker_lease_handle.lock_fd is None or worker_lease_handle.lifecycle_read_fd is None:
            raise ValueError("worker lease descriptors are already closed")
        inherited_fds = tuple(
            descriptor
            for descriptor in (
                worker_lease_handle.lock_fd,
                worker_lease_handle.lifecycle_read_fd,
                worker_lease_handle.global_lock_fd,
            )
            if descriptor is not None
        )
        safe_environment[_WORKER_LOCK_FD_ENV] = str(worker_lease_handle.lock_fd)
        safe_environment[_LIFECYCLE_FD_ENV] = str(worker_lease_handle.lifecycle_read_fd)
        if worker_lease_handle.global_lock_fd is not None:
            safe_environment[_GLOBAL_LOCK_FD_ENV] = str(worker_lease_handle.global_lock_fd)
    working_directory = (cwd or Path.cwd()).resolve()
    allowed_read_paths = _request_allowed_read_paths(
        argv,
        execution_read_manifest,
        verification_token=verification_token,
    )
    safe_environment[_PARENT_ALLOWED_READ_PATHS_ENV] = json.dumps(
        [str(path) for path in allowed_read_paths],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    sandbox = detect_os_sandbox()
    if sandbox is None:
        raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
    profile_path: Optional[Path] = None
    linux_runtime_code_roots: tuple[Path, ...] = ()
    linux_host_transport_capability: Optional[HostTransportLibraryCapability] = None
    macos_runtime_read_roots: tuple[Path, ...] = ()
    if sandbox.kind == "sandbox-exec":
        discovered_roots = _runtime_read_roots(argv, working_directory)
        macos_runtime_read_roots, runtime_package_data_paths = _macos_runtime_read_capabilities(
            execution_read_manifest, discovered_roots
        )
        profile_path = scratch_root / "worker-sandbox.sb"
        profile_path.write_text(
            generate_macos_sandbox_profile(
                write_roots,
                allowed_read_paths=(*allowed_read_paths, *runtime_package_data_paths),
                runtime_read_roots=macos_runtime_read_roots,
                execution_read_manifest=execution_read_manifest,
                verification_token=verification_token,
            ),
            encoding="utf-8",
        )
    elif sandbox.kind == "bubblewrap":
        discovered_roots = _linux_runtime_code_roots(argv, working_directory)
        if execution_read_manifest is None:
            linux_runtime_code_roots = discovered_roots
        elif isinstance(execution_read_manifest, ExecutionReadManifestV3):
            linux_runtime_code_roots = (execution_read_manifest.environment_authority.prefix,)
        elif isinstance(execution_read_manifest, ExecutionReadManifestV2):
            linux_runtime_code_roots = ()
        linux_host_transport_capability = _linux_host_transport_library_capability(Path(argv[0]))
    sandboxed_argv = wrap_with_os_sandbox(
        sandbox,
        argv,
        working_directory,
        write_roots,
        macos_profile_path=profile_path,
        allowed_read_paths=allowed_read_paths,
        host_transport_capability=linux_host_transport_capability,
    )
    denial_audit_path: Optional[Path] = None
    denial_audit_identity: Optional[tuple[int, int]] = None
    macos_audit_channel: Optional[_MacOSAuditChannel] = None
    success_attestation_path: Optional[Path] = None
    success_attestation_identity: Optional[tuple[int, int]] = None
    if worker_completion_challenge is not None and request_nonce is None:
        success_attestation_path, success_attestation_identity = _parent_owned_audit_path(
            scratch_root,
            write_roots,
            filename="worker-success-attestation.json",
        )
    if sandbox.kind == "bubblewrap":
        denial_audit_executable = shutil.which("strace")
        if denial_audit_executable is None:
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
        denial_audit_path, denial_audit_identity = _parent_owned_audit_path(
            scratch_root, write_roots
        )
        sandboxed_argv = _linux_audited_argv(
            sandboxed_argv,
            denial_audit_executable,
            denial_audit_path,
        )
    elif sandbox.kind == "sandbox-exec":
        if profile_path is None:
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
        macos_audit_channel = _start_macos_denial_audit(
            scratch_root,
            write_roots,
            sandbox_executable=sandbox.executable,
            profile_path=profile_path,
            worker_executable=argv[0],
        )
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    started_at = utc_now()
    timed_out = False
    rss_exceeded = False
    shutdown_requested = False
    peak_rss = 0
    peak_rss_basis = "root-process"
    group_teardown_outcome: Optional[str] = None
    group_survivor_pids: tuple[int, ...] = ()
    darwin_libomp_blocker_path: Optional[Path] = None
    darwin_libomp_blocker_identity: Optional[tuple[int, int]] = None
    darwin_libomp_blocker_cleanup_failed = False
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            if isinstance(execution_read_manifest, ExecutionReadManifestV3):
                if verification_token is None:
                    raise ValueError("v3 worker spawn lacks a cache verification token")
                execution_read_manifest.environment_authority._cache.mark_spawned(
                    verification_token
                )
            process = subprocess.Popen(
                list(sandboxed_argv),
                cwd=working_directory,
                env=safe_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
                start_new_session=True,
                close_fds=True,
                pass_fds=inherited_fds,
                preexec_fn=partial(
                    _child_limit,
                    rss_limit_bytes,
                    (worker_lease_handle.record_path if worker_lease_handle is not None else None),
                    worker_lease_handle.lease if worker_lease_handle is not None else None,
                    sandbox.kind == "sandbox-exec",
                ),
            )
            # Bind the worker's group identity before anything else can run, while the
            # PID provably still belongs to this child. Every later force teardown
            # signals this bound identity instead of re-reading ``process.pid``.
            process_group = capture_process_group(process)
            if sandbox.kind == "sandbox-exec":
                darwin_libomp_blocker_path = _darwin_libomp_registration_blocker(process.pid)
                try:
                    darwin_libomp_blocker_identity = _verify_darwin_libomp_registration_blocker(
                        darwin_libomp_blocker_path
                    )
                except SandboxUnavailableError:
                    _kill_process_group(process_group)
                    _reap_worker_root(process, process_group)
                    raise
            if worker_lease_handle is not None:
                if worker_lease_handle.lock_fd is not None:
                    os.close(worker_lease_handle.lock_fd)
                    worker_lease_handle.lock_fd = None
                if worker_lease_handle.global_lock_fd is not None:
                    os.close(worker_lease_handle.global_lock_fd)
                    worker_lease_handle.global_lock_fd = None
                if worker_lease_handle.lifecycle_read_fd is not None:
                    os.close(worker_lease_handle.lifecycle_read_fd)
                    worker_lease_handle.lifecycle_read_fd = None
                try:
                    worker_lease_handle.lease = _read_worker_lease(worker_lease_handle.record_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    _kill_process_group(process_group)
                else:
                    if on_lease_started is not None:
                        try:
                            on_lease_started(worker_lease_handle.lease)
                        except Exception:
                            _kill_process_group(process_group)
            if macos_audit_channel is not None:
                macos_audit_channel.worker_pid = process.pid
            poll_iteration = 0
            # DX-8/RE-8: the kill fires at the soft watermark BELOW the hard
            # cap, so the sampler wins the race against a fast allocator (near-
            # cap completions become deferrals -- the JMT-adopted B3-XQ-4
            # behavior change). Group-sweep pressure escalates above 60% of the
            # threshold.
            rss_kill_threshold = effective_rss_kill_threshold_bytes(rss_limit_bytes)
            rss_pressure = False
            while True:
                elapsed = time.monotonic() - started
                # Containment is over the WHOLE bound process group, not the
                # root PID: a periodic group sweep catches multi-process
                # models that split their footprint across children, and on
                # Linux it measures the tracees instead of strace itself
                # (BL-12(d)). The cheap root sample runs every iteration, and
                # above the pressure ratio the group sweep does too -- a child
                # spawned between strided sweeps is exactly the overshoot the
                # watermark exists to bound (DX-8).
                if rss_pressure or poll_iteration % _GROUP_RSS_SAMPLE_STRIDE == 0:
                    current_rss, sample_basis = _process_group_rss(
                        process_group.pgid, process.pid
                    )
                    if sample_basis == "process-group-sum":
                        peak_rss_basis = "process-group-sum"
                else:
                    current_rss = _child_rss(process.pid)
                poll_iteration += 1
                peak_rss = max(peak_rss, current_rss)
                if rss_kill_threshold > 0 and current_rss:
                    rss_pressure = (
                        current_rss > rss_kill_threshold * RSS_GROUP_SAMPLE_PRESSURE_RATIO
                    )
                if current_rss and rss_kill_threshold > 0 and current_rss > rss_kill_threshold:
                    rss_exceeded = True
                    _kill_process_group(process_group)
                    break
                if process.poll() is not None:
                    break
                if shutdown_event is not None and shutdown_event.is_set():
                    shutdown_requested = True
                    _kill_process_group(process_group)
                    break
                if elapsed >= timeout_seconds:
                    timed_out = True
                    _kill_process_group(process_group)
                    break
                time.sleep(0.01)
            _reap_worker_root(process, process_group)
            # No-survivors is part of the lease-close contract on EVERY path,
            # normal exit included (BL-12(c)): orphaned grandchildren inherit
            # both execution flocks and previously outlived supervision.
            if process_group.isolated:
                group_teardown_outcome, group_survivor_pids = _teardown_group_after_reap(
                    process_group.pgid
                )
            else:
                group_teardown_outcome, group_survivor_pids = ("foreign-group", ())
    finally:
        if worker_lease_handle is not None and worker_lease_handle.lifecycle_write_fd is not None:
            os.close(worker_lease_handle.lifecycle_write_fd)
            worker_lease_handle.lifecycle_write_fd = None
        if macos_audit_channel is not None:
            _finish_macos_denial_audit(macos_audit_channel)
        if (
            darwin_libomp_blocker_path is not None
            and darwin_libomp_blocker_identity is not None
            and not _remove_darwin_libomp_registration_blocker(
                darwin_libomp_blocker_path,
                darwin_libomp_blocker_identity,
            )
        ):
            darwin_libomp_blocker_cleanup_failed = True
    wall_seconds = time.monotonic() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = max(0.0, _rusage_seconds(usage_after) - _rusage_seconds(usage_before))
    peak_rss = max(peak_rss, _rusage_peak_rss_floor_bytes(usage_before, usage_after))
    return_code = process.returncode
    signal_number = -return_code if return_code is not None and return_code < 0 else None
    exit_code = return_code if return_code is not None and return_code >= 0 else None
    # Streams are hashed and tailed with constant memory; the parent never
    # materializes a whole model-owned stream again (BL-12(h)).
    stdout_sha256_value, stdout_byte_count = _hash_file_streaming(stdout_path)
    stderr_sha256_value, stderr_byte_count = _hash_file_streaming(stderr_path)
    stdout_window = _final_window_bytes(stdout_path)
    denial = SandboxDenialObservation()
    if macos_audit_channel is not None:
        denial = _parse_macos_denial_audit(
            macos_audit_channel.path,
            expected_identity=macos_audit_channel.expected_identity,
            expected_process_ids=(
                (macos_audit_channel.worker_pid,)
                if macos_audit_channel.worker_pid is not None
                else ()
            ),
            expected_process_roots=macos_runtime_read_roots,
            ignored_process_ids=macos_audit_channel.sentinel_process_ids,
        )
        if darwin_libomp_blocker_cleanup_failed:
            denial = _merge_denial_observations(
                denial,
                _telemetry_failure_observation("libomp-blocker-cleanup"),
            )
    elif denial_audit_path is not None:
        denial = _parse_linux_denial_audit(
            denial_audit_path,
            working_directory,
            write_roots,
            expected_identity=denial_audit_identity,
            allowed_read_paths=allowed_read_paths,
            runtime_code_roots=linux_runtime_code_roots,
            host_transport_capability=linux_host_transport_capability,
            standard_input_asset=(
                None
                if execution_read_manifest is None
                or execution_read_manifest.standard_input_asset is None
                else execution_read_manifest.standard_input_asset[0]
            ),
        )
    if (
        denial.poisoned
        and policy_failure_receipt_path is not None
        and policy_failure_request is not None
        and not policy_failure_receipt_path.exists()
    ):
        _write_parent_policy_failure_receipt(
            policy_failure_receipt_path,
            policy_failure_request,
            denial,
        )
    _poison_receipts_in_roots(additional_write_roots, denial)
    parent_observation = {
        "exit_code": exit_code,
        "signal": signal_number,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": peak_rss,
        "stdout_sha256": stdout_sha256_value,
        "stderr_sha256": stderr_sha256_value,
    }
    completion = (
        _verified_worker_completion(
            stdout_window,
            worker_completion_challenge,
            request_nonce=request_nonce,
            request_sha256=request_sha256,
        )
        if worker_completion_challenge is not None
        and exit_code == 0
        and signal_number is None
        and not timed_out
        and not rss_exceeded
        else None
    )
    success_attestation_sha256: Optional[str] = None
    attested_receipt_sha256: Optional[str] = None
    if (
        completion is not None
        and success_attestation_path is not None
        and success_attestation_identity is not None
    ):
        named_receipt_sha256 = completion.get(
            "raw_award_receipt_sha256", completion.get("receipt_sha256")
        )
        success_attestation_sha256 = parent_success_attestation_sha256(
            completion["completion_line"], parent_observation
        )
        attestation = {
            "version": "menagerie.crawler.parent-success-attestation.v1",
            "receipt_sha256": named_receipt_sha256,
            "completion_line": completion["completion_line"],
            "observation": parent_observation,
            "attestation_sha256": success_attestation_sha256,
        }
        try:
            status = success_attestation_path.stat()
            if (status.st_dev, status.st_ino) == success_attestation_identity:
                with success_attestation_path.open("r+b") as handle:
                    handle.seek(0)
                    handle.truncate()
                    handle.write(canonical_json_bytes(attestation) + b"\n")
                    handle.flush()
                    durable_fsync(handle.fileno())
                attested_receipt_sha256 = named_receipt_sha256
            else:
                success_attestation_sha256 = None
        except OSError:
            success_attestation_sha256 = None
    return SupervisorObservation(
        argv=sandboxed_argv,
        cwd=str(working_directory),
        exit_code=exit_code,
        signal_number=signal_number,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_rss_bytes=peak_rss,
        timed_out=timed_out,
        rss_exceeded=rss_exceeded,
        stdout_sha256=stdout_sha256_value,
        stdout_bytes=stdout_byte_count,
        stdout_tail=_tail_file(stdout_path),
        stderr_sha256=stderr_sha256_value,
        stderr_bytes=stderr_byte_count,
        stderr_tail=_tail_file(stderr_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        failed_read_probe_paths=denial.failed_read_probe_paths,
        success_attestation_sha256=success_attestation_sha256,
        attested_receipt_sha256=attested_receipt_sha256,
        started_at=started_at,
        finished_at=utc_now(),
        shutdown_requested=shutdown_requested,
        group_teardown_outcome=group_teardown_outcome,
        group_survivor_pids=group_survivor_pids,
        peak_rss_basis=peak_rss_basis,
        rss_kill_threshold_bytes=rss_kill_threshold,
        rss_overshoot_bytes=(
            max(0, peak_rss - rss_kill_threshold) if rss_exceeded else 0
        ),
    )


def _sandbox_unavailable_observation(
    argv: Sequence[str], scratch_root: Path, working_directory: Path
) -> SupervisorObservation:
    """Create honest parent facts for a worker refused before launch.

    Parameters
    ----------
    argv:
        Worker command that was not launched.
    scratch_root:
        Supervisor log root.
    working_directory:
        Requested worker working directory.

    Returns
    -------
    SupervisorObservation
        Zero-resource observation with a durable closed failure log.
    """

    scratch_root.mkdir(parents=True, exist_ok=True)
    stdout_path = scratch_root / "stdout.log"
    stderr_path = scratch_root / "stderr.log"
    stdout = b""
    status = f"failed:{FailureStage.SANDBOX_UNAVAILABLE.value}"
    stderr = f"{status}\n".encode("utf-8")
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    timestamp = utc_now()
    return SupervisorObservation(
        argv=tuple(argv),
        cwd=str(working_directory),
        exit_code=None,
        signal_number=None,
        wall_seconds=0.0,
        cpu_seconds=0.0,
        peak_rss_bytes=0,
        timed_out=False,
        rss_exceeded=False,
        stdout_sha256=hash_bytes(stdout),
        stdout_bytes=0,
        stdout_tail="",
        stderr_sha256=hash_bytes(stderr),
        stderr_bytes=len(stderr),
        stderr_tail=_tail(stderr),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        started_at=timestamp,
        finished_at=timestamp,
    )


def _load_receipt(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Load and verify a complete atomic worker receipt.

    Parameters
    ----------
    path:
        Expected final receipt path.

    Returns
    -------
    tuple[dict[str, Any] | None, str | None]
        Verified receipt or a parent-owned error.
    """

    if not path.exists():
        return None, "missing-receipt"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid-receipt:{type(exc).__name__}"
    if not isinstance(value, dict):
        return None, "invalid-receipt:not-an-object"
    return _load_worker_result_value(value)


def _worker_declared_failure_v3(
    observation: SupervisorObservation,
    receipt: Optional[Mapping[str, Any]],
    receipt_error: Optional[str],
    completion: Optional[Mapping[str, Any]],
) -> bool:
    """Return whether a v3 result is the worker's own designed failure shape.

    The v3 success protocol is claim-then-attest: ``worker.main`` mints the raw
    award receipt and the challenge-bound completion file ONLY when the
    requested forward completed cleanly, and exits ``1`` with neither while
    recording the failure in its receipt (``_raw_award_receipt`` is documented
    success-only). That shape is a worker-owned NEGATIVE disposition, not an
    attestation problem: minting ``missing-or-mismatched-v3-attestation`` for
    it publishes machinery weather, the scheduler keeps requeueing, and every
    deterministically failing model loops through quarantine forever (the
    m12646/m13258/m13279 loop, step-200 legs 31-36).

    The shape is a closed conjunction and everything else keeps the
    fail-closed attestation-mismatch path:

    * the receipt parsed cleanly and the worker minted NO success claim --
      no raw award receipt, no raw award digest, no verified completion file;
    * the parent observed the designed failure exit exactly -- ``exit_code``
      ``1`` with no signal, no timeout, and no RSS breach;
    * the receipt's diagnostic carries the worker's own recorded failure: a
      structured global error, or a structured per-mode error on a mode whose
      ``forward_completed`` stayed false.

    Passing grants nothing positive: the raw award receipt stays absent, so no
    award or attestation can derive from this lane -- it only lets the
    already-negative disposition reach the receipt-evidence classification
    arms (``forward/mode-run`` and friends) instead of a machinery reason.
    """

    if receipt_error is not None or not isinstance(receipt, Mapping):
        return False
    if completion is not None:
        return False
    if receipt.get("raw_award_receipt") is not None:
        return False
    if receipt.get("raw_award_receipt_sha256") is not None:
        return False
    if (
        observation.exit_code != 1
        or observation.signal_number is not None
        or observation.timed_out
        or observation.rss_exceeded
    ):
        return False
    diagnostic = receipt.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        return False
    per_mode = diagnostic.get("per_mode")
    if isinstance(diagnostic.get("error"), Mapping):
        # DX-12/RE-13: a GLOBAL error beside a per-mode receipt whose forward
        # COMPLETED is a contradiction, not a designed negative -- the worker
        # claims it failed while its own mode evidence claims work finished.
        # A contradictory receipt stays on the fail-closed machinery
        # protocol-park path instead of reaching the model-owned
        # receipt-evidence arms.
        if isinstance(per_mode, Mapping) and any(
            isinstance(mode_observation, Mapping)
            and mode_observation.get("forward_completed")
            for mode_observation in per_mode.values()
        ):
            return False
        return True
    if not isinstance(per_mode, Mapping):
        return False
    for mode_observation in per_mode.values():
        if (
            isinstance(mode_observation, Mapping)
            and isinstance(mode_observation.get("error"), Mapping)
            and not mode_observation.get("forward_completed")
        ):
            return True
    return False


def supervise_worker(
    request_path: Path,
    receipt_path: Path,
    scratch_root: Path,
    *,
    timeout_seconds: float = DEFAULT_FORWARD_TIMEOUT_SECONDS,
    rss_limit_bytes: int = 12 * 1024**3,
    cwd: Optional[Path] = None,
    execution_read_manifest: Optional[ExecutionReadManifestV2 | ExecutionReadManifestV3] = None,
    worker_lease_handle: Optional[WorkerLeaseHandle] = None,
    shutdown_event: Optional[threading.Event] = None,
    on_lease_started: Optional[Callable[[WorkerLease], None]] = None,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> SupervisedResult:
    """Launch the standard worker and attach only a verified atomic receipt.

    Parameters
    ----------
    request_path, receipt_path:
        Immutable request and expected atomic child result.
    scratch_root:
        Fresh logs/caches root.
    timeout_seconds, rss_limit_bytes:
        Parent resource caps.
    cwd:
        Source working directory.
    execution_read_manifest:
        Frozen out-of-band read capability for the v3 protocol.
    worker_lease_handle:
        Open child-inherited worker lease. Required by fully integrated v3 callers.
    shutdown_event:
        Driver-owned event polled during supervision.
    on_lease_started:
        Single-writer callback for the durable ``worker-lease-started`` event.
    verification_token:
        Optional cache-created spawn proof. Direct v3 callers receive one here when absent.

    Returns
    -------
    SupervisedResult
        Parent observation plus an optional verified child receipt.
    """

    if isinstance(execution_read_manifest, ExecutionReadManifestV3) and verification_token is None:
        authority = execution_read_manifest.environment_authority
        with authority._cache.spawn_verification(authority) as created_token:
            return supervise_worker(
                request_path,
                receipt_path,
                scratch_root,
                timeout_seconds=timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
                cwd=cwd,
                execution_read_manifest=execution_read_manifest,
                worker_lease_handle=worker_lease_handle,
                shutdown_event=shutdown_event,
                on_lease_started=on_lease_started,
                verification_token=created_token,
            )

    request_bytes = request_path.read_bytes()
    request_sha256 = hash_bytes(request_bytes)
    try:
        request_value = json.loads(request_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("worker request is not valid JSON") from exc
    if not isinstance(request_value, Mapping):
        raise ValueError("worker request must be an object")
    request_nonce: Optional[str] = None
    if execution_read_manifest is not None:
        verify_execution_read_manifest(
            execution_read_manifest,
            verification_token=verification_token,
        )
        if not isinstance(execution_read_manifest, ExecutionReadManifestV3):
            raise ValueError("live v3 model worker spawn requires execution-read-manifest.v3")
        if request_value.get("protocol_version") != "menagerie.crawler.worker-request.v3":
            raise ValueError("compiled execution manifests require worker-request.v3")
        if worker_lease_handle is None:
            raise ValueError("v3 model worker spawn requires an inherited live worker lease")
        request_nonce_value = request_value.get("request_nonce")
        if not isinstance(request_nonce_value, str) or not request_nonce_value:
            raise ValueError("v3 worker request requires a nonce")
        request_nonce = request_nonce_value
        expected_association = {
            "stable_id": execution_read_manifest.stable_id,
            "work_id": execution_read_manifest.work_id,
            "execution_identity": execution_read_manifest.execution_identity,
            "code_manifest_identity": execution_read_manifest.code_manifest_identity,
            "execution_read_manifest_identity": execution_read_manifest.manifest_id,
        }
        mismatches = [
            field
            for field, expected in expected_association.items()
            if request_value.get(field) != expected
        ]
        if mismatches:
            raise ValueError(
                "worker request differs from execution manifest: " + ",".join(mismatches)
            )
        input_contract = request_value.get("input_contract")
        if isinstance(input_contract, Mapping) and "code_path" in input_contract:
            raise ValueError("v3 execution forbids input_contract.code_path presence")
        if request_value.get("input_manifest") is not None:
            raise ValueError("v3 execution forbids legacy input_manifest grants")
        expected_asset = execution_read_manifest.standard_input_asset
        request_asset = request_value.get("standard_input_asset")
        expected_asset_value = (
            None
            if expected_asset is None
            else {"asset_id": expected_asset[2], "sha256": expected_asset[1]}
        )
        if request_asset != expected_asset_value:
            raise ValueError("worker request standard asset differs from execution manifest")
        if worker_lease_handle is not None:
            lease = worker_lease_handle.lease
            if (
                lease.nonce != request_nonce
                or lease.stable_id != execution_read_manifest.stable_id
                or lease.work_id != execution_read_manifest.work_id
                or lease.request_identity != request_sha256
                or lease.execution_identity != execution_read_manifest.execution_identity
                or lease.receipt_path.resolve() != receipt_path.resolve()
            ):
                raise ValueError("worker lease differs from the exact v3 request")
    elif worker_lease_handle is not None:
        raise ValueError("worker lease inheritance requires an execution read manifest")
    receipt_path.unlink(missing_ok=True)
    completion_file_path = receipt_path.parent / WORKER_COMPLETION_FILE_NAME
    completion_file_path.unlink(missing_ok=True)
    worker_executable = (
        str(execution_read_manifest.environment_authority.selected_interpreter)
        if isinstance(execution_read_manifest, ExecutionReadManifestV3)
        else sys.executable
    )
    argv = (
        worker_executable,
        "-B",
        "-m",
        "menagerie.crawler.worker",
        "--request",
        str(request_path),
        "--receipt",
        str(receipt_path),
    )
    working_directory = (cwd or Path.cwd()).resolve()
    completion_challenge = secrets.token_hex(32)
    try:
        observation = run_isolated_subprocess(
            argv,
            scratch_root,
            timeout_seconds=timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
            cwd=working_directory,
            additional_write_roots=(receipt_path.parent,),
            worker_completion_challenge=completion_challenge,
            execution_read_manifest=execution_read_manifest,
            shutdown_event=shutdown_event,
            worker_lease_handle=worker_lease_handle,
            request_nonce=request_nonce,
            request_sha256=request_sha256 if request_nonce is not None else None,
            on_lease_started=on_lease_started,
            verification_token=verification_token,
            policy_failure_receipt_path=(
                receipt_path if execution_read_manifest is not None else None
            ),
            policy_failure_request=(request_value if execution_read_manifest is not None else None),
        )
    except SandboxUnavailableError:
        observation = _sandbox_unavailable_observation(argv, scratch_root, working_directory)
        status = f"failed:{FailureStage.SANDBOX_UNAVAILABLE.value}"
        if execution_read_manifest is not None and request_nonce is not None:
            parent_attestation = build_parent_attestation(
                request_nonce=request_nonce,
                request_sha256=request_sha256,
                completion=None,
                observation=observation,
            )
            return SupervisedResult(
                observation=observation,
                worker_receipt=None,
                receipt_error=status,
                success_attestation_sha256=str(parent_attestation["attestation_sha256"]),
                parent_attestation=parent_attestation,
                unattested_partial={
                    "state": "unattested-partial",
                    "stage": "policy",
                    "reason_code": "sandbox-unavailable-v1",
                    "diagnostic_sha256": None,
                },
            )
        return SupervisedResult(observation, None, status)
    receipt, receipt_error = _load_receipt(receipt_path)
    caught_policy_reason = _caught_policy_reason(receipt) if receipt is not None else None
    if caught_policy_reason is not None:
        _poison_receipt_for_policy_failure(receipt_path, caught_policy_reason)
        receipt, receipt_error = _load_receipt(receipt_path)
    if execution_read_manifest is not None and request_nonce is not None:
        # v3 completion is valid ONLY through its dedicated atomic file, bound
        # to the parent challenge (BL-12(a)): stdout can neither destroy nor
        # forge the attestation of a successful run.
        completion = _verified_worker_completion_file(
            completion_file_path,
            completion_challenge,
            request_nonce=request_nonce,
            request_sha256=request_sha256,
        )
        raw_receipt_value = receipt.get("raw_award_receipt") if receipt is not None else None
        raw_digest_value = receipt.get("raw_award_receipt_sha256") if receipt is not None else None
        raw_receipt = dict(raw_receipt_value) if isinstance(raw_receipt_value, Mapping) else None
        raw_digest = raw_digest_value if isinstance(raw_digest_value, str) else None
        association_clean = bool(
            raw_receipt is not None
            and raw_digest == stable_hash(raw_receipt)
            and raw_receipt.get("request_nonce") == request_nonce
            and raw_receipt.get("request_sha256") == request_sha256
            and raw_receipt.get("stable_id") == execution_read_manifest.stable_id
            and raw_receipt.get("work_id") == execution_read_manifest.work_id
            and raw_receipt.get("execution_identity") == execution_read_manifest.execution_identity
            and raw_receipt.get("code_manifest_identity")
            == execution_read_manifest.code_manifest_identity
            and completion is not None
            and completion.get("raw_award_receipt_sha256") == raw_digest
            and completion.get("completion_line")
            == completion_line_for_raw_award_receipt(raw_receipt)
            # DX-11/RE-12: the group teardown gates the award. A surviving
            # grandchild, an unverifiable sweep, or a foreign PGID means the
            # run's containment cannot be attested -- association is unclean,
            # so the award is impossible and the host park lane owns it.
            and observation.group_teardown_outcome in _AWARDABLE_TEARDOWN_OUTCOMES
            and observation.group_survivor_pids == ()
        )
        success_parent_attestation: Optional[dict[str, Any]] = None
        if association_clean:
            started_at = observation.started_at
            finished_at = observation.finished_at
            if not isinstance(started_at, str) or not isinstance(finished_at, str):
                association_clean = False
            else:
                assert completion is not None
                assert raw_receipt is not None
                success_parent_attestation = derive_parent_attestation(
                    raw_receipt,
                    str(completion["completion_line"]),
                    observation.to_dict(),
                    started_at=started_at,
                    finished_at=finished_at,
                )
        worker_declared_failure = False
        if not association_clean:
            raw_receipt = None
            raw_digest = None
            # The worker's own designed failure shape (recorded failure, no
            # success claim, exit 1) is a model-owned negative disposition,
            # not an unclean attestation: leave receipt_error unset so the
            # driver's receipt-evidence arms classify it, and mint no
            # protocol-violation partial for it. Every other raw-receipt-less
            # shape stays on the fail-closed attestation-mismatch path.
            worker_declared_failure = _worker_declared_failure_v3(
                observation, receipt, receipt_error, completion
            )
            if receipt_error is None and not worker_declared_failure:
                receipt_error = "missing-or-mismatched-v3-attestation"
        partial = None
        if raw_receipt is None and not worker_declared_failure:
            diagnostic = receipt.get("diagnostic") if receipt is not None else None
            partial = {
                "state": "unattested-partial",
                "stage": "forward",
                "reason_code": "protocol-violation",
                "diagnostic_sha256": (
                    stable_hash(diagnostic) if isinstance(diagnostic, Mapping) else None
                ),
            }
        parent_attestation = success_parent_attestation or build_parent_attestation(
            request_nonce=request_nonce,
            request_sha256=request_sha256,
            completion=None,
            observation=observation,
        )
        return SupervisedResult(
            observation=observation,
            worker_receipt=receipt,
            receipt_error=receipt_error,
            success_attestation_sha256=(
                str(success_parent_attestation["attestation_sha256"])
                if success_parent_attestation is not None
                else None
            ),
            raw_award_receipt=raw_receipt,
            raw_award_receipt_sha256=raw_digest,
            parent_attestation=parent_attestation,
            unattested_partial=partial,
        )
    success_attestation = observation.success_attestation_sha256
    if receipt is None or observation.attested_receipt_sha256 != receipt.get("result_sha256"):
        success_attestation = None
    if observation.exit_code != 0 or observation.signal_number is not None:
        return SupervisedResult(
            observation,
            receipt,
            receipt_error or ("worker-exit-nonzero" if receipt is None else None),
            None,
        )
    if receipt is not None and success_attestation is None:
        return SupervisedResult(observation, receipt, "missing-parent-success-attestation", None)
    return SupervisedResult(observation, receipt, receipt_error, success_attestation)
