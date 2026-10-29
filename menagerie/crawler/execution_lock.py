"""Host-global execution serialization for crawler model workers.

BL-12(b) / AREA_WORKER opus W-4 / catalog C-19: the acquire used to be a
blocking, timeout-free, SIGTERM-immune (PEP 475), ownerless ``flock`` -- one
orphan from any campaign clone silently hung every later leg on the host. The
lock is now a bounded, observable, reconcilable wait:

* acquisition polls ``LOCK_NB`` under an explicit deadline and consults the
  driver's shutdown event every iteration, so an operator SIGTERM works;
* the holder writes an ownership record into the lock file while holding it,
  so a blocked acquirer (and a human with ``cat``) can name the owner;
* deadline expiry raises the typed :class:`ExecutionLockTimeout` carrying the
  holder record instead of blocking forever.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
from pathlib import Path
import pwd
import stat
import threading
import time
from typing import Any, Mapping, Optional

from menagerie.crawler.identity import durable_fsync

#: Default bounded acquire deadline. The lock is held for at most one supervised
#: forward (parent wall timeout <= 1800 s), so twice that bound distinguishes a
#: slow legitimate holder from a wedged or orphaned one.
DEFAULT_EXECUTION_LOCK_TIMEOUT_SECONDS = 3600.0

#: Poll cadence for the non-blocking acquire loop.
_EXECUTION_LOCK_POLL_SECONDS = 0.1

#: Version marker for the holder record written into the lock file.
_HOLDER_RECORD_VERSION = "menagerie.crawler.execution-flock-holder.v1"


class ExecutionLockTimeout(RuntimeError):
    """Bounded global-execution-lock acquisition ended without the lock.

    Parameters
    ----------
    detail:
        Human-readable diagnosis naming the deadline or the shutdown request.
    holder:
        Parsed holder record from the lock file, or ``None`` when the holder
        wrote none (a pre-record holder or a foreign process).
    reason:
        ``"deadline"`` or ``"shutdown"``.
    """

    def __init__(
        self,
        detail: str,
        *,
        holder: Optional[Mapping[str, Any]] = None,
        reason: str = "deadline",
    ) -> None:
        super().__init__(detail)
        self.holder = dict(holder) if holder is not None else None
        self.reason = reason


def global_execution_flock_path() -> Path:
    """Return the one execution lock path shared by every campaign clone.

    Returns
    -------
    pathlib.Path
        Operator-account-global lock path outside every campaign runtime root.

    The path is derived from the kernel user identity rather than ``HOME`` or a
    campaign environment variable. Separate launchd environments and clones
    therefore cannot accidentally select different locks.
    """

    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    return account_home / ".cache" / "menagerie" / "crawler" / "execution.flock"


def read_global_execution_flock_holder(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Read the last holder record written into the execution lock file.

    Parameters
    ----------
    path:
        Explicit test/operator path. The production default is shared across clones.

    Returns
    -------
    dict[str, Any] | None
        Parsed holder record, or ``None`` when absent or unparsable. The record
        names the LAST process that acquired with metadata; it is diagnostic
        provenance, never an authority.
    """

    lock_path = path or global_execution_flock_path()
    try:
        text = lock_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping) or value.get("version") != _HOLDER_RECORD_VERSION:
        return None
    return dict(value)


def _write_holder_record(descriptor: int, holder_metadata: Optional[Mapping[str, Any]]) -> None:
    """Write this process's ownership record while the lock is held.

    Parameters
    ----------
    descriptor:
        Open descriptor that owns the exclusive kernel flock.
    holder_metadata:
        Caller-supplied lease facts (campaign, work, lease identifiers).
    """

    record: dict[str, Any] = {
        "version": _HOLDER_RECORD_VERSION,
        "pid": os.getpid(),
        "opened_at_epoch": time.time(),
    }
    if holder_metadata is not None:
        for key, value in holder_metadata.items():
            record.setdefault(str(key), value)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, payload)
        durable_fsync(descriptor)
    except OSError:
        # The record is diagnostic provenance; failing to write it must never
        # fail an acquisition the kernel already granted.
        pass


def acquire_global_execution_flock(
    path: Optional[Path] = None,
    *,
    timeout_seconds: float = DEFAULT_EXECUTION_LOCK_TIMEOUT_SECONDS,
    poll_seconds: float = _EXECUTION_LOCK_POLL_SECONDS,
    shutdown_event: Optional[threading.Event] = None,
    holder_metadata: Optional[Mapping[str, Any]] = None,
) -> int:
    """Acquire the host-global execution flock under an explicit deadline.

    The returned descriptor must remain open for the full worker lifetime. Production
    callers also inherit it into the worker process, so an abrupt driver death cannot
    release the slot while its detached child is still running.

    Parameters
    ----------
    path:
        Explicit test/operator path. The production default is shared across clones.
    timeout_seconds:
        Bounded acquire deadline. Every wait has a bound (sprint invariant I2).
    poll_seconds:
        Non-blocking poll cadence.
    shutdown_event:
        Driver-owned shutdown signal consulted every poll iteration, so an
        operator SIGTERM interrupts the wait instead of being retried by PEP 475.
    holder_metadata:
        Lease facts written into the lock file while held, so a blocked
        acquirer can name the owner.

    Returns
    -------
    int
        Open descriptor owning the exclusive kernel flock.

    Raises
    ------
    ExecutionLockTimeout
        If the deadline expires or shutdown is requested while waiting. The
        error carries the holder record when one exists.
    OSError
        If the lock path cannot be created safely.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    lock_path = (path or global_execution_flock_path()).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    blocked_reported = False
    deadline = time.monotonic() + timeout_seconds
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise OSError(f"global execution flock is not a regular file: {lock_path}")
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    raise
            if not blocked_reported:
                blocked_reported = True
                holder = read_global_execution_flock_holder(lock_path)
                notice = (
                    "menagerie: global execution flock is busy; waiting up to "
                    f"{timeout_seconds:.0f}s (holder={holder!r})\n"
                ).encode("utf-8", errors="backslashreplace")
                try:
                    os.write(2, notice[:4096])
                except OSError:
                    pass
            if shutdown_event is not None and shutdown_event.is_set():
                raise ExecutionLockTimeout(
                    "shutdown requested while waiting for the global execution flock",
                    holder=read_global_execution_flock_holder(lock_path),
                    reason="shutdown",
                )
            if time.monotonic() >= deadline:
                holder = read_global_execution_flock_holder(lock_path)
                raise ExecutionLockTimeout(
                    "global execution flock not acquired within "
                    f"{timeout_seconds:.0f}s; execution-lock-held-by:{holder!r}",
                    holder=holder,
                    reason="deadline",
                )
            time.sleep(poll_seconds)
        _write_holder_record(descriptor, holder_metadata)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def release_global_execution_flock(descriptor: Optional[int]) -> None:
    """Release one host-global execution descriptor.

    Parameters
    ----------
    descriptor:
        Descriptor returned by :func:`acquire_global_execution_flock`, or ``None``.
    """

    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
