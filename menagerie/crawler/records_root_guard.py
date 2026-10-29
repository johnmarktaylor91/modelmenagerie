"""Records-root write-safety guard (LC-22 / UU-9 interim refusals, WS7).

Every durability guarantee in the crawler's ledger spine rests on ONE
unstated assumption: ``fcntl.flock`` on the records root actually excludes
concurrent writers. On a network or interchange filesystem (NFS, SMB,
exFAT on a carried SSD) that assumption silently fails -- flock either
raises, no-ops, or holds only per-host -- and every torn-tail recovery,
append lock, and chain verification downstream is reasoning about bytes it
never actually owned. The step-500 multi-host charter designs the real
lease/fencing story; these are the two INTERIM refusals the plan makes
blocking NOW:

* :func:`assert_records_root_flock_safe` -- leg open refuses when the
  records root sits on a filesystem outside the closed local-flock
  allowlist, when the filesystem cannot be determined, or when an actual
  flock probe round-trip fails. A caller may pass a ``filesystem_capability``
  naming the EXACT observed filesystem to accept it deliberately -- the
  capability names what it allows, never "skip the check".
* :func:`assert_and_stamp_records_root_writer` -- the moved-root arm: a
  records root whose durable last-writer stamp names ANOTHER machine
  refuses until :func:`claim_records_root` is invoked deliberately, because
  a root carried between hosts may still have a live orchestrator on the
  other side and flock cannot see across hosts. Era-honest: a root with no
  stamp (every pre-guard root) is stamped and admitted.

The same-host two-orchestrator race is already excluded by the host-global
execution flock (:mod:`menagerie.crawler.execution_lock`); the
foreign-live-``leg_run`` arm needs per-invocation machine identity on leg
rows and lands with the Wave-2 identity batch (D-22/DX-23).
"""

from __future__ import annotations

import json
import fcntl
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from menagerie.crawler.identity import atomic_replace_bytes

#: Filesystems with reliable LOCAL ``fcntl.flock`` semantics. Closed on
#: purpose: an unknown filesystem refuses (with the observed name and the
#: capability spelling in the refusal), never passes by default. Network and
#: interchange filesystems are exactly the ones that must never appear here.
FLOCK_SAFE_FSTYPES: frozenset[str] = frozenset(
    {
        "apfs",
        "hfs",
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "zfs",
        "tmpfs",
        "overlay",
        "overlayfs",
    }
)

_WRITER_STAMP_NAME = ".records-root-writer.json"
_WRITER_STAMP_SCHEMA = "menagerie.crawler.records-root-writer.v1"

#: Per-process fstype cache keyed by device id: the resolution shells out to
#: the mount table and leg opens are frequent in tests.
_FSTYPE_CACHE: dict[int, Optional[str]] = {}


class RecordsRootGuardError(RuntimeError):
    """The records root is not safe to write from this process."""


def _mount_table() -> list[tuple[str, str]]:
    """Return (mountpoint, fstype) rows for this host, best-effort."""

    rows: list[tuple[str, str]] = []
    if sys.platform.startswith("linux"):
        try:
            payload = Path("/proc/mounts").read_text(encoding="utf-8")
        except OSError:
            return rows
        for line in payload.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                rows.append((parts[1], parts[2]))
        return rows
    if sys.platform == "darwin":
        try:
            output = subprocess.run(
                ["mount"], capture_output=True, text=True, check=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return rows
        for line in output.splitlines():
            # "<device> on <mountpoint> (<fstype>, <flags>...)"
            if " on " not in line or "(" not in line:
                continue
            _, _, rest = line.partition(" on ")
            mountpoint, _, detail = rest.rpartition(" (")
            fstype = detail.split(",", 1)[0].rstrip(")").strip()
            if mountpoint and fstype:
                rows.append((mountpoint, fstype))
        return rows
    return rows


def resolve_records_fstype(path: Path) -> Optional[str]:
    """Return the filesystem type holding ``path``, or ``None`` when unknown."""

    resolved = Path(path).resolve()
    try:
        device = os.stat(resolved).st_dev
    except OSError:
        device = -1
    if device in _FSTYPE_CACHE:
        return _FSTYPE_CACHE[device]
    best: Optional[tuple[int, str]] = None
    text = str(resolved)
    for mountpoint, fstype in _mount_table():
        if text == mountpoint or text.startswith(mountpoint.rstrip("/") + "/") or mountpoint == "/":
            candidate = (len(mountpoint), fstype)
            if best is None or candidate[0] > best[0]:
                best = candidate
    result = best[1].lower() if best is not None else None
    if device != -1:
        _FSTYPE_CACHE[device] = result
    return result


def _probe_flock(records_root: Path) -> None:
    """Round-trip one exclusive flock on the root, or raise the OS refusal."""

    probe = records_root / ".flock-probe"
    with probe.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    try:
        probe.unlink()
    except OSError:
        pass  # a leftover empty probe file is inert


def assert_records_root_flock_safe(
    records_root: Path,
    *,
    filesystem_capability: Optional[str] = None,
) -> str:
    """Refuse a records root whose filesystem cannot honour local flock.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory (created if absent, exactly as
        the ledger writers would).
    filesystem_capability:
        Deliberate acceptance of ONE exact filesystem type outside the
        allowlist (or of an undeterminable one, spelled ``"unknown"``). The
        capability names what it allows; it can never blanket-disable the
        probe.

    Returns
    -------
    str
        The observed filesystem type (``"unknown"`` when undeterminable).

    Raises
    ------
    RecordsRootGuardError
        If the filesystem is outside the allowlist without a matching
        capability, or the flock probe round-trip fails.
    """

    records_root = Path(records_root)
    records_root.mkdir(parents=True, exist_ok=True)
    observed = resolve_records_fstype(records_root) or "unknown"
    if observed not in FLOCK_SAFE_FSTYPES and filesystem_capability != observed:
        raise RecordsRootGuardError(
            f"records root {records_root} sits on filesystem {observed!r}, which is "
            "not in the local-flock allowlist: every ledger guarantee assumes "
            "flock actually excludes writers, and network/interchange filesystems "
            "break that silently. Move the root to a local volume, or accept THIS "
            f"exact filesystem deliberately with filesystem_capability={observed!r}."
        )
    try:
        _probe_flock(records_root)
    except OSError as exc:
        raise RecordsRootGuardError(
            f"records root {records_root} (filesystem {observed!r}) refused an "
            f"exclusive flock probe: {exc}. A root whose locks do not hold cannot "
            "be written safely from any process."
        ) from exc
    return observed


def _writer_stamp_path(records_root: Path) -> Path:
    return Path(records_root) / _WRITER_STAMP_NAME


def last_writer_stamp(records_root: Path) -> Optional[dict[str, Any]]:
    """Return the durable last-writer stamp, or ``None`` for a pre-guard root."""

    path = _writer_stamp_path(records_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    except ValueError as exc:
        raise RecordsRootGuardError(
            f"records-root writer stamp {path} is unreadable ({exc}): a corrupt "
            "ownership stamp must be adjudicated by a human, never guessed over"
        ) from exc
    if not isinstance(payload, dict):
        raise RecordsRootGuardError(
            f"records-root writer stamp {path} is not an object"
        )
    return payload


def _write_stamp(records_root: Path, machine_id: str, *, claim: Optional[str]) -> dict[str, Any]:
    stamp: dict[str, Any] = {
        "schema_version": _WRITER_STAMP_SCHEMA,
        "machine_id": machine_id,
    }
    if claim is not None:
        stamp["claimed_reason"] = claim
    atomic_replace_bytes(
        _writer_stamp_path(records_root),
        json.dumps(stamp, sort_keys=True).encode("utf-8") + b"\n",
    )
    return stamp


def assert_and_stamp_records_root_writer(
    records_root: Path,
    *,
    machine_id: Optional[str] = None,
) -> dict[str, Any]:
    """Refuse a root last written by ANOTHER machine; stamp this one.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    machine_id:
        This writer's host identity; defaults to ``platform.node()``.

    Returns
    -------
    dict[str, Any]
        The stamp now in force.

    Raises
    ------
    RecordsRootGuardError
        If the durable stamp names a different machine. flock cannot see
        across hosts, so a root moved (or shared) between machines may still
        have a live writer on the other side; only a deliberate
        :func:`claim_records_root` transfers ownership.
    """

    this_machine = machine_id or platform.node() or "unknown-host"
    stamp = last_writer_stamp(records_root)
    if stamp is not None and str(stamp.get("machine_id")) != this_machine:
        raise RecordsRootGuardError(
            f"records root {records_root} was last written by machine "
            f"{stamp.get('machine_id')!r}, not {this_machine!r}. A moved or shared "
            "root may still have a live orchestrator on the other host and flock "
            "cannot see across hosts. If the other writer is KNOWN dead, claim the "
            "root deliberately: menagerie.crawler.records_root_guard."
            "claim_records_root(records_root, reason=...)"
        )
    if stamp is None or str(stamp.get("machine_id")) != this_machine:
        return _write_stamp(records_root, this_machine, claim=None)
    return stamp


def claim_records_root(
    records_root: Path,
    *,
    reason: str,
    machine_id: Optional[str] = None,
) -> dict[str, Any]:
    """Deliberately transfer a records root's writer stamp to this machine.

    The capability arm of the foreign-writer refusal: an explicit, reasoned
    operator act, never a default. The reason is stored in the stamp so the
    transfer is legible after the fact.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    reason:
        Why the previous writer is known not to be live.
    machine_id:
        Claiming host identity; defaults to ``platform.node()``.

    Returns
    -------
    dict[str, Any]
        The stamp now in force.
    """

    if not reason or not reason.strip():
        raise RecordsRootGuardError(
            "claiming a records root requires a real reason: the claim is the "
            "audit trail for overriding the foreign-writer refusal"
        )
    this_machine = machine_id or platform.node() or "unknown-host"
    return _write_stamp(records_root, this_machine, claim=reason.strip())


__all__ = [
    "FLOCK_SAFE_FSTYPES",
    "RecordsRootGuardError",
    "assert_and_stamp_records_root_writer",
    "assert_records_root_flock_safe",
    "claim_records_root",
    "last_writer_stamp",
    "resolve_records_fstype",
]
