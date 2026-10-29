"""ONE-command "is it stuck?" probe with a closed verdict vocabulary (D-35).

200a answered "is it stuck?" with expert-hours: lsof on driver.lock, ps for
daemonized pids, hand UTC arithmetic, knowing which of two gates ledgers is
live, and knowing which watchers were themselves trustworthy. Watchers keyed
on process-name pattern greps false-read twice (a watcher matched its own shell as the
driver; a kill aimed at a phantom match while the real driver worked). This
probe replaces all of it:

* Liveness comes from the ``driver.lock`` kernel flock holder ONLY -- the one
  authority a dead process cannot keep and a stranger cannot fake -- with the
  holder's recorded ``pid`` + ``boot_id`` + ``start_token`` cross-checked so a
  reused PID can never impersonate the driver. NEVER a process-name pattern.
* Every age is computed UTC-vs-UTC here; no human subtracts timestamps.
* The verdict is a CLOSED vocabulary mirrored by the exit code, so watchers
  consume the code and never string-match rendered lines:

  ========================  ====
  verdict                   exit
  ========================  ====
  RUNNING-HEALTHY           0
  RUNNING-SUSPECT:<signal>  1
  WEDGED:<signal>           2
  DEAD                      3
  HOLDING:<reason>          4
  ========================  ====

* ``--self-test`` proves the probe against a simulated-dead and a
  simulated-held lock BEFORE a watcher arms on it: an instrument that has
  never seen a dead driver is untested at its only job.

The probe shares ONE snapshot implementation with the launch monitor
(``launch_monitor.build_snapshot``); the W18 per-stage funnel lands there and
deepens the WEDGED diagnosis without a second derivation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from menagerie.crawler.tools.launch_monitor import (
    MonitorEvidenceUnavailable,
    Snapshot,
    Thresholds,
    build_snapshot,
    render_snapshot,
    within_first_author_wall,
)
from menagerie.crawler.worker_supervisor import current_boot_id, process_start_token

#: Closed verdict vocabulary; the qualifier after ``:`` is detail, never a
#: new verdict. Watchers branch on the exit code.
VERDICTS = (
    "RUNNING-HEALTHY",
    "RUNNING-SUSPECT",
    "WEDGED",
    "DEAD",
    "HOLDING",
)

VERDICT_EXIT_CODES = {
    "RUNNING-HEALTHY": 0,
    "RUNNING-SUSPECT": 1,
    "WEDGED": 2,
    "DEAD": 3,
    "HOLDING": 4,
}

DRIVER_STATUS_MARKER_SCHEMA_VERSION = "menagerie.crawler.driver-status-marker.v1"


@dataclass(frozen=True)
class DriverLiveness:
    """The lock-holder liveness fact: held XOR dead, identity cross-checked."""

    held: bool
    owner: Optional[dict[str, Any]]
    identity_verified: Optional[bool]
    detail: str


def driver_lock_path(runtime_root: Path) -> Path:
    """Return the driver single-writer lock path below one runtime root."""

    return runtime_root / "locks" / "driver.lock"


def read_driver_liveness(lock_path: Path) -> DriverLiveness:
    """Adjudicate driver liveness from the kernel flock holder ONLY.

    A missing or unheld lock file is DEAD evidence; a held flock is alive
    evidence regardless of anything a process listing would claim. When the
    holder metadata records ``pid``/``boot_id``/``start_token``, they are
    cross-checked (PID-reuse-proof); a held lock whose recorded identity does
    not check out is disclosed as unverified, never upgraded to healthy.
    """

    try:
        handle = open(lock_path, "r", encoding="utf-8")
    except FileNotFoundError:
        return DriverLiveness(
            held=False, owner=None, identity_verified=None,
            detail=f"no driver lock at {lock_path}",
        )
    except OSError as exc:
        return DriverLiveness(
            held=False, owner=None, identity_verified=None,
            detail=f"driver lock unreadable: {exc}",
        )
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            owner = _read_owner(handle)
            verified, detail = _verify_owner_identity(owner)
            return DriverLiveness(
                held=True, owner=owner, identity_verified=verified, detail=detail
            )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return DriverLiveness(
            held=False, owner=_read_owner(handle), identity_verified=None,
            detail="driver lock exists but is UNHELD (no live driver process)",
        )


def _read_owner(handle) -> Optional[dict[str, Any]]:
    try:
        handle.seek(0)
        value = json.loads(handle.readline() or "null")
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _verify_owner_identity(owner: Optional[dict[str, Any]]) -> tuple[Optional[bool], str]:
    if owner is None:
        return None, "lock is HELD; holder metadata unreadable"
    pid = owner.get("pid")
    if not isinstance(pid, int):
        return None, "lock is HELD; holder records no pid"
    checks: list[str] = []
    boot = owner.get("boot_id")
    if isinstance(boot, str) and boot:
        if boot != current_boot_id():
            return False, f"lock is HELD but holder boot_id {boot!r} is not this boot"
        checks.append("boot_id")
    token = owner.get("start_token")
    if isinstance(token, str) and token:
        observed = process_start_token(pid)
        if observed != token:
            return False, (
                f"lock is HELD but pid {pid} start_token mismatch"
                " (recorded holder is not the running process)"
            )
        checks.append("start_token")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, f"lock is HELD but recorded pid {pid} is not running"
    except PermissionError:
        pass
    checks.append("pid")
    return True, f"driver pid {pid} holds the lock ({'+'.join(checks)} verified)"


def driver_status_marker_path(runtime_root: Path) -> Path:
    """Return the driver-owned status marker path below one runtime root."""

    return runtime_root / "driver-status.json"


def read_status_marker(
    runtime_root: Path, *, max_age_seconds: float, now: Optional[datetime] = None
) -> tuple[str, Optional[dict[str, Any]]]:
    """Read the driver-owned status marker with staleness-as-death semantics.

    Returns ``(state, payload)`` with state in ``{"fresh", "stale", "absent"}``.
    A stale marker's claims are DEAD claims: they are disclosed but never used
    as liveness or arm evidence (the 200a supervisor heartbeat lied frozen for
    2.5 days while daemonized drivers bypassed it; a lying instrument is worse
    than a missing one).
    """

    path = driver_status_marker_path(runtime_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "absent", None
    except (OSError, ValueError):
        return "stale", None
    if not isinstance(payload, dict):
        return "stale", None
    written = payload.get("written_at")
    try:
        instant = datetime.fromisoformat(str(written).replace("Z", "+00:00"))
    except ValueError:
        return "stale", payload
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    age = ((now or datetime.now(timezone.utc)) - instant).total_seconds()
    return ("fresh" if age <= max_age_seconds else "stale"), payload


@dataclass(frozen=True)
class CrawlDoctorReport:
    """One probe result: closed verdict, exit code, and its evidence lines."""

    verdict: str
    qualifier: str
    exit_code: int
    liveness: DriverLiveness
    snapshot: Optional[Snapshot]
    evidence: tuple[str, ...]

    @property
    def verdict_line(self) -> str:
        return self.verdict if not self.qualifier else f"{self.verdict}:{self.qualifier}"


def evaluate(
    records_root: Path,
    runtime_root: Path,
    *,
    thresholds: Optional[Thresholds] = None,
    rung_label: Optional[str] = None,
    outbox_path: Optional[Path] = None,
    marker_max_age_seconds: float = 900.0,
    now: Optional[datetime] = None,
) -> CrawlDoctorReport:
    """Adjudicate the one-line verdict from lock liveness + the shared snapshot."""

    thresholds = thresholds or Thresholds()
    liveness = read_driver_liveness(driver_lock_path(runtime_root))
    evidence: list[str] = [f"driver: {liveness.detail}"]

    snapshot: Optional[Snapshot] = None
    snapshot_error: Optional[str] = None
    try:
        snapshot = build_snapshot(
            records_root,
            rung_label=rung_label,
            outbox_path=outbox_path,
            now=now,
        )
    except MonitorEvidenceUnavailable as exc:
        snapshot_error = str(exc)
        evidence.append(f"evidence: {snapshot_error}")

    marker_state, marker = read_status_marker(
        runtime_root, max_age_seconds=marker_max_age_seconds, now=now
    )
    if marker_state == "fresh" and marker is not None:
        evidence.append(
            "driver status marker: fresh"
            f" (leg {marker.get('leg_ordinal')}, arm {marker.get('arm')})"
        )
    else:
        evidence.append(f"driver status marker: {marker_state}")

    if snapshot is not None:
        evidence.extend(render_snapshot(snapshot).split("\n"))

    if not liveness.held:
        qualifier = ""
        if snapshot is not None and snapshot.complete:
            evidence.append("campaign reads COMPLETE; a dead driver is expected")
        return _report("DEAD", qualifier, liveness, snapshot, evidence)

    if snapshot is None:
        return _report(
            "RUNNING-SUSPECT", "monitor-evidence-unavailable", liveness, None, evidence
        )

    if snapshot.quota_paused:
        return _report("HOLDING", "quota-pause", liveness, snapshot, evidence)

    stalled = (
        not snapshot.complete
        and snapshot.in_flight > 0
        and snapshot.last_activity_age_minutes is not None
        and snapshot.last_activity_age_minutes >= thresholds.stall_minutes
        and not within_first_author_wall(snapshot, thresholds)
    )
    if stalled:
        return _report(
            "WEDGED",
            f"no-ledger-write-{snapshot.last_activity_age_minutes:.0f}m",
            liveness,
            snapshot,
            evidence,
        )

    suspects: list[str] = []
    if liveness.identity_verified is False:
        suspects.append("lock-holder-identity-mismatch")
    if snapshot.unrecognized:
        suspects.append(f"unrecognized-heads-{snapshot.unrecognized}")
    if snapshot.corrupt_lines:
        suspects.append(f"corrupt-ledger-lines-{snapshot.corrupt_lines}")
    if snapshot.roster_source != "registry":
        suspects.append("roster-degraded")
    if snapshot.notify_transport_disposition != "live":
        suspects.append(f"notify-transport-{snapshot.notify_transport_disposition}")
    if snapshot.notify_pending_critical:
        suspects.append(
            f"notify-pending-critical-{snapshot.notify_pending_critical}"
        )
    if marker_state == "stale":
        suspects.append("driver-status-marker-stale")
    if suspects:
        return _report(
            "RUNNING-SUSPECT", ",".join(suspects), liveness, snapshot, evidence
        )
    return _report("RUNNING-HEALTHY", "", liveness, snapshot, evidence)


def _report(
    verdict: str,
    qualifier: str,
    liveness: DriverLiveness,
    snapshot: Optional[Snapshot],
    evidence: list[str],
) -> CrawlDoctorReport:
    assert verdict in VERDICTS
    return CrawlDoctorReport(
        verdict=verdict,
        qualifier=qualifier,
        exit_code=VERDICT_EXIT_CODES[verdict],
        liveness=liveness,
        snapshot=snapshot,
        evidence=tuple(evidence),
    )


def self_test() -> int:
    """Arm-time self-test: the probe must name a dead and a held lock correctly.

    A watcher arms on this probe only after it has SEEN a simulated-dead
    fixture read DEAD and a simulated-held lock read alive -- the 200a
    watchers were never tested against a dead driver and false-read twice.
    """

    with tempfile.TemporaryDirectory() as raw:
        runtime_root = Path(raw)
        lock_path = driver_lock_path(runtime_root)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        lock_path.write_text(json.dumps({"pid": 999999999}) + "\n", encoding="utf-8")
        dead = read_driver_liveness(lock_path)
        if dead.held:
            print("crawl-doctor self-test FAILED: unheld lock read as held")
            return 1

        owner = {
            "pid": os.getpid(),
            "boot_id": current_boot_id(),
            "start_token": process_start_token(os.getpid()),
        }
        with open(lock_path, "w", encoding="utf-8") as holder:
            holder.write(json.dumps(owner) + "\n")
            holder.flush()
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
            held = read_driver_liveness(lock_path)
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        if not held.held or held.identity_verified is not True:
            print("crawl-doctor self-test FAILED: held lock not verified")
            return 1
    print("crawl-doctor self-test OK: dead reads DEAD, held reads alive")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.tools.crawl_doctor",
        description=__doc__.split("\n", 1)[0],
    )
    parser.add_argument("--records-root", type=Path, default=None)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="the campaign .crawl-local root holding locks/driver.lock",
    )
    parser.add_argument("--rung-label", default=None)
    parser.add_argument("--outbox", type=Path, default=None)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--marker-max-age-seconds", type=float, default=900.0)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="verify the probe against simulated dead/held locks and exit",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.records_root is None or args.runtime_root is None:
        print(
            "crawl-doctor: --records-root and --runtime-root are required",
            file=sys.stderr,
        )
        return 64
    report = evaluate(
        args.records_root,
        args.runtime_root,
        thresholds=Thresholds(stall_minutes=args.stall_minutes),
        rung_label=args.rung_label,
        outbox_path=args.outbox,
        marker_max_age_seconds=args.marker_max_age_seconds,
    )
    if args.json:
        payload = {
            "verdict": report.verdict,
            "qualifier": report.qualifier,
            "verdict_line": report.verdict_line,
            "exit_code": report.exit_code,
            "evidence": list(report.evidence),
            "snapshot": None if report.snapshot is None else asdict(report.snapshot),
        }
        print(json.dumps(payload, sort_keys=True))
    else:
        print(report.verdict_line)
        for line in report.evidence:
            print(f"  {line}")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
