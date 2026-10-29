"""D-58: the dead-man watchdog pair with an expiry-bound HOLD sentinel.

The 200a campaign went dark for 5h46m INSIDE an expiry-less operator hold: the
hold suppressed every alarm, nothing owned re-arming them, and silence read as
health. The pair closes both halves:

* ON-HOST: a boot-surviving LaunchAgent runs ``--check`` on an interval --
  monitor heartbeat fresh AND (driver alive XOR campaign complete) -- and
  pages on failure. It also writes a BEAT file into a synced share on every
  healthy check.
* OFF-HOST: a cron on another machine runs ``--check-beat`` against
  the synced beat file and pages when it goes stale -- so a host that dies
  entirely (power, kernel, disk) is detected by hardware it cannot take down.

HOLD sentinel: an operator hold is a JSON file with a REQUIRED ``expires_at``
and a REQUIRED ``default_action`` from a closed vocabulary. While live it
suppresses paging (the operator owns the window); once expired the default
executes -- ``page`` alarms immediately, ``release`` resumes normal checks.
An expiry-less or malformed sentinel is INVALID and pages at once: a hold
that cannot end is exactly the outage class this exists to kill.

Per-machine installation (LaunchAgent on the crawl host, cron on the off-host checker) is a
fleet operator act; ``--emit-launchd-plist`` / ``--emit-remote-cron`` print
the exact artifacts. The pre-200b supervised reboot drill must observe
detection within 15 minutes end to end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.tools.crawl_doctor import driver_lock_path, read_driver_liveness
from menagerie.crawler.tools.launch_monitor import check_heartbeat, default_state_path
from menagerie.local_config import local_value

HOLD_SENTINEL_SCHEMA_VERSION = "menagerie.crawler.deadman-hold.v1"

#: Closed default actions an expired hold may take.
HOLD_DEFAULT_ACTIONS = frozenset({"page", "release"})

#: Closed check verdicts, mirrored by exit codes.
CHECK_VERDICTS = {
    "healthy": 0,
    "holding": 0,
    "page": 1,
    "invalid-hold": 1,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class HoldDisposition:
    """One sentinel adjudication: state + the reason it reads that way."""

    state: str  # "none" | "active" | "expired-page" | "expired-release" | "invalid"
    detail: str


def read_hold_sentinel(
    path: Path, *, now: Optional[datetime] = None
) -> HoldDisposition:
    """Adjudicate the operator HOLD sentinel, fail-closed on malformation.

    A hold WITHOUT a parseable ``expires_at`` or with an off-vocabulary
    ``default_action`` is INVALID and pages immediately -- never a silent
    indefinite mute (the 5h46m outage ran inside exactly that).
    """

    now = now or _utc_now()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return HoldDisposition("none", "no hold sentinel")
    except (OSError, ValueError):
        return HoldDisposition("invalid", f"unreadable hold sentinel at {path}")
    if not isinstance(payload, dict):
        return HoldDisposition("invalid", "hold sentinel is not an object")
    expires = _parse(payload.get("expires_at"))
    action = payload.get("default_action")
    if expires is None:
        return HoldDisposition(
            "invalid", "hold sentinel has no parseable expires_at (expiry-less holds are forbidden)"
        )
    if action not in HOLD_DEFAULT_ACTIONS:
        return HoldDisposition(
            "invalid", f"hold default_action {action!r} outside {sorted(HOLD_DEFAULT_ACTIONS)}"
        )
    if now <= expires:
        reason = payload.get("reason", "")
        return HoldDisposition("active", f"hold live until {expires.isoformat()} ({reason})")
    if action == "page":
        return HoldDisposition(
            "expired-page", f"hold EXPIRED at {expires.isoformat()} with default_action=page"
        )
    return HoldDisposition(
        "expired-release", f"hold expired at {expires.isoformat()}; released"
    )


def evaluate_check(
    *,
    records_root: Path,
    runtime_root: Path,
    state_file: Path,
    hold_path: Path,
    heartbeat_max_age_seconds: float,
    now: Optional[datetime] = None,
) -> tuple[str, list[str]]:
    """One on-host dead-man check: closed verdict + evidence lines."""

    now = now or _utc_now()
    hold = read_hold_sentinel(hold_path, now=now)
    evidence = [f"hold: {hold.detail}"]
    if hold.state == "invalid":
        return "invalid-hold", evidence
    if hold.state == "active":
        return "holding", evidence
    if hold.state == "expired-page":
        return "page", evidence

    heartbeat_verdict, heartbeat_detail = check_heartbeat(
        state_file, max_age_seconds=heartbeat_max_age_seconds, now=now
    )
    evidence.append(f"monitor heartbeat: {heartbeat_verdict} -- {heartbeat_detail}")
    liveness = read_driver_liveness(driver_lock_path(runtime_root))
    evidence.append(f"driver: {liveness.detail}")

    complete = _campaign_complete(records_root)
    evidence.append(f"campaign complete: {complete}")
    if heartbeat_verdict != "fresh":
        return "page", evidence
    if not liveness.held and not complete:
        return "page", evidence
    return "healthy", evidence


def _campaign_complete(records_root: Path) -> bool:
    """Best-effort completion read; a failed read is NOT complete."""

    try:
        from menagerie.crawler.tools.launch_monitor import build_snapshot

        return build_snapshot(records_root, check_disk=False).complete
    except Exception:  # noqa: BLE001 -- fail toward paging, never toward quiet
        return False


def write_beat(path: Path, *, verdict: str, now: Optional[datetime] = None) -> None:
    """Write the synced beat file the OFF-HOST watcher consumes."""

    payload = {
        "schema_version": "menagerie.crawler.deadman-beat.v1",
        "written_at": (now or _utc_now()).isoformat(),
        "host_pid": os.getpid(),
        "verdict": verdict,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def check_beat(
    path: Path, *, max_age_seconds: float, now: Optional[datetime] = None
) -> tuple[str, str]:
    """OFF-HOST verdict over the synced beat file (missing/stale/fresh)."""

    now = now or _utc_now()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        written = _parse(payload.get("written_at"))
    except FileNotFoundError:
        return "missing", f"no beat at {path}"
    except (OSError, ValueError):
        return "unreadable", f"unreadable beat at {path}"
    if written is None:
        return "unreadable", "beat carries no written_at"
    age = (now - written).total_seconds()
    if age > max_age_seconds:
        return "stale", f"beat is {age:.0f}s old (max {max_age_seconds:.0f}s)"
    return "fresh", f"beat is {age:.0f}s old"


def render_launchd_plist(args: argparse.Namespace) -> str:
    """The on-host boot-surviving agent: RunAtLoad + StartInterval check."""

    import plistlib

    argv = [
        sys.executable,
        "-m",
        "menagerie.crawler.tools.deadman_watchdog",
        "--records-root",
        str(Path(args.records_root).resolve()),
        "--runtime-root",
        str(Path(args.runtime_root).resolve()),
        "--check",
        "--send",
    ]
    if args.beat is not None:
        argv.extend(("--beat", str(Path(args.beat).resolve())))
    return plistlib.dumps(
        {
            "Label": "com.menagerie.deadman-watchdog",
            "ProgramArguments": argv,
            "RunAtLoad": True,
            "StartInterval": int(args.interval),
            "WorkingDirectory": str(Path.cwd()),
        },
        fmt=plistlib.FMT_XML,
    ).decode("utf-8")


def render_remote_cron(args: argparse.Namespace) -> str:
    """The off-host checker's crontab line over the synced beat file."""

    beat = Path(args.beat).resolve() if args.beat else Path(local_value("DEADMAN_BEAT", "~/.local/state/menagerie/deadman-beat.json"))
    minutes = max(1, int(args.interval // 60))
    return (
        f"*/{minutes} * * * * PYTHONPATH={Path.cwd()} {sys.executable} -m"
        " menagerie.crawler.tools.deadman_watchdog"
        f" --check-beat {beat} --max-age-seconds {3 * args.interval:.0f} --send"
    )


def _send(message: str, enabled: bool) -> None:
    print(message, flush=True)
    if not enabled:
        return
    import subprocess

    script = Path("~/.claude/scripts/send-to-jmt.sh").expanduser()
    if not script.exists():
        print(f"[deadman] send script missing: {script}", file=sys.stderr)
        return
    try:
        subprocess.run([str(script), message], check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[deadman] send failed: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.tools.deadman_watchdog",
        description=__doc__.split("\n", 1)[0],
    )
    parser.add_argument("--records-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--hold", type=Path, default=None, help="HOLD sentinel path")
    parser.add_argument("--beat", type=Path, default=None, help="synced beat file")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-beat", type=Path, default=None)
    parser.add_argument("--max-age-seconds", type=float, default=900.0)
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--emit-launchd-plist", action="store_true")
    parser.add_argument("--emit-remote-cron", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_beat is not None:
        verdict, detail = check_beat(args.check_beat, max_age_seconds=args.max_age_seconds)
        print(f"deadman beat: {verdict} -- {detail}")
        if verdict != "fresh":
            _send(
                f"crawl host DARK: dead-man beat {verdict} ({detail}) -- the"
                " crawl host is not proving liveness; check it",
                args.send,
            )
            return 1
        return 0
    if args.emit_launchd_plist or args.emit_remote_cron:
        if args.emit_launchd_plist:
            if args.records_root is None or args.runtime_root is None:
                print("--emit-launchd-plist needs --records-root/--runtime-root", file=sys.stderr)
                return 64
            print(render_launchd_plist(args))
        if args.emit_remote_cron:
            print(render_remote_cron(args))
        return 0
    if not args.check:
        print("pass one of --check / --check-beat / --emit-*", file=sys.stderr)
        return 64
    if args.records_root is None or args.runtime_root is None:
        print("--check needs --records-root and --runtime-root", file=sys.stderr)
        return 64
    state_file = args.state_file or default_state_path(args.records_root)
    hold_path = args.hold or (args.runtime_root / "deadman-hold.json")
    verdict, evidence = evaluate_check(
        records_root=args.records_root,
        runtime_root=args.runtime_root,
        state_file=state_file,
        hold_path=hold_path,
        heartbeat_max_age_seconds=args.max_age_seconds,
    )
    print(f"deadman check: {verdict}")
    for line in evidence:
        print(f"  {line}")
    if verdict in ("page", "invalid-hold"):
        _send(
            f"crawl DEAD-MAN {verdict.upper()}: " + "; ".join(evidence),
            args.send,
        )
    if args.beat is not None and verdict in ("healthy", "holding"):
        write_beat(args.beat, verdict=verdict)
    return CHECK_VERDICTS[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
