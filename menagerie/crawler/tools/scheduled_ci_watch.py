"""TC-21: scheduled-CI reds routed to iMessage (nightly/weekly/canary).

The nightly, weekly, and latest-canary workflows run on GitHub's schedule and
their failures land in an unwatched Actions tab -- red and dark. A hosted
runner cannot reach the phone, so the router lives HOST-SIDE: this poller asks
``gh run list`` for each scheduled workflow's newest completed run and texts
exactly once per newly-red run (state-file dedup, so a standing red does not
re-page every poll and a NEW red after a green does). Recovery texts once too:
silence never has to mean healthy.

Install: ``--emit-launchd-plist`` prints the polling agent (per-machine
operator act, same rule as the dead-man pair).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from menagerie.local_config import local_value

#: The scheduled workflows under watch (closed; extend in a reviewed change).
SCHEDULED_WORKFLOWS = ("nightly.yml", "weekly.yml", "latest-canary.yml")

STATE_SCHEMA_VERSION = "menagerie.crawler.scheduled-ci-watch-state.v1"


def _gh_latest_completed(workflow: str, repo_root: Path) -> Optional[dict[str, Any]]:
    """Return the newest completed run for one workflow via the gh CLI."""

    completed = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--status",
            "completed",
            "--limit",
            "1",
            "--json",
            "databaseId,conclusion,displayTitle,url,updatedAt",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        print(
            f"[ci-watch] gh failed for {workflow}: {completed.stderr.strip()}",
            file=sys.stderr,
        )
        return None
    try:
        runs = json.loads(completed.stdout or "[]")
    except ValueError:
        return None
    return runs[0] if isinstance(runs, list) and runs else None


def check_workflows(
    *,
    state_path: Path,
    repo_root: Path,
    fetch: Callable[[str, Path], Optional[dict[str, Any]]] = _gh_latest_completed,
    send: Callable[[str], None],
) -> int:
    """Poll every scheduled workflow; text on new reds and on recovery.

    Returns the number of currently-red scheduled workflows. An unreachable
    workflow (gh failure) is DISCLOSED via a text after two consecutive
    misses -- an unwatchable watcher is the dark state this tool exists to
    kill, so it never reads as green.
    """

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            state = {}
    except (OSError, ValueError):
        state = {}
    reported: dict[str, Any] = dict(state.get("reported", {}))
    misses: dict[str, int] = {k: int(v) for k, v in dict(state.get("misses", {})).items()}
    red = 0
    for workflow in SCHEDULED_WORKFLOWS:
        run = fetch(workflow, repo_root)
        if run is None:
            misses[workflow] = misses.get(workflow, 0) + 1
            if misses[workflow] == 2:
                send(
                    f"scheduled CI watch cannot read {workflow} (gh failing"
                    " twice); the scheduled suites are UNWATCHED until fixed"
                )
            continue
        misses[workflow] = 0
        run_id = str(run.get("databaseId"))
        conclusion = str(run.get("conclusion", ""))
        prior = reported.get(workflow, {})
        if conclusion != "success":
            red += 1
            if prior.get("run_id") != run_id or prior.get("conclusion") != conclusion:
                send(
                    f"scheduled CI RED: {workflow} concluded {conclusion}"
                    f" ({run.get('displayTitle', '')}) {run.get('url', '')}"
                )
                reported[workflow] = {"run_id": run_id, "conclusion": conclusion}
        else:
            if prior and prior.get("conclusion") not in (None, "success"):
                send(f"scheduled CI recovered: {workflow} is green again")
            reported[workflow] = {"run_id": run_id, "conclusion": "success"}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp = state_path.with_suffix(state_path.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "reported": reported,
                "misses": misses,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, state_path)
    return red


def _send_imessage(message: str) -> None:
    print(message, flush=True)
    script = Path("~/.claude/scripts/send-to-jmt.sh").expanduser()
    if not script.exists():
        print(f"[ci-watch] send script missing: {script}", file=sys.stderr)
        return
    try:
        subprocess.run([str(script), message], check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[ci-watch] send failed: {exc}", file=sys.stderr)


def render_launchd_plist(interval_seconds: float, repo_root: Path) -> str:
    import plistlib

    return plistlib.dumps(
        {
            "Label": "com.menagerie.scheduled-ci-watch",
            "ProgramArguments": [
                sys.executable,
                "-m",
                "menagerie.crawler.tools.scheduled_ci_watch",
                "--repo-root",
                str(repo_root.resolve()),
                "--send",
            ],
            "RunAtLoad": True,
            "StartInterval": int(interval_seconds),
            "WorkingDirectory": str(repo_root.resolve()),
        },
        fmt=plistlib.FMT_XML,
    ).decode("utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.tools.scheduled_ci_watch",
        description=__doc__.split("\n", 1)[0],
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--interval", type=float, default=1800.0)
    parser.add_argument("--emit-launchd-plist", action="store_true")
    args = parser.parse_args(argv)
    if args.emit_launchd_plist:
        print(render_launchd_plist(args.interval, args.repo_root))
        return 0
    state_file = args.state_file or (
        Path(local_value("LAUNCH_MONITOR_STATE_DIR", "~/.local/state/menagerie/launch-monitor")).expanduser()
        / "scheduled-ci-watch.json"
    )
    send = _send_imessage if args.send else lambda message: print(message, flush=True)
    red = check_workflows(state_path=state_file, repo_root=args.repo_root, send=send)
    return 1 if red else 0


if __name__ == "__main__":
    raise SystemExit(main())
