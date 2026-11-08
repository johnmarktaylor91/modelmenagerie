"""``oneoff``: a big one-off Menagerie job on the runner host waits for runner B's day size.

A one-off job is anything a helper or agent submits by hand: a scan, a retype, a test run. Runners
A, B and G, axon and the tick's own units submit through their own code and never come here. A
one-off job declaring more than ``BIG_GIB`` that may land on the runner host is refused while B is
below its day size, is stopping, waits for capacity or has a resize pending; the refusal says when
to ask again (Lead, 29 Oct: a 24 GiB scan and a 20 GiB test job held B at one worker for four and a
half hours, and the stage 3 backlog grew by 70 rows).

Submit through it (it checks, then runs ``run-on submit`` with the same arguments)::

    python3 scripts/oneoff_submit.py -- --project menagerie --host HOST --memory-gib 24 ... -- CMD

or ask alone with ``--check --memory-gib G [--host H]``. Exit 0 admitted (or submitted, with
run-on's own exit code), 75 refused for now. ``--because TEXT`` submits anyway and logs the reason
to ``<tick folder>/oneoff-overrides.jsonl``; it is for a job JMT asked for by name.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from menagerie.local_config import local_value
from menagerie.paths import work_dir
from menagerie.pipeline.runner_job import B_DAY, NIGHT_ZONE, night_open

# the trace host as run-on names it (the tick's runner host); local_config key TRACE_HOST
RUNNER_HOST = local_value("TRACE_HOST", "trace-host")
BIG_GIB = 8.0  # declared memory above this waits for B's day size
STALE = timedelta(minutes=30)  # older tick state cannot vouch for B
TICK_EVERY = timedelta(minutes=10)
REFUSED = 75  # EX_TEMPFAIL: ask again later


@dataclass(frozen=True)
class Verdict:
    ok: bool
    why: str


def _pt(when: datetime) -> str:
    return when.astimezone(NIGHT_ZONE).strftime("%-I:%M %p PT")


def _retry(now: datetime) -> str:
    if night_open(now):
        return "B is off or small while G runs at night; ask again after 6 AM PT, once B is at its day size"
    nxt = now + TICK_EVERY - timedelta(minutes=now.minute % 10, seconds=now.second,
                                       microseconds=now.microsecond)
    return f"ask again after the next tick pass, about {_pt(nxt + timedelta(minutes=5))}"


def b_short(record: Mapping[str, Any] | None) -> str:
    """Why runner B is not settled at its day size, from its tick record ('' when it is).

    One flat check per way the tick records an unsettled B (CC 12: each branch is its own message)."""

    if not record or not record.get("job"):
        return "runner B is not running"
    if record.get("cancelled") and record.get("cancelled") == record.get("job"):
        why = {"g": "to give way to G at night", "resize": "to change size"}.get(record.get("cancelled_for"), "")
        return f"runner B is stopping {why}".rstrip()
    if record.get("refused_at") and record["refused_at"] >= (record.get("submitted_at") or ""):
        return "runner B waits for run-on capacity"
    if record.get("resize_to") is not None or record.get("draining"):
        return "runner B has a resize pending"
    size = record.get("size")
    if size is not None and size < B_DAY:
        return f"runner B is at size {size}, below its day size {B_DAY}"
    return ""


def verdict(memory_gib: float, host: str, state: Mapping[str, Any] | None, now: datetime,
            state_at: datetime | None) -> Verdict:
    """Whether a one-off job of ``memory_gib`` declared on ``host`` may go now."""

    if memory_gib <= BIG_GIB:
        return Verdict(True, f"{memory_gib:g} GiB is at most {BIG_GIB:g} GiB")
    if host not in (RUNNER_HOST, "any"):
        return Verdict(True, f"host {host} is not the runner host")
    if state is None or state_at is None or now - state_at > STALE:
        age = "missing" if state_at is None else f"{(now - state_at).total_seconds() / 60:.0f} minutes old"
        return Verdict(False, f"the tick state is {age}, so B's size is unknown; ask Crawler")
    short = b_short((state.get("runners") or {}).get("B"))
    if short:
        return Verdict(False, f"{short}; a {memory_gib:g} GiB one-off job would hold its room. {_retry(now)}")
    return Verdict(True, "runner B is at its day size")


def _state() -> tuple[dict[str, Any] | None, datetime | None]:
    path = work_dir("tick") / "state.json"
    try:
        return (json.loads(path.read_text(encoding="utf-8")),
                datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))
    except (OSError, ValueError):
        return None, None


def default_host(repo: Path) -> str:
    """run-on's own default: the checkout's ``.compute-host``, else any host."""

    path = repo / ".compute-host"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "any"


def submit_options(args: Sequence[str]) -> tuple[float, str | None]:
    """The declared memory and host in ``run-on submit`` arguments (before the payload's ``--``)."""

    head = list(args[:args.index("--")] if "--" in args else args)
    found: dict[str, str] = {}
    for i, word in enumerate(head):
        for flag in ("--memory-gib", "--host"):
            if word == flag and i + 1 < len(head):
                found[flag] = head[i + 1]
            elif word.startswith(flag + "="):
                found[flag] = word.split("=", 1)[1]
    return float(found.get("--memory-gib", 0)), found.get("--host")


def _override(because: str, args: Sequence[str], why: str, now: datetime) -> None:
    line = {"at": now.isoformat(timespec="seconds"), "because": because, "refusal": why, "args": list(args)}
    with open(work_dir("tick") / "oneoff-overrides.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps(line) + "\n")


def main(argv: Sequence[str] | None = None, *, repo: Path | None = None,
         run: Callable[[list[str]], int] | None = None, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oneoff_submit.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="only say whether a job may go now")
    parser.add_argument("--memory-gib", type=float, default=0.0, help="with --check: the declaration")
    parser.add_argument("--host", help="with --check: the host (default: run-on's)")
    parser.add_argument("--because", help="submit despite a refusal; the reason is logged")
    parser.add_argument("submit", nargs=argparse.REMAINDER, help="-- then the run-on submit arguments")
    opts = parser.parse_args(argv)
    args = opts.submit[1:] if opts.submit[:1] == ["--"] else opts.submit
    repo = repo or Path(__file__).resolve().parents[2]
    now = now or datetime.now(timezone.utc)
    memory, host = (opts.memory_gib, opts.host) if opts.check else submit_options(args)
    state, state_at = _state()
    answer = verdict(memory, host or default_host(repo), state, now, state_at)
    if opts.check or not args:
        print(("admitted: " if answer.ok else "refused: ") + answer.why)
        return 0 if answer.ok else REFUSED
    return _submit(args, answer, opts.because, now, run or _run_on)


def _submit(args: Sequence[str], answer: Verdict, because: str | None, now: datetime,
            run: Callable[[list[str]], int]) -> int:
    if not answer.ok and not because:
        print(f"refused: {answer.why}")
        return REFUSED
    if not answer.ok:
        _override(because, args, answer.why, now)
    return run(["run-on", "submit", *args])


def _run_on(argv: list[str]) -> int:
    return subprocess.run(argv, check=False).returncode
