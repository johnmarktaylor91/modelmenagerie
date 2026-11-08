"""The worker pause: the tick config's ``zmachine_paused`` hands the trace host to another project.

While it is set (``{"for": <who has the host>, "since": <ISO time>, "until_note": <when it ends>}``,
optional ``"host"``, default the trace host, ``DEFAULT_HOST``), the tick drains runners A, B and G and cancels each once
its spool is idle (``runner_job.pause``; a planned stop, never a restart), starts no runner, store
trim, reclaim, axon render or 2b lane, gives the host no lane slots (``mini_placement.host_cap``),
and stops the timers in ``TIMERS``. It still collects the spools and commits, and the hub and Mac
stages keep running. Clearing the key undoes all of it: the next pass starts the runners again and
the stopped timers (``timers``), and the host's cap is the configured one.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from typing import Any

from menagerie.local_config import local_value

KEY = "zmachine_paused"
DEFAULT_HOST = local_value("TRACE_HOST", "trace-host")  # the trace host as run-on names it
# Timers whose services run work on the trace host (counts on the worker, the daily store-box
# batch). A transient timer is never listed: stopping one deletes it. A retired timer is removed
# here, as the catalog night boost was (its grant ended 12 Oct; decisions.md 2026-10-30).
TIMERS = ("menagerie-counts.timer", "menagerie-store-box-daily.timer")


def active(cfg: Mapping[str, Any]) -> dict[str, str] | None:
    """The pause when the config sets one, else None; a malformed one is refused."""

    given = cfg.get(KEY)
    if given is None or given is False:
        return None
    if not isinstance(given, Mapping) or not all(
            isinstance(given.get(k), str) and given.get(k) for k in ("for", "since", "until_note")):
        raise ValueError(f"{KEY} must be an object with for, since and until_note strings")
    return {"host": DEFAULT_HOST, **given}


def held_host(cfg: Mapping[str, Any], host: str) -> bool:
    """Whether ``host`` is the paused worker."""

    pause = active(cfg)
    return pause is not None and pause["host"] == host


def note(cfg: Mapping[str, Any]) -> str | None:
    """The status line's words, such as ``paused for TorchLens since ... (until ...)``."""

    pause = active(cfg)
    if pause is None:
        return None
    return (f"{pause['host']} paused for {pause['for']} since {pause['since']} "
            f"({pause['until_note']})")


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True,
                          check=False, timeout=30)


def timers(paused: bool, held: list[str] | None, *, dry_run: bool = False,
           run: Callable[..., subprocess.CompletedProcess[str]] = _systemctl,
           ) -> tuple[list[str] | None, dict[str, list[str]]]:
    """Hold the timers in ``TIMERS`` for a pause: on its first pass, ``held`` (the tick state's
    list, None outside a pause) becomes every enabled one that is running, and each is stopped; once
    the pause is cleared, each held one still enabled is started and ``held`` goes back to None. A
    timer that was already stopped, or was disabled during the pause, stays as it is.
    Returns the new ``held`` and what changed."""

    if paused:
        held = held if held is not None else [
            u for u in TIMERS if run("is-enabled", u).stdout.strip() == "enabled"
            and run("is-active", u).stdout.strip() == "active"]
        stopped = [u for u in held if run("is-active", u).stdout.strip() == "active"]
        for unit in [] if dry_run else stopped:
            run("stop", unit)
        return held, {"stopped": stopped} if stopped else {}
    if held is None:
        return None, {}
    start = [u for u in held if run("is-enabled", u).stdout.strip() == "enabled"]
    for unit in [] if dry_run else start:
        run("start", unit)
    left = [u for u in held if u not in start]
    return (held if dry_run else None), {"started": start, **({"left_disabled": left} if left else {})}
