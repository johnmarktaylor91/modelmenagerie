"""Subprocess runner for real-SIGKILL scheduler rehearsals.

``test_scheduler_under_fire`` launches this module as a child process; the
child builds the standard fake-lane driver rig and hard-kills ITSELF
(``os.kill(getpid(), SIGKILL)``) at an exact boundary occurrence. Unlike the
in-process ``InjectedKill`` matrix, nothing unwinds: no ``finally`` blocks, no
pool drain, no flock release beyond what the kernel does at process death.
The parent then resumes the campaign in-process and asserts reconciliation.

Run: ``python -m menagerie.crawler.tests.scheduler_fire_support --root DIR
--count N --scheduler pipelined --kill-boundary after-gate --kill-occurrence 1``
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run one fake-lane driver leg, self-SIGKILLing at the target boundary.

    Parameters
    ----------
    argv:
        Optional argument vector override.

    Returns
    -------
    int
        Driver exit disposition when the kill boundary never fired.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--scheduler", default="pipelined")
    parser.add_argument("--kill-boundary", default=None)
    parser.add_argument("--kill-occurrence", type=int, default=1)
    args = parser.parse_args(argv)

    from menagerie.crawler.intake import load_intake_snapshot
    from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot

    intake_root = args.root / "intake"
    if intake_root.exists():
        snapshot = load_intake_snapshot(intake_root)
    else:
        snapshot = _snapshot(args.root, count=args.count)

    seen = {"n": 0}

    def hard_kill(boundary: str, _stable_id: str) -> None:
        if boundary != args.kill_boundary:
            return
        seen["n"] += 1
        if seen["n"] == args.kill_occurrence:
            # Flush nothing, unwind nothing: the whole point.
            os.kill(os.getpid(), signal.SIGKILL)

    result = _driver(args.root, snapshot, boundary=hard_kill, scheduler=args.scheduler).run()
    print(f"completed-without-kill:{result.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
