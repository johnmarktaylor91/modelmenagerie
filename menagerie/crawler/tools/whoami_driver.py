"""Lock-based driver liveness: the anti-pgrep authority.

``pgrep``-shaped process identification misread driver liveness in 200a
(name matching is neither necessary nor sufficient: a dead driver leaves no
process but its lock record; a foreign python matches the name). The global
execution flock IS the liveness fact: a live driver HOLDS it; a dead one's
kernel lock evaporated with its process. This tool probes the lock
non-destructively and reports the recorded holder metadata beside the
kernel-observed liveness.

Exit codes: 0 = probe answered (JSON on stdout); with ``--expect-live`` /
``--expect-free``, 3 = expectation not met (for gate composition).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from menagerie.crawler.execution_lock import (
    global_execution_flock_path,
    read_global_execution_flock_holder,
)


def probe(lock_path: Optional[Path] = None) -> dict[str, Any]:
    """Probe the global execution flock without disturbing a live holder.

    A shared, non-blocking flock attempt answers liveness: an exclusive
    holder makes it fail with EWOULDBLOCK; acquiring (and instantly
    releasing) the shared lock proves no driver holds it.
    """

    path = lock_path or global_execution_flock_path()
    holder = read_global_execution_flock_holder(path)
    if not path.exists():
        return {"lock_path": str(path), "live": False, "holder": None, "lock_file": "absent"}
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            live = True
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            live = False
    finally:
        os.close(descriptor)
    return {"lock_path": str(path), "live": live, "holder": holder, "lock_file": "present"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Answer "is a driver alive here?" from the lock, never from pgrep."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-path", type=Path, default=None)
    expectation = parser.add_mutually_exclusive_group()
    expectation.add_argument("--expect-live", action="store_true")
    expectation.add_argument("--expect-free", action="store_true")
    args = parser.parse_args(argv)
    report = probe(args.lock_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.expect_live and not report["live"]:
        print("whoami-driver: expected a live driver; the lock is free", file=sys.stderr)
        return 3
    if args.expect_free and report["live"]:
        print("whoami-driver: expected no driver; the lock is HELD", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
