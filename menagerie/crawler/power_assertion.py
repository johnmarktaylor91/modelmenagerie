"""Driver-held sleep immunity (DX-64).

200a ran with sleep immunity borrowed from a STRAY host ``caffeinate`` process
nothing owned, monitored, or restarted: killing that unrelated process (or a
reboot) would let the host sleep mid-leg with paid sessions in flight. The
driver now spawns and holds its OWN assertion for its exact lifetime, and the
resulting state is recorded in the driver-lock owner record so the leg-open
armed-guard census can surface an unarmed guard instead of silently relying on
ambient host state.

The mechanism is ``caffeinate -w <driver pid>``: the assertion child watches
the driver pid, so even a SIGKILLed driver releases its immunity — there is no
way to leak a system-wide no-sleep assertion past the leg.
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

#: Absolute pin: the assertion must never resolve through a writable PATH.
CAFFEINATE_EXECUTABLE = "/usr/bin/caffeinate"


@dataclass(frozen=True)
class PowerAssertionState:
    """The four census facts the driver-lock owner record publishes.

    Parameters
    ----------
    mechanism:
        ``caffeinate`` on Darwin, ``not-applicable`` elsewhere.
    held:
        Whether a live assertion child is holding sleep immunity.
    pid:
        Assertion child pid when one was spawned.
    detail:
        Why the guard is unarmed, when it is.
    """

    mechanism: str
    held: bool
    pid: Optional[int]
    detail: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the exact owner-record mapping."""

        return {
            "mechanism": self.mechanism,
            "held": self.held,
            "pid": self.pid,
            "detail": self.detail,
        }


@contextmanager
def driver_power_assertion(*, watched_pid: int) -> Iterator[PowerAssertionState]:
    """Hold sleep immunity for one driver's lifetime and report its state.

    An unarmed guard is RECORDED, never fatal: refusing the whole leg because
    an assertion could not spawn would turn a disclosure into an outage. The
    leg-open armed-guard census is the consumer that decides whether an
    unarmed power guard blocks a launch.

    Parameters
    ----------
    watched_pid:
        Driver pid the assertion child binds its lifetime to.
    """

    if sys.platform != "darwin":
        yield PowerAssertionState(
            mechanism="not-applicable",
            held=False,
            pid=None,
            detail=f"no driver-held assertion mechanism on {sys.platform}",
        )
        return
    if not Path(CAFFEINATE_EXECUTABLE).is_file():
        yield PowerAssertionState(
            mechanism="caffeinate",
            held=False,
            pid=None,
            detail=f"{CAFFEINATE_EXECUTABLE} is absent",
        )
        return
    try:
        child = subprocess.Popen(
            (
                CAFFEINATE_EXECUTABLE,
                "-i",
                "-m",
                "-s",
                "-w",
                str(watched_pid),
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        yield PowerAssertionState(
            mechanism="caffeinate", held=False, pid=None, detail=str(exc)
        )
        return
    if child.poll() is not None:
        yield PowerAssertionState(
            mechanism="caffeinate",
            held=False,
            pid=child.pid,
            detail="assertion child exited at spawn",
        )
        return
    try:
        yield PowerAssertionState(
            mechanism="caffeinate", held=True, pid=child.pid, detail=None
        )
    finally:
        try:
            child.terminate()
            child.wait(timeout=5)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
