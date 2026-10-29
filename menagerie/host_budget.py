"""How much of the host a Menagerie job may use.

Worker counts, thread counts and memory budgets come from the job's declaration,
never from ``os.cpu_count()``. The rule on shared machines:
a job declares its CPUs and memory, keeps the machine's reserve free, and takes at
most half of a shared machine.

Declaration, in order of precedence:

- ``MENAGERIE_CPUS`` / ``MENAGERIE_MEMORY_GB`` environment variables.
- CPUs: the process's CPU affinity when it is narrower than the machine
  (``run-on --cpus N`` pins the job with ``taskset``, so affinity is N).
- Otherwise half of the usable machine (cores or memory minus the reserve).
"""

from __future__ import annotations

import os

CPUS_ENV = "MENAGERIE_CPUS"
MEMORY_ENV = "MENAGERIE_MEMORY_GB"
MIN_MEMORY_GB = 1.0
RESERVE_MEMORY_FLOOR_GB = 8.0
RESERVE_MEMORY_FRACTION = 0.1


def reserve_cpus(total: int) -> int:
    """Return the cores the fleet keeps free for ssh, Tailscale and probes."""

    if total >= 8:
        return 2
    if total >= 4:
        return 1
    return 0


def reserve_memory_gb(total_gb: float) -> float:
    """Return the memory the fleet keeps free: the larger of 8 GiB or 10 percent."""

    return max(RESERVE_MEMORY_FLOOR_GB, RESERVE_MEMORY_FRACTION * total_gb)


def _affinity_count() -> int | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    return len(getter(0))


def _total_memory_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)


def declared_cpus(affinity: int | None = None, total: int | None = None) -> int:
    """Return the CPU count this job may use.

    Parameters
    ----------
    affinity:
        Cores the process may run on; read from the OS when omitted.
    total:
        Cores on the machine; ``os.cpu_count()`` when omitted.
    """

    raw = os.environ.get(CPUS_ENV)
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value < 1:
            raise ValueError(f"{CPUS_ENV} must be a positive integer, got {raw!r}")
        return value
    total = total if total is not None else (os.cpu_count() or 1)
    affinity = affinity if affinity is not None else (_affinity_count() or total)
    if affinity < total:
        return max(1, affinity)
    return max(1, (total - reserve_cpus(total)) // 2)


def declared_memory_gb(total_gb: float | None = None) -> float:
    """Return the memory in GiB this job may use across all its workers.

    Parameters
    ----------
    total_gb:
        Physical memory in GiB; read from the OS when omitted.
    """

    raw = os.environ.get(MEMORY_ENV)
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value <= 0:
            raise ValueError(f"{MEMORY_ENV} must be a positive number of GiB, got {raw!r}")
        return value
    total_gb = total_gb if total_gb is not None else _total_memory_gb()
    return max(MIN_MEMORY_GB, (total_gb - reserve_memory_gb(total_gb)) / 2)


def threads_per_worker(cpus: int, jobs: int) -> int:
    """Return threads per worker so workers times threads stays within ``cpus``."""

    return max(1, cpus // max(1, jobs))
