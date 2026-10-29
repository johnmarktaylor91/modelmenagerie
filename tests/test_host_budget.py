"""Tests for the host resource budget that sizes Menagerie workers."""

from __future__ import annotations

import pytest

from menagerie import host_budget
from menagerie.host_budget import (
    declared_cpus,
    declared_memory_gb,
    reserve_cpus,
    reserve_memory_gb,
    threads_per_worker,
)


def test_reserve_matches_fleet_rule() -> None:
    assert reserve_cpus(20) == 2
    assert reserve_cpus(8) == 2
    assert reserve_cpus(6) == 1
    assert reserve_cpus(2) == 0
    assert reserve_memory_gb(125.0) == pytest.approx(12.5)
    assert reserve_memory_gb(32.0) == 8.0


def test_env_declaration_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_CPUS", "6")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "40")
    assert declared_cpus(affinity=20, total=20) == 6
    assert declared_memory_gb(total_gb=125.0) == 40.0


def test_pinned_affinity_is_the_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    """run-on pins a job with taskset, so affinity below the core count is the declaration."""

    monkeypatch.delenv("MENAGERIE_CPUS", raising=False)
    assert declared_cpus(affinity=9, total=20) == 9


def test_unpinned_job_gets_half_the_usable_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job started without a declaration never sizes itself to nproc."""

    monkeypatch.delenv("MENAGERIE_CPUS", raising=False)
    monkeypatch.delenv("MENAGERIE_MEMORY_GB", raising=False)
    assert declared_cpus(affinity=20, total=20) == 9
    assert declared_cpus(affinity=1, total=1) == 1
    assert declared_memory_gb(total_gb=125.0) == pytest.approx(56.25)
    assert declared_memory_gb(total_gb=4.0) == 1.0


def test_bad_env_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_CPUS", "0")
    with pytest.raises(ValueError, match="MENAGERIE_CPUS"):
        declared_cpus(affinity=20, total=20)
    monkeypatch.setenv("MENAGERIE_CPUS", "4")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "lots")
    with pytest.raises(ValueError, match="MENAGERIE_MEMORY_GB"):
        declared_memory_gb(total_gb=125.0)


@pytest.mark.parametrize(("cpus", "jobs", "expected"), [(9, 3, 3), (9, 2, 4), (9, 4, 2), (1, 5, 1)])
def test_threads_times_workers_stay_within_declaration(cpus: int, jobs: int, expected: int) -> None:
    threads = threads_per_worker(cpus, jobs)
    assert threads == expected
    assert threads * jobs <= max(cpus, jobs)


def test_live_host_values_are_sane() -> None:
    assert host_budget.declared_cpus() >= 1
    assert host_budget.declared_memory_gb() >= 1.0


def test_default_worker_memory_cap_splits_the_share(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.validate_menagerie import default_worker_memory_cap_gb

    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "56")
    assert default_worker_memory_cap_gb(1) == 56.0
    assert default_worker_memory_cap_gb(4) == 14.0
    assert default_worker_memory_cap_gb(0) == 56.0
