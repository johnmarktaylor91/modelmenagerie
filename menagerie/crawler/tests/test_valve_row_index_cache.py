"""Valve-row reads index once per file change instead of once per work item.

T4-T5 (C-PROJ-CACHE memoization half): ``_still_valid_deferral_row`` re-read
and re-validated EVERY durable valve ledger from disk per scheduled work item,
and the grant-park/quarantine fact rechecks full-scanned the models shard and
the operational ledger per row. The stat-signature cache keeps each derived
read exactly as fresh as its source files: any append, truncation, creation,
or removal changes the ``(size, mtime_ns)`` signature and recomputes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler.driver_admission import (
    _file_stat_signature,
    _persisted_deferral_rows,
    _stat_signature_cached,
)
from menagerie.crawler.host_capacity import (
    HostCapacity,
    append_capacity_deferral_row,
    assess_model_capacity,
    build_capacity_deferral_row,
    capacity_deferral_path,
)
from menagerie.crawler.tests.test_host_capacity import MIXTRAL_RECIPE

#: Sub-second load-bearing coverage promoted into the mandatory smoke tier (T5-7).
pytestmark = pytest.mark.smoke

#: 16 GiB test host: small enough that the Mixtral recipe refuses.
SMALL_HOST = HostCapacity(physical_memory_bytes=16 * 2**30, memory_source="test")


def _driver(tmp_path: Path) -> SimpleNamespace:
    """Return the duck-typed driver shape ``_persisted_deferral_rows`` reads."""

    models = tmp_path / "records" / "models" / "current-shard.jsonl"
    models.parent.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        paths=SimpleNamespace(ledgers=SimpleNamespace(models=models)),
        config=SimpleNamespace(machine_id="machine-1"),
    )


def _capacity_row(stable_id: str) -> dict[str, Any]:
    """Build one validator-passing capacity deferral row."""

    return build_capacity_deferral_row(
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        name="Mixtral 8x7B",
        campaign_id="c1-test",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-04T02:52:37.527339Z",
        assessment=assess_model_capacity(MIXTRAL_RECIPE, host=SMALL_HOST),
    )


def test_persisted_deferral_rows_serve_cached_until_a_valve_file_changes(
    tmp_path: Path,
) -> None:
    """Unchanged valve files serve the cached index; an append recomputes."""

    driver = _driver(tmp_path)
    records_root = driver.paths.ledgers.models.parent.parent
    valve_path = capacity_deferral_path(records_root)
    append_capacity_deferral_row(valve_path, _capacity_row("m_one"))

    first = _persisted_deferral_rows(driver)
    assert set(first) == {"m_one"}
    assert _persisted_deferral_rows(driver) is first, (
        "unchanged valve ledgers must serve the cached index"
    )

    append_capacity_deferral_row(valve_path, _capacity_row("m_two"))
    second = _persisted_deferral_rows(driver)
    assert second is not first
    assert set(second) == {"m_one", "m_two"}


def test_stat_signature_cache_recomputes_on_creation_and_removal(
    tmp_path: Path,
) -> None:
    """File creation and removal both change the signature."""

    target = tmp_path / "ledger.jsonl"
    driver = SimpleNamespace()
    calls = [0]

    def compute() -> int:
        calls[0] += 1
        return calls[0]

    assert _stat_signature_cached(driver, "probe", (target,), compute) == 1
    assert _stat_signature_cached(driver, "probe", (target,), compute) == 1
    target.write_text("{}\n", encoding="utf-8")
    assert _stat_signature_cached(driver, "probe", (target,), compute) == 2
    target.unlink()
    assert _stat_signature_cached(driver, "probe", (target,), compute) == 3
    # Distinct cache keys never collide.
    assert _stat_signature_cached(driver, "other", (target,), compute) == 4


def test_stat_signature_distinguishes_missing_from_empty(tmp_path: Path) -> None:
    """A missing file and an empty file are different signature arms."""

    target = tmp_path / "ledger.jsonl"
    missing = _file_stat_signature((target,))
    target.write_text("", encoding="utf-8")
    empty = _file_stat_signature((target,))
    assert missing != empty


def test_stat_signature_cache_degrades_to_live_reads_without_settable_driver() -> None:
    """A driver refusing attributes still gets correct (uncached) answers."""

    class Frozen:
        __slots__ = ()

    calls = [0]

    def compute() -> int:
        calls[0] += 1
        return calls[0]

    assert _stat_signature_cached(Frozen(), "probe", (), compute) == 1
    assert _stat_signature_cached(Frozen(), "probe", (), compute) == 2
