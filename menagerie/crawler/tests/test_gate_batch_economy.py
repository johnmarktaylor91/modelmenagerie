"""Metadata batch economy: a leg must not buy one session per model.

The checker session is the crawler's expensive unit -- the pilot burned ~15h of
checker wall against 3.4 minutes of model execution -- and the batch cap is
calibrated (``METADATA_BATCH_MIN == METADATA_BATCH_MAX``, constants.py; the
rung-10 incident proved 10+ item batches blow the lane's output ceiling and
terminalized 41 innocent models, and the 2026-08-29 width rotation pinned the
cap at ONE after the Arm A matrix measured width-5 acceptance indeterminacy).
So the batcher has exactly one job at scale: fill to the cap whenever the cap's
worth are available, and dispatch a SHORT batch only when it can prove no
further member can arrive. At width 1 every batch is trivially full, and this
suite's arithmetic is deliberately symbolic in the cap so the economy pins keep
binding: the batcher must still emit exactly ceil(n/cap) sessions, never a
re-buy and never an abandoned tail.

Both halves are load-bearing at 20k, in opposite directions:

* Never packing means paying more sessions than the ceil(n/cap) optimum -- the
  session class this axis exists to protect.
* Never flushing a short tail means fully authored, fully executed models
  silently left ungated at leg end, their work discarded and re-bought.

The M3 pipelined scheduler is where this is decided (``scheduler.py`` step 3:
"cap-width eager, short only with the tail proof"), because per-model pipelining
removed the intake-wide barrier that used to make every batch full by
construction. Its accumulator and final-tail proof had no direct test.

One session class is EXEMPT from packing by reviewed design: the H-5
duplicate-accept write door (rule-2, 2026-08-29) buys exactly ONE extra blind
width-1 confirmation session per ACCEPTED gate item -- never batched, never
pooled -- so an all-accept leg over n models lawfully buys ceil(n/cap)
first-round sessions PLUS n confirmations. The witness below classifies the
two classes at dispatch time so the packing pins keep binding to the batcher
alone and the door's per-accept cost is pinned exactly (one each, width one,
never a re-buy of a first-round session).

Marked ``slow``: each case runs a full driver leg (~1 minute), which is past the
``heavy`` band and must not inflate the mid backstop tier. The four intake sizes
are the structurally distinct residue classes mod the cap -- tail only, exact
fill, full plus a minimal tail, and multiple full batches plus a tail -- and two
further legs put a model that never authors beside a tail, which is the case
where the proof has to distinguish "no further member can arrive" from "a member
is still pending".
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import METADATA_BATCH_MAX
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    AuthorScript,
    FakeChecker,
    ScriptedAuthor,
    _driver,
    _paths,
    _snapshot,
)


class BatchSizeWitness(FakeChecker):
    """Record the width of every metadata session bought, by session class.

    ``mark_confirmations`` must be armed on the driver before the run: the H-5
    duplicate-accept door dispatches its confirmation session synchronously
    inside ``_require_confirmed_accuracy_acceptance`` (never a pool capsule),
    so a thread-local flag around the door separates the door's per-accept
    confirmations from the batcher's first-round sessions exactly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.first_round_sizes: list[int] = []
        self.confirmation_sizes: list[int] = []
        self._door_thread_state = threading.local()

    def mark_confirmations(self, driver: Any) -> None:
        """Flag the H-5 door's synchronous confirmation buys on this driver."""

        inner = driver._require_confirmed_accuracy_acceptance  # noqa: SLF001

        def marked(*args: Any, **kwargs: Any) -> Any:
            self._door_thread_state.active = True
            try:
                return inner(*args, **kwargs)
            finally:
                self._door_thread_state.active = False

        driver._require_confirmed_accuracy_acceptance = marked  # noqa: SLF001

    def check_metadata(self, artifacts, work_root, config):  # noqa: ANN001, ANN201
        """Count one bought metadata session and its width, classified."""

        width = len(tuple(artifacts))
        if getattr(self._door_thread_state, "active", False):
            self.confirmation_sizes.append(width)
        else:
            self.first_round_sizes.append(width)
        return super().check_metadata(artifacts, work_root, config)


def _run(tmp_path: Path, count: int, scheduler: str) -> tuple[BatchSizeWitness, Any]:
    """Run one full leg over ``count`` models and return the witness and paths."""

    snapshot = _snapshot(tmp_path, count=count)
    checker = BatchSizeWitness()
    driver = _driver(tmp_path, snapshot, checker=checker, scheduler=scheduler)
    checker.mark_confirmations(driver)
    result = driver.run()
    assert result.status == "complete", result.status
    return checker, _paths(tmp_path, snapshot)


@pytest.mark.slow
@pytest.mark.parametrize("count", (4, 5, 6, 11))
def test_pipelined_leg_packs_to_the_cap_and_flushes_its_tail_exactly_once(
    tmp_path: Path, count: int
) -> None:
    """Every model is gated, and the session count is the packing optimum.

    ``ceil(count / 5)`` is the floor any correct batcher must reach and the
    ceiling a well-packed one must not exceed, so pinning equality pins BOTH
    failure directions at once: an unpacked trickle (one session per model)
    exceeds it, and an unflushed tail leaves models ungated below it.
    """

    checker, paths = _run(tmp_path, count, "pipelined")
    sizes = checker.first_round_sizes

    optimum = -(-count // METADATA_BATCH_MAX)  # ceil division
    assert len(sizes) == optimum, (
        f"{count} models bought {sizes} metadata sessions; the packing optimum is {optimum}"
    )
    # Every model actually reached a gate: the tail was flushed, not abandoned,
    # and no model rode two first-round sessions.
    assert sum(sizes) == count
    # No batch ever exceeds the token-calibrated cap (the rung-10 incident).
    assert max(sizes) <= METADATA_BATCH_MAX
    # At most one short batch, and it comes last: a tail flushed before the
    # accumulator could fill would fragment every batch after it.
    assert len([size for size in sizes if size < METADATA_BATCH_MAX]) <= 1, sizes
    assert sizes == sorted(sizes, key=lambda size: size != METADATA_BATCH_MAX), sizes
    # H-5: every accepted item buys exactly ONE blind width-1 confirmation --
    # per accept, never batched, and never a re-buy of a first-round session.
    assert checker.confirmation_sizes == [1] * count, checker.confirmation_sizes
    assert len(scan_jsonl(paths.ledgers.models)) == count


@pytest.mark.slow
@pytest.mark.parametrize("count", (4, 7))
def test_a_model_that_never_authors_does_not_hold_the_tail_hostage(
    tmp_path: Path, count: int
) -> None:
    """A failed author must not block the short flush of everyone else's tail.

    The final-tail proof asks whether any model outside the accumulator could
    still become gate-ready this leg, and answers it from author/exec readiness.
    A model whose author command failed is the interesting case: it is neither
    gate-ready nor going to be, so it must read as SETTLED and drop out of the
    proof. If it read as a possible future arrival instead, the tail would never
    flush and the leg would end on the scheduler's no-durable-progress tripwire
    with everyone else's authored, executed work discarded.

    Both sizes matter: 4 leaves a tail with no full batch ahead of it (the proof
    is the only thing that can flush anything at all), 7 leaves a full batch plus
    a tail (the proof runs after a flush has already moved the accumulator).
    """

    snapshot = _snapshot(tmp_path, count=count)
    failed_id = snapshot.items[0].stable_id
    checker = BatchSizeWitness()
    driver = _driver(
        tmp_path,
        snapshot,
        author=ScriptedAuthor(AuthorScript(failed_id=failed_id)),
        checker=checker,
        scheduler="pipelined",
    )
    checker.mark_confirmations(driver)
    result = driver.run()
    # A model that cannot author terminalizes, so this leg legitimately closes a
    # terminal partition; the economy claim is about the OTHER models' sessions,
    # so the status is accepted either way and the batching is pinned below.
    assert result.status in {"complete", "terminal-partition-complete"}, result.status

    gated = count - 1
    sizes = checker.first_round_sizes
    assert sum(sizes) == gated, f"{sizes} gated {sum(sizes)} of {gated} authorable models"
    assert len(sizes) == -(-gated // METADATA_BATCH_MAX), sizes
    # H-5: confirmations track ACCEPTS -- the failed author never buys one.
    assert checker.confirmation_sizes == [1] * gated, checker.confirmation_sizes

    records = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert records[failed_id]["status"]["code"].startswith("failed:"), records[failed_id]["status"]
    assert len(records) == count


class PooledSettleWitness(BatchSizeWitness):
    """Record per-session occupancy so pooled settle is measurable (SD-14).

    ``check_metadata`` is called from checker-pool threads, so every mutation
    is lock-guarded; the ``hold`` keeps each session open long enough that
    genuinely concurrent submissions must overlap on the wall clock.
    """

    def __init__(self, hold_seconds: float = 0.0) -> None:
        super().__init__()
        self._hold_seconds = hold_seconds
        self._witness_lock = threading.Lock()
        self._live = 0
        self.peak_concurrency = 0

    def check_metadata(self, artifacts, work_root, config):  # noqa: ANN001, ANN201
        with self._witness_lock:
            self._live += 1
            self.peak_concurrency = max(self.peak_concurrency, self._live)
        try:
            if self._hold_seconds:
                time.sleep(self._hold_seconds)
            with self._witness_lock:
                return super().check_metadata(artifacts, work_root, config)
        finally:
            with self._witness_lock:
                self._live -= 1


@pytest.mark.slow
def test_accumulator_flushes_all_full_batches_in_one_pooled_call(tmp_path: Path) -> None:
    """SD-14: the accumulator hands ALL full batches to ONE ``_ensure_gates`` call.

    The serial shape this pins against: step 3 used to loop ``flush_gate_batch``
    once per width-5 chunk, so each call built a fresh checker pool, submitted
    exactly one metadata session into a 24-wide pool, and settled it before the
    next chunk was even offered -- N/5 sequential session walls where the pool
    could have paid ~1. With 17 models authored at full width, the 15-member
    full prefix must arrive as a single call (its three width-5 batches are
    composed inside ``_run_gate_rounds`` exactly as before) and the pool must
    genuinely overlap the three bought sessions. The 2-member tail still rides
    the final-tail proof, unchanged.
    """

    snapshot = _snapshot(tmp_path, count=17)
    checker = PooledSettleWitness(hold_seconds=0.2)
    driver = _driver(
        tmp_path,
        snapshot,
        checker=checker,
        scheduler="pipelined",
        author_concurrency=17,
        checker_concurrency=24,
    )
    checker.mark_confirmations(driver)

    ensure_gates_call_widths: list[int] = []
    inner_ensure_gates = driver._ensure_gates  # noqa: SLF001

    def counting_ensure_gates(work, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        ensure_gates_call_widths.append(len(tuple(work)))
        return inner_ensure_gates(work, *args, **kwargs)

    driver._ensure_gates = counting_ensure_gates  # noqa: SLF001
    result = driver.run()
    assert result.status == "complete", result.status

    # The batch economy is untouched: same widths, same count, short tail last.
    sizes = checker.first_round_sizes
    assert sum(sizes) == 17
    assert len(sizes) == -(-17 // METADATA_BATCH_MAX), sizes
    assert max(sizes) <= METADATA_BATCH_MAX
    assert sizes == sorted(sizes, key=lambda size: size != METADATA_BATCH_MAX), sizes
    # H-5: one blind width-1 confirmation per accepted item, outside the pool.
    assert checker.confirmation_sizes == [1] * 17, checker.confirmation_sizes

    # The whole-flush pin: every full batch of the accumulated prefix rides ONE
    # call (width >= 3 * cap), never one call per width-5 chunk.
    assert max(ensure_gates_call_widths) >= 3 * METADATA_BATCH_MAX, (
        f"_ensure_gates call widths {ensure_gates_call_widths}: the full prefix "
        "was chunked into per-batch calls instead of one pooled flush"
    )

    # Pooled settle: the three prefetched sessions overlap on the wall clock.
    assert checker.peak_concurrency >= 3, (
        f"peak checker concurrency {checker.peak_concurrency}: the pool settled "
        "the full prefix serially"
    )


@pytest.mark.slow
def test_wave_and_pipelined_schedulers_agree_on_batch_economy(tmp_path: Path) -> None:
    """The pipelined scheduler must not be economically worse than waves.

    The wave driver batches a whole intake at once, so it is the natural
    reference for packing. M3 removed that barrier for latency; it must not have
    paid for latency in checker sessions.
    """

    waves_root = tmp_path / "waves"
    pipelined_root = tmp_path / "pipelined"
    waves_root.mkdir()
    pipelined_root.mkdir()

    waves, _ = _run(waves_root, 7, "waves")
    pipelined, _ = _run(pipelined_root, 7, "pipelined")

    assert sum(pipelined.first_round_sizes) == sum(waves.first_round_sizes) == 7
    assert len(pipelined.first_round_sizes) <= len(waves.first_round_sizes), (
        f"pipelined bought {pipelined.first_round_sizes} vs waves {waves.first_round_sizes}"
    )
    # The H-5 confirmation spend is scheduler-invariant: one per accepted item.
    assert pipelined.confirmation_sizes == waves.confirmation_sizes == [1] * 7
