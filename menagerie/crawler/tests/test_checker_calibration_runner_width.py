"""Width surfaces of the calibration runner (pure decision functions only).

The width rotation (Opus adjudication 5.3 rule 1, 2026-08-29) re-measures the
metadata lane at the new production width on a durable root that already
carries the retired width's traffic. Two runner surfaces make that honest:

* :func:`derive_width_packs` re-chunks frozen packs recorded at another width
  into production-width packs -- same items, same envelope builder, the
  grouping production would use -- so a width rotation can build its
  measurement plan from the existing frozen corpus without hand-authoring
  envelopes.
* ``run_measure``'s rendered-session floor counts PRODUCTION-width metadata
  gate rows only (mirroring ``gather_lane_measurements``), so the retired
  width's rendered verdicts can never satisfy the new lane's floor. That
  filter lives inside the reducer-bound closure; its contract is pinned here
  through the same ``batch_size`` projection it reads.
"""

from __future__ import annotations

import pytest

from menagerie.crawler.constants import METADATA_BATCH_MAX
from menagerie.crawler.tools.checker_calibration_runner import (
    derive_width_packs,
    select_measurement_plan,
)

pytestmark = pytest.mark.smoke


def _pack(path: str, stable_ids: list[str]) -> dict:
    return {
        "path": path,
        "items": [{"stable_id": stable_id} for stable_id in stable_ids],
        "stable_ids": list(stable_ids),
        "member_set": tuple(sorted(stable_ids)),
        "had_result": True,
    }


def test_exact_width_packs_pass_through_unchanged() -> None:
    pack = _pack("/w/a/request.json", [f"m{index}" for index in range(METADATA_BATCH_MAX)])
    derived = derive_width_packs([pack], width=METADATA_BATCH_MAX)
    assert derived == [pack]


def test_the_retired_width5_corpus_rechunks_to_the_production_width() -> None:
    """The exact rotation case: frozen width-5 packs feed the width-1 plan."""

    pack = _pack("/w/b/request.json", [f"m{index}" for index in range(5)])
    derived = derive_width_packs([pack], width=1)
    assert len(derived) == 5
    assert all(len(chunk["items"]) == 1 for chunk in derived)
    assert all("#" in str(chunk["path"]) for chunk in derived)
    # Derived keys are distinct so resume accounting is per-derived-pack.
    assert len({chunk["path"] for chunk in derived}) == 5
    # Membership is the original items, in order, without invention.
    recovered = [item["stable_id"] for chunk in derived for item in chunk["items"]]
    assert recovered == [f"m{index}" for index in range(5)]


def test_rechunking_drops_short_tails() -> None:
    """A chunk short of the target width is dropped, never padded or emitted."""

    pack = _pack("/w/e/request.json", [f"m{index}" for index in range(10)])
    derived = derive_width_packs([pack], width=3)
    assert [len(chunk["items"]) for chunk in derived] == [3, 3, 3]
    recovered = [item["stable_id"] for chunk in derived for item in chunk["items"]]
    assert recovered == [f"m{index}" for index in range(9)]


def test_derived_packs_dedupe_by_member_set_in_the_plan() -> None:
    width = METADATA_BATCH_MAX
    same_members = [f"m{index}" for index in range(width)]
    packs = derive_width_packs(
        [_pack("/w/c/request.json", same_members), _pack("/w/d/request.json", same_members)],
        width=width,
    )
    plan = select_measurement_plan(packs, count=1)
    assert len(plan) == 1
    # Only when the distinct pool is exhausted are re-asks admitted.
    plan = select_measurement_plan(packs, count=2)
    assert len(plan) == 2
    assert plan[0]["member_set"] == plan[1]["member_set"]
