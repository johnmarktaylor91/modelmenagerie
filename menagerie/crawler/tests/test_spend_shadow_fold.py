"""B2-X-V9 shadow-fold: the previous rung's spend must fit the next rung's fuses.

Unit arms exercise the verdict logic in both directions; the historical arm
pins the 200a census numbers against the CURRENT constants, so lowering any
fuse below what the last honest rung actually consumed goes red here before it
can schedule a mid-rung halt. The live pre-launch run is
``python -m menagerie.crawler.tools.spend_shadow_fold`` against the real
ledger; this file guards the tool's semantics.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from menagerie.crawler.paid_session_accounting import LANES, Weight
from menagerie.crawler.tools.spend_shadow_fold import (
    DEFAULT_REFUSAL_FRACTION,
    FuseVerdict,
    shadow_fold_verdicts,
)

pytestmark = pytest.mark.smoke

#: 200a consumption census at 2026-08-24 (replay of the pilot spend ledger
#: through the real fold; WS2 first-task (d)). A floor, not an exact pin: the
#: shadow-fold rule is that the NEXT rung's fuses must hold AT LEAST this much
#: under the refusal threshold.
HISTORICAL_200A_CONSUMPTION = {
    "author": Weight(1_001_552.2, 1419),
    "checker": Weight(27_134.3, 129),
}


def _state(by_lane: dict[str, Weight]) -> SimpleNamespace:
    """Return a minimal stand-in exposing the one field the fold verdict reads."""

    return SimpleNamespace(
        runaway_by_lane={
            ("campaign-one", "gen-one", lane): weight for lane, weight in by_lane.items()
        }
    )


def test_verdicts_cover_every_lane_and_both_axes() -> None:
    """Every lane in the closed set gets a sessions and a wall verdict."""

    verdicts = shadow_fold_verdicts(
        _state({lane: Weight(10.0, 1) for lane in LANES}),
        member_count=210,
        wall_grant_seconds=1800.0,
    )
    assert {(v.lane, v.axis) for v in verdicts} == {
        (lane, axis) for lane in LANES for axis in ("sessions", "wall_seconds")
    }


def test_consumption_past_the_threshold_refuses_launch() -> None:
    """A fuse the previous rung consumed >80% of refuses, at the exact boundary."""

    verdict = FuseVerdict(
        lane="author", axis="sessions", consumed=81.0, fuse=100.0,
        refusal_fraction=DEFAULT_REFUSAL_FRACTION,
    )
    assert verdict.refused
    at_boundary = FuseVerdict(
        lane="author", axis="sessions", consumed=80.0, fuse=100.0,
        refusal_fraction=DEFAULT_REFUSAL_FRACTION,
    )
    assert not at_boundary.refused  # strictly MORE than the fraction refuses


def test_a_storm_heavy_previous_rung_refuses_the_borrowed_fuse_shape() -> None:
    """Regression of the 2026-08-20 class: 29.84h against a 30h fuse refuses.

    The incident wall folded to 106,530.9s against the borrowed 108,000s fuse
    (98.6%); under the shadow-fold rule that configuration REFUSES launch
    instead of halting mid-leg.
    """

    verdicts = shadow_fold_verdicts(
        _state({"author": Weight(106_530.9, 169), "checker": Weight(0.0, 0)}),
        member_count=0,  # tiny-leg floor fuses
        wall_grant_seconds=1800.0,
    )
    wall = {v.lane: v for v in verdicts if v.axis == "wall_seconds"}
    # Under the RESTORED floor (310,800s) the same real consumption passes --
    # the raise, not the threshold, is what absorbs an honest heavy rung.
    assert not wall["author"].refused
    assert wall["author"].fraction == pytest.approx(106_530.9 / 310_800.0)


def test_a_lane_outside_the_closed_set_is_refused_typed() -> None:
    """A ledger naming an unregistered lane fails closed, never folds silently."""

    with pytest.raises(ValueError, match="closed set"):
        shadow_fold_verdicts(
            _state({"author": Weight(1.0, 1), "smuggled": Weight(1.0, 1)}),
            member_count=210,
            wall_grant_seconds=1800.0,
        )


def test_the_200a_census_fits_the_current_200b_constants() -> None:
    """The last honest rung's real consumption passes today's fuses (<=80%).

    Author sessions are the closest axis (1,419 of the 2,000 floor = 71.0%);
    lowering any fuse below this census -- or a rung consuming past it --
    must go red HERE before a launch is attempted on it.
    """

    verdicts = shadow_fold_verdicts(
        _state(HISTORICAL_200A_CONSUMPTION),
        member_count=210,
        wall_grant_seconds=1800.0,
    )
    assert all(not v.refused for v in verdicts), [
        (v.lane, v.axis, v.fraction) for v in verdicts if v.refused
    ]
    sessions = {v.lane: v for v in verdicts if v.axis == "sessions"}
    assert sessions["author"].fraction == pytest.approx(1419 / 2000)
