"""D-2: every paid lane's wall fuse dominates its own maximum reserved hold.

The 200a checker-lane wedge (the sharpest money finding of the tri-lab audit):
a checker reservation holds its full ceiling against the leg's RUNAWAY wall
fuse until it settles, so ``concurrency x reservation_ceiling`` of PHANTOM wall
accumulates with zero real spend. At the configured checker width the borrowed
30h fuse was crossed by holds alone at ~17 concurrent reservations on a fresh
epoch -- a lane that succeeds hard enough wedges itself ("success makes it
fail"). 200a never saw it only because its outcome starved the checker lane to
~2 concurrent sessions.

The rule this file pins, for EVERY lane in the closed ``LANES`` set: the lane's
minimum wall fuse must exceed its maximum possible reserved hold
(``max configured concurrency x its own reservation ceiling``) by at least the
historical real-wall runaway bound (30h), so

* reserved holds ALONE can never trip the fuse at any lawful width, and
* the fuse still trips at exactly the historical bound of REAL settled wall
  even under a maximal hold -- the raise RESTORES the breaker's semantics
  (bind on real runaway, never on configured concurrency); it does not loosen
  them.

Replay evidence for "holds count against the fuse by design" (kept, not
removed): the 2026-08-20 author halt's first predicate crossing at
2026-08-20T11:57:44.731858Z carried 11 in-flight 1983s holds (21.5% of the
folded accumulator) -- reserved-but-unsettled wall deliberately counts, and
THIS file's domination margin is the sizing rule that makes that conservatism
safe (WS2 first-task (c) replay, .lane-logs/ft_c_instants.txt).

MC-Q9 / PD-18 width pin: 200b runs the proven configuration -- author 12,
checker 24 (BL-11's measured campaign widths). The pin is ratified HERE, in
the same change that makes the fuse dominate even the 32-wide maximum, so no
width inside the legal range can wedge its lane.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable

import pytest

from menagerie.crawler.constants import (
    CHECKER_LEG_SESSION_BUDGET,
    CHECKER_LEG_WALL_BUDGET_SECONDS,
    DEFAULT_AUTHOR_WAVE_CONCURRENCY,
    DEFAULT_CHECKER_WAVE_CONCURRENCY,
    MAX_AUTHOR_WAVE_CONCURRENCY,
    MAX_CHECKER_WAVE_CONCURRENCY,
    author_leg_runaway_budget,
)
from menagerie.crawler.driver_admission import (
    CHECKER_LANE_BACKOFF_SLACK_SECONDS,
    CHECKER_LANE_WALL_GRACE_SECONDS,
    AdmissionEnvironmentMixin,
)
from menagerie.crawler.operator_protocol import (
    OPERATOR_DEADLINE_SECONDS,
    checker_session_kill_seconds,
)
from menagerie.crawler.paid_session_accounting import LANES

pytestmark = pytest.mark.smoke

#: The historical real-wall runaway bound: strictly more REAL settled wall in
#: one leg than the entire 72-model pilot consumed lifetime (28.5h) is a
#: runaway leg, whatever the cause. The domination margin preserves exactly
#: this trip point for real spend under a maximal reserved hold.
REAL_WALL_RUNAWAY_BOUND_SECONDS = 30 * 60 * 60


def _checker_reservation_ceiling_seconds() -> float:
    """Return the exact wall one checker reservation holds until it settles."""

    return (
        checker_session_kill_seconds()
        + CHECKER_LANE_BACKOFF_SLACK_SECONDS
        + CHECKER_LANE_WALL_GRACE_SECONDS
    )


def _author_reservation_ceiling_seconds() -> float:
    """Return the exact wall one author reservation holds until it settles.

    Mirrors the live ``_admit_author_spend`` spelling
    (``reservation_wall_seconds=float(OPERATOR_DEADLINE_SECONDS)``). D-4
    re-derives the author reservation from the author lane's own wall bound;
    when it does, the AST mirror test below fails loudly so this ceiling is
    updated in the same reviewed change instead of drifting.
    """

    return float(OPERATOR_DEADLINE_SECONDS)


def _minimum_wall_fuse_seconds(lane: str) -> float:
    """Return the smallest RUNAWAY wall fuse the lane can run under."""

    if lane == "author":
        # The member-derived fuse floors at the checker constants; the
        # minimum over every intake size is the member_count=0 floor.
        _sessions, wall = author_leg_runaway_budget(0, 1800.0)
        return float(wall)
    if lane == "checker":
        return float(CHECKER_LEG_WALL_BUDGET_SECONDS)
    raise AssertionError(f"lane {lane!r} has no registered fuse row")


_LANE_ROWS: dict[str, dict[str, Any]] = {
    "author": {
        "max_concurrency": MAX_AUTHOR_WAVE_CONCURRENCY,
        "reservation_ceiling": _author_reservation_ceiling_seconds,
        "session_fuse": CHECKER_LEG_SESSION_BUDGET,
    },
    "checker": {
        "max_concurrency": MAX_CHECKER_WAVE_CONCURRENCY,
        "reservation_ceiling": _checker_reservation_ceiling_seconds,
        "session_fuse": CHECKER_LEG_SESSION_BUDGET,
    },
}


def test_lane_fuse_registry_covers_the_closed_lane_set() -> None:
    """A new paid lane cannot ship without registering its fuse row here."""

    assert set(_LANE_ROWS) == set(LANES)


@pytest.mark.parametrize("lane", sorted(LANES))
def test_every_lane_wall_fuse_dominates_its_own_maximum_reserved_hold(lane: str) -> None:
    """Reserved holds alone can never trip any lane's wall fuse."""

    row = _LANE_ROWS[lane]
    max_hold = row["max_concurrency"] * row["reservation_ceiling"]()
    fuse = _minimum_wall_fuse_seconds(lane)
    assert fuse > max_hold, (
        f"{lane} lane wall fuse {fuse:.1f}s does not dominate its own maximum "
        f"reserved hold {max_hold:.1f}s ({row['max_concurrency']} x "
        f"{row['reservation_ceiling']():.1f}s): the lane wedges on phantom "
        f"wall when it is busy enough -- the 200a checker-wedge class"
    )
    # Restoration, not loosening: under a maximal hold the fuse still trips
    # at the historical bound of REAL settled wall.
    assert fuse - max_hold >= REAL_WALL_RUNAWAY_BOUND_SECONDS


@pytest.mark.parametrize("lane", sorted(LANES))
def test_every_lane_session_fuse_dominates_its_own_maximum_session_hold(lane: str) -> None:
    """In-flight reservations alone can never trip any lane's session fuse."""

    row = _LANE_ROWS[lane]
    assert row["session_fuse"] > row["max_concurrency"]


def test_the_borrowed_30h_fuse_wedged_the_checker_lane_at_configured_width() -> None:
    """Regression pin of the wedge arithmetic the raise repaired (U-O1)."""

    ceiling = _checker_reservation_ceiling_seconds()
    # Historical fact: holds alone crossed the borrowed 30h fuse below the
    # CONFIGURED width -- a healthy full-width checker lane could not run.
    assert int(REAL_WALL_RUNAWAY_BOUND_SECONDS // ceiling) < DEFAULT_CHECKER_WAVE_CONCURRENCY
    # Repaired fact: the raised fuse admits the full legal width in holds.
    assert int(_minimum_wall_fuse_seconds("checker") // ceiling) >= MAX_CHECKER_WAVE_CONCURRENCY


def test_checker_width_pin_mc_q9() -> None:
    """PD-18/MC-Q9: 200b runs the proven widths -- author 12, checker 24.

    24 was never script-pinned in 200a (no launch script set
    ``MENAGERIE_CHECKER_CONCURRENCY``); it stands ratified here as BL-11's
    measured campaign width, safe because the fuse now dominates even the
    32-wide maximum. Changing any of these four numbers re-opens this file's
    domination arithmetic in the same reviewed change.
    """

    assert DEFAULT_AUTHOR_WAVE_CONCURRENCY == 12
    assert DEFAULT_CHECKER_WAVE_CONCURRENCY == 24
    assert MAX_AUTHOR_WAVE_CONCURRENCY == 32
    assert MAX_CHECKER_WAVE_CONCURRENCY == 32


def _loaded_names(function: Callable[..., Any]) -> set[str]:
    """Return every identifier one production function's source loads."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def test_reservation_ceiling_mirrors_track_the_live_admission_spellings() -> None:
    """The ceilings this file guards are the ones admission actually reserves.

    If D-4 re-derives the author reservation (or the checker ceiling gains a
    term), the corresponding mirror above must change in the same reviewed
    change -- this assertion is what makes that drift loud instead of silent.
    """

    author_names = _loaded_names(AdmissionEnvironmentMixin._admit_author_spend)
    assert "OPERATOR_DEADLINE_SECONDS" in author_names, (
        "author admission no longer reserves float(OPERATOR_DEADLINE_SECONDS): "
        "update _author_reservation_ceiling_seconds() to the new derivation"
    )
    checker_names = _loaded_names(AdmissionEnvironmentMixin._admit_checker_spend)
    assert {
        "checker_session_kill_seconds",
        "CHECKER_LANE_BACKOFF_SLACK_SECONDS",
        "CHECKER_LANE_WALL_GRACE_SECONDS",
    } <= checker_names, (
        "checker admission's reservation wall changed terms: update "
        "_checker_reservation_ceiling_seconds() to the new derivation"
    )
