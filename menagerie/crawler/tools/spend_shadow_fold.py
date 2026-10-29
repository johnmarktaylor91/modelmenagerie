"""B2-X-V9: shadow-fold a finished rung's spend ledger through the NEXT rung's fuses.

The pre-launch money rehearsal the 200a method never ran: it validated the
LEDGER exhaustively and never evaluated a LIMIT at production width. This tool
replays the previous rung's ACTUAL spend ledger -- through the real accounting
fold, never a proxy -- and compares each lane's real consumption against the
fuse constants the NEXT rung will launch under. Any fuse the last honest rung
consumed more than the refusal fraction of (default 80%) REFUSES LAUNCH with a
typed line and a nonzero exit: a fuse that barely contained the previous rung
is not headroom, it is a scheduled mid-rung halt.

Scope, stated (the broad-mandate rule): the two lane RUNAWAY fuses on both
axes (sessions, wall), for the closed ``LANES`` set. Deliberately excluded:
per-member QUOTA allowances (200b mints fresh work generations, so prior-rung
per-member consumption does not carry) and grant-authorized capacity vs the
fuse (D-4 (b) owns that census). Reserved-hold domination is a separate,
width-derived property owned by ``test_lane_fuse_domination.py``.

Usage::

    python -m menagerie.crawler.tools.spend_shadow_fold \
        --ledger <records>/operational/spend.jsonl \
        --members 210 [--wall-grant-seconds 1800] [--refusal-fraction 0.8]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from menagerie.crawler.constants import (
    CHECKER_LEG_SESSION_BUDGET,
    CHECKER_LEG_WALL_BUDGET_SECONDS,
    author_leg_runaway_budget,
)
from menagerie.crawler.paid_session_accounting import (
    AccountingState,
    LANES,
    Weight,
)
from menagerie.crawler.paid_session_durability import preserved_lineage_verifier

__all__ = ["FuseVerdict", "shadow_fold_verdicts", "main"]

#: Consuming more than this fraction of any next-rung fuse refuses launch.
DEFAULT_REFUSAL_FRACTION = 0.8


@dataclass(frozen=True)
class FuseVerdict:
    """One (lane, axis) fuse compared against the previous rung's consumption."""

    lane: str
    axis: str
    consumed: float
    fuse: float
    refusal_fraction: float

    @property
    def fraction(self) -> float:
        """Return the consumed fraction of the fuse."""

        return self.consumed / self.fuse if self.fuse else float("inf")

    @property
    def refused(self) -> bool:
        """Return whether this fuse refuses the launch."""

        return self.fraction > self.refusal_fraction


def _fold_ledger(ledger: Path) -> AccountingState:
    """Fold one spend ledger through the real accounting state, fail-closed."""

    lineage = preserved_lineage_verifier(
        ledger.parent, ledger_name=ledger.name, subject="paid-session"
    )
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line]
    return AccountingState(lineage_verifier=lineage).extend(rows)


def _next_rung_fuses(
    lane: str, member_count: int, wall_grant_seconds: float
) -> tuple[int, float]:
    """Return the next rung's ``(sessions, wall_seconds)`` fuse for one lane."""

    if lane == "author":
        return author_leg_runaway_budget(member_count, wall_grant_seconds)
    if lane == "checker":
        return CHECKER_LEG_SESSION_BUDGET, float(CHECKER_LEG_WALL_BUDGET_SECONDS)
    raise ValueError(f"lane {lane!r} has no registered fuse derivation")


def shadow_fold_verdicts(
    state: AccountingState,
    *,
    member_count: int,
    wall_grant_seconds: float,
    refusal_fraction: float = DEFAULT_REFUSAL_FRACTION,
) -> tuple[FuseVerdict, ...]:
    """Compare every lane's folded consumption against the next rung's fuses.

    Consumption is summed across every generation in the folded ledger for
    the lane (a rung that spanned several paid generations still consumed the
    whole number against ONE rung's worth of fuse epochs; summing is the
    conservative direction).

    Parameters
    ----------
    state:
        Folded accounting state of the finished rung's ledger.
    member_count:
        The NEXT rung's frozen intake size (fuse derivation input).
    wall_grant_seconds:
        The next rung's per-session author wall grant.
    refusal_fraction:
        Fraction of any fuse above which launch refuses.

    Returns
    -------
    tuple[FuseVerdict, ...]
        One verdict per (lane, axis), every lane in the closed set present.
    """

    consumed: dict[str, Weight] = {lane: Weight() for lane in LANES}
    for (_campaign, _generation, lane), weight in state.runaway_by_lane.items():
        if lane not in consumed:
            raise ValueError(f"ledger names a lane outside the closed set: {lane!r}")
        consumed[lane] = Weight(
            consumed[lane].wall_seconds + weight.wall_seconds,
            consumed[lane].sessions + weight.sessions,
        )
    verdicts: list[FuseVerdict] = []
    for lane in sorted(LANES):
        sessions_fuse, wall_fuse = _next_rung_fuses(lane, member_count, wall_grant_seconds)
        verdicts.append(
            FuseVerdict(
                lane=lane,
                axis="sessions",
                consumed=float(consumed[lane].sessions),
                fuse=float(sessions_fuse),
                refusal_fraction=refusal_fraction,
            )
        )
        verdicts.append(
            FuseVerdict(
                lane=lane,
                axis="wall_seconds",
                consumed=consumed[lane].wall_seconds,
                fuse=wall_fuse,
                refusal_fraction=refusal_fraction,
            )
        )
    return tuple(verdicts)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shadow fold and print one typed verdict line per fuse."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--members", type=int, required=True,
                        help="next rung's frozen intake member count")
    parser.add_argument("--wall-grant-seconds", type=float, default=1800.0)
    parser.add_argument("--refusal-fraction", type=float,
                        default=DEFAULT_REFUSAL_FRACTION)
    args = parser.parse_args(argv)

    state = _fold_ledger(args.ledger)
    verdicts = shadow_fold_verdicts(
        state,
        member_count=args.members,
        wall_grant_seconds=args.wall_grant_seconds,
        refusal_fraction=args.refusal_fraction,
    )
    refused = False
    for verdict in verdicts:
        status = "REFUSES-LAUNCH" if verdict.refused else "ok"
        refused = refused or verdict.refused
        print(
            "shadow-fold %s lane=%s axis=%s consumed=%.1f fuse=%.1f "
            "fraction=%.3f threshold=%.2f"
            % (status, verdict.lane, verdict.axis, verdict.consumed,
               verdict.fuse, verdict.fraction, verdict.refusal_fraction)
        )
    if refused:
        print("shadow-fold verdict: REFUSES LAUNCH (a fuse the previous rung "
              "consumed past the threshold would halt the next rung mid-leg)")
        return 1
    print("shadow-fold verdict: pass")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
