"""W26/TE-23: the correlated CROSS-MODEL provider-storm fuse.

The 200a author lane had per-model valves (attempt allowances, refund bounds)
and lane-wide MONEY fuses (QUOTA/RUNAWAY), but nothing that recognized the
actual 2026-08-20 storm shape: MANY DISTINCT models dying the SAME
machine-classified way in a short window -- the org-wide provider storm.
Per-model instruments each see one death and keep buying; the money fuse trips
only after tens of hours of charged wall (~150 sessions and ~38 models'
opportunities burned before the 08-20 halt). This fuse recognizes the
correlation itself: K DISTINCT models settling paid author sessions with the
SAME closed machine infrastructure classification inside one window is lane
weather, whatever each individual death looked like.

Distinct from DX-61's identical-error storm breaker by predicate: DX-61 keys
on K identical provider error BODIES (one retired model's echo); this fuse
keys on K distinct MODELS x one closed machine classification (the org-wide
storm). Composes with W1's deferral valve (Wave 2) and AL-9's storm alarm.

Scope, stated: the fuse observes AUTHOR-lane paid-session settlements whose
``local_outcome`` is one of the closed machine infrastructure refund classes
(:data:`CROSS_MODEL_STORM_CLASSIFICATIONS`). Deliberately excluded: bare
``failed`` (model-content conduct -- a content storm is W5's
publication-drought lane-health question, WS3), checker-lane weather (its own
bounded park ladder exists), and ``campaign-teardown`` (a teardown is not
weather). Requiring DISTINCT models is the single-pathological-model
false-trip guard.

On trip, author ADMISSION refuses with the typed retryable
:class:`CrossModelProviderStorm` BEFORE any money row is minted: the lane
halts loudly and resumably at the campaign scope (the same
``RetryableOperatorError`` ladder every lane-weather stop rides), affected
models stay UNRESOLVED with ZERO terminals, and the halt self-expires as the
window slides (a fresh leg or a drained window re-opens the lane bounded).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from menagerie.crawler.constants import (
    TIER_CAMPAIGN_AUTHOR_MODELS,
    resolve_author_wall_seconds,
)
from menagerie.crawler.driver_contracts import RetryableOperatorError
from menagerie.crawler.identity import stable_hash

__all__ = [
    "CROSS_MODEL_STORM_CLASSIFICATIONS",
    "CROSS_MODEL_STORM_DISTINCT_MODELS",
    "CROSS_MODEL_STORM_WINDOW_SECONDS",
    "CrossModelProviderStorm",
    "CrossModelStormFuse",
    "CrossModelStormWitness",
    "FuseTable",
    "derive_fuse_table",
]

#: Distinct models that must fail the SAME way inside the window to trip.
#: Five is the tri-lab shape (a 3-of-5 circuit would have stopped the 08-20
#: storm after ~5 deaths instead of ~150 sessions); requiring five DISTINCT
#: stable ids means no single incorrigible model can trip lane weather alone.
CROSS_MODEL_STORM_DISTINCT_MODELS = 5

#: Sliding window. Storm-era failed sessions burned real 10-21 minute walls
#: (ledger census; the chronicle's "near-instant" impression was wrong), so a
#: 12-wide storming lane puts well over five distinct machine-classified
#: deaths inside 30 minutes, while 200a's healthy legs averaged well under one
#: machine-classified author death per hour.
CROSS_MODEL_STORM_WINDOW_SECONDS = 30.0 * 60.0

#: The closed author-lane machine infrastructure classifications the fuse
#: counts -- the ATTEMPT_REFUND vocabulary minus the checker-lane wall class
#: and campaign teardown (neither is author-lane weather). Widening this set
#: is a reviewed change: every addition can only make the lane halt MORE
#: eagerly, but it also widens what counts as "the same way".
CROSS_MODEL_STORM_CLASSIFICATIONS = frozenset(
    {
        "failed-author-queue-stalled",
        "failed-custody-reaped",
        # Item 9 (recovery F1): the R6b shape -- 52 distinct models dying
        # exit 78 in ~2 s each -- was invisible to this fuse because the deaths
        # settled bare ``failed``. Five distinct typed executor outages inside
        # one window now HALT the lane instead of burning 52 allowances.
        "failed-executor-typed-outage",
        "failed-harness-aborted-streaming-sigterm",
        "failed-harness-api-error",
        "failed-machine-spawn",
    }
)


@dataclass(frozen=True)
class FuseTable:
    """One tier campaign's derived storm-fuse parameters (W32 rider, F-O23).

    Session-shape economics are a function of the AUTHOR MODEL TIER: a
    mid-rung tier change (an Opus fallback) changes wall grants and therefore
    what a storm looks like inside a fixed window. The table is DERIVED —
    never hand-set per tier — composing with D-4's wall-bound derivation:
    the window covers at least three tier session walls and never narrows
    below the reviewed constant. Staging re-derives the table and refuses a
    tier change whose committed toolchain attestation was not re-minted.
    """

    campaign_id: Optional[str]
    author_model: Optional[str]
    wall_seconds: float
    distinct_models: int
    window_seconds: float

    @property
    def digest(self) -> str:
        """Return the content digest the toolchain attestation binds."""

        return stable_hash(
            {
                "campaign_id": self.campaign_id,
                "author_model": self.author_model,
                "wall_seconds": self.wall_seconds,
                "distinct_models": self.distinct_models,
                "window_seconds": self.window_seconds,
            }
        )


def derive_fuse_table(campaign_id: Optional[str]) -> FuseTable:
    """Derive one campaign's fuse table from its tier's wall grant.

    Parameters
    ----------
    campaign_id:
        Frozen tier campaign, or ``None`` outside a tier campaign (the
        reviewed constants apply unchanged).
    """

    author_model = (
        TIER_CAMPAIGN_AUTHOR_MODELS.get(campaign_id) if campaign_id is not None else None
    )
    wall_seconds = resolve_author_wall_seconds(campaign_id)
    window_seconds = (
        max(CROSS_MODEL_STORM_WINDOW_SECONDS, 3.0 * wall_seconds)
        if campaign_id is not None
        else CROSS_MODEL_STORM_WINDOW_SECONDS
    )
    return FuseTable(
        campaign_id=campaign_id,
        author_model=author_model,
        wall_seconds=wall_seconds,
        distinct_models=CROSS_MODEL_STORM_DISTINCT_MODELS,
        window_seconds=window_seconds,
    )


class CrossModelProviderStorm(RetryableOperatorError):
    """Typed, retryable, campaign-scoped lane halt: an org-wide provider storm.

    Raised at author admission BEFORE a reservation is minted, so a tripped
    lane buys nothing and terminalizes nobody; the campaign pauses resumably
    through the same loud ladder every ``RetryableOperatorError`` rides.
    """


@dataclass(frozen=True)
class CrossModelStormWitness:
    """The disclosure a tripped fuse carries: who failed how, inside when.

    Attributes
    ----------
    classification:
        The one closed machine classification every counted death shares.
    stable_ids:
        Sorted distinct models that died that way inside the window.
    window_seconds:
        The sliding window the correlation was observed in.
    """

    classification: str
    stable_ids: tuple[str, ...]
    window_seconds: float

    def describe(self) -> str:
        """Return the typed one-line disclosure for the halt message."""

        return (
            f"{len(self.stable_ids)} distinct models settled paid author sessions "
            f"with the same machine classification {self.classification!r} within "
            f"{self.window_seconds:.0f}s: {', '.join(self.stable_ids)}"
        )


@dataclass
class CrossModelStormFuse:
    """Windowed cross-model correlation witness over author session deaths.

    In-memory and driver-scoped by design: a restart clears it, and a still
    raging storm re-arms it within one window from real observations. Money
    truth is untouched -- the fuse never writes a row and never changes a
    charge; it only refuses the NEXT buy while the correlation stands.
    """

    distinct_models: int = CROSS_MODEL_STORM_DISTINCT_MODELS
    window_seconds: float = CROSS_MODEL_STORM_WINDOW_SECONDS
    _observations: dict[str, dict[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, *, stable_id: str, lane: str, local_outcome: str, at: float) -> None:
        """Record one settled paid-session outcome, if the fuse counts it.

        Parameters
        ----------
        stable_id:
            The model whose session settled.
        lane:
            The paid lane; only ``author`` observations count.
        local_outcome:
            The settlement's machine-observed local disposition.
        at:
            Writer-monotonic observation time (seconds).
        """

        if lane != "author" or local_outcome not in CROSS_MODEL_STORM_CLASSIFICATIONS:
            return
        with self._lock:
            per_model = self._observations.setdefault(local_outcome, {})
            # Latest death per (classification, model): a model dying twice is
            # still ONE distinct model.
            per_model[stable_id] = max(at, per_model.get(stable_id, float("-inf")))

    def tripped(self, at: float) -> "CrossModelStormWitness | None":
        """Return the storm witness while any classification's window holds K.

        Parameters
        ----------
        at:
            Writer-monotonic evaluation time (seconds).

        Returns
        -------
        CrossModelStormWitness | None
            The correlation disclosure, or ``None`` when the lane is open.
        """

        floor = at - self.window_seconds
        with self._lock:
            for classification, per_model in self._observations.items():
                stale = [sid for sid, seen in per_model.items() if seen < floor]
                for sid in stale:
                    del per_model[sid]
                if len(per_model) >= self.distinct_models:
                    return CrossModelStormWitness(
                        classification=classification,
                        stable_ids=tuple(sorted(per_model)),
                        window_seconds=self.window_seconds,
                    )
        return None
