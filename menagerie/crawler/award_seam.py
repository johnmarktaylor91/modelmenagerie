"""AT-6: ONE closed award-seam exception classifier for BOTH publication lanes.

Two lanes spend artifact chains: the scheduler AWARD WORKER
(``driver_receipts._award_worker_result``) and the TERMINAL-FINALIZATION lane
(``driver._authorize_terminal_artifact`` via ``_record_terminal``) -- m12646's
12:42Z publication ran on the terminal-finalization path, and the two lanes'
exception handling had no shared authority. This module is that authority: a
CLOSED exception -> disposition map, walked in declaration order, consumed by
both lanes' disclosure events (which also stamp the AT-Q9 ``publication_lane``
field) and pinned by TW-EXHAUSTIVE-AWARD (an unmapped exception class fails
the test; at runtime an unmapped class disclosed at a seam stamps the loud
``unclassified-award-seam-exception`` value and never silently classifies).

Dispositions (closed):

- ``spent-chain-cache-miss`` -- the cached artifact chain lawfully cannot
  serve (spent or otherwise transition-refused): invalidate loudly, route to
  fresh authorship (W9's composed retirement authority is the invalidation).
- ``pre-commit-refusal-park`` -- the reducer refused BEFORE committing
  anything (``PublicationAuthorizationRefused``): nothing durable happened;
  model-scoped terminal with human review.
- ``post-commit-resume`` -- a physical write failed AFTER the authorization
  committed: a torn chain with a committed capability, owned by the resume
  protocol (AT-7), never a model terminal.
- ``store-integrity-campaign-stop`` -- shared-store corruption (checkpoint or
  mirror integrity): propagate; silently re-authoring the campaign on top of
  a corrupt store is worse than stopping.
- ``append-refusal-award-blocked`` -- the reducer's typed append refusal
  (``ReductionError``): pre-spend it terminalizes model-scoped; when it
  manifests only AFTER the chain is spent, the member parks typed
  ``award-blocked:spent-chain`` instead of pairing a published final event
  with a failure head (m12646's ``published_with_failure_head`` shape).
"""

from __future__ import annotations

from typing import Any, Mapping

from menagerie.crawler.artifact_transactions import (
    ArtifactChainSpentError,
    ArtifactCheckpointError,
    ArtifactPublicationError,
    ArtifactTransitionError,
)
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.driver_contracts import PublicationAuthorizationRefused
from menagerie.crawler.mirrors import MirrorError
from menagerie.crawler.models import JsonObject
from menagerie.crawler.reducer import ReductionError

__all__ = [
    "AWARD_SEAM_LANES",
    "AWARD_SEAM_DISPOSITIONS",
    "AWARD_SEAM_UNCLASSIFIED",
    "AwardSeamClassificationError",
    "classify_award_seam_exception",
    "disclosed_award_seam_classification",
    "award_seam_refusal_event",
]

AWARD_SEAM_LANES = frozenset({"award-worker", "terminal-finalization"})
"""Closed publication-lane vocabulary (AT-Q9 disclosure field)."""

AWARD_SEAM_DISPOSITIONS = frozenset(
    {
        "spent-chain-cache-miss",
        "pre-commit-refusal-park",
        "post-commit-resume",
        "store-integrity-campaign-stop",
        "append-refusal-award-blocked",
    }
)
"""Closed award-seam disposition vocabulary; exactly one per exception class."""

AWARD_SEAM_UNCLASSIFIED = "unclassified-award-seam-exception"
"""Loud disclosure value for a class outside the closed map (never a route)."""


class AwardSeamClassificationError(RuntimeError):
    """An award-seam exception class is outside the closed classification map."""


#: THE closed map, walked in declaration order (subclasses before parents).
#: Extending it is a reviewed vocabulary change pinned by
#: TW-EXHAUSTIVE-AWARD in ``tests/test_award_batch_at456.py``.
_CLASSIFICATION_ORDER: tuple[tuple[type[BaseException], str], ...] = (
    (ArtifactChainSpentError, "spent-chain-cache-miss"),
    (ArtifactCheckpointError, "store-integrity-campaign-stop"),
    (ArtifactPublicationError, "store-integrity-campaign-stop"),
    (ArtifactTransitionError, "spent-chain-cache-miss"),
    (PublicationAuthorizationRefused, "pre-commit-refusal-park"),
    (MirrorError, "store-integrity-campaign-stop"),
    (ReductionError, "append-refusal-award-blocked"),
    (OSError, "post-commit-resume"),
)


def classify_award_seam_exception(exc: BaseException) -> str:
    """Return the single closed disposition for one award-seam exception.

    Raises
    ------
    AwardSeamClassificationError
        For a class outside the closed map -- fail-closed: the caller either
        propagates (campaign stop) or discloses
        :data:`AWARD_SEAM_UNCLASSIFIED` loudly, never routes on a guess.
    """

    for exception_type, disposition in _CLASSIFICATION_ORDER:
        if isinstance(exc, exception_type):
            return disposition
    raise AwardSeamClassificationError(
        f"award-seam exception class {type(exc).__module__}.{type(exc).__qualname__} "
        "is outside the closed classification map"
    )


def disclosed_award_seam_classification(exc: BaseException) -> str:
    """Return the closed classification, or the loud unclassified value.

    Disclosure-only: routing never consumes this fallback; a novel class at a
    seam is disclosed as :data:`AWARD_SEAM_UNCLASSIFIED` and the strict
    classifier (plus TW-EXHAUSTIVE-AWARD) is what keeps the map total.
    """

    try:
        return classify_award_seam_exception(exc)
    except AwardSeamClassificationError:
        return AWARD_SEAM_UNCLASSIFIED


def award_seam_refusal_event(
    *,
    publication_lane: str,
    stable_id: str,
    work_id: str,
    failure: BaseException,
    run_id: str,
    machine_id: str,
    created_at: str,
    extra_details: Mapping[str, Any] | None = None,
) -> JsonObject:
    """Build one honest award-seam refusal disclosure event (unminted).

    DX-36's honest-status rule: an award-seam refusal event is DEGRADED,
    never healthy. The event stamps the AT-Q9 ``publication_lane`` field and
    the classifier's disposition; the caller mints and appends it.
    """

    if publication_lane not in AWARD_SEAM_LANES:
        raise ValueError(f"unknown publication lane: {publication_lane!r}")
    details: JsonObject = {
        "disposition": "award-seam-refusal",
        "publication_lane": publication_lane,
        "award_seam_classification": disclosed_award_seam_classification(failure),
        "stable_id": stable_id,
        "work_id": work_id,
        "error_type": f"{type(failure).__module__}.{type(failure).__qualname__}",
        "error": str(failure)[:1_500],
    }
    if extra_details:
        details.update(dict(extra_details))
    return {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": created_at,
        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
        "status": OperationalEventStatus.DEGRADED.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": run_id,
        "machine_id": machine_id,
        "details": details,
    }
