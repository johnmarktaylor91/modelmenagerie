"""W26/TE-23: K distinct models failing the same machine way => lane halt.

The 2026-08-20 storm shape no 200a instrument recognized: per-model valves
each saw one death and kept buying; the money fuse tripped only after tens of
charged hours. The cross-model fuse recognizes the correlation itself and
refuses the NEXT author buy typed, retryable, campaign-scoped -- zero model
terminals, nothing minted. The e2e storm-pack arm runs in R4's V4 rehearsal;
this file pins the predicate matrix and the admission/settlement wiring.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable

import pytest

from menagerie.crawler.driver_admission import AdmissionEnvironmentMixin
from menagerie.crawler.driver_contracts import RetryableOperatorError
from menagerie.crawler.storm_fuse import (
    CROSS_MODEL_STORM_CLASSIFICATIONS,
    CROSS_MODEL_STORM_DISTINCT_MODELS,
    CROSS_MODEL_STORM_WINDOW_SECONDS,
    CrossModelProviderStorm,
    CrossModelStormFuse,
)

pytestmark = pytest.mark.smoke

API_ERROR = "failed-harness-api-error"


def _observe_deaths(fuse: CrossModelStormFuse, count: int, *, at: float,
                    classification: str = API_ERROR, prefix: str = "m") -> None:
    """Record ``count`` distinct-model author deaths of one classification."""

    for index in range(count):
        fuse.observe(
            stable_id=f"{prefix}{index}", lane="author",
            local_outcome=classification, at=at,
        )


def test_k_distinct_models_same_classification_trips() -> None:
    """Exactly K distinct models inside the window is the storm."""

    fuse = CrossModelStormFuse()
    _observe_deaths(fuse, CROSS_MODEL_STORM_DISTINCT_MODELS - 1, at=100.0)
    assert fuse.tripped(101.0) is None
    fuse.observe(stable_id="m_last", lane="author", local_outcome=API_ERROR, at=102.0)
    witness = fuse.tripped(103.0)
    assert witness is not None
    assert witness.classification == API_ERROR
    assert len(witness.stable_ids) == CROSS_MODEL_STORM_DISTINCT_MODELS
    assert "distinct models" in witness.describe()


def test_one_pathological_model_never_trips_alone() -> None:
    """K deaths of ONE model are one distinct model -- the false-trip guard."""

    fuse = CrossModelStormFuse()
    for index in range(CROSS_MODEL_STORM_DISTINCT_MODELS * 3):
        fuse.observe(
            stable_id="m_pathological", lane="author",
            local_outcome=API_ERROR, at=100.0 + index,
        )
    assert fuse.tripped(200.0) is None


def test_mixed_classifications_do_not_compose_into_a_storm() -> None:
    """K distinct models across DIFFERENT classifications is not the predicate."""

    fuse = CrossModelStormFuse()
    for index, classification in enumerate(sorted(CROSS_MODEL_STORM_CLASSIFICATIONS)):
        fuse.observe(
            stable_id=f"m{index}", lane="author",
            local_outcome=classification, at=100.0,
        )
    assert fuse.tripped(101.0) is None


def test_the_window_slides_and_the_lane_reopens() -> None:
    """Observations age out; a tripped lane self-expires as the window drains."""

    fuse = CrossModelStormFuse()
    _observe_deaths(fuse, CROSS_MODEL_STORM_DISTINCT_MODELS, at=100.0)
    assert fuse.tripped(100.0 + CROSS_MODEL_STORM_WINDOW_SECONDS - 1.0) is not None
    assert fuse.tripped(100.0 + CROSS_MODEL_STORM_WINDOW_SECONDS + 1.0) is None


def test_only_author_lane_machine_classifications_count() -> None:
    """Checker weather, content failures, and completions are out of scope."""

    fuse = CrossModelStormFuse()
    for index in range(CROSS_MODEL_STORM_DISTINCT_MODELS):
        fuse.observe(stable_id=f"c{index}", lane="checker",
                     local_outcome=API_ERROR, at=100.0)
        fuse.observe(stable_id=f"f{index}", lane="author",
                     local_outcome="failed", at=100.0)
        fuse.observe(stable_id=f"ok{index}", lane="author",
                     local_outcome="completed", at=100.0)
        fuse.observe(stable_id=f"t{index}", lane="author",
                     local_outcome="campaign-teardown", at=100.0)
    assert fuse.tripped(101.0) is None


def test_the_countable_set_is_closed_and_author_shaped() -> None:
    """The fuse's classification set stays inside the refund vocabulary."""

    from menagerie.crawler.paid_session_accounting import ATTEMPT_REFUND_LOCAL_OUTCOMES

    assert CROSS_MODEL_STORM_CLASSIFICATIONS < ATTEMPT_REFUND_LOCAL_OUTCOMES
    assert "failed-checker-session-wall" not in CROSS_MODEL_STORM_CLASSIFICATIONS
    assert "campaign-teardown" not in CROSS_MODEL_STORM_CLASSIFICATIONS


def test_the_halt_is_typed_retryable_and_campaign_scoped() -> None:
    """The storm halt rides the loud resumable ladder, never a model terminal."""

    assert issubclass(CrossModelProviderStorm, RetryableOperatorError)


def _loaded_names(function: Callable[..., Any]) -> set[str]:
    """Return every identifier one production function's source loads."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def test_admission_checks_the_fuse_before_any_money_row() -> None:
    """Author admission consults the fuse and raises the typed storm halt."""

    loaded = _loaded_names(AdmissionEnvironmentMixin._admit_author_spend)
    assert "CrossModelProviderStorm" in loaded


def test_the_retry_ladder_never_grinds_a_storm_halt_into_a_terminal() -> None:
    """A mid-attempt storm trip re-raises immediately, like the leg breaker.

    ``_is_infrastructure_error`` is True for every ``RetryableOperatorError``,
    so without the early-raise arm the bounded retry ladder would back off
    against a window fuse it cannot drain and exhaust into a model-scoped
    ``AuthorSessionRetryExhausted`` -- the exact terminal the halt exists to
    prevent.
    """

    source = textwrap.dedent(
        inspect.getsource(AdmissionEnvironmentMixin._retry_infrastructure_call)
    )
    breaker_arm = source.find("CrossModelProviderStorm")
    assert breaker_arm != -1
    # The early raise sits BEFORE the infrastructure-retry classification.
    retry_classification = source.find("if not self._is_infrastructure_error(exc)")
    assert retry_classification != -1
    assert breaker_arm < retry_classification


def test_settlement_feeds_the_fuse_after_the_durable_settle() -> None:
    """The observe side rides the one author settle-with-outcome seam."""

    loaded = _loaded_names(AdmissionEnvironmentMixin._settle_paid_session)
    assert "storm_model" in loaded and "resolved_outcome" in loaded
