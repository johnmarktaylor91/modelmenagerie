"""Author-session spend derived from append-only history, enforced pre-dispatch.

Docket C-AUTHOR-SPEND (T2-07) and the T1-13 allowance half. The pilot ran 31
work generations over the declared 3-session cap (worst: eleven staged results
on one m8650 work_id) because enforcement was an in-memory per-leg counter a
rehydration reset, and the canonical ``budget.author_sessions_used`` field was
hardcoded to 0/1. Chargeability is now a pure replay of provider-billed
``author-session-bought`` events per exact ``(stable_id, work_id)``; historical
staged results remain a lower-bound fallback, never the counting unit.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Optional

import pytest

from menagerie.crawler import driver as driver_module
from menagerie.crawler import driver_admission, driver_models, driver_receipts
from menagerie.crawler.author_spend import (
    author_session_allowance,
    author_sessions_exhaustion_dimension,
    author_spend_census,
    chargeable_author_sessions,
    enforce_author_session_allowance,
)
from menagerie.crawler.constants import (
    AUTHOR_SESSIONS_MAX,
    EFFORT_EXHAUSTION_REASON_CODES,
    FAILURE_REASON_CODES,
    EnvironmentPhase,
)
from menagerie.crawler.driver_contracts import (
    AuthorEffortCapExceeded,
    DriverConfig,
    IntentRoute,
    WorkItem,
)
from menagerie.crawler.paid_session_accounting import PAID_SESSION_SETTLEMENT_SCHEMA
from menagerie.crawler.driver_models import _assemble_terminal_model
from menagerie.crawler.intake import IntakeItem

STABLE_ID = "m_spend"
WORK_ID = "work-m_spend"


def _staged(stable_id: str, work_id: str, result_id: str) -> dict[str, Any]:
    """Return one minimal staged-private artifact event.

    Parameters
    ----------
    stable_id, work_id, result_id:
        Exact staging identities.

    Returns
    -------
    dict[str, Any]
        Ledger-shaped artifact event.
    """

    return {
        "event_kind": "staged-private",
        "stable_id": stable_id,
        "work_id": work_id,
        "author_result_id": result_id,
    }


def _bought(stable_id: str, work_id: str, ordinal: int) -> dict[str, Any]:
    """Return one minimal provider-billed author-session event.

    Parameters
    ----------
    stable_id, work_id:
        Exact work-generation identity.
    ordinal:
        Monotonic bought-session ordinal.

    Returns
    -------
    dict[str, Any]
        Ledger-shaped operational event.
    """

    return {
        "event_id": f"author-session-{ordinal}",
        "details": {
            "disposition": "author-session-bought",
            "stable_id": stable_id,
            "work_id": work_id,
            "session_ordinal": ordinal,
        },
    }


def _work_item(*, granted_attempts: Optional[int] = None) -> WorkItem:
    """Return one standalone scheduled work item for the harness model.

    Parameters
    ----------
    granted_attempts:
        Optional requeue-granted session extension.

    Returns
    -------
    WorkItem
        Routed item whose ``active_work_id`` is ``work-m_spend``.
    """

    intake = IntakeItem(
        stable_id=STABLE_ID,
        name="SpendNet",
        zoo="test-zoo",
        variant="base",
        discovery_source="crawl_roster",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=(),
        variant_scope="standalone",
        family_representative_id=STABLE_ID,
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(STABLE_ID, "core", EnvironmentPhase.PYTORCH),
        granted_attempts=granted_attempts,
    )


@pytest.mark.smoke
def test_chargeable_sessions_count_distinct_staged_results() -> None:
    """The 2.11x fixture counts bought sessions, not published results."""

    published = [_staged(STABLE_ID, WORK_ID, f"result-{index}") for index in range(100)]
    bought = [_bought(STABLE_ID, WORK_ID, index) for index in range(1, 212)]
    events = [
        *published,
        *bought,
        # Another generation and another model never leak into the count.
        _bought(STABLE_ID, "work-other-generation", 1),
        _bought("m_other", WORK_ID, 1),
        # Non-staging transitions are not sessions.
        {
            "event_kind": "published",
            "stable_id": STABLE_ID,
            "work_id": WORK_ID,
            "author_result_id": "result-0",
        },
    ]
    assert len(bought) / len(published) == 2.11
    assert chargeable_author_sessions(events, STABLE_ID, WORK_ID) == 211


def _reserved(
    spend_id: str, *, attempt_group: Optional[str] = None
) -> dict[str, Any]:
    """Return one minimal merged author paid-ledger reservation row."""

    return {
        "event": "session_reserved",
        "lane": "author",
        "spend_id": spend_id,
        "attempt_group": attempt_group,
        "axis": {
            "work_generations": ({"stable_id": STABLE_ID, "work_id": WORK_ID},)
        },
    }


def _settled(
    spend_id: str, local_outcome: str, *, accepted: bool = True
) -> dict[str, Any]:
    """Return one minimal merged current-schema settlement row."""

    return {
        "event": "session_settled",
        "spend_id": spend_id,
        "schema_version": PAID_SESSION_SETTLEMENT_SCHEMA,
        "local_outcome": local_outcome,
        "provider": {"accepted": accepted},
    }


@pytest.mark.smoke
def test_merged_rows_refund_an_attempt_whose_terminal_session_died_infra() -> None:
    """The merged-row floor shares the fold's attempt-group refund rule.

    The 2026-08-20 refund gap, mirrored at the artifact/operational floor: a
    two-guard attempt whose source-request guard COMPLETED and whose terminal
    session settled with a closed machine infrastructure outcome weighs zero
    attempts, exactly like the paid authority's own projection.  The refund
    keys only on the driver-written settlement outcome, never on any
    producer-supplied claim.
    """

    infra_attempt = [
        _reserved("spend-head"),
        _settled("spend-head", "completed"),
        _reserved("spend-join", attempt_group="spend-head"),
        _settled("spend-join", "failed-harness-api-error"),
    ]
    assert chargeable_author_sessions(infra_attempt, STABLE_ID, WORK_ID) == 0

    # A fully declined group keeps the historical R4 refund.
    declined = [
        _reserved("spend-declined"),
        _settled("spend-declined", "failed", accepted=False),
    ]
    assert chargeable_author_sessions(declined, STABLE_ID, WORK_ID) == 0


@pytest.mark.smoke
def test_merged_rows_still_charge_healthy_session_failures() -> None:
    """A generic failed settlement and an unsettled member stay chargeable.

    The precision half at the merged-row floor: a completed-session contract
    violation settles ``failed`` (no machine infrastructure evidence) and must
    keep consuming; a group with an unsettled member charges conservatively.
    """

    healthy_failure = [
        _reserved("spend-head"),
        _settled("spend-head", "completed"),
        _reserved("spend-join", attempt_group="spend-head"),
        _settled("spend-join", "failed"),
    ]
    assert chargeable_author_sessions(healthy_failure, STABLE_ID, WORK_ID) == 1

    completed_attempt = [
        _reserved("spend-head"),
        _settled("spend-head", "completed"),
        _reserved("spend-join", attempt_group="spend-head"),
        _settled("spend-join", "completed"),
    ]
    assert chargeable_author_sessions(completed_attempt, STABLE_ID, WORK_ID) == 1

    unsettled = [_reserved("spend-head")]
    assert chargeable_author_sessions(unsettled, STABLE_ID, WORK_ID) == 1


@pytest.mark.smoke
def test_allowance_is_base_plus_granted_extension() -> None:
    """A consumed grant EXTENDS its generation by the granted attempts."""

    assert author_session_allowance(None) == AUTHOR_SESSIONS_MAX
    assert author_session_allowance(2) == AUTHOR_SESSIONS_MAX + 2


@pytest.mark.smoke
def test_enforcement_raises_the_typed_author_sessions_cap() -> None:
    """The m8650 shape: a rehydrated leg can never re-buy staged spend."""

    events = [
        _staged(STABLE_ID, WORK_ID, f"result-{index}") for index in range(AUTHOR_SESSIONS_MAX)
    ]
    with pytest.raises(AuthorEffortCapExceeded) as caught:
        enforce_author_session_allowance(events, _work_item())
    assert caught.value.stage == "author"
    assert caught.value.reason_code == "effort-exhausted:author-sessions"
    assert caught.value.reason_code in FAILURE_REASON_CODES["author"]

    # One session below the allowance still dispatches.
    enforce_author_session_allowance(events[:-1], _work_item())
    # A granted extension re-opens exactly the granted headroom.
    enforce_author_session_allowance(events, _work_item(granted_attempts=1))
    with pytest.raises(AuthorEffortCapExceeded):
        enforce_author_session_allowance(
            events + [_staged(STABLE_ID, WORK_ID, "result-extra")],
            _work_item(granted_attempts=1),
        )


@pytest.mark.smoke
def test_staging_an_already_staged_result_is_idempotent_at_the_cap() -> None:
    """The m4851 valve runs at staging too, without re-charging a replayed id.

    Enforcement at the staging chokepoint must not refuse a rehydration replay
    that re-stages an ``author_result_id`` the ledger already holds: the
    distinct-id counting rule says that staging buys nothing new. Only a FRESH
    result id at the cap is an overshoot.
    """

    events = [
        _staged(STABLE_ID, WORK_ID, f"result-{index}") for index in range(AUTHOR_SESSIONS_MAX)
    ]
    # A replayed staging of an already-counted id passes even at the cap.
    enforce_author_session_allowance(events, _work_item(), staging_author_result_id="result-0")
    # A fresh id at the cap is exactly the m4851 overshoot and refuses typed.
    with pytest.raises(AuthorEffortCapExceeded) as caught:
        enforce_author_session_allowance(
            events, _work_item(), staging_author_result_id="result-fresh"
        )
    assert caught.value.reason_code == "effort-exhausted:author-sessions"

    # A session charged pre-dispatch at the last slot may stage its result;
    # staging is not a second purchase.
    backed = events[:-1] + [_bought(STABLE_ID, WORK_ID, AUTHOR_SESSIONS_MAX)]
    enforce_author_session_allowance(
        backed, _work_item(), staging_author_result_id="result-fresh"
    )


@pytest.mark.smoke
def test_superseded_generation_stagings_stay_unchargeable_at_staging() -> None:
    """A supersession observation cannot remint one generation's allowance."""

    events = [
        _staged(STABLE_ID, WORK_ID, f"result-{index}") for index in range(AUTHOR_SESSIONS_MAX)
    ]
    discard = {
        "details": {
            "disposition": "author-cache-discarded",
            "stable_id": STABLE_ID,
            "work_id": WORK_ID,
            "superseded_author_result_ids": [
                f"result-{index}" for index in range(AUTHOR_SESSIONS_MAX)
            ],
        }
    }
    with pytest.raises(AuthorEffortCapExceeded):
        enforce_author_session_allowance(
            events,
            _work_item(),
            operational_events=[discard],
            staging_author_result_id="result-fresh",
        )


@pytest.mark.smoke
def test_the_staging_chokepoint_consults_the_one_allowance_owner() -> None:
    """m4851 staged 6 of 3: ``_stage_author_result`` itself now runs the valve.

    The repair loops (metadata/fidelity, detected-modes, recipe-refusal) each
    stage results under their own bounded budgets without re-consulting the
    author-session valve, which is how m4851 staged double its allowance. The
    static inventory proves the staging chokepoint calls the SAME owner
    function -- never a fork of the arithmetic -- before any byte lands.
    """

    source = inspect.getsource(driver_module.CrawlerDriver._stage_author_result)
    tree = ast.parse(source.lstrip())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "enforce_author_session_allowance"
    ]
    assert calls, "_stage_author_result must consult enforce_author_session_allowance"
    keywords = {keyword.arg for call in calls for keyword in call.keywords}
    assert "staging_author_result_id" in keywords
    assert "operational_events" in keywords


@pytest.mark.smoke
def test_author_spend_census_flags_over_allowance_generations() -> None:
    """The census reads like the pilot adjudication: over-cap pairs, named."""

    events = [_staged("m8650", "work-one", f"result-{index}") for index in range(11)] + [
        _staged(STABLE_ID, WORK_ID, "result-a"),
    ]
    census = author_spend_census(events)
    assert census["work_generations"] == 2
    assert census["sessions_staged"] == 12
    assert census["sessions_bought"] == 0
    assert census["sessions_chargeable"] == 12
    assert census["generations_over_allowance"] == 1
    assert census["over_allowance"] == {"m8650/work-one": 11}


@pytest.mark.smoke
def test_terminal_budget_records_the_derived_spend() -> None:
    """The head's budget block is the replay-derived truth, never a snapshot."""

    model = _assemble_terminal_model(
        _work_item(granted_attempts=2),
        None,
        "failed:author",
        "session-crashed",
        None,
        [],
        [],
        DriverConfig(),
        "2026-08-12T08:00:00Z",
        human_review=False,
        root_cause_fingerprint="fingerprint-spend",
        author_sessions_used=4,
    )
    assert model["budget"]["author_sessions_used"] == 4
    assert model["budget"]["author_sessions_max"] == AUTHOR_SESSIONS_MAX + 2


@pytest.mark.smoke
def test_every_author_buy_site_consults_the_derived_allowance() -> None:
    """The three buy paths bind to the enforcement, and the assemblers to truth.

    Source-binding in the anti-substitution style: deleting any call reverts to
    the unenforced world T2-07 measured, so each binding is pinned.
    """

    commit_wave = inspect.getsource(driver_admission.AdmissionEnvironmentMixin._commit_author_wave)
    assert "enforce_author_session_allowance(" in commit_wave
    assert "reducer.artifact_ledger.events," in commit_wave
    assert "operational_events=operational.records," in commit_wave
    speculative = inspect.getsource(
        driver_admission.AdmissionEnvironmentMixin._author_session_is_certain
    )
    assert "enforce_author_session_allowance(" in speculative
    speculative_tree = ast.parse(speculative.lstrip())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "author_session_allowance"
        for node in ast.walk(speculative_tree)
    )
    repair = inspect.getsource(driver_module.CrawlerDriver._repair_author)
    assert "enforce_author_session_allowance(" in repair
    assert "*operational_events" in repair

    # The assembler bindings are audited over the WHOLE production module rather
    # than over one named writer's source: M3's scheduler refactor extracted the
    # run award out of ``_forward_and_reduce`` into ``_award_worker_result``, and
    # a pin naming the old method reads as "the binding was deleted" when the
    # binding merely MOVED. Auditing every call site is strictly stronger -- it
    # also covers a NEW writer added later -- and it still fails the instant the
    # derived count stops being passed.
    unbound: list[str] = []
    for module in (driver_module, driver_receipts, driver_admission, driver_models):
        source = inspect.getsource(module)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in {"_assemble_run_model", "_assemble_terminal_model"}:
                continue
            derived = any(
                keyword.arg == "author_sessions_used"
                and isinstance(keyword.value, ast.Call)
                and isinstance(keyword.value.func, ast.Name)
                and keyword.value.func.id == "chargeable_author_sessions"
                for keyword in node.keywords
            )
            if not derived:
                unbound.append(f"{module.__name__}:{node.lineno} {node.func.id}")
    assert not unbound, (
        "every production model assembly must pass the replay-derived session"
        f" count; unbound call sites: {unbound}"
    )


# ---------------------------------------------------------------------------
# Provider-storm exhaustion split (2026-08-21): the reason is replay-derived
# from settlement evidence only, and uncertainty never upgrades a terminal.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_majority_infra_settlements_read_provider_storm() -> None:
    """A storm-burned allowance names the storm, not this model's authoring."""

    events = [
        # Two consumed sessions from real attempts.
        _reserved("spend-real-1"),
        _settled("spend-real-1", "completed"),
        _reserved("spend-real-2"),
        _settled("spend-real-2", "failed"),
        # Five storm casualties: provider-charged, machine-classified infra.
        *(
            row
            for index in range(5)
            for row in (
                _reserved(f"spend-storm-{index}"),
                _settled(f"spend-storm-{index}", "failed-harness-api-error"),
            )
        ),
    ]
    assert (
        author_sessions_exhaustion_dimension(events, STABLE_ID, WORK_ID) == "provider-storm"
    )


@pytest.mark.smoke
def test_minority_or_tied_infra_evidence_stays_plain_exhaustion() -> None:
    """The split needs a strict majority; a tie is not a storm."""

    tied = [
        _reserved("spend-real"),
        _settled("spend-real", "completed"),
        _reserved("spend-storm"),
        _settled("spend-storm", "failed-harness-aborted-streaming-sigterm"),
    ]
    assert author_sessions_exhaustion_dimension(tied, STABLE_ID, WORK_ID) == "author-sessions"
    minority = tied + [
        _reserved("spend-real-2"),
        _settled("spend-real-2", "completed"),
    ]
    assert (
        author_sessions_exhaustion_dimension(minority, STABLE_ID, WORK_ID) == "author-sessions"
    )


@pytest.mark.smoke
def test_uncertain_or_foreign_evidence_never_upgrades_to_storm() -> None:
    """Empty, declined, unreadable, and foreign rows all fail toward plain."""

    # No evidence at all: plain exhaustion.
    assert author_sessions_exhaustion_dimension([], STABLE_ID, WORK_ID) == "author-sessions"
    # Provider-declined rows are never consumed and never storm evidence.
    declined_only = [
        _reserved("spend-declined"),
        _settled("spend-declined", "failed-harness-api-error", accepted=False),
    ]
    assert (
        author_sessions_exhaustion_dimension(declined_only, STABLE_ID, WORK_ID)
        == "author-sessions"
    )
    # An unreadable settlement counts as consumed NON-infra: producer
    # influence fails toward charging, so it dilutes rather than supports.
    unreadable = [
        _reserved("spend-storm"),
        _settled("spend-storm", "failed-machine-spawn"),
        _reserved("spend-torn"),
        {
            "event": "session_settled",
            "spend_id": "spend-torn",
            "schema_version": PAID_SESSION_SETTLEMENT_SCHEMA,
            "local_outcome": "completed",
            "provider": {"accepted": "not-a-bool"},
        },
    ]
    assert (
        author_sessions_exhaustion_dimension(unreadable, STABLE_ID, WORK_ID)
        == "author-sessions"
    )
    # Another generation's storm is not this generation's storm.
    foreign = [
        {
            "event": "session_reserved",
            "lane": "author",
            "spend_id": "spend-foreign",
            "attempt_group": None,
            "axis": {
                "work_generations": (
                    {"stable_id": STABLE_ID, "work_id": "work-other-generation"},
                )
            },
        },
        _settled("spend-foreign", "failed-harness-api-error"),
    ]
    assert (
        author_sessions_exhaustion_dimension(foreign, STABLE_ID, WORK_ID) == "author-sessions"
    )


@pytest.mark.smoke
def test_enforcement_terminal_carries_the_storm_split_reason() -> None:
    """An exhaustion raise under majority-infra evidence names the storm.

    The step-200 residual: the raw-session fuse terminalized sustained-storm
    models under the same ``effort-exhausted:author-sessions`` as genuine
    exhaustion, so triage pooled restorable storm damage with real
    exhaustion. The valve's own raise now derives the dimension from the same
    merged settlement evidence it already receives.
    """

    events = [
        _staged(STABLE_ID, WORK_ID, f"result-{index}") for index in range(AUTHOR_SESSIONS_MAX)
    ] + [
        row
        for index in range(2 * AUTHOR_SESSIONS_MAX + 1)
        for row in (
            _reserved(f"spend-storm-{index}"),
            _settled(f"spend-storm-{index}", "failed-harness-api-error"),
        )
    ]
    with pytest.raises(AuthorEffortCapExceeded) as caught:
        enforce_author_session_allowance(events, _work_item())
    assert caught.value.stage == "author"
    assert caught.value.reason_code == "effort-exhausted:provider-storm"
    assert caught.value.reason_code in FAILURE_REASON_CODES["author"]
    assert caught.value.reason_code in EFFORT_EXHAUSTION_REASON_CODES
