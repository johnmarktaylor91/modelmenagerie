"""AL-13: the retry counter counts, and the fingerprint clusters causes.

Two dead instruments measured on 200a's 113 ``failed:author`` members, pinned
here through the REAL builders:

1. ``root_cause_fingerprint`` had zero clustering power -- 113 distinct
   fingerprints for a handful of causes -- because the preimage hashed
   ``stable_id`` and the raw exception message (which embeds the member id
   and its work paths). Any triage, grant-loop refusal (AL-18/PD-12), or
   ``root_cause_repeat`` keyed on it was structurally blind.
2. ``status.retries.author == 1`` for all 113 while ``budget.
   author_sessions_used`` spanned 0-5: the writer stamped a 0/1 stage
   indicator, never a count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.constants import EnvironmentPhase
from menagerie.crawler.driver_contracts import DriverConfig, WorkItem
from menagerie.crawler.driver_models import (
    _assemble_terminal_model,
    _driver_failure_attempt,
)
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.routing import IntentRoute

_CREATED_AT = "2026-08-24T00:00:00Z"


def _work_item(stable_id: str) -> WorkItem:
    """Return a minimal routed work item for row-builder tests."""

    intake = IntakeItem(
        stable_id=stable_id,
        name="FingerprintNet",
        zoo="fixtures",
        variant="base",
        discovery_source="crawl_roster",
        legacy_row_sha256="0" * 64,
        preserved_legacy_flags=(),
        variant_scope="standalone",
        family_representative_id=stable_id,
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(stable_id=stable_id, intent="core", phase=EnvironmentPhase.PYTORCH),
    )


def _failure_attempt(
    tmp_path: Path,
    stable_id: str,
    *,
    stage: str = "author",
    message_template: str = "author session for {sid} died on the shared cause",
    stage_attempt: int = 1,
) -> dict:
    """Build one driver-observed failure row through the real builder."""

    return _driver_failure_attempt(
        _work_item(stable_id),
        None,
        stage,
        "session-crashed" if stage == "author" else "unreachable",
        RuntimeError(message_template.format(sid=stable_id)),
        DriverConfig(campaign_id="c2-disco", machine_id="fixture-machine"),
        diagnostics_root=tmp_path / ".crawl-local" / "diagnostics",
        environment=None,
        created_at=_CREATED_AT,
        stage_attempt=stage_attempt,
    )


@pytest.mark.smoke
def test_one_injected_cause_across_members_is_one_fingerprint(tmp_path: Path) -> None:
    """N members, one cause => ONE fingerprint, even with the member id in the
    exception message (the shape every real message has)."""

    rows = [_failure_attempt(tmp_path, stable_id) for stable_id in ("m101", "m202", "m303")]
    fingerprints = {row["error"]["root_cause_fingerprint"] for row in rows}
    assert len(fingerprints) == 1


@pytest.mark.smoke
def test_distinct_causes_keep_distinct_fingerprints(tmp_path: Path) -> None:
    """Clustering must not collapse genuinely different causes."""

    shared = _failure_attempt(tmp_path, "m101")
    different = _failure_attempt(
        tmp_path,
        "m101",
        message_template="forge refused {sid}'s primary reference outright",
    )
    assert (
        shared["error"]["root_cause_fingerprint"]
        != different["error"]["root_cause_fingerprint"]
    )


@pytest.mark.smoke
def test_terminal_fallback_fingerprint_also_clusters_across_members() -> None:
    """The artifact-free terminal's derived fingerprint follows the same rule."""

    def terminal(stable_id: str) -> dict:
        return _assemble_terminal_model(
            _work_item(stable_id),
            None,
            "failed:author",
            "session-crashed",
            f"author session for {stable_id} died on the shared cause",
            (),
            (),
            DriverConfig(),
            _CREATED_AT,
            human_review=False,
            root_cause_fingerprint=None,
        )

    first = terminal("m101")["status"]["root_cause_fingerprint"]
    second = terminal("m202")["status"]["root_cause_fingerprint"]
    assert first == second


@pytest.mark.smoke
def test_terminal_retries_report_real_per_stage_consumption(tmp_path: Path) -> None:
    """``status.retries`` counts consumed attempts per stage, never a 0/1 flag."""

    attempts = (
        _failure_attempt(tmp_path, "m101"),
        _failure_attempt(tmp_path, "m101", stage_attempt=2),
        _failure_attempt(tmp_path, "m101", stage="fetch"),
    )
    model = _assemble_terminal_model(
        _work_item("m101"),
        None,
        "failed:author",
        "session-crashed",
        "detail",
        attempts,
        (),
        DriverConfig(),
        _CREATED_AT,
        human_review=False,
        root_cause_fingerprint=None,
    )
    retries = model["status"]["retries"]
    # 1 + 2 ladder-consumed author invocations, one fetch row, nothing else.
    assert retries["author"] == 3
    assert retries["fetch"] == 1
    assert retries["gate"] == 0


@pytest.mark.smoke
def test_terminal_retries_author_equals_the_replay_derived_session_count(
    tmp_path: Path,
) -> None:
    """The tripwire exactly: ``retries.author == sessions consumed``.

    The replay-derived chargeable count the driver already passes (the same
    authority ``budget.author_sessions_used`` records) wins over row counting.
    """

    model = _assemble_terminal_model(
        _work_item("m101"),
        None,
        "failed:author",
        "session-crashed",
        "detail",
        (_failure_attempt(tmp_path, "m101"),),
        (),
        DriverConfig(),
        _CREATED_AT,
        human_review=False,
        root_cause_fingerprint=None,
        author_sessions_used=4,
    )
    assert model["status"]["retries"]["author"] == 4
    assert model["budget"]["author_sessions_used"] == 4
