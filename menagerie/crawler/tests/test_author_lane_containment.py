"""W5 (B3-XQ-25): member-fault containment -- one member's death stays its own.

The leg-27 killer this retires: the lane-health boundary's clause (3) read
"the second consecutive DISTINCT model to die without publishing" as a
lane-wide pattern. At width 12-18 two independent members dying 5-10 seconds
apart is Tuesday, not lane death -- the exact operational sequence 5821-5831
(m12687 -> m12724, 10 s apart, 41 publication witnesses in the window) killed
a ~15 h leg. The merged tri-lab design pinned here:

- F arm: host-checks-pass => member scope ALWAYS. An author death on a
  healthy host can never escalate past the member; the containment terminal
  is a DEFERRED-FAMILY park (site ``author-wave``), never a machine
  ``failed:author`` without adjudication (OC-Q6).
- O arm: the lane-health detector is a publication-DROUGHT predicate --
  windowed RATE, normalized by the configured width (``max(2, ceil(w/2))``
  distinct unpublished deaths AND zero publications in the window), never
  adjacency of two deaths.
- S arm: a typed primary-implementation-unavailable death (the
  ``AuthorSourceDeterministicallyUnfetchableError`` chain) is content
  evidence, not lane evidence -- excluded from the drought entirely.
- Storm shape (drought whose deaths are all machine-class) routes to the
  DURABLE usage-pause lane; a non-infra lane-wide pattern (content refusals
  across the lane) still stops LOUD, campaign-scoped.

Money-custody note: the boundary itself moves no money; the adversarial
money-custody review mandated by the plan rides the merge queue.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.driver_contracts import (
    AuthorSessionRetryExhausted,
    AuthorUsagePause,
    DriverConfig,
    WorkItem,
)
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    EnvironmentPhase,
)
from menagerie.crawler.scheduler_parks import (
    load_scheduler_quarantine_rows,
    scheduler_quarantine_path,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot


def _work_item(stable_id: str) -> WorkItem:
    """Return a minimal routed work item for boundary tests."""

    intake = IntakeItem(
        stable_id=stable_id,
        name=f"Containment-{stable_id}",
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


def _exhausted(
    stable_id: str, *, failure_class: Optional[str] = "session-crashed"
) -> AuthorSessionRetryExhausted:
    """Build one retry-exhaustion failure with a classified cause chain."""

    cause = RuntimeError(f"author session for {stable_id} died")
    if failure_class is not None:
        cause.failure_class = failure_class  # type: ignore[attr-defined]
        cause.failure_class_basis = "structured-notice"  # type: ignore[attr-defined]
    exhausted = AuthorSessionRetryExhausted(
        stable_id, attempts=3, backoff_seconds=(1.0, 2.0), detail=str(cause)
    )
    exhausted.__cause__ = cause
    return exhausted


def _source_unfetchable(stable_id: str) -> AuthorSessionRetryExhausted:
    """Build an exhaustion whose chain carries the typed unfetchable class."""

    from menagerie.crawler.author_dispatch import (
        AuthorSourceDeterministicallyUnfetchableError,
    )

    cause = AuthorSourceDeterministicallyUnfetchableError(stable_id, "no such source")
    exhausted = AuthorSessionRetryExhausted(
        stable_id, attempts=3, backoff_seconds=(1.0,), detail=str(cause)
    )
    exhausted.__cause__ = cause
    return exhausted


class _Rig:
    """One driver + operational ledger bound for direct boundary calls."""

    def __init__(self, tmp_path: Path, *, author_concurrency: int = 12) -> None:
        snapshot = _snapshot(tmp_path, count=1)
        self.paths = _paths(tmp_path, snapshot)
        self.driver = _driver(tmp_path, snapshot, author_concurrency=author_concurrency)
        self.records_root = self.paths.ledgers.models.parent.parent
        self.ledger_path = tmp_path / "boundary-operational.jsonl"
        # The rig calls the boundary directly, outside a running leg, so the
        # driver lock is not held; the F-arm tests assert the HEALTHY-host
        # law, so the host checks read healthy unless a test overrides them.
        self.driver._author_lane_health_checks = lambda: {  # type: ignore[method-assign]
            "disk_ok": True,
            "ledger_dir_writable": True,
            "driver_lock_held": True,
            "passed": True,
        }

    def decide(self, stable_id: str, exc: AuthorSessionRetryExhausted) -> bool:
        with JsonlLedger(self.ledger_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
            return self.driver._scope_author_retry_exhaustion(  # noqa: SLF001
                _work_item(stable_id), None, exc, None, operational, {}
            )

    def witness(self, stable_id: str) -> None:
        self.driver._note_author_lane_publication(stable_id)  # noqa: SLF001

    def quarantined(self) -> list[Any]:
        return [
            row
            for row in load_scheduler_quarantine_rows(
                [scheduler_quarantine_path(self.records_root)]
            )
        ]

    def decisions(self) -> list[str]:
        if not self.ledger_path.is_file():
            return []
        return [
            str(event["details"]["decision"])
            for event in scan_jsonl(self.ledger_path)
            if event.get("details", {}).get("kind") == "author-no-publication-boundary"
        ]


@pytest.mark.smoke
def test_operational_replay_seq_5821_5831_the_leg_survives(tmp_path: Path) -> None:
    """The exact leg-27 shape: two deaths 10 s apart with 41 witnesses.

    Both members are CONTAINED as author-wave deferred-family parks; no
    campaign escalation, no ``failed:author`` machine terminal.
    """

    rig = _Rig(tmp_path, author_concurrency=12)
    for index in range(41):
        rig.witness(f"m_witness_{index}")

    assert rig.decide("m12687", _exhausted("m12687")) is True
    assert rig.decide("m12724", _exhausted("m12724")) is True

    rows = rig.quarantined()
    assert sorted(row["stable_id"] for row in rows) == ["m12687", "m12724"]
    assert {row["scheduler_quarantine"]["site"] for row in rows} == {"author-wave"}
    assert rig.decisions() == ["member-contained", "member-contained"]
    # OC-Q6: no machine failed:author was minted -- the models ledger holds
    # nothing for either member.
    models_ledger = rig.paths.ledgers.models
    if models_ledger.is_file():
        terminal_ids = {
            row.get("stable_id")
            for row in scan_jsonl(models_ledger)
            if str(row.get("status", "")).startswith("failed:author")
        }
        assert not ({"m12687", "m12724"} & terminal_ids)


@pytest.mark.smoke
def test_a_first_death_on_a_healthy_host_is_member_contained(tmp_path: Path) -> None:
    """F arm: healthy host => member scope, even with no witness at all."""

    rig = _Rig(tmp_path)
    assert rig.decide("m_first", _exhausted("m_first")) is True
    assert [row["stable_id"] for row in rig.quarantined()] == ["m_first"]
    assert rig.decisions() == ["member-contained"]


def test_the_drought_threshold_is_width_normalized(tmp_path: Path) -> None:
    """O arm: deaths below max(2, ceil(width/2)) never read as lane-wide."""

    rig = _Rig(tmp_path, author_concurrency=12)
    threshold = max(2, math.ceil(12 / 2))
    for index in range(threshold - 1):
        assert rig.decide(f"m_dying_{index}", _exhausted(f"m_dying_{index}")) is True
    assert rig.decisions() == ["member-contained"] * (threshold - 1)


def test_a_content_refusal_drought_still_stops_loud(tmp_path: Path) -> None:
    """A lane-wide NON-infra pattern is a campaign stop, never a silent park."""

    rig = _Rig(tmp_path, author_concurrency=2)
    # width 2 => threshold 2. Two distinct content-refusal deaths with zero
    # publications in the window are lane-wide content weather.
    assert (
        rig.decide("m_refused_1", _exhausted("m_refused_1", failure_class="content-refusal"))
        is True
    ), "the first death is below the drought threshold and stays contained"
    assert (
        rig.decide("m_refused_2", _exhausted("m_refused_2", failure_class="content-refusal"))
        is False
    ), "the drought with content evidence must re-raise campaign-scoped"
    assert rig.decisions()[-1] == "campaign-scoped"


def test_a_machine_storm_drought_routes_to_the_durable_usage_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Storm shape: an all-machine-class drought pauses durably, never crashes."""

    rig = _Rig(tmp_path, author_concurrency=2)
    captured: dict[str, Any] = {}

    def _fake_pause(signal: Any, operational: Any, queued: int) -> str:
        captured["signal"] = signal
        return "usage-limit-anthropic"

    monkeypatch.setattr(rig.driver, "_pause_for_usage", _fake_pause)
    assert rig.decide("m_storm_1", _exhausted("m_storm_1")) is True
    with pytest.raises(AuthorUsagePause):
        rig.decide("m_storm_2", _exhausted("m_storm_2"))
    signal = captured["signal"]
    assert signal.provider == "anthropic"
    assert "drought" in signal.response_excerpt
    assert rig.decisions()[-1] == "usage-pause-storm"


def test_source_unavailability_is_excluded_from_lane_death_evidence(
    tmp_path: Path,
) -> None:
    """S arm: typed primary-implementation-unavailable deaths are not lane data."""

    rig = _Rig(tmp_path, author_concurrency=2)
    # Two source-unfetchable deaths would otherwise reach the width-2
    # threshold; excluded, the lane never reads as in drought and a third
    # ordinary death stays member-contained.
    assert rig.decide("m_nosource_1", _source_unfetchable("m_nosource_1")) is True
    assert rig.decide("m_nosource_2", _source_unfetchable("m_nosource_2")) is True
    assert rig.decide("m_ordinary", _exhausted("m_ordinary")) is True
    assert rig.decisions() == ["member-contained"] * 3


def test_a_failed_host_check_still_escalates_campaign_scoped(tmp_path: Path) -> None:
    """The F arm's contrapositive: a sick host is never a member's fault."""

    rig = _Rig(tmp_path)
    rig.witness("m_ok")

    def _sick_host() -> dict[str, Any]:
        return {"passed": False, "disk_ok": False}

    rig.driver._author_lane_health_checks = _sick_host  # type: ignore[method-assign]
    assert rig.decide("m_on_sick_host", _exhausted("m_on_sick_host")) is False
    assert rig.decisions() == ["campaign-scoped"]
    assert rig.quarantined() == []


@pytest.mark.smoke
def test_every_exhaustion_except_site_consults_the_boundary_and_reraises() -> None:
    """Blast-radius census: the mutation-raise contract at every catch site.

    Every ``except AuthorSessionRetryExhausted`` handler in driver_admission
    must consult ``_scope_author_retry_exhaustion`` and re-raise on a falsy
    verdict -- a handler that silently absorbs an unadjudicated exhaustion is
    exactly the silent-swallow class the census exists to refuse. The count
    is pinned so a new catch site must join the contract in the same change.
    """

    source = Path("menagerie/crawler/driver_admission.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    handlers: list[ast.ExceptHandler] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            names = {
                child.id
                for child in ast.walk(node.type)
                if isinstance(child, ast.Name)
            }
            if "AuthorSessionRetryExhausted" in names:
                handlers.append(node)
    assert len(handlers) == 4, "the exhaustion catch-site census moved; re-derive it"
    for handler in handlers:
        body_calls = {
            child.func.attr
            for child in ast.walk(handler)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        assert "_scope_author_retry_exhaustion" in body_calls
        assert any(
            isinstance(child, ast.Raise) and child.exc is None
            for child in ast.walk(handler)
        ), "every catch site must re-raise the campaign-scoped verdict"


def test_the_boundary_event_discloses_the_drought_facts(tmp_path: Path) -> None:
    """The decision's evidence (witnesses, drought census, host checks) is durable."""

    rig = _Rig(tmp_path)
    rig.witness("m_pub")
    rig.decide("m_x", _exhausted("m_x"))
    events = [
        event
        for event in scan_jsonl(rig.ledger_path)
        if event.get("details", {}).get("kind") == "author-no-publication-boundary"
    ]
    details = events[-1]["details"]
    assert details["decision"] == "member-contained"
    assert details["publications_this_run"] == 1
    assert "drought" in details
    assert details["drought"]["width"] == DriverConfig().author_concurrency
    assert details["host_checks"]["passed"] is True
