"""DX-7/RE-7 + DX-8/RE-8: deferral suppresses re-drives; RSS kills at a watermark.

200a re-spawned RSS-deferred members into the identical kill 14 times (m3767;
38/41 RSS spawns wasted = 20.5% of worker wall) because the observed-peak memo
compared against physical RAM -- always satisfied on the very host that killed
the worker -- and because sibling mode slots each bought their own full kill.
DX-8's watermark keeps the parent sampler ahead of a fast allocator (the
suspected 8/23 host-crash mechanism) and records the overshoot per attempt.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler.constants import FAILURE_REASON_CODES
from menagerie.crawler.driver_admission import _deferral_row_fact_still_holds
from menagerie.crawler.driver_receipts import (
    SupervisedForwardLane,
    WORKER_FAILURE_REASON_OWNERS,
    _worker_observed_rss_deferral_row,
    derive_worker_rss_limit_bytes,
)
from menagerie.crawler.host_capacity import CAPACITY_DEFERRAL_DISPOSITION
from menagerie.crawler.tests.test_driver_host_capacity_admission import (
    MIXTRAL_RECIPE,
    _rss_cap_attempt,
    _StubArtifact,
    _work_item,
)
from menagerie.crawler.worker_supervisor import (
    RSS_SOFT_WATERMARK_RATIO,
    RSS_WATERMARK_HEADROOM_FLOOR_BYTES,
    effective_rss_kill_threshold_bytes,
)

pytestmark = pytest.mark.smoke

_GIB = 1024**3
_MIB = 1024**2


def test_watermark_threshold_schedule() -> None:
    """The kill threshold is 85% of the cap with a 256 MiB headroom floor."""

    # Production-sized cap: the ratio binds (headroom 1.8 GiB > 256 MiB).
    assert effective_rss_kill_threshold_bytes(12 * _GIB) == 12 * _GIB - int(
        12 * _GIB * (1.0 - RSS_SOFT_WATERMARK_RATIO)
    )
    # Small cap: the 256 MiB floor binds.
    assert effective_rss_kill_threshold_bytes(1 * _GIB) == 1 * _GIB - 256 * _MIB
    assert RSS_WATERMARK_HEADROOM_FLOOR_BYTES == 256 * _MIB
    # Tiny cap: the floor is bounded at half the cap so the threshold stays
    # positive and monotone.
    assert effective_rss_kill_threshold_bytes(300 * _MIB) == 150 * _MIB
    # Disabled containment stays disabled.
    assert effective_rss_kill_threshold_bytes(0) == 0
    assert effective_rss_kill_threshold_bytes(-1) == 0
    # Monotone in the cap (never a bigger cap with a smaller threshold).
    thresholds = [
        effective_rss_kill_threshold_bytes(limit)
        for limit in range(64 * _MIB, 16 * _GIB, 512 * _MIB)
    ]
    assert thresholds == sorted(thresholds)


def test_blocked_by_sibling_rides_the_closed_vocabulary() -> None:
    """Schema + constants + owner map extend together (DX-15 cross-pin rides CI)."""

    assert "blocked-by-sibling" in FAILURE_REASON_CODES["resource"]
    assert WORKER_FAILURE_REASON_OWNERS["blocked-by-sibling"] == "model"


def test_lane_suppresses_siblings_after_a_deterministic_kill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """After one kill, no further slot of the generation spawns (one timeout max).

    The resolver replays the recorded timeout kill for slot (0, train); every
    remaining slot must route through the blocked-attempt factory without the
    lane ever reaching its spawn machinery.
    """

    import menagerie.crawler.driver_receipts as receipts

    monkeypatch.setattr(
        receipts,
        "_driver_receipt_dependencies",
        lambda: SimpleNamespace(execution_identity=lambda a, e, i: "exec-test"),
    )
    proposal = {
        "stable_id": "m-sibling",
        "work_id": "work-sibling-1",
        "proposed_facts": {
            "modes": {"meaningful_modes": ["train", "eval"], "train_eval_divergence": "none"},
            "source_resolution": {"rung": "R1"},
        },
    }
    artifact = SimpleNamespace(proposal=proposal)
    environment = SimpleNamespace(
        environment_authority=object(),
        environment_authority_cache=object(),
        family="test-family",
    )
    killed_row = {
        "attempt_id": "attempt-killed",
        "result": "failed",
        "error": {"stage": "resource", "reason_code": "timeout"},
    }
    factory_calls: list[tuple[int, str, Mapping[str, Any]]] = []

    def resolver(cold_index: int, mode: str) -> Optional[Mapping[str, Any]]:
        if (cold_index, mode) == (0, "train"):
            return killed_row
        return None

    def factory(cold_index: int, mode: str, kill: Mapping[str, Any]) -> dict[str, Any]:
        factory_calls.append((cold_index, mode, kill))
        return {
            "attempt_id": f"blocked-{cold_index}-{mode}",
            "result": "failed",
            "error": {"stage": "resource", "reason_code": "blocked-by-sibling"},
        }

    sunk: list[Mapping[str, Any]] = []
    lane = SupervisedForwardLane(timeout_seconds=1, rss_limit_bytes=1, cwd=None)
    attempts = lane.forward(
        artifact,  # type: ignore[arg-type]
        environment,  # type: ignore[arg-type]
        2,
        tmp_path,
        attempt_sink=sunk.append,
        attempt_resolver=resolver,
        closure=SimpleNamespace(identity="closure-test"),  # type: ignore[arg-type]
        blocked_attempt_factory=factory,
    )

    # The reducer policy sizes required cold runs (one for this fixture); the
    # invariant under test is that EVERY slot after the kill routes through the
    # factory and none reaches spawn machinery.
    assert factory_calls, "the sibling slot must be suppressed"
    assert all(call[2]["kind"] == "timeout" for call in factory_calls)
    assert [call[:2] for call in factory_calls] == [(0, "eval")]
    assert len(attempts) == 1 + len(factory_calls)
    assert [attempt["attempt_id"] for attempt in sunk] == ["blocked-0-eval"]


def test_observed_peak_memo_holds_on_the_killing_host() -> None:
    """DX-7 admission memo: the memo compares against the WORKER ALLOWANCE.

    On the host that killed the worker, physical RAM always exceeds the peak
    (the cap sat below RAM), so a RAM comparison re-spawned the member into the
    identical kill. The refusal must hold while the current effective kill
    threshold does not clear the observed peak.
    """

    allowance, _basis = derive_worker_rss_limit_bytes()
    threshold = effective_rss_kill_threshold_bytes(allowance)
    driver = SimpleNamespace(config=SimpleNamespace(run_id="run-test"))
    item = _work_item("m-memo", "Memo member")

    def _row(peak: int) -> dict[str, Any]:
        return {
            "disposition": CAPACITY_DEFERRAL_DISPOSITION,
            "capacity": {
                "estimate": {"origin": "worker-observed-rss", "max_peak_rss_bytes": peak}
            },
        }

    # A peak this host's threshold cannot clear: the member is NOT re-spawned.
    assert _deferral_row_fact_still_holds(driver, item, _row(threshold)) is True
    # A peak safely below the threshold: released toward attempting.
    assert _deferral_row_fact_still_holds(driver, item, _row(max(1, threshold // 4))) is False


def test_rss_valve_tolerates_derived_sibling_rows() -> None:
    """An all-RSS generation still defers when suppressed siblings ride along."""

    item = _work_item("m-rss-with-sibling", "RSS with sibling")
    rss = _rss_cap_attempt("attempt-rss", "train", 12_908_324_872, work_id=item.active_work_id)
    blocked = {
        "attempt_id": "attempt-blocked",
        "work_id": item.active_work_id,
        "result": "failed",
        "error": {"stage": "resource", "reason_code": "blocked-by-sibling"},
    }
    row = _worker_observed_rss_deferral_row(
        item,
        _StubArtifact({"library_recipe": MIXTRAL_RECIPE}),
        (rss, blocked),
        campaign_id="c1-mech",
        run_id="rung9b-replay",
        machine_id="worker-a",
        created_at="2026-08-05T23:40:50Z",
    )
    assert row is not None
    # The observation set is the RSS evidence alone.
    entries = row["capacity"]["estimate"]["attempts"]
    assert [entry["attempt_id"] for entry in entries] == ["attempt-rss"]
    # A blocked row alone can never defer (nothing was observed).
    alone = _worker_observed_rss_deferral_row(
        item,
        _StubArtifact({"library_recipe": MIXTRAL_RECIPE}),
        (blocked,),
        campaign_id="c1-mech",
        run_id="rung9b-replay",
        machine_id="worker-a",
        created_at="2026-08-05T23:40:50Z",
    )
    assert alone is None
