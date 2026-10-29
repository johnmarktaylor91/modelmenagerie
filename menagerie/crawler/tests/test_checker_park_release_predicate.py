"""BL-10 (f): checker-lane parks hold under an unchanged identity, release causally.

Audit CH-F6 / sol B07: checker-lane deferral rows had NO still-holds arm --
``_deferral_row_fact_still_holds`` fell through to ``False`` -- so a
deterministically-failing model re-bought the full 5->3->2->1 halving ladder
(~4 sessions, ~20-25 min) on EVERY later leg, forever; the pilot's six parks
were all perpetual re-buyers. The predicate, pinned here:

* a v2 row (recording the checker adjudication identity it failed under) HOLDS
  once the model has spent its bounded recidivism budget
  (``CHECKER_LANE_PARK_RECIDIVISM_LIMIT`` parks on the durable operational
  ledger) while that identity is UNCHANGED;
* it RELEASES the moment the lane's identity rotates -- model, version, or
  prompt: the causal change the remedy asks for;
* within the recidivism budget it releases toward attempting (a stochastic
  flake earns a fresh-leg retry);
* v1 rows (no identity recorded) keep their historical requeue-every-run
  behavior, and an unknown current identity errs toward attempting.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest

from menagerie.crawler.checker_lane import (
    CHECKER_LANE_DEFERRAL_SCHEMA_VERSION_V1,
    build_checker_lane_deferral_row,
)
from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.constants import CHECKER_LANE_PARK_RECIDIVISM_LIMIT
from menagerie.crawler.driver_admission import _deferral_row_fact_still_holds
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.gate_reuse import CheckerAdjudicationIdentity
from menagerie.crawler.identity import stable_hash

pytestmark = pytest.mark.smoke

_IDENTITY = {
    "provider": "openai",
    "model": "gpt-5.6-sol",
    "version": "current",
    "prompt_sha256": "sha256:" + "a" * 64,
}


class _FixedIdentityChecker:
    """A lane advertising one fixed adjudication identity."""

    def __init__(self, identity: Optional[dict[str, str]]) -> None:
        self._identity = identity

    def adjudication_identity(self, gate_kind: str, config: Any) -> Any:
        del gate_kind, config
        if self._identity is None:
            raise RuntimeError("no identity")
        return CheckerAdjudicationIdentity(**self._identity)


def _row(*, checker_identity: Optional[dict[str, str]] = _IDENTITY) -> dict[str, Any]:
    return build_checker_lane_deferral_row(
        stable_id="m_parked",
        work_id="work-m_parked",
        name="Parked Example",
        campaign_id="c1-mech",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-17T00:00:00Z",
        gate_kind="metadata_batch",
        failure_class="permanent-contract-rejection",
        failure_evidence="refused at every ladder size",
        attempted_batch_sizes=(5, 3, 2, 1),
        checker_identity=checker_identity,
    )


def _driver(tmp_path: Path, identity: Optional[dict[str, str]], parks: int) -> Any:
    """Fake driver with an operational ledger carrying ``parks`` park events."""

    models = tmp_path / "records" / "models" / "current-shard.jsonl"
    operational = canonical_operational_ledger_path(models)
    operational.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(parks):
        lines.append(
            json.dumps(
                {
                    "event_kind": "campaign-health",
                    "created_at": f"2026-08-1{index}T00:00:00Z",
                    "details": {
                        "disposition": "checker-lane-parked",
                        "stable_id": "m_parked",
                        "work_id": "work-m_parked",
                    },
                }
            )
        )
    operational.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return SimpleNamespace(
        config=SimpleNamespace(machine_id="machine-1"),
        dependencies=SimpleNamespace(checker=_FixedIdentityChecker(identity)),
        paths=SimpleNamespace(ledgers=SimpleNamespace(models=models)),
    )


_ITEM = cast(WorkItem, SimpleNamespace(route=SimpleNamespace(intent="core")))


def test_row_holds_after_recidivism_cap_under_unchanged_identity(tmp_path: Path) -> None:
    driver = _driver(tmp_path, _IDENTITY, parks=CHECKER_LANE_PARK_RECIDIVISM_LIMIT)
    assert _deferral_row_fact_still_holds(driver, _ITEM, _row()) is True


def test_row_releases_within_the_recidivism_budget(tmp_path: Path) -> None:
    """A first park earns its bounded fresh-leg retry (stochastic flakes recover)."""

    driver = _driver(tmp_path, _IDENTITY, parks=CHECKER_LANE_PARK_RECIDIVISM_LIMIT - 1)
    assert _deferral_row_fact_still_holds(driver, _ITEM, _row()) is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model", "gpt-7-rotated"),
        ("version", "next-channel"),
        ("prompt_sha256", "sha256:" + "b" * 64),
        ("provider", "azure-openai"),
    ),
)
def test_identity_rotation_is_the_causal_release(
    tmp_path: Path, field: str, value: str
) -> None:
    """Any rotated identity component releases the park, whatever the count."""

    rotated = dict(_IDENTITY, **{field: value})
    driver = _driver(tmp_path, rotated, parks=CHECKER_LANE_PARK_RECIDIVISM_LIMIT + 3)
    assert _deferral_row_fact_still_holds(driver, _ITEM, _row()) is False


def test_v1_rows_keep_their_historical_requeue_behavior(tmp_path: Path) -> None:
    """A row without a recorded identity can never hold (append-only history)."""

    driver = _driver(tmp_path, _IDENTITY, parks=CHECKER_LANE_PARK_RECIDIVISM_LIMIT + 3)
    legacy = _row(checker_identity=None)
    legacy["schema_version"] = CHECKER_LANE_DEFERRAL_SCHEMA_VERSION_V1
    legacy.pop("row_sha256")
    legacy["row_sha256"] = stable_hash(legacy)
    assert _deferral_row_fact_still_holds(driver, _ITEM, legacy) is False


def test_unknown_current_identity_errs_toward_attempting(tmp_path: Path) -> None:
    driver = _driver(tmp_path, None, parks=CHECKER_LANE_PARK_RECIDIVISM_LIMIT + 3)
    assert _deferral_row_fact_still_holds(driver, _ITEM, _row()) is False


def test_unreadable_operational_ledger_errs_toward_attempting(tmp_path: Path) -> None:
    """Unknown recidivism is never 'provably exhausted'."""

    driver = _driver(tmp_path, _IDENTITY, parks=0)
    operational = canonical_operational_ledger_path(driver.paths.ledgers.models)
    operational.unlink()
    assert _deferral_row_fact_still_holds(driver, _ITEM, _row()) is False


def test_park_events_for_other_models_do_not_count(tmp_path: Path) -> None:
    driver = _driver(tmp_path, _IDENTITY, parks=0)
    operational = canonical_operational_ledger_path(driver.paths.ledgers.models)
    lines = [
        json.dumps(
            {
                "event_kind": "campaign-health",
                "details": {
                    "disposition": "checker-lane-parked",
                    "stable_id": "m_other",
                    "work_id": "work-m_other",
                },
            }
        )
        for _ in range(CHECKER_LANE_PARK_RECIDIVISM_LIMIT + 2)
    ]
    operational.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _deferral_row_fact_still_holds(driver, _ITEM, _row()) is False
