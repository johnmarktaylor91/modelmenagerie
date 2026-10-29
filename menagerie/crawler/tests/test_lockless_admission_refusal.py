"""EN-2: no member routes to a lockless non-exempt intent -- compose and admission refuse.

The pilot's m4714 (``tf-keras-arm64``) died at a LOCKLESS env turn after a full
paid author session: routing to an intent without a committed lock family is a
composition defect that must refuse PRE-AUTHOR, typed, not after the spend.
Both doors derive the route with THE driver derivation
(``constants.framework_from_markers`` -> ``routing.route_model``):

* COMPOSE (``rung_staging.stage_rung``): a member routing to a lockless
  non-exempt intent refuses BEFORE any durable byte exists;
* ADMISSION (``driver_admission._ordered_work``): the same refusal fires when a
  leg starts (a snapshot staged before this law, or a requeue re-target,
  cannot route around it).

``allow_lockless`` is the explicit operator exemption. It is RE-DERIVED at the
door against the LIVE registry -- an entry naming an intent that is actually
LOCKED refuses as stale, and no tool-written field is ever trusted for it
(producer-input rule).

Whole-catalog census: over the committed 200b composition corpus
(``menagerie/data/master_catalog.jsonl``), ZERO rows route to a lockless
intent under the unified derivation -- the reviewed zoo allowlist is EMPTY
today (the roster's ``discovered-jax`` class is the known future member and
lives outside the 200b corpus).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.constants import framework_from_markers
from menagerie.crawler.driver_contracts import DriverConfig, DriverIntegrationError
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, load_environment_registry
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.routing import route_model
from menagerie.crawler.rung_staging import RungStagingError
from menagerie.crawler.tests.test_slice_f_driver import _driver, _write_jsonl

pytestmark = pytest.mark.smoke

_CATALOG = Path(__file__).resolve().parents[2] / "data" / "master_catalog.jsonl"

#: Reviewed zoo allowlist for the whole-catalog census (EN-2). EMPTY by
#: adjudication: under the unified derivation zero master_catalog rows route
#: to a lockless intent. The crawl_roster's ``discovered-jax`` zoo (171 rows)
#: is the known legitimate lockless-routing class and lives OUTSIDE the 200b
#: composition corpus; adding any member here is a reviewed change.
_CENSUS_ZOO_ALLOWLIST: frozenset[str] = frozenset()


def _mixed_snapshot(tmp_path: Path):
    """One torch row plus one tensorflow-zoo row that routes to tf-keras-arm64."""

    master = tmp_path / "master.jsonl"
    _write_jsonl(
        master,
        [
            {"name": "TorchExample", "zoo": "fixtures", "variant": "base"},
            {"name": "KerasThing", "zoo": "tensorflow-models", "variant": "base"},
        ],
    )
    _write_jsonl(tmp_path / "deferred.jsonl", [])
    return create_intake_snapshot(master, tmp_path / "deferred.jsonl", tmp_path / "intake")


def _registry():
    return load_environment_registry(DEFAULT_ENVS_ROOT, target="osx-arm64")


def test_fixture_row_really_routes_to_a_lockless_intent(tmp_path: Path) -> None:
    """The fixture exercises the real derivation, not a hand-picked intent."""

    snapshot = _mixed_snapshot(tmp_path)
    item = next(entry for entry in snapshot.items if entry.name == "KerasThing")
    route = route_model(item.to_model_requirements(framework_from_markers(item.zoo, item.name)))
    registry = _registry()
    assert registry.intents[route.intent].lock.status != "locked"


# ---------------------------------------------------------------------------
# Admission (driver_admission._ordered_work)
# ---------------------------------------------------------------------------


def test_admission_refuses_a_lockless_route_pre_author(tmp_path: Path) -> None:
    """The m4714 shape refuses at leg start, before any author spend."""

    snapshot = _mixed_snapshot(tmp_path)
    driver = _driver(tmp_path, snapshot, registry=_registry())
    with pytest.raises(DriverIntegrationError, match="lockless|lock family"):
        driver._ordered_work(snapshot, {})


def test_admission_admits_the_same_route_with_the_explicit_exemption(tmp_path: Path) -> None:
    snapshot = _mixed_snapshot(tmp_path)
    driver = _driver(tmp_path, snapshot, registry=_registry())
    driver.config = DriverConfig(
        campaign_id=driver.config.campaign_id,
        allow_lockless_intents=("tf-keras-arm64",),
    )
    items = driver._ordered_work(snapshot, {})
    assert {item.route.intent for item in items} >= {"tf-keras-arm64"}


def test_a_stale_exemption_naming_a_locked_intent_refuses(tmp_path: Path) -> None:
    """The exemption is re-derived against the LIVE registry, never trusted."""

    snapshot = _mixed_snapshot(tmp_path)
    driver = _driver(tmp_path, snapshot, registry=_registry())
    driver.config = DriverConfig(
        campaign_id=driver.config.campaign_id,
        allow_lockless_intents=("core", "tf-keras-arm64"),
    )
    with pytest.raises(DriverIntegrationError, match="locked|stale"):
        driver._ordered_work(snapshot, {})


def test_a_requeue_cannot_retarget_onto_a_lockless_intent(tmp_path: Path) -> None:
    """The requeue door goes through the same refusal."""

    master = tmp_path / "master.jsonl"
    _write_jsonl(master, [{"name": "TorchExample", "zoo": "fixtures", "variant": "base"}])
    _write_jsonl(tmp_path / "deferred.jsonl", [])
    snapshot = create_intake_snapshot(master, tmp_path / "deferred.jsonl", tmp_path / "intake")
    driver = _driver(tmp_path, snapshot, registry=_registry())
    stable_id = snapshot.items[0].stable_id
    requeues = {stable_id: {"target_intent": "jax-flax-arm64"}}
    with pytest.raises(DriverIntegrationError):
        driver._ordered_work(snapshot, {}, requeues)


# ---------------------------------------------------------------------------
# Compose (rung_staging.stage_rung) -- the door that stops it earliest
# ---------------------------------------------------------------------------


def test_compose_refuses_a_lockless_route_with_no_durable_bytes(tmp_path: Path) -> None:
    """Staging refuses typed and leaves nothing for the register door to adopt."""

    from menagerie.crawler.rung_staging import refuse_lockless_member_routes

    snapshot = _mixed_snapshot(tmp_path)
    with pytest.raises(RungStagingError, match="lockless|lock family"):
        refuse_lockless_member_routes(
            tuple(item.stable_id for item in snapshot.items),
            snapshot,
            registry=_registry(),
            allow_lockless=(),
        )


def test_compose_honors_the_explicit_exemption_and_refuses_stale_ones(tmp_path: Path) -> None:
    from menagerie.crawler.rung_staging import refuse_lockless_member_routes

    snapshot = _mixed_snapshot(tmp_path)
    members = tuple(item.stable_id for item in snapshot.items)
    refuse_lockless_member_routes(
        members, snapshot, registry=_registry(), allow_lockless=("tf-keras-arm64",)
    )
    with pytest.raises(RungStagingError, match="locked|stale"):
        refuse_lockless_member_routes(
            members, snapshot, registry=_registry(), allow_lockless=("core",)
        )


# ---------------------------------------------------------------------------
# Whole-catalog census
# ---------------------------------------------------------------------------


def test_no_master_catalog_row_routes_to_a_lockless_intent() -> None:
    """The 200b composition corpus lands zero members on lockless intents."""

    registry = _registry()
    lockless = {
        name for name, intent in registry.intents.items() if intent.lock.status != "locked"
    }
    assert lockless  # the census is vacuous if everything is locked
    violations: list[tuple[str, str, str]] = []
    with _CATALOG.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            zoo = str(row.get("zoo", ""))
            name = str(row.get("name", ""))
            framework = framework_from_markers(zoo, name)
            intent = {
                "tf": "tf-keras-arm64",
                "jax": "jax-flax-arm64",
                "paddle": "paddle-arm64",
            }.get(framework)
            if intent in lockless and zoo not in _CENSUS_ZOO_ALLOWLIST:
                violations.append((name, zoo, intent))
    assert violations == [], violations[:20]
