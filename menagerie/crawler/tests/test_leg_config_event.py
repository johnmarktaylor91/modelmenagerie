"""B-09: every leg durably attests the config it ran under.

At the audit base the configuration a leg ran under lived in an operator's
shell history and nowhere else. Which scheduler ran it, how wide its author and
checker lanes were, which checker model adjudicated its gates, whether it was
an ordinary leg or an ``--only-unsettled`` scope cut, and which stable IDs a
named lane admitted -- none of it was in the ledgers. So when two legs behaved
differently, the config that caused the difference could not be recovered, and
a post-hoc "which leg ran with what?" was a reconstruction exercise rather than
a read.

Pinned here: exactly ONE ``leg-config`` operational event per leg, appended
BEFORE the first admission decision (so a leg that refuses at a door still
attests what it was launched as), carrying the effective config plus a
content-addressed ``config_digest``; per-occurrence identity, so a relaunch
never dedups into the first leg's row; and the registry/vocabulary wiring that
makes the kind legal (identity class, wake transition, JSON schema contract).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.constants import (
    CLAUDE_CLI_VERSION,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.operational_identity import (
    OPERATIONAL_EVENT_IDENTITY_REGISTRY,
    OperationalEventIdentityClass,
    mint_operational_event,
)
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    FakeForward,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke

_KIND = OperationalEventKind.LEG_CONFIG.value


def _config_events(paths: Any) -> list[Mapping[str, Any]]:
    """Return every leg-config event this campaign appended, in ledger order."""

    if not paths.operational_ledger.is_file():
        return []
    return [
        event for event in scan_jsonl(paths.operational_ledger) if event.get("event_kind") == _KIND
    ]


def _emit(driver: Any, paths: Any) -> Mapping[str, Any]:
    """Append one leg-config event through the REAL ledger writer.

    The ledger validates every append against the operational-event schema, so
    this exercises the B-09 schema contract (status const, required details
    keys, digest pattern) as well as the emitter.
    """

    with JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        driver._record_leg_config(operational)  # noqa: SLF001 -- the unit under test
    return _config_events(paths)[-1]


# ---------------------------------------------------------------------------
# The recorded facts.
# ---------------------------------------------------------------------------


def test_config_event_records_the_effective_leg_shape(tmp_path: Path) -> None:
    """The event names the scheduler, both widths, the models, and the scope."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(
        tmp_path,
        snapshot,
        scheduler="pipelined",
        author_concurrency=4,
        checker_concurrency=3,
        checker_model="gpt-5.6-sol",
    )
    driver.config = replace(driver.config, only_unsettled=True)

    details = _emit(driver, paths)["details"]
    assert details["scheduler"] == "pipelined"
    assert details["author_concurrency"] == 4
    assert details["checker_concurrency"] == 3
    assert details["checker_model"] == "gpt-5.6-sol"
    assert details["claude_cli_version"] == CLAUDE_CLI_VERSION
    assert details["only_unsettled"] is True
    assert details["redrive_targets"] == []
    assert details["gate_catchup_targets"] == []
    assert details["config_digest"].startswith("sha256:")


def test_config_event_records_the_named_lanes(tmp_path: Path) -> None:
    """A named lane's enumerated stable IDs are the leg's admission authority.

    They ARE the scope of such a leg, so they are recorded verbatim: a named
    re-drive whose targets are not durable is a leg nobody can audit.
    """

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    targets = tuple(item.stable_id for item in snapshot.items)
    driver = _driver(tmp_path, snapshot)
    driver.config = replace(
        driver.config,
        redrive_targets=targets,
        redrive_reason="operator: audit-visible named re-drive",
    )

    details = _emit(driver, paths)["details"]
    assert tuple(details["redrive_targets"]) == targets
    # D-54: the rationale is part of the durable record, beside its scope.
    assert details["redrive_reason"] == "operator: audit-visible named re-drive"
    assert details["gate_catchup_targets"] == []
    assert details["only_unsettled"] is False


def test_config_digest_distinguishes_exactly_the_config(tmp_path: Path) -> None:
    """One field answers "did these two legs run the same config?".

    The digest must move when the leg's shape moves and hold when it does not,
    or an auditor comparing eleven fields by eye is back where they started.
    """

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)

    def digest(**overrides: Any) -> str:
        driver.config = replace(DriverConfig(campaign_id="c1-mech"), **overrides)
        return str(driver._leg_config_details()["config_digest"])  # noqa: SLF001

    baseline = digest()
    assert digest() == baseline
    assert digest(scheduler="pipelined") != baseline
    assert digest(author_concurrency=DriverConfig().author_concurrency + 1) != baseline
    # b05 made gpt-5.6-sol the DriverConfig default, so proving the digest
    # distinguishes the checker axis needs a value that actually differs from
    # the default -- any distinct string does (this is a digest-distinction
    # probe, not a checker-authority surface, so it never re-endorses terra).
    assert digest(checker_model="gpt-5.6-sol-variant-probe") != baseline
    assert digest(only_unsettled=True) != baseline
    assert (
        digest(redrive_targets=("m_abc",), redrive_reason="operator: probe") != baseline
    )
    # D-54: the digest distinguishes the rationale too -- two legs re-driving
    # the same member for different reasons are different acts.
    assert digest(
        redrive_targets=("m_abc",), redrive_reason="operator: probe"
    ) != digest(redrive_targets=("m_abc",), redrive_reason="operator: other")


# ---------------------------------------------------------------------------
# Identity: one per leg, never deduped across legs.
# ---------------------------------------------------------------------------


def test_kind_is_registered_per_occurrence() -> None:
    """The kind is per-occurrence (_B) and refuses an idempotent key.

    An idempotent-first-clock registration would be wrong in the one way that
    matters: two legs launched with identical flags would collapse into ONE
    attestation, and the second leg would have no durable config at all.
    """

    spec = OPERATIONAL_EVENT_IDENTITY_REGISTRY[OperationalEventKind.LEG_CONFIG]
    assert spec.identity_class is OperationalEventIdentityClass.PER_OCCURRENCE

    event = {
        "event_kind": _KIND,
        "created_at": "2026-08-16T00:00:00Z",
        "details": {"scheduler": "waves"},
    }
    assert mint_operational_event(event)["event_id"].startswith(f"{_KIND}-")
    with pytest.raises(ValueError, match="forbids an idempotent identity payload"):
        mint_operational_event(event, identity_payload={"run_id": "run-test"})
    with pytest.raises(ValueError, match="requires created_at"):
        mint_operational_event({"event_kind": _KIND, "details": {}})


def test_identical_configs_at_different_clocks_are_distinct_events(tmp_path: Path) -> None:
    """Two legs with the same flags are two attestations, not one."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    details = driver._leg_config_details()  # noqa: SLF001
    base = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_kind": _KIND,
        "status": OperationalEventStatus.HEALTHY.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 0},
        "current_environment": None,
        "run_id": "run-test",
        "machine_id": "machine-test",
        "details": details,
    }
    first = mint_operational_event({**base, "created_at": "2026-08-16T00:00:00Z"})
    second = mint_operational_event({**base, "created_at": "2026-08-16T01:00:00Z"})
    assert first["event_id"] != second["event_id"]
    # Same config, same clock: the SAME occurrence, so the same identity.
    replayed = mint_operational_event({**base, "created_at": "2026-08-16T00:00:00Z"})
    assert replayed["event_id"] == first["event_id"]


def test_the_kind_is_pure_provenance_to_the_wake_machinery() -> None:
    """A leg-config event never opens, fires, or resolves a wake episode.

    A wake-callback leg emits one exactly like an ordinary leg does; if the
    transition table treated it as anything but an observation, recording the
    config would itself perturb the episode being resumed.
    """

    from menagerie.crawler.wakeup import _WAKE_EVENT_TRANSITIONS, _WakeEventTransition

    assert _WAKE_EVENT_TRANSITIONS[OperationalEventKind.LEG_CONFIG] is _WakeEventTransition.OBSERVE


def test_the_schema_pins_the_details_contract() -> None:
    """The operational-event schema knows the kind and its required details."""

    from menagerie.crawler.schema import load_schema, validate_payload

    schema = load_schema(OPERATIONAL_EVENT_SCHEMA_VERSION)
    assert _KIND in schema["properties"]["event_kind"]["enum"]
    event = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_id": f"{_KIND}-000000000000000000000000",
        "ledger_seq": 1,
        "payload_sha256": "sha256:" + "0" * 64,
        "created_at": "2026-08-16T00:00:00Z",
        "event_kind": _KIND,
        "status": OperationalEventStatus.HEALTHY.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 0},
        "current_environment": None,
        "run_id": "run-test",
        "machine_id": "machine-test",
        "details": {
            "scheduler": "waves",
            "author_concurrency": 1,
            "checker_concurrency": 1,
            "author_model": "claude-fable-5-1",
            "author_version": "current",
            "claude_cli_version": CLAUDE_CLI_VERSION,
            "checker_model": "gpt-5.6-terra",
            "checker_version": "current",
            "campaign_id": "c1-mech",
            "phase": "pytorch",
            "review_checkpoint_at": 1000,
            "only_unsettled": False,
            "only_status": None,
            "redrive_targets": [],
            "gate_catchup_targets": [],
            "config_digest": "sha256:" + "0" * 64,
        },
    }
    validate_payload(event, OPERATIONAL_EVENT_SCHEMA_VERSION)
    # A blocking status on a provenance record, and a missing required fact,
    # must both be rejected -- otherwise the contract is decorative.
    with pytest.raises(ValueError):
        validate_payload({**event, "status": "degraded"}, OPERATIONAL_EVENT_SCHEMA_VERSION)
    details = event["details"]
    assert isinstance(details, dict)
    thinned = {key: value for key, value in details.items() if key != "scheduler"}
    with pytest.raises(ValueError):
        validate_payload({**event, "details": thinned}, OPERATIONAL_EVENT_SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# The real leg: exactly one event, and it lands before the first door.
# ---------------------------------------------------------------------------


@pytest.mark.heavy
def test_a_real_leg_appends_exactly_one_config_event(tmp_path: Path) -> None:
    """One event per leg, first among the leg's provenance, and per relaunch."""

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    assert (
        _driver(
            tmp_path,
            snapshot,
            author=FakeAuthor(),
            checker=FakeChecker(),
            forward=FakeForward(),
            scheduler="pipelined",
            checker_concurrency=2,
        )
        .run()
        .status
        == "complete"
    )
    events = _config_events(paths)
    assert len(events) == 1
    assert events[0]["details"]["scheduler"] == "pipelined"
    assert events[0]["details"]["checker_concurrency"] == 2
    assert events[0]["status"] == OperationalEventStatus.HEALTHY.value

    # The config is attested BEFORE the leg makes any admission decision: its
    # ledger_seq precedes the selection census the same leg emits.
    census_seqs = [
        int(event["ledger_seq"])
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("event_kind") == OperationalEventKind.WORK_SELECTION_CENSUS.value
    ]
    assert census_seqs
    assert int(events[0]["ledger_seq"]) < min(census_seqs)

    # A second leg is a second attestation, with its own identity.
    assert _driver(tmp_path, snapshot, scheduler="waves").run().status == "complete"
    relaunched = _config_events(paths)
    assert len(relaunched) == 2
    assert relaunched[1]["details"]["scheduler"] == "waves"
    assert relaunched[0]["event_id"] != relaunched[1]["event_id"]


@pytest.mark.heavy
def test_a_leg_that_refuses_at_a_door_still_attested_its_config(tmp_path: Path) -> None:
    """A refused leg is exactly the leg whose config an auditor needs.

    The config event is appended before the first admission decision, so a
    launch that refuses at one -- here ``--after-review`` presented with no
    pending checkpoint, which raises before any work is selected -- has still
    recorded what it was launched as. A refusal that erased its own config
    would leave the operator diagnosing a leg they cannot describe.
    """

    from menagerie.crawler.driver_contracts import DriverIntegrationError

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot, scheduler="pipelined", author_concurrency=2)
    with pytest.raises(DriverIntegrationError, match="requires a pending review checkpoint"):
        driver.run(after_review=True)
    events = _config_events(paths)
    assert len(events) == 1
    assert events[0]["details"]["scheduler"] == "pipelined"
    assert events[0]["details"]["author_concurrency"] == 2


# ---------------------------------------------------------------------------
# D-63 (PM-1): tree identity, width provenance, and the rung-invariant digest.
# ---------------------------------------------------------------------------


def test_leg_config_carries_tree_identity_and_width_provenance(tmp_path: Path) -> None:
    """The attestation names the tree and which channel set each width.

    The 200a ledger claimed a width override was "in effect" that nothing
    consumed; the provenance fields beside the widths make that lie
    unrepeatable, and the observed tree lets a rung census prove its legs
    shared one tree without turning a dirty dev checkout into an outage.
    """

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    driver.config = replace(driver.config, author_concurrency_source="explicit-flag")

    details = _emit(driver, paths)["details"]

    assert details["author_concurrency_source"] == "explicit-flag"
    assert details["checker_concurrency_source"] == "default"
    # Disclosure, never refusal: outside a git repository both fields are
    # null; inside one they are the 40-hex tree and the dirty bit.
    assert "repo_tree_sha" in details
    assert "repo_tree_dirty" in details
    tree_sha = details["repo_tree_sha"]
    assert tree_sha is None or (isinstance(tree_sha, str) and len(tree_sha) == 40)
    assert details["rung_config_digest"].startswith("sha256:")


def test_rung_config_digest_is_invariant_across_per_leg_axes(tmp_path: Path) -> None:
    """Per-leg axes never move the rung digest; a rung-invariant axis does.

    Rung certification compares ``rung_config_digest`` across a rung's legs,
    so a wake-resumed leg, a manual leg, and a named-lane leg of the SAME
    configuration must agree on it while their full ``config_digest`` values
    differ -- and a width change must break it.
    """

    from menagerie.crawler.constants import InvocationOrigin

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    base = _emit(driver, paths)["details"]

    driver.config = replace(
        driver.config,
        invocation_origin=InvocationOrigin.WAKE_CALLBACK,
        wake_episode_id="wake-episode-1",
        only_unsettled=True,
        acknowledge_award_holds=True,
    )
    wake = _emit(driver, paths)["details"]
    assert wake["rung_config_digest"] == base["rung_config_digest"]
    assert wake["config_digest"] != base["config_digest"]

    driver.config = replace(
        driver.config,
        invocation_origin=InvocationOrigin.ORDINARY_RUN,
        wake_episode_id=None,
        author_concurrency=driver.config.author_concurrency + 1,
    )
    widened = _emit(driver, paths)["details"]
    assert widened["rung_config_digest"] != base["rung_config_digest"]


def test_the_width_provenance_vocabulary_is_closed_on_the_config() -> None:
    """A free-text source spelling refuses at construction."""

    with pytest.raises(ValueError, match="closed vocabulary"):
        DriverConfig(author_concurrency_source="operator-vibes")
