"""WS-1 behavioral fixtures: the ONE work-selection policy (AUDIT100 SYNAUD-A §3).

These are the behavioral pins the audit demanded in place of source greps
(TP-1): a five-row campaign -- (a) never-recorded, (b) settled
dependency-CURRENT head, (c) settled dependency-STALE head, (d) parked,
(e) host-capacity-deferred -- run through the REAL pipelined path, the REAL
waves path, and the wake-resume argv builder, asserting the scheduled set is
exactly {a}. Row (c) is the incident mechanism (vector 8b): the dependency
currency projection returns ``None`` for a recorded-but-stale head, so
"recorded-but-stale" and "never recorded" were the same input and finish-7/8
re-authored 20 recorded members, 19 of them award holders.

Verified RED against 2f22635f (S2-1): at the audit base the five-row fixture
schedules {a, c, d} on both scheduler paths, grants are consumed BEFORE
selection even under ``--only-unsettled``, and the wake callback argv drops
the scheduler and every scope flag.

Sol's S2 extension rows are pinned too: (f) host-deferred on the same
incapable host stays excluded, (g) host-deferred on a now-capable host is
admitted, (h) a raw ``runs`` head plus a generic requeue grant stays excluded
absent an exact award-work authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    build_canonical_requeue_void,
    append_canonical_requeue_void,
    canonical_operational_ledger_path,
    canonical_records_root,
    canonical_requeue_grants_path,
)
from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.host_capacity import HOST_MEMORY_ENV_VAR
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.refusal_parks import (
    append_refusal_park_row,
    build_refusal_park_row,
    refusal_park_path,
)
from menagerie.crawler.routing import EnvironmentPhase, IntentRoute
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.partition_proof import PartitionProofError
from menagerie.crawler.tests.test_slice_f_driver import (
    NOW,
    CheckerScript,
    FakeAuthor,
    FakeChecker,
    FakeForward,
    ScriptedChecker,
    _driver,
    _paths,
    _write_jsonl,
)
from menagerie.crawler.tests.test_host_capacity import MIXTRAL_RECIPE


def _rotate_publication_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotate ``publication_policy_identity`` for the next legs (the incident axis).

    51cdae77 -- the IDSWEEP crash-fix itself -- moved artifact_transactions.py
    and thereby rotated ``publication_policy_identity``, staling 32 of 33
    awards on the campaign's own projection while review-held terminals
    stayed current through the recorded-basis carve-out. This helper
    reproduces exactly that world: the live context builder returns a rotated
    publication policy identity, and nothing else changes.
    """

    from dataclasses import replace as dataclass_replace

    from menagerie.crawler import driver as driver_module
    from menagerie.crawler import driver_admission as admission_module
    from menagerie.crawler.authority import build_authority_context as real_builder

    def rotated(**kwargs: Any) -> Any:
        context = real_builder(**kwargs)
        return dataclass_replace(
            context,
            publication_policy_identity=stable_hash(
                "ws1-fixture: publication policy bytes moved between legs"
            ),
        )

    monkeypatch.setattr(driver_module, "build_authority_context", rotated)
    monkeypatch.setattr(admission_module, "build_authority_context", rotated)


def _five_row_snapshot(tmp_path: Path):
    """Create the five-row intake: b, c, e PyTorch; a, d native-tail.

    The native-tail rows exist so leg 1 (``--phase pytorch``) can settle b, c,
    and e while leaving a and d untouched -- a stays never-recorded and d gets
    its park row appended out-of-band, exactly the m9911 treadmill shape (a
    live refusal park with no model head).
    """

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(
        master,
        [
            {"name": "RowNeverRecorded", "zoo": "tensorflow", "variant": "base"},
            {"name": "RowCurrentHead", "zoo": "fixtures", "variant": "base"},
            {"name": "RowStaleAward", "zoo": "fixtures", "variant": "base"},
            {"name": "RowParked", "zoo": "tensorflow", "variant": "base"},
            {"name": "RowDeferred", "zoo": "fixtures", "variant": "base"},
        ],
    )
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "intake")
    by_name = {item.name: item.stable_id for item in snapshot.items}
    return snapshot, by_name


def _seed_capacity_deferral(tmp_path: Path, snapshot, stable_id: str, name: str) -> None:
    """Append a REAL durable host-capacity deferral row for one member.

    Uses the production admission gate against this campaign's records root
    (the same construction as test_driver_host_capacity_admission), so the row
    carries the exact work generation the scheduler would later see.
    """

    from types import SimpleNamespace

    models_path = _paths(tmp_path, snapshot).ledgers.models
    duck = SimpleNamespace(
        dependencies=SimpleNamespace(clock=lambda: NOW),
        paths=SimpleNamespace(ledgers=SimpleNamespace(models=models_path)),
        config=SimpleNamespace(campaign_id="c1-mech", run_id="run-test", machine_id="machine-test"),
    )

    class _StubArtifact:
        proposal = {"proposed_facts": {"implementation": {"library_recipe": MIXTRAL_RECIPE}}}

    class _StubLedger:
        def append(self, event):
            return event

    item = WorkItem(
        intake=next(entry for entry in snapshot.items if entry.stable_id == stable_id),
        route=IntentRoute(stable_id=stable_id, intent="torch-core", phase=EnvironmentPhase.PYTORCH),
    )
    admitted = CrawlerDriver._admit_within_host_capacity(  # noqa: SLF001
        duck, (item,), {stable_id: _StubArtifact()}, _StubLedger()
    )
    assert admitted == ()


def _seed_refusal_park(tmp_path: Path, snapshot, stable_id: str, name: str) -> None:
    """Append a live terminal-refusal park row for a member with NO model head.

    The row records the RUNNING tree as its parking tree, so D-48's machine
    release lane reads it as already-attempted-under-this-tree and the member
    stays parked -- these fixtures pin clause 5's absolute half. The lane's own
    one-generation-per-tree flip cells live in ``test_park_release_lane.py``.
    """

    from menagerie.crawler.park_release import current_park_tree_identity

    records_root = _paths(tmp_path, snapshot).ledgers.models.parent.parent
    append_refusal_park_row(
        refusal_park_path(records_root),
        build_refusal_park_row(
            stable_id=stable_id,
            work_id=f"work-{stable_id}",
            name=name,
            campaign_id="c1-mech",
            run_id="run-test",
            machine_id="machine-test",
            created_at=NOW,
            refusal_class="ReductionError",
            record_error_type="menagerie.crawler.reducer.ReductionError",
            record_error="model environment generation is absent from active context",
            fallback_error_type=None,
            fallback_error=None,
            intended_status_code="failed:environment",
            intended_reason_code="environment-generation-stale",
            blocked_parent_revision=None,
            candidate_sha256=None,
            tree_sha=current_park_tree_identity(),
        ),
    )


def _five_row_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build the five-row campaign at leg-1 close and return (snapshot, ids).

    Leg 1 runs ``--phase pytorch`` with a per-model rejecting checker so that
    afterwards:

    * b holds a review-held ``failed:accuracy-gate`` terminal (dependency-
      current under any later rotation via the recorded-basis carve-out),
    * c holds a ``runs`` award (dependency-STALE once leg 2 rotates the
      checker authority),
    * e holds a durable host-capacity deferral row and no head,
    * a and d were never touched (native-tail, out of leg-1 scope);
      d then gains a live refusal-park row with no head.
    """

    monkeypatch.setenv(HOST_MEMORY_ENV_VAR, str(16 * 2**30))
    snapshot, by_name = _five_row_snapshot(tmp_path)
    ids = {
        "a": by_name["RowNeverRecorded"],
        "b": by_name["RowCurrentHead"],
        "c": by_name["RowStaleAward"],
        "d": by_name["RowParked"],
        "e": by_name["RowDeferred"],
    }
    # The capacity valve row belongs to leg 1's evidence window. Opening the
    # generation before the lower-level fixture emits the row mirrors the real
    # driver order and gives the row durable run provenance under BL-5.
    from menagerie.crawler.leg_generation import open_leg_generation

    paths = _paths(tmp_path, snapshot)
    open_leg_generation(
        paths.ledgers.models.parent.parent,
        campaign_id="c1-mech",
        intake_snapshot_id=snapshot.snapshot_id,
        intake_snapshot_sha256=snapshot.snapshot_sha256,
        phase="pytorch",
        run_id="run-test",
        opened_at=NOW,
    )
    _seed_capacity_deferral(tmp_path, snapshot, ids["e"], "RowDeferred")
    rejecting = ScriptedChecker(
        script=CheckerScript(metadata_verdict="inaccurate", selected_id=ids["b"])
    )
    first = _driver(tmp_path, snapshot, checker=rejecting, phase="pytorch", scheduler="waves").run()
    assert first.status.startswith("phase-terminal-partition-complete")
    heads = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert heads[ids["b"]]["status"]["code"] == "failed:accuracy-gate"
    assert heads[ids["b"]]["status"]["human_review"]["required"] is True
    assert heads[ids["c"]]["status"]["kind"] == "runs"
    assert ids["a"] not in heads and ids["d"] not in heads and ids["e"] not in heads
    _seed_refusal_park(tmp_path, snapshot, ids["d"], "RowParked")
    return snapshot, ids


@pytest.mark.heavy
@pytest.mark.parametrize("scheduler", ("waves", "pipelined"))
def test_five_row_fixture_schedules_exactly_the_never_recorded_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheduler: str
) -> None:
    """Rows (a)-(e) under ``--only-unsettled``: the scheduled set is exactly {a}.

    Leg 2 rotates the publication policy. Before RC-2 that made row (c)'s
    award dependency-STALE (the close-era world, 32 of 33 awards stale) and
    tripped the award-custody door; since RC-2 every settled head replays
    CURRENT at its recorded basis, so (c) is simply a settled head. Either
    way the leg must author and execute ONLY the never-recorded member. At
    2f22635f this fixture is RED on both paths: (c) reads as never-recorded
    through the projection and is re-driven, and (d)'s live park is invisible
    so its refusal treadmills.
    """

    snapshot, ids = _five_row_campaign(tmp_path, monkeypatch)
    _rotate_publication_policy(monkeypatch)
    author = FakeAuthor()
    forward = FakeForward()
    second = _driver(
        tmp_path,
        snapshot,
        author=author,
        forward=forward,
        checker=FakeChecker(),
        scheduler=scheduler,
    )
    second.config = _with_only_unsettled(second.config)
    # RC-2: row (c)'s award replays CURRENT at its recorded basis under the
    # rotation (no award hold, no stale exclusion, no PartitionError naming
    # it); it is a settled head selection never re-opens, and the recheck
    # census -- not the projection -- carries the unmeasured move. The leg
    # now reaches the leg-era close, whose exact-citation proof refuses on the
    # fixture's parked no-head row (d), seeded in leg 1's window -- a fixture
    # residual of the leg close, never a statement about (a) or (c).
    with pytest.raises(PartitionProofError) as refusal:
        second.run()
    assert ids["a"] not in str(refusal.value) and ids["c"] not in str(refusal.value)
    assert set(author.calls) == {ids["a"]}, (
        "the leg authored beyond the never-recorded member: "
        f"{sorted(set(author.calls))} (expected exactly {{{ids['a']}}})"
    )
    assert set(forward.calls) == {ids["a"]}
    census = _selection_census_events(tmp_path, snapshot)
    assert census, "the leg emitted no work-selection census event"
    details = census[-1]["details"]
    assert details["never_recorded"] == 1
    assert details["stale_excluded"] == 0
    assert details["parked"] == 1
    assert details["deferred"] == 1


@pytest.mark.heavy
def test_ordinary_mode_refuses_to_start_over_stale_awards_without_operator_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary leg over a stale-award campaign walks TWO doors, in order.

    The projection-staleness census is the single cheapest diagnostic nobody
    ran in three days; a leg that would silently EXCLUDE settled-but-stale
    members must surface the count and stop until the operator acknowledges.

    COMPOSE ruling (AW x WS-1): the award-custody door (AW-1/AW-2 --
    projection-stale runs awards + live award parks, every mode, released by
    registry rotation / park drain / ``acknowledge_award_holds``) fires FIRST
    at leg start; selection's visibility door (WS-1 -- every stale-excluded
    settled head in ordinary mode, released by a named mode /
    ``ack_stale_excluded``) fires second. Neither ack subsumes the other:
    they authorize different things (launching without registering rotations
    vs skipping stale members without re-driving them), so a stale-award
    ordinary launch needs BOTH.
    """


    snapshot, ids = _five_row_campaign(tmp_path, monkeypatch)
    _rotate_publication_policy(monkeypatch)
    # RC-2: the rotation stales nothing, so neither door has anything to
    # refuse -- the ordinary leg launches, authors only the never-recorded
    # member, and closes without a PartitionError. The unmeasured move is the
    # recheck census's business (release-excluded there, never held here).
    author = FakeAuthor()
    ordinary = _driver(tmp_path, snapshot, author=author, scheduler="waves")
    # The leg-era close's exact-citation proof refuses on the parked no-head
    # row (d) seeded in leg 1's window (fixture residual, see the sibling test).
    with pytest.raises(PartitionProofError) as refusal:
        ordinary.run()
    assert ids["a"] not in str(refusal.value) and ids["c"] not in str(refusal.value)
    assert set(author.calls) == {ids["a"]}
    from menagerie.crawler.recheck_census import recheck_census
    from menagerie.crawler.reducer import _select_current
    from menagerie.crawler.tests.conftest import make_authority_context

    paths = _paths(tmp_path, snapshot)
    heads = _select_current(scan_jsonl(paths.ledgers.models))
    rotated = replace_publication_policy(make_authority_context(list(ids.values())))
    census = recheck_census(heads, (), rotated, scope=ids.values())
    assert ids["c"] in census["by_axis"]["publication_policy_identity"]


def replace_publication_policy(context: Any) -> Any:
    """Apply the same rotated publication policy identity the fixture installs."""

    from dataclasses import replace as dataclass_replace

    return dataclass_replace(
        context,
        publication_policy_identity=stable_hash(
            "ws1-fixture: publication policy bytes moved between legs"
        ),
    )


def _with_only_unsettled(config):
    """Return the config with the operator only-unsettled scope cut enabled."""

    from dataclasses import replace

    return replace(config, only_unsettled=True)


def _digest_from_refusal(message: str) -> str:
    """Extract the census digest a D-47 door refusal disclosed (the ceremony)."""

    import re

    match = re.search(r"sha256:[0-9a-f]{64}", message)
    assert match is not None, f"refusal disclosed no census digest: {message!r}"
    return match.group(0)


def _with_ack_stale_excluded(config, digest):
    """Return the config with the stale-excluded operator acknowledgment.

    D-47: the blanket boolean is retired; the acknowledgment binds the exact
    standing census digest the refusal disclosed (the ONE authority is
    ``census_ack.stale_excluded_ack_digest`` over member ids plus closed
    reason classes -- the operator copies the digest from the door's own
    message, never re-derives it by hand).
    """

    from dataclasses import replace

    return replace(
        config,
        ack_stale_excluded=True,
        stale_excluded_digest=digest,
    )


def _with_acknowledge_award_holds(config, stale_awards=None):
    """Return the config with AW's award-custody-door acknowledgment.

    COMPOSE ruling: the award-holds door (AW-1/AW-2, leg start, every mode,
    stale runs awards + live award parks) and the stale-exclusion door (WS-1,
    ordinary mode, every stale-excluded settled head) are SEPARATE authorities
    with separate release levers; neither ack subsumes the other. A leg over a
    stale-award campaign passes the award door first, then selection's.

    D-47: an acknowledgment that actually passes the door binds the exact
    standing census digest (``census_ack.award_holds_ack_digest`` over the
    stale runs awards; these fixtures hold no live award-supersession parks).
    Call sites that never reach the door (the wake-argv round-trip) omit
    ``stale_awards`` and keep the boolean-only shape.
    """

    from dataclasses import replace

    from menagerie.crawler.census_ack import award_holds_ack_digest

    digest = (
        award_holds_ack_digest(stale_awards, ())
        if stale_awards is not None
        else config.award_holds_digest
    )
    return replace(config, acknowledge_award_holds=True, award_holds_digest=digest)


def _selection_census_events(tmp_path: Path, snapshot) -> list[Mapping[str, Any]]:
    """Return every work-selection census event this campaign ever appended."""

    paths = _paths(tmp_path, snapshot)
    events: list[Mapping[str, Any]] = []
    sources = {canonical_operational_ledger_path(paths.ledgers.models), paths.operational_ledger}
    for source in sources:
        if not source.is_file():
            continue
        try:
            rows = scan_jsonl(source, validate=False)
        except (OSError, ValueError):
            continue
        events.extend(row for row in rows if row.get("event_kind") == "work-selection-census")
    return events


def _consumption_events(tmp_path: Path, snapshot) -> list[Mapping[str, Any]]:
    """Return every requeue-grant consumption event this campaign holds."""

    paths = _paths(tmp_path, snapshot)
    events: list[Mapping[str, Any]] = []
    sources = {canonical_operational_ledger_path(paths.ledgers.models), paths.operational_ledger}
    for source in sources:
        if not source.is_file():
            continue
        try:
            rows = scan_jsonl(source, validate=False)
        except (OSError, ValueError):
            continue
        events.extend(
            row
            for row in rows
            if row.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
        )
    return events


def _single_member_terminal_campaign(tmp_path: Path):
    """Run one member to a review-held ``failed:accuracy-gate`` terminal."""

    from menagerie.crawler.tests.test_slice_f_driver import InaccurateChecker, _snapshot

    snapshot = _snapshot(tmp_path, count=1)
    result = _driver(tmp_path, snapshot, checker=InaccurateChecker()).run()
    assert result.status == "terminal-partition-complete"
    stable_id = snapshot.items[0].stable_id
    return snapshot, stable_id


def _append_grant(tmp_path: Path, snapshot, stable_id: str, *, marker: int = 1) -> str:
    """Append one durable accuracy-gate requeue grant and return its ID."""

    paths = _paths(tmp_path, snapshot)
    reason = "human reviewed corrected evidence"
    grant_id = stable_hash(
        {"stable_id": stable_id, "stage": "accuracy-gate", "reason": reason, "grant": marker}
    )
    grant = {
        "grant_id": grant_id,
        "stable_id": stable_id,
        "stage": "accuracy-gate",
        "reason": reason,
        "attempts": 1,
        "created_at": NOW,
    }
    paths.requeue_grants.parent.mkdir(parents=True, exist_ok=True)
    with paths.requeue_grants.open("ab") as handle:
        handle.write(canonical_json_bytes(grant) + b"\n")
    return grant_id


@pytest.mark.heavy
def test_a_live_grant_is_never_consumed_under_only_unsettled(tmp_path: Path) -> None:
    """Scope-off: ``--only-unsettled`` neither schedules nor CONSUMES a live grant.

    At 2f22635f grant consumption ran before selection (`_consume_requeue_grants`
    at driver.py:1188), so a finishing leg burned the grant -- minted its work
    generation, appended the consumption event -- and then excluded the member,
    leaving an undischargeable in-flight identity (the 48-generation residue).
    Consumption is a side effect and side effects follow selection.
    """

    snapshot, stable_id = _single_member_terminal_campaign(tmp_path)
    _append_grant(tmp_path, snapshot, stable_id)
    forward = FakeForward()
    second = _driver(tmp_path, snapshot, forward=forward)
    second.config = _with_only_unsettled(second.config)
    second_result = second.run()
    assert forward.calls == {}, "only-unsettled re-drove a settled terminal"
    assert _consumption_events(tmp_path, snapshot) == [], (
        "a live grant was consumed by a leg whose scope excludes granted work"
    )
    paths = _paths(tmp_path, snapshot)
    revisions = [
        record for record in scan_jsonl(paths.ledgers.models) if record["stable_id"] == stable_id
    ]
    assert [record["status"]["code"] for record in revisions] == ["failed:accuracy-gate"]
    assert second_result.status == "terminal-partition-complete"


@pytest.mark.heavy
def test_a_live_grant_admits_consumes_atomically_and_rebuys_in_ordinary_scope(
    tmp_path: Path,
) -> None:
    """Scope-on: an ordinary leg admits the granted member and consumes exactly once."""

    snapshot, stable_id = _single_member_terminal_campaign(tmp_path)
    grant_id = _append_grant(tmp_path, snapshot, stable_id)
    forward = FakeForward()
    second = _driver(tmp_path, snapshot, forward=forward).run()
    assert second.status in {"complete", "terminal-partition-complete"}
    assert forward.calls == {stable_id: 1}
    consumed = _consumption_events(tmp_path, snapshot)
    assert len(consumed) == 1
    assert consumed[0]["details"]["grant_id"] == grant_id
    paths = _paths(tmp_path, snapshot)
    revisions = [
        record for record in scan_jsonl(paths.ledgers.models) if record["stable_id"] == stable_id
    ]
    assert [record["status"]["code"] for record in revisions] == [
        "failed:accuracy-gate",
        "runs",
    ]
    assert revisions[-1]["budget"]["explicit_grants"] == [grant_id]
    replay_forward = FakeForward()
    third = _driver(tmp_path, snapshot, forward=replay_forward).run()
    assert third.status in {"complete", "terminal-partition-complete"}
    assert replay_forward.calls == {}
    assert len(_consumption_events(tmp_path, snapshot)) == 1


@pytest.mark.heavy
def test_a_voided_grant_neither_schedules_nor_consumes(tmp_path: Path) -> None:
    """A grant the operator voided is dead: no admission, no consumption."""

    snapshot, stable_id = _single_member_terminal_campaign(tmp_path)
    grant_id = _append_grant(tmp_path, snapshot, stable_id)
    paths = _paths(tmp_path, snapshot)
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    grant = {
        "grant_id": grant_id,
        "stable_id": stable_id,
        "stage": "accuracy-gate",
        "reason": "human reviewed corrected evidence",
        "attempts": 1,
        "created_at": NOW,
    }
    append_canonical_requeue_grant(canonical, grant)
    void = build_canonical_requeue_void(
        canonical,
        void_of=grant_id,
        reason="issued against the wrong member",
        voided_by="operator",
    )
    append_canonical_requeue_void(canonical, void)
    forward = FakeForward()
    result = _driver(tmp_path, snapshot, forward=forward).run()
    assert result.status == "terminal-partition-complete"
    assert forward.calls == {}
    assert _consumption_events(tmp_path, snapshot) == []


@pytest.mark.heavy
def test_a_runs_head_with_a_generic_grant_stays_excluded(tmp_path: Path) -> None:
    """Row (h): a generic requeue grant never reopens an award.

    Retry permission and downgrade permission are separate authorities
    (§2.1): admission of a ``runs`` head requires an exact award-work
    authority naming its current revision, which a generic accuracy-gate
    grant is not. The member stays excluded, nothing consumes, and the
    grant hold is disclosed durably.
    """

    from menagerie.crawler.tests.test_slice_f_driver import _snapshot

    snapshot = _snapshot(tmp_path, count=1)
    first = _driver(tmp_path, snapshot).run()
    assert first.status in {"complete", "terminal-partition-complete"}
    stable_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    heads = {row["stable_id"]: row for row in scan_jsonl(paths.ledgers.models)}
    assert heads[stable_id]["status"]["kind"] == "runs"
    _append_grant(tmp_path, snapshot, stable_id)
    forward = FakeForward()
    second = _driver(tmp_path, snapshot, forward=forward).run()
    assert second.status in {"complete", "terminal-partition-complete"}
    assert forward.calls == {}
    assert _consumption_events(tmp_path, snapshot) == []
    revisions = [row for row in scan_jsonl(paths.ledgers.models) if row["stable_id"] == stable_id]
    assert [row["status"]["kind"] for row in revisions] == ["runs"]


@pytest.mark.heavy
def test_a_now_capable_host_admits_a_deferred_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row (g): a host-capacity deferral whose fact no longer holds releases.

    The deferral row records a parameter bound; on a host whose admissible
    ceiling exceeds it the recorded refusal no longer holds and the member is
    ordinary never-recorded work again (WS-2's capability release, exercised
    through selection). COMPOSE (WS-2 B8): a leg whose ENTIRE target set is
    host-capability deferred refuses at the leg-start preflight rather than
    quietly closing -- recording that leg as healthy activity is how eight
    step-100 finishing legs looked like progress. Row (f)'s contract (the
    incapable host never re-attempts the deferral) is preserved BY the
    refusal: nothing runs at all.
    """

    from menagerie.crawler.driver_contracts import HostCapabilityPreflightError

    monkeypatch.setenv(HOST_MEMORY_ENV_VAR, str(16 * 2**30))
    from menagerie.crawler.tests.test_slice_f_driver import _snapshot

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    _seed_capacity_deferral(tmp_path, snapshot, stable_id, snapshot.items[0].name)
    forward = FakeForward()
    with pytest.raises(HostCapabilityPreflightError):
        _driver(tmp_path, snapshot, forward=forward).run()
    assert forward.calls == {}, "row (f): the incapable host re-attempted the deferral"
    monkeypatch.setenv(HOST_MEMORY_ENV_VAR, str(4096 * 2**30))
    released_forward = FakeForward()
    released = _driver(tmp_path, snapshot, forward=released_forward).run()
    assert released_forward.calls == {stable_id: 1}
    assert released.status in {"complete", "terminal-partition-complete"}


@pytest.mark.smoke
def test_wake_callback_argv_carries_the_complete_launching_config(
    tmp_path: Path,
) -> None:
    """The wake continuation argv replays scheduler, scope, and width flags.

    DELTA F5/D8: the step-100 wake argv carried only resume/intake/target/
    run-id, so a fired wake resumed UNSCOPED under the DEFAULT scheduler --
    with the wave award-currentness lane live, a fully composed award-eating
    path. Continuation identity must be complete: parsing the persisted argv
    back through the CLI must reproduce the launching selection config.
    """

    from menagerie.crawler import cli as cli_module
    from menagerie.crawler.tests.test_slice_f_driver import _snapshot

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot, scheduler="pipelined")
    driver.config = _with_acknowledge_award_holds(_with_only_unsettled(driver.config))
    argv = driver._wakeup_callback_argv()  # noqa: SLF001
    assert "run" in argv
    run_args = list(argv[argv.index("run") :])
    assert "--repo-root" in argv
    repo_root_index = argv.index("--repo-root")
    parsed = cli_module.build_parser().parse_args(
        ["--repo-root", argv[repo_root_index + 1], *run_args]
    )
    assert parsed.scheduler == "pipelined", (
        "the wake argv dropped --scheduler: a fired wake would resume under "
        f"the default scheduler (argv: {argv})"
    )
    assert parsed.only_unsettled is True, f"the wake argv dropped --only-unsettled (argv: {argv})"
    assert parsed.acknowledge_award_holds is True, (
        "the wake argv dropped --acknowledge-award-holds: a wake fired on an "
        f"acked launch would strand at the award-custody door (argv: {argv})"
    )
    assert parsed.author_concurrency == driver.config.author_concurrency
    assert parsed.checker_concurrency == driver.config.checker_concurrency


@pytest.mark.smoke
def test_wake_continuation_identity_is_byte_stable_and_flag_sensitive(
    tmp_path: Path,
) -> None:
    """Continuation identity: same config, same bytes; any scope flag flip differs."""

    from menagerie.crawler.tests.test_slice_f_driver import _snapshot
    from menagerie.crawler.work_selection import selection_continuation_identity

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot, scheduler="pipelined")
    baseline = selection_continuation_identity(driver.config)
    assert baseline == selection_continuation_identity(driver.config)
    driver.config = _with_only_unsettled(driver.config)
    assert selection_continuation_identity(driver.config) != baseline


def _raw_head(
    kind: str,
    code: str,
    *,
    authored_metadata_state: str = "pending",
    record_revision: str = "sha256:head-rev",
) -> dict[str, Any]:
    """Build one raw ledger head carrying every field the policy READS.

    The policy reads three fields off a raw head: ``status.kind``,
    ``record_revision`` (the award-naming authority match), and
    ``authored_metadata_state`` (the GATE_CATCHUP clause's runs@pending shape).
    A hand-built head that omits any of them does not merely under-specify the
    fixture -- it makes the clause reading it VACUOUS, so the door passes no
    matter what the test varies. Every raw head in this module goes through
    here so no clause can go silent again.

    Parameters
    ----------
    kind, code:
        Closed status kind and its terminal code.
    authored_metadata_state:
        Author-metadata disposition; ``pending`` is the honest default for the
        module's base ``failed:forward`` head (a failed run never landed
        accepted metadata).
    record_revision:
        Exact raw revision an award-naming authority must name.

    Returns
    -------
    dict[str, Any]
        Raw head shaped like a real ledger row's policy-visible fields.
    """

    return {
        "status": {"kind": kind, "code": code},
        "authored_metadata_state": authored_metadata_state,
        "record_revision": record_revision,
    }


@pytest.mark.smoke
def test_selection_predicate_is_pure_and_mode_invariant_to_invocation_origin() -> None:
    """The predicate is one pure function; invocation origin has zero effect."""

    from menagerie.crawler.driver_contracts import InvocationOrigin
    from menagerie.crawler.work_selection import (
        MemberFacts,
        SelectionMode,
        work_selection_predicate,
    )

    facts = MemberFacts(
        stable_id="m-x",
        raw_head=None,
        dependency_current=False,
        stale_reason=None,
        parked=False,
        deferral_live=False,
        live_grant=None,
        grant_bindable=False,
        active_generation=None,
        active_generation_discharged=False,
    )
    ordinary = work_selection_predicate(facts, mode=SelectionMode.ORDINARY)
    assert ordinary.admitted and ordinary.decision.value == "never-recorded"
    for origin in InvocationOrigin:
        del origin  # the predicate signature admits no origin: nothing to vary
    unsettled = work_selection_predicate(facts, mode=SelectionMode.ONLY_UNSETTLED)
    assert unsettled.admitted


@pytest.mark.smoke
def test_selection_predicate_raw_head_exclusions_and_exceptions() -> None:
    """Clause-level truth table for raw-headed members."""

    from menagerie.crawler.work_selection import (
        MemberFacts,
        SelectionDecision,
        SelectionMode,
        work_selection_predicate,
    )

    def facts(**overrides: Any) -> Any:
        base: dict[str, Any] = {
            "stable_id": "m-x",
            "raw_head": _raw_head("failed", "failed:forward"),
            "dependency_current": True,
            "stale_reason": None,
            "parked": False,
            "deferral_live": False,
            "live_grant": None,
            "grant_bindable": False,
            "active_generation": None,
            "active_generation_discharged": False,
        }
        base.update(overrides)
        return MemberFacts(**base)

    settled = work_selection_predicate(facts(), mode=SelectionMode.ORDINARY)
    assert not settled.admitted
    assert settled.decision is SelectionDecision.SETTLED_HEAD

    stale = work_selection_predicate(
        facts(dependency_current=False, stale_reason="authority rotated"),
        mode=SelectionMode.ORDINARY,
    )
    assert not stale.admitted, "a stale settled head is STILL settled (clause 1)"
    assert stale.decision is SelectionDecision.STALE_EXCLUDED

    parked = work_selection_predicate(facts(parked=True), mode=SelectionMode.ORDINARY)
    assert not parked.admitted
    assert parked.decision is SelectionDecision.PARKED

    named = work_selection_predicate(
        facts(),
        mode=SelectionMode.REDRIVE,
        mode_targets=frozenset({"m-x"}),
    )
    assert named.admitted, "an enumerated re-drive target IS an admission authority"
    assert named.decision is SelectionDecision.NAMED_REDRIVE

    award = work_selection_predicate(
        facts(raw_head=_raw_head("runs", "runs")),
        mode=SelectionMode.REDRIVE,
        mode_targets=frozenset({"m-x"}),
    )
    assert not award.admitted, "a CLI target list is not an award-work authority"
    assert award.decision is SelectionDecision.AWARD_CUSTODY

    parked_named = work_selection_predicate(
        facts(parked=True),
        mode=SelectionMode.REDRIVE,
        mode_targets=frozenset({"m-x"}),
    )
    assert not parked_named.admitted, "a live park is absolute (clause 5)"

    # Clause 4's OTHER named mode. GATE_CATCHUP keys on the R1/R2 catch-up
    # SHAPE -- runs@pending -- not on membership in the target list, and both
    # halves of that conjunction are load-bearing. The heads below vary exactly
    # the two fields the clause reads; a raw head that OMITS
    # ``authored_metadata_state`` (as every hand-built head in this module did)
    # makes the clause unreachable, so a catch-up leg reads
    # NOT_IN_MODE_TARGETS for every target it was handed and nothing fails.
    catchup = work_selection_predicate(
        facts(raw_head=_raw_head("runs", "runs", authored_metadata_state="pending")),
        mode=SelectionMode.GATE_CATCHUP,
        mode_targets=frozenset({"m-x"}),
    )
    assert catchup.admitted, "a runs@pending catch-up target IS the gate-catchup lane"
    assert catchup.decision is SelectionDecision.GATE_CATCHUP

    # An ACCEPTED runs head is not catch-up work: the lane exists only to finish
    # metadata the author never landed, so a named target is still refused.
    accepted_target = work_selection_predicate(
        facts(raw_head=_raw_head("runs", "runs", authored_metadata_state="accepted")),
        mode=SelectionMode.GATE_CATCHUP,
        mode_targets=frozenset({"m-x"}),
    )
    assert not accepted_target.admitted
    assert accepted_target.decision is SelectionDecision.NOT_IN_MODE_TARGETS

    # ...and pending metadata over a NON-runs head is ordinary unfinished work,
    # never an award whose metadata is owed.
    pending_terminal = work_selection_predicate(
        facts(), mode=SelectionMode.GATE_CATCHUP, mode_targets=frozenset({"m-x"})
    )
    assert not pending_terminal.admitted
    assert pending_terminal.decision is SelectionDecision.NOT_IN_MODE_TARGETS


@pytest.mark.smoke
def test_only_unsettled_cut_is_absolute_even_for_capability_reopens() -> None:
    """The operator scope cut admits NOTHING recorded (red-team break 7).

    ``--only-unsettled`` was sold as the guillotine remedy for vectors 4/6, yet
    the WS-2 capability-reopen arm sat ABOVE the cut and re-admitted a recorded
    head with ``work_authority=None`` -- no grant, no enumerated target, no
    named mode, no acknowledgment gate. The arm belongs BELOW the cut: the
    host-fact remedy stays automatic in every mode EXCEPT the absolute one.
    """

    from menagerie.crawler.work_selection import (
        MemberFacts,
        SelectionDecision,
        SelectionMode,
        work_selection_predicate,
    )

    facts = MemberFacts(
        stable_id="m-x",
        raw_head={"status": {"kind": "deferred", "code": "deferred:needs-more-memory"}},
        dependency_current=True,
        stale_reason=None,
        parked=False,
        deferral_live=False,
        live_grant=None,
        grant_bindable=False,
        active_generation=None,
        active_generation_discharged=False,
        capability_reopened=True,
    )
    cut = work_selection_predicate(facts, mode=SelectionMode.ONLY_UNSETTLED)
    assert not cut.admitted, (
        "a recorded head entered an --only-unsettled leg via capability-reopened"
    )
    assert cut.decision is SelectionDecision.SETTLED_HEAD
    # The WS-2 remedy itself survives: ordinary scope still re-opens the head.
    ordinary = work_selection_predicate(facts, mode=SelectionMode.ORDINARY)
    assert ordinary.admitted
    assert ordinary.decision is SelectionDecision.CAPABILITY_REOPENED


@pytest.mark.smoke
def test_a_dead_grants_generation_cannot_upgrade_into_award_work() -> None:
    """Consuming a grant never upgrades its authority (red-team break 6).

    The active-generation arm fires exactly when ``live_grant is None`` -- i.e.
    after the grant is consumed and dead -- and consulted neither head kind nor
    any award-naming authority, so the same ``runs`` member flipped from
    refused (award custody) to admitted by the act of consumption. The
    generation carries EXACTLY its originating grant's authority: a generic
    grant's generation never buys award work, while an award-naming grant's
    generation keeps re-admitting the exact revision it named (a paused award
    re-drive must converge across resumes, not demand a fresh grant per leg).
    """

    from menagerie.crawler.work_selection import (
        MemberFacts,
        SelectionDecision,
        SelectionMode,
        work_selection_predicate,
    )

    def facts(head_kind: str, **generation_extra: Any) -> MemberFacts:
        return MemberFacts(
            stable_id="m-x",
            raw_head={
                "status": {"kind": head_kind, "code": head_kind},
                "record_revision": "sha256:award-rev",
            },
            dependency_current=True,
            stale_reason=None,
            parked=False,
            deferral_live=False,
            live_grant=None,
            grant_bindable=False,
            active_generation={
                "grant_id": "grant-dead",
                "new_work_generation": 1,
                **generation_extra,
            },
            active_generation_discharged=False,
        )

    awarded = work_selection_predicate(facts("runs"), mode=SelectionMode.ORDINARY)
    assert not awarded.admitted, (
        "an ordinary leg admitted an awarded head as active-granted-generation "
        "on a consumed (dead) GENERIC grant's leftover generation"
    )
    assert awarded.decision is SelectionDecision.AWARD_CUSTODY
    # A consumed grant naming a DIFFERENT award revision releases nothing.
    moved = work_selection_predicate(
        facts("runs", blocked_award_revision="sha256:some-older-rev"),
        mode=SelectionMode.ORDINARY,
    )
    assert not moved.admitted
    assert moved.decision is SelectionDecision.AWARD_CUSTODY
    # The award-naming generation keeps exactly the authority it was minted
    # with: the named current revision stays schedulable until discharged.
    named = work_selection_predicate(
        facts("runs", blocked_award_revision="sha256:award-rev"),
        mode=SelectionMode.ORDINARY,
    )
    assert named.admitted
    assert named.decision is SelectionDecision.ACTIVE_GRANT_GENERATION
    # A non-award undischarged generation still settles rather than vanishing.
    terminal = work_selection_predicate(facts("failed"), mode=SelectionMode.ORDINARY)
    assert terminal.admitted
    assert terminal.decision is SelectionDecision.ACTIVE_GRANT_GENERATION


@pytest.mark.smoke
def test_discharge_vocabulary_is_closed_and_derivable() -> None:
    """The generation-discharge vocabulary is a closed set with one spelling each."""

    from menagerie.crawler.work_selection import GenerationDischarge

    assert {state.value for state in GenerationDischarge} == {
        "satisfied",
        "failed-terminal",
        "parked-policy-refusal",
        "voided-before-start",
        "superseded-by-generation",
        "leg-exit-abandoned",
    }


@pytest.mark.heavy
def test_a_voided_generation_is_discharged_and_never_replays(tmp_path: Path) -> None:
    """Clause 6: ``voided-before-start`` discharges an in-flight granted identity.

    A consumed grant whose generation never started (the m4933 shape) is an
    identity nothing discharges at 2f22635f: every ``--resume`` re-binds it as
    active work forever. The operator void tool appends the typed discharge
    event and selection stops replaying it.
    """

    snapshot, stable_id = _single_member_terminal_campaign(tmp_path)
    grant_id = _append_grant(tmp_path, snapshot, stable_id)
    paths = _paths(tmp_path, snapshot)
    canonical_grants = canonical_requeue_grants_path(paths.ledgers.models)
    append_canonical_requeue_grant(
        canonical_grants,
        {
            "grant_id": grant_id,
            "stable_id": stable_id,
            "stage": "accuracy-gate",
            "reason": "human reviewed corrected evidence",
            "attempts": 1,
            "created_at": NOW,
        },
    )
    heads = {row["stable_id"]: row for row in scan_jsonl(paths.ledgers.models)}
    source_revision = heads[stable_id]["record_revision"]
    new_work_id = stable_hash(
        {
            "stable_id": stable_id,
            "grant_id": grant_id,
            "parent_revision": source_revision,
            "generation": 1,
        }
    )
    consumption = {
        "schema_version": "menagerie.crawler.operational-event.v1",
        "event_id": f"requeue-consumed-{grant_id[7:31]}",
        "created_at": NOW,
        "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
        "status": "requeue-grant-consumed",
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": "run-test",
        "machine_id": "machine-test",
        "details": {
            "grant_id": grant_id,
            "stable_id": stable_id,
            "stage": "accuracy-gate",
            "reason": "human reviewed corrected evidence",
            "attempts": 1,
            "source_record_revision": source_revision,
            "new_work_generation": 1,
            "new_work_id": new_work_id,
        },
    }
    from menagerie.crawler.recordio import JsonlLedger

    events_path = canonical_operational_ledger_path(paths.ledgers.models)
    with JsonlLedger(
        events_path, "menagerie.crawler.operational-event.v1", lock_timeout=10.0
    ) as ledger:
        ledger.append(consumption)

    from menagerie.crawler.tools import discharge_generation

    exit_code = discharge_generation.main(
        [
            "--records-root",
            str(canonical_records_root(paths.ledgers.models)),
            "--repo-root",
            str(tmp_path),
            "void",
            "--stable-id",
            stable_id,
            "--work-id",
            new_work_id,
            "--reason",
            "generation superseded by the fix wave; never started",
            "--discharged-by",
            "operator",
        ]
    )
    assert exit_code == 0
    forward = FakeForward()
    resumed = _driver(tmp_path, snapshot, forward=forward).run()
    assert resumed.status == "terminal-partition-complete"
    assert forward.calls == {}, "a discharged generation replayed on resume"
    revisions = [row for row in scan_jsonl(paths.ledgers.models) if row["stable_id"] == stable_id]
    assert [row["status"]["code"] for row in revisions] == ["failed:accuracy-gate"]


@pytest.mark.smoke
def test_discharge_migration_classifies_generations_from_ledger_evidence(
    tmp_path: Path,
) -> None:
    """The migration dry-run assigns each consumed generation a discharge state.

    A generation whose grant appears in a landed revision's budget is
    ``satisfied``/``failed-terminal`` by that revision's status; a consumed
    generation with no landed revision, no park, no void, and no successor is
    LIVE and listed as needing an operator decision (exactly m4933's shape).
    """

    from menagerie.crawler.tools import discharge_generation

    records_root = tmp_path / "records"
    models = records_root / "models" / "current-shard.jsonl"
    models.parent.mkdir(parents=True)
    events = records_root / "operational" / "events.jsonl"
    events.parent.mkdir(parents=True)

    def consumption(stable_id: str, grant_id: str, work_id: str) -> dict[str, Any]:
        return {
            "event_kind": "requeue-grant-consumed",
            "details": {
                "grant_id": grant_id,
                "stable_id": stable_id,
                "source_record_revision": "sha256:" + "1" * 64,
                "new_work_generation": 1,
                "new_work_id": work_id,
            },
        }

    models.write_text(
        json.dumps(
            {
                "stable_id": "m-satisfied",
                "record_revision": "sha256:" + "2" * 64,
                "status": {"kind": "runs", "code": "runs"},
                "budget": {"explicit_grants": ["grant-a"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events.write_text(
        json.dumps(consumption("m-satisfied", "grant-a", "work-a"))
        + "\n"
        + json.dumps(consumption("m-live", "grant-b", "work-b"))
        + "\n",
        encoding="utf-8",
    )
    report = discharge_generation.migration_report(records_root)
    by_work_id = {entry["work_id"]: entry for entry in report}
    assert by_work_id["work-a"]["discharge"] == "satisfied"
    assert by_work_id["work-b"]["discharge"] is None
    assert by_work_id["work-b"]["needs_operator_decision"] is True


# ---------------------------------------------------------------------------
# Award-custody hardening (2026-08-16 red-team ROUND 2, break 5): the reducer
# half of round-1 #7 refuses a voided grant; the SELECTION half did not. Voiding
# killed the live-grant arm and left the same grant fully potent on its consumed
# GENERATION arm, so the withdrawn authority kept buying work.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_a_voided_grants_generation_is_discharged_not_admitted() -> None:
    """A void discharges the generation it authorized, at the one derivation.

    ``derive_generation_discharge`` is the ONE authority selection consults for
    clause 6, so the void has to land there rather than in a second opinion
    downstream: an operator who withdraws a grant withdraws every identity that
    grant minted. A LANDED revision still wins -- work that actually settled is
    ``satisfied``, and rewriting that as "voided" would be the dishonest half of
    the same mistake.
    """

    from menagerie.crawler.work_selection import (
        GenerationDischarge,
        derive_generation_discharge,
    )

    details = {
        "grant_id": "grant-voided",
        "stable_id": "m-x",
        "new_work_id": "work-1",
        "new_work_generation": 1,
    }
    live, evidence = derive_generation_discharge(
        details,
        model_revisions=[],
        live_park_work_ids=frozenset(),
        discharge_events=[],
        all_consumptions=[],
        voided_grant_ids=frozenset(),
    )
    assert live is None and evidence is None

    discharged, evidence = derive_generation_discharge(
        details,
        model_revisions=[],
        live_park_work_ids=frozenset(),
        discharge_events=[],
        all_consumptions=[],
        voided_grant_ids=frozenset({"grant-voided"}),
    )
    assert discharged is GenerationDischarge.VOIDED_BEFORE_START
    assert "grant-voided" in str(evidence)

    landed, evidence = derive_generation_discharge(
        details,
        model_revisions=[
            {
                "stable_id": "m-x",
                "record_revision": "sha256:landed",
                "status": {"kind": "runs", "code": "runs"},
                "budget": {"explicit_grants": ["grant-voided"]},
            }
        ],
        live_park_work_ids=frozenset(),
        discharge_events=[],
        all_consumptions=[],
        voided_grant_ids=frozenset({"grant-voided"}),
    )
    assert landed is GenerationDischarge.SATISFIED


@pytest.mark.heavy
def test_a_voided_grants_generation_never_admits_at_selection(tmp_path: Path) -> None:
    """The withdrawn authority stops buying work (round-2 break 5).

    ``_requeue_grant_state`` filtered ``voided_ids`` out of the LIVE-grant arm
    (driver_admission L5998) and out of nothing else, so the consumed
    generation of a voided grant kept re-admitting its member on every leg --
    one wasted author/execution purchase per generation per member, on an
    authority the operator explicitly withdrew. ``checkpoint.py`` refuses to
    void an already-consumed grant, but it reads consumptions from the
    CANONICAL events ledger while the driver writes the receipt run-locally
    first, so the mid-leg void window is real (23 grants were voided against a
    running leg at step-100).
    """

    from menagerie.crawler.recordio import JsonlLedger
    from menagerie.crawler.work_selection import SelectionDecision

    snapshot, stable_id = _single_member_terminal_campaign(tmp_path)
    grant_id = _append_grant(tmp_path, snapshot, stable_id)
    paths = _paths(tmp_path, snapshot)
    canonical_grants = canonical_requeue_grants_path(paths.ledgers.models)
    grant = {
        "grant_id": grant_id,
        "stable_id": stable_id,
        "stage": "accuracy-gate",
        "reason": "human reviewed corrected evidence",
        "attempts": 1,
        "created_at": NOW,
    }
    append_canonical_requeue_grant(canonical_grants, grant)
    heads = {row["stable_id"]: row for row in scan_jsonl(paths.ledgers.models)}
    source_revision = heads[stable_id]["record_revision"]
    new_work_id = stable_hash(
        {
            "stable_id": stable_id,
            "grant_id": grant_id,
            "parent_revision": source_revision,
            "generation": 1,
        }
    )
    # The void is minted while the consumption receipt is still RUN-LOCAL --
    # ``build_canonical_requeue_void`` reads the canonical events ledger only,
    # which is exactly why the mid-leg window exists -- and the receipt reaches
    # the canonical ledger afterwards.
    append_canonical_requeue_void(
        canonical_grants,
        build_canonical_requeue_void(
            canonical_grants,
            void_of=grant_id,
            reason="the re-adjudication was withdrawn mid-leg",
            voided_by="operator",
        ),
    )
    consumption = {
        "schema_version": "menagerie.crawler.operational-event.v1",
        "event_id": f"requeue-consumed-{grant_id[7:31]}",
        "created_at": NOW,
        "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
        "status": "requeue-grant-consumed",
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": "run-test",
        "machine_id": "machine-test",
        "details": {
            "grant_id": grant_id,
            "stable_id": stable_id,
            "stage": "accuracy-gate",
            "reason": "human reviewed corrected evidence",
            "attempts": 1,
            "source_record_revision": source_revision,
            "new_work_generation": 1,
            "new_work_id": new_work_id,
        },
    }
    with JsonlLedger(
        canonical_operational_ledger_path(paths.ledgers.models),
        "menagerie.crawler.operational-event.v1",
        lock_timeout=10.0,
    ) as ledger:
        ledger.append(consumption)

    forward = FakeForward()
    driver = _driver(tmp_path, snapshot, forward=forward)
    resumed = driver.run()
    assert forward.calls == {}, (
        "a VOIDED grant's consumed generation bought work the operator withdrew"
    )
    assert resumed.status == "terminal-partition-complete"
    plan = driver._selection_plan
    assert plan is not None
    decision = plan.decision_of(stable_id)
    assert decision is not None and not decision.admits
    # The generation is discharged at the ONE derivation, so the member falls
    # through to its head: the driver's facts null a DISCHARGED generation
    # rather than carrying it into the predicate, which is why the census
    # reads it as a settled head (a pre-existing disclosure gap, not a fresh
    # one -- the same reading a tool-issued discharge produces today).
    assert decision is SelectionDecision.SETTLED_HEAD
    assert len(_consumption_events(tmp_path, snapshot)) == 1, (
        "the voided grant was consumed a second time after its void"
    )
    revisions = [row for row in scan_jsonl(paths.ledgers.models) if row["stable_id"] == stable_id]
    assert [row["status"]["code"] for row in revisions] == ["failed:accuracy-gate"]


@pytest.mark.heavy
def test_a_consumption_receipt_cannot_forge_award_naming_authority(
    tmp_path: Path,
) -> None:
    """The generation's authority is the durable grant row, not its receipt copy.

    The active-generation arm reads ``blocked_award_revision`` out of the
    consumption receipt -- a producer-written copy -- while the receipt/grant
    agreement check covered stage, reason, attempts, and target_intent only.
    A receipt naming an award its grant never named would have been award-work
    authority nobody granted (four-part waiver doctrine part 3).
    """

    from menagerie.crawler.driver_contracts import DriverIntegrationError
    from menagerie.crawler.recordio import JsonlLedger

    snapshot, stable_id = _single_member_terminal_campaign(tmp_path)
    grant_id = _append_grant(tmp_path, snapshot, stable_id)
    paths = _paths(tmp_path, snapshot)
    canonical_grants = canonical_requeue_grants_path(paths.ledgers.models)
    append_canonical_requeue_grant(
        canonical_grants,
        {
            "grant_id": grant_id,
            "stable_id": stable_id,
            "stage": "accuracy-gate",
            "reason": "human reviewed corrected evidence",
            "attempts": 1,
            "created_at": NOW,
        },
    )
    heads = {row["stable_id"]: row for row in scan_jsonl(paths.ledgers.models)}
    source_revision = heads[stable_id]["record_revision"]
    new_work_id = stable_hash(
        {
            "stable_id": stable_id,
            "grant_id": grant_id,
            "parent_revision": source_revision,
            "generation": 1,
        }
    )
    with JsonlLedger(
        canonical_operational_ledger_path(paths.ledgers.models),
        "menagerie.crawler.operational-event.v1",
        lock_timeout=10.0,
    ) as ledger:
        ledger.append(
            {
                "schema_version": "menagerie.crawler.operational-event.v1",
                "event_id": f"requeue-consumed-{grant_id[7:31]}",
                "created_at": NOW,
                "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
                "status": "requeue-grant-consumed",
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 1},
                "current_environment": None,
                "run_id": "run-test",
                "machine_id": "machine-test",
                "details": {
                    "grant_id": grant_id,
                    "stable_id": stable_id,
                    "stage": "accuracy-gate",
                    "reason": "human reviewed corrected evidence",
                    "attempts": 1,
                    "source_record_revision": source_revision,
                    "new_work_generation": 1,
                    "new_work_id": new_work_id,
                    "blocked_award_revision": source_revision,
                },
            }
        )
    with pytest.raises(DriverIntegrationError, match="grant facts mismatch"):
        _driver(tmp_path, snapshot, forward=FakeForward()).run()
