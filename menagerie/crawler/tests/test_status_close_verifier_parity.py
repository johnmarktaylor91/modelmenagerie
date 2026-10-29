"""D-17 (LC-3): the CLI boundary gate and the driver's close verify as ONE.

``crawl status --verify-partition`` shipped under a comment claiming "same
production tripwire the driver's close uses" while diverging from the real
close in two confirmed ways (tri-lab synthesis §2-D6, both spot-checked):

* its claims view loaded seven of the eight machine-deferral sources -- the
  scheduler-quarantine ledger was absent, so a member whose only accounting
  was a quarantine row read as a coverage HOLE at the boundary while the
  close accounted it;
* it passed no ``stale_accounted_ids`` at all, while the close derives them
  from the durable park/valve ledgers -- so a stale head a durable row
  accounts refused at the boundary while the close completed (the leg-30
  shape, strictly more refusing).

These tests pin the unification: the CLI's partition inputs now come from
:func:`menagerie.crawler.cli._status_partition_accounting`, whose stale
accounting is the close's own
:func:`menagerie.crawler.driver.durable_readiness_accounting`. The matrix
drives BOTH derivations over one records root and demands identical verdicts,
including the refusing direction -- unification must never weaken the
tripwire.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler import cli
from menagerie.crawler.authority import AuthorityContext, validate_currency
from menagerie.crawler.checker_lane import (
    append_checker_lane_deferral_row,
    build_checker_lane_deferral_row,
    checker_lane_deferral_path,
)
from menagerie.crawler.driver import _scoped_stale_reasons, durable_readiness_accounting
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import (
    CanonicalReducer,
    _award_recorded_basis_context,
)
from menagerie.crawler.scheduler_parks import (
    append_scheduler_quarantine_row,
    build_scheduler_quarantine_row,
    scheduler_quarantine_path,
)
from menagerie.crawler.status import (
    UNCLASSIFIED_STALE_AXIS,
    PartitionError,
    assert_partition,
    stale_axis,
)
from menagerie.crawler.tests.conftest import (
    make_authority_context,
    make_failed_attempt,
    make_model,
)
from menagerie.crawler.tests.rt3_support import (
    build_repo,
    campaign_member_ids,
    records_root_of,
    seed_heads,
)

pytestmark = pytest.mark.smoke

CAMPAIGN = "c3-classics"

#: A stale reason in the exact shape `validate_currency` emits: the axis is
#: recoverable, and the partition buckets it instead of inventing one.
_STALE_REASON = "dependency-vector: stale author_dispatcher_identity (fixture rotation)"


def _append_quarantine(records_root: Path, stable_id: str) -> Mapping[str, Any]:
    """Append one live scheduler-quarantine row through the real builder."""

    row = build_scheduler_quarantine_row(
        stable_id=stable_id,
        work_id=f"w-{stable_id}",
        name=stable_id,
        campaign_id=CAMPAIGN,
        run_id="pytest-close",
        machine_id="pytest-host",
        created_at="2026-08-01T00:00:10Z",
        site="scheduler-round",
        reason_code="no-durable-routing-progress",
        fault_evidence="fixture scheduler fault",
    )
    return append_scheduler_quarantine_row(scheduler_quarantine_path(records_root), row)


def _append_checker_lane_valve(records_root: Path, stable_id: str) -> Mapping[str, Any]:
    """Append one ordinary (non-park) valve row through the real builder."""

    row = build_checker_lane_deferral_row(
        stable_id=stable_id,
        work_id=f"w-{stable_id}",
        name=stable_id,
        campaign_id=CAMPAIGN,
        run_id="pytest-close",
        machine_id="pytest-host",
        created_at="2026-08-01T00:00:10Z",
        gate_kind="metadata_batch",
        failure_class="partial-result",
        failure_evidence="fixture checker partial result",
        attempted_batch_sizes=(2, 1),
    )
    return append_checker_lane_deferral_row(
        checker_lane_deferral_path(records_root), row
    )


def _heads(records_root: Path) -> dict[str, Mapping[str, Any]]:
    """Project the fixture models shard onto its current head per stable ID."""

    path = records_root / "models" / "current-shard.jsonl"
    heads: dict[str, Mapping[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            heads[str(row["stable_id"])] = row
    return heads


def _verdict(
    intake_ids: tuple[str, ...],
    current: list[Mapping[str, Any]],
    *,
    machine_deferrals: Mapping[str, Mapping[str, Any]],
    ledger_heads: Mapping[str, Mapping[str, Any]],
    stale_reasons: Mapping[str, str],
    stale_accounted_ids: Optional[frozenset[str]],
) -> Optional[str]:
    """Run the one shared assertion; return its refusal text or None."""

    try:
        assert_partition(
            intake_ids,
            current,
            machine_deferrals=machine_deferrals,
            ledger_heads=ledger_heads,
            stale_reasons=stale_reasons,
            stale_accounted_ids=stale_accounted_ids,
        )
    except PartitionError as exc:
        return str(exc)
    return None


def _close_arm_inputs(
    records_root: Path,
    *,
    scoped_ids: frozenset[str],
    ledger_heads: Mapping[str, Mapping[str, Any]],
    run_claims: Mapping[str, Mapping[str, Any]],
):
    """Derive the close's exact partition inputs (driver.py completion path)."""

    accounting = durable_readiness_accounting(
        SimpleNamespace(
            paths=SimpleNamespace(
                ledgers=SimpleNamespace(models=records_root / "models" / "current-shard.jsonl")
            )
        ),
        scoped_ids=scoped_ids,
        ledger_heads=ledger_heads,
        claimed_ids=frozenset(run_claims),
    )
    return run_claims, accounting.stale_accounted_ids


def test_status_gate_and_close_render_one_verdict_across_every_source_kind(
    tmp_path: Path,
) -> None:
    """The full matrix: terminal, quarantine-claimed, stale+valve, stale+quarantine.

    The two RED cells this pin was born from: the quarantine-claimed headless
    member (absent from the CLI claims view entirely) and the stale-headed
    member whose durable capacity row the close accounts (the CLI passed no
    stale accounting at all). Both must now render the CLOSE's verdict.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    m_terminal, m_quarantine, m_stale_valve, m_stale_quarantine = ids[:4]
    intake = (m_terminal, m_quarantine, m_stale_valve, m_stale_quarantine)

    seed_heads(rr, [m_terminal, m_stale_valve, m_stale_quarantine])
    quarantine_row = _append_quarantine(rr, m_quarantine)
    _append_checker_lane_valve(rr, m_stale_valve)
    _append_quarantine(rr, m_stale_quarantine)

    heads = _heads(rr)
    stale_reasons = {
        m_stale_valve: _STALE_REASON,
        m_stale_quarantine: _STALE_REASON,
    }
    current = [heads[m_terminal]]  # the projection vouches only for the terminal

    close_claims, close_stale_accounted = _close_arm_inputs(
        rr,
        scoped_ids=frozenset(intake),
        ledger_heads=heads,
        run_claims={m_quarantine: dict(quarantine_row)},
    )
    close_verdict = _verdict(
        intake,
        current,
        machine_deferrals=close_claims,
        ledger_heads=heads,
        stale_reasons=stale_reasons,
        stale_accounted_ids=close_stale_accounted,
    )

    cli_claims, cli_stale_accounted = cli._status_partition_accounting(
        rr, scope_ids=intake, ledger_heads=heads
    )
    cli_verdict = _verdict(
        intake,
        current,
        machine_deferrals=cli_claims,
        ledger_heads=heads,
        stale_reasons=stale_reasons,
        stale_accounted_ids=cli_stale_accounted,
    )

    # The close completes this rung; the boundary gate must agree.
    assert close_verdict is None
    assert cli_verdict == close_verdict

    # The two confirmed divergences, pinned as cells:
    # (1) the scheduler-quarantine ledger is IN the CLI claims view;
    assert m_quarantine in cli_claims
    assert str(cli_claims[m_quarantine].get("disposition")) == str(
        quarantine_row["disposition"]
    )
    # (2) the durable stale accounting reaches the CLI exactly as the close
    # derives it for every STALE member (the only members the assertion
    # consumes it for; the close's run-claim exclusion may shrink its view of
    # non-stale members without any verdict effect).
    assert {m_stale_valve, m_stale_quarantine} <= cli_stale_accounted
    stale = {m_stale_valve, m_stale_quarantine}
    assert cli_stale_accounted & stale == close_stale_accounted & stale


def test_unification_does_not_weaken_the_refusing_direction(tmp_path: Path) -> None:
    """A member NOTHING accounts still refuses identically in both views.

    The D-17 change may only move the boundary gate TOWARD the close's
    verdict; a coverage hole (rung 9b's honest tripwire) and an unaccounted
    stale head must keep refusing in both, with the member named.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    m_terminal, m_hole, m_stale_bare = ids[:3]
    intake = (m_terminal, m_hole, m_stale_bare)

    seed_heads(rr, [m_terminal, m_stale_bare])
    heads = _heads(rr)
    stale_reasons = {m_stale_bare: _STALE_REASON}
    current = [heads[m_terminal]]

    close_claims, close_stale_accounted = _close_arm_inputs(
        rr,
        scoped_ids=frozenset(intake),
        ledger_heads=heads,
        run_claims={},
    )
    close_verdict = _verdict(
        intake,
        current,
        machine_deferrals=close_claims,
        ledger_heads=heads,
        stale_reasons=stale_reasons,
        stale_accounted_ids=close_stale_accounted,
    )

    cli_claims, cli_stale_accounted = cli._status_partition_accounting(
        rr, scope_ids=intake, ledger_heads=heads
    )
    cli_verdict = _verdict(
        intake,
        current,
        machine_deferrals=cli_claims,
        ledger_heads=heads,
        stale_reasons=stale_reasons,
        stale_accounted_ids=cli_stale_accounted,
    )

    assert close_verdict is not None and cli_verdict is not None
    assert m_hole in close_verdict and m_hole in cli_verdict
    assert m_stale_bare in close_verdict and m_stale_bare in cli_verdict
    assert cli_verdict == close_verdict


def test_status_machine_deferrals_covers_the_whole_closed_disposition_set(
    tmp_path: Path,
) -> None:
    """Set-equality tripwire: every accounted disposition has a loaded source.

    The class-closer for the defect shape: the CLI view's loaded ledgers must
    produce rows covering EXACTLY the closed
    :data:`~menagerie.crawler.status.MACHINE_DEFERRAL_DISPOSITIONS` set, so
    the next disposition added to the close's accounting cannot be silently
    absent from the boundary gate (the scheduler-quarantine omission was
    exactly this, undetected for a full rung).
    """

    from menagerie.crawler.status import MACHINE_DEFERRAL_DISPOSITIONS

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)

    from menagerie.crawler.tests.rt3_support import append_refusal_park

    # One row per family through the real builders/loaders; the dispositions
    # whose builders this fixture does not exercise (capacity, coverage,
    # rebuild, disk-floor, award park, grant park) are covered by the AST
    # half below.
    _append_quarantine(rr, ids[0])
    _append_checker_lane_valve(rr, ids[1])
    append_refusal_park(rr, ids[2])
    selected = cli._status_machine_deferrals(rr)
    observed = {str(row.get("disposition")) for row in selected.values()}
    assert observed <= MACHINE_DEFERRAL_DISPOSITIONS
    assert {
        "deferred:scheduler-fault-quarantined",
    } <= observed

    # AST half: the loader block in _status_machine_deferrals names one
    # loader per durable source the driver's own read
    # (_persisted_deferral_rows) consumes -- eight, including the
    # scheduler-quarantine ledger.
    import ast
    import inspect

    source = inspect.getsource(cli._status_machine_deferrals)
    loader_names = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "load_capacity_deferral_rows",
        "load_coverage_deferral_rows",
        "load_rebuild_deferral_rows",
        "load_disk_floor_deferral_rows",
        "load_checker_lane_deferral_rows",
        "load_scheduler_quarantine_rows",
        "load_grant_park_rows",
        "load_refusal_park_rows",
    } <= loader_names


#: A non-identity replay refusal: the recorded-basis replay refused the head
#: on a validator CODE, so no closed prefix names an identity axis and
#: :func:`stale_axis` classifies it ``unclassified``.
_NON_IDENTITY_REPLAY_REASON = (
    "terminal disposition rejected: failed:accuracy-gate first-minted with "
    "terminal-disposition-rejected and no standing verdict"
)


def _reduced_terminal_head(tmp_path: Path) -> tuple[dict[str, Any], AuthorityContext]:
    """Append one failed terminal through the real reducer; return (head, context)."""

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    context = make_authority_context(["m_example"])
    attempt = make_failed_attempt()
    model = make_model(accepted=False, status_code="failed:source")
    model["status"]["attempt_ids"] = [attempt["attempt_id"]]
    model["execution"]["accepted_attempt_ids"] = []
    with CanonicalReducer(paths, context) as reducer:
        reducer.append_attempt(attempt)
        record = reducer.append_model(reducer.prepare_model(model)).record
    return dict(record), context


def test_close_quotes_the_projection_reason_for_a_non_identity_replay_refusal(
    tmp_path: Path,
) -> None:
    """D-17 second label mechanism: the close never re-attributes to an identity axis.

    The pinned incident: 26 review-held heads whose every identity move was
    REGISTERED-refinement covered were refused by the projection on a
    non-identity replay class (a status-validator refusal of the head's own
    first mint), yet the close re-attributed them through registry-blind
    ``validate_currency`` and printed ``stale:author_dispatcher_identity``
    with the register-the-rotation remedy -- wrong axis, unactionable (the
    rotation was already registered) -- while ``status --verify-partition``
    printed ``stale:unclassified`` for the SAME members. The close must quote
    the projection's OWN reason, so both surfaces render one label.
    """

    head, context = _reduced_terminal_head(tmp_path)
    stable_id = str(head["stable_id"])
    rotated = replace(context, author_dispatcher_identity="sha256:" + "d" * 64)
    # The registry-blind per-record authority names the identity axis first --
    # exactly the label the close used to headline...
    assert validate_currency(rotated, head) == (
        "dependency-vector: stale author_dispatcher_identity"
    )
    # ...while the projection's own refusal carries no identity axis at all.
    assert stale_axis(_NON_IDENTITY_REPLAY_REASON) == UNCLASSIFIED_STALE_AXIS

    close_reasons = _scoped_stale_reasons(
        rotated,
        {stable_id: head},
        frozenset(),
        {stable_id: _NON_IDENTITY_REPLAY_REASON},
    )
    assert close_reasons == {stable_id: _NON_IDENTITY_REPLAY_REASON}

    intake = (stable_id,)
    close_verdict = _verdict(
        intake,
        [],
        machine_deferrals={},
        ledger_heads={stable_id: head},
        stale_reasons=close_reasons,
        stale_accounted_ids=None,
    )
    # The boundary gate passes the projection's map verbatim (cli.py
    # `stale_reasons=projection.stale_reasons`); with the close quoting the
    # same map, the two surfaces render ONE refusal for one mechanism.
    boundary_verdict = _verdict(
        intake,
        [],
        machine_deferrals={},
        ledger_heads={stable_id: head},
        stale_reasons={stable_id: _NON_IDENTITY_REPLAY_REASON},
        stale_accounted_ids=None,
    )
    assert close_verdict is not None
    assert close_verdict == boundary_verdict
    assert f"stale:{UNCLASSIFIED_STALE_AXIS}=['{stable_id}']" in close_verdict
    assert "author_dispatcher_identity" not in close_verdict


def test_a_policy_axis_move_pins_the_recorded_basis_and_the_fallback_authority_still_names_the_axis(
    tmp_path: Path,
) -> None:
    """RC-2: a settled head replays at its recorded basis for EVERY policy move.

    The projection no longer emits an award-release-hold for an unregistered
    or soundness rotation -- the move's class feeds the recheck census, never
    currency -- so the recorded-basis context pins the recorded identity and
    the head projects current. A head OUTSIDE the projection's domain (no
    projection reason to quote) still falls back to the exact per-record
    identity authority, which names the moved axis.
    """

    head, context = _reduced_terminal_head(tmp_path)
    stable_id = str(head["stable_id"])
    rotated = replace(context, author_dispatcher_identity="sha256:" + "d" * 64)

    pinned = _award_recorded_basis_context(rotated, head, attested_bindings={})
    assert pinned.author_dispatcher_identity == head["dependency_vector"]["author_dispatcher_identity"]
    assert stale_axis("dependency-vector: stale author_dispatcher_identity") == (
        "author_dispatcher_identity"
    )

    fallback = _scoped_stale_reasons(rotated, {stable_id: head}, frozenset(), {})
    assert fallback == {stable_id: "dependency-vector: stale author_dispatcher_identity"}
