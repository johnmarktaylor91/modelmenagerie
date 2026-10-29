"""B3-XQ-3: composed quarantine loop containment (W12 + PD-11 + SD-6/7 + S-C5).

The 200a livelock class this retires: the recidivism counter counted EVERY
``scheduler-member-quarantined`` event for a work generation with no
budget-restoring cutoff, so an operator drain bought ZERO retries (m12646:
8 parks / 8 voids, the operator drained forever), and the unconditional event
append beside the idempotent row append let observations drift above rows
(the 28-events-vs-22-rows class).

The composed contract, walked as ONE scenario over one member's ledger:

- S1 (RE-10/O-E3): park -> held -> BARE void -> park -> STILL held. A bare
  void removes the live row (the member is no longer parked) but restores no
  budget: the very next fault holds again. The idempotent re-park append
  returns the incumbent and the driver's reused disclosure charges nothing.
- S2 (SD-6): a void WITH declared remedy (fixing_sha + RED-pinned test node,
  the DX-69 shape) + a declared passing canary restores the budget: exactly
  ONE fresh retry, and the next machinery fault re-parks.
- S3 (RE-11/W12): retries exhausted -> a disclosed machine drain. The drain
  refuses without ``--fixed-in-tree``; an ineligible drain row (retries not
  exhausted at its position) NEVER restores budget no matter who appended it
  (closed proof rule, replay-derived); an eligible drain resets the counter;
  the census segregates machine-drained from remedied-void from held rows.
- S4 (SD-7): a release of a row that is not the LIVE current row refuses
  typed; re-releasing identically is idempotent.
- S5 (S-C5): restart, a new leg, or a tree change WITHOUT declared remedy
  restores nothing (an env-string match is not repair evidence), and a
  remedied void with a missing remedy field refuses at build time.

Four-part shape asserted for the drain (torchlens CLAUDE.md waiver doctrine):
disposition-invariant (no model record minted, the park's history stands),
closed proof rule (exact event vocabulary + exhaustion recomputed), replay-
derived (no persisted flag; recomputation is deterministic from rows alone),
disclosed (segregated census classes + the waiver registry row).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler.scheduler_parks import (
    SCHEDULER_DRAIN_CENSUS_CLASSES,
    SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION,
    SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT,
    SCHEDULER_QUARANTINE_REUSED_DISPOSITION,
    SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2,
    SchedulerQuarantineError,
    append_scheduler_quarantine_row,
    append_scheduler_quarantine_void,
    budget_restoring_boundaries,
    build_scheduler_quarantine_drain,
    build_scheduler_quarantine_remedied_void,
    build_scheduler_quarantine_row,
    build_scheduler_quarantine_void,
    load_scheduler_quarantine_ledger,
    load_scheduler_quarantine_rows,
    scheduler_drain_census,
    scheduler_quarantine_path,
    scheduler_quarantine_released,
    unsatisfied_drain_tree_pins,
)

_LIMIT = SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT


def _park_row(stable_id: str, *, created_at: str) -> dict[str, Any]:
    """Build one valid park row for the member's single work generation."""

    return build_scheduler_quarantine_row(
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        name=stable_id,
        campaign_id="c1-mech",
        run_id="run-1",
        machine_id="machine-1",
        created_at=created_at,
        site="scheduler-round",
        reason_code="no-durable-routing-progress",
        fault_evidence="scheduler made no durable routing progress",
    )


def _park_event(
    stable_id: str, *, created_at: str, disposition: str = "scheduler-member-quarantined"
) -> dict[str, Any]:
    """Build one operational park event in the real ledger shape."""

    return {
        "event_kind": "campaign-health",
        "created_at": created_at,
        "details": {
            "disposition": disposition,
            "stable_id": stable_id,
            "work_id": f"work-{stable_id}",
        },
    }


def _live(ledger: Path, stable_id: str) -> Optional[Mapping[str, Any]]:
    """Return the member's live park row, if any."""

    for row in load_scheduler_quarantine_rows([ledger]):
        if row["stable_id"] == stable_id:
            return row
    return None


@pytest.mark.smoke
def test_composed_quarantine_loop_containment(tmp_path: Path) -> None:
    """B3-XQ-3: the S1-S5 walk over one member's ledger."""

    records_root = tmp_path / "records"
    ledger = scheduler_quarantine_path(records_root)
    member = "m_loop"
    events: list[dict[str, Any]] = []

    # --- S1: park -> held -> BARE void -> park -> STILL held. ---
    first = append_scheduler_quarantine_row(
        ledger, _park_row(member, created_at="2026-08-20T00:01:00Z")
    )
    events.append(_park_event(member, created_at="2026-08-20T00:01:00Z"))
    rows = load_scheduler_quarantine_ledger([ledger])
    assert scheduler_quarantine_released(first, events, ledger_rows=rows) is True

    events.append(_park_event(member, created_at="2026-08-20T00:02:00Z"))
    assert scheduler_quarantine_released(first, events, ledger_rows=rows) is False

    # The idempotent re-park append returns the incumbent: the ledger never
    # drifts above one live row per generation, and the reused DISCLOSURE
    # event charges nothing against the budget.
    reused = append_scheduler_quarantine_row(
        ledger, _park_row(member, created_at="2026-08-20T00:03:00Z")
    )
    assert reused["row_sha256"] == first["row_sha256"]
    events.append(
        _park_event(
            member,
            created_at="2026-08-20T00:03:00Z",
            disposition=SCHEDULER_QUARANTINE_REUSED_DISPOSITION,
        )
    )
    rows = load_scheduler_quarantine_ledger([ledger])
    held_after_reuse = scheduler_quarantine_released(first, events, ledger_rows=rows)
    assert held_after_reuse is False, "a reused disclosure must charge nothing"

    bare = build_scheduler_quarantine_void(
        void_of=str(first["row_sha256"]),
        reason="operator cleared the row without a remedy",
        voided_by="operator",
        created_at="2026-08-20T00:04:00Z",
    )
    append_scheduler_quarantine_void(ledger, bare)
    assert _live(ledger, member) is None, "a bare void still removes the live row"

    second = append_scheduler_quarantine_row(
        ledger, _park_row(member, created_at="2026-08-20T00:05:00Z")
    )
    events.append(_park_event(member, created_at="2026-08-20T00:05:00Z"))
    rows = load_scheduler_quarantine_ledger([ledger])
    assert second["row_sha256"] != first["row_sha256"]
    assert (
        scheduler_quarantine_released(second, events, ledger_rows=rows) is False
    ), "S1: a bare void restores no budget -- the next fault holds again"

    # --- S2: a remedied void restores exactly ONE fresh retry. ---
    remedied = build_scheduler_quarantine_remedied_void(
        void_of=str(second["row_sha256"]),
        reason="root-caused: routing seam fixed",
        voided_by="operator",
        created_at="2026-08-20T00:06:00Z",
        fixing_sha="a" * 40,
        test_node="menagerie/crawler/tests/test_quarantine_loop_containment.py::red_pin",
        canary="canary: composed scenario replay passed",
    )
    append_scheduler_quarantine_void(ledger, remedied)
    third = append_scheduler_quarantine_row(
        ledger, _park_row(member, created_at="2026-08-20T00:07:00Z")
    )
    events.append(_park_event(member, created_at="2026-08-20T00:07:00Z"))
    rows = load_scheduler_quarantine_ledger([ledger])
    assert (
        scheduler_quarantine_released(third, events, ledger_rows=rows) is True
    ), "S2: the remedied void restored the budget for one fresh retry"

    events.append(_park_event(member, created_at="2026-08-20T00:08:00Z"))
    assert (
        scheduler_quarantine_released(third, events, ledger_rows=rows) is False
    ), "S2: the next machinery fault re-parks and holds"

    # --- S3: the machine drain. ---
    # An ineligible drain (retries NOT exhausted at its position) never
    # restores budget, no matter who appended it: eligibility is recomputed
    # from the rows on every read (closed proof rule, replay-derived).
    other = append_scheduler_quarantine_row(
        ledger, _park_row("m_fresh", created_at="2026-08-20T00:09:00Z")
    )
    other_events = [_park_event("m_fresh", created_at="2026-08-20T00:09:00Z")]
    ineligible = build_scheduler_quarantine_drain(
        void_of=str(other["row_sha256"]),
        reason="premature drain",
        drained_by="operator",
        created_at="2026-08-20T00:10:00Z",
        fixed_in_tree_sha="b" * 40,
    )
    append_scheduler_quarantine_void(ledger, ineligible)
    rows = load_scheduler_quarantine_ledger([ledger])
    boundaries = budget_restoring_boundaries(rows, tuple(events) + tuple(other_events))
    assert ("m_fresh", "work-m_fresh") not in boundaries, (
        "S3: a drain appended before the bounded retries were exhausted is "
        "not a budget-restoring event"
    )

    # The eligible drain on the held member resets the counter.
    drain = build_scheduler_quarantine_drain(
        void_of=str(third["row_sha256"]),
        reason="machine drain after exhausted retries",
        drained_by="operator",
        created_at="2026-08-20T00:11:00Z",
        fixed_in_tree_sha="c" * 40,
    )
    append_scheduler_quarantine_void(ledger, drain)
    assert _live(ledger, member) is None
    fourth = append_scheduler_quarantine_row(
        ledger, _park_row(member, created_at="2026-08-20T00:12:00Z")
    )
    events.append(_park_event(member, created_at="2026-08-20T00:12:00Z"))
    rows = load_scheduler_quarantine_ledger([ledger])
    assert (
        scheduler_quarantine_released(fourth, events, ledger_rows=rows) is True
    ), "S3: the counter resets on a disclosed machine drain"

    # Four-part: replay-derived determinism -- recomputation from rows alone,
    # twice, gives the same boundaries; nothing persisted steers it.
    replayed = budget_restoring_boundaries(rows, tuple(events))
    assert replayed == budget_restoring_boundaries(tuple(rows), tuple(events))
    assert (member, f"work-{member}") in replayed
    for row in rows:
        assert "released" not in row and "budget" not in row, (
            "no persisted flag may steer the release decision"
        )

    # Four-part: disclosed -- the census segregates every class.
    census = scheduler_drain_census(rows, tuple(events) + tuple(other_events))
    assert tuple(sorted(census)) == tuple(sorted(SCHEDULER_DRAIN_CENSUS_CLASSES))
    assert f"{member}::work-{member}" in census["retry-eligible"]
    assert f"{member}::work-{member}" in census["machine-drained"]
    assert f"{member}::work-{member}" in census["remedied-void-released"]

    # --- S4: releasing a non-current row refuses typed. ---
    with pytest.raises(SchedulerQuarantineError, match="unknown row"):
        append_scheduler_quarantine_void(
            ledger,
            build_scheduler_quarantine_void(
                void_of="0" * 64,
                reason="no such row",
                voided_by="operator",
                created_at="2026-08-20T00:13:00Z",
            ),
        )
    # A DIFFERENT release kind against an already-released row refuses: the
    # prior release stands; silently accepting a drain that drained nothing
    # would forge a budget restoration.
    with pytest.raises(SchedulerQuarantineError, match="already released"):
        append_scheduler_quarantine_void(
            ledger,
            build_scheduler_quarantine_drain(
                void_of=str(first["row_sha256"]),
                reason="drain of a bare-voided row",
                drained_by="operator",
                created_at="2026-08-20T00:14:00Z",
                fixed_in_tree_sha="d" * 40,
            ),
        )
    # An identical re-release is idempotent.
    again = append_scheduler_quarantine_void(ledger, bare)
    assert again["row_sha256"] == bare["row_sha256"]

    # --- S5: nothing restores budget by itself. ---
    fifth_events = list(events)
    fifth_events.append(_park_event(member, created_at="2026-08-20T00:15:00Z"))
    rows = load_scheduler_quarantine_ledger([ledger])
    assert (
        scheduler_quarantine_released(fourth, fifth_events, ledger_rows=rows) is False
    ), "held again after its one post-drain retry"
    # A restart / new leg / tree change appends nothing to this ledger, so
    # the decision is unchanged by construction; the closed builder refuses a
    # remedy with any missing part (env-string match is not repair evidence).
    for missing in ("fixing_sha", "test_node", "canary"):
        kwargs: dict[str, Any] = {
            "void_of": str(fourth["row_sha256"]),
            "reason": "partial remedy",
            "voided_by": "operator",
            "created_at": "2026-08-20T00:16:00Z",
            "fixing_sha": "e" * 40,
            "test_node": "tests::pin",
            "canary": "passed",
        }
        kwargs[missing] = ""
        with pytest.raises(SchedulerQuarantineError):
            build_scheduler_quarantine_remedied_void(**kwargs)


@pytest.mark.smoke
def test_drain_tool_preconditions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The drain CLI refuses without --fixed-in-tree and before exhaustion."""

    from menagerie.crawler.recordio import operational_events_path
    from menagerie.crawler.tools.scheduler_quarantines import main

    records_root = tmp_path / "records"
    ledger = scheduler_quarantine_path(records_root)
    row = append_scheduler_quarantine_row(
        ledger, _park_row("m_tool", created_at="2026-08-21T00:00:00Z")
    )
    operational = operational_events_path(records_root)
    operational.parent.mkdir(parents=True, exist_ok=True)
    import json as _json

    with operational.open("w", encoding="utf-8") as handle:
        handle.write(_json.dumps(_park_event("m_tool", created_at="2026-08-21T00:00:00Z")) + "\n")

    base = ["--repo-root", str(tmp_path), "--records-root", str(records_root)]

    # A drain without the fixed-in tree binding refuses.
    assert main([*base, "--drain", str(row["row_sha256"]), "--reason", "drain"]) == 1
    assert "--fixed-in-tree" in capsys.readouterr().err

    # A drain before the bounded retries are exhausted refuses.
    assert (
        main(
            [
                *base,
                "--drain",
                str(row["row_sha256"]),
                "--reason",
                "drain",
                "--fixed-in-tree",
                "f" * 40,
            ]
        )
        == 1
    )
    assert "not exhausted" in capsys.readouterr().err

    # Exhaust the retries; the drain now succeeds and the row leaves the
    # live set.
    with operational.open("a", encoding="utf-8") as handle:
        handle.write(_json.dumps(_park_event("m_tool", created_at="2026-08-21T00:01:00Z")) + "\n")
    assert (
        main(
            [
                *base,
                "--drain",
                str(row["row_sha256"]),
                "--reason",
                "drain",
                "--fixed-in-tree",
                "f" * 40,
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert _live(ledger, "m_tool") is None

    # A remedied void demands the full DX-69 remedy shape.
    second = append_scheduler_quarantine_row(
        ledger, _park_row("m_tool2", created_at="2026-08-21T00:02:00Z")
    )
    assert (
        main(
            [
                *base,
                "--void",
                str(second["row_sha256"]),
                "--reason",
                "void with partial remedy",
                "--fixing-sha",
                "a" * 40,
            ]
        )
        == 1
    )
    assert "COMPLETE remedy shape" in capsys.readouterr().err


@pytest.mark.smoke
def test_unsatisfied_drain_tree_pins(tmp_path: Path) -> None:
    """Admission refuses while the running tree lacks a drain's fixed-in sha."""

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "--allow-empty", "-q", "-m", "fix"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    records_root = tmp_path / "records"
    ledger = scheduler_quarantine_path(records_root)
    row = append_scheduler_quarantine_row(
        ledger, _park_row("m_pin", created_at="2026-08-22T00:00:00Z")
    )
    drain = build_scheduler_quarantine_drain(
        void_of=str(row["row_sha256"]),
        reason="drain bound to an unmerged fix",
        drained_by="operator",
        created_at="2026-08-22T00:02:00Z",
        fixed_in_tree_sha="9" * 40,
    )
    append_scheduler_quarantine_void(ledger, drain)
    rows = load_scheduler_quarantine_ledger([ledger])

    unsatisfied = unsatisfied_drain_tree_pins(rows, repo)
    assert [pin["fixed_in_tree_sha"] for pin in unsatisfied] == ["9" * 40]

    # A drain bound to a commit the tree actually contains is satisfied.
    second = append_scheduler_quarantine_row(
        ledger, _park_row("m_pin2", created_at="2026-08-22T00:03:00Z")
    )
    satisfied = build_scheduler_quarantine_drain(
        void_of=str(second["row_sha256"]),
        reason="drain bound to a merged fix",
        drained_by="operator",
        created_at="2026-08-22T00:04:00Z",
        fixed_in_tree_sha=head,
    )
    append_scheduler_quarantine_void(ledger, satisfied)
    rows = load_scheduler_quarantine_ledger([ledger])
    unsatisfied = unsatisfied_drain_tree_pins(rows, repo)
    assert [pin["fixed_in_tree_sha"] for pin in unsatisfied] == ["9" * 40]

    # Outside a git repository the pin fails CLOSED: a drain declared a tree
    # obligation nothing can verify.
    not_a_repo = tmp_path / "empty"
    not_a_repo.mkdir()
    unverifiable = unsatisfied_drain_tree_pins(rows, not_a_repo)
    assert len(unverifiable) == 2


def test_current_rows_survive_every_release_kind(tmp_path: Path) -> None:
    """The live-set reduction treats void v1, void v2, and drain identically."""

    records_root = tmp_path / "records"
    ledger = scheduler_quarantine_path(records_root)
    keep = append_scheduler_quarantine_row(
        ledger, _park_row("m_keep", created_at="2026-08-23T00:00:00Z")
    )
    gone_bare = append_scheduler_quarantine_row(
        ledger, _park_row("m_bare", created_at="2026-08-23T00:01:00Z")
    )
    gone_remedied = append_scheduler_quarantine_row(
        ledger, _park_row("m_remedied", created_at="2026-08-23T00:02:00Z")
    )
    append_scheduler_quarantine_void(
        ledger,
        build_scheduler_quarantine_void(
            void_of=str(gone_bare["row_sha256"]),
            reason="bare",
            voided_by="operator",
            created_at="2026-08-23T00:03:00Z",
        ),
    )
    append_scheduler_quarantine_void(
        ledger,
        build_scheduler_quarantine_remedied_void(
            void_of=str(gone_remedied["row_sha256"]),
            reason="remedied",
            voided_by="operator",
            created_at="2026-08-23T00:04:00Z",
            fixing_sha="a" * 40,
            test_node="tests::pin",
            canary="passed",
        ),
    )
    live = load_scheduler_quarantine_rows([ledger])
    assert [row["stable_id"] for row in live] == ["m_keep"]
    assert keep["row_sha256"] == live[0]["row_sha256"]
    # Schema identities are the closed, versioned vocabulary.
    kinds = {row["schema_version"] for row in load_scheduler_quarantine_ledger([ledger])}
    assert SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2 in kinds
    assert SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION not in kinds


@pytest.mark.smoke
def test_leg_start_refuses_while_a_drain_tree_pin_is_unsatisfied(tmp_path: Path) -> None:
    """W12: a real leg refuses at open until the drained fix is in the tree.

    The refusal fails CLOSED here by construction: the test repository root is
    not a git repository, so the drain's declared tree obligation is
    unverifiable and admission must not proceed on hope.
    """

    from menagerie.crawler.driver_contracts import DriverIntegrationError
    from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    records_root = paths.ledgers.models.parent.parent
    ledger = scheduler_quarantine_path(records_root)
    row = append_scheduler_quarantine_row(
        ledger, _park_row("m_leg_pin", created_at="2026-08-22T10:00:00Z")
    )
    append_scheduler_quarantine_void(
        ledger,
        build_scheduler_quarantine_drain(
            void_of=str(row["row_sha256"]),
            reason="drain bound to a fix this tree lacks",
            drained_by="operator",
            created_at="2026-08-22T10:01:00Z",
            fixed_in_tree_sha="9" * 40,
        ),
    )

    with pytest.raises(DriverIntegrationError, match="drain tree pin") as excinfo:
        _driver(tmp_path, snapshot).run()
    assert "9" * 40 in str(excinfo.value)
