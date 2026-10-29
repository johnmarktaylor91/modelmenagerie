"""D-48: the park exit is mechanically reachable (the record-less release lane).

The RED core is the no-head-park admission fixture (PLAN_v6 §2 D-48): on
pre-D-48 main, clause-5 member-keyed absolute parks fire BEFORE any grant-hold,
``_park_released_for_selection`` releases only the grant/award dispositions,
and ``refusal_park_released`` keys on a landed revision that can only land
through the admission clause 5 refuses -- so a terminal-refusal park's release
mechanism was mechanically unreachable while the refusal-park module docstring
asserted the opposite ("the next leg attempts the model normally"). D1-OS is
decided definitively by ``test_no_head_park_release_was_mechanically_unreachable``.

The fix under test: a member whose EVERY live park is in the closed
machinery-fault class readmits for ONE fresh generation per change of the
running code tree, on ordinary legs only; the re-park records the tree it
re-parked under (the tree joins the park FACT identity) and closes the lane
until the next tree change; every live park names its computed release path at
close; requeue refuses minting provably-unbindable grants.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from menagerie.crawler.grant_parks import GRANT_PENDING_DISPOSITION
from menagerie.crawler.park_release import (
    MACHINERY_FAULT_PARK_DISPOSITIONS,
    RELEASE_PATH_MACHINE_RETRY_NOW,
    RELEASE_PATH_MACHINE_RETRY_ON_TREE_CHANGE,
    RELEASE_PATH_OPERATOR_GRANT_STAGE,
    RELEASE_PATH_VOCABULARY,
    ParkReleasePathError,
    compute_release_path,
    derive_park_machine_readmission,
    park_release_path_census,
)
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_DISPOSITION,
    append_refusal_park_row,
    build_refusal_park_row,
    load_refusal_park_rows,
)
from menagerie.crawler.work_selection import (
    MemberFacts,
    SelectionDecision,
    SelectionMode,
    build_selection_plan,
    work_selection_predicate,
)


def _park_row(
    stable_id: str = "m-parked",
    *,
    tree_sha: str | None = None,
    work_id: str = "work-1",
    record_error: str = "reducer refused the candidate",
) -> dict:
    """Build one validated terminal-refusal park row for the lane fixtures."""

    return build_refusal_park_row(
        stable_id=stable_id,
        work_id=work_id,
        name="Parked Model",
        campaign_id="camp-1",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-24T00:00:00Z",
        refusal_class="ReductionError",
        record_error_type="ReductionError",
        record_error=record_error,
        fallback_error_type=None,
        fallback_error=None,
        intended_status_code="failed:runner",
        intended_reason_code="internal-error",
        blocked_parent_revision=None,
        candidate_sha256=None,
        tree_sha=tree_sha,
    )


def _facts(**overrides: object) -> MemberFacts:
    """Return the no-head parked member's selection facts."""

    base: dict = {
        "stable_id": "m-parked",
        "raw_head": None,
        "dependency_current": False,
        "stale_reason": None,
        "parked": True,
        "deferral_live": False,
        "live_grant": None,
        "grant_bindable": False,
        "active_generation": None,
        "active_generation_discharged": False,
    }
    base.update(overrides)
    return MemberFacts(**base)


@pytest.mark.smoke
def test_no_head_park_release_was_mechanically_unreachable() -> None:
    """The D-48 RED core: clause 5 fires before ANY grant-hold, so a no-head

    refusal-parked member has no machine path to the landed revision its own
    release predicate requires -- and the bounded machine lane is the exit.
    """

    grant = {"grant_id": "g-1", "stage": "runner"}
    # Clause 5 beats the grant arm: even a live, bindable grant admits nothing.
    held = work_selection_predicate(
        _facts(live_grant=grant, grant_bindable=True), mode=SelectionMode.ORDINARY
    )
    assert not held.admitted and held.decision is SelectionDecision.PARKED
    # ... in every named mode too.
    for mode in (SelectionMode.REDRIVE, SelectionMode.GATE_CATCHUP):
        named = work_selection_predicate(
            _facts(live_grant=grant, grant_bindable=True),
            mode=mode,
            mode_targets=frozenset({"m-parked"}),
        )
        assert not named.admitted and named.decision is SelectionDecision.PARKED
    # The lane: a replay-derived machine readmission admits for ONE fresh
    # generation, carrying the governing park row as its work authority.
    fact = {"row_sha256": "deadbeef", "parked_tree_sha": None, "current_tree_sha": "tree-b"}
    readmitted = work_selection_predicate(
        _facts(park_machine_readmission=fact), mode=SelectionMode.ORDINARY
    )
    assert readmitted.admitted
    assert readmitted.decision is SelectionDecision.PARK_MACHINE_RETRY
    assert readmitted.work_authority == "deadbeef"


@pytest.mark.smoke
def test_machine_readmission_never_widens_operator_scope_cuts() -> None:
    """The lane fires on ordinary legs only; cuts and named modes stay parked."""

    fact = {"row_sha256": "deadbeef", "parked_tree_sha": None, "current_tree_sha": "tree-b"}
    for mode, targets in (
        (SelectionMode.ONLY_UNSETTLED, frozenset()),
        (SelectionMode.ONLY_STATUS, frozenset()),
        (SelectionMode.REDRIVE, frozenset({"m-parked"})),
        (SelectionMode.GATE_CATCHUP, frozenset({"m-parked"})),
    ):
        selection = work_selection_predicate(
            _facts(park_machine_readmission=fact), mode=mode, mode_targets=targets
        )
        assert not selection.admitted, mode
        assert selection.decision is SelectionDecision.PARKED


@pytest.mark.smoke
def test_machinery_fault_readmission_is_one_generation_per_tree_change() -> None:
    """The repark pair: tree change opens the lane once; the re-park closes it."""

    parked_under_a = _park_row(tree_sha="tree-a")
    # Same tree: the refusal already spent this tree's generation -- closed.
    assert (
        derive_park_machine_readmission([parked_under_a], tree_identity="tree-a") is None
    )
    # Changed tree: one fresh generation opens, naming the governing row.
    fact = derive_park_machine_readmission([parked_under_a], tree_identity="tree-b")
    assert fact is not None
    assert fact["row_sha256"] == parked_under_a["row_sha256"]
    assert fact["parked_tree_sha"] == "tree-a"
    assert fact["current_tree_sha"] == "tree-b"
    # The idempotent re-park under tree-b records tree-b and closes the lane.
    reparked_under_b = _park_row(tree_sha="tree-b")
    assert (
        derive_park_machine_readmission([reparked_under_b], tree_identity="tree-b") is None
    )
    assert (
        derive_park_machine_readmission([reparked_under_b], tree_identity="tree-c")
        is not None
    )


@pytest.mark.smoke
def test_readmission_fails_closed_and_respects_operator_parks() -> None:
    """Underivable trees never readmit; any operator park vetoes the lane."""

    row = _park_row(tree_sha="tree-a")
    # Fail-closed: an underivable running tree cannot prove the causal remedy.
    assert derive_park_machine_readmission([row], tree_identity=None) is None
    # Pre-D-48 era row (no recorded tree): readmits once under any provable tree.
    pre_era = {key: value for key, value in row.items() if key != "tree_sha"}
    assert derive_park_machine_readmission([pre_era], tree_identity="tree-b") is not None
    # A grant-pending (operator) park beside the machinery row vetoes outright.
    operator_row = {"disposition": GRANT_PENDING_DISPOSITION, "row_sha256": "op-1"}
    assert (
        derive_park_machine_readmission([row, operator_row], tree_identity="tree-b") is None
    )
    assert derive_park_machine_readmission([], tree_identity="tree-b") is None


@pytest.mark.smoke
def test_repark_under_a_new_tree_appends_a_fresh_governing_row(tmp_path: Path) -> None:
    """The parking tree joins the park FACT identity.

    Byte-identical refusals under one tree stay idempotent; the same refusal
    re-derived under a NEW tree appends a fresh governing row whose recorded
    tree closes the lane again.
    """

    ledger = tmp_path / "terminal-refusal-parks.jsonl"
    first = append_refusal_park_row(ledger, _park_row(tree_sha="tree-a"))
    replay = append_refusal_park_row(ledger, _park_row(tree_sha="tree-a"))
    assert replay["row_sha256"] == first["row_sha256"], "same fact, same tree: no-op"
    reparked = append_refusal_park_row(ledger, _park_row(tree_sha="tree-b"))
    assert reparked["row_sha256"] != first["row_sha256"], "new tree: new governing fact"
    # RULING W7 clause 9 (per-row liveness): BOTH facts stay live -- the fresh
    # row never silently erases the earlier unvoided one. The lane's governing
    # selection keys on the NEWEST live row, so the new tree still closes it.
    live = load_refusal_park_rows([ledger])
    assert [row["row_sha256"] for row in live] == [
        first["row_sha256"],
        reparked["row_sha256"],
    ]
    assert live[-1]["tree_sha"] == "tree-b"


@pytest.mark.smoke
def test_recheck_hint_names_the_real_levers() -> None:
    """The hint no longer promises a requeue release clause 5 makes unreachable."""

    hint = _park_row()["recheck_hint"]
    assert "machine release lane" in hint
    assert "voids this park row explicitly" in hint
    assert "requeue grants cannot bind it" in hint
    assert "re-drive or requeue" not in hint, "the pre-D-48 false promise"


@pytest.mark.smoke
def test_release_path_census_is_total_over_the_closed_park_vocabulary() -> None:
    """Every live park computes exactly one closed release path; unknown refuses."""

    machinery_now = _park_row(tree_sha="tree-a")
    machinery_closed = _park_row(tree_sha="tree-b")
    assert (
        compute_release_path(machinery_now, tree_identity="tree-b")
        == RELEASE_PATH_MACHINE_RETRY_NOW
    )
    assert (
        compute_release_path(machinery_closed, tree_identity="tree-b")
        == RELEASE_PATH_MACHINE_RETRY_ON_TREE_CHANGE
    )
    census = park_release_path_census(
        {
            REFUSAL_PARK_DISPOSITION: {"m-parked": machinery_now},
            GRANT_PENDING_DISPOSITION: {"m-held": {"disposition": GRANT_PENDING_DISPOSITION}},
        },
        tree_identity="tree-b",
    )
    assert census[REFUSAL_PARK_DISPOSITION]["m-parked"] == RELEASE_PATH_MACHINE_RETRY_NOW
    assert census[GRANT_PENDING_DISPOSITION]["m-held"] == RELEASE_PATH_OPERATOR_GRANT_STAGE
    for paths in census.values():
        assert set(paths.values()) <= RELEASE_PATH_VOCABULARY
    with pytest.raises(ParkReleasePathError, match="outside the closed park vocabulary"):
        compute_release_path(
            {"disposition": "review-held:never-registered", "row_sha256": "x"},
            tree_identity="tree-b",
        )
    with pytest.raises(ParkReleasePathError):
        park_release_path_census(
            {"review-held:never-registered": {"m-x": {"disposition": "review-held:never-registered"}}},
            tree_identity="tree-b",
        )


@pytest.mark.smoke
def test_close_wires_the_release_path_assertion_and_selection_wires_the_lane() -> None:
    """Source-binding pins: the derivations are consumed at their real seams."""

    import menagerie.crawler.driver as driver_module

    close_source = inspect.getsource(driver_module.CrawlerDriver._run_locked)
    assert "park_release_path_census(" in close_source, "close release-path assertion"
    assert '"park_release_paths"' in close_source, "close currency disclosure key"
    # D-49-style parity contract: the honest fixture close runs the same
    # derivation and refuses the same uncomputable paths.
    import menagerie.crawler.tests.close_object_support as close_support

    support_source = inspect.getsource(close_support)
    assert "park_release_path_census(" in support_source, "fixture-close parity"
    selection_source = inspect.getsource(driver_module.CrawlerDriver._select_work)
    assert "derive_park_machine_readmission(" in selection_source
    assert "current_park_tree_identity()" in selection_source
    assert "park_machine_readmission=park_readmissions.get(stable_id)" in selection_source
    # The fresh-generation half: a park-machine-retry admission re-generations.
    assert "SelectionDecision.PARK_MACHINE_RETRY}" in selection_source


@pytest.mark.smoke
def test_selection_census_segregates_machine_retries() -> None:
    """A machine retry is censused apart from granted (reviewed) re-drives."""

    fact = {"row_sha256": "deadbeef", "parked_tree_sha": None, "current_tree_sha": "tree-b"}
    plan = build_selection_plan(
        [
            _facts(park_machine_readmission=fact),
            _facts(stable_id="m-still-parked"),
        ],
        mode=SelectionMode.ORDINARY,
    )
    census = plan.census()
    assert census["park_machine_retry"] == 1
    assert census["parked"] == 1
    assert census["granted"] == 0
    assert census["admitted"] == ["m-parked"]


@pytest.mark.smoke
def test_requeue_refuses_provably_unbindable_grants(tmp_path: Path, capsys) -> None:
    """A grant no selection arm can consume refuses at mint, naming the levers."""

    from menagerie.crawler.tools.requeue import main

    records_root = tmp_path / "records"
    operational = records_root / "operational"
    operational.mkdir(parents=True)
    ledger = operational / "requeue-grants.jsonl"
    append_refusal_park_row(
        records_root / "intake-extensions" / "terminal-refusal-parks.jsonl",
        _park_row(tree_sha="tree-a"),
    )
    exit_code = main(
        [
            "m-parked",
            "--grant",
            "1",
            "--stage",
            "runner",
            "--reason",
            "operator hopes this works",
            "--repo-root",
            str(tmp_path),
            "--ledger",
            str(ledger),
        ]
    )
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "provably unbindable" in err
    assert "machine release lane" in err
    assert "void the park row" in err
    assert not ledger.exists() or "grant_id" not in ledger.read_text(encoding="utf-8")
    # Positive control: an unparked member's grant still mints.
    exit_code = main(
        [
            "m-free",
            "--grant",
            "1",
            "--stage",
            "runner",
            "--reason",
            "legitimate re-drive",
            "--repo-root",
            str(tmp_path),
            "--ledger",
            str(ledger),
        ]
    )
    assert exit_code == 0
    minted = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert minted["stable_id"] == "m-free"


@pytest.mark.smoke
def test_machinery_fault_class_is_closed() -> None:
    """Extending the machinery-fault class is a rotation-batch vocabulary change."""

    assert MACHINERY_FAULT_PARK_DISPOSITIONS == frozenset({REFUSAL_PARK_DISPOSITION})


@pytest.mark.heavy
def test_pre_era_park_member_machine_readmits_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the REAL driver: a pre-era machinery park readmits its member.

    This is the production shape of the §6 dossier's park re-drives: the fix
    wave changes the tree, the next ordinary leg readmits each parked member
    for one fresh generation, and a landed revision (or an idempotent re-park
    under the new tree) settles it. The closed half -- a park recorded under
    the RUNNING tree stays absolutely parked through the real driver and its
    close -- is pinned by the five-row campaign tests in
    ``test_work_selection_policy.py`` (their fixture park records the running
    tree since D-48).
    """

    from menagerie.crawler.refusal_parks import refusal_park_path as park_path
    from menagerie.crawler.tests.test_slice_f_driver import FakeAuthor, _driver, _paths
    from menagerie.crawler.tests.test_work_selection_policy import (
        _five_row_snapshot,
        _selection_census_events,
    )

    snapshot, by_name = _five_row_snapshot(tmp_path)
    parked_id = by_name["RowParked"]
    records_root = _paths(tmp_path, snapshot).ledgers.models.parent.parent
    append_refusal_park_row(
        park_path(records_root),
        _park_row(parked_id, tree_sha=None, work_id=f"work-{parked_id}"),
    )
    author = FakeAuthor()
    _driver(tmp_path, snapshot, author=author, scheduler="waves").run()
    census = _selection_census_events(tmp_path, snapshot)[-1]["details"]
    assert parked_id in set(author.calls), "the lane readmitted the parked member"
    assert census["park_machine_retry"] == 1
    assert census["parked"] == 0
