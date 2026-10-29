"""W5 merge gate: the uncatchable-kill durability matrix (T5-4, C-W5-MERGE).

The graceful S01-S13 matrix proves every HANDLED signal boundary; every cell
here is deliberately UNCATCHABLE: a real ``SIGKILL`` delivered inside the
provably check-free atomic award-commit windows, during partition close, and
mid-ledger-append -- plus flock reclaim after holder death and an ENOSPC
short-write at append.  The assert pattern throughout is the serial no-kill
oracle: the identical campaign, run to completion at the same absolute path,
must be byte-reproduced (modulo the one wall-clock-stamped authorization
event) by the killed-then-resumed campaign, with no double award and an exact
artifact<->model bijection.  A fault may leave loud, typed, disclosed residue;
it may never leave silent divergence.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.driver_contracts import DriverLockError
from menagerie.crawler.grant_parks import (
    GRANT_PARK_EVENT_DISPOSITION,
    grant_park_path,
    load_grant_park_rows,
)
from menagerie.crawler.refusal_parks import (
    append_refusal_park_void,
    build_refusal_park_void,
    load_refusal_park_rows,
    refusal_park_path,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.kill_matrix_support import (
    assert_artifact_model_bijection,
    assert_no_double_award,
    assert_projection_equals_oracle,
    capture_durable_state,
    durable_projection,
    files_below,
    install_enospc_short_write_on_ledger,
    install_sigkill_after_park_row_before_stash,
    install_sigkill_before_close_state_write,
    install_sigkill_before_final_artifact_event,
    install_sigkill_on_append_model,
    read_child_outcome,
    recreate_campaign_dir,
    run_forked,
    sigkill_self,
    write_child_outcome,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    ChangedInputAuthor,
    ForwardScript,
    InaccurateChecker,
    RepairingIdentityAuthor,
    ScriptedForward,
    _driver,
    _paths,
    _snapshot,
)
from menagerie.crawler.tests.test_grant_pending_parks import (
    FakeForward,
    _rewrite_intake_with_changed_sources,
)

pytestmark = pytest.mark.heavy


def _run_plain(root: Path, count: int, *, forward_failed_id: Optional[str] = None) -> Any:
    """Run the deterministic fake campaign to completion and return its result."""

    snapshot = _snapshot(root, count=count)
    forward = (
        ScriptedForward(ForwardScript(failed_id=forward_failed_id))
        if forward_failed_id is not None
        else None
    )
    return _driver(root, snapshot, forward=forward).run()


def _plain_oracle(root: Path, count: int, *, fail_first: bool = False) -> dict[str, Any]:
    """Capture the serial no-kill oracle projection at exactly this path."""

    snapshot = _snapshot(root, count=count)
    failed_id = snapshot.items[0].stable_id if fail_first else None
    result = _run_plain(root, count, forward_failed_id=failed_id)
    assert result.status == "complete"
    projection = durable_projection(capture_durable_state(root, _paths(root, snapshot)))
    recreate_campaign_dir(root)
    return projection


def _child_run_plain_with_injection(
    root: Path, count: int, injection: str, fail_first: bool, kill_boundary: Optional[str]
) -> None:
    """Forked child: install one fault and run the campaign; SIGKILL never returns."""

    if injection == "sigkill-on-append-model":
        install_sigkill_on_append_model()
    elif injection == "sigkill-before-final-artifact-event":
        install_sigkill_before_final_artifact_event()
    elif injection == "sigkill-before-close-state-write":
        install_sigkill_before_close_state_write()
    elif injection == "enospc-short-write-models":
        install_enospc_short_write_on_ledger(_paths(root, _snapshot(root, count)).ledgers.models)
    elif injection != "none":
        raise AssertionError(f"unknown injection {injection!r}")

    snapshot = _snapshot(root, count=count)
    failed_id = snapshot.items[0].stable_id if fail_first else None
    forward = ScriptedForward(ForwardScript(failed_id=failed_id)) if failed_id is not None else None

    def boundary(name: str, stable_id: str) -> None:
        del stable_id
        if kill_boundary is not None and name == kill_boundary:
            sigkill_self()

    try:
        result = _driver(root, snapshot, forward=forward, boundary=boundary).run()
    except BaseException as exc:  # a loud, typed refusal is a legitimate outcome
        write_child_outcome(root, {"outcome": "raised", "error": repr(exc)})
        raise SystemExit(3) from exc
    write_child_outcome(root, {"outcome": "completed", "status": result.status})
    raise SystemExit(0)


def _final_award_keys(state: dict[str, Any]) -> set[tuple[str, str]]:
    """Return the (stable_id, work_id) key of every final artifact event."""

    return {
        (str(event["stable_id"]), str(event["work_id"]))
        for event in state["artifact_events"]
        if event.get("event_kind") in ("published", "private-committed")
    }


def _awarded_model_ids(state: dict[str, Any]) -> set[str]:
    """Return the stable ids of model rows carrying committed artifact authority."""

    return {str(model["stable_id"]) for model in state["models"] if model.get("artifact_authority")}


def test_sigkill_between_publication_and_model_append_resumes_exactly_once(
    tmp_path: Path,
) -> None:
    """T5-4 headline: kill inside the check-free window; resume awards exactly once.

    The child dies inside ``source[publication:append]`` -- the final artifact
    event is durable, the models row is not.  One production resume must land
    the missing row from the recorded artifact authority without re-publishing
    and converge to the serial oracle byte-for-byte.
    """

    root = tmp_path / "campaign"
    root.mkdir()
    oracle = _plain_oracle(root, count=2)

    exitcode = run_forked(
        _child_run_plain_with_injection, (root, 2, "sigkill-on-append-model", False, None)
    )
    assert exitcode == -signal.SIGKILL
    assert read_child_outcome(root) is None

    snapshot = _snapshot(root, count=2)
    paths = _paths(root, snapshot)
    before = capture_durable_state(root, paths)
    assert before["models"] == []
    half_awarded = _final_award_keys(before)
    assert len(half_awarded) == 1
    assert_no_double_award(before)

    resumed = _driver(root, snapshot).run()
    assert resumed.status == "complete"
    after = capture_durable_state(root, paths)
    assert_no_double_award(after)
    assert_artifact_model_bijection(root, after)
    assert half_awarded <= _final_award_keys(after)
    assert_projection_equals_oracle(durable_projection(after), oracle)


def test_sigkill_after_model_append_before_post_award_boundary_is_idempotent(
    tmp_path: Path,
) -> None:
    """The other ordering: publication AND models append durable, then SIGKILL.

    Dying at the ``post-award-commit`` boundary hook -- after
    ``reducer.append_model`` returned, before any post-award observation or
    driver-state write -- must leave a resumable campaign whose resume changes
    nothing about the already-committed award.
    """

    root = tmp_path / "campaign"
    root.mkdir()
    oracle = _plain_oracle(root, count=2)

    exitcode = run_forked(
        _child_run_plain_with_injection, (root, 2, "none", False, "post-award-commit")
    )
    assert exitcode == -signal.SIGKILL

    snapshot = _snapshot(root, count=2)
    paths = _paths(root, snapshot)
    before = capture_durable_state(root, paths)
    assert len(before["models"]) == 1
    assert _awarded_model_ids(before) == {str(before["models"][0]["stable_id"])}
    assert_artifact_model_bijection(root, before)

    resumed = _driver(root, snapshot).run()
    assert resumed.status == "complete"
    after = capture_durable_state(root, paths)
    assert before["models"] == after["models"][: len(before["models"])]
    assert_no_double_award(after)
    assert_artifact_model_bijection(root, after)
    assert_projection_equals_oracle(durable_projection(after), oracle)


def test_sigkill_inside_terminal_award_commit_window_resumes_exactly_once(
    tmp_path: Path,
) -> None:
    """Kill the terminal-award section: terminal publication durable, no model row."""

    root = tmp_path / "campaign"
    root.mkdir()
    oracle = _plain_oracle(root, count=1, fail_first=True)

    exitcode = run_forked(
        _child_run_plain_with_injection, (root, 1, "sigkill-on-append-model", True, None)
    )
    assert exitcode == -signal.SIGKILL

    snapshot = _snapshot(root, count=1)
    paths = _paths(root, snapshot)
    before = capture_durable_state(root, paths)
    assert before["models"] == []
    assert len(_final_award_keys(before)) == 1

    failed_id = snapshot.items[0].stable_id
    resumed = _driver(
        root, snapshot, forward=ScriptedForward(ForwardScript(failed_id=failed_id))
    ).run()
    assert resumed.status == "complete"
    after = capture_durable_state(root, paths)
    assert [model["status"]["kind"] for model in after["models"]] == ["failed"]
    assert_no_double_award(after)
    assert_artifact_model_bijection(root, after)
    assert_projection_equals_oracle(durable_projection(after), oracle)


def test_sigkill_between_reconstruction_commit_and_final_event_never_silently_awards(
    tmp_path: Path,
) -> None:
    """Kill mid-publication: a chain without its final event is never authority.

    The reconstruction anchor and its ledger event are durable but the final
    ``published`` event never lands.  Resume must treat the incomplete chain
    as a quarantined non-award -- no model row derives from it -- and drive
    the model to a complete award whose final state matches the serial oracle
    everywhere a fresh award is comparable.
    """

    root = tmp_path / "campaign"
    root.mkdir()
    oracle = _plain_oracle(root, count=1)

    exitcode = run_forked(
        _child_run_plain_with_injection,
        (root, 1, "sigkill-before-final-artifact-event", False, None),
    )
    assert exitcode == -signal.SIGKILL

    snapshot = _snapshot(root, count=1)
    paths = _paths(root, snapshot)
    before = capture_durable_state(root, paths)
    assert before["models"] == []
    assert _final_award_keys(before) == set()
    kinds = [event["event_kind"] for event in before["artifact_events"]]
    assert "reconstruction-committed" in kinds

    resumed = _driver(root, snapshot).run()
    assert resumed.status == "complete"
    after = capture_durable_state(root, paths)
    assert_no_double_award(after)
    assert_artifact_model_bijection(root, after)
    assert len(_final_award_keys(after)) == 1
    projection = durable_projection(after)
    assert projection["models"] == oracle["models"]
    assert projection["repository_files"] == oracle["repository_files"]
    assert projection["public_mirror_files"] == oracle["public_mirror_files"]
    assert projection["driver_state"] == oracle["driver_state"]


def test_sigkill_during_close_before_status_write_reproduces_the_close(
    tmp_path: Path,
) -> None:
    """Kill the close after the partition holds but before the status persists."""

    root = tmp_path / "campaign"
    root.mkdir()
    oracle = _plain_oracle(root, count=2)

    exitcode = run_forked(
        _child_run_plain_with_injection, (root, 2, "sigkill-before-close-state-write", False, None)
    )
    assert exitcode == -signal.SIGKILL

    snapshot = _snapshot(root, count=2)
    paths = _paths(root, snapshot)
    before = capture_durable_state(root, paths)
    assert len(before["models"]) == 2
    assert (before["driver_state"] or {}).get("status") != "complete"
    assert_artifact_model_bijection(root, before)

    resumed = _driver(root, snapshot).run()
    assert resumed.status == "complete"
    after = capture_durable_state(root, paths)
    assert after["driver_state"]["status"] == "complete"
    assert_no_double_award(after)
    assert_projection_equals_oracle(durable_projection(after), oracle)


def _run_park_leg1(root: Path) -> Any:
    """Drive every model to a review-held terminal (leg 1 of the park scenario)."""

    snapshot = _snapshot(root, count=2)
    return _driver(
        root,
        snapshot,
        author=RepairingIdentityAuthor(),
        checker=InaccurateChecker(),
    ).run()


def _park_leg2_driver(root: Path, changed: Any) -> Any:
    """Build the leg-2 driver as the operator's NAMED re-drive of the stale heads.

    COMPOSE (AW x WS-1, clause 4 / V3): changed intake bytes stale every
    settled leg-1 head, and an ordinary leg never re-drives a settled head,
    stale or not -- it refuses to start with a nonzero stale census.  The
    re-drive of the staled heads is therefore the operator's explicit
    ``--redrive-targets`` decision, exactly as pinned by
    ``test_grant_pending_parks.test_three_legs_park_then_grant_then_consume``.
    A live park still excludes its member in EVERY mode, so the named list may
    name parked heads without re-driving them.
    """

    driver = _driver(root, changed, author=ChangedInputAuthor(), forward=FakeForward())
    driver.config = replace(
        driver.config,
        redrive_targets=tuple(sorted(item.stable_id for item in changed.items)),
        # D-54: a named re-drive is an operator act with a blast radius; its
        # rationale is required and recorded on the durable leg config event.
        redrive_reason="kill-matrix leg 2: re-drive the heads staled by changed intake bytes",
    )
    return driver


def _child_run_park_leg2_with_kill(root: Path) -> None:
    """Forked child: leg 2 dies between the durable park row and its stash/event."""

    install_sigkill_after_park_row_before_stash()
    changed = _rewrite_intake_with_changed_sources(root, 2)
    _park_leg2_driver(root, changed).run()
    raise SystemExit(0)


def test_sigkill_between_park_append_and_health_event_recovers_parks_and_status(
    tmp_path: Path,
) -> None:
    """T5-4 close arm: a kill mid-park-mint leaves parks and status recoverable.

    The grant-pending park row is durable but its in-run stash and operational
    health event are not.  Resume must re-claim the durable row (no duplicate
    rows, no unaccounted model at partition time) and close with the exact
    review-holds status the serial oracle produces -- never a silent
    ``complete``.  The pre-crash row's parked health event died with the
    child and is never re-emitted.  COMPOSE (AW x WS-1, clause 5): the park
    treadmill is gone -- a live park excludes its member at SELECTION, before
    any author-lane spend, so the reuse is disclosed through the resume leg's
    ``work-selection-census`` (the parked count) and the close's review-holds
    block, never through a re-bought ``machine-deferral-row-reused`` claim
    (that disclosure remains valve-row-only).  The park stays visible where
    operators look: the census, the driver state, and the durable park ledger
    itself.
    """

    root = tmp_path / "campaign"
    root.mkdir()

    first = _run_park_leg1(root)
    assert first.status == "terminal-partition-complete"
    changed = _rewrite_intake_with_changed_sources(root, 2)
    second = _park_leg2_driver(root, changed).run()
    assert second.status == "terminal-partition-complete-with-review-holds"
    paths = _paths(root, changed)
    records_root = paths.ledgers.models.parent.parent
    oracle = durable_projection(capture_durable_state(root, paths))
    oracle_parks = load_grant_park_rows([grant_park_path(records_root)])
    assert len(oracle_parks) == 2
    recreate_campaign_dir(root)

    assert _run_park_leg1(root).status == "terminal-partition-complete"
    exitcode = run_forked(_child_run_park_leg2_with_kill, (root,))
    assert exitcode == -signal.SIGKILL

    before_parks = load_grant_park_rows([grant_park_path(records_root)])
    assert len(before_parks) == 1
    operational_path = canonical_operational_ledger_path(paths.ledgers.models)
    before_events = scan_jsonl(operational_path) if operational_path.exists() else []
    parked_before = [
        event
        for event in before_events
        if event.get("details", {}).get("disposition") == GRANT_PARK_EVENT_DISPOSITION
    ]
    assert parked_before == []

    # The resume names the same re-drive: one head is already parked (excluded
    # in every mode) and the other is settled-but-stale, which an ordinary
    # resume would refuse to start over rather than silently re-drive.
    resumed = _park_leg2_driver(root, changed).run()
    assert resumed.status == "terminal-partition-complete-with-review-holds"
    after_parks = load_grant_park_rows([grant_park_path(records_root)])
    assert len(after_parks) == 2
    assert {row["stable_id"] for row in after_parks} == {row["stable_id"] for row in oracle_parks}
    parked_after = [
        event
        for event in scan_jsonl(operational_path)
        if event.get("details", {}).get("disposition") == GRANT_PARK_EVENT_DISPOSITION
    ]
    minted_by_resume = {str(row["stable_id"]) for row in after_parks} - {
        str(row["stable_id"]) for row in before_parks
    }
    assert {event["details"]["stable_id"] for event in parked_after} == minted_by_resume
    # The durable pre-crash row was re-claimed at selection, not re-bought:
    # the resume leg's census discloses exactly one parked member, and the
    # close's review-holds block names BOTH parks (the reused row and the
    # resume-minted one) -- the loud place operators look.
    census_after = [
        event
        for event in scan_jsonl(operational_path)
        if event.get("event_kind") == "work-selection-census"
    ]
    assert census_after[-1]["details"]["parked"] == 1
    state = capture_durable_state(root, paths)["driver_state"]
    assert state is not None
    assert {entry["stable_id"] for entry in state["review_holds"]["pending"]} == {
        str(row["stable_id"]) for row in after_parks
    }
    oracle_state = deepcopy(oracle["driver_state"])
    resumed_state = deepcopy(state)
    oracle_pending = {
        str(entry["stable_id"]): entry for entry in oracle_state["review_holds"]["pending"]
    }
    resumed_pending = {
        str(entry["stable_id"]): entry for entry in resumed_state["review_holds"]["pending"]
    }
    for stable_id in sorted(resumed_pending):
        for entry in (oracle_pending[stable_id], resumed_pending[stable_id]):
            entry.pop("row_sha256")
    assert resumed_state == oracle_state
    oracle_park_by_id = {str(row["stable_id"]): deepcopy(row) for row in oracle_parks}
    resumed_park_by_id = {str(row["stable_id"]): deepcopy(row) for row in after_parks}
    for stable_id in sorted(resumed_park_by_id):
        if stable_id in minted_by_resume:
            oracle_used = oracle_park_by_id[stable_id]["pending"]["pending_model"]["budget"][
                "author_sessions_used"
            ]
            resumed_used = resumed_park_by_id[stable_id]["pending"]["pending_model"]["budget"][
                "author_sessions_used"
            ]
            # BL-11 makes the killed speculative buy durable money truth. The
            # resumed row therefore discloses exactly one additional session;
            # it is not a second award or a hidden state divergence.
            assert resumed_used == oracle_used + 1
        for row in (oracle_park_by_id[stable_id], resumed_park_by_id[stable_id]):
            row.pop("row_sha256")
            row["pending"].pop("pending_model_sha256")
            row["pending"]["pending_model"]["budget"].pop("author_sessions_used")
    assert resumed_park_by_id == oracle_park_by_id
    assert_projection_equals_oracle(
        durable_projection(capture_durable_state(root, paths)),
        oracle,
        exclude=("operational", "driver_state"),
    )


_TORN_PREFIX = b'{"schema_version": "menagerie.crawler.'


def _tear_tail(path: Path) -> None:
    """Append a partial, newline-free record prefix -- the crash torn tail."""

    with path.open("ab") as handle:
        handle.write(_TORN_PREFIX)


@pytest.mark.parametrize("ledger", ["models", "gates"])
def test_driver_resumes_from_torn_core_ledger_tail(tmp_path: Path, ledger: str) -> None:
    """A torn models/gates tail is evidenced, truncated, and changes nothing."""

    root = tmp_path / "campaign"
    root.mkdir()
    snapshot = _snapshot(root, count=2)
    assert _driver(root, snapshot).run().status == "complete"
    paths = _paths(root, snapshot)
    oracle = durable_projection(capture_durable_state(root, paths))

    target = getattr(paths.ledgers, ledger)
    _tear_tail(target)
    assert not target.read_bytes().endswith(b"\n")

    resumed = _driver(root, snapshot).run()
    assert resumed.status == "complete"
    evidence = target.parent / f"{target.name}.recovery.jsonl"
    assert evidence.exists()
    rows = scan_jsonl(evidence, validate=False)
    assert rows[-1]["byte_count"] == len(_TORN_PREFIX)
    assert_projection_equals_oracle(durable_projection(capture_durable_state(root, paths)), oracle)


def test_driver_resumes_from_torn_operational_ledger_tail(tmp_path: Path) -> None:
    """A torn operational tail is evidenced and the resume leg reproduces exactly.

    Oracle and subject both run three legs (terminals, parks, idempotent
    resume); the subject's operational ledger is torn between legs two and
    three, so the driver's startup recovery must evidence and truncate the
    partial line and the third leg must then be indistinguishable from the
    serial one.
    """

    def _three_legs(root: Path, tear_before_leg3: bool) -> dict[str, Any]:
        from menagerie.crawler.tests.campaign_binding_support import (
            adopt_successor_snapshot,
            bind_snapshot_at_records_root,
        )

        prior = _snapshot(root, count=2)
        # The registered changed-input rung below only resolves at close
        # inside a partitioned campaign repository (`introduced_member_ids`
        # verifies the campaign root and the durable lineage snapshots), so
        # the rig binds leg 1's snapshot exactly like the CLI fixtures do.
        bind_snapshot_at_records_root(root / "records", prior)
        assert _run_park_leg1(root).status == "terminal-partition-complete"
        changed = _rewrite_intake_with_changed_sources(root, 2)
        adopt_successor_snapshot(
            root / "records", changed, label="durability-matrix changed-input rung"
        )
        second = _park_leg2_driver(root, changed).run()
        assert second.status == "terminal-partition-complete-with-review-holds"
        paths = _paths(root, changed)
        operational_path = canonical_operational_ledger_path(paths.ledgers.models)
        assert scan_jsonl(operational_path)
        if tear_before_leg3:
            _tear_tail(operational_path)
            assert not operational_path.read_bytes().endswith(b"\n")
        third = _driver(root, changed, author=ChangedInputAuthor(), forward=FakeForward()).run()
        assert third.status == "terminal-partition-complete-with-review-holds"
        return durable_projection(capture_durable_state(root, paths))

    root = tmp_path / "campaign"
    root.mkdir()
    oracle = _three_legs(root, tear_before_leg3=False)
    recreate_campaign_dir(root)

    resumed = _three_legs(root, tear_before_leg3=True)
    paths = _paths(root, _snapshot(root, count=2))
    operational_path = canonical_operational_ledger_path(paths.ledgers.models)
    evidence = operational_path.parent / f"{operational_path.name}.recovery.jsonl"
    assert evidence.exists()
    assert scan_jsonl(evidence, validate=False)[-1]["byte_count"] == len(_TORN_PREFIX)
    assert_projection_equals_oracle(resumed, oracle)


def test_torn_last_model_award_line_is_rederived_from_artifact_authority(
    tmp_path: Path,
) -> None:
    """Tearing the last models row mid-line loses only bytes, never the award.

    Recovery truncates the half-written award row; the artifact ledger still
    holds the committed final event, so the resume must re-derive the missing
    model row through rehydration -- exactly once -- and land back on the
    serial oracle.
    """

    root = tmp_path / "campaign"
    root.mkdir()
    snapshot = _snapshot(root, count=2)
    assert _driver(root, snapshot).run().status == "complete"
    paths = _paths(root, snapshot)
    oracle = durable_projection(capture_durable_state(root, paths))

    raw = paths.ledgers.models.read_bytes()
    body = raw[:-1]  # drop the final newline so the last row becomes the tail
    torn_at = len(body) - (len(body) - body.rfind(b"\n") - 1) // 2
    paths.ledgers.models.write_bytes(body[:torn_at])
    assert not paths.ledgers.models.read_bytes().endswith(b"\n")

    resumed = _driver(root, snapshot).run()
    assert resumed.status == "complete"
    evidence = paths.ledgers.models.parent / f"{paths.ledgers.models.name}.recovery.jsonl"
    assert evidence.exists()
    after = capture_durable_state(root, paths)
    assert_no_double_award(after)
    assert_artifact_model_bijection(root, after)
    assert_projection_equals_oracle(durable_projection(after), oracle)


_HOLDER_SCRIPT = """
import fcntl, sys, time
handle = open(sys.argv[1], "a+b")
fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep(120)
"""


def _spawn_flock_holder(lock_path: Path) -> subprocess.Popen[str]:
    """Hold an exclusive flock on ``lock_path`` from a disposable subprocess."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, str(lock_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "locked"
    return holder


def test_models_ledger_lock_refuses_while_held_and_reclaims_after_holder_sigkill(
    tmp_path: Path,
) -> None:
    """Ledger single-writer flock: typed refusal while held, clean reclaim after death."""

    root = tmp_path / "campaign"
    root.mkdir()
    snapshot = _snapshot(root, count=1)
    paths = _paths(root, snapshot)
    sidecar = paths.ledgers.models.parent / f"{paths.ledgers.models.name}.lock"
    holder = _spawn_flock_holder(sidecar)
    try:
        with pytest.raises(DriverLockError, match="ledger already has a writer"):
            _driver(root, snapshot).run()
        holder.send_signal(signal.SIGKILL)
        assert holder.wait(timeout=10) == -signal.SIGKILL
        resumed = _driver(root, snapshot).run()
        assert resumed.status == "complete"
        after = capture_durable_state(root, paths)
        assert len(after["models"]) == 1
        assert_artifact_model_bijection(root, after)
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=10)


def test_driver_lock_refuses_while_held_and_reclaims_after_holder_sigkill(
    tmp_path: Path,
) -> None:
    """Driver flock: typed refusal while held, clean reclaim after holder SIGKILL."""

    root = tmp_path / "campaign"
    root.mkdir()
    snapshot = _snapshot(root, count=1)
    paths = _paths(root, snapshot)
    holder = _spawn_flock_holder(paths.lock_path)
    try:
        with pytest.raises(DriverLockError):
            _driver(root, snapshot).run()
        holder.send_signal(signal.SIGKILL)
        assert holder.wait(timeout=10) == -signal.SIGKILL
        resumed = _driver(root, snapshot).run()
        assert resumed.status == "complete"
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=10)


def test_enospc_short_write_at_models_append_recovers_without_double_award(
    tmp_path: Path,
) -> None:
    """ENOSPC mid-append: contained torn-write, honest degraded close, exact recovery.

    The child's models-ledger append persists half a record line and raises
    ``OSError(ENOSPC)`` after the publication durably completed. Under the
    composed substrate the failed append is truncated back to the exact
    validated extent at write time (never a torn tail left on disk), the
    record-path death is contained as a durable ``terminal-unrecordable``
    operational event (streak-limited, so a systemically full disk still
    aborts loudly), and the leg closes with refusal parks instead of a clean
    ``complete`` -- it must never claim the award it could not record.

    COMPOSE (AW x WS-1 clause 5 + WS-2): the containment's terminal-refusal
    park is operator property -- a live park excludes its member in EVERY
    mode and nothing releases it automatically, so a blind resume must stay
    loudly parked (no model row, no silent ``complete``, the committed award
    untouched) instead of silently re-driving.  Only after the operator
    drains the park through the sanctioned void lane does recovery re-derive
    the model row from the already-durable artifact authority -- exactly
    once.  The containment attempt is audit history, not authority for
    choosing a different award path.
    """

    root = tmp_path / "campaign"
    root.mkdir()
    oracle = _plain_oracle(root, count=1)
    snapshot = _snapshot(root, count=1)
    paths = _paths(root, snapshot)

    exitcode = run_forked(
        _child_run_plain_with_injection, (root, 1, "enospc-short-write-models", False, None)
    )
    assert exitcode == 0
    outcome = read_child_outcome(root)
    assert outcome is not None
    assert outcome["outcome"] == "completed"
    assert outcome["status"] != "complete", "an unrecordable award must never read as a clean close"
    assert "refusal-parks" in outcome["status"]
    unrecordable = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("event_kind") == "terminal-unrecordable"
    ]
    assert unrecordable, "the containment must be durable operational evidence"

    raw = paths.ledgers.models.read_bytes()
    assert raw == b"" or raw.endswith(b"\n"), (
        "the failed append must restore the exact validated extent, never a torn tail"
    )
    before = {
        "artifact_events": scan_jsonl(paths.ledgers.artifacts),
        "repository_files": files_below(root / "menagerie"),
        "public_mirror_files": files_below(paths.runtime_root / "mirrors" / "public"),
    }
    assert len(_final_award_keys(before)) == 1
    assert_no_double_award(before)

    # A blind resume finds the live park and must stay loudly parked: the
    # member is excluded from selection, no model row is minted, and the
    # durable award is neither re-published nor silently claimed.
    parked_resume = _driver(root, snapshot).run()
    assert parked_resume.status == outcome["status"]
    parked_state = capture_durable_state(root, paths)
    assert parked_state["models"] == []
    assert _final_award_keys(parked_state) == _final_award_keys(before)
    assert parked_state["repository_files"] == before["repository_files"]
    assert parked_state["public_mirror_files"] == before["public_mirror_files"]

    # The operator drains the park through the sanctioned void lane -- voids
    # are decisions, never mechanics -- and only then may recovery land the
    # unrecorded award.
    records_root = paths.ledgers.models.parent.parent
    park_ledger = refusal_park_path(records_root)
    park_rows = load_refusal_park_rows([park_ledger])
    assert len(park_rows) == 1
    append_refusal_park_void(
        park_ledger,
        build_refusal_park_void(
            void_of=str(park_rows[0]["row_sha256"]),
            reason="disk restored; the unrecorded award may re-land",
            voided_by="operator",
            created_at="2026-08-16T00:00:00Z",
        ),
    )
    assert load_refusal_park_rows([park_ledger]) == ()

    resumed = _driver(root, snapshot).run()
    assert resumed.status == "complete"

    after = capture_durable_state(root, paths)
    assert len(after["models"]) == 1
    assert _final_award_keys(after) == _final_award_keys(before)
    assert_no_double_award(after)
    assert_artifact_model_bijection(root, after)
    assert after["repository_files"] == before["repository_files"]
    assert after["public_mirror_files"] == before["public_mirror_files"]
    assert_projection_equals_oracle(
        durable_projection(after),
        oracle,
        exclude=("attempts", "operational", "driver_state"),
    )
