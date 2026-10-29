"""X1 rehearsals: the pipelined scheduler under fire (widths, kills, migration).

The M3 merge-gate suite proves one clean width-7 leg. This module rehearses the
space between "tests pass" and "20k models flow through it for weeks":

- width sweep (1/4/8) with mixed dead/alive intakes, closed against the wave
  reference with the strongest applicable oracle at each width;
- a kill at EVERY dispatch boundary (author, triage, gate, award, close) with a
  pipelined resume that must reconcile without phantom strands or double spend;
- mid-campaign scheduler rollback (pipelined -> waves) and cutover
  (waves -> pipelined) from a killed leg, M4's migration story;
- graceful SIGTERM drain (shutdown event) inside a pipelined round.

Oracle notes. The digest-aliased canonical projection (append-order strict) is
only valid across schedulers for intakes whose record CONTENT is scheduler
independent -- all-alive intakes, where both modes buy the same authors, gates,
and awards in the same per-model shapes. Mixed dead/alive intakes differ BY
DESIGN (execute-first terminalizes machine deaths gate-free: `pending`
authored-metadata state, zero gate rows, different batch composition for the
survivors), so those legs close against semantic parity: same close status,
same per-model terminal heads, same awarded set, and a canonical replay of the
pipelined records through the reducer's own tripwires.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import pytest

from menagerie.crawler.checker_conflicts import (
    checker_verdict_conflict_path,
    load_checker_verdict_conflict_rows,
)
from menagerie.crawler.constants import METADATA_BATCH_MAX
from menagerie.crawler.driver_contracts import AuthorArtifact, EnvironmentBinding
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.tests.test_author_wave_concurrency import _canonical_projection
from menagerie.crawler.tests.test_scheduler_pipelined_leg import ImportDeadForward
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    FakeForward,
    InjectedKill,
    ScriptedForward,
    _driver,
    _paths,
    _snapshot,
)


class SelectivelyDeadForward(ImportDeadForward):
    """Import-kill an exact stable-id subset; run everything else clean."""

    def __init__(self, dead_ids: Sequence[str]) -> None:
        """Record the import-dead subset."""

        super().__init__()
        self._dead_ids = frozenset(dead_ids)

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Return the import-death shape for dead models, clean attempts else."""

        if str(artifact.proposal["stable_id"]) in self._dead_ids:
            return super().forward(artifact, environment, cold_runs, work_root)
        return ScriptedForward.forward(self, artifact, environment, cold_runs, work_root)


def _model_heads(paths: Any) -> dict[str, Mapping[str, Any]]:
    """Return the last (current) model record per stable_id, in-scope only."""

    heads: dict[str, Mapping[str, Any]] = {}
    for record in scan_jsonl(paths.ledgers.models):
        heads[str(record["stable_id"])] = record
    return heads


def _assert_per_model_lineage(paths: Any) -> None:
    """Assert parent-chained per-model revisions and strict global sequences."""

    models = scan_jsonl(paths.ledgers.models)
    assert [record["record_seq"] for record in models] == list(range(1, len(models) + 1))
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for record in models:
        by_model.setdefault(str(record["stable_id"]), []).append(record)
    for revisions in by_model.values():
        assert revisions[0]["parent_revision"] is None
        for earlier, later in zip(revisions, revisions[1:]):
            assert later["parent_revision"] == earlier["record_revision"]
    attempts = scan_jsonl(paths.ledgers.attempts)
    assert [record["ledger_seq"] for record in attempts] == list(range(1, len(attempts) + 1))


def _assert_canonical_replay(root: Path, snapshot: Any, config_driver: Any) -> None:
    """Re-reduce the campaign's ledgers through every reducer replay tripwire.

    Partition validity was already adjudicated by the driver's own close (the
    asserted ``result.status``); this replays the raw ledgers through
    ``CanonicalReducer``'s append-time invariants and checks every in-scope
    model still projects a current record.
    """

    paths = _paths(root, snapshot)
    # The driver's post-run context is the replay authority: it carries the
    # live environment generations (update_context census) without which every
    # record projects environment-STALE and the replay is vacuous. Any offline
    # audit must reconstruct those generations from ledger evidence first.
    context = config_driver._authority_context  # noqa: SLF001
    with CanonicalReducer(paths.ledgers, context) as reducer:
        current = dict(reducer.current_records)
    for item in snapshot.items:
        assert item.stable_id in current


def _run_leg(
    root: Path,
    count: int,
    dead_indices: Sequence[int],
    scheduler: str,
) -> tuple[Any, Any, Any, FakeChecker, ScriptedForward]:
    """Run one whole synthetic campaign leg and return its artifacts."""

    root.mkdir(exist_ok=True)
    snapshot = _snapshot(root, count=count)
    dead_ids = tuple(snapshot.items[index].stable_id for index in dead_indices)
    forward: ScriptedForward = SelectivelyDeadForward(dead_ids) if dead_ids else FakeForward()
    checker = FakeChecker()
    driver = _driver(root, snapshot, checker=checker, forward=forward, scheduler=scheduler)
    result = driver.run()
    return snapshot, driver, result, checker, forward


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("count", "dead_indices"),
    [
        pytest.param(1, (), id="width-1-alive"),
        pytest.param(1, (0,), id="width-1-dead"),
        pytest.param(4, (1, 3), id="width-4-mixed"),
        pytest.param(8, (0, 3, 4, 6), id="width-8-mixed"),
        pytest.param(8, tuple(range(8)), id="width-8-all-dead"),
    ],
)
def test_pipelined_width_sweep_closes_semantically_equal_to_waves(
    tmp_path: Path, count: int, dead_indices: Sequence[int]
) -> None:
    """Pipelined legs at width 1/4/8 with mixed dead/alive intakes match waves.

    Semantic-parity oracle (see module docstring): close status, per-model
    terminal heads, awarded set, per-model lineage, and a full canonical
    replay of the pipelined records. All-alive widths additionally get the
    strict append-order projection oracle via
    ``test_pipelined_all_alive_widths_project_identically``.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    count, dead_indices:
        Intake width and which member indices die at import.
    """

    waves_root = tmp_path / "waves"
    pipelined_root = tmp_path / "pipelined"
    _, waves_driver, waves_result, waves_checker, _ = _run_leg(
        waves_root, count, dead_indices, "waves"
    )
    snapshot, pipelined_driver, pipelined_result, pipelined_checker, _ = _run_leg(
        pipelined_root, count, dead_indices, "pipelined"
    )

    assert pipelined_result.status == waves_result.status
    waves_heads = _model_heads(_paths(waves_root, snapshot))
    pipelined_paths = _paths(pipelined_root, snapshot)
    pipelined_heads = _model_heads(pipelined_paths)
    assert set(pipelined_heads) == set(waves_heads)
    for stable_id, head in pipelined_heads.items():
        assert head["status"]["code"] == waves_heads[stable_id]["status"]["code"]
    awarded = {
        stable_id for stable_id, head in pipelined_heads.items() if head["status"]["kind"] == "runs"
    }
    assert awarded == {
        stable_id for stable_id, head in waves_heads.items() if head["status"]["kind"] == "runs"
    }
    _assert_per_model_lineage(pipelined_paths)
    _assert_canonical_replay(pipelined_root, snapshot, pipelined_driver)
    # Execute-first economics must hold at every width: the pipelined leg
    # never buys MORE checker sessions than waves, and buys strictly fewer
    # whenever the intake carries machine deaths.
    assert pipelined_checker.metadata_calls <= waves_checker.metadata_calls
    if dead_indices and len(dead_indices) < count:
        assert pipelined_checker.metadata_calls <= waves_checker.metadata_calls


#: Every boundary hook the pipelined leg's seams fire, in pipeline order.
#: "after-notification-identity" is deliberately absent: a plain completion
#: records no notification identity, so the hook never fires on this path
#: (probed 2026-08-12); the mid-close kill is covered by
#: ``test_pipelined_kill_mid_close_resumes`` instead.
_KILL_BOUNDARIES = (
    "after-author",
    "pre-forward",
    "after-attempt",
    "after-forward",
    "post-attempt-pre-award",
    "after-gate",
    "pre-publication",
    "pre-award-commit",
    "award-commit-entered",
    "post-award-commit",
    "after-reduce",
)


def _kill_once(target: str) -> Any:
    """Return a boundary hook raising ``InjectedKill`` at the first ``target``."""

    fired = {"n": 0}

    def hook(boundary: str, _stable_id: str) -> None:
        if boundary == target and fired["n"] == 0:
            fired["n"] = 1
            raise InjectedKill(target)

    return hook


def _spend(author: FakeAuthor, checker: FakeChecker, forward: ScriptedForward) -> tuple:
    """Return (author sessions, metadata sessions, forward executions)."""

    return (
        sum(author.calls.values()),
        checker.metadata_calls,
        sum(forward.calls.values()),
    )


class _ClassifiedChecker(FakeChecker):
    """Record every metadata session's members as (stable_id, class) pairs.

    The H-5 duplicate-accept door (rule-2, 2026-08-29) buys one extra blind
    width-1 confirmation session per ACCEPTED gate item, synchronously inside
    ``_require_confirmed_accuracy_acceptance`` (never a pool capsule), so a
    thread-local flag armed around the door via ``mark_confirmations``
    separates the two session classes exactly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[tuple[str, str]] = []
        self._door_thread_state = threading.local()

    def mark_confirmations(self, driver: Any) -> None:
        """Flag the H-5 door's synchronous confirmation buys on this driver."""

        inner = driver._require_confirmed_accuracy_acceptance  # noqa: SLF001

        def marked(*args: Any, **kwargs: Any) -> Any:
            self._door_thread_state.active = True
            try:
                return inner(*args, **kwargs)
            finally:
                self._door_thread_state.active = False

        driver._require_confirmed_accuracy_acceptance = marked  # noqa: SLF001

    def check_metadata(self, artifacts, work_root, config):  # noqa: ANN001, ANN201
        """Record each bought session member with its session class."""

        session_class = (
            "confirm" if getattr(self._door_thread_state, "active", False) else "first"
        )
        for artifact in tuple(artifacts):
            self.sessions.append((str(artifact.proposal["stable_id"]), session_class))
        return super().check_metadata(artifacts, work_root, config)

    def bought(self, session_class: str) -> set[str]:
        """Return the stable ids bought in one session class."""

        return {sid for sid, cls in self.sessions if cls == session_class}


def _durable_metadata_state(paths: Any) -> tuple[set[str], set[str]]:
    """Return the (first-round-durable, confirmation-durable) stable-id sets.

    A ``metadata_batch`` row on the gates ledger is the durable first-round
    result; a settled ``confirmed`` duplicate-accept row on the conflict
    ledger is the durable H-5 confirmation (confirmation sessions publish no
    gate row of their own). Anything a killed leg bought WITHOUT reaching
    these ledgers died with the process and must be re-bought exactly once.
    """

    durable_first = {
        str(item["stable_id"])
        for gate in scan_jsonl(paths.ledgers.gates)
        if gate.get("gate_kind") == "metadata_batch"
        for item in gate["items"]
    }
    rows = load_checker_verdict_conflict_rows(
        (checker_verdict_conflict_path(paths.ledgers.models.parent.parent),)
    )
    durable_confirm = {
        str(row["stable_id"])
        for row in rows
        if row.get("row_kind") == "duplicate-accept-confirmation"
        and row.get("status") == "confirmed"
    }
    return durable_first, durable_confirm


@pytest.mark.heavy
@pytest.mark.parametrize("boundary", _KILL_BOUNDARIES)
def test_pipelined_kill_at_every_boundary_resumes_without_double_spend(
    tmp_path: Path, boundary: str
) -> None:
    """A death at any dispatch boundary resumes to the wave-equal close.

    The reconciliation bar, per kill point: the resumed leg completes, per-model
    heads equal the wave reference, exactly one ``runs`` revision per model (no
    phantom strands, no double award), and cross-leg spend never re-buys
    DURABLE work -- author sessions rehydrate (W5), adjudicated gates reuse
    (W4), and persisted triage matrices award without re-execution.

    Metadata spend under the 2026-08-29 width-1 rotation + the H-5 K=2 door:
    the clean one-leg budget is ``ceil(count / cap)`` first-round sessions
    plus one blind confirmation per accepted model. A kill can catch several
    pooled width-1 sessions IN FLIGHT (bought, result never persisted) --
    those are lawful crash-window losses, re-bought EXACTLY once -- so the
    pin is the durable diff, not a constant: the resumed leg buys precisely
    the sessions the ledgers do NOT already witness, and never double-buys
    within itself.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    boundary:
        Boundary hook name to kill at (first occurrence).
    """

    count = 4
    waves_root = tmp_path / "waves"
    waves_snapshot, _, waves_result, _, _ = _run_leg(waves_root, count, (), "waves")
    assert waves_result.status == "complete"
    waves_heads = {
        stable_id: head["status"]["code"]
        for stable_id, head in _model_heads(_paths(waves_root, waves_snapshot)).items()
    }

    root = tmp_path / "pipelined"
    root.mkdir()
    snapshot = _snapshot(root, count=count)
    author1, checker1, forward1 = FakeAuthor(), _ClassifiedChecker(), FakeForward()
    driver1 = _driver(
        root,
        snapshot,
        author=author1,
        checker=checker1,
        forward=forward1,
        boundary=_kill_once(boundary),
        scheduler="pipelined",
    )
    checker1.mark_confirmations(driver1)
    with pytest.raises(InjectedKill):
        driver1.run()

    # The durable ledger state between the legs is the re-buy authority.
    paths = _paths(root, snapshot)
    durable_first, durable_confirm = _durable_metadata_state(paths)

    author2, checker2, forward2 = FakeAuthor(), _ClassifiedChecker(), FakeForward()
    driver2 = _driver(
        root,
        snapshot,
        author=author2,
        checker=checker2,
        forward=forward2,
        scheduler="pipelined",
    )
    checker2.mark_confirmations(driver2)
    resumed = driver2.run()
    assert resumed.status == "complete"

    heads = _model_heads(paths)
    assert {k: v["status"]["code"] for k, v in heads.items()} == waves_heads
    assert (
        sum(1 for record in scan_jsonl(paths.ledgers.models) if record["status"]["kind"] == "runs")
        == count
    )
    _assert_per_model_lineage(paths)

    spend1 = _spend(author1, checker1, forward1)
    spend2 = _spend(author2, checker2, forward2)
    total = tuple(a + b for a, b in zip(spend1, spend2))
    # Exactly the one-leg author/forward budget: 4 author sessions, 4 forward
    # matrices -- measured on the probe matrix 2026-08-12; a regression in W5
    # rehydration or recorded-attempt award shows up here as a cross-leg
    # re-buy.
    assert (total[0], total[2]) == (count, count), total
    # Metadata: neither leg double-buys a (model, class) session within
    # itself, and the resumed leg buys exactly the durable-state complement --
    # every persisted first-round gate and settled confirmation reuses (W4 /
    # the door's ledger re-derivation), every in-flight loss re-buys once.
    for witness in (checker1, checker2):
        assert len(set(witness.sessions)) == len(witness.sessions), witness.sessions
    all_ids = {item.stable_id for item in snapshot.items}
    assert checker2.bought("first") == all_ids - durable_first, checker2.sessions
    assert checker2.bought("confirm") == all_ids - durable_confirm, checker2.sessions


@pytest.mark.heavy
def test_pipelined_kill_mid_close_resumes(tmp_path: Path, monkeypatch: Any) -> None:
    """A death INSIDE partition close (post-leg, mid-rebuild) resumes cleanly.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    monkeypatch:
        Pytest monkeypatch fixture.
    """

    import menagerie.crawler.driver as driver_module

    snapshot = _snapshot(tmp_path, count=4)
    real_rebuild = driver_module.rebuild_state
    calls = {"n": 0}

    def rebuild_then_die(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        result = real_rebuild(*args, **kwargs)
        # Call 1 is the leg-start rebuild; call 2 is the close-path rebuild.
        if calls["n"] == 2:
            raise InjectedKill("mid-close")
        return result

    monkeypatch.setattr(driver_module, "rebuild_state", rebuild_then_die)
    with pytest.raises(InjectedKill):
        _driver(tmp_path, snapshot, scheduler="pipelined").run()
    monkeypatch.setattr(driver_module, "rebuild_state", real_rebuild)

    author2, checker2, forward2 = FakeAuthor(), FakeChecker(), FakeForward()
    resumed = _driver(
        tmp_path,
        snapshot,
        author=author2,
        checker=checker2,
        forward=forward2,
        scheduler="pipelined",
    ).run()
    assert resumed.status == "complete"
    # The records were already complete when the close died: the resumed leg
    # is a pure re-close and buys NOTHING.
    assert _spend(author2, checker2, forward2) == (0, 0, 0)
    _assert_per_model_lineage(_paths(tmp_path, snapshot))


@pytest.mark.heavy
def test_pipelined_sigterm_drains_and_resumes(tmp_path: Path) -> None:
    """A SIGTERM mid-leg returns interrupted:shutdown and resumes to complete.

    Faithful emulation: the production handler
    (``worker_supervisor.shutdown_signal_handlers``) only sets the driver's
    shutdown event; this test sets the same event from a mid-triage boundary
    hook, so the next ``_check_shutdown`` seam raises ``DriverShutdown`` and
    the driver drains -- never cancels -- exactly as a real signal would.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=4)
    cell: dict[str, Any] = {}
    fired = {"n": 0}

    def request_shutdown(boundary: str, _stable_id: str) -> None:
        if boundary == "pre-forward" and fired["n"] == 0:
            fired["n"] = 1
            cell["driver"]._shutdown_event.set()  # noqa: SLF001

    author1, checker1, forward1 = FakeAuthor(), FakeChecker(), FakeForward()
    driver = _driver(
        tmp_path,
        snapshot,
        author=author1,
        checker=checker1,
        forward=forward1,
        boundary=request_shutdown,
        scheduler="pipelined",
    )
    cell["driver"] = driver
    result = driver.run()
    assert result.status == "interrupted:shutdown"

    author2, checker2, forward2 = FakeAuthor(), FakeChecker(), FakeForward()
    resumed = _driver(
        tmp_path,
        snapshot,
        author=author2,
        checker=checker2,
        forward=forward2,
        scheduler="pipelined",
    ).run()
    assert resumed.status == "complete"
    paths = _paths(tmp_path, snapshot)
    assert (
        sum(1 for record in scan_jsonl(paths.ledgers.models) if record["status"]["kind"] == "runs")
        == 4
    )
    _assert_per_model_lineage(paths)
    total = tuple(
        a + b
        for a, b in zip(_spend(author1, checker1, forward1), _spend(author2, checker2, forward2))
    )
    # A drain settles everything it bought (never cancels mid-session), so the
    # cross-leg metadata spend is EXACTLY the clean one-leg budget under the
    # width-1 rotation + H-5 door: ceil(4/cap) first-round sessions plus one
    # confirmation per accepted model -- no crash-window loss to tolerate.
    assert total == (4, -(-4 // METADATA_BATCH_MAX) + 4, 4)


class InjectedRecordFault(RuntimeError):
    """A record-path FAILURE (an ordinary exception), not a process death.

    BL-8 (a): ``InjectedKill`` is now a true ``BaseException`` -- uncatchable,
    like the SIGKILL it stands in for. The terminal recording ladder's
    absorbed-death coverage models a record-path exception instead, which is
    exactly the class (``Exception``) the ladder is contracted to absorb into
    ``terminal-unrecordable``.
    """


def _kill_at(
    target: str,
    occurrence: int,
    exc_type: type[BaseException] = InjectedKill,
) -> Any:
    """Return a hook raising ``exc_type`` at the Nth ``target`` boundary."""

    seen = {"n": 0}

    def hook(boundary: str, _stable_id: str) -> None:
        if boundary == target:
            seen["n"] += 1
            if seen["n"] == occurrence:
                raise exc_type(f"{target}#{occurrence}")

    return hook


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("boundary", "occurrence", "absorbed"),
    [
        pytest.param("after-attempt", 1, False, id="mid-triage"),
        pytest.param("after-gate", 1, False, id="mid-gate"),
        # Occurrence 1 of post-award-commit belongs to a DEAD model's gate-free
        # TERMINAL commit: `_terminalize` converts a record-path death into a
        # durable terminal-unrecordable event (streak-limited) and the leg
        # SELF-HEALS -- the scheduler re-drives the model within the same leg.
        pytest.param("post-award-commit", 1, True, id="mid-terminal-commit"),
        # Occurrence 3 is a RUNS award commit: the death propagates as a
        # campaign abort and the resume must not double-award.
        pytest.param("post-award-commit", 3, False, id="mid-runs-award"),
    ],
)
def test_pipelined_kill_with_mixed_dead_intake_resumes(
    tmp_path: Path, boundary: str, occurrence: int, absorbed: bool
) -> None:
    """Kills interleaved with gate-free machine-death terminals still reconcile.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    boundary, occurrence:
        Boundary hook name and firing ordinal to kill at.
    absorbed:
        Whether the recording path absorbs the death (terminal-unrecordable)
        instead of propagating it.
    """

    count, dead_indices = 4, (1, 3)
    waves_root = tmp_path / "waves"
    _, _, waves_result, _, _ = _run_leg(waves_root, count, dead_indices, "waves")

    root = tmp_path / "pipelined"
    root.mkdir()
    snapshot = _snapshot(root, count=count)
    dead_ids = tuple(snapshot.items[index].stable_id for index in dead_indices)
    paths = _paths(root, snapshot)
    driver1 = _driver(
        root,
        snapshot,
        forward=SelectivelyDeadForward(dead_ids),
        # The absorbed arm models a record-path FAILURE (the exception class
        # the recording ladder is contracted to absorb); the propagating arms
        # model a process death, which stays uncatchable.
        boundary=_kill_at(
            boundary, occurrence, exc_type=InjectedRecordFault if absorbed else InjectedKill
        ),
        scheduler="pipelined",
    )
    if absorbed:
        first = driver1.run()
        assert first.status == waves_result.status
        unrecordable = [
            event
            for event in scan_jsonl(paths.operational_ledger)
            if event["event_kind"] == "terminal-unrecordable"
        ]
        assert len(unrecordable) == 1
    else:
        with pytest.raises(InjectedKill):
            driver1.run()

    durable_first, durable_confirm = _durable_metadata_state(paths)
    checker2 = _ClassifiedChecker()
    driver2 = _driver(
        root,
        snapshot,
        forward=SelectivelyDeadForward(dead_ids),
        checker=checker2,
        scheduler="pipelined",
    )
    checker2.mark_confirmations(driver2)
    resumed = driver2.run()
    assert resumed.status == waves_result.status
    heads = _model_heads(paths)
    for index, item in enumerate(snapshot.items):
        expected = "failed:import" if index in dead_indices else "runs"
        assert heads[item.stable_id]["status"]["code"] == expected
        if index in dead_indices:
            assert heads[item.stable_id]["authored_metadata_state"] == "pending"
    runs_revisions = sum(
        1 for record in scan_jsonl(paths.ledgers.models) if record["status"]["kind"] == "runs"
    )
    assert runs_revisions == count - len(dead_indices)
    _assert_per_model_lineage(paths)
    # Dead models never reach the checker lane, killed leg or not, and the
    # resume buys exactly the ALIVE models' durable-state complement: every
    # persisted gate/confirmation reuses; only in-flight kill losses re-buy.
    alive_ids = {
        item.stable_id
        for index, item in enumerate(snapshot.items)
        if index not in dead_indices
    }
    assert len(set(checker2.sessions)) == len(checker2.sessions), checker2.sessions
    assert checker2.bought("first") == alive_ids - durable_first, checker2.sessions
    assert checker2.bought("confirm") == alive_ids - durable_confirm, checker2.sessions


@pytest.mark.heavy
@pytest.mark.parametrize("boundary", ["after-attempt", "after-gate", "award-commit-entered"])
def test_pipelined_sigkill_subprocess_resumes(tmp_path: Path, boundary: str) -> None:
    """A real SIGKILL (no unwinding, no drain) still resumes to the same close.

    The child process (``scheduler_fire_support``) hard-kills ITSELF at the
    target boundary, so no ``finally`` blocks run, pool threads die mid-flight,
    and the flock releases only by kernel action. The parent resume must
    acquire the stale ``DriverLock``, recover any torn ledger tail, and close
    the same partition -- with exactly one ``runs`` revision per model.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    boundary:
        Boundary at which the child SIGKILLs itself.
    """

    import os
    import subprocess
    import sys

    from menagerie.crawler.intake import load_intake_snapshot

    env = dict(os.environ)
    repo_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "menagerie.crawler.tests.scheduler_fire_support",
            "--root",
            str(tmp_path),
            "--count",
            "4",
            "--scheduler",
            "pipelined",
            "--kill-boundary",
            boundary,
        ],
        env=env,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert child.returncode == -9, (child.returncode, child.stdout, child.stderr)

    snapshot_dirs = sorted((tmp_path / "intake").glob("intake-*"))
    assert len(snapshot_dirs) == 1
    snapshot = load_intake_snapshot(snapshot_dirs[0])
    resumed = _driver(tmp_path, snapshot, scheduler="pipelined").run()
    assert resumed.status == "complete"
    paths = _paths(tmp_path, snapshot)
    heads = _model_heads(paths)
    assert len(heads) == 4
    assert all(head["status"]["kind"] == "runs" for head in heads.values())
    assert (
        sum(1 for record in scan_jsonl(paths.ledgers.models) if record["status"]["kind"] == "runs")
        == 4
    )
    _assert_per_model_lineage(paths)


@pytest.mark.heavy
def test_pipelined_resume_recovers_torn_ledger_tails(tmp_path: Path) -> None:
    """A SIGKILL-torn trailing half-line on canonical ledgers cannot brick resume.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=4)
    with pytest.raises(InjectedKill):
        _driver(
            tmp_path,
            snapshot,
            boundary=_kill_once("after-attempt"),
            scheduler="pipelined",
        ).run()
    paths = _paths(tmp_path, snapshot)
    for ledger in (paths.ledgers.models, paths.ledgers.attempts):
        if ledger.exists():
            with ledger.open("ab") as handle:
                handle.write(b'{"stable_id": "m_torn-tail-half-')

    resumed = _driver(tmp_path, snapshot, scheduler="pipelined").run()
    assert resumed.status == "complete"
    heads = _model_heads(paths)
    assert len(heads) == 4
    assert all(head["status"]["kind"] == "runs" for head in heads.values())
    _assert_per_model_lineage(paths)


@pytest.mark.heavy
def test_pipelined_review_threshold_pauses_mid_leg_and_stays_one_shot(
    tmp_path: Path,
) -> None:
    """The review checkpoint stops the pipelined leg AT the threshold, once.

    Beyond the merge-gate's threshold-1 case: a mid-leg threshold (3 of 7)
    must pause at exactly the third committed record, a re-run without
    ``--after-review`` must re-block WITHOUT buying any session, the signoff
    resume must complete the leg, and a fourth run must not re-pause (the
    policy is one-shot per intake snapshot).

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=7)
    paths = _paths(tmp_path, snapshot)

    first = _driver(tmp_path, snapshot, review_at=3, scheduler="pipelined").run()
    assert first.status == "paused:review-checkpoint"
    committed_at_pause = len(scan_jsonl(paths.ledgers.models))
    assert committed_at_pause == 3

    author2, checker2, forward2 = FakeAuthor(), FakeChecker(), FakeForward()
    reblocked = _driver(
        tmp_path,
        snapshot,
        author=author2,
        checker=checker2,
        forward=forward2,
        review_at=3,
        scheduler="pipelined",
    ).run()
    assert reblocked.status == "paused:review-checkpoint"
    # The re-block is adjudicated at leg start from the durable checkpoint
    # event: no session of any kind may be bought while blocked.
    assert _spend(author2, checker2, forward2) == (0, 0, 0)
    assert len(scan_jsonl(paths.ledgers.models)) == committed_at_pause

    resumed = _driver(tmp_path, snapshot, review_at=3, scheduler="pipelined").run(after_review=True)
    assert resumed.status == "complete"
    heads = _model_heads(paths)
    assert len(heads) == 7
    assert all(head["status"]["kind"] == "runs" for head in heads.values())

    fourth = _driver(tmp_path, snapshot, review_at=3, scheduler="pipelined").run()
    assert fourth.status == "complete"
    reviews = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event["event_kind"] in ("checkpoint-review", "review-signoff")
    ]
    assert [event["event_kind"] for event in reviews] == [
        "checkpoint-review",
        "review-signoff",
    ]
    _assert_per_model_lineage(paths)


@pytest.mark.heavy
def test_pipelined_review_threshold_from_triage_terminals_costs_no_checker(
    tmp_path: Path,
) -> None:
    """Gate-free machine-death terminals crossing the threshold buy no gates.

    Under waves a review threshold this low would still have bought the whole
    metadata wave first; under execute-first the pause lands before ANY
    checker session when the crossing terminals are triage deaths.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=4)
    dead_ids = tuple(item.stable_id for item in snapshot.items)
    checker = FakeChecker()
    first = _driver(
        tmp_path,
        snapshot,
        checker=checker,
        forward=SelectivelyDeadForward(dead_ids),
        review_at=2,
        scheduler="pipelined",
    ).run()
    assert first.status == "paused:review-checkpoint"
    assert checker.metadata_calls == 0
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.gates)) == 0

    resumed = _driver(
        tmp_path,
        snapshot,
        forward=SelectivelyDeadForward(dead_ids),
        review_at=2,
        scheduler="pipelined",
    ).run(after_review=True)
    assert resumed.status == "terminal-partition-complete"
    heads = _model_heads(paths)
    assert all(head["status"]["code"] == "failed:import" for head in heads.values())
    assert len(scan_jsonl(paths.ledgers.gates)) == 0


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("first_mode", "second_mode", "boundary"),
    [
        pytest.param("pipelined", "waves", "after-gate", id="rollback-mid-campaign"),
        pytest.param("waves", "pipelined", "after-gate", id="cutover-mid-campaign"),
        pytest.param("pipelined", "waves", "after-attempt", id="rollback-mid-triage"),
        pytest.param("waves", "pipelined", "after-author", id="cutover-mid-author"),
    ],
)
def test_scheduler_mode_switch_mid_campaign_finishes_the_leg(
    tmp_path: Path, first_mode: str, second_mode: str, boundary: str
) -> None:
    """A leg killed under one scheduler is finished by the OTHER scheduler.

    M4's migration story, rehearsed in both directions: records are
    scheduler-agnostic, so a rollback (pipelined -> waves) or cutover
    (waves -> pipelined) from any intermediate state must drive the same
    campaign to the same close as a pure-wave reference -- same heads, same
    single ``runs`` revision per model, no re-bought sessions for settled
    evidence.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    first_mode, second_mode:
        Scheduler modes for the killed leg and the finishing leg.
    boundary:
        Kill point inside the first leg.
    """

    count = 4
    waves_root = tmp_path / "reference"
    waves_snapshot, _, waves_result, _, _ = _run_leg(waves_root, count, (), "waves")
    assert waves_result.status == "complete"
    waves_heads = {
        stable_id: head["status"]["code"]
        for stable_id, head in _model_heads(_paths(waves_root, waves_snapshot)).items()
    }

    root = tmp_path / "campaign"
    root.mkdir()
    snapshot = _snapshot(root, count=count)
    with pytest.raises(InjectedKill):
        _driver(
            root,
            snapshot,
            boundary=_kill_once(boundary),
            scheduler=first_mode,
        ).run()

    author2, checker2, forward2 = FakeAuthor(), FakeChecker(), FakeForward()
    finished = _driver(
        root,
        snapshot,
        author=author2,
        checker=checker2,
        forward=forward2,
        scheduler=second_mode,
    ).run()
    assert finished.status == "complete"
    paths = _paths(root, snapshot)
    assert {k: v["status"]["code"] for k, v in _model_heads(paths).items()} == waves_heads
    assert (
        sum(1 for record in scan_jsonl(paths.ledgers.models) if record["status"]["kind"] == "runs")
        == count
    )
    _assert_per_model_lineage(paths)
    # Cross-scheduler resume must not re-buy author sessions (W5 rehydration
    # is scheduler-agnostic).
    assert sum(author2.calls.values()) == 0


@pytest.mark.heavy
def test_pipelined_all_slow_metadata_repair_ladder_closes(tmp_path: Path) -> None:
    """An intake whose every gate adjudicates inaccurate terminalizes, never spins.

    The all-slow-metadata pathological case: the checker rejects every model's
    metadata until the disposition/accuracy machinery terminalizes it. The
    pipelined leg must reach the same close as waves without a livelock (the
    zero-progress tripwire never fires because every round commits durable
    routing progress).

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    from menagerie.crawler.tests.test_slice_f_driver import InaccurateChecker

    waves_root = tmp_path / "waves"
    waves_root.mkdir()
    waves_snapshot = _snapshot(waves_root, count=7)
    waves_result = _driver(
        waves_root, waves_snapshot, checker=InaccurateChecker(), scheduler="waves"
    ).run()

    pipelined_root = tmp_path / "pipelined"
    pipelined_root.mkdir()
    snapshot = _snapshot(pipelined_root, count=7)
    result = _driver(
        pipelined_root, snapshot, checker=InaccurateChecker(), scheduler="pipelined"
    ).run()
    assert result.status == waves_result.status == "terminal-partition-complete"
    paths = _paths(pipelined_root, snapshot)
    waves_heads = _model_heads(_paths(waves_root, waves_snapshot))
    for stable_id, head in _model_heads(paths).items():
        assert head["status"]["code"] == "failed:accuracy-gate"
        assert head["status"]["code"] == waves_heads[stable_id]["status"]["code"]
    _assert_per_model_lineage(paths)


@pytest.mark.heavy
def test_pipelined_mode_repair_identity_rotation_reexecutes_and_awards(
    tmp_path: Path,
) -> None:
    """A gate-round repair that rotates the recipe identity still awards.

    The exec-ready-award lane (scheduler step 5 fallback): when the gate round
    repairs the artifact, the recorded triage attempts belong to the OLD
    identity, `_award_recorded_run` must refuse, and the wave-equivalent
    execute-and-award env turn must re-run the model under the repaired
    identity. The merge-gate suite never exercises this lane.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    from menagerie.crawler.tests.test_slice_f_driver import DetectedModeRepairAuthor

    waves_root = tmp_path / "waves"
    waves_root.mkdir()
    waves_snapshot = _snapshot(waves_root, count=3)
    waves_forward = FakeForward()
    waves_result = _driver(
        waves_root,
        waves_snapshot,
        author=DetectedModeRepairAuthor(),
        forward=waves_forward,
        scheduler="waves",
    ).run()

    root = tmp_path / "pipelined"
    root.mkdir()
    snapshot = _snapshot(root, count=3)
    forward = FakeForward()
    result = _driver(
        root,
        snapshot,
        author=DetectedModeRepairAuthor(),
        forward=forward,
        scheduler="pipelined",
    ).run()
    assert result.status == waves_result.status
    paths = _paths(root, snapshot)
    waves_heads = _model_heads(_paths(waves_root, waves_snapshot))
    for stable_id, head in _model_heads(paths).items():
        assert head["status"]["code"] == waves_heads[stable_id]["status"]["code"]
    # The repair rotated identity after triage, so each model re-executes:
    # exactly the wave driver's execution count, never more.
    assert forward.calls == waves_forward.calls
    _assert_per_model_lineage(paths)


@pytest.mark.heavy
def test_pipelined_intent_skew_completes_with_bounded_env_turns(tmp_path: Path) -> None:
    """A 10:1 intent skew starves nothing and stays within the turn budget.

    One env intent holds ten models, the other one; the threshold-gated turn
    triggers plus the leg-end sweep must close both without starving the
    minority intent and without buying more than the per-intent turn budget
    (normal legs: <=2 non-currentness turns per intent, wave parity).

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    from collections import Counter

    from menagerie.crawler.tests.test_slice_f_driver import (
        FakeEnvironments,
        _mixed_phase_snapshot,
    )

    snapshot = _mixed_phase_snapshot(tmp_path)
    environments = FakeEnvironments(tmp_path / "fake-envs")
    result = _driver(tmp_path, snapshot, environments=environments, scheduler="pipelined").run()
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    heads = _model_heads(paths)
    assert len(heads) == len(snapshot.items)
    assert all(head["status"]["kind"] == "runs" for head in heads.values())
    creates = Counter(event for event in environments.events if event.startswith("create:"))
    assert creates and all(count <= 3 for count in creates.values()), creates
    _assert_per_model_lineage(paths)


@pytest.mark.heavy
def test_pipelined_dead_family_representative_matches_wave_outcome(
    tmp_path: Path,
) -> None:
    """An import-dead family representative routes variants identically to waves.

    The nastiest ordering case for out-of-order close: variants become
    author-ready only through the representative's settlement, and a
    representative that settles UNUSABLE (terminal failure) hands routing to
    the author lane's fail-closed authority rule. Whatever that rule decides,
    both schedulers must decide it identically.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    from menagerie.crawler.tests.test_slice_f_driver import _family_snapshot

    outcomes: dict[str, object] = {}
    for mode in ("waves", "pipelined"):
        root = tmp_path / mode
        root.mkdir()
        snapshot, representative_id, _variant_ids = _family_snapshot(root)
        dead = SelectivelyDeadForward((representative_id,))
        try:
            result = _driver(root, snapshot, forward=dead, scheduler=mode).run()
            outcomes[mode] = (
                "closed",
                result.status,
                {
                    stable_id: head["status"]["code"]
                    for stable_id, head in _model_heads(_paths(root, snapshot)).items()
                },
            )
        except Exception as exc:  # noqa: BLE001 -- parity of the raised class is the oracle
            outcomes[mode] = ("raised", type(exc).__name__, str(exc)[:120])
    assert outcomes["pipelined"] == outcomes["waves"], outcomes


@pytest.mark.heavy
def test_scheduler_transition_events_record_each_mode_change(tmp_path: Path) -> None:
    """Cutover/rollback provenance events land exactly once per transition.

    D1 migration ceremony: a campaign starting under ``pipelined`` records one
    ``scheduler-cutover`` from the ``waves`` default; a same-mode leg appends
    nothing; each subsequent direction change appends one event with a
    ledger-derived occurrence and canonical tail counts/digests.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)

    def transitions() -> list[tuple[str, str, str, int]]:
        return [
            (
                str(event["event_kind"]),
                str(event["details"]["from_scheduler"]),
                str(event["details"]["to_scheduler"]),
                int(event["details"]["occurrence"]),
            )
            for event in scan_jsonl(paths.operational_ledger)
            if event["event_kind"] in ("scheduler-cutover", "scheduler-rollback")
        ]

    assert _driver(tmp_path, snapshot, scheduler="pipelined").run().status == "complete"
    assert transitions() == [("scheduler-cutover", "waves", "pipelined", 1)]
    assert _driver(tmp_path, snapshot, scheduler="pipelined").run().status == "complete"
    assert transitions() == [("scheduler-cutover", "waves", "pipelined", 1)]
    assert _driver(tmp_path, snapshot, scheduler="waves").run().status == "complete"
    assert _driver(tmp_path, snapshot, scheduler="pipelined").run().status == "complete"
    assert transitions() == [
        ("scheduler-cutover", "waves", "pipelined", 1),
        ("scheduler-rollback", "pipelined", "waves", 2),
        ("scheduler-cutover", "waves", "pipelined", 3),
    ]
    events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event["event_kind"] in ("scheduler-cutover", "scheduler-rollback")
    ]
    # The first transition attests empty ledgers; later ones attest the tails.
    first_tails = events[0]["details"]["ledger_tails"]
    assert first_tails["models"] == {"count": 0, "last_payload_sha256": None}
    last_tails = events[-1]["details"]["ledger_tails"]
    assert last_tails["models"]["count"] == 2
    assert str(last_tails["models"]["last_payload_sha256"]).startswith("sha256:")


@pytest.mark.heavy
def test_shadow_audit_explains_every_intake_id_at_every_campaign_stage(
    tmp_path: Path,
) -> None:
    """The projection-only shadow audit refuses nothing on honest campaigns.

    Fresh (nothing run), mid-flight (killed after the first gate), and closed
    campaigns must each classify EVERY intake ID into exactly one explainable
    next state -- the D1 cutover acceptance audit, using only durable state.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    from menagerie.crawler.tools.scheduler_shadow_audit import (
        resolve_ledger_paths,
        shadow_audit,
    )

    snapshot = _snapshot(tmp_path, count=4)
    dead_ids = (snapshot.items[1].stable_id,)
    runtime = tmp_path / "runtime"

    fresh = shadow_audit(
        snapshot,
        _paths(tmp_path, snapshot).ledgers,
        runtime,
    )
    assert fresh["counts"] == {"awaiting-author": 4}
    assert not fresh["unexplained"]

    with pytest.raises(InjectedKill):
        _driver(
            tmp_path,
            snapshot,
            forward=SelectivelyDeadForward(dead_ids),
            boundary=_kill_once("after-gate"),
            scheduler="pipelined",
        ).run()
    ledgers = resolve_ledger_paths(tmp_path / "records")
    mid = shadow_audit(snapshot, ledgers, runtime)
    assert not mid["unexplained"]
    assert sum(mid["counts"].values()) == 4

    result = _driver(
        tmp_path, snapshot, forward=SelectivelyDeadForward(dead_ids), scheduler="pipelined"
    ).run()
    assert result.status == "terminal-partition-complete"
    closed = shadow_audit(snapshot, ledgers, runtime)
    assert closed["counts"] == {"settled-runs": 3, "settled-terminal": 1}
    assert not closed["unexplained"]


@pytest.mark.heavy
@pytest.mark.parametrize("count", [1, 4, 8])
def test_pipelined_all_alive_widths_project_identically(tmp_path: Path, count: int) -> None:
    """All-alive widths 1/4/8 are append-order projection-identical to waves.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    count:
        Intake width.
    """

    waves_root = tmp_path / "waves"
    pipelined_root = tmp_path / "pipelined"
    waves_snapshot, _, waves_result, _, _ = _run_leg(waves_root, count, (), "waves")
    snapshot, _, pipelined_result, _, _ = _run_leg(pipelined_root, count, (), "pipelined")
    assert waves_result.status == "complete"
    assert pipelined_result.status == "complete"
    assert _canonical_projection(
        _paths(pipelined_root, snapshot), pipelined_root
    ) == _canonical_projection(_paths(waves_root, waves_snapshot), waves_root)
