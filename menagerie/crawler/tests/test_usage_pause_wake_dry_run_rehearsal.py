"""Usage-pause and wake must survive the acceptance dry-run CLI composition.

The pause/wake machinery has dense unit coverage (``test_slice_e_wakeup_events``,
``test_provider_weather_arms``, ``test_slice_f_driver``) but had no end-to-end
rehearsal through the composition an actual campaign runs: the CLI's driver
factory, the real ``WakeupManager``, the canonical operational ledger, the paid
spend authority, and the wake-callback resume. A launch that hits provider quota
-- the expected case, not the exotic one -- exercises exactly that composed path
for the first time in production unless something rehearses it first.

Both rehearsals drive the SAME injected lane
(``dry_run_support.UsagePauseAuthor``), which raises the production
``AuthorBackoffError``, and assert the DURABLE evidence at each step of the cycle:

    run -> paused:usage-limit (pause row + episode + verified projection)
        -> wake callback BEFORE the reset (guarded fire, zero work)
        -> wake callback AFTER the reset (usage-resume, campaign completes)

The OS-scheduler boundary is recorded, never executed (see
``dry_run_support.RecordingWakeScheduler``); the durable records are what is under
test, not ``launchctl``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterator, Mapping, Optional, Sequence

import pytest

from menagerie.crawler import cli as cli_module
from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.cli import EXIT_OK, EXIT_PAUSED
from menagerie.crawler.constants import AuthorPauseReason
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests import dry_run_support
from menagerie.crawler.tests.conftest import NOW, RealEnvironmentFixture
from menagerie.crawler.tests.dry_run_support import (
    CLOCK_OFFSET_ENV,
    DEFAULT_INJECTED_RESET_SECONDS,
    INJECT_USAGE_PAUSE_ENV,
    INJECTED_USAGE_RESPONSE,
    USAGE_PAUSE_MARKER,
    UsagePauseInjection,
    build_dry_run_driver,
    create_dry_run_snapshot,
    dry_run_paths,
    read_wake_os_registrations,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeEnvironments,
    FakeForward,
    _unlocked_registry,
)
from menagerie.crawler.wakeup import reduce_wake_episodes

#: Injected reset lands five seconds after the pause, so the "before reset" wake
#: fires at the frozen clock and the "after reset" wake advances past it. 1800s is
#: two recurrence buckets away (interval 900s), so the second fire is never
#: swallowed as an idempotent replay of the first.
_AFTER_RESET_OFFSET_SECONDS = 1800

#: Every model in the dry-run corpus, all of which must complete after the resume.
_EXPECTED_MODELS = 10


def _operational_rows(campaign_root: Path) -> tuple[Mapping[str, Any], ...]:
    """Read the canonical operational event ledger for one dry-run campaign.

    Parameters
    ----------
    campaign_root:
        Disposable dry-run campaign root.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Append-ordered operational events.
    """

    snapshot = create_dry_run_snapshot(campaign_root)
    paths = dry_run_paths(campaign_root, snapshot)
    path = canonical_operational_ledger_path(paths.ledgers.models)
    if not path.is_file():
        return ()
    return tuple(scan_jsonl(path))


def _spend_rows(campaign_root: Path) -> tuple[Mapping[str, Any], ...]:
    """Read the chain-verified paid-session spend ledger.

    Parameters
    ----------
    campaign_root:
        Disposable dry-run campaign root.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Append-ordered paid-session rows, or empty when no session was admitted.
    """

    snapshot = create_dry_run_snapshot(campaign_root)
    paths = dry_run_paths(campaign_root, snapshot)
    if not paths.paid_session_ledger.is_file():
        return ()
    return tuple(scan_jsonl(paths.paid_session_ledger, validate=False))


def _author_session_scopes(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return one identity per admitted author paid session, in append order.

    Parameters
    ----------
    rows:
        Paid-session ledger rows.

    Returns
    -------
    tuple[str, ...]
        ``<stable_id>/<work_id>/<spend_id>`` for each reserved author session.
    """

    scopes: list[str] = []
    for row in rows:
        if row.get("event") != "session_reserved" or row.get("lane") != "author":
            continue
        axis = row.get("axis")
        assert isinstance(axis, Mapping)
        generations = axis.get("work_generations")
        assert isinstance(generations, Sequence) and generations
        member = generations[0]
        scopes.append(f"{member['stable_id']}/{member['work_id']}/{row['spend_id']}")
    return tuple(scopes)


def _unsettled_author_sessions(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return author spend IDs reserved without a durable settlement.

    An unsettled reservation is the ambiguity that makes the driver refuse paid
    admission until the provider reconciler answers, so it is the exact quantity a
    pause must not leave behind.

    Parameters
    ----------
    rows:
        Paid-session ledger rows.

    Returns
    -------
    tuple[str, ...]
        Sorted spend IDs with a reservation and no settlement.
    """

    reserved = {
        str(row["spend_id"])
        for row in rows
        if row.get("event") == "session_reserved" and row.get("lane") == "author"
    }
    settled = {
        str(row["spend_id"]) for row in rows if row.get("event") == "session_settled"
    }
    return tuple(sorted(reserved - settled))


def _completed_author_models(rows: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """Return models whose author session was PAID and settled as completed.

    Parameters
    ----------
    rows:
        Paid-session ledger rows.

    Returns
    -------
    frozenset[str]
        Stable IDs already charged for a successful author session.
    """

    owner: dict[str, str] = {}
    for row in rows:
        if row.get("event") == "session_reserved" and row.get("lane") == "author":
            axis = row.get("axis")
            assert isinstance(axis, Mapping)
            generations = axis.get("work_generations")
            assert isinstance(generations, Sequence) and generations
            owner[str(row["spend_id"])] = str(generations[0]["stable_id"])
    return frozenset(
        owner[str(row["spend_id"])]
        for row in rows
        if row.get("event") == "session_settled"
        and row.get("local_outcome") == "completed"
        and str(row["spend_id"]) in owner
    )


def _author_reserves_by_model(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    """Count reserved author sessions per model.

    Parameters
    ----------
    rows:
        Paid-session ledger rows.

    Returns
    -------
    Mapping[str, int]
        Stable ID to reserved author-session count.
    """

    counts: dict[str, int] = {}
    for scope in _author_session_scopes(rows):
        stable_id = scope.split("/", 1)[0]
        counts[stable_id] = counts.get(stable_id, 0) + 1
    return counts


def _pause_rows(rows: Sequence[Mapping[str, Any]], kind: str) -> tuple[Mapping[str, Any], ...]:
    """Return operational rows of one event kind.

    Parameters
    ----------
    rows, kind:
        Operational rows and the exact ``event_kind`` to select.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Matching rows in append order.
    """

    return tuple(row for row in rows if row.get("event_kind") == kind)


def _episode_id(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the single opened wake episode's identity.

    Parameters
    ----------
    rows:
        Operational rows containing exactly one opened episode.

    Returns
    -------
    str
        Frozen episode identity.
    """

    projection = reduce_wake_episodes(list(rows))
    assert len(projection.episodes) == 1, projection.episodes
    return next(iter(projection.episodes))


def _reset_instant() -> str:
    """Return the injected reset instant relative to the frozen clock.

    Returns
    -------
    str
        RFC 3339 ``Z`` reset timestamp the pause must record.
    """

    hour, minute, second = 12, 0, DEFAULT_INJECTED_RESET_SECONDS
    assert NOW == f"2026-07-14T{hour:02d}:{minute:02d}:00Z"
    return f"2026-07-14T{hour:02d}:{minute:02d}:{second:02d}Z"


class _SyntheticUsagePauseAuthor(UsagePauseInjection, FakeAuthor):
    """The shipped one-shot usage-pause injection over the synthetic unit corpus.

    Identical injected signal and identical one-shot marker semantics as
    ``dry_run_support.UsagePauseAuthor``; only the corpus differs. The tiny corpus
    exists so a REAL worker executes a REAL adapter inside the materialized
    environment, and its adapters import the fixture-only sentinel -- neither of
    which the faked forward lane can provide. Substituting the synthetic corpus is
    therefore part of the same "fake exactly the lanes that need real execution"
    boundary, not a weakening of the pause path under test.
    """


def _in_process_factory(
    repo_root: Path, campaign_root: Path, prefix: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Return a driver factory composing the dry run with in-process lanes.

    The composition under test is ``build_dry_run_driver`` itself: its config, its
    paths, its notifier, clock, wake stubs, provider reconciler, and the injected
    usage-pause signal are all real, as is every ledger it writes. The three lanes
    that need a materialized conda prefix and a working OS sandbox -- the
    environment lifecycle, the supervised forward, and (because nothing executes an
    adapter here) the tiny corpus behind the injection -- are swapped for the shared
    in-process fakes, so this rehearsal runs on any host. The real-subprocess
    variant below covers those three for real.

    Parameters
    ----------
    repo_root, campaign_root, prefix, monkeypatch:
        Checked-out source, disposable campaign, fake prefix root, and patcher.

    Returns
    -------
    Any
        Factory accepting parsed CLI arguments and returning the composed driver.
    """

    monkeypatch.setattr(
        dry_run_support,
        "_materialized_environment",
        lambda selected: (FakeEnvironments(selected), _unlocked_registry()),
    )
    monkeypatch.setattr(
        dry_run_support,
        "SupervisedForwardLane",
        lambda **_kwargs: FakeForward(),
    )
    monkeypatch.setattr(dry_run_support, "UsagePauseAuthor", _SyntheticUsagePauseAuthor)
    # EN-2 (Wave-2 batch): these in-process lanes swap in the UNLOCKED fixture
    # registry, so the composition must carry the lockless-route exemption
    # explicitly (as the slice-f harness does); the real-prefix variant keeps
    # the locked registry and no exemption.
    real_driver_config = dry_run_support.DriverConfig

    def _lockless_exempt_config(**kwargs: Any) -> Any:
        return real_driver_config(
            **kwargs,
            allow_lockless_intents=tuple(sorted(_unlocked_registry().intents)),
        )

    monkeypatch.setattr(dry_run_support, "DriverConfig", _lockless_exempt_config)

    def factory(_args: Any) -> CrawlerDriver:
        """Compose one dry-run driver exactly as the CLI's dry-run branch does."""

        return build_dry_run_driver(
            repo_root,
            campaign_root,
            prefix,
            review_checkpoint_at=None,
            progress_milestones=(),
            run_id="usage-pause-rehearsal",
        )

    return factory


def _cli_json(capsys: pytest.CaptureFixture[str]) -> Mapping[str, Any]:
    """Decode the final structured stdout line from one in-process CLI call.

    Parameters
    ----------
    capsys:
        Pytest capture fixture bound to the CLI call.

    Returns
    -------
    Mapping[str, Any]
        Final JSON object the CLI printed.
    """

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip().startswith("{")]
    assert lines, (captured.out, captured.err)
    value = json.loads(lines[-1])
    assert isinstance(value, dict)
    return value


@pytest.fixture
def paused_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Iterator[tuple[Path, Path, Any, Mapping[str, Any]]]:
    """Pause one in-process dry-run campaign on the injected provider backoff.

    Yields
    ------
    tuple[pathlib.Path, pathlib.Path, Any, Mapping[str, Any]]
        Campaign root, selected prefix, the CLI driver factory, and the paused output.
    """

    repo_root = Path(__file__).resolve().parents[3]
    campaign_root = tmp_path / "usage-pause-campaign"
    prefix = tmp_path / "fake-prefix"
    factory = _in_process_factory(repo_root, campaign_root, prefix, monkeypatch)
    monkeypatch.setenv(INJECT_USAGE_PAUSE_ENV, "author")
    monkeypatch.delenv(CLOCK_OFFSET_ENV, raising=False)
    argv = [
        "--repo-root",
        str(repo_root),
        "run",
        "--dry-run",
        "--dry-run-root",
        str(campaign_root),
        "--dry-run-environment-prefix",
        str(prefix),
    ]
    assert cli_module.main(argv, driver_factory=factory) == EXIT_PAUSED
    output = _cli_json(capsys)
    assert output["status"] == "paused:usage-limit"
    yield campaign_root, prefix, factory, output


def test_injected_backoff_pauses_the_composed_dry_run_with_durable_wake_records(
    paused_campaign: tuple[Path, Path, Any, Mapping[str, Any]],
) -> None:
    """The composed dry run must record the pause, the episode, and its projection."""

    campaign_root, _prefix, _factory, output = paused_campaign
    assert output["paused_reason"] == AuthorPauseReason.QUOTA_EXHAUSTED.value

    rows = _operational_rows(campaign_root)
    pauses = _pause_rows(rows, "usage-pause")
    assert len(pauses) == 1
    pause = pauses[0]
    assert pause["status"] == "paused:usage-limit"
    assert pause["provider"] == "anthropic"
    assert pause["reset_at"] == _reset_instant()
    assert pause["observed_response"] == INJECTED_USAGE_RESPONSE
    assert pause["details"]["reset_exact"] is True
    assert pause["details"]["episode"]["reset_observation"] == "observed"
    assert pause["details"]["episode"]["not_before"] == _reset_instant()

    installs = _pause_rows(rows, "wakeup-installed")
    assert len(installs) == 1
    assert installs[0]["details"]["episode_id"] == _episode_id(rows)
    assert installs[0]["details"]["verified"] is True
    assert installs[0]["details"]["recurring"] is True

    # Every paid author session admitted before the pause is durably settled: a
    # pause must not strand a reservation, because an unsettled one refuses paid
    # admission on the resume until the provider reconciler answers for it.
    assert _unsettled_author_sessions(_spend_rows(campaign_root)) == ()

    projection = reduce_wake_episodes(list(rows))
    state = projection.episodes[_episode_id(rows)]
    assert state.active
    assert state.install_verified
    assert state.episode.not_before == _reset_instant()
    assert state.episode.callback_argv[-2:] == ("--wake-episode-id", state.episode.episode_id)

    # The OS registration is recorded, never executed.
    registrations = read_wake_os_registrations(campaign_root)
    assert [row["action"] for row in registrations] == ["install", "verify"]
    assert {row["episode_id"] for row in registrations} == {state.episode.episode_id}

    # The pause fired exactly once and durably said so.
    marker = json.loads((campaign_root / USAGE_PAUSE_MARKER).read_text(encoding="utf-8"))
    assert marker["reset_at"] == _reset_instant()
    assert marker["reason"] == AuthorPauseReason.QUOTA_EXHAUSTED.value

    # A usage pause is never a model terminal: nothing was published.
    snapshot = create_dry_run_snapshot(campaign_root)
    paths = dry_run_paths(campaign_root, snapshot)
    assert not paths.ledgers.models.is_file() or scan_jsonl(paths.ledgers.models) == []


def test_wake_before_reset_does_no_work_and_after_reset_completes_the_campaign(
    paused_campaign: tuple[Path, Path, Any, Mapping[str, Any]],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarded wake must do nothing; the wake past the reset must finish the run."""

    campaign_root, prefix, factory, _output = paused_campaign
    repo_root = Path(__file__).resolve().parents[3]
    paused_rows = _operational_rows(campaign_root)
    episode_id = _episode_id(paused_rows)
    spend_before = _spend_rows(campaign_root)
    author_sessions_before = _author_session_scopes(spend_before)

    argv = [
        "--repo-root",
        str(repo_root),
        "run",
        "--dry-run",
        "--dry-run-root",
        str(campaign_root),
        "--dry-run-environment-prefix",
        str(prefix),
        "--wake-episode-id",
        episode_id,
    ]

    # 1. The scheduled callback fires BEFORE the provider reset. The guard exits
    #    before any driver work, and the campaign is STILL PAUSED at a resumable
    #    point (the episode's not_before), which is exactly the state EXIT_PAUSED
    #    names: the CLI's own paused carve-out already outranks acceptance grading
    #    (the run that CREATED the pause exits EXIT_PAUSED, not EXIT_ERROR, with
    #    acceptance pending), so the guarded wake that bounces off the same pause
    #    grades pending too. EXIT_OK stays reserved for a passed acceptance and
    #    EXIT_ERROR for a graded failure -- a dry run that never ran still cannot
    #    exit 0.
    assert cli_module.main(argv, driver_factory=factory) == EXIT_PAUSED
    guarded_output = _cli_json(capsys)
    assert guarded_output["status"] == "wake-noop"
    assert guarded_output["models_reduced"] == 0
    assert guarded_output["acceptance"]["status"] == "pending"
    assert guarded_output["acceptance"]["reason_code"] == "driver-paused"
    guarded_rows = _operational_rows(campaign_root)
    fires = _pause_rows(guarded_rows, "wakeup-fired")
    assert len(fires) == 1
    assert fires[0]["details"]["guarded_not_before"] is True
    assert _pause_rows(guarded_rows, "usage-resume") == ()
    assert reduce_wake_episodes(list(guarded_rows)).episodes[episode_id].active
    # A guarded wake is not a resumption: no model, and no new paid session.
    assert _author_session_scopes(_spend_rows(campaign_root)) == author_sessions_before

    # 2. The scheduled callback fires AFTER the provider reset.
    monkeypatch.setenv(CLOCK_OFFSET_ENV, str(_AFTER_RESET_OFFSET_SECONDS))
    assert cli_module.main(argv, driver_factory=factory) == EXIT_OK
    resumed_output = _cli_json(capsys)
    assert resumed_output["status"] == "complete"
    assert resumed_output["acceptance"]["status"] == "passed"
    assert resumed_output["funnel"]["models:total"] == _EXPECTED_MODELS
    assert resumed_output["funnel"]["status:runs"] == _EXPECTED_MODELS

    resumed_rows = _operational_rows(campaign_root)
    resumed_fires = _pause_rows(resumed_rows, "wakeup-fired")
    assert len(resumed_fires) == 2
    assert resumed_fires[-1]["details"]["guarded_not_before"] is False
    resumes = _pause_rows(resumed_rows, "usage-resume")
    assert len(resumes) == 1
    assert resumes[0]["details"]["episode_id"] == episode_id
    assert _pause_rows(resumed_rows, "wakeup-deactivated")
    assert not reduce_wake_episodes(list(resumed_rows)).episodes[episode_id].active
    assert [row["action"] for row in read_wake_os_registrations(campaign_root)][-1] == "deactivate"

    # 3. No paid author session was spent twice across the pause. The spend ledger
    #    is append-only, so the pre-pause reservations must still be exactly those
    #    rows, every reservation must be settled (an unsettled one is the ambiguity
    #    that refuses paid admission), and -- the actual double-spend hazard -- a
    #    model already CHARGED for a completed author session before the pause must
    #    not be charged again by the resumed leg.
    spend_after = _spend_rows(campaign_root)
    author_sessions_after = _author_session_scopes(spend_after)
    assert author_sessions_after[: len(author_sessions_before)] == author_sessions_before
    assert len(set(author_sessions_after)) == len(author_sessions_after)
    assert _unsettled_author_sessions(spend_after) == ()
    paid_before = _completed_author_models(spend_before)
    assert paid_before, "the pause must land after at least one paid author session"
    reserves_before = _author_reserves_by_model(spend_before)
    reserves_after = _author_reserves_by_model(spend_after)
    recharged = {
        stable_id: (reserves_before[stable_id], reserves_after[stable_id])
        for stable_id in sorted(paid_before)
        if reserves_after[stable_id] != reserves_before[stable_id]
    }
    assert not recharged, f"models re-charged for an already paid author session: {recharged}"


def _subprocess_dry_run(
    repo_root: Path,
    campaign_root: Path,
    environment_prefix: Path,
    *,
    wake_episode_id: Optional[str] = None,
    clock_offset_seconds: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Run one real dry-run CLI process with the usage-pause injection armed.

    Parameters
    ----------
    repo_root, campaign_root, environment_prefix:
        Checked-out source, disposable campaign, and strictly bound real prefix.
    wake_episode_id:
        Scheduled wake episode this process is the callback for, when any.
    clock_offset_seconds:
        Deterministic advance of the frozen dry-run clock.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Captured real CLI process result.
    """

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    environment["MENAGERIE_RELEASE_GATE"] = "1"
    environment[INJECT_USAGE_PAUSE_ENV] = "author"
    environment[CLOCK_OFFSET_ENV] = str(clock_offset_seconds)
    argv = [
        str(environment_prefix / "bin" / "python"),
        "-B",
        "-m",
        "menagerie.crawler",
        "--repo-root",
        str(repo_root),
        "run",
        "--dry-run",
        "--dry-run-root",
        str(campaign_root),
        "--dry-run-environment-prefix",
        str(environment_prefix),
        "--progress-milestones",
        "",
    ]
    if wake_episode_id is not None:
        argv.extend(("--wake-episode-id", wake_episode_id))
    return subprocess.run(
        argv,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


def _subprocess_json(completed: subprocess.CompletedProcess[str]) -> Mapping[str, Any]:
    """Decode the final structured stdout line from one CLI subprocess.

    Parameters
    ----------
    completed:
        Captured CLI subprocess result.

    Returns
    -------
    Mapping[str, Any]
        Final JSON object printed by the crawler CLI.
    """

    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    assert lines, (completed.stdout, completed.stderr)
    value = json.loads(lines[-1])
    assert isinstance(value, dict)
    return value


@pytest.mark.heavy
def test_usage_pause_wake_cycle_survives_the_real_subprocess_dry_run(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """The same cycle must hold across real CLI processes and the real prefix.

    The in-process rehearsals above own the ledger detail; this one owns the parts
    only separate processes can prove: the durable records are re-derived from disk
    by a fresh interpreter, the pause survives process exit, and the wake callback
    resumes the real supervised worker inside the real materialized environment.
    """

    repo_root = Path(__file__).resolve().parents[3]
    campaign_root = tmp_path / "real-usage-pause-campaign"
    prefix = real_environment_fixture.prefix

    paused = _subprocess_dry_run(repo_root, campaign_root, prefix)
    assert paused.returncode == EXIT_PAUSED, (paused.stdout, paused.stderr)
    paused_output = _subprocess_json(paused)
    assert paused_output["status"] == "paused:usage-limit"
    assert paused_output["paused_reason"] == AuthorPauseReason.QUOTA_EXHAUSTED.value

    paused_rows = _operational_rows(campaign_root)
    pauses = _pause_rows(paused_rows, "usage-pause")
    assert len(pauses) == 1
    assert pauses[0]["reset_at"] == _reset_instant()
    assert pauses[0]["observed_response"] == INJECTED_USAGE_RESPONSE
    episode_id = _episode_id(paused_rows)
    assert reduce_wake_episodes(list(paused_rows)).episodes[episode_id].install_verified
    assert [row["action"] for row in read_wake_os_registrations(campaign_root)] == [
        "install",
        "verify",
    ]
    author_sessions_before = _author_session_scopes(_spend_rows(campaign_root))

    # The guarded wake leaves the campaign paused at a resumable point, so the
    # dry-run CLI reports it as EXIT_PAUSED (acceptance pending), same as the
    # in-process leg: one contract for both compositions.
    guarded = _subprocess_dry_run(repo_root, campaign_root, prefix, wake_episode_id=episode_id)
    assert guarded.returncode == EXIT_PAUSED, (guarded.stdout, guarded.stderr)
    guarded_json = _subprocess_json(guarded)
    assert guarded_json["status"] == "wake-noop"
    assert guarded_json["acceptance"]["status"] == "pending"
    guarded_rows = _operational_rows(campaign_root)
    assert _pause_rows(guarded_rows, "usage-resume") == ()
    assert _pause_rows(guarded_rows, "wakeup-fired")[0]["details"]["guarded_not_before"] is True
    assert _author_session_scopes(_spend_rows(campaign_root)) == author_sessions_before

    resumed = _subprocess_dry_run(
        repo_root,
        campaign_root,
        prefix,
        wake_episode_id=episode_id,
        clock_offset_seconds=_AFTER_RESET_OFFSET_SECONDS,
    )
    assert resumed.returncode == EXIT_OK, (resumed.stdout, resumed.stderr)
    resumed_output = _subprocess_json(resumed)
    assert resumed_output["status"] == "complete"
    assert resumed_output["acceptance"]["status"] == "passed"
    assert resumed_output["funnel"]["models:total"] == _EXPECTED_MODELS
    assert resumed_output["funnel"]["status:runs"] == _EXPECTED_MODELS

    resumed_rows = _operational_rows(campaign_root)
    assert len(_pause_rows(resumed_rows, "usage-resume")) == 1
    assert not reduce_wake_episodes(list(resumed_rows)).episodes[episode_id].active
    author_sessions_after = _author_session_scopes(_spend_rows(campaign_root))
    assert author_sessions_after[: len(author_sessions_before)] == author_sessions_before
    assert len(set(author_sessions_after)) == len(author_sessions_after)
