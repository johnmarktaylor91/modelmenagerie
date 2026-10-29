"""Tests for the W1.6 campaign supervisor and host-global execution flock."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import time
from typing import Any, BinaryIO, Optional, Sequence

import pytest

from menagerie.crawler.campaign_config import CampaignConfig
from menagerie.crawler.cli import (
    EXIT_ERROR,
    EXIT_LOCKED,
    EXIT_OPERATOR_OUTAGE,
    EXIT_PAUSED,
    EXIT_REVIEW_PAUSED,
)
from menagerie.crawler.driver_progress import CommandNotifier
from menagerie.crawler.driver_receipts import SupervisedForwardLane
from menagerie.crawler import worker_supervisor
from menagerie.crawler.supervisor import (
    CONSECUTIVE_FLAT_RESTART_LIMIT,
    CRASH_LIMIT,
    CrawlerSupervisor,
    QueueStall,
    SupervisorConfig,
    SupervisorError,
    render_launchd_plist,
    stalled_queue_jobs,
    validate_supervised_campaign,
)

NOW = datetime(2026, 7, 27, 16, 0, tzinfo=timezone.utc)


class FakeNotifier:
    """Record supervisor notifications and optionally fail initial attempts."""

    def __init__(self, *, failures: int = 0) -> None:
        """Configure the number of missing-receipt attempts."""

        self.failures = failures
        self.messages: list[tuple[str, str]] = []
        self.notification_classes: list[Optional[str]] = []

    def notify(
        self,
        summary: str,
        *,
        idempotency_key: str,
        notification_class: Optional[str] = None,
    ) -> bool:
        """Record one notification and return its scripted receipt status."""

        self.messages.append((summary, idempotency_key))
        self.notification_classes.append(notification_class)
        if self.failures:
            self.failures -= 1
            return False
        return True


class FakeProcess:
    """Pollable driver process with a scripted status sequence."""

    def __init__(self, returncode: int, *, running_polls: int = 0) -> None:
        """Store the exit code and number of pre-exit polls."""

        self.returncode = returncode
        self.running_polls = running_polls

    def poll(self) -> Optional[int]:
        """Return ``None`` for the scripted running polls, then the status."""

        if self.running_polls:
            self.running_polls -= 1
            return None
        return self.returncode

    def wait(self) -> int:
        """Return the scripted final status."""

        return self.returncode


class FakeProcessFactory:
    """Create fake processes from an ordered exit-code script."""

    def __init__(self, codes: Sequence[int], *, running_polls: int = 0) -> None:
        """Store process exit codes and optional initial running polls."""

        self.codes = list(codes)
        self.running_polls = running_polls
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: Sequence[str],
        cwd: Path,
        stdout: BinaryIO,
        stderr: BinaryIO,
    ) -> FakeProcess:
        """Return the next fake process and record its exact argv."""

        del cwd, stdout, stderr
        self.commands.append(tuple(command))
        return FakeProcess(self.codes.pop(0), running_polls=self.running_polls)


def _config(
    tmp_path: Path,
    *,
    queue_root: Optional[Path] = None,
    wake_episode_id: Optional[str] = None,
) -> SupervisorConfig:
    """Build a test supervisor configuration."""

    campaign_config = tmp_path / "campaign.json"
    campaign_config.write_text("{}\n", encoding="utf-8")
    return SupervisorConfig(
        campaign_id="c1-mech",
        repo_root=tmp_path.resolve(),
        campaign_config_path=campaign_config.resolve(),
        author_queue_root=queue_root.resolve() if queue_root is not None else None,
        runtime_root=(tmp_path / ".crawl-local" / "supervisor" / "c1-mech").resolve(),
        wake_episode_id=wake_episode_id,
        poll_seconds=1.0,
        notification_retry_seconds=1.0,
    )


@pytest.mark.parametrize(
    ("driver_exit", "message_fragment"),
    [
        (EXIT_PAUSED, "usage"),
        (EXIT_REVIEW_PAUSED, "review checkpoint"),
        (EXIT_OPERATOR_OUTAGE, "RETRYABLE INFRASTRUCTURE"),
    ],
)
def test_supervisor_handles_meaningful_stop_codes_without_restart(
    tmp_path: Path,
    driver_exit: int,
    message_fragment: str,
) -> None:
    """Usage, review, and operator stops each alert once and halt cleanly."""

    notifier = FakeNotifier()
    factory = FakeProcessFactory([driver_exit])
    supervisor = CrawlerSupervisor(
        _config(tmp_path),
        notifier,
        process_factory=factory,
        now=lambda: NOW,
        sleep=lambda _seconds: None,
    )

    assert supervisor.run() == 0
    assert len(factory.commands) == 1
    assert message_fragment in notifier.messages[0][0]


def test_supervisor_backs_off_lock_contention_then_restarts(tmp_path: Path) -> None:
    """Exit 3 is ordinary driver ownership contention, not a crash."""

    notifier = FakeNotifier()
    factory = FakeProcessFactory([EXIT_LOCKED, EXIT_PAUSED])
    sleeps: list[float] = []
    supervisor = CrawlerSupervisor(
        _config(tmp_path),
        notifier,
        process_factory=factory,
        now=lambda: NOW,
        sleep=sleeps.append,
    )

    assert supervisor.run() == 0
    assert len(factory.commands) == 2
    assert sleeps == [30.0]


def test_supervisor_crash_circuit_is_persisted_and_alerted(tmp_path: Path) -> None:
    """Both crash circuits page, persist, and refuse to exit successfully.

    BL-6 rewrote this contract. The old assertions -- exit 0 after an open circuit,
    and exactly one alert for five crashes -- encoded the two defects round 1
    measured: a 30-minute windowed counter that at 20k-model leg durations provably
    never armed (AREA_SCHED_opus SCHED-04), and an ordinary crash that notified
    nothing at all (AREA_OPS_opus OPS-06). The circuit now counts CONSECUTIVE
    restarts that made no durable progress, an ordinary crash pages its class
    immediately, and an open circuit exits NON-ZERO so the OS layer sees it.

    Parameters
    ----------
    tmp_path:
        Test-owned root.
    """

    notifier = FakeNotifier(failures=1)
    factory = FakeProcessFactory([EXIT_ERROR] * (CONSECUTIVE_FLAT_RESTART_LIMIT + 1))
    sleeps: list[float] = []
    supervisor = CrawlerSupervisor(
        _config(tmp_path),
        notifier,
        process_factory=factory,
        now=lambda: NOW,
        sleep=sleeps.append,
    )

    assert supervisor.run() == EXIT_ERROR
    # The first crash only establishes the progress baseline the later ones are
    # measured against, so the circuit opens one restart after the limit.
    assert len(factory.commands) == CONSECUTIVE_FLAT_RESTART_LIMIT + 1
    assert "circuit opened" in notifier.messages[-1][0]
    # One retried crash-class page (the first attempt returns no receipt) plus the
    # circuit alert; the three identical later crashes are deduplicated by class.
    assert len(notifier.messages) == 3
    state = json.loads(
        (tmp_path / ".crawl-local" / "supervisor" / "c1-mech" / "state.json").read_text()
    )
    assert len(state["crash_timestamps"]) == CONSECUTIVE_FLAT_RESTART_LIMIT + 1
    assert state["consecutive_flat_restarts"] == CONSECUTIVE_FLAT_RESTART_LIMIT

    # The historical burst circuit is retained as the second tripwire: it still arms
    # when every crashed leg DID move durable progress, so the flat counter stays 0.
    burst_root = tmp_path / "burst"
    (burst_root / ".crawl-local").mkdir(parents=True)
    burst_notifier = FakeNotifier()
    burst = CrawlerSupervisor(
        _config(burst_root),
        burst_notifier,
        process_factory=FakeProcessFactory([EXIT_ERROR] * CRASH_LIMIT),
        now=lambda: NOW,
        sleep=lambda _seconds: None,
    )
    driver_state = burst_root / ".crawl-local" / "driver-state.json"
    terminals = {"value": 0}
    measured = burst._progress_fingerprint

    def _moving() -> str:
        """Record one more terminal, then fingerprint the real durable witnesses."""

        terminals["value"] += 1
        driver_state.write_text(
            json.dumps({"status": "running", "last_terminal_count": terminals["value"]}),
            encoding="utf-8",
        )
        return measured()

    burst._progress_fingerprint = _moving  # type: ignore[method-assign]
    assert burst.run() == EXIT_ERROR
    assert "in 30 minutes" in burst_notifier.messages[-1][0]


def test_crash_circuit_fires_on_count_at_20k_scale_leg_durations(tmp_path: Path) -> None:
    """BL-8 (c) tripwire: the crash circuit is COUNT-based, proven at 20k scale.

    At 20k-model leg durations a restart-to-crash cycle is HOURS (step-100's
    four crashes spanned ~55h), so any 30-minute windowed counter holds at most
    one crash at a time and provably never arms: a deterministic mid-leg crash
    was an unbounded restart loop at full spend with zero messages. This test
    spaces every crash 14 hours apart and asserts the flat-restart circuit
    still opens on the COUNT of consecutive no-progress restarts, while the
    persisted burst window demonstrably never held more than one crash.
    """

    clock = {"now": NOW}

    def _now() -> datetime:
        return clock["now"]

    class AdvancingFactory(FakeProcessFactory):
        """Advance the clock one 20k-scale leg duration per driver restart."""

        def __call__(self, command, cwd, stdout, stderr):  # noqa: ANN001, ANN202
            clock["now"] = clock["now"] + timedelta(hours=14)
            return super().__call__(command, cwd, stdout, stderr)

    factory = AdvancingFactory([EXIT_ERROR] * (CONSECUTIVE_FLAT_RESTART_LIMIT + 1))
    notifier = FakeNotifier()
    supervisor = CrawlerSupervisor(
        _config(tmp_path),
        notifier,
        process_factory=factory,
        now=_now,
        sleep=lambda _seconds: None,
    )

    assert supervisor.run() == EXIT_ERROR
    # The count circuit opened: baseline crash + LIMIT flat restarts, then refusal.
    assert len(factory.commands) == CONSECUTIVE_FLAT_RESTART_LIMIT + 1
    assert "circuit opened" in notifier.messages[-1][0]
    assert "consecutive restarts" in notifier.messages[-1][0]
    state = json.loads(
        (tmp_path / ".crawl-local" / "supervisor" / "c1-mech" / "state.json").read_text()
    )
    assert state["consecutive_flat_restarts"] == CONSECUTIVE_FLAT_RESTART_LIMIT
    # The wall-clock projection provably never armed: at 14h spacing the
    # persisted 30-minute window is pruned to exactly the newest crash, so a
    # windowed limit of any size >1 can never fire at this cadence.
    assert len(state["crash_timestamps"]) == 1
    assert CRASH_LIMIT > 1


def test_queue_watchdog_reports_stall_as_retryable_infrastructure(tmp_path: Path) -> None:
    """A >45-minute pending job is surfaced without creating a model result."""

    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "claimed").mkdir()
    job = {
        "job_id": "author-job-1",
        "attempt_nonce": "attempt-1",
        "stable_id": "m_stalled",
        "enqueued_at": (NOW - timedelta(minutes=46)).isoformat().replace("+00:00", "Z"),
    }
    (queue / "pending" / "author-job-1.json").write_text(json.dumps(job), encoding="utf-8")
    (queue / "claimed" / "author-job-1.json").write_text("{}\n", encoding="utf-8")

    assert stalled_queue_jobs(queue, now=NOW) == (
        QueueStall("author-job-1", "attempt-1", "m_stalled", 46 * 60.0, True),
    )
    notifier = FakeNotifier()
    supervisor = CrawlerSupervisor(
        _config(tmp_path, queue_root=queue),
        notifier,
        process_factory=FakeProcessFactory([EXIT_OPERATOR_OUTAGE], running_polls=1),
        now=lambda: NOW,
        sleep=lambda _seconds: None,
    )

    assert supervisor.run() == 0
    assert "author queue stalled >45 min" in notifier.messages[0][0]
    assert "RETRYABLE INFRASTRUCTURE" in notifier.messages[0][0]
    assert not (tmp_path / "menagerie" / "crawler" / "records" / "models").exists()


def test_global_execution_flock_serializes_separate_campaign_clones(tmp_path: Path) -> None:
    """Two processes with different campaign roots cannot own the global slot together."""

    lock_path = tmp_path / "host" / "execution.flock"
    holder_ready = tmp_path / "holder-ready"
    release_holder = tmp_path / "release-holder"
    contender_ready = tmp_path / "contender-ready"
    program = (
        "import os,sys,time; from pathlib import Path; "
        "from menagerie.crawler.execution_lock import "
        "acquire_global_execution_flock,release_global_execution_flock; "
        "fd=acquire_global_execution_flock(Path(sys.argv[1])); "
        "Path(sys.argv[2]).write_text('acquired'); "
        "\nwhile not Path(sys.argv[3]).exists(): time.sleep(0.01)\n"
        "release_global_execution_flock(fd)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", program, str(lock_path), str(holder_ready), str(release_holder)]
    )
    contender_release = tmp_path / "release-contender"
    contender: Optional[subprocess.Popen[bytes]] = None
    try:
        deadline = datetime.now().timestamp() + 5
        while not holder_ready.exists() and datetime.now().timestamp() < deadline:
            time.sleep(0.01)
        assert holder_ready.exists()
        contender = subprocess.Popen(
            [
                sys.executable,
                "-c",
                program,
                str(lock_path),
                str(contender_ready),
                str(contender_release),
            ]
        )
        time.sleep(0.2)
        assert not contender_ready.exists()
        release_holder.write_text("release", encoding="utf-8")
        deadline = datetime.now().timestamp() + 5
        while not contender_ready.exists() and datetime.now().timestamp() < deadline:
            time.sleep(0.01)
        assert contender_ready.exists()
    finally:
        release_holder.touch()
        contender_release.touch()
        holder.wait(timeout=5)
        if contender is not None:
            contender.wait(timeout=5)


def test_global_execution_flock_default_is_outside_campaign_runtime(
    tmp_path: Path,
) -> None:
    """The production default cannot accidentally resolve inside a clone."""

    from menagerie.crawler.execution_lock import global_execution_flock_path

    assert not global_execution_flock_path().is_relative_to(tmp_path / ".crawl-local")


def test_production_worker_spawn_inherits_the_global_flock() -> None:
    """The only model-worker spawn path carries the global descriptor into its child."""

    lane_source = inspect.getsource(SupervisedForwardLane.forward)
    spawn_source = inspect.getsource(worker_supervisor.run_isolated_subprocess)

    assert "global_lock_path=self.execution_flock_path" in lane_source
    assert "worker_lease_handle.global_lock_fd" in spawn_source
    assert "pass_fds=inherited_fds" in spawn_source


def test_command_notifier_requires_the_operator_shim_nonce_receipt(tmp_path: Path) -> None:
    """Exit zero without a receipt is failure; the shipped shim makes it success.

    The success arm uses ``/bin/echo``, not ``/usr/bin/true``: the no-op mute
    is now a denylisted transport the shim REFUSES to execute (OB-M3), so a
    receipt can only come from a transport that actually ran the message.
    """

    receipt_root = tmp_path / "receipts"
    silent = CommandNotifier(
        f"{sys.executable} -c pass",
        receipt_root=receipt_root,
    )
    assert silent.notify("silent", idempotency_key="nonce-silent") is False

    shim = CommandNotifier(
        (f"{sys.executable} -m menagerie.crawler.operator_notify --transport /bin/echo"),
        receipt_root=receipt_root,
    )
    assert shim.notify("visible", idempotency_key="nonce-visible") is True

    muted = CommandNotifier(
        (f"{sys.executable} -m menagerie.crawler.operator_notify --transport /usr/bin/true"),
        receipt_root=receipt_root,
    )
    assert muted.notify("visible", idempotency_key="nonce-muted") is False


def test_launchd_plist_keeps_only_unexpected_supervisor_death_alive(tmp_path: Path) -> None:
    """The rendered agent does not relaunch handled pause/complete exits."""

    payload = plistlib.loads(
        render_launchd_plist(
            campaign_id="c1-mech",
            repo_root=tmp_path,
            campaign_config_path=tmp_path / "campaign.json",
            author_queue_root=tmp_path / "queue",
            python_executable=Path(sys.executable),
            author_concurrency=6,
        )
    )

    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["ProgramArguments"][0].endswith("menagerie/crawler/tools/crawler_supervisor.sh")
    # The author fan-out is carried explicitly: it is not part of the frozen campaign
    # config, so a supervised restart that dropped it would silently retune the
    # campaign's throughput back to the default.
    assert payload["ProgramArguments"][-4:] == [
        "--author-queue",
        str(tmp_path / "queue"),
        "--author-concurrency",
        "6",
    ]


def test_checkpoint_policy_requires_only_c1_gate_and_advance_notifications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1 keeps 1000+900/950/1000 while C2-C4 require a zero checkpoint."""

    from menagerie.crawler.partitioner import CampaignBinding, CAMPAIGN_SPECS
    import menagerie.crawler.supervisor as supervisor_module

    snapshot = object()
    monkeypatch.setattr(supervisor_module, "load_intake_snapshot", lambda _path: snapshot)

    def binding(campaign_id: str) -> CampaignBinding:
        """Return a minimal manifest binding for one frozen campaign."""

        spec = next(item for item in CAMPAIGN_SPECS if item.campaign_id == campaign_id)
        return CampaignBinding(spec, 1, "p", "sha256:x", "i", "intake-x", "sha256:y", 1)

    base: dict[str, Any] = dict(
        repo_root=tmp_path,
        intake_root=tmp_path / "intake",
        target="osx-arm64",
        run_id="run",
        author_queue_root=tmp_path / "queue",
        author_command=(sys.executable,),
        checker_command=(sys.executable,),
        environment_command=(sys.executable,),
        notify_command=(sys.executable,),
        public_mirror=tmp_path / "public",
        private_mirror=tmp_path / "private",
        phase="pytorch",
        only_status=None,
        author_wall_seconds=None,
    )
    c1 = CampaignConfig(
        **base,
        review_checkpoint_at=1000,
        progress_milestones=(900, 950, 1000),
    )
    monkeypatch.setattr(
        supervisor_module, "find_campaign_binding", lambda _path, _snapshot: binding("c1-mech")
    )
    validate_supervised_campaign(tmp_path, "c1-mech", c1)
    with pytest.raises(SupervisorError, match="notify"):
        validate_supervised_campaign(
            tmp_path,
            "c1-mech",
            CampaignConfig(
                **base,
                review_checkpoint_at=1000,
                progress_milestones=(900, 950),
            ),
        )

    monkeypatch.setattr(
        supervisor_module, "find_campaign_binding", lambda _path, _snapshot: binding("c2-disco")
    )
    with pytest.raises(SupervisorError, match="must be 0"):
        validate_supervised_campaign(tmp_path, "c2-disco", c1)


# ---------------------------------------------------------------------------
# W6: the bounded, attested restart ladder.
# ---------------------------------------------------------------------------


def _ladder_supervisor(tmp_path: Path, **config_kwargs: object) -> CrawlerSupervisor:
    """Return one supervisor for direct ladder adjudication."""

    return CrawlerSupervisor(
        _config(tmp_path, **config_kwargs),  # type: ignore[arg-type]
        FakeNotifier(),
        process_factory=FakeProcessFactory([]),
        now=lambda: NOW,
        sleep=lambda _seconds: None,
    )


@pytest.mark.smoke
def test_restart_ladder_refuses_a_non_restartable_exit(tmp_path: Path) -> None:
    """A lawful end (close/pause/operator stop) owns its own continuation."""

    from menagerie.crawler.leg_generation import record_leg_ended, record_leg_started
    from menagerie.crawler.leg_exit_vocabulary import cause_fingerprint

    supervisor = _ladder_supervisor(tmp_path)
    records_root = tmp_path / "menagerie" / "crawler" / "records"
    started = record_leg_started(
        records_root,
        campaign_id="c1-mech",
        generation_id="sha256:" + "d" * 64,
        run_id="run-1",
        created_at="2026-08-24T00:00:00Z",
        invocation_origin="ordinary-run",
        invocation_ordinal=0,
        pid=1,
    )
    record_leg_ended(
        records_root,
        campaign_id="c1-mech",
        generation_id="sha256:" + "d" * 64,
        run_id="run-1",
        created_at="2026-08-24T00:01:00Z",
        leg_started_sha256=str(started["row_sha256"]),
        exit_class="paused_typed",
        cause_fingerprint=cause_fingerprint("paused_typed"),
        terminal_delta=0,
        invocation_origin="ordinary-run",
    )
    supervisor._leg_events_baseline = 0  # noqa: SLF001
    refusal = supervisor._restart_ladder_refusal(EXIT_ERROR)  # noqa: SLF001
    assert refusal is not None and "not restartable" in refusal


@pytest.mark.smoke
def test_restart_ladder_bounds_the_ordinal_and_honors_holds(tmp_path: Path) -> None:
    """Exhausted ordinal and a standing hold sentinel both refuse typed."""

    import json as _json

    import menagerie.crawler.supervisor as supervisor_module

    supervisor = _ladder_supervisor(tmp_path)
    supervisor._restart_ordinal = supervisor_module.SUPERVISED_RESTART_ORDINAL_LIMIT  # noqa: SLF001
    refusal = supervisor._restart_ladder_refusal(EXIT_ERROR)  # noqa: SLF001
    assert refusal is not None and "exhausted" in refusal

    held = _ladder_supervisor(tmp_path)
    control = tmp_path / "menagerie" / "crawler" / "records" / "control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "sprint-pause.json").write_text(
        _json.dumps(
            {
                "format": "menagerie.crawler.sprint-pause.v1",
                "reason": "orchestrator hold",
                "paused_by": "orchestrator",
                "created_at": "2026-08-24T00:00:00Z",
                "expires_at": None,
                "default_action": "stay-paused",
            }
        ),
        encoding="utf-8",
    )
    refusal = held._restart_ladder_refusal(EXIT_ERROR)  # noqa: SLF001
    assert refusal is not None and "hold_sentinel_present" in refusal


@pytest.mark.smoke
def test_supervised_restart_drops_the_consumed_wake_episode(tmp_path: Path) -> None:
    """SINGLE continuation authority: the episode continues a generation once.

    The first invocation of a wake-launched supervision episode carries the
    episode id; every ladder restart drops it and attests its ordinal
    instead, so the wake episode and the restart ladder can never both
    continue the same generation (the dual-callback race). The DX-23
    invocation ledger will bind these ordinals when its lane merges in the
    same batch.
    """

    import menagerie.crawler.supervisor as supervisor_module

    config = _config(tmp_path, wake_episode_id="episode-1")
    first = supervisor_module._driver_command(config)  # noqa: SLF001
    assert "--wake-episode-id" in first
    assert "--supervised-restart-ordinal" not in first

    restart = supervisor_module._driver_command(config, restart_ordinal=2)  # noqa: SLF001
    assert "--wake-episode-id" not in restart
    ordinal_index = restart.index("--supervised-restart-ordinal")
    assert restart[ordinal_index + 1] == "2"
