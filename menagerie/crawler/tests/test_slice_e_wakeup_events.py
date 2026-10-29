"""Round-14 recurring wake-episode and operational-event tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.authority import WakeEpisode
from menagerie.crawler.checkpoint import (
    FunnelSnapshot,
    build_checkpoint_review_event,
    build_progress_notification_event,
    build_review_signoff_event,
)
from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION, OperationalEventKind
from menagerie.crawler.recordio import JsonlLedger, LedgerConflictError, scan_jsonl
from menagerie.crawler.schema import PayloadValidationError, validate_payload
from menagerie.crawler.wakeup import (
    DEFAULT_RETRY_INTERVAL_SECONDS,
    OperationalContext,
    RenderedWakeupDefinition,
    WakeupBackend,
    WakeupConfigurationError,
    WakeupManager,
    _WAKE_EVENT_TRANSITIONS,
    _WakeEventTransition,
    build_wake_episode,
    build_usage_pause_event,
    detect_wakeup_backend,
    reduce_wake_episodes,
    render_wakeup_definition,
    render_wakeup_status,
    write_fire_intent,
)

NOW = "2026-07-14T12:00:00Z"
RESET = "2026-07-14T13:00:00Z"


class FakeScheduler:
    """In-memory verified scheduler backend."""

    def __init__(self) -> None:
        """Initialize empty registrations and operation history."""

        self.registrations: dict[str, str] = {}
        self.installed: list[str] = []
        self.deactivated: list[str] = []
        self.fail_install = False

    def install(self, definition: RenderedWakeupDefinition) -> None:
        """Install the exact rendered definition or raise a fake failure."""

        if self.fail_install:
            raise RuntimeError("synthetic scheduler failure")
        episode_id = definition.episode.episode_id
        self.registrations[episode_id] = definition.scheduler_definition
        self.installed.append(episode_id)

    def verify(self, definition: RenderedWakeupDefinition) -> bool:
        """Return whether the exact rendered bytes are registered."""

        return (
            self.registrations.get(definition.episode.episode_id) == definition.scheduler_definition
        )

    def deactivate(self, definition: RenderedWakeupDefinition) -> None:
        """Remove one registration idempotently."""

        episode_id = definition.episode.episode_id
        self.registrations.pop(episode_id, None)
        self.deactivated.append(episode_id)


def _context() -> OperationalContext:
    """Return shared synthetic operational context.

    Returns
    -------
    OperationalContext
        Valid event context.
    """

    return OperationalContext("run-test", "machine-test", {"author": 2}, "env-test")


def _manager(
    root: Path,
    ledger: JsonlLedger,
    scheduler: FakeScheduler,
    *,
    backend: WakeupBackend = WakeupBackend.LAUNCHD,
    health_threshold_fires: int = 96,
) -> WakeupManager:
    """Return a recurring manager bound to the fake scheduler."""

    return WakeupManager(
        root,
        ledger,
        ["python", "-m", "menagerie.crawler", "run", "--resume"],
        backend=backend,
        installer=scheduler.install,
        verifier=scheduler.verify,
        deactivator=scheduler.deactivate,
        health_threshold_fires=health_threshold_fires,
    )


def test_frozen_episode_and_recurring_renderers_have_backend_parity(tmp_path: Path) -> None:
    """Every backend renders recurrence and delegates reset guarding to the callback."""

    episode = build_wake_episode(
        provider="anthropic",
        reset_at=RESET,
        reset_observation="observed",
        callback_argv=["python", "-m", "menagerie.crawler", "run", "--resume"],
    )
    assert isinstance(episode, WakeEpisode)
    assert episode.retry_interval_seconds == DEFAULT_RETRY_INTERVAL_SECONDS
    assert episode.not_before == RESET
    assert episode.callback_argv[-2:] == ("--wake-episode-id", episode.episode_id)

    launchd = render_wakeup_definition(tmp_path, episode, WakeupBackend.LAUNCHD)
    systemd = render_wakeup_definition(tmp_path, episode, WakeupBackend.SYSTEMD_TIMER)
    cron = render_wakeup_definition(tmp_path, episode, WakeupBackend.CRON)
    assert "<key>StartInterval</key><integer>900</integer>" in launchd.scheduler_definition
    assert "StartCalendarInterval" not in launchd.scheduler_definition
    assert "OnUnitInactiveSec=900s" in systemd.scheduler_definition
    assert "Persistent=true" in systemd.scheduler_definition
    assert "OnCalendar=" not in systemd.scheduler_definition
    assert "*/15 * * * *" in cron.scheduler_definition
    assert "one-shot" not in cron.scheduler_definition
    for definition in (launchd, systemd, cron):
        assert episode.episode_id in definition.scheduler_definition or (
            definition.service_definition is not None
            and episode.episode_id in definition.service_definition
        )


@pytest.mark.parametrize("retry", [299, 901, 301, 420])
def test_episode_rejects_out_of_contract_or_nonminute_cadence(retry: int) -> None:
    """The shared callback cadence remains representable and bounded."""

    with pytest.raises(WakeupConfigurationError):
        build_wake_episode(
            provider="openai",
            reset_at=RESET,
            reset_observation="guessed",
            callback_argv=["python", "resume.py"],
            retry_interval_seconds=retry,
        )


def test_pause_install_replay_and_missing_definition_repair_are_visible(tmp_path: Path) -> None:
    """An active episode self-heals a deleted projection without reopening authority."""

    ledger_path = tmp_path / "operational.jsonl"
    scheduler = FakeScheduler()
    with JsonlLedger(ledger_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / "wakeups", ledger, scheduler)
        first = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="usage limit reached",
            reset_at=RESET,
            context=_context(),
            created_at=NOW,
        )
        replay = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="usage limit reached",
            reset_at=RESET,
            context=_context(),
            created_at="2026-07-14T12:01:00Z",
        )
        assert first.created and first.verified
        assert not replay.created and not replay.repaired
        assert scheduler.installed == [first.episode.episode_id]

        first.definition.definition_path.unlink()
        repaired = manager.reconcile(context=_context(), created_at="2026-07-14T12:05:00Z")
        assert repaired.repaired_episode_ids == (first.episode.episode_id,)
        assert scheduler.installed == [first.episode.episode_id, first.episode.episode_id]

    events = scan_jsonl(ledger_path)
    assert [event["event_kind"] for event in events] == [
        "usage-pause",
        "wakeup-installed",
        "wakeup-repaired",
    ]
    assert all(event["details"].get("recurring") is True for event in events[1:])
    for event in events:
        validate_payload(event)


def test_install_verification_failure_is_visible_and_cannot_pause_silently(
    tmp_path: Path,
) -> None:
    """A failed scheduler install records failed:runner and raises."""

    ledger_path = tmp_path / "operational.jsonl"
    scheduler = FakeScheduler()
    scheduler.fail_install = True
    with JsonlLedger(ledger_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / "wakeups", ledger, scheduler)
        with pytest.raises(WakeupConfigurationError, match="cannot install recurring"):
            manager.record_pause_and_schedule(
                provider="openai",
                observed_response="quota exhausted",
                reset_at=RESET,
                context=_context(),
                created_at=NOW,
            )
    events = scan_jsonl(ledger_path)
    assert [event["event_kind"] for event in events] == ["usage-pause", "wakeup-failed"]
    assert events[-1]["status"] == "failed:runner"


def test_guessed_reset_is_superseded_without_disarming_recurrence(tmp_path: Path) -> None:
    """Improved provider evidence replaces one active guessed episode."""

    scheduler = FakeScheduler()
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / "wakeups", ledger, scheduler)
        guessed = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="limit without reset",
            reset_at="2026-07-14T12:30:00Z",
            reset_observation="guessed",
            context=_context(),
            created_at=NOW,
        )
        observed = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="resets at 13:00 UTC",
            reset_at=RESET,
            reset_observation="observed",
            context=_context(),
            created_at="2026-07-14T12:10:00Z",
        )
        projection = manager.projection
        assert observed.episode.supersedes_episode_id == guessed.episode.episode_id
        assert projection.episodes[guessed.episode.episode_id].disposition == "superseded"
        assert projection.active_episode_ids == (observed.episode.episode_id,)
        assert scheduler.deactivated == [guessed.episode.episode_id]
        reconciled = manager.reconcile(context=_context(), created_at="2026-07-14T12:11:00Z")
        assert reconciled.deactivated_episode_ids == ()


@pytest.mark.parametrize(
    ("resolution", "event_kind", "disposition"),
    [
        ("manual-resume", "usage-resume", "resumed"),
        ("campaign-completed", "campaign-completed", "completed"),
        ("operator-cancelled", "operator-cancelled", "cancelled"),
    ],
)
def test_durable_resolution_precedes_self_deactivation(
    tmp_path: Path, resolution: str, event_kind: str, disposition: str
) -> None:
    """Manual resume, completion, and cancellation resolve before OS removal."""

    scheduler = FakeScheduler()
    ledger_path = tmp_path / f"{resolution}.jsonl"
    with JsonlLedger(ledger_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / resolution, ledger, scheduler)
        scheduled = manager.record_pause_and_schedule(
            provider="openai",
            observed_response="quota exhausted",
            reset_at=RESET,
            context=_context(),
            created_at=NOW,
        )
        assert manager.resolve_episode(
            scheduled.episode.episode_id,
            resolution=resolution,
            context=_context(),
            created_at="2026-07-14T12:20:00Z",
        )
        assert manager.projection.episodes[scheduled.episode.episode_id].disposition == disposition
    kinds = [event["event_kind"] for event in scan_jsonl(ledger_path)]
    assert kinds.index(event_kind) < kinds.index("wakeup-deactivated")


def test_fake_clock_fire_guard_bucket_idempotency_health_and_resume(tmp_path: Path) -> None:
    """Recurring fires stay bounded until a durable resume precedes deactivation."""

    scheduler = FakeScheduler()
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / "wakeups", ledger, scheduler, health_threshold_fires=2)
        scheduled = manager.record_pause_and_schedule(
            provider="openai",
            observed_response="quota exhausted",
            reset_at=RESET,
            context=_context(),
            created_at=NOW,
        )
        episode_id = scheduled.episode.episode_id
        first = manager.handle_fire(episode_id, fired_at="2026-07-14T12:15:00Z", context=_context())
        replay = manager.handle_fire(
            episode_id, fired_at="2026-07-14T12:16:00Z", context=_context()
        )
        second = manager.handle_fire(
            episode_id, fired_at="2026-07-14T12:30:00Z", context=_context()
        )
        resumed = manager.handle_fire(episode_id, fired_at=RESET, context=_context())
        assert first.guarded_not_before and first.appended
        assert replay.guarded_not_before and not replay.appended
        assert second.guarded_not_before and second.appended
        assert resumed.should_resume and resumed.deactivated
        assert scheduler.deactivated == [episode_id]
        state = manager.projection.episodes[episode_id]
        assert state.disposition == "resumed"
        assert state.health_degraded
        assert not state.active

    kinds = [event["event_kind"] for event in scan_jsonl(tmp_path / "operational.jsonl")]
    assert kinds.index("usage-resume") < kinds.index("wakeup-deactivated")
    assert kinds.count("wakeup-health-degraded") == 1


def test_lock_contention_intent_is_fsynced_bucket_idempotent_and_ingested(
    tmp_path: Path,
) -> None:
    """A live driver gets a durable no-op intent and recurrence stays active."""

    scheduler = FakeScheduler()
    ledger_path = tmp_path / "operational.jsonl"
    with JsonlLedger(ledger_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / "wakeups", ledger, scheduler)
        result = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="usage limit reached",
            reset_at=RESET,
            context=_context(),
            created_at=NOW,
        )
        first = write_fire_intent(
            tmp_path / "wakeups",
            result.episode,
            fired_at="2026-07-14T12:15:00Z",
        )
        replay = write_fire_intent(
            tmp_path / "wakeups",
            result.episode,
            fired_at="2026-07-14T12:16:00Z",
        )
        assert first == replay and first.is_file()
        assert (
            manager.ingest_fire_intents(context=_context(), created_at="2026-07-14T12:17:00Z") == 1
        )
        assert not first.exists()
        assert manager.projection.episodes[result.episode.episode_id].active
        assert result.episode.episode_id in scheduler.registrations

    kinds = [event["event_kind"] for event in scan_jsonl(ledger_path)]
    assert "wakeup-fired" in kinds
    assert "wake-noop-already-running" in kinds
    assert "usage-resume" not in kinds
    assert "wakeup-deactivated" not in kinds


def test_status_renderer_uses_same_operational_reducer(tmp_path: Path) -> None:
    """Status exposes active provider, reset evidence, recurrence, and health."""

    scheduler = FakeScheduler()
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / "wakeups", ledger, scheduler)
        scheduled = manager.record_pause_and_schedule(
            provider="openai",
            observed_response="quota exhausted",
            reset_at=RESET,
            reset_observation="guessed",
            context=_context(),
            created_at=NOW,
        )
        rendered = render_wakeup_status(reduce_wake_episodes(ledger.records))
    assert rendered["active_count"] == 1
    episode = rendered["episodes"][0]
    assert episode["episode_id"] == scheduled.episode.episode_id
    assert episode["provider"] == "openai"
    assert episode["reset_observation"] == "guessed"
    assert episode["retry_interval_seconds"] == 900
    assert episode["install_verified"] is True


def test_wakeup_event_transition_table_is_exhaustive_and_preserves_diagnostics() -> None:
    """Every closed event kind has a transition and install errors remain exact."""

    assert set(_WAKE_EVENT_TRANSITIONS) == set(OperationalEventKind)
    assert all(
        isinstance(value, _WakeEventTransition) for value in _WAKE_EVENT_TRANSITIONS.values()
    )
    episode = build_wake_episode(
        provider="openai",
        reset_at=RESET,
        reset_observation="observed",
        callback_argv=["python", "resume.py"],
    )
    opening = build_usage_pause_event(
        episode=episode,
        observed_response="quota exhausted",
        context=_context(),
        created_at=NOW,
    )
    invalid_install = {
        "event_id": "invalid-install",
        "event_kind": OperationalEventKind.WAKEUP_INSTALLED.value,
        "details": {
            "episode_id": episode.episode_id,
            "verified": True,
            "backend": "invalid-backend",
        },
    }
    with pytest.raises(
        WakeupConfigurationError,
        match="unknown wakeup backend in event invalid-install: 'invalid-backend'",
    ):
        reduce_wake_episodes([opening, invalid_install])


def test_conflicting_episode_open_replay_is_rejected(tmp_path: Path) -> None:
    """An immutable opening event cannot silently accept changed episode facts."""

    scheduler = FakeScheduler()
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / "wakeups", ledger, scheduler)
        manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="first exact response",
            reset_at=RESET,
            context=_context(),
            created_at=NOW,
        )
        conflicting_open = deepcopy(ledger.records[0])
        conflicting_open.pop("ledger_seq")
        conflicting_open.pop("payload_sha256")
        conflicting_open["details"]["episode"]["not_before"] = "2026-07-14T13:15:00Z"
        with pytest.raises(LedgerConflictError):
            ledger.append(conflicting_open)


def test_changed_usage_pause_excerpt_is_disclosed_without_reopening(tmp_path: Path) -> None:
    """A provider rewording appends an observation while retaining the opening.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    scheduler = FakeScheduler()
    path = tmp_path / "operational.jsonl"
    with JsonlLedger(path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / "wakeups", ledger, scheduler)
        first = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="first exact response",
            reset_at=RESET,
            context=_context(),
            created_at=NOW,
        )
        second = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="provider reworded the same pause",
            reset_at=RESET,
            context=_context(),
            created_at="2026-07-14T12:05:00Z",
        )
        projection = manager.projection

    events = scan_jsonl(path)
    openings = [event for event in events if event["event_kind"] == "usage-pause"]
    observations = [
        event for event in events if event["event_kind"] == "usage-pause-observation.v1"
    ]
    assert first.episode == second.episode
    assert len(openings) == 1
    assert openings[0]["observed_response"] == "first exact response"
    assert len(observations) == 1
    assert observations[0]["observed_response"] == "provider reworded the same pause"
    assert projection.active_episode_ids == (first.episode.episode_id,)


def test_dual_lane_pauses_coalesce_to_exactly_one_resume_at_max_reset(
    tmp_path: Path,
) -> None:
    """DX-67: a resume requires ALL paused lanes' resets; one resume at max.

    Author and checker pauses are per-provider episodes and supersession is
    provider-scoped, so two lanes hold two active episodes at once. The first
    lane's reset used to fire a resume straight into the second lane's
    still-throttled provider; now that fire stays GUARDED (disclosed
    ``coalesced_behind_reset``), the recurring schedule keeps firing, and the
    fire at max(reset) performs the one and only resume.
    """

    scheduler = FakeScheduler()
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _manager(tmp_path / "wakeups", ledger, scheduler)
        checker = manager.record_pause_and_schedule(
            provider="openai",
            observed_response="quota exhausted",
            reset_at=RESET,
            context=_context(),
            created_at=NOW,
        )
        author = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="usage limit until 15:00 UTC",
            reset_at="2026-07-14T15:00:00Z",
            context=_context(),
            created_at="2026-07-14T12:05:00Z",
        )
        # Both lanes are active at once: supersession is provider-scoped.
        assert set(manager.projection.active_episode_ids) == {
            checker.episode.episode_id,
            author.episode.episode_id,
        }
        # The checker lane's own reset has passed; the author lane is still
        # throttled. The fire is guarded, disclosed, and resumes NOTHING.
        early = manager.handle_fire(
            checker.episode.episode_id,
            fired_at="2026-07-14T13:30:00Z",
            context=_context(),
        )
        assert early.guarded_not_before is True
        assert early.should_resume is False
        assert manager.projection.episodes[checker.episode.episode_id].active
        fires = [
            event
            for event in scan_jsonl(tmp_path / "operational.jsonl")
            if event["event_kind"] == "wakeup-fired"
        ]
        assert fires[-1]["details"]["coalesced_behind_reset"] == "2026-07-14T15:00:00Z"
        assert fires[-1]["details"]["guarded_not_before"] is False
        # At max(reset) the fire resumes -- exactly once.
        resumed = manager.handle_fire(
            author.episode.episode_id,
            fired_at="2026-07-14T15:00:00Z",
            context=_context(),
        )
        assert resumed.should_resume is True
        assert resumed.deactivated is True
        resumes = [
            event
            for event in scan_jsonl(tmp_path / "operational.jsonl")
            if event["event_kind"] == "usage-resume"
        ]
        assert len(resumes) == 1


def test_pause_refuses_typed_without_a_boot_persistent_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W15: pause and boot-persistent resume are one transaction, or typed refusal.

    200a's wake plists were empty after the host reboot: the pause rows stood,
    the resume path did not, and every recovery was a human. With the
    requirement armed, a pause whose wake would not survive a reboot refuses
    LOUDLY (with a durable wakeup-failed row) instead of durably recording a
    reboot-mortal pause; arranging the byte-exact agent-root plist makes the
    same pause succeed and disclose ``boot_persistent.persistent`` true. The
    requirement DERIVES from whether the real OS scheduler operations are in
    play, so production managers (which pass no fakes) are always armed.
    """

    from menagerie.crawler.wakeup import (
        LAUNCHD_AGENT_ROOT_ENV,
        _write_boot_persistent_definition,
    )

    monkeypatch.setenv(LAUNCHD_AGENT_ROOT_ENV, str(tmp_path / "LaunchAgents"))
    scheduler = FakeScheduler()
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        # Derivation: injected fakes leave the requirement OFF by default (a
        # fake cannot testify about the OS agent root); the production
        # constructor path (no injected operations) arms it.
        assert _manager(tmp_path / "wakeups", ledger, scheduler).require_boot_persistence is False

        refusing = WakeupManager(
            tmp_path / "wakeups",
            ledger,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.LAUNCHD,
            installer=scheduler.install,
            verifier=scheduler.verify,
            deactivator=scheduler.deactivate,
            require_boot_persistence=True,
        )
        with pytest.raises(WakeupConfigurationError, match="not boot-persistent"):
            refusing.record_pause_and_schedule(
                provider="openai",
                observed_response="quota exhausted",
                reset_at=RESET,
                context=_context(),
                created_at=NOW,
            )
        failures = [
            event
            for event in scan_jsonl(tmp_path / "operational.jsonl")
            if event["event_kind"] == "wakeup-failed"
        ]
        assert failures and "not boot-persistent" in failures[-1]["details"]["reason"]
        # No installed event durably stood for the refused pause.
        assert not [
            event
            for event in scan_jsonl(tmp_path / "operational.jsonl")
            if event["event_kind"] == "wakeup-installed"
        ]

    # The arming half runs in an ISOLATED store: the refused attempt above
    # legitimately left a fake OS registration behind (started installs are
    # facts), and this half proves a persistence-writing install passes.
    persistent_scheduler = FakeScheduler()

    def persistent_install(definition) -> None:  # noqa: ANN001
        persistent_scheduler.install(definition)
        _write_boot_persistent_definition(definition)

    with JsonlLedger(
        tmp_path / "operational-armed.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION
    ) as armed_ledger:
        arming = WakeupManager(
            tmp_path / "wakeups-armed",
            armed_ledger,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.LAUNCHD,
            installer=persistent_install,
            verifier=persistent_scheduler.verify,
            deactivator=persistent_scheduler.deactivate,
            require_boot_persistence=True,
        )
        scheduled = arming.record_pause_and_schedule(
            provider="openai",
            observed_response="quota exhausted",
            reset_at=RESET,
            context=_context(),
            created_at="2026-07-14T12:01:00Z",
        )
        assert scheduled.scheduler_event.record["details"]["boot_persistent"][
            "persistent"
        ] is True


def test_wakeup_backend_feature_detection_prefers_ruled_platform_tools() -> None:
    """macOS uses launchd; other hosts prefer systemd and fall back to cron."""

    assert (
        detect_wakeup_backend(
            platform_name="Darwin", command_exists=lambda name: name == "launchctl"
        )
        is WakeupBackend.LAUNCHD
    )
    assert (
        detect_wakeup_backend(
            platform_name="Linux", command_exists=lambda name: name == "systemctl"
        )
        is WakeupBackend.SYSTEMD_TIMER
    )
    assert (
        detect_wakeup_backend(
            platform_name="FreeBSD", command_exists=lambda name: name == "crontab"
        )
        is WakeupBackend.CRON
    )


def test_review_and_progress_event_types_validate_strictly(tmp_path: Path) -> None:
    """All three additive event kinds validate and unknown kinds remain closed."""

    snapshot = FunnelSnapshot(runs=7, deferred=1, skipped=1, failed=1)
    logical_events = [
        build_checkpoint_review_event(
            models_completed=10,
            funnel_snapshot=snapshot,
            report_path="menagerie/crawler/views/review-10.json",
            context=_context(),
            created_at=NOW,
        ),
        build_review_signoff_event(
            approved_by_note="JMT approved review report 10",
            resume_after=10,
            context=_context(),
            created_at=NOW,
        ),
        build_progress_notification_event(
            models_completed=10,
            milestone=10,
            funnel_snapshot=snapshot,
            context=_context(),
            created_at=NOW,
        ),
    ]
    with JsonlLedger(tmp_path / "events.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        persisted = [ledger.append(event).record for event in logical_events]
    assert {event["event_kind"] for event in persisted} == {
        "checkpoint-review",
        "review-signoff",
        "progress-notification",
    }
    for event in persisted:
        validate_payload(event)
    unknown = deepcopy(persisted[0])
    unknown["event_kind"] = "unknown-event"
    with pytest.raises(PayloadValidationError):
        validate_payload(unknown)
