"""Durable recurring wake episodes for usage-limit pauses.

The append-only operational ledger is the authority for episode state. Scheduler
files, OS registrations, and fire-intent files are disposable projections that
this module verifies and repairs from that authority.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from menagerie.crawler.scheduler_ownership import SchedulerOwnershipRegistry

from menagerie.crawler.authority import WakeEpisode
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    SCHEDULED_RECHECK_PROVIDERS,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    fsync_directory,
    stable_hash,
)
from menagerie.crawler.models import AppendResult, JsonObject
from menagerie.crawler.operational_identity import (
    OPERATIONAL_EVENT_IDENTITY_REGISTRY,
    OperationalEventIdentityClass,
    idempotent_operational_event_id,
    mint_operational_event,
)
from menagerie.crawler.recordio import JsonlLedger

DEFAULT_RETRY_INTERVAL_SECONDS = 15 * 60
MIN_RETRY_INTERVAL_SECONDS = 5 * 60
MAX_RETRY_INTERVAL_SECONDS = 15 * 60
DEFAULT_HEALTH_THRESHOLD_FIRES = 96
RESET_OBSERVATIONS = frozenset({"observed", "guessed"})

#: THE name of the wake projection directory, derived exactly once.
#:
#: ``doctor`` and ``wake cancel`` used to spell it ``wakeups`` while the driver
#: installed into ``wakeup``, so the doctor's wake check reported broken exactly
#: when an episode was active -- i.e. while paused, when it matters -- and cancel
#: left the real projection installed while bootstrapping duplicates into a
#: directory nothing reads (AREA_SCHED_fable SCHED-06e, AREA_OPS_sol OPS-B08).
WAKEUP_PROJECTION_DIRNAME = "wakeup"

#: How long past ``max(reset_at, not_before)`` an active episode may go without a
#: recorded fire before it is OVERDUE. One recurrence interval plus a minute of
#: scheduler slop: the callback is meant to fire every ``retry_interval_seconds``,
#: so missing a full interval is already the failure, not noise.
WAKE_OVERDUE_GRACE_SECONDS = MAX_RETRY_INTERVAL_SECONDS + 60

#: What a report prints for a quantity it did not measure. BL-6 (f): reports
#: print ``unknown``, never ``0`` -- an unmeasured liveness window that renders
#: ``0`` is affirmatively reassuring about the thing it never looked at.
UNMEASURED = "unknown"

#: The exact flag a wake callback uses to declare the root its imports resolve
#: from. A closed rule, not a heuristic: the renderer reads THIS token or it
#: falls back to a derivation it can state, never to a guess.
CALLBACK_ROOT_FLAG = "--repo-root"

#: Operator-blessed wrapper that may schedule a power wake. ``pmset schedule``
#: needs root, so the crawler never invokes ``pmset`` directly: an operator points
#: this at a wrapper with a narrow sudoers line, or the power-wake path stays
#: honestly unarmed. Round 1 found ZERO occurrences of pmset/caffeinate in the
#: tree while the plan claimed two wake paths (AREA_AUTHOR_fable A-3, TB-03).
POWER_WAKE_COMMAND_ENV = "MENAGERIE_POWER_WAKE_COMMAND"

#: How far before the reset the power wake is scheduled, so the host is already
#: awake when the recurring callback's next interval lands.
POWER_WAKE_LEAD_SECONDS = 120

#: Where launchd looks for per-user jobs at login. Overridable ONLY so tests can
#: prove the boot-persistence rule without touching the operator's real agent
#: root; production always resolves the real one.
#:
#: BL-6 (b): the job used to be bootstrapped straight out of the gitignored
#: ``.crawl-local`` runtime directory, which launchd never scans at login, so a
#: reboot while paused left NOTHING scheduled and the campaign became a
#: permanent silent park. A definition the OS must re-read at boot has to live
#: where the OS looks for it (AREA_OPS_sol OPS-B10, TRACKB_sol TB-14).
LAUNCHD_AGENT_ROOT_ENV = "MENAGERIE_LAUNCHD_AGENT_ROOT"

#: The per-user launchd search path, relative to ``$HOME``. Kept as parts rather
#: than a joined literal so the projection-root census cannot mistake it for a
#: second derivation of the wake projection directory.
LAUNCHD_AGENT_ROOT_PARTS = ("Library", "LaunchAgents")

#: The argv that decides whether a ``systemd --user`` timer survives a reboot.
#: Without lingering, user units stop at logout and a rebooted, never-logged-in
#: host runs nothing -- so this is probed, never assumed.
LINGER_PROBE_ARGV = ("loginctl", "show-user", "--property=Linger")


class WakeupBackend(str, Enum):
    """Supported recurring OS scheduler mechanisms."""

    LAUNCHD = "launchd"
    SYSTEMD_TIMER = "systemd-timer"
    CRON = "cron"


class _WakeEventTransition(str, Enum):
    """Closed reducer transition applied by one operational event kind."""

    OPEN = "open"
    RESUME = "resume"
    RESOLVE_COMPLETED = "resolve-completed"
    RESOLVE_CANCELLED = "resolve-cancelled"
    INSTALL = "install"
    INSTALL_FAILED = "install-failed"
    FIRE = "fire"
    HEALTH_DEGRADED = "health-degraded"
    OBSERVE = "observe"


class _WakeResolution(str, Enum):
    """Closed operator resolution accepted by ``resolve_episode``."""

    MANUAL_RESUME = "manual-resume"
    CAMPAIGN_COMPLETED = "campaign-completed"
    OPERATOR_CANCELLED = "operator-cancelled"


_WAKE_EVENT_TRANSITIONS: dict[OperationalEventKind, _WakeEventTransition] = {
    OperationalEventKind.USAGE_PAUSE: _WakeEventTransition.OPEN,
    OperationalEventKind.USAGE_PAUSE_OBSERVATION: _WakeEventTransition.OBSERVE,
    OperationalEventKind.USAGE_RESUME: _WakeEventTransition.RESUME,
    OperationalEventKind.WAKEUP: _WakeEventTransition.OBSERVE,
    OperationalEventKind.CHECKPOINT: _WakeEventTransition.OBSERVE,
    OperationalEventKind.CAMPAIGN_HEALTH: _WakeEventTransition.OBSERVE,
    OperationalEventKind.WAKE_NOOP_ALREADY_RUNNING: _WakeEventTransition.FIRE,
    OperationalEventKind.CHECKPOINT_REVIEW: _WakeEventTransition.OBSERVE,
    OperationalEventKind.REVIEW_SIGNOFF: _WakeEventTransition.OBSERVE,
    OperationalEventKind.PROGRESS_NOTIFICATION: _WakeEventTransition.OBSERVE,
    OperationalEventKind.NOTIFICATION_DELIVERY: _WakeEventTransition.OBSERVE,
    OperationalEventKind.REQUEUE_GRANT_CONSUMED: _WakeEventTransition.OBSERVE,
    OperationalEventKind.WORKER_LEASE_OPENED: _WakeEventTransition.OBSERVE,
    OperationalEventKind.WORKER_LEASE_STARTED: _WakeEventTransition.OBSERVE,
    OperationalEventKind.WORKER_LEASE_CLOSED: _WakeEventTransition.OBSERVE,
    OperationalEventKind.WORKER_LEASE_REAPED: _WakeEventTransition.OBSERVE,
    OperationalEventKind.ORPHAN_WORKER_RECOVERED: _WakeEventTransition.OBSERVE,
    OperationalEventKind.WAKEUP_INSTALLED: _WakeEventTransition.INSTALL,
    OperationalEventKind.WAKEUP_REPAIRED: _WakeEventTransition.INSTALL,
    OperationalEventKind.WAKEUP_FIRED: _WakeEventTransition.FIRE,
    OperationalEventKind.WAKEUP_DEACTIVATED: _WakeEventTransition.OBSERVE,
    OperationalEventKind.WAKEUP_HEALTH_DEGRADED: _WakeEventTransition.HEALTH_DEGRADED,
    OperationalEventKind.WAKEUP_FAILED: _WakeEventTransition.INSTALL_FAILED,
    OperationalEventKind.CAMPAIGN_COMPLETED: _WakeEventTransition.RESOLVE_COMPLETED,
    OperationalEventKind.OPERATOR_CANCELLED: _WakeEventTransition.RESOLVE_CANCELLED,
    OperationalEventKind.WORKER_SHUTDOWN_INTERRUPTED: _WakeEventTransition.OBSERVE,
    # An unrecordable terminal is a durable per-model fact, never a wake transition:
    # it must not open, resume, fire or resolve a wake episode.
    OperationalEventKind.TERMINAL_UNRECORDABLE: _WakeEventTransition.OBSERVE,
    # A model-local lane failure is a durable per-model diagnostic, never a wake
    # transition, for the same reason an unrecordable terminal is not one.
    OperationalEventKind.MODEL_LANE_FAILED: _WakeEventTransition.OBSERVE,
    # Scheduler mode transitions are pure provenance: the same canonical
    # records drive either mode, so neither direction opens, fires, or
    # resolves a wake episode.
    OperationalEventKind.SCHEDULER_CUTOVER: _WakeEventTransition.OBSERVE,
    OperationalEventKind.SCHEDULER_ROLLBACK: _WakeEventTransition.OBSERVE,
    # WS-1 selection disclosures are pure per-leg provenance: a census, a
    # staleness report, or a generation discharge never opens, fires, or
    # resolves a wake episode.
    OperationalEventKind.WORK_SELECTION_CENSUS: _WakeEventTransition.OBSERVE,
    OperationalEventKind.PROJECTION_STALENESS: _WakeEventTransition.OBSERVE,
    OperationalEventKind.REQUEUE_GENERATION_DISCHARGED: _WakeEventTransition.OBSERVE,
    # B-09: the leg-start config attestation is pure provenance. A wake-callback
    # leg emits one exactly like an ordinary leg does, and it must not fire,
    # open, or resolve the episode that woke it.
    OperationalEventKind.LEG_CONFIG: _WakeEventTransition.OBSERVE,
    # W6/OB-M12: the leg-outcome mirror is pure provenance exactly like the
    # config attestation it pairs with.
    OperationalEventKind.LEG_OUTCOME: _WakeEventTransition.OBSERVE,
    # D-20: the leg-start close preview is a read-only diagnostic; it never
    # opens, fires, or resolves a wake episode.
    OperationalEventKind.CLOSE_PREVIEW: _WakeEventTransition.OBSERVE,
    # Overhaul RC-1: crawl-invocation facts are pure provenance. A typed
    # refusal, the paced-start census, and a limit raise never open, fire, or
    # resolve a wake episode.
    OperationalEventKind.INVOCATION_REFUSED: _WakeEventTransition.OBSERVE,
    OperationalEventKind.CONFIGURATION_CANARY: _WakeEventTransition.OBSERVE,
    OperationalEventKind.CRAWL_LIMIT: _WakeEventTransition.OBSERVE,
}
if set(_WAKE_EVENT_TRANSITIONS) != set(OperationalEventKind):
    raise RuntimeError("wakeup event transition table is not exhaustive")

_WAKE_RESOLUTION_TRANSITIONS: dict[
    _WakeResolution, tuple[OperationalEventKind, OperationalEventStatus]
] = {
    _WakeResolution.MANUAL_RESUME: (
        OperationalEventKind.USAGE_RESUME,
        OperationalEventStatus.USAGE_RESUMED,
    ),
    _WakeResolution.CAMPAIGN_COMPLETED: (
        OperationalEventKind.CAMPAIGN_COMPLETED,
        OperationalEventStatus.CAMPAIGN_COMPLETED,
    ),
    _WakeResolution.OPERATOR_CANCELLED: (
        OperationalEventKind.OPERATOR_CANCELLED,
        OperationalEventStatus.OPERATOR_CANCELLED,
    ),
}
if set(_WAKE_RESOLUTION_TRANSITIONS) != set(_WakeResolution):
    raise RuntimeError("wakeup resolution transition table is not exhaustive")


class WakeupError(RuntimeError):
    """Base class for wake episode failures."""


class WakeupConfigurationError(WakeupError):
    """Raised when a recurring wake episode cannot be represented or verified."""


@dataclass(frozen=True)
class OperationalContext:
    """Common immutable fields carried by every operational event."""

    run_id: str
    machine_id: str
    queued_work_counts: Mapping[str, int]
    current_environment: Optional[str]

    def __post_init__(self) -> None:
        """Validate common event context.

        Raises
        ------
        ValueError
            If an identity or queue count is invalid.
        """

        if not self.run_id.strip() or not self.machine_id.strip():
            raise ValueError("run_id and machine_id must be non-empty")
        if any(not key.strip() or value < 0 for key, value in self.queued_work_counts.items()):
            raise ValueError("queued work counts require non-empty keys and non-negative values")


@dataclass(frozen=True)
class RenderedWakeupDefinition:
    """Disposable backend projection of one frozen wake episode."""

    episode: WakeEpisode
    backend: WakeupBackend
    definition_path: Path
    scheduler_definition: str
    service_path: Optional[Path]
    service_definition: Optional[str]


@dataclass(frozen=True)
class WakeEpisodeState:
    """Reducer-derived operational state for one frozen wake episode."""

    episode: WakeEpisode
    disposition: str
    terminal_event_id: Optional[str]
    fire_buckets: tuple[int, ...]
    install_verified: bool
    health_degraded: bool
    last_event_id: str
    backend: Optional[WakeupBackend]

    @property
    def active(self) -> bool:
        """Return whether recurrence must remain installed.

        Returns
        -------
        bool
            True until a durable resolving or superseding fact exists.
        """

        return self.disposition == "active"


@dataclass(frozen=True)
class WakeEpisodeProjection:
    """Pure operational-ledger projection of every wake episode."""

    episodes: Mapping[str, WakeEpisodeState]
    active_episode_ids: tuple[str, ...]

    @property
    def active_episodes(self) -> tuple[WakeEpisodeState, ...]:
        """Return active episode states in deterministic opening order.

        Returns
        -------
        tuple[WakeEpisodeState, ...]
            Active reducer-derived states.
        """

        return tuple(self.episodes[episode_id] for episode_id in self.active_episode_ids)


@dataclass(frozen=True)
class WakeupScheduleResult:
    """Result of opening and verifying one recurring wake episode."""

    episode: WakeEpisode
    definition: RenderedWakeupDefinition
    created: bool
    repaired: bool
    verified: bool
    pause_event: AppendResult
    scheduler_event: AppendResult


@dataclass(frozen=True)
class WakeupCallbackResult:
    """Result of one callback fire handled under the driver lock."""

    episode_id: str
    time_bucket: int
    appended: bool
    guarded_not_before: bool
    should_resume: bool
    deactivated: bool


@dataclass(frozen=True)
class WakeupReconciliationResult:
    """Summary of scheduler repair/deactivation work."""

    active_episode_ids: tuple[str, ...]
    installed_episode_ids: tuple[str, ...]
    repaired_episode_ids: tuple[str, ...]
    deactivated_episode_ids: tuple[str, ...]
    failures: Mapping[str, str]


@dataclass(frozen=True)
class WakeupInspection:
    """Read-only comparison of active authority and scheduler projections."""

    backend: WakeupBackend
    active_episode_ids: tuple[str, ...]
    missing_or_mismatched_episode_ids: tuple[str, ...]
    unverified_episode_ids: tuple[str, ...]
    #: Active episodes the OS accepts NOW but would forget on reboot. BL-6 (b):
    #: a live projection and a projection that survives a restart are different
    #: facts, and only the second one makes a multi-week pause resumable.
    not_boot_persistent_episode_ids: tuple[str, ...] = ()

    @property
    def healthy(self) -> bool:
        """Return whether every active episode has an exact live projection.

        Returns
        -------
        bool
            True when no active projection needs repair and every one of them
            survives a reboot.
        """

        return (
            not self.missing_or_mismatched_episode_ids
            and not self.unverified_episode_ids
            and not self.not_boot_persistent_episode_ids
        )


CommandExists = Callable[[str], bool]
DefinitionAction = Callable[[RenderedWakeupDefinition], None]
DefinitionVerifier = Callable[[RenderedWakeupDefinition], bool]


def detect_wakeup_backend(
    *,
    platform_name: Optional[str] = None,
    command_exists: Optional[CommandExists] = None,
) -> WakeupBackend:
    """Feature-detect launchd, systemd timer, or cron.

    Parameters
    ----------
    platform_name:
        Optional test override for ``platform.system()``.
    command_exists:
        Optional command-presence probe.

    Returns
    -------
    WakeupBackend
        launchd on macOS, systemd when available elsewhere, otherwise cron.

    Raises
    ------
    WakeupConfigurationError
        If no supported recurring scheduler exists.
    """

    system = (platform_name or platform.system()).lower()
    exists = command_exists or (lambda name: shutil.which(name) is not None)
    if system == "darwin":
        if not exists("launchctl"):
            raise WakeupConfigurationError("macOS wake episodes require launchctl")
        return WakeupBackend.LAUNCHD
    if exists("systemctl"):
        return WakeupBackend.SYSTEMD_TIMER
    if exists("crontab"):
        return WakeupBackend.CRON
    raise WakeupConfigurationError("no launchd, systemd, or cron wakeup mechanism found")


def build_wake_episode(
    *,
    provider: str,
    reset_at: str,
    reset_observation: str,
    callback_argv: Sequence[str],
    not_before: Optional[str] = None,
    retry_interval_seconds: int = DEFAULT_RETRY_INTERVAL_SECONDS,
    supersedes_episode_id: Optional[str] = None,
) -> WakeEpisode:
    """Build the frozen durable episode identity and callback argv.

    Parameters
    ----------
    provider:
        Provider requiring a scheduled recheck: ``anthropic``, ``openai``,
        ``research-tools``, or ``forge``.
    reset_at:
        Provider reset timestamp, whether observed or guessed.
    reset_observation:
        Closed evidence class, ``observed`` or ``guessed``.
    callback_argv:
        Exact base crawler-resume argv. The episode identity argument is added.
    not_before:
        Callback guard; defaults to ``reset_at``.
    retry_interval_seconds:
        Fixed recurring cadence in the inclusive five-to-fifteen-minute bound.
    supersedes_episode_id:
        Optional earlier episode replaced by improved reset evidence.

    Returns
    -------
    WakeEpisode
        Exact Phase-0 frozen episode value.

    Raises
    ------
    WakeupConfigurationError
        If a frozen episode invariant is invalid.
    """

    _validate_provider(provider)
    _validate_reset_observation(reset_observation)
    _parse_utc(reset_at)
    guarded_at = not_before or reset_at
    _parse_utc(guarded_at)
    _validate_retry_interval(retry_interval_seconds)
    base_argv = tuple(callback_argv)
    if not base_argv or any(not argument for argument in base_argv):
        raise WakeupConfigurationError("callback argv must contain non-empty arguments")
    if "--wake-episode-id" in base_argv:
        raise WakeupConfigurationError("base callback argv must not predeclare an episode ID")
    callback_identity = stable_hash({"callback_argv": list(base_argv)})
    identity = stable_hash(
        {
            "provider": provider,
            "reset_at": reset_at,
            "reset_observation": reset_observation,
            "not_before": guarded_at,
            "retry_interval_seconds": retry_interval_seconds,
            "callback_identity": callback_identity,
            "supersedes_episode_id": supersedes_episode_id,
        }
    ).removeprefix("sha256:")
    episode_id = f"wake-episode-{identity[:32]}"
    if supersedes_episode_id == episode_id:
        raise WakeupConfigurationError("a wake episode cannot supersede itself")
    opened_event_id = idempotent_operational_event_id(
        OperationalEventKind.USAGE_PAUSE,
        {"episode_id": episode_id},
    )
    return WakeEpisode(
        episode_id=episode_id,
        provider=provider,
        reset_at=reset_at,
        reset_observation=reset_observation,
        not_before=guarded_at,
        retry_interval_seconds=retry_interval_seconds,
        callback_identity=callback_identity,
        callback_argv=(*base_argv, "--wake-episode-id", episode_id),
        opened_event_id=opened_event_id,
        supersedes_episode_id=supersedes_episode_id,
    )


def build_usage_pause_event(
    *,
    episode: WakeEpisode,
    observed_response: str,
    context: OperationalContext,
    created_at: str,
) -> JsonObject:
    """Construct the authoritative opening event for one wake episode.

    Parameters
    ----------
    episode:
        Frozen episode being opened.
    observed_response:
        Exact redacted provider limit observation.
    context:
        Common campaign context.
    created_at:
        Exact event timestamp.

    Returns
    -------
    dict[str, object]
        Logical operational event ready for append.
    """

    _validate_episode(episode)
    _parse_utc(created_at)
    event = _base_event(
        created_at=created_at,
        event_kind=OperationalEventKind.USAGE_PAUSE,
        status=OperationalEventStatus.USAGE_PAUSED,
        context=context,
        provider=episode.provider,
        observed_response=observed_response,
        reset_at=episode.reset_at,
        details={
            "episode": _episode_payload(episode),
            "reset_exact": episode.reset_observation == "observed",
        },
    )
    minted = mint_operational_event(
        event,
        identity_payload={"episode_id": episode.episode_id},
    )
    if minted["event_id"] != episode.opened_event_id:
        raise WakeupConfigurationError("wake episode opening identity is inconsistent")
    return minted


def build_usage_pause_observation_event(
    *,
    episode: WakeEpisode,
    observed_response: str,
    context: OperationalContext,
    created_at: str,
) -> JsonObject:
    """Build one occurrence-addressed changed usage-pause observation.

    Parameters
    ----------
    episode:
        Existing immutable wake episode whose opening remains authoritative.
    observed_response:
        Changed exact redacted provider response excerpt.
    context:
        Common campaign context.
    created_at:
        Exact observation timestamp and occurrence axis.

    Returns
    -------
    dict[str, object]
        Typed non-state-changing operational disclosure.
    """

    _validate_episode(episode)
    _parse_utc(created_at)
    event = _base_event(
        created_at=created_at,
        event_kind=OperationalEventKind.USAGE_PAUSE_OBSERVATION,
        status=OperationalEventStatus.USAGE_PAUSED,
        context=context,
        provider=episode.provider,
        observed_response=observed_response,
        reset_at=episode.reset_at,
        details={
            "episode_id": episode.episode_id,
            "opened_event_id": episode.opened_event_id,
        },
    )
    return mint_operational_event(event)


def record_operational_event(ledger: JsonlLedger, event: Mapping[str, object]) -> AppendResult:
    """Append one operational event through the canonical strict ledger.

    Parameters
    ----------
    ledger:
        Operational-event ledger.
    event:
        Complete logical event without sequence/hash fields.

    Returns
    -------
    AppendResult
        Idempotent append result.
    """

    if ledger.schema_version != OPERATIONAL_EVENT_SCHEMA_VERSION:
        raise ValueError("record_operational_event requires an operational-event ledger")
    return ledger.append(event)


def reduce_wake_episodes(events: Sequence[Mapping[str, object]]) -> WakeEpisodeProjection:
    """Derive current wake episode state solely from operational events.

    Parameters
    ----------
    events:
        Persisted or logical operational events in ledger order.

    Returns
    -------
    WakeEpisodeProjection
        Immutable derived episode view.

    Raises
    ------
    WakeupConfigurationError
        If immutable episode identity is conflicting or malformed.
    """

    episodes: dict[str, WakeEpisode] = {}
    opening_order: list[str] = []
    dispositions: dict[str, str] = {}
    terminal_events: dict[str, Optional[str]] = {}
    fire_buckets: dict[str, set[int]] = {}
    install_verified: dict[str, bool] = {}
    health_degraded: dict[str, bool] = {}
    last_event_ids: dict[str, str] = {}
    backends: dict[str, Optional[WakeupBackend]] = {}

    for event in events:
        raw_event_kind = event.get("event_kind")
        try:
            event_kind = OperationalEventKind(raw_event_kind)
        except (TypeError, ValueError):
            event_kind = None
        transition = _WAKE_EVENT_TRANSITIONS[event_kind] if event_kind is not None else None
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        event_id = str(event.get("event_id", ""))
        if transition is _WakeEventTransition.OPEN and isinstance(details.get("episode"), Mapping):
            episode = _episode_from_payload(details["episode"])
            existing = episodes.get(episode.episode_id)
            if existing is not None and existing != episode:
                raise WakeupConfigurationError(
                    f"conflicting operational episode identity: {episode.episode_id}"
                )
            if existing is None:
                episodes[episode.episode_id] = episode
                opening_order.append(episode.episode_id)
                dispositions[episode.episode_id] = "active"
                terminal_events[episode.episode_id] = None
                fire_buckets[episode.episode_id] = set()
                install_verified[episode.episode_id] = False
                health_degraded[episode.episode_id] = False
                backends[episode.episode_id] = None
            last_event_ids[episode.episode_id] = event_id
            superseded = episode.supersedes_episode_id
            if superseded in episodes and dispositions[superseded] == "active":
                dispositions[superseded] = "superseded"
                terminal_events[superseded] = episode.opened_event_id
            continue

        episode_id = details.get("episode_id")
        if transition in {
            _WakeEventTransition.RESOLVE_COMPLETED,
            _WakeEventTransition.RESOLVE_CANCELLED,
        }:
            targets = (
                [str(episode_id)] if isinstance(episode_id, str) and episode_id else list(episodes)
            )
            disposition = (
                "completed" if transition is _WakeEventTransition.RESOLVE_COMPLETED else "cancelled"
            )
            for target in targets:
                if target in episodes:
                    dispositions[target] = disposition
                    terminal_events[target] = event_id
                    last_event_ids[target] = event_id
            continue
        if not isinstance(episode_id, str) or episode_id not in episodes:
            continue
        last_event_ids[episode_id] = event_id
        if transition is _WakeEventTransition.RESUME:
            dispositions[episode_id] = "resumed"
            terminal_events[episode_id] = event_id
        elif transition is _WakeEventTransition.INSTALL:
            install_verified[episode_id] = details.get("verified") is True
            backend_value = details.get("backend")
            if isinstance(backend_value, str):
                try:
                    backends[episode_id] = WakeupBackend(backend_value)
                except ValueError as exc:
                    raise WakeupConfigurationError(
                        f"unknown wakeup backend in event {event_id}: {backend_value!r}"
                    ) from exc
        elif transition is _WakeEventTransition.INSTALL_FAILED:
            install_verified[episode_id] = False
        elif transition is _WakeEventTransition.FIRE:
            bucket = details.get("time_bucket")
            if isinstance(bucket, int) and bucket >= 0:
                fire_buckets[episode_id].add(bucket)
        elif transition is _WakeEventTransition.HEALTH_DEGRADED:
            health_degraded[episode_id] = True

    states: dict[str, WakeEpisodeState] = {}
    for episode_id in opening_order:
        states[episode_id] = WakeEpisodeState(
            episode=episodes[episode_id],
            disposition=dispositions[episode_id],
            terminal_event_id=terminal_events[episode_id],
            fire_buckets=tuple(sorted(fire_buckets[episode_id])),
            install_verified=install_verified[episode_id],
            health_degraded=health_degraded[episode_id],
            last_event_id=last_event_ids[episode_id],
            backend=backends[episode_id],
        )
    active_ids = tuple(episode_id for episode_id in opening_order if states[episode_id].active)
    return WakeEpisodeProjection(states, active_ids)


@dataclass(frozen=True)
class WakeLivenessAlarm:
    """One active wake episode that is past its deadline with no recorded fire."""

    episode_id: str
    provider: str
    deadline_at: str
    #: Seconds past ``deadline + grace``: the amount by which the fire is LATE,
    #: so ``0`` means "not overdue" on every surface that reports it.
    overdue_seconds: int
    seconds_since_deadline: int
    fire_count: int
    install_verified: bool
    remedy_command: str

    def summary(self) -> str:
        """Return one ASCII operator line naming the remedy.

        Returns
        -------
        str
            Alarm summary suitable for a notification transport.
        """

        return (
            f"wake episode {self.episode_id} ({self.provider}) is OVERDUE by "
            f"{self.overdue_seconds}s past {self.deadline_at} with {self.fire_count} "
            f"recorded fires; remedy: {self.remedy_command}"
        )


def _episode_deadline(state: WakeEpisodeState) -> datetime:
    """Return the instant after which an active episode owes a fire."""

    episode = state.episode
    reset = _parse_utc(episode.reset_at)
    not_before = _parse_utc(episode.not_before) if episode.not_before else reset
    return max(reset, not_before)


def _power_wake_disclosure(
    events: Sequence[Mapping[str, object]], episode_id: str
) -> Optional[JsonObject]:
    """Return the newest recorded power-wake disclosure for one episode."""

    installs = {
        OperationalEventKind.WAKEUP_INSTALLED.value,
        OperationalEventKind.WAKEUP_REPAIRED.value,
    }
    for event in reversed(tuple(events)):
        if event.get("event_kind") not in installs:
            continue
        details = event.get("details")
        if not isinstance(details, Mapping) or details.get("episode_id") != episode_id:
            continue
        disclosure = details.get("power_wake")
        if isinstance(disclosure, Mapping):
            return dict(disclosure)
    return None


def _fires_since(state: WakeEpisodeState, deadline: datetime) -> int:
    """Count recorded fire buckets at or after an episode's deadline bucket."""

    interval = state.episode.retry_interval_seconds
    floor = _time_bucket(deadline, interval)
    return sum(1 for bucket in state.fire_buckets if bucket >= floor)


def wake_liveness_alarms(
    events: Sequence[Mapping[str, object]],
    *,
    now: str,
    grace_seconds: int = WAKE_OVERDUE_GRACE_SECONDS,
) -> tuple[WakeLivenessAlarm, ...]:
    """Return every active episode whose fire is overdue, from the ledger alone.

    This is the INDEPENDENT half of wake liveness. A heartbeat emitted by the
    process that can die cannot report its own death, so overdue-ness is derived
    by a pure function over canonical operational events: no driver lock, no
    runtime state, no scheduler probe. An external poller, ``status``, the
    supervisor, and ``doctor`` all consult this one derivation, so they can never
    disagree about whether the campaign's only automated resume path is alive.

    Parameters
    ----------
    events:
        Canonical operational events in ledger order.
    now:
        Exact UTC instant to measure against.
    grace_seconds:
        Slack past the deadline before an absent fire is an alarm.

    Returns
    -------
    tuple[WakeLivenessAlarm, ...]
        Overdue active episodes in deterministic opening order.
    """

    instant = _parse_utc(now)
    alarms: list[WakeLivenessAlarm] = []
    for state in reduce_wake_episodes(events).active_episodes:
        deadline = _episode_deadline(state)
        elapsed = int((instant - deadline).total_seconds())
        overdue = elapsed - grace_seconds
        if overdue <= 0 or _fires_since(state, deadline):
            continue
        alarms.append(
            WakeLivenessAlarm(
                episode_id=state.episode.episode_id,
                provider=state.episode.provider,
                deadline_at=deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
                overdue_seconds=overdue,
                seconds_since_deadline=elapsed,
                fire_count=len(state.fire_buckets),
                install_verified=state.install_verified,
                remedy_command=(
                    "python -m menagerie.crawler --repo-root <root> wake status "
                    f"# then: wake cancel --episode-id {state.episode.episode_id} and resume"
                ),
            )
        )
    return tuple(alarms)


def render_wakeup_status(
    projection: WakeEpisodeProjection,
    *,
    now: Optional[str] = None,
    grace_seconds: int = WAKE_OVERDUE_GRACE_SECONDS,
    events: Optional[Sequence[Mapping[str, object]]] = None,
) -> JsonObject:
    """Render operational episode history and active LIVENESS for status consumers.

    ``healthy`` used to be ``install_verified and not health_degraded`` -- two
    configuration facts, no clock -- so an episode that was installed, verified
    and never fired rendered ``healthy: true`` forever while the campaign sat
    dead past its reset. That is exactly what the pilot's only wake trial
    reported (AREA_OPS_opus OPS-08, AREA_OPS_sol OPS-B08). A health predicate
    that cannot go false when the thing is dead is worse than no predicate,
    because it is affirmatively reassuring.

    When ``now`` is absent nothing is measured, so every liveness field renders
    :data:`UNMEASURED` and ``healthy`` renders ``unknown`` -- never ``0`` and
    never ``true`` (BL-6 (f)).

    Parameters
    ----------
    projection:
        Reducer-derived wake state.
    now:
        Exact UTC instant to measure liveness against.
    grace_seconds:
        Slack past the deadline before an absent fire is overdue.
    events:
        Canonical operational events, read only for the recorded power-wake
        disclosure. Omitting them renders that field ``unknown`` -- never
        "unarmed", which would be a claim nothing measured.

    Returns
    -------
    dict[str, object]
        JSON-safe wake status summary.
    """

    instant = _parse_utc(now) if now is not None else None
    episodes: list[JsonObject] = []
    overdue_ids: list[str] = []
    for state in projection.episodes.values():
        episode = state.episode
        overdue_seconds: object = UNMEASURED
        fires_since: object = UNMEASURED
        if instant is not None:
            deadline = _episode_deadline(state)
            fires_since = _fires_since(state, deadline)
            if not state.active:
                overdue_seconds = 0
            elif fires_since:
                # The scheduler IS firing for this window; whether the fire
                # resumed anything is the health counter's separate question.
                overdue_seconds = 0
            else:
                overdue_seconds = max(
                    0, int((instant - deadline).total_seconds()) - grace_seconds
                )
                if overdue_seconds:
                    overdue_ids.append(episode.episode_id)
        episodes.append(
            {
                "episode_id": episode.episode_id,
                "provider": episode.provider,
                "reset_at": episode.reset_at,
                "reset_observation": episode.reset_observation,
                "not_before": episode.not_before,
                "retry_interval_seconds": episode.retry_interval_seconds,
                "disposition": state.disposition,
                "active": state.active,
                "install_verified": state.install_verified,
                "backend": state.backend.value if state.backend is not None else None,
                "fire_count": len(state.fire_buckets),
                "fires_since_not_before": fires_since,
                "overdue_seconds": overdue_seconds,
                "last_time_bucket": state.fire_buckets[-1] if state.fire_buckets else None,
                "health_degraded": state.health_degraded,
                "terminal_event_id": state.terminal_event_id,
                "supersedes_episode_id": episode.supersedes_episode_id,
                "power_wake": (
                    UNMEASURED
                    if events is None
                    else (_power_wake_disclosure(events, episode.episode_id) or UNMEASURED)
                ),
            }
        )
    configured = all(
        state.install_verified and not state.health_degraded for state in projection.active_episodes
    )
    healthy: object = UNMEASURED if instant is None else (configured and not overdue_ids)
    return {
        "active_count": len(projection.active_episode_ids),
        "active_episode_ids": list(projection.active_episode_ids),
        "configured": configured,
        "grace_seconds": grace_seconds,
        "overdue_episode_ids": UNMEASURED if instant is None else overdue_ids,
        "healthy": healthy,
        "episodes": episodes,
    }


def inspect_wakeup_definitions(
    wakeup_root: Path,
    events: Sequence[Mapping[str, object]],
    backend: WakeupBackend,
    *,
    verifier: Optional[DefinitionVerifier] = None,
    linger_runner: Optional[Callable[[Sequence[str]], tuple[int, str]]] = None,
) -> WakeupInspection:
    """Compare active operational authority with local and OS definitions.

    Parameters
    ----------
    wakeup_root:
        Gitignored scheduler projection root.
    events:
        Canonical operational events.
    backend:
        Host scheduler backend.
    verifier:
        Optional injected OS registration verifier.
    linger_runner:
        Optional injected ``systemd`` lingering probe.

    Returns
    -------
    WakeupInspection
        Exact missing/mismatched, unverified, and non-boot-persistent active
        episode IDs.
    """

    projection = reduce_wake_episodes(events)
    active_ids = projection.active_episode_ids
    missing: list[str] = []
    unverified: list[str] = []
    transient: list[str] = []
    check = verifier or _verify_wakeup
    for state in projection.active_episodes:
        definition = render_wakeup_definition(wakeup_root, state.episode, backend)
        if not _definition_matches(definition):
            missing.append(state.episode.episode_id)
            continue
        if not check(definition):
            unverified.append(state.episode.episode_id)
            continue
        # A verified projection that a reboot would erase is the OPS-1 silent
        # park: the doctor must be able to say so while the episode is active,
        # which is precisely when it matters.
        if not boot_persistence(definition, linger_runner=linger_runner).persistent:
            transient.append(state.episode.episode_id)
    return WakeupInspection(
        backend, active_ids, tuple(missing), tuple(unverified), tuple(transient)
    )


class WakeupManager:
    """Reconcile recurring scheduler projections with operational episode authority."""

    def __init__(
        self,
        wakeup_root: Path,
        ledger: JsonlLedger,
        callback_argv: Sequence[str],
        *,
        backend: Optional[WakeupBackend] = None,
        platform_name: Optional[str] = None,
        command_exists: Optional[CommandExists] = None,
        installer: Optional[DefinitionAction] = None,
        verifier: Optional[DefinitionVerifier] = None,
        deactivator: Optional[DefinitionAction] = None,
        health_threshold_fires: int = DEFAULT_HEALTH_THRESHOLD_FIRES,
        power_wake_command: Optional[Sequence[str]] = None,
        power_wake_runner: Optional[Callable[[Sequence[str]], int]] = None,
        require_boot_persistence: Optional[bool] = None,
        ownership_registry: Optional["SchedulerOwnershipRegistry"] = None,
    ) -> None:
        """Bind scheduler effects and the canonical operational writer.

        Parameters
        ----------
        wakeup_root:
            Gitignored runtime root for definitions and fire intents.
        ledger:
            Canonical operational-event writer held under the driver lock.
        callback_argv:
            Base callback argv used when opening a new episode.
        backend, platform_name, command_exists:
            Explicit backend or feature-detection controls.
        installer, verifier, deactivator:
            Injectable OS scheduler operations.
        health_threshold_fires:
            Fire count that emits one visible degraded-health fact.
        power_wake_command, power_wake_runner:
            Second-wake-path wrapper argv and injected runner. Absent means the
            power wake stays unarmed and says so; it never blocks the pause.
        require_boot_persistence:
            W15: a pause must arm a BOOT-PERSISTENT resume in the same durable
            transaction or refuse typed -- a reboot while paused otherwise
            permanently silences the only automated resume path. ``None``
            derives the requirement from whether the REAL OS scheduler
            operations are in play (injected fakes cannot testify about the
            OS agent root, so the requirement is only meaningful -- and only
            armed by default -- when installer and verifier are the real
            ones); pass an explicit bool to force either arm.
        ownership_registry:
            D-59: the machine-scoped scheduler ownership registry. Every
            install appends a ``registered`` row in the same transaction and
            every deactivation a ``decommissioned`` row, so a host entry can
            always be traced to its owner even after the owning ledger is
            gone (the b1e1e9ee stale-cron class). ``None`` derives the
            default registry exactly when the REAL OS scheduler operations
            are in play (injected fakes must not write host state); pass a
            registry to force one.
        """

        if not callback_argv or any(not argument for argument in callback_argv):
            raise WakeupConfigurationError("callback argv must contain non-empty arguments")
        if health_threshold_fires <= 0:
            raise WakeupConfigurationError("health threshold must be positive")
        self.root = wakeup_root
        self.ledger = ledger
        self.callback_argv = tuple(callback_argv)
        self.backend = backend or detect_wakeup_backend(
            platform_name=platform_name, command_exists=command_exists
        )
        self.installer = installer or _install_wakeup
        self.verifier = verifier or _verify_wakeup
        self.deactivator = deactivator or _deactivate_wakeup
        self.require_boot_persistence = (
            (installer is None and verifier is None)
            if require_boot_persistence is None
            else require_boot_persistence
        )
        if ownership_registry is None and installer is None and verifier is None:
            from menagerie.crawler.scheduler_ownership import SchedulerOwnershipRegistry

            ownership_registry = SchedulerOwnershipRegistry.from_env()
        self.ownership_registry = ownership_registry
        self.health_threshold_fires = health_threshold_fires
        self.power_wake_command = power_wake_command
        self.power_wake_runner = power_wake_runner

    @property
    def projection(self) -> WakeEpisodeProjection:
        """Return current event-derived episode state.

        Returns
        -------
        WakeEpisodeProjection
            Current pure operational projection.
        """

        return reduce_wake_episodes(self.ledger.records)

    def record_pause_and_schedule(
        self,
        *,
        provider: str,
        observed_response: str,
        reset_at: str,
        context: OperationalContext,
        created_at: str,
        reset_observation: str = "observed",
        not_before: Optional[str] = None,
        retry_interval_seconds: int = DEFAULT_RETRY_INTERVAL_SECONDS,
    ) -> WakeupScheduleResult:
        """Open an episode and verify its recurring scheduler projection.

        A newer pause for the same provider supersedes the currently active
        episode. Replaying the exact pause is append- and install-idempotent.

        Parameters
        ----------
        provider, observed_response, reset_at:
            Exact provider observation and reset timestamp.
        context:
            Common campaign event context.
        created_at:
            Exact pause timestamp.
        reset_observation:
            ``observed`` or ``guessed``.
        not_before:
            Optional callback guard, defaulting to the reset timestamp.
        retry_interval_seconds:
            Recurrence in the frozen five-to-fifteen-minute range.

        Returns
        -------
        WakeupScheduleResult
            Frozen episode and verified disposable projection.

        Raises
        ------
        WakeupConfigurationError
            If installation cannot be durably verified.
        """

        projection = self.projection
        prior_for_provider = [
            state.episode
            for state in projection.episodes.values()
            if state.episode.provider == provider
        ]
        supersedes = prior_for_provider[-1].episode_id if prior_for_provider else None
        candidate = build_wake_episode(
            provider=provider,
            reset_at=reset_at,
            reset_observation=reset_observation,
            callback_argv=self.callback_argv,
            not_before=not_before,
            retry_interval_seconds=retry_interval_seconds,
            supersedes_episode_id=supersedes,
        )
        if supersedes is not None and projection.episodes[supersedes].active:
            prior = projection.episodes[supersedes].episode
            if _same_episode_request(prior, candidate):
                candidate = prior
        existing_open = next(
            (
                event
                for event in self.ledger.records
                if event.get("event_id") == candidate.opened_event_id
            ),
            None,
        )
        if existing_open is not None:
            pause = AppendResult(existing_open, appended=False)
            if existing_open.get("observed_response") != observed_response:
                record_operational_event(
                    self.ledger,
                    build_usage_pause_observation_event(
                        episode=candidate,
                        observed_response=observed_response,
                        context=context,
                        created_at=created_at,
                    ),
                )
        else:
            pause = record_operational_event(
                self.ledger,
                build_usage_pause_event(
                    episode=candidate,
                    observed_response=observed_response,
                    context=context,
                    created_at=created_at,
                ),
            )
        definition = render_wakeup_definition(self.root, candidate, self.backend)
        created, repaired, verified, scheduler_event = self._ensure_installed(
            definition, context=context, created_at=created_at
        )
        if supersedes is not None and candidate.episode_id != supersedes:
            prior = projection.episodes[supersedes].episode
            prior_definition = render_wakeup_definition(self.root, prior, self.backend)
            if self._projection_or_registration_exists(prior_definition):
                self._deactivate(
                    prior_definition,
                    context=context,
                    created_at=created_at,
                )
        return WakeupScheduleResult(
            candidate,
            definition,
            created,
            repaired,
            verified,
            pause,
            scheduler_event,
        )

    def reconcile(
        self,
        *,
        context: OperationalContext,
        created_at: str,
    ) -> WakeupReconciliationResult:
        """Repair active definitions and deactivate every resolved projection.

        Parameters
        ----------
        context:
            Common campaign event context.
        created_at:
            Exact reconciliation timestamp.

        Returns
        -------
        WakeupReconciliationResult
            Installed, repaired, deactivated, and failed episode IDs.
        """

        _parse_utc(created_at)
        projection = self.projection
        installed: list[str] = []
        repaired: list[str] = []
        deactivated: list[str] = []
        failures: dict[str, str] = {}
        for state in projection.episodes.values():
            definition = render_wakeup_definition(self.root, state.episode, self.backend)
            if state.active:
                try:
                    created, was_repaired, _, _ = self._ensure_installed(
                        definition, context=context, created_at=created_at
                    )
                except WakeupConfigurationError as exc:
                    failures[state.episode.episode_id] = str(exc)
                    continue
                if created:
                    installed.append(state.episode.episode_id)
                elif was_repaired:
                    repaired.append(state.episode.episode_id)
            elif self._projection_or_registration_exists(definition):
                try:
                    self._deactivate(definition, context=context, created_at=created_at)
                except WakeupConfigurationError as exc:
                    failures[state.episode.episode_id] = str(exc)
                else:
                    deactivated.append(state.episode.episode_id)
        return WakeupReconciliationResult(
            self.projection.active_episode_ids,
            tuple(installed),
            tuple(repaired),
            tuple(deactivated),
            failures,
        )

    def handle_fire(
        self,
        episode_id: str,
        *,
        fired_at: str,
        context: OperationalContext,
    ) -> WakeupCallbackResult:
        """Handle one callback fire while the caller holds ``driver.lock``.

        Parameters
        ----------
        episode_id:
            Exact active or recently resolved episode.
        fired_at:
            Actual callback timestamp used for the idempotency bucket.
        context:
            Common campaign event context.

        Returns
        -------
        WakeupCallbackResult
            Guard, resume, append, and deactivation decisions.

        Raises
        ------
        WakeupConfigurationError
            If the episode is unknown or scheduler deactivation fails.
        """

        fired = _parse_utc(fired_at)
        projection = self.projection
        try:
            state = projection.episodes[episode_id]
        except KeyError as exc:
            raise WakeupConfigurationError(f"unknown wake episode: {episode_id}") from exc
        episode = state.episode
        bucket = _time_bucket(fired, episode.retry_interval_seconds)
        guarded = fired < _parse_utc(episode.not_before)
        # DX-67 dual-lane pause coalescing: a resume is lawful only once EVERY
        # paused lane's reset has passed. Pauses are per-provider episodes
        # (author -> anthropic, checker -> openai) and supersession is
        # provider-scoped, so two lanes legitimately hold two active episodes
        # at once -- and the first lane's reset used to fire a resume straight
        # into the second lane's still-throttled provider. The gate: this fire
        # resumes only at max(reset) over ALL active episodes; earlier fires
        # stay guarded (the recurring schedule re-fires until the gate opens),
        # so exactly one resume happens, at max(reset).
        latest_reset = max(
            (
                _parse_utc(sibling.episode.not_before)
                for sibling in projection.active_episodes
            ),
            default=None,
        )
        sibling_guarded = (
            not guarded
            and state.active
            and latest_reset is not None
            and fired < latest_reset
        )
        if bucket in state.fire_buckets:
            return WakeupCallbackResult(
                episode_id, bucket, False, guarded or sibling_guarded, False, False
            )
        details: JsonObject = {
            "time_bucket": bucket,
            "guarded_not_before": guarded,
            "source": "scheduler-callback",
        }
        if sibling_guarded:
            details["coalesced_behind_reset"] = latest_reset.strftime("%Y-%m-%dT%H:%M:%SZ")
        fire_event = _episode_event(
            episode,
            event_kind=OperationalEventKind.WAKEUP_FIRED,
            status=OperationalEventStatus.WAKEUP_FIRED,
            context=context,
            created_at=fired_at,
            identity={"time_bucket": bucket},
            details=details,
        )
        fire_result = record_operational_event(self.ledger, fire_event)
        self._record_health_if_needed(episode, context=context, created_at=fired_at)
        state = self.projection.episodes[episode_id]
        if guarded or sibling_guarded:
            return WakeupCallbackResult(
                episode_id, bucket, fire_result.appended, True, False, False
            )
        if not state.active:
            definition = render_wakeup_definition(self.root, episode, self.backend)
            deactivated = self._deactivate(
                definition, context=context, created_at=fired_at, time_bucket=bucket
            )
            return WakeupCallbackResult(
                episode_id, bucket, fire_result.appended, False, False, deactivated
            )
        resume_event = _episode_event(
            episode,
            event_kind=OperationalEventKind.USAGE_RESUME,
            status=OperationalEventStatus.USAGE_RESUMED,
            context=context,
            created_at=fired_at,
            identity={"time_bucket": bucket},
            details={"time_bucket": bucket, "source": "scheduler-callback"},
        )
        record_operational_event(self.ledger, resume_event)
        definition = render_wakeup_definition(self.root, episode, self.backend)
        deactivated = self._deactivate(
            definition, context=context, created_at=fired_at, time_bucket=bucket
        )
        return WakeupCallbackResult(
            episode_id, bucket, fire_result.appended, False, True, deactivated
        )

    def resolve_episode(
        self,
        episode_id: str,
        *,
        resolution: str,
        context: OperationalContext,
        created_at: str,
    ) -> bool:
        """Durably resolve an episode before removing its scheduler projection.

        Parameters
        ----------
        episode_id:
            Exact frozen episode identity.
        resolution:
            ``manual-resume``, ``campaign-completed``, or ``operator-cancelled``.
        context:
            Common campaign operational context.
        created_at:
            Exact resolution timestamp.

        Returns
        -------
        bool
            True when a new deactivation fact was appended.

        Raises
        ------
        WakeupConfigurationError
            If the episode or resolution is unknown.
        """

        projection = self.projection
        try:
            episode = projection.episodes[episode_id].episode
        except KeyError as exc:
            raise WakeupConfigurationError(f"unknown wake episode: {episode_id}") from exc
        try:
            transition = _WakeResolution(resolution)
        except ValueError as exc:
            raise WakeupConfigurationError(f"unknown wake resolution: {resolution}") from exc
        event_kind, status = _WAKE_RESOLUTION_TRANSITIONS[transition]
        event = _episode_event(
            episode,
            event_kind=event_kind,
            status=status,
            context=context,
            created_at=created_at,
            identity={"resolution": resolution},
            details={"source": resolution},
        )
        record_operational_event(self.ledger, event)
        definition = render_wakeup_definition(self.root, episode, self.backend)
        return self._deactivate(definition, context=context, created_at=created_at)

    def ingest_fire_intents(
        self,
        *,
        context: OperationalContext,
        created_at: str,
    ) -> int:
        """Ingest fsynced lock-contention intents under the live driver lock.

        Parameters
        ----------
        context:
            Current live-driver operational context.
        created_at:
            Ingestion timestamp.

        Returns
        -------
        int
            Number of distinct durable intent files consumed.
        """

        _parse_utc(created_at)
        consumed = 0
        intent_root = self.root / "fire-intents"
        for path in sorted(intent_root.glob("*/*.json")) if intent_root.exists() else ():
            payload = _load_fire_intent(path)
            episode_id = str(payload["episode_id"])
            projection = self.projection
            if episode_id not in projection.episodes:
                raise WakeupConfigurationError(f"fire intent names unknown episode: {episode_id}")
            episode = projection.episodes[episode_id].episode
            bucket = int(payload["time_bucket"])
            fired_at = str(payload["fired_at"])
            if bucket in projection.episodes[episode_id].fire_buckets:
                path.unlink()
                fsync_directory(path.parent)
                consumed += 1
                continue
            event = _episode_event(
                episode,
                event_kind=OperationalEventKind.WAKEUP_FIRED,
                status=OperationalEventStatus.WAKEUP_FIRED,
                context=context,
                created_at=fired_at,
                identity={"time_bucket": bucket},
                details={
                    "time_bucket": bucket,
                    "guarded_not_before": _parse_utc(fired_at) < _parse_utc(episode.not_before),
                    "source": "lock-contention-intent",
                },
            )
            record_operational_event(self.ledger, event)
            noop = _episode_event(
                episode,
                event_kind=OperationalEventKind.WAKE_NOOP_ALREADY_RUNNING,
                status=OperationalEventStatus.WAKE_NOOP_ALREADY_RUNNING,
                context=context,
                created_at=fired_at,
                identity={"time_bucket": bucket},
                details={
                    "time_bucket": bucket,
                    "authoritative_driver_lock_live": True,
                    "source": "lock-contention-intent",
                },
            )
            record_operational_event(self.ledger, noop)
            self._record_health_if_needed(episode, context=context, created_at=created_at)
            path.unlink()
            fsync_directory(path.parent)
            consumed += 1
        return consumed

    def _ensure_installed(
        self,
        definition: RenderedWakeupDefinition,
        *,
        context: OperationalContext,
        created_at: str,
    ) -> tuple[bool, bool, bool, AppendResult]:
        """Write, install, verify, and visibly record one projection."""

        _parse_utc(created_at)
        projection_matches = _definition_matches(definition)
        registration_matches = projection_matches and self.verifier(definition)
        episode_id = definition.episode.episode_id
        previously_projected = any(
            event.get("event_kind")
            in {
                OperationalEventKind.WAKEUP_INSTALLED.value,
                OperationalEventKind.WAKEUP_REPAIRED.value,
            }
            and isinstance(event.get("details"), Mapping)
            and event["details"].get("episode_id") == episode_id
            for event in self.ledger.records
        )
        created = not previously_projected
        repaired = previously_projected and not registration_matches
        if registration_matches and previously_projected:
            # W15: a replayed install is still held to boot persistence -- a
            # deleted or drifted agent-root plist after the original install
            # is exactly the reboot-mortal state this refusal exists to catch.
            self._require_boot_persistent(definition, context=context, created_at=created_at)
            persisted = next(
                event
                for event in reversed(self.ledger.records)
                if event.get("event_kind")
                in {
                    OperationalEventKind.WAKEUP_INSTALLED.value,
                    OperationalEventKind.WAKEUP_REPAIRED.value,
                }
                and isinstance(event.get("details"), Mapping)
                and event["details"].get("episode_id") == episode_id
            )
            return False, False, True, AppendResult(persisted, appended=False)
        if not registration_matches:
            _write_definition(definition)
            try:
                self.installer(definition)
                verified = self.verifier(definition)
            except Exception as exc:
                self._record_install_failure(
                    definition.episode,
                    context=context,
                    created_at=created_at,
                    reason=str(exc),
                )
                raise WakeupConfigurationError(
                    f"cannot install recurring wake episode {definition.episode.episode_id}: {exc}"
                ) from exc
            if not verified:
                reason = "scheduler verification returned false after install"
                self._record_install_failure(
                    definition.episode,
                    context=context,
                    created_at=created_at,
                    reason=reason,
                )
                raise WakeupConfigurationError(
                    f"cannot verify recurring wake episode {definition.episode.episode_id}"
                )
            # D-59: ownership registration rides the install transaction. An
            # entry we cannot register would recreate the untraceable-orphan
            # class, so a failed append refuses the install.
            if self.ownership_registry is not None:
                self.ownership_registry.register_installation(
                    entry_id=definition.episode.episode_id,
                    backend=definition.backend.value,
                    label=_ownership_label(definition),
                    callback_argv=definition.episode.callback_argv,
                    run_id=context.run_id,
                    machine_id=context.machine_id,
                    owning_ledger=self.ledger.path,
                    created_at=created_at,
                )
        else:
            verified = True
        # W15: the pause and its boot-persistent resume are ONE durable
        # transaction -- the installed event below is appended only after the
        # persistence requirement holds, so "paused" can never durably stand
        # over a wake a reboot would silently unschedule.
        persistence = self._require_boot_persistent(
            definition, context=context, created_at=created_at
        )
        # Armed AFTER the recurring projection verifies, so a failed install never
        # leaves a scheduled power wake for an episode that does not exist.
        power_wake = arm_power_wake(
            definition.episode,
            backend=definition.backend,
            command=self.power_wake_command,
            runner=self.power_wake_runner,
        )
        kind = (
            OperationalEventKind.WAKEUP_REPAIRED
            if repaired
            else OperationalEventKind.WAKEUP_INSTALLED
        )
        status = (
            OperationalEventStatus.WAKEUP_REPAIRED
            if repaired
            else OperationalEventStatus.WAKEUP_INSTALLED
        )
        event = _episode_event(
            definition.episode,
            event_kind=kind,
            status=status,
            context=context,
            created_at=created_at,
            identity={
                "backend": definition.backend.value,
                "definition_sha256": stable_hash(definition.scheduler_definition),
                "repair_at": created_at if repaired else None,
            },
            details={
                "backend": definition.backend.value,
                "definition_path": str(definition.definition_path),
                "definition_sha256": stable_hash(definition.scheduler_definition),
                "verified": verified,
                "recurring": True,
                "retry_interval_seconds": definition.episode.retry_interval_seconds,
                # BL-6 (b): the second wake path is DISCLOSED on every install,
                # armed or not. The plan claimed two wake paths while the pmset
                # half had no code at all; an absent disclosure is how that went
                # unnoticed, so this field is required, never conditional.
                "power_wake": power_wake.to_dict(),
                "wake_paths_armed": 1 + int(power_wake.armed),
                # BL-6 (b): DERIVED per backend, never a literal -- whether the
                # OS still has this job after a restart is the difference between
                # a pause that resumes itself and a permanent silent park.
                "boot_persistent": persistence.to_dict(),
            },
        )
        return created, repaired, verified, record_operational_event(self.ledger, event)

    def _require_boot_persistent(
        self,
        definition: RenderedWakeupDefinition,
        *,
        context: OperationalContext,
        created_at: str,
    ) -> BootPersistence:
        """Refuse typed when a required boot-persistent resume does not stand.

        W15: 200a's wake plists were empty after the reboot -- the pause rows
        stood, the resume path did not, and every recovery was a human. When
        the requirement is armed (real OS scheduler operations, or an explicit
        opt-in), a non-persistent wake refuses the pause LOUDLY instead of
        durably recording a pause whose resume dies with the next boot. The
        verdict is always returned so the install event disclosure and this
        refusal can never disagree.
        """

        persistence = boot_persistence(definition)
        if self.require_boot_persistence and not persistence.persistent:
            reason = f"wake is not boot-persistent: {persistence.reason}"
            self._record_install_failure(
                definition.episode,
                context=context,
                created_at=created_at,
                reason=reason,
            )
            raise WakeupConfigurationError(
                f"pause refuses for wake episode {definition.episode.episode_id}: "
                f"{reason}. A reboot while paused would permanently silence the "
                "only automated resume path (W15)"
            )
        return persistence

    def _record_install_failure(
        self,
        episode: WakeEpisode,
        *,
        context: OperationalContext,
        created_at: str,
        reason: str,
    ) -> None:
        """Append one visible fail-closed scheduler installation fact."""

        event = _episode_event(
            episode,
            event_kind=OperationalEventKind.WAKEUP_FAILED,
            status=OperationalEventStatus.RUNNER_FAILED,
            context=context,
            created_at=created_at,
            identity={"reason": reason, "failed_at": created_at},
            details={"backend": self.backend.value, "reason": reason, "verified": False},
        )
        record_operational_event(self.ledger, event)

    def _record_health_if_needed(
        self,
        episode: WakeEpisode,
        *,
        context: OperationalContext,
        created_at: str,
    ) -> None:
        """Emit one health fact without changing recurrence or episode state."""

        state = self.projection.episodes[episode.episode_id]
        if len(state.fire_buckets) < self.health_threshold_fires or state.health_degraded:
            return
        event = _episode_event(
            episode,
            event_kind=OperationalEventKind.WAKEUP_HEALTH_DEGRADED,
            status=OperationalEventStatus.WAKEUP_HEALTH_DEGRADED,
            context=context,
            created_at=created_at,
            identity={"threshold": self.health_threshold_fires},
            details={
                "fire_count": len(state.fire_buckets),
                "threshold": self.health_threshold_fires,
                "recurrence_retained": True,
            },
        )
        record_operational_event(self.ledger, event)

    def _deactivate(
        self,
        definition: RenderedWakeupDefinition,
        *,
        context: OperationalContext,
        created_at: str,
        time_bucket: Optional[int] = None,
    ) -> bool:
        """Deactivate only after reducer state has a durable resolving fact."""

        state = self.projection.episodes[definition.episode.episode_id]
        if state.active:
            raise WakeupConfigurationError(
                f"refusing to deactivate active episode {definition.episode.episode_id}"
            )
        try:
            self.deactivator(definition)
        except Exception as exc:
            self._record_install_failure(
                definition.episode,
                context=context,
                created_at=created_at,
                reason=f"deactivation failed: {exc}",
            )
            raise WakeupConfigurationError(
                f"cannot deactivate wake episode {definition.episode.episode_id}: {exc}"
            ) from exc
        _remove_definition_projection(definition)
        # D-59: the removal is bookkept in the machine-scoped ownership
        # registry so the host census can distinguish a lawful deactivation
        # from residue.
        if self.ownership_registry is not None:
            self.ownership_registry.record_decommission(
                entry_id=definition.episode.episode_id,
                backend=definition.backend.value,
                verb="wakeup-deactivation",
                reason=f"episode resolved: terminal event {state.terminal_event_id}",
                removed_content=definition.scheduler_definition,
                run_id=context.run_id,
                machine_id=context.machine_id,
                created_at=created_at,
            )
        identity: JsonObject = {"terminal_event_id": state.terminal_event_id}
        if time_bucket is not None:
            identity["time_bucket"] = time_bucket
        event = _episode_event(
            definition.episode,
            event_kind=OperationalEventKind.WAKEUP_DEACTIVATED,
            status=OperationalEventStatus.WAKEUP_DEACTIVATED,
            context=context,
            created_at=created_at,
            identity=identity,
            details={
                "backend": definition.backend.value,
                "terminal_event_id": state.terminal_event_id,
                "time_bucket": time_bucket,
            },
        )
        return record_operational_event(self.ledger, event).appended

    def _projection_or_registration_exists(self, definition: RenderedWakeupDefinition) -> bool:
        """Return whether local or OS disposable state still exists."""

        return definition.definition_path.exists() or self.verifier(definition)


@dataclass(frozen=True)
class PowerWakeReceipt:
    """Disclosure of the second, independent wake path for one episode.

    A launchd ``StartInterval`` timer does NOT wake a sleeping Mac -- it fires
    late, after something else wakes the host -- so on macOS the recurring
    callback is one wake path, not two. This receipt is how that fact becomes
    visible instead of assumed: ``armed`` is true only when a real scheduling
    command exited zero, and ``reason`` always names why it did not.
    """

    supported: bool
    armed: bool
    reason: str
    argv: tuple[str, ...] = ()
    wake_at: Optional[str] = None

    def to_dict(self) -> JsonObject:
        """Return the JSON-safe disclosure carried by install events and status.

        Returns
        -------
        dict[str, object]
            Closed disclosure payload.
        """

        return {
            "supported": self.supported,
            "armed": self.armed,
            "reason": self.reason,
            "argv": list(self.argv),
            "wake_at": self.wake_at,
        }


def render_power_wake_plan(
    episode: WakeEpisode, *, command: Sequence[str]
) -> tuple[tuple[str, ...], str]:
    """Return the power-wake argv and the exact instant it schedules.

    ``pmset schedule`` reads LOCAL time, so the episode's UTC reset is converted
    once, here, rather than at a call site that might forget.

    Parameters
    ----------
    episode:
        Frozen episode authority.
    command:
        Operator-blessed wrapper argv (for example ``["sudo", "pmset"]``).

    Returns
    -------
    tuple[tuple[str, ...], str]
        Complete argv, and the UTC instant it arms.
    """

    deadline = _parse_utc(episode.not_before or episode.reset_at)
    wake_at = deadline - timedelta(seconds=POWER_WAKE_LEAD_SECONDS)
    local = wake_at.astimezone()
    return (
        (*command, "schedule", "wakeorpoweron", local.strftime("%m/%d/%y %H:%M:%S")),
        wake_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def arm_power_wake(
    episode: WakeEpisode,
    *,
    backend: WakeupBackend,
    command: Optional[Sequence[str]] = None,
    runner: Optional[Callable[[Sequence[str]], int]] = None,
) -> PowerWakeReceipt:
    """Attempt the second wake path and return an honest receipt either way.

    This never raises and never blocks the pause: a host that cannot arm a power
    wake must still park with a recurring callback installed. What it must not do
    is report two wake paths when it has one.

    Parameters
    ----------
    episode:
        Frozen episode authority.
    backend:
        Recurring backend in use; power wake is a macOS/launchd pairing.
    command:
        Explicit wrapper argv, defaulting to :data:`POWER_WAKE_COMMAND_ENV`.
    runner:
        Injected argv runner returning an exit status.

    Returns
    -------
    PowerWakeReceipt
        Disclosure carried into the install event and the status surface.
    """

    if backend is not WakeupBackend.LAUNCHD:
        return PowerWakeReceipt(
            supported=False,
            armed=False,
            reason=f"power wake is a launchd/macOS pairing; backend is {backend.value}",
        )
    configured = command if command is not None else os.environ.get(POWER_WAKE_COMMAND_ENV)
    if not configured:
        return PowerWakeReceipt(
            supported=True,
            armed=False,
            reason=(
                "no power-wake command configured: a launchd StartInterval timer does "
                f"not wake a sleeping host, so set {POWER_WAKE_COMMAND_ENV} to an "
                "operator-blessed `pmset schedule` wrapper or accept ONE wake path"
            ),
        )
    argv_prefix = shlex.split(configured) if isinstance(configured, str) else tuple(configured)
    if not argv_prefix:
        return PowerWakeReceipt(
            supported=True, armed=False, reason="power-wake command resolved to empty argv"
        )
    argv, wake_at = render_power_wake_plan(episode, command=argv_prefix)
    execute = runner or _run_power_wake
    try:
        status = execute(argv)
    except Exception as exc:  # noqa: BLE001 -- an unarmed power wake is disclosed, never fatal
        return PowerWakeReceipt(
            supported=True,
            armed=False,
            reason=f"power-wake command failed: {exc}",
            argv=tuple(argv),
            wake_at=wake_at,
        )
    if status != 0:
        return PowerWakeReceipt(
            supported=True,
            armed=False,
            reason=f"power-wake command exited {status}",
            argv=tuple(argv),
            wake_at=wake_at,
        )
    return PowerWakeReceipt(
        supported=True,
        armed=True,
        reason="power wake scheduled",
        argv=tuple(argv),
        wake_at=wake_at,
    )


@dataclass(frozen=True)
class BootPersistence:
    """Whether one installed definition survives a reboot, and why.

    Two orthogonal facts describe an armed wake: ``verified`` says the scheduler
    accepted the job NOW, and this says whether the OS will still have it after
    a restart. Round 1's pass reported the second as a hardcoded ``False``, which
    was both unfixed for launchd and simply wrong for cron -- an untrue constant
    in the direction that hides which backend actually needs the work.
    """

    persistent: bool
    reason: str
    install_path: Optional[str] = None

    def to_dict(self) -> JsonObject:
        """Return the JSON-safe disclosure carried by install events.

        Returns
        -------
        dict[str, object]
            Closed disclosure payload.
        """

        return {
            "persistent": self.persistent,
            "reason": self.reason,
            "install_path": self.install_path,
        }


def launchd_agent_root() -> Path:
    """Return THE login-persistent launchd agent root, derived exactly once.

    Returns
    -------
    Path
        Directory launchd scans for per-user jobs at login.
    """

    override = os.environ.get(LAUNCHD_AGENT_ROOT_ENV)
    if override:
        return Path(override)
    return Path.home().joinpath(*LAUNCHD_AGENT_ROOT_PARTS)


def boot_persistent_install_path(definition: RenderedWakeupDefinition) -> Optional[Path]:
    """Return where this definition must live to survive a reboot, if anywhere.

    Parameters
    ----------
    definition:
        Rendered recurring projection.

    Returns
    -------
    Path | None
        Boot-persistent install path, or ``None`` for backends whose own store
        is already boot-persistent.
    """

    if definition.backend is WakeupBackend.LAUNCHD:
        return launchd_agent_root() / f"{_launchd_label(definition.episode)}.plist"
    return None


def boot_persistence(
    definition: RenderedWakeupDefinition,
    *,
    linger_runner: Optional[Callable[[Sequence[str]], tuple[int, str]]] = None,
) -> BootPersistence:
    """Return an honest, per-backend verdict on surviving a reboot.

    The rule is closed and enumerated per backend rather than inferred:

    * **launchd** -- persistent only when the byte-exact plist sits in the
      login-persistent agent root. A drifted copy is NOT persistence: launchd
      would reload bytes that no longer describe this episode.
    * **cron** -- persistent, because the daemon re-reads the crontab at boot.
    * **systemd --user** -- persistent only when lingering is enabled, probed
      rather than assumed, because user units otherwise stop at logout.

    Parameters
    ----------
    definition:
        Rendered recurring projection.
    linger_runner:
        Injected probe returning ``(exit status, stdout)``.

    Returns
    -------
    BootPersistence
        Disclosure carried into the install event and the status surface.
    """

    if definition.backend is WakeupBackend.CRON:
        return BootPersistence(
            persistent=True,
            reason="cron re-reads the crontab at boot",
        )
    if definition.backend is WakeupBackend.SYSTEMD_TIMER:
        status, stdout = _probe_linger(linger_runner)
        if status != 0:
            return BootPersistence(
                persistent=False,
                reason=(
                    "cannot prove `systemd --user` survives a reboot: the linger probe "
                    f"`{shlex.join(LINGER_PROBE_ARGV)}` exited {status}. Without lingering, "
                    "user units stop at logout and a rebooted host runs nothing"
                ),
            )
        if "linger=yes" not in stdout.strip().lower():
            return BootPersistence(
                persistent=False,
                reason=(
                    "user lingering is disabled, so this timer dies at logout and does "
                    "not come back on reboot; enable it with `loginctl enable-linger`"
                ),
            )
        return BootPersistence(
            persistent=True,
            reason="user lingering is enabled and the timer unit is enabled",
        )
    install_path = boot_persistent_install_path(definition)
    if install_path is None:  # pragma: no cover -- launchd always names a path
        raise WakeupConfigurationError(
            f"backend {definition.backend.value} states no boot-persistence rule"
        )
    if not install_path.is_file():
        return BootPersistence(
            persistent=False,
            reason=(
                "launchd does not scan the campaign runtime root at login, so this job "
                f"is lost by any reboot until it is installed at {install_path}"
            ),
            install_path=str(install_path),
        )
    if install_path.read_text(encoding="utf-8") != definition.scheduler_definition:
        return BootPersistence(
            persistent=False,
            reason=(
                f"the boot-persistent plist at {install_path} does not match the rendered "
                "episode, so a reboot would reload a stale job"
            ),
            install_path=str(install_path),
        )
    return BootPersistence(
        persistent=True,
        reason=f"launchd reloads {install_path} at login",
        install_path=str(install_path),
    )


def _probe_linger(
    runner: Optional[Callable[[Sequence[str]], tuple[int, str]]],
) -> tuple[int, str]:
    """Return the linger probe's exit status and stdout, never raising."""

    if runner is not None:
        return runner(LINGER_PROBE_ARGV)
    if shutil.which(LINGER_PROBE_ARGV[0]) is None:
        return 127, ""
    try:
        completed = subprocess.run(
            [*LINGER_PROBE_ARGV, str(os.getuid())],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:  # noqa: BLE001 -- an unprovable probe is disclosed, never fatal
        return 1, ""
    return completed.returncode, completed.stdout


def _write_boot_persistent_definition(
    definition: RenderedWakeupDefinition,
) -> Optional[Path]:
    """Atomically place the definition where the OS reloads it at boot.

    Parameters
    ----------
    definition:
        Rendered recurring projection.

    Returns
    -------
    Path | None
        The installed path, or ``None`` for a natively boot-persistent backend.
    """

    install_path = boot_persistent_install_path(definition)
    if install_path is None:
        return None
    install_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(install_path, definition.scheduler_definition.encode("utf-8"))
    return install_path


def _remove_boot_persistent_definition(definition: RenderedWakeupDefinition) -> None:
    """Remove the boot-persistent copy, so a finished episode cannot resume.

    A leftover agent-root plist would relaunch a campaign that already closed on
    the next reboot, which is the same class of defect in the other direction.
    """

    install_path = boot_persistent_install_path(definition)
    if install_path is None:
        return
    install_path.unlink(missing_ok=True)
    if install_path.parent.exists():
        fsync_directory(install_path.parent)


def _run_power_wake(argv: Sequence[str]) -> int:
    """Run one power-wake wrapper argv with a bounded deadline."""

    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.returncode


def wakeup_projection_root(runtime_root: Path) -> Path:
    """Return THE wake projection directory for one campaign runtime root.

    Every consumer -- driver, doctor, ``status``, ``wake cancel``, reaper --
    derives the path here so a plural/singular drift cannot split the brain
    again.

    Parameters
    ----------
    runtime_root:
        Campaign ``.crawl-local`` runtime root.

    Returns
    -------
    Path
        Wake definition and fire-intent root.
    """

    return runtime_root / WAKEUP_PROJECTION_DIRNAME


def callback_working_directory(callback_argv: Sequence[str]) -> Optional[Path]:
    """Return the ABSOLUTE root a callback declared for its own imports.

    The rule is closed and exact: the value following :data:`CALLBACK_ROOT_FLAG`,
    accepted only when it is absolute. A relative claim is not a working
    directory -- launchd would resolve it against ``/``, which is the AU-1 defect
    itself -- so it is refused rather than normalized against an ambient cwd the
    scheduler does not share.

    Parameters
    ----------
    callback_argv:
        Rendered callback argv.

    Returns
    -------
    Path | None
        The declared absolute root, or ``None`` when the callback declares none.
    """

    argv = tuple(callback_argv)
    for index, argument in enumerate(argv[:-1]):
        if argument != CALLBACK_ROOT_FLAG:
            continue
        candidate = Path(argv[index + 1])
        if candidate.is_absolute():
            return candidate
    return None


def _resolve_working_directory(wakeup_root: Path, episode: WakeEpisode) -> Path:
    """Resolve the absolute directory the rendered callback must execute from.

    Preference order, each a statable derivation rather than a guess: the root
    the callback itself declares; the campaign root the projection lives under
    (``<root>/.crawl-local/wakeup`` implies ``<root>``); the projection root.
    """

    declared = callback_working_directory(episode.callback_argv)
    if declared is not None:
        return declared
    root = wakeup_root.resolve()
    if root.name == WAKEUP_PROJECTION_DIRNAME and root.parent.name == ".crawl-local":
        return root.parent.parent
    return root


def render_wakeup_definition(
    wakeup_root: Path,
    episode: WakeEpisode,
    backend: WakeupBackend,
    *,
    working_directory: Optional[Path] = None,
) -> RenderedWakeupDefinition:
    """Render an equivalent recurring backend projection that can EXECUTE.

    Every backend definition names an absolute working directory and injects it
    on ``PYTHONPATH``. Without those two facts the callback dies pre-import,
    pre-ledger and traceless on every fire: launchd execs from ``/``, ``systemd
    --user`` and cron from ``$HOME``, and ``import menagerie`` resolves from the
    campaign root only. That is not a hypothetical -- it is why the pilot's one
    real quota pause never fired once and ended in a manual resume 1h51m past
    its reset (AREA_AUTHOR_opus AU-1, TRACKB_opus F1, AREA_OPS_fable OPS-1).

    Parameters
    ----------
    wakeup_root:
        Gitignored definition root.
    episode:
        Frozen episode authority.
    backend:
        OS scheduler backend.
    working_directory:
        Explicit absolute callback root. Defaults to the callback's own declared
        root, then to the campaign root implied by ``wakeup_root``.

    Returns
    -------
    RenderedWakeupDefinition
        Deterministic disposable scheduler files.
    """

    _validate_episode(episode)
    suffix = {
        WakeupBackend.LAUNCHD: "plist",
        WakeupBackend.SYSTEMD_TIMER: "timer",
        WakeupBackend.CRON: "cron",
    }[backend]
    path = wakeup_root / f"{episode.episode_id}.{suffix}"
    work_dir = working_directory or _resolve_working_directory(wakeup_root, episode)
    if not work_dir.is_absolute():
        raise WakeupConfigurationError(
            f"wake callback working directory must be absolute, not {work_dir}"
        )
    cwd = str(work_dir)
    if backend is WakeupBackend.LAUNCHD:
        args = "".join(f"<string>{_xml_escape(arg)}</string>" for arg in episode.callback_argv)
        # A traceless crash is how AU-1 stayed invisible for a whole pause: the
        # callback's own stderr is the only forensic record of a pre-ledger death.
        log_root = wakeup_root / "logs"
        content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0"><dict>'
            f"<key>Label</key><string>{_launchd_label(episode)}</string>"
            f"<key>ProgramArguments</key><array>{args}</array>"
            f"<key>WorkingDirectory</key><string>{_xml_escape(cwd)}</string>"
            "<key>EnvironmentVariables</key><dict>"
            f"<key>PYTHONPATH</key><string>{_xml_escape(cwd)}</string>"
            "</dict>"
            "<key>StandardOutPath</key>"
            f"<string>{_xml_escape(str(log_root / f'{episode.episode_id}.out'))}</string>"
            "<key>StandardErrorPath</key>"
            f"<string>{_xml_escape(str(log_root / f'{episode.episode_id}.err'))}</string>"
            f"<key>StartInterval</key><integer>{episode.retry_interval_seconds}</integer>"
            "<key>RunAtLoad</key><true/></dict></plist>\n"
        )
        return RenderedWakeupDefinition(episode, backend, path, content, None, None)
    if backend is WakeupBackend.SYSTEMD_TIMER:
        service_path = path.with_suffix(".service")
        timer = (
            "[Unit]\nDescription=TorchLens crawler recurring usage-limit wake\n"
            "[Timer]\n"
            # OnUnitInactiveSec alone is relative to a last-deactivation
            # timestamp a never-run service does not have, so the timer would
            # very likely never fire even once (AREA_OPS_fable OPS-1.4).
            # OnActiveSec gives the first fire an anchor the timer does have.
            f"OnActiveSec={episode.retry_interval_seconds}s\n"
            f"OnUnitInactiveSec={episode.retry_interval_seconds}s\n"
            "Persistent=true\n"
            f"Unit={service_path.name}\n"
            "[Install]\nWantedBy=timers.target\n"
        )
        service = (
            "[Unit]\nDescription=TorchLens crawler wake callback\n"
            "[Service]\nType=oneshot\n"
            f"WorkingDirectory={cwd}\n"
            f'Environment="PYTHONPATH={cwd}"\n'
            f"ExecStart={shlex.join(episode.callback_argv)}\n"
        )
        return RenderedWakeupDefinition(episode, backend, path, timer, service_path, service)
    minutes = episode.retry_interval_seconds // 60
    quoted_cwd = shlex.quote(cwd)
    command = (
        f"cd {quoted_cwd} && PYTHONPATH={quoted_cwd} {shlex.join(episode.callback_argv)}"
    )
    marker = _cron_marker(episode)
    content = f"{marker} begin\nCRON_TZ=UTC\n*/{minutes} * * * * {command}\n{marker} end\n"
    return RenderedWakeupDefinition(episode, backend, path, content, None, None)


def write_fire_intent(
    wakeup_root: Path,
    episode: WakeEpisode,
    *,
    fired_at: str,
) -> Path:
    """Atomically fsync one lock-contention callback intent per time bucket.

    Parameters
    ----------
    wakeup_root:
        Gitignored wake runtime root.
    episode:
        Frozen episode read from canonical operational history.
    fired_at:
        Actual callback timestamp.

    Returns
    -------
    Path
        Durable intent path. Replays return the same path.

    Raises
    ------
    WakeupConfigurationError
        If a same-bucket intent has conflicting bytes.
    """

    fired = _parse_utc(fired_at)
    bucket = _time_bucket(fired, episode.retry_interval_seconds)
    payload: JsonObject = {
        "episode_id": episode.episode_id,
        "time_bucket": bucket,
        "fired_at": fired_at,
        "callback_identity": episode.callback_identity,
    }
    path = wakeup_root / "fire-intents" / episode.episode_id / f"{bucket}.json"
    encoded = canonical_json_bytes(payload) + b"\n"
    if path.exists():
        existing = _load_fire_intent(path)
        if (
            existing["episode_id"] != episode.episode_id
            or existing["time_bucket"] != bucket
            or existing["callback_identity"] != episode.callback_identity
        ):
            raise WakeupConfigurationError(f"conflicting fire intent: {path}")
        return path
    _write_bytes_atomic(path, encoded)
    return path


def _same_episode_request(prior: WakeEpisode, candidate: WakeEpisode) -> bool:
    """Return whether a candidate is an exact replay apart from supersession."""

    return bool(
        prior.provider == candidate.provider
        and prior.reset_at == candidate.reset_at
        and prior.reset_observation == candidate.reset_observation
        and prior.not_before == candidate.not_before
        and prior.retry_interval_seconds == candidate.retry_interval_seconds
        and prior.callback_identity == candidate.callback_identity
    )


def _episode_payload(episode: WakeEpisode) -> JsonObject:
    """Return the JSON-safe exact frozen episode payload."""

    payload = asdict(episode)
    payload["callback_argv"] = list(episode.callback_argv)
    return payload


def _episode_from_payload(payload: Mapping[str, object]) -> WakeEpisode:
    """Parse and validate an exact frozen episode payload."""

    expected = {
        "episode_id",
        "provider",
        "reset_at",
        "reset_observation",
        "not_before",
        "retry_interval_seconds",
        "callback_identity",
        "callback_argv",
        "opened_event_id",
        "supersedes_episode_id",
    }
    if set(payload) != expected:
        raise WakeupConfigurationError(
            f"wake episode payload fields differ: expected={sorted(expected)}, got={sorted(payload)}"
        )
    argv = payload["callback_argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise WakeupConfigurationError("wake episode callback_argv must be a string list")
    supersedes = payload["supersedes_episode_id"]
    if supersedes is not None and not isinstance(supersedes, str):
        raise WakeupConfigurationError("wake episode supersedes ID must be a string or null")
    retry_interval = payload["retry_interval_seconds"]
    if not isinstance(retry_interval, int):
        raise WakeupConfigurationError("wake episode retry interval must be an integer")
    episode = WakeEpisode(
        episode_id=str(payload["episode_id"]),
        provider=str(payload["provider"]),
        reset_at=str(payload["reset_at"]),
        reset_observation=str(payload["reset_observation"]),
        not_before=str(payload["not_before"]),
        retry_interval_seconds=retry_interval,
        callback_identity=str(payload["callback_identity"]),
        callback_argv=tuple(argv),
        opened_event_id=str(payload["opened_event_id"]),
        supersedes_episode_id=supersedes,
    )
    _validate_episode(episode)
    return episode


def _validate_episode(episode: WakeEpisode) -> None:
    """Validate frozen episode fields without creating another authority type."""

    _validate_provider(episode.provider)
    _validate_reset_observation(episode.reset_observation)
    _parse_utc(episode.reset_at)
    _parse_utc(episode.not_before)
    _validate_retry_interval(episode.retry_interval_seconds)
    expected_opening_id = idempotent_operational_event_id(
        OperationalEventKind.USAGE_PAUSE,
        {"episode_id": episode.episode_id},
    )
    legacy_opening_id = f"usage-pause-{episode.episode_id}"
    if not episode.episode_id or episode.opened_event_id not in {
        expected_opening_id,
        legacy_opening_id,
    }:
        raise WakeupConfigurationError("wake episode/opening event identity is invalid")
    if not episode.callback_argv or any(not argument for argument in episode.callback_argv):
        raise WakeupConfigurationError("wake episode callback argv is invalid")
    try:
        flag_index = episode.callback_argv.index("--wake-episode-id")
    except ValueError as exc:
        raise WakeupConfigurationError("wake episode callback omits its identity") from exc
    if (
        flag_index + 1 >= len(episode.callback_argv)
        or episode.callback_argv[flag_index + 1] != episode.episode_id
    ):
        raise WakeupConfigurationError("wake episode callback identity argument is invalid")
    base_argv = episode.callback_argv[:flag_index] + episode.callback_argv[flag_index + 2 :]
    if stable_hash({"callback_argv": list(base_argv)}) != episode.callback_identity:
        raise WakeupConfigurationError("wake episode callback identity hash is invalid")
    if episode.supersedes_episode_id == episode.episode_id:
        raise WakeupConfigurationError("wake episode cannot supersede itself")


def _validate_provider(provider: str) -> None:
    """Validate the closed provider vocabulary."""

    if provider not in SCHEDULED_RECHECK_PROVIDERS:
        raise WakeupConfigurationError(f"unsupported scheduled-recheck provider: {provider!r}")


def _validate_reset_observation(reset_observation: str) -> None:
    """Validate observed-versus-guessed reset evidence."""

    if reset_observation not in RESET_OBSERVATIONS:
        raise WakeupConfigurationError(
            f"reset_observation must be one of {sorted(RESET_OBSERVATIONS)}"
        )


def _validate_retry_interval(retry_interval_seconds: int) -> None:
    """Validate the frozen recurring cadence and cron parity."""

    if not MIN_RETRY_INTERVAL_SECONDS <= retry_interval_seconds <= MAX_RETRY_INTERVAL_SECONDS:
        raise WakeupConfigurationError("retry cadence must be between 300 and 900 seconds")
    if retry_interval_seconds % 60 != 0:
        raise WakeupConfigurationError("retry cadence must be an exact whole minute")
    if 3600 % retry_interval_seconds != 0:
        raise WakeupConfigurationError(
            "retry cadence must divide one hour for exact recurring cron parity"
        )


def _episode_event(
    episode: WakeEpisode,
    *,
    event_kind: OperationalEventKind,
    status: OperationalEventStatus,
    context: OperationalContext,
    created_at: str,
    identity: Mapping[str, object],
    details: JsonObject,
) -> JsonObject:
    """Build one deterministic episode-linked operational event."""

    _parse_utc(created_at)
    event = _base_event(
        created_at=created_at,
        event_kind=event_kind,
        status=status,
        context=context,
        provider=episode.provider,
        observed_response=None,
        reset_at=episode.reset_at,
        details={"episode_id": episode.episode_id, **details},
    )
    spec = OPERATIONAL_EVENT_IDENTITY_REGISTRY[event_kind]
    identity_payload = (
        {"episode_id": episode.episode_id, **dict(identity)}
        if spec.identity_class is OperationalEventIdentityClass.IDEMPOTENT_FIRST_CLOCK
        else None
    )
    return mint_operational_event(event, identity_payload=identity_payload)


def _base_event(
    *,
    created_at: str,
    event_kind: OperationalEventKind,
    status: OperationalEventStatus,
    context: OperationalContext,
    provider: Optional[str],
    observed_response: Optional[str],
    reset_at: Optional[str],
    details: JsonObject,
) -> JsonObject:
    """Build the strict common operational-event envelope."""

    return {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": created_at,
        "event_kind": event_kind.value,
        "status": status.value,
        "provider": provider,
        "observed_response": observed_response,
        "reset_at": reset_at,
        "queued_work_counts": dict(context.queued_work_counts),
        "current_environment": context.current_environment,
        "run_id": context.run_id,
        "machine_id": context.machine_id,
        "details": details,
    }


def _definition_matches(definition: RenderedWakeupDefinition) -> bool:
    """Return whether every local projection byte matches the renderer."""

    if not definition.definition_path.is_file():
        return False
    if definition.definition_path.read_text(encoding="utf-8") != definition.scheduler_definition:
        return False
    if definition.service_path is None:
        return True
    return bool(
        definition.service_path.is_file()
        and definition.service_path.read_text(encoding="utf-8") == definition.service_definition
    )


def _write_definition(definition: RenderedWakeupDefinition) -> None:
    """Atomically replace disposable definition projections and fsync them."""

    _write_bytes_atomic(definition.definition_path, definition.scheduler_definition.encode("utf-8"))
    if definition.service_path is not None and definition.service_definition is not None:
        _write_bytes_atomic(definition.service_path, definition.service_definition.encode("utf-8"))


def _remove_definition_projection(definition: RenderedWakeupDefinition) -> None:
    """Remove local projections only after scheduler deactivation succeeds."""

    definition.definition_path.unlink(missing_ok=True)
    if definition.service_path is not None:
        definition.service_path.unlink(missing_ok=True)
    if definition.definition_path.parent.exists():
        fsync_directory(definition.definition_path.parent)


def _install_wakeup(definition: RenderedWakeupDefinition) -> None:
    """Install or repair one recurring backend definition."""

    if definition.backend is WakeupBackend.LAUNCHD:
        domain = f"gui/{os.getuid()}"
        label = _launchd_label(definition.episode)
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{label}"],
            check=False,
            capture_output=True,
            text=True,
        )
        # BL-6 (b): bootstrap from the LOGIN-PERSISTENT agent root, not from the
        # gitignored runtime projection, so the job launchd reloads at the next
        # login is this episode's job. Bootstrapping the runtime copy is what
        # made a reboot-while-paused unschedule the only automated resume path.
        install_path = _write_boot_persistent_definition(definition)
        _run_checked(["launchctl", "bootstrap", domain, str(install_path)])
        return
    if definition.backend is WakeupBackend.SYSTEMD_TIMER:
        if definition.service_path is None:
            raise WakeupConfigurationError("systemd timer projection omits its service")
        _run_checked(
            [
                "systemctl",
                "--user",
                "link",
                str(definition.service_path),
                str(definition.definition_path),
            ]
        )
        _run_checked(["systemctl", "--user", "daemon-reload"])
        _run_checked(["systemctl", "--user", "enable", "--now", definition.definition_path.name])
        return
    current = _read_crontab()
    stripped = _remove_cron_block(current, definition.episode)
    if stripped and not stripped.endswith("\n"):
        stripped += "\n"
    _write_crontab(stripped + definition.scheduler_definition)


def _verify_wakeup(definition: RenderedWakeupDefinition) -> bool:
    """Return whether the OS scheduler has the exact recurring projection active."""

    if not _definition_matches(definition):
        return False
    if definition.backend is WakeupBackend.LAUNCHD:
        domain = f"gui/{os.getuid()}"
        result = subprocess.run(
            ["launchctl", "print", f"{domain}/{_launchd_label(definition.episode)}"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    if definition.backend is WakeupBackend.SYSTEMD_TIMER:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", definition.definition_path.name],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "active"
    return definition.scheduler_definition in _read_crontab()


def _deactivate_wakeup(definition: RenderedWakeupDefinition) -> None:
    """Remove one recurring backend definition idempotently."""

    if definition.backend is WakeupBackend.LAUNCHD:
        domain = f"gui/{os.getuid()}"
        result = subprocess.run(
            ["launchctl", "bootout", f"{domain}/{_launchd_label(definition.episode)}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in {0, 3, 113}:
            raise WakeupConfigurationError(f"launchd deactivation failed: {result.stderr.strip()}")
        # Only after the scheduler let go: a surviving agent-root plist would
        # resume a closed campaign on the next reboot.
        _remove_boot_persistent_definition(definition)
        return
    if definition.backend is WakeupBackend.SYSTEMD_TIMER:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "disable",
                "--now",
                definition.definition_path.name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and "not loaded" not in result.stderr.lower():
            raise WakeupConfigurationError(f"systemd deactivation failed: {result.stderr.strip()}")
        return
    current = _read_crontab()
    updated = _remove_cron_block(current, definition.episode)
    if updated != current:
        _write_crontab(updated)


def _ownership_label(definition: RenderedWakeupDefinition) -> str:
    """Return the host-visible identifier the ownership registry records."""

    if definition.backend is WakeupBackend.LAUNCHD:
        return _launchd_label(definition.episode)
    if definition.backend is WakeupBackend.CRON:
        return _cron_marker(definition.episode)
    return definition.definition_path.name


def _launchd_label(episode: WakeEpisode) -> str:
    """Return the deterministic launchd label for an episode."""

    return f"org.menagerie.crawler-wake.{episode.episode_id}"


def _cron_marker(episode: WakeEpisode) -> str:
    """Return the unique cron projection marker."""

    return f"# menagerie-wake-episode {episode.episode_id}"


def _read_crontab() -> str:
    """Read the current user crontab, treating absence as empty."""

    current = subprocess.run(["crontab", "-l"], check=False, capture_output=True, text=True)
    if current.returncode not in {0, 1}:
        raise WakeupConfigurationError(f"cannot inspect current crontab: {current.stderr.strip()}")
    return current.stdout if current.returncode == 0 else ""


def _write_crontab(content: str) -> None:
    """Replace the user crontab without invoking a shell."""

    installed = subprocess.run(
        ["crontab", "-"], input=content, check=False, capture_output=True, text=True
    )
    if installed.returncode != 0:
        raise WakeupConfigurationError(
            f"cannot install recurring cron wakeup: {installed.stderr.strip()}"
        )


def _remove_cron_block(content: str, episode: WakeEpisode) -> str:
    """Remove exactly one marked cron episode block."""

    begin = f"{_cron_marker(episode)} begin"
    end = f"{_cron_marker(episode)} end"
    output: list[str] = []
    skipping = False
    for line in content.splitlines(keepends=True):
        if line.rstrip("\n") == begin:
            skipping = True
            continue
        if skipping and line.rstrip("\n") == end:
            skipping = False
            continue
        if not skipping:
            output.append(line)
    if skipping:
        raise WakeupConfigurationError(f"unterminated cron wake block for {episode.episode_id}")
    return "".join(output)


def _run_checked(argv: Sequence[str]) -> None:
    """Run one scheduler argv and raise a typed error on failure."""

    completed = subprocess.run(list(argv), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise WakeupConfigurationError(
            f"scheduler command {list(argv)!r} failed: {completed.stderr.strip()}"
        )


def _load_fire_intent(path: Path) -> JsonObject:
    """Load and validate one exact local fire-intent object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WakeupConfigurationError(f"invalid fire intent {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "episode_id",
        "time_bucket",
        "fired_at",
        "callback_identity",
    }:
        raise WakeupConfigurationError(f"invalid fire intent fields: {path}")
    if not isinstance(payload["time_bucket"], int) or payload["time_bucket"] < 0:
        raise WakeupConfigurationError(f"invalid fire intent bucket: {path}")
    _parse_utc(str(payload["fired_at"]))
    return payload


def _time_bucket(instant: datetime, interval_seconds: int) -> int:
    """Return the deterministic UTC recurrence bucket."""

    return int(instant.timestamp()) // interval_seconds


def _parse_utc(value: str) -> datetime:
    """Parse an exact RFC 3339 UTC timestamp."""

    if not value.endswith("Z"):
        raise WakeupConfigurationError("wakeup timestamps must be RFC 3339 UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise WakeupConfigurationError(f"invalid wakeup timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise WakeupConfigurationError("wakeup timestamps must use exact UTC")
    return parsed


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    """Atomically write and fsync one disposable runtime file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _xml_escape(value: str) -> str:
    """Escape an argv leaf for a launchd plist."""

    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
