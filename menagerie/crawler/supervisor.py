"""Per-campaign crawler keep-alive and launchd integration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import plistlib
import shlex
import subprocess
import sys
import time
from typing import Any, BinaryIO, Callable, Mapping, Optional, Protocol, Sequence

from menagerie.crawler.campaign_config import CampaignConfig, load_campaign_config
from menagerie.crawler.continuation_manifest import forwarded_continuation_argv
from menagerie.crawler.notification_policy import (
    NotificationClass,
    classify_notification_event,
)
from menagerie.crawler.cli import (
    EXIT_ERROR,
    EXIT_LOCKED,
    EXIT_OK,
    EXIT_OPERATOR_OUTAGE,
    EXIT_PAUSED,
    EXIT_REVIEW_PAUSED,
)
from menagerie.crawler.constants import (
    AUTHOR_QUEUE_STALL_SECONDS,
    DEFAULT_AUTHOR_WAVE_CONCURRENCY,
    MAX_AUTHOR_WAVE_CONCURRENCY,
)
from menagerie.crawler.driver_progress import CommandNotifier
from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes, stable_hash
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.partitioner import DEFAULT_CAMPAIGN_MANIFEST, find_campaign_binding
from menagerie.crawler.wakeup import UNMEASURED

#: Fast-burst crash accounting, kept as the SECOND tripwire: five unexpected exits
#: inside half an hour is a crash loop whatever progress each leg claims. It is
#: deliberately no longer the ONLY one -- see CONSECUTIVE_FLAT_RESTART_LIMIT.
CRASH_WINDOW_SECONDS = 30 * 60
CRASH_LIMIT = 5
#: SCHED-04: the wall-clock-INDEPENDENT crash circuit. At 20k-model leg durations a
#: restart-to-crash cycle is hours -- step-100's four crashes spanned ~55h -- so
#: ``len(recent)`` inside the 30-minute window above never exceeded 1 and the burst
#: circuit provably never armed: a deterministic mid-leg crash was an unbounded
#: restart loop at full spend with zero messages. Three is this project's
#: anti-flailing number: three consecutive restarts that moved no durable progress
#: marker are a flail, and there is never a fourth attempt at a failing approach.
CONSECUTIVE_FLAT_RESTART_LIMIT = 3

#: W6: bounded supervised-restart ordinal per supervision episode. The ladder
#: restarts a RESTARTABLE typed exit at most this many times (each attested
#: via --supervised-restart-ordinal on the driver argv and the leg_started
#: row) before paging and stopping; the flat-restart and burst circuits above
#: keep their own independent bounds.
SUPERVISED_RESTART_ORDINAL_LIMIT = 5
MAX_RESTART_BACKOFF_SECONDS = 15 * 60
LOCK_BACKOFF_SECONDS = 30.0
WATCHDOG_POLL_SECONDS = 30.0
NOTIFICATION_RETRY_SECONDS = 60.0
#: OPS-05: the alert retry was ``while not notify(): sleep(60)`` -- unbounded, so a
#: broken transport turned every terminal stop into a live silent zombie that
#: launchd's keep-alive read as healthy. Five attempts at the 60s retry interval
#: bound one terminal alert to about five minutes; a transport that failed five
#: consecutive times is broken, not busy, and further retries only delay the
#: durable second channel that an independent poller can actually see.
NOTIFICATION_MAX_ATTEMPTS = 5
#: OPS-06/OPS-08: ``EXIT_LOCKED`` slept 30s and retried FOREVER, with no crash
#: accounting, no bound and no alert. Twenty refusals under the bounded exponential
#: backoff below is a little over four hours of continuous contention -- far past
#: any legitimate driver handoff -- after which the holder identity is paged.
LOCK_CONTENTION_MAX_ATTEMPTS = 20
#: OPS-01: milestone pushes are COUNT-triggered with 5000-model gaps, so at
#: 45.8 min/model the last two gaps are 159 days of silence BY DESIGN. The
#: heartbeat is LEVEL-triggered instead, which makes a MISSING file -- not a
#: missing message -- the death signal an independent poller keys on. Fifteen
#: minutes is well inside one model's wall time, so a stall shows up long before
#: the campaign's own reporting cadence could describe it.
HEARTBEAT_SECONDS = 15 * 60

SUPERVISOR_STATE_FORMAT = "menagerie.crawler.supervisor-state.v2"
#: v1 documents predate the flat-restart circuit and the alert-class projection.
#: They load with those fields at their defaults; nothing is guessed from them.
LEGACY_SUPERVISOR_STATE_FORMAT = "menagerie.crawler.supervisor-state.v1"
HEARTBEAT_FORMAT = "menagerie.crawler.supervisor-heartbeat.v1"
ALERT_DROP_FORMAT = "menagerie.crawler.supervisor-alert-drop.v1"
#: The loud marker the OPS-01 deadman agent reads: its mere existence means the
#: campaign tried to raise its hand and the transport refused.
NOTIFY_FAILED_MARKER = "NOTIFY_FAILED"

#: OPS-B06 closed allowlist, arm one: a VERIFIED completion. These are the exact
#: driver statuses the completion notification has always recognized.
VERIFIED_COMPLETION_PREFIXES = (
    "complete",
    "phase-complete",
    "terminal-partition-complete",
    "phase-terminal-partition-complete",
)
#: OPS-B06 closed allowlist, arms two and three: an explicit durable pause, and the
#: verified wake handoff the driver refuses to write unless the wake projection
#: verified (``paused:usage-limit`` is written only after that check).
DURABLE_PAUSE_PREFIXES = ("paused:",)
#: RC-1: the crawl invocation's typed lawful end under its limit.
WORK_EXHAUSTED_STATUS = "work_exhausted"

#: Closed declared-stop vocabulary: a driver exit status the driver DESCRIBED, its
#: alert event identity, and the operator summary. Every row pages once and stops.
DECLARED_STOPS: tuple[tuple[int, str, str], ...] = (
    (
        EXIT_PAUSED,
        "usage-pause",
        (
            "Menagerie crawler {campaign_id} paused for provider usage; the durable "
            "reset-time wake episode now owns resume"
        ),
    ),
    (
        EXIT_REVIEW_PAUSED,
        "review-pause",
        (
            "Menagerie crawler {campaign_id} reached its human review checkpoint; "
            "resume requires --after-review"
        ),
    ),
    (
        EXIT_OPERATOR_OUTAGE,
        "operator-outage",
        (
            "Menagerie crawler {campaign_id} stopped for RETRYABLE INFRASTRUCTURE; "
            "restore the managing session/operator lane"
        ),
    ),
)


class SupervisorError(RuntimeError):
    """Raised when the supervisor configuration cannot preserve its contracts."""


class NotificationLane(Protocol):
    """Receipt-bearing operator notification boundary."""

    def notify(
        self,
        summary: str,
        *,
        idempotency_key: str,
        notification_class: Optional[str] = None,
    ) -> bool:
        """Deliver one idempotent summary with a nonce receipt.

        ``notification_class`` is the closed
        :class:`~menagerie.crawler.notification_policy.NotificationClass`
        value the caller derived for this alert; the transport-owning
        implementation applies the class policy, and ``None`` always
        attempts delivery.
        """

        ...


class ManagedProcess(Protocol):
    """Minimal subprocess surface polled by the keep-alive."""

    def poll(self) -> Optional[int]:
        """Return the exit status, or ``None`` while running."""

        ...

    def wait(self) -> int:
        """Wait for process completion and return its status."""

        ...


ProcessFactory = Callable[
    [Sequence[str], Path, BinaryIO, BinaryIO],
    ManagedProcess,
]


@dataclass(frozen=True)
class QueueStall:
    """One author queue job older than the R6 liveness deadline."""

    job_id: str
    attempt_nonce: str
    stable_id: str
    age_seconds: float
    claimed: bool
    marker_path: Optional[Path] = None


@dataclass(frozen=True)
class SupervisorConfig:
    """Frozen per-campaign keep-alive configuration."""

    campaign_id: str
    repo_root: Path
    campaign_config_path: Path
    author_queue_root: Optional[Path]
    runtime_root: Path
    #: Author-session fan-out the supervised driver must keep using. It is carried
    #: explicitly because it is NOT part of the frozen campaign config: without it a
    #: supervised restart would silently drop back to the default and quietly halve a
    #: tuned campaign's throughput for the rest of the month.
    author_concurrency: int = DEFAULT_AUTHOR_WAVE_CONCURRENCY
    wake_episode_id: Optional[str] = None
    #: WS-1: pre-built selection argv fragments (scheduler, scope flags,
    #: continuation identity) forwarded VERBATIM into the driver resume so a
    #: supervised wake replays the complete launching config.
    selection_argv: tuple[str, ...] = ()
    stall_seconds: float = float(AUTHOR_QUEUE_STALL_SECONDS)
    poll_seconds: float = WATCHDOG_POLL_SECONDS
    notification_retry_seconds: float = NOTIFICATION_RETRY_SECONDS
    #: Overhaul RC-1 (D.1 behavior 6): ``resume`` is the historical leg-shaped
    #: driver command (kept for one release); ``crawl`` launches the continuous
    #: crawl (``--records-root``, ``--campaign``, ``--limit``, ``--author-width``).
    driver_shape: str = "resume"
    records_root: Optional[Path] = None
    crawl_limit: Optional[int] = None
    target: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate absolute paths and positive watchdog intervals."""

        if not self.campaign_id:
            raise ValueError("campaign_id cannot be empty")
        self._validate_driver_shape()
        for path in (self.repo_root, self.campaign_config_path, self.runtime_root):
            if not path.is_absolute():
                raise ValueError(f"supervisor path must be absolute: {path}")
        if self.author_queue_root is not None and not self.author_queue_root.is_absolute():
            raise ValueError("author_queue_root must be absolute")
        if not 1 <= self.author_concurrency <= MAX_AUTHOR_WAVE_CONCURRENCY:
            raise ValueError(
                "author_concurrency must be between 1 and "
                f"{MAX_AUTHOR_WAVE_CONCURRENCY}, not {self.author_concurrency}"
            )
        if min(self.stall_seconds, self.poll_seconds, self.notification_retry_seconds) <= 0:
            raise ValueError("supervisor intervals must be positive")


    def _validate_driver_shape(self) -> None:
        """Validate the RC-1 driver shape fields."""

        if self.driver_shape not in {"resume", "crawl"}:
            raise ValueError(f"driver_shape must be 'resume' or 'crawl', not {self.driver_shape!r}")
        if self.driver_shape != "crawl":
            return
        if self.records_root is None or not self.records_root.is_absolute():
            raise ValueError("crawl supervision requires an absolute records_root")
        if self.crawl_limit is not None and self.crawl_limit < 1:
            raise ValueError("crawl_limit must be positive")


@dataclass(frozen=True)
class CrashAccounting:
    """One unexpected driver exit, accounted both ways.

    ``window`` is the historical rolling burst projection; ``consecutive_flat_restarts``
    is the wall-clock-independent SCHED-04 counter, which is the one that can actually
    fire at 20k-model leg durations.
    """

    window: tuple[str, ...]
    consecutive_flat_restarts: int
    progress_fingerprint: str


@dataclass(frozen=True)
class SupervisorState:
    """Durable crash, flat-restart, and delivered-alert projection."""

    crash_timestamps: tuple[str, ...] = ()
    notified_stalls: tuple[str, ...] = ()
    #: SCHED-04: consecutive restarts that moved NO durable progress marker.
    consecutive_flat_restarts: int = 0
    #: Durable-progress fingerprint observed at the last unexpected exit. An empty
    #: string means no crash has been accounted yet, never "no progress".
    progress_fingerprint: str = ""
    #: Alert classes already DELIVERED, so a repeating class pages once. Silence is
    #: never recorded here: an undelivered class stays un-deduplicated.
    notified_alert_classes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return the closed durable representation.

        Returns
        -------
        dict[str, object]
            JSON-compatible supervisor state.
        """

        return {
            "format": SUPERVISOR_STATE_FORMAT,
            "crash_timestamps": list(self.crash_timestamps),
            "notified_stalls": list(self.notified_stalls),
            "consecutive_flat_restarts": self.consecutive_flat_restarts,
            "progress_fingerprint": self.progress_fingerprint,
            "notified_alert_classes": list(self.notified_alert_classes),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SupervisorState":
        """Parse one durable state document.

        Parameters
        ----------
        value:
            Decoded JSON mapping.

        Returns
        -------
        SupervisorState
            Validated state.

        Raises
        ------
        SupervisorError
            If the format is not one of the two known versions, or any field has the
            wrong type. Nothing is defaulted except fields a v1 document predates.
        """

        if value.get("format") not in (SUPERVISOR_STATE_FORMAT, LEGACY_SUPERVISOR_STATE_FORMAT):
            raise SupervisorError("supervisor state format is invalid")
        crashes = value.get("crash_timestamps")
        stalls = value.get("notified_stalls")
        if (
            not isinstance(crashes, list)
            or any(not isinstance(item, str) for item in crashes)
            or not isinstance(stalls, list)
            or any(not isinstance(item, str) for item in stalls)
        ):
            raise SupervisorError("supervisor state fields are invalid")
        flat = value.get("consecutive_flat_restarts", 0)
        fingerprint = value.get("progress_fingerprint", "")
        classes = value.get("notified_alert_classes", [])
        if (
            not isinstance(flat, int)
            or isinstance(flat, bool)
            or flat < 0
            or not isinstance(fingerprint, str)
            or not isinstance(classes, list)
            or any(not isinstance(item, str) for item in classes)
        ):
            raise SupervisorError("supervisor state liveness fields are invalid")
        return cls(
            crash_timestamps=tuple(crashes),
            notified_stalls=tuple(stalls),
            consecutive_flat_restarts=flat,
            progress_fingerprint=fingerprint,
            notified_alert_classes=tuple(classes),
        )


class CrawlerSupervisor:
    """Restart one campaign driver and alert on every terminal stop condition."""

    def __init__(
        self,
        config: SupervisorConfig,
        notifier: NotificationLane,
        *,
        process_factory: Optional[ProcessFactory] = None,
        now: Optional[Callable[[], datetime]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Bind injected process, clock, sleep, and receipt notifier effects."""

        self.config = config
        self.notifier = notifier
        self._process_factory = process_factory or _start_process
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or time.sleep
        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        self._state_path = self.config.runtime_root / "state.json"
        self._state = _load_state(self._state_path)
        self._heartbeat_path = self.config.runtime_root / "heartbeat.json"
        self._alerts_root = self.config.runtime_root / "alerts"
        self._marker_path = self.config.runtime_root / NOTIFY_FAILED_MARKER
        #: Events whose operator alert exhausted every attempt. A non-empty list makes
        #: the process exit NON-ZERO so the OS layer sees the silence (OPS-05).
        self._undelivered: list[str] = []
        self._last_heartbeat: Optional[datetime] = None
        self._lock_refusals = 0

    def run(self) -> int:
        """Supervise until a described stop, a bounded refusal, or an open circuit.

        Returns
        -------
        int
            ``EXIT_OK`` only for a stop the campaign DESCRIBED -- a verified
            completion, an explicit durable pause, or a verified wake handoff -- whose
            alert was delivered. Every other outcome, including an exhausted operator
            alert, is ``EXIT_ERROR``: launchd's unsuccessful-exit keep-alive is how an
            indeterminate stop gets restarted and how a broken notification transport
            becomes visible outside this process (OPS-05, OPS-B06).
        """

        armed = self._settle_armed_circuit()
        if armed is not None:
            return armed
        while True:
            self._write_heartbeat("supervising")
            returncode = self._run_driver_once()
            if returncode == EXIT_OK:
                return self._settle_exit_zero()
            declared = _declared_stop(returncode)
            if declared is not None:
                event, template = declared
                self._notify_until_delivered(
                    event, template.format(campaign_id=self.config.campaign_id)
                )
                return self._stop(event, EXIT_OK)
            if returncode == EXIT_LOCKED:
                contended = self._settle_lock_contention()
                if contended is not None:
                    return contended
                continue
            crashed = self._settle_crash(returncode)
            if crashed is not None:
                return crashed

    def _run_driver_once(self) -> int:
        """Spawn one driver invocation while polling the author queue watchdog.

        Returns
        -------
        int
            Driver exit status.
        """

        self._leg_events_baseline = self._leg_events_count()
        command = _driver_command(
            self.config, restart_ordinal=getattr(self, "_restart_ordinal", 0)
        )
        stdout_path = self.config.runtime_root / "driver.stdout.log"
        stderr_path = self.config.runtime_root / "driver.stderr.log"
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = self._process_factory(command, self.config.repo_root, stdout, stderr)
            while process.poll() is None:
                self._check_queue_stalls()
                self._write_heartbeat_if_due("supervising")
                self._sleep(self.config.poll_seconds)
            return process.wait()

    def _settle_armed_circuit(self) -> Optional[int]:
        """Refuse to spend another leg while an open flat-restart circuit is still armed.

        The launchd keep-alive is ``SuccessfulExit: False``, so the non-zero exit that
        makes an open circuit visible to the OS also gets this supervisor restarted.
        Without this pre-flight the circuit would buy exactly one throttle interval and
        the campaign would keep burning a full leg per restart -- the unbounded
        restart loop at full spend that SCHED-04 measured. The refusal is cheap, loud,
        and holds until durable progress lands or an operator clears the state.

        Returns
        -------
        int | None
            ``EXIT_ERROR`` while the circuit holds, else ``None`` to supervise.
        """

        if self._state.consecutive_flat_restarts < CONSECUTIVE_FLAT_RESTART_LIMIT:
            return None
        fingerprint = self._progress_fingerprint()
        if fingerprint != self._state.progress_fingerprint:
            # Durable progress landed after the circuit opened -- an operator drained
            # the blocker, or a partial leg committed -- so the flail is over.
            self._state = replace(
                self._state,
                consecutive_flat_restarts=0,
                progress_fingerprint=fingerprint,
            )
            _write_state(self._state_path, self._state)
            return None
        self._notify_alert_class(
            "flat-restart-circuit-armed",
            {
                "campaign_id": self.config.campaign_id,
                "event": "flat-restart-circuit-armed",
                "progress_fingerprint": fingerprint,
            },
            (
                f"Menagerie crawler {self.config.campaign_id} refused to launch: the "
                f"flat-restart circuit is open at {self._state.consecutive_flat_restarts} "
                "consecutive restarts with no durable progress; clear "
                f"{self._state_path} or land durable progress to re-arm"
            ),
        )
        return self._stop("flat-restart-circuit-armed", EXIT_ERROR)

    def _settle_exit_zero(self) -> int:
        """Settle a zero driver exit against the closed honest-stop allowlist.

        OPS-B06: exit 0 with a missing or fieldless ``driver-state.json`` -- an
        accidental zero exit, a lost state write, a graceful SIGTERM -- used to return
        0 and notify NOTHING, and launchd's ``SuccessfulExit: False`` keep-alive then
        never restarts. Only a verified completion, an explicit durable pause, or the
        verified wake handoff the driver refuses to record unverified may end a
        supervised campaign quietly.

        Returns
        -------
        int
            Supervisor exit status.
        """

        status = _driver_state_status(self.config.repo_root / ".crawl-local")
        if status == WORK_EXHAUSTED_STATUS:
            # RC-1: the crawl ran its scheduled work under its limit to
            # settlement. A described stop; the operator raises the limit.
            self._notify_until_delivered(
                "work-exhausted",
                f"Menagerie crawler {self.config.campaign_id} exhausted its crawl "
                f"limit ({status}); raise --limit to continue",
            )
            return self._stop("work-exhausted", EXIT_OK)
        if status is not None and status.startswith(VERIFIED_COMPLETION_PREFIXES):
            self._notify_until_delivered(
                "campaign-complete",
                f"Menagerie crawler {self.config.campaign_id} completed: {status}",
            )
            return self._stop("campaign-complete", EXIT_OK)
        if status is not None and status.startswith(DURABLE_PAUSE_PREFIXES):
            return self._stop(f"durable-pause:{status}", EXIT_OK)
        self._notify_until_delivered(
            "indeterminate-stop",
            (
                f"Menagerie crawler {self.config.campaign_id} stopped INDETERMINATE at "
                f"exit 0: driver state {status or 'absent'} is neither a verified "
                "completion, an explicit durable pause, nor a verified wake handoff, so "
                "this is a silent campaign death and not a clean stop"
            ),
        )
        return self._stop("indeterminate-stop", EXIT_ERROR)

    def _settle_lock_contention(self) -> Optional[int]:
        """Back off one bounded driver-lock refusal, or give up and page the holder.

        Returns
        -------
        int | None
            Supervisor exit status once the bound is exhausted, else ``None`` to retry.
        """

        self._lock_refusals += 1
        if self._lock_refusals >= LOCK_CONTENTION_MAX_ATTEMPTS:
            holder = _lock_holder_summary(_driver_lock_path(self.config.repo_root))
            waited = sum(
                _lock_backoff_seconds(attempt) for attempt in range(1, self._lock_refusals)
            )
            self._notify_until_delivered(
                "lock-contention",
                (
                    f"Menagerie crawler {self.config.campaign_id} gave up after "
                    f"{self._lock_refusals} driver lock refusals over {waited:.0f}s of "
                    f"backoff; the lock holder is {holder}"
                ),
            )
            return self._stop("lock-contention", EXIT_ERROR)
        self._sleep(_lock_backoff_seconds(self._lock_refusals))
        return None

    def _settle_crash(self, returncode: int) -> Optional[int]:
        """Account one unexpected driver exit, page its class, and test both circuits.

        Parameters
        ----------
        returncode:
            Unexpected driver exit status.

        Returns
        -------
        int | None
            Supervisor exit status once a circuit opens, else ``None`` to restart.
        """

        crash = self._record_crash()
        # OPS-06: an ordinary crash pages IMMEDIATELY. Step-100's four crashes spanned
        # ~55h, the windowed counter never reached five, so no notification was
        # possible at all and an operator found them by reading logs. Deduplication is
        # by CLASS -- campaign, exit reason, progress fingerprint -- so a repeating
        # class pages once and a NEW class pages again; silence is never deduplicated.
        self._notify_alert_class(
            "driver-crash",
            {
                "campaign_id": self.config.campaign_id,
                "event": "driver-crash",
                "exit_reason": returncode,
                "progress_fingerprint": crash.progress_fingerprint,
            },
            (
                f"Menagerie crawler {self.config.campaign_id} driver crash: exit "
                f"{returncode}, restart {len(crash.window)} in the burst window, "
                f"{crash.consecutive_flat_restarts} consecutive restarts with no "
                "durable progress"
            ),
        )
        if crash.consecutive_flat_restarts >= CONSECUTIVE_FLAT_RESTART_LIMIT:
            self._notify_until_delivered(
                "flat-restart-circuit-open",
                (
                    f"Menagerie crawler {self.config.campaign_id} circuit opened after "
                    f"{crash.consecutive_flat_restarts} consecutive restarts that made "
                    f"no durable progress (exit {returncode}); the campaign is flailing "
                    "at full spend and will not restart again"
                ),
            )
            return self._stop("flat-restart-circuit-open", EXIT_ERROR)
        if len(crash.window) >= CRASH_LIMIT:
            self._notify_until_delivered(
                "crash-circuit-open",
                (
                    f"Menagerie crawler {self.config.campaign_id} circuit opened after "
                    f"{len(crash.window)} unexpected exits in 30 minutes"
                ),
            )
            return self._stop("crash-circuit-open", EXIT_ERROR)
        # W6: the bounded, attested restart ladder. Restart ONLY when the
        # dead leg's typed exit class is restartable, no StopClass holds,
        # and the per-episode ordinal bound is unspent -- everything else
        # pages and stops instead of grinding an identical death.
        refusal = self._restart_ladder_refusal(returncode)
        if refusal is not None:
            self._notify_until_delivered(
                "restart-ladder-refused",
                (
                    f"Menagerie crawler {self.config.campaign_id} restart refused "
                    f"(exit {returncode}): {refusal}"
                ),
            )
            return self._stop("restart-ladder-refused", EXIT_ERROR)
        self._restart_ordinal = getattr(self, "_restart_ordinal", 0) + 1
        self._sleep(_restart_backoff_seconds(len(crash.window)))
        return None

    def _leg_events_count(self) -> int:
        """Return the durable leg-event row count, ``-1`` when unreadable."""

        from menagerie.crawler.leg_generation import load_leg_events

        try:
            return len(load_leg_events(_models_ledger_root(self.config).parent))
        except Exception:  # noqa: BLE001 -- absent ledger reads as empty history
            return -1

    def _restart_ladder_refusal(self, returncode: int) -> Optional[str]:
        """Adjudicate one crash against the W6 restart-admission law.

        Returns
        -------
        str | None
            The typed refusal detail, or ``None`` when the ladder may
            restart (bounded, attested).
        """

        from menagerie.crawler.leg_exit_vocabulary import (
            RESTARTABLE_EXIT_CLASSES,
            evaluate_stop_classes,
        )
        from menagerie.crawler.leg_generation import load_leg_events

        ordinal = getattr(self, "_restart_ordinal", 0) + 1
        if ordinal > SUPERVISED_RESTART_ORDINAL_LIMIT:
            return (
                f"the supervised restart ladder is exhausted (ordinal {ordinal} > "
                f"{SUPERVISED_RESTART_ORDINAL_LIMIT})"
            )
        records_root = _models_ledger_root(self.config).parent
        try:
            events = load_leg_events(records_root)
        except Exception:  # noqa: BLE001 -- no history: bounded restart proceeds
            events = ()
        baseline = getattr(self, "_leg_events_baseline", -1)
        fresh = events[baseline:] if 0 <= baseline <= len(events) else events
        ended = [row for row in fresh if row.get("event") == "leg_ended"]
        if ended:
            exit_class = str(ended[-1].get("exit_class", ""))
            if exit_class not in RESTARTABLE_EXIT_CLASSES:
                return (
                    f"the dead leg's typed exit class {exit_class!r} is not "
                    "restartable (paused/close/operator exits own their own "
                    "continuation)"
                )
        findings = [
            finding
            for finding in evaluate_stop_classes(
                records_root,
                self.config.repo_root,
                now=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            # The driver's own leg-open refusal adjudicates absent secrets
            # CONTEXTUALLY (only receipt-configured production legs demand
            # them, W15); the ladder deferring to it avoids refusing every
            # restart on a host whose supervisor shell simply lacks the
            # exports the driver would not require.
            if finding["stop_class"] != "secrets_absent_at_wake"
        ]
        if findings:
            summary = "; ".join(
                f"{finding['stop_class']}: {finding['detail']}" for finding in findings
            )
            return f"standing StopClass refusals: {summary}"
        return None

    def _stop(self, event: str, status: int) -> int:
        """Record the terminal heartbeat and honor any undelivered alert.

        Parameters
        ----------
        event:
            Terminal event identity, recorded as the heartbeat status.
        status:
            Exit status this disposition earns when every alert was delivered.

        Returns
        -------
        int
            ``status``, or ``EXIT_ERROR`` when an operator alert exhausted its bound:
            an unreported stop is never allowed to read as a clean one (OPS-05).
        """

        self._write_heartbeat(event)
        if self._undelivered:
            return EXIT_ERROR
        return status

    def _check_queue_stalls(self) -> None:
        """Notify once for each queue job crossing the 45-minute deadline."""

        queue_root = self.config.author_queue_root
        if queue_root is None:
            return
        notified = set(self._state.notified_stalls)
        changed = False
        for stall in stalled_queue_jobs(
            queue_root,
            now=self._now(),
            stall_seconds=self.config.stall_seconds,
        ):
            identity = stable_hash(
                {
                    "campaign_id": self.config.campaign_id,
                    "job_id": stall.job_id,
                    "attempt_nonce": stall.attempt_nonce,
                }
            )
            if identity in notified:
                continue
            delivered = self.notifier.notify(
                (
                    f"Menagerie crawler {self.config.campaign_id} RETRYABLE INFRASTRUCTURE: "
                    f"author queue stalled >45 min for {stall.stable_id} "
                    f"({'claimed' if stall.claimed else 'unclaimed'})"
                ),
                idempotency_key=identity,
                notification_class=NotificationClass.STALL.value,
            )
            if delivered:
                notified.add(identity)
                changed = True
                if stall.marker_path is not None:
                    stall.marker_path.unlink(missing_ok=True)
        if changed:
            self._state = replace(self._state, notified_stalls=tuple(sorted(notified)))
            _write_state(self._state_path, self._state)

    def _record_crash(self) -> CrashAccounting:
        """Account one unexpected driver exit in the window AND the flat-restart counter.

        Returns
        -------
        CrashAccounting
            Both projections plus the durable-progress fingerprint that decided the
            flat counter. Only durable progress resets it, so the counter is a fact
            about the campaign's forward motion and not about wall-clock cadence
            (SCHED-04).
        """

        now = self._now()
        cutoff = now - timedelta(seconds=CRASH_WINDOW_SECONDS)
        recent = [
            timestamp
            for timestamp in self._state.crash_timestamps
            if _parse_timestamp(timestamp) >= cutoff
        ]
        recent.append(_format_timestamp(now))
        fingerprint = self._progress_fingerprint()
        # An empty persisted fingerprint means no crash has been accounted yet, which
        # is UNKNOWN progress, never flat: the first crash after a fresh supervisor
        # start only establishes the baseline the next one is measured against.
        progressed = (
            not self._state.progress_fingerprint
            or fingerprint != self._state.progress_fingerprint
        )
        flat = 0 if progressed else self._state.consecutive_flat_restarts + 1
        self._state = replace(
            self._state,
            crash_timestamps=tuple(recent),
            consecutive_flat_restarts=flat,
            progress_fingerprint=fingerprint,
        )
        _write_state(self._state_path, self._state)
        return CrashAccounting(
            window=tuple(recent),
            consecutive_flat_restarts=flat,
            progress_fingerprint=fingerprint,
        )

    def _progress_fingerprint(self) -> str:
        """Digest every durable progress witness this campaign can present.

        Durable progress is what SURVIVED the crashed leg: the driver's persisted
        cursor (which carries ``last_terminal_count``) and the append-only model
        ledger's extents. A moving fingerprint is progress; an identical one across a
        restart is a flat restart. Absence contributes an explicit ``None`` rather
        than a zero, so "nothing recorded" can never read as "nothing progressed".

        Returns
        -------
        str
            Stable digest over the durable progress witnesses.
        """

        runtime_root = self.config.repo_root / ".crawl-local"
        return stable_hash(
            {
                "driver_state": _file_digest(runtime_root / "driver-state.json"),
                "model_ledger": [
                    list(extent) for extent in _ledger_extents(_models_ledger_root(self.config))
                ],
            }
        )

    def _notify_until_delivered(self, event: str, summary: str) -> bool:
        """Deliver one terminal alert under a bound, then escalate durably.

        Parameters
        ----------
        event:
            Alert event identity.
        summary:
            Operator message.

        Returns
        -------
        bool
            Whether the notifier produced its nonce receipt.
        """

        return self._notify_bounded(
            event,
            stable_hash(
                {
                    "campaign_id": self.config.campaign_id,
                    "event": event,
                    "wake_episode_id": self.config.wake_episode_id,
                }
            ),
            summary,
        )

    def _notify_alert_class(
        self,
        event: str,
        identity_payload: Mapping[str, Any],
        summary: str,
    ) -> bool:
        """Page one alert CLASS at most once, deduplicating on delivery only.

        Parameters
        ----------
        event:
            Alert event identity.
        identity_payload:
            The class this alert belongs to: campaign, event, and the exact facts that
            make two occurrences the same problem.
        summary:
            Operator message.

        Returns
        -------
        bool
            Whether this class is now delivered, either just now or previously.
        """

        identity = stable_hash(dict(identity_payload))
        if identity in self._state.notified_alert_classes:
            return True
        if not self._notify_bounded(event, identity, summary):
            # Deduplicating here would deduplicate on SILENCE, which is exactly how
            # the round-1 stack reported nothing at all.
            return False
        self._state = replace(
            self._state,
            notified_alert_classes=tuple(
                sorted({*self._state.notified_alert_classes, identity})
            ),
        )
        _write_state(self._state_path, self._state)
        return True

    def _notify_bounded(self, event: str, identity: str, summary: str) -> bool:
        """Attempt one idempotent alert at most ``NOTIFICATION_MAX_ATTEMPTS`` times.

        Parameters
        ----------
        event:
            Alert event identity.
        identity:
            Idempotency key the transport receipt must echo.
        summary:
            Operator message.

        Returns
        -------
        bool
            Whether the notifier produced its nonce receipt within the bound.
        """

        notification_class = classify_notification_event(event)
        for attempt in range(1, NOTIFICATION_MAX_ATTEMPTS + 1):
            if self.notifier.notify(
                summary,
                idempotency_key=identity,
                notification_class=(
                    None if notification_class is None else notification_class.value
                ),
            ):
                return True
            if attempt < NOTIFICATION_MAX_ATTEMPTS:
                self._sleep(self.config.notification_retry_seconds)
        self._record_undelivered(event, summary, NOTIFICATION_MAX_ATTEMPTS)
        return False

    def _record_undelivered(self, event: str, summary: str, attempts: int) -> None:
        """Escalate one exhausted alert onto the independent durable channel.

        OPS-01/OPS-05: the primary transport is the thing that failed, so the
        escalation may not use it. A JSON drop an independent poller can read, plus a
        loud marker naming the event, plus a non-zero process exit are three
        mechanisms that do not share the transport's failure mode.

        Parameters
        ----------
        event:
            Alert event identity that could not be delivered.
        summary:
            Operator message that never arrived.
        attempts:
            Exact number of delivery attempts made.
        """

        self._undelivered.append(event)
        drop = {
            "format": ALERT_DROP_FORMAT,
            "campaign_id": self.config.campaign_id,
            "event": event,
            "attempts": attempts,
            "summary": summary,
            "observed_at": _format_timestamp(self._now()),
            "wake_episode_id": self.config.wake_episode_id,
        }
        self._alerts_root.mkdir(parents=True, exist_ok=True)
        digest = stable_hash(drop).removeprefix("sha256:")[:16]
        atomic_replace_bytes(
            self._alerts_root / f"{event}.{digest}.json",
            canonical_json_bytes(drop) + b"\n",
        )
        marker = "\n".join(
            [
                f"menagerie crawler {self.config.campaign_id}: operator notification FAILED",
                f"undelivered events: {', '.join(self._undelivered)}",
                f"attempts per event: {attempts}",
                f"durable drops: {self._alerts_root}",
                "",
            ]
        )
        atomic_replace_bytes(self._marker_path, marker.encode("utf-8"))

    def _write_heartbeat_if_due(self, status: str) -> None:
        """Rewrite the level-triggered heartbeat once its declared floor elapses.

        Parameters
        ----------
        status:
            Current supervisor disposition.
        """

        if self._last_heartbeat is None:
            self._write_heartbeat(status)
            return
        elapsed = (self._now() - self._last_heartbeat).total_seconds()
        if elapsed >= HEARTBEAT_SECONDS:
            self._write_heartbeat(status)

    def _write_heartbeat(self, status: str) -> None:
        """Write the level-triggered liveness file an independent poller reads.

        OPS-01: milestone pushes are count-triggered with 5000-model gaps, so the
        maximum designed silence window was 35-159 days. This file is rewritten at
        least every :data:`HEARTBEAT_SECONDS`, which makes its ABSENCE the death
        signal -- the only shape that reports a failure the campaign cannot describe.

        BL-6 (f): every DERIVED quantity that was not measured renders
        :data:`~menagerie.crawler.wakeup.UNMEASURED`, never ``0``. Only declared
        configuration and observed identity may be numbers or strings here.

        Parameters
        ----------
        status:
            Current supervisor disposition.
        """

        now = self._now()
        payload: dict[str, object] = {
            "format": HEARTBEAT_FORMAT,
            "campaign_id": self.config.campaign_id,
            "status": status,
            "observed_at": _format_timestamp(now),
            "heartbeat_seconds": HEARTBEAT_SECONDS,
            "progress_fingerprint": self._progress_fingerprint(),
            "models_completed": self._models_completed(),
            "newest_event_age_seconds": self._newest_event_age_seconds(now),
        }
        atomic_replace_bytes(self._heartbeat_path, canonical_json_bytes(payload) + b"\n")
        self._last_heartbeat = now

    def _models_completed(self) -> object:
        """Return the driver's persisted terminal count, or ``UNMEASURED``.

        Returns
        -------
        object
            Recorded terminal count, or :data:`~menagerie.crawler.wakeup.UNMEASURED`
            when no cursor recorded one. An absent cursor is not zero models.
        """

        value = _driver_state_field(self.config.repo_root / ".crawl-local", "last_terminal_count")
        if isinstance(value, bool) or not isinstance(value, int):
            return UNMEASURED
        return value

    def _newest_event_age_seconds(self, now: datetime) -> object:
        """Return the age of the newest durable operational event, or ``UNMEASURED``.

        Parameters
        ----------
        now:
            Current aware UTC instant.

        Returns
        -------
        object
            Whole seconds since the newest recorded operational event, or
            :data:`~menagerie.crawler.wakeup.UNMEASURED` when no event ledger, no
            readable tail record, or no parseable instant exists. A campaign nothing
            looked at never reports a fresh zero-second age.
        """

        created_at = _newest_operational_event_instant(_operational_ledger_path(self.config))
        if created_at is None:
            return UNMEASURED
        return max(0, int((now - created_at).total_seconds()))


def stalled_queue_jobs(
    queue_root: Path,
    *,
    now: datetime,
    stall_seconds: float = float(AUTHOR_QUEUE_STALL_SECONDS),
) -> tuple[QueueStall, ...]:
    """Return pending queue jobs older than the retryable-infrastructure deadline.

    Parameters
    ----------
    queue_root:
        Campaign-local author queue.
    now:
        Current aware UTC instant.
    stall_seconds:
        Queue latency threshold.

    Returns
    -------
    tuple[QueueStall, ...]
        Deterministically ordered stalled jobs. Malformed jobs are left to the lane's
        fail-closed protocol validation and are not guessed here.
    """

    if now.tzinfo is None:
        raise ValueError("queue watchdog requires an aware timestamp")
    stalls: list[QueueStall] = []
    marked_attempts: set[tuple[str, str]] = set()
    for path in sorted((queue_root / "watchdog").glob("*.stall.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("format") != "menagerie.crawler.author-queue-stall.v1":
                continue
            job_id = str(value["job_id"])
            attempt_nonce = str(value["attempt_nonce"])
            stable_id = str(value["stable_id"])
            raw_claimed = value["claimed"]
            if not isinstance(raw_claimed, bool):
                continue
            enqueued_at = _parse_timestamp(str(value["enqueued_at"]))
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        age = max(stall_seconds, (now - enqueued_at).total_seconds())
        stalls.append(
            QueueStall(job_id, attempt_nonce, stable_id, age, raw_claimed, marker_path=path)
        )
        marked_attempts.add((job_id, attempt_nonce))
    for path in sorted((queue_root / "pending").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            enqueued_at = _parse_timestamp(str(value["enqueued_at"]))
            job_id = str(value["job_id"])
            attempt_nonce = str(value["attempt_nonce"])
            stable_id = str(value["stable_id"])
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        if (job_id, attempt_nonce) in marked_attempts:
            continue
        age = max(0.0, (now - enqueued_at).total_seconds())
        if age <= stall_seconds:
            continue
        claim_path = queue_root / "claimed" / f"{job_id}.json"
        stalls.append(QueueStall(job_id, attempt_nonce, stable_id, age, claim_path.is_file()))
    return tuple(sorted(stalls, key=lambda stall: (stall.job_id, stall.attempt_nonce)))


def validate_supervised_campaign(
    repo_root: Path,
    campaign_id: str,
    persisted: CampaignConfig,
) -> None:
    """Validate checkpoint policy before launchd is allowed to run a campaign.

    Parameters
    ----------
    repo_root, campaign_id, persisted:
        Repository, requested campaign label, and mode-0600 campaign config.

    Raises
    ------
    SupervisorError
        If a production binding could lose C1's review gate, impose it on C2-C4,
        or reach the checkpoint without 900/950/1000 outbox identities.
    """

    snapshot = load_intake_snapshot(persisted.intake_root)
    binding = find_campaign_binding(repo_root / DEFAULT_CAMPAIGN_MANIFEST, snapshot)
    if binding is None:
        return
    if binding.spec.campaign_id != campaign_id:
        raise SupervisorError(
            f"campaign config belongs to {binding.spec.campaign_id}, not {campaign_id}"
        )
    if persisted.review_checkpoint_at != binding.spec.review_checkpoint_at:
        raise SupervisorError(
            f"{campaign_id} review checkpoint must be {binding.spec.review_checkpoint_at}"
        )
    if campaign_id == "c1-mech" and not {900, 950, 1000}.issubset(persisted.progress_milestones):
        raise SupervisorError("c1-mech must notify at 900, 950, and 1000")


def render_launchd_plist(
    *,
    campaign_id: str,
    repo_root: Path,
    campaign_config_path: Path,
    author_queue_root: Optional[Path],
    python_executable: Path,
    author_concurrency: int = DEFAULT_AUTHOR_WAVE_CONCURRENCY,
) -> bytes:
    """Render one concrete per-campaign launchd agent.

    Parameters
    ----------
    campaign_id, repo_root, campaign_config_path, author_queue_root, python_executable:
        Exact launch inputs embedded as argv leaves without shell interpolation.

    Returns
    -------
    bytes
        Valid XML plist with unsuccessful-exit keep-alive for supervisor death only.
    """

    script = repo_root / "menagerie" / "crawler" / "tools" / "crawler_supervisor.sh"
    arguments = [
        str(script),
        "--python",
        str(python_executable),
        "run",
        "--campaign-id",
        campaign_id,
        "--repo-root",
        str(repo_root),
        "--campaign-config",
        str(campaign_config_path),
    ]
    if author_queue_root is not None:
        arguments.extend(("--author-queue", str(author_queue_root)))
    arguments.extend(("--author-concurrency", str(author_concurrency)))
    payload = {
        "Label": f"org.menagerie.crawler.{campaign_id}",
        "ProgramArguments": arguments,
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Background",
        "ThrottleInterval": 30,
        "StandardOutPath": str(repo_root / ".crawl-local" / "supervisor" / "launchd.out.log"),
        "StandardErrorPath": str(repo_root / ".crawl-local" / "supervisor" / "launchd.err.log"),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the supervisor and launchd-rendering CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(prog="python -m menagerie.crawler.supervisor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "render-launchd"):
        command = subparsers.add_parser(name)
        command.add_argument("--campaign-id", required=True)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--campaign-config", type=Path, required=True)
        command.add_argument("--author-queue", type=Path)
        command.add_argument(
            "--author-concurrency", type=int, default=DEFAULT_AUTHOR_WAVE_CONCURRENCY
        )
    # RC-1 (D.1 behavior 6): the continuous-crawl shape. No frozen campaign
    # config is required to supervise: the driver resolves it from the bound
    # campaign root on its first launch and reuses it afterwards.
    crawl = subparsers.add_parser("crawl")
    crawl.add_argument("--campaign-id", required=True)
    crawl.add_argument("--repo-root", type=Path, required=True)
    crawl.add_argument("--records-root", type=Path, required=True)
    crawl.add_argument("--limit", type=int)
    crawl.add_argument("--author-width", type=int, default=DEFAULT_AUTHOR_WAVE_CONCURRENCY)
    crawl.add_argument("--checker-concurrency", type=int)
    crawl.add_argument("--target", default="osx-arm64")
    crawl.add_argument("--environment-readiness-receipt", type=Path)
    crawl.add_argument("--author-queue", type=Path)
    crawl.add_argument(
        "--render-launchd",
        type=Path,
        help="write the keep-alive plist for this crawl shape instead of supervising",
    )
    crawl.add_argument("--python", type=Path, default=Path(sys.executable))
    run = subparsers.choices["run"]
    run.add_argument("--wake-episode-id")
    # WS-1 continuation completeness: the wake callback forwards the COMPLETE
    # launching selection config through the supervisor into the driver
    # resume; absent flags historically meant an unscoped default-scheduler
    # resume. Typed and forwarded verbatim, never interpreted here.
    run.add_argument("--target")
    run.add_argument("--phase")
    run.add_argument("--scheduler")
    run.add_argument("--only-unsettled", action="store_true")
    run.add_argument("--redrive-targets")
    run.add_argument("--redrive-reason")
    run.add_argument("--redrive-manifest")
    run.add_argument("--gate-catchup-targets")
    run.add_argument("--leg-purpose")
    run.add_argument("--ack-stale-excluded", action="store_true")
    run.add_argument("--acknowledge-award-holds", action="store_true")
    # D-47: the digest companions binding the census acknowledgments ride
    # the continuation verbatim like every other vocabulary flag.
    run.add_argument("--award-holds-digest")
    run.add_argument("--stale-excluded-digest")
    run.add_argument("--checker-concurrency", type=int)
    run.add_argument("--continuation-identity")
    # W15 (adjudicated D1): the readiness receipt is part of the continuation
    # contract; a supervisor parser without it strands every supervised wake
    # at the CLI readiness gate.
    run.add_argument("--environment-readiness-receipt")
    render = subparsers.choices["render-launchd"]
    render.add_argument("--python", type=Path, default=Path(sys.executable))
    render.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the keep-alive or render a concrete launchd plist.

    Parameters
    ----------
    argv:
        Optional CLI arguments.

    Returns
    -------
    int
        Stable supervisor command status.
    """

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repo_root = args.repo_root.resolve()
    if args.command == "crawl":
        return _crawl_main(args, repo_root)
    config_path = args.campaign_config.resolve()
    persisted = load_campaign_config(config_path)
    author_queue = (
        args.author_queue.resolve()
        if isinstance(args.author_queue, Path)
        else persisted.author_queue_root
    )
    validate_supervised_campaign(repo_root, args.campaign_id, persisted)
    if args.command == "render-launchd":
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_bytes(
            output,
            render_launchd_plist(
                campaign_id=args.campaign_id,
                repo_root=repo_root,
                campaign_config_path=config_path,
                author_queue_root=author_queue,
                python_executable=args.python.resolve(),
                author_concurrency=args.author_concurrency,
            ),
        )
        return EXIT_OK
    runtime_root = repo_root / ".crawl-local" / "supervisor" / args.campaign_id
    notifier = CommandNotifier(
        (None if persisted.notify_command is None else shlex.join(persisted.notify_command)),
        receipt_root=runtime_root / "notification-receipts",
    )
    # W15: the forwarded contract derives from the ONE continuation
    # vocabulary (menagerie.crawler.continuation_manifest), so a flag the
    # driver's callback renders can never be silently dropped here -- the
    # hand-enumerated forwarding list is the class that dropped the readiness
    # receipt (adjudicated D1). Typed and forwarded verbatim, never
    # interpreted. ``--target`` stays supervisor-owned (it is not a
    # continuation flag; both callback branches carry it independently).
    selection_argv: list[str] = []
    if getattr(args, "target", None):
        selection_argv.extend(("--target", args.target))
    selection_argv.extend(forwarded_continuation_argv(args))
    supervisor = CrawlerSupervisor(
        SupervisorConfig(
            campaign_id=args.campaign_id,
            repo_root=repo_root,
            campaign_config_path=config_path,
            author_queue_root=author_queue,
            runtime_root=runtime_root,
            author_concurrency=args.author_concurrency,
            wake_episode_id=args.wake_episode_id,
            selection_argv=tuple(selection_argv),
        ),
        notifier,
    )
    return supervisor.run()


def _crawl_main(args: argparse.Namespace, repo_root: Path) -> int:
    """Supervise (or render the plist for) one continuous-crawl shape (RC-1)."""

    records_root = args.records_root.resolve()
    if not (records_root / "campaigns.json").is_file():
        raise SupervisorError(f"records root has no campaigns.json: {records_root}")
    crawl_flags: list[str] = ["--records-root", str(records_root), "--campaign", args.campaign_id]
    if args.limit is not None:
        crawl_flags.extend(("--limit", str(args.limit)))
    crawl_flags.extend(("--author-width", str(args.author_width), "--target", args.target))
    if args.checker_concurrency is not None:
        crawl_flags.extend(("--checker-concurrency", str(args.checker_concurrency)))
    if args.environment_readiness_receipt is not None:
        crawl_flags.extend(
            ("--environment-readiness-receipt", str(args.environment_readiness_receipt.resolve()))
        )
    author_queue = args.author_queue.resolve() if isinstance(args.author_queue, Path) else None
    if author_queue is not None:
        crawl_flags.extend(("--author-queue", str(author_queue)))
    if args.render_launchd is not None:
        output = args.render_launchd.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_bytes(
            output,
            render_crawl_launchd_plist(
                campaign_id=args.campaign_id,
                repo_root=repo_root,
                python_executable=args.python.resolve(),
                crawl_flags=tuple(crawl_flags),
            ),
        )
        return EXIT_OK
    runtime_root = repo_root / ".crawl-local" / "supervisor" / args.campaign_id
    notifier = CommandNotifier(None, receipt_root=runtime_root / "notification-receipts")
    supervisor = CrawlerSupervisor(
        SupervisorConfig(
            campaign_id=args.campaign_id,
            repo_root=repo_root,
            # No frozen config is consulted by the crawl shape; the driver owns
            # it. The field stays absolute for the config invariant.
            campaign_config_path=repo_root / ".crawl-local" / "campaign-configs",
            author_queue_root=author_queue,
            runtime_root=runtime_root,
            author_concurrency=int(args.author_width),
            selection_argv=tuple(crawl_flags),
            driver_shape="crawl",
            records_root=records_root,
            crawl_limit=args.limit,
            target=args.target,
        ),
        notifier,
    )
    return supervisor.run()


def render_crawl_launchd_plist(
    *,
    campaign_id: str,
    repo_root: Path,
    python_executable: Path,
    crawl_flags: Sequence[str],
) -> bytes:
    """Render the keep-alive plist for the crawl shape (RC-1 behavior 6)."""

    script = repo_root / "menagerie" / "crawler" / "tools" / "crawler_supervisor.sh"
    arguments = [
        str(script),
        "--python",
        str(python_executable),
        "crawl",
        "--campaign-id",
        campaign_id,
        "--repo-root",
        str(repo_root),
        *crawl_flags,
    ]
    payload = {
        "Label": f"org.menagerie.crawler.{campaign_id}",
        "ProgramArguments": arguments,
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Background",
        "ThrottleInterval": 30,
        "StandardOutPath": str(repo_root / ".crawl-local" / "supervisor" / "launchd.out.log"),
        "StandardErrorPath": str(repo_root / ".crawl-local" / "supervisor" / "launchd.err.log"),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def _driver_command(
    config: SupervisorConfig, *, restart_ordinal: int = 0
) -> tuple[str, ...]:
    """Return the clean-environment driver resume argv.

    W6 single-continuation authority: the wake episode continues a generation
    exactly ONCE (the first invocation of a wake-launched supervision
    episode). Every later iteration is the bounded restart LADDER's own
    continuation -- it drops the consumed ``--wake-episode-id`` and attests
    itself with ``--supervised-restart-ordinal`` instead, so the two
    authorities can never both continue one generation (the historical argv
    re-appended the episode id on every restart, re-entering ``handle_fire``
    for the same episode).
    """

    if config.driver_shape == "crawl":
        # RC-1 (behavior 6): the crawl shape. ``selection_argv`` carries the
        # complete crawl flag set verbatim (records root, campaign, limit,
        # width, target); the restart ladder attests its ordinal the same way.
        command = [
            sys.executable,
            "-m",
            "menagerie.crawler",
            "--repo-root",
            str(config.repo_root),
            "crawl",
            *config.selection_argv,
        ]
        if restart_ordinal > 0:
            command.extend(("--supervised-restart-ordinal", str(restart_ordinal)))
        return tuple(command)
    command = [
        sys.executable,
        "-m",
        "menagerie.crawler",
        "--repo-root",
        str(config.repo_root),
        "resume",
        "--campaign-config",
        str(config.campaign_config_path),
    ]
    if config.author_queue_root is not None:
        command.extend(("--author-queue", str(config.author_queue_root)))
    command.extend(("--author-concurrency", str(config.author_concurrency)))
    command.extend(config.selection_argv)
    if restart_ordinal > 0:
        command.extend(("--supervised-restart-ordinal", str(restart_ordinal)))
    elif config.wake_episode_id is not None:
        command.extend(("--wake-episode-id", config.wake_episode_id))
    return tuple(command)


def _start_process(
    command: Sequence[str],
    cwd: Path,
    stdout: BinaryIO,
    stderr: BinaryIO,
) -> ManagedProcess:
    """Start one non-shell driver child.

    Parameters
    ----------
    command, cwd, stdout, stderr:
        Exact argv, working directory, and append-only logs.

    Returns
    -------
    ManagedProcess
        Pollable driver subprocess.
    """

    return subprocess.Popen(
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        shell=False,
        start_new_session=False,
    )


def _load_state(path: Path) -> SupervisorState:
    """Load durable supervisor state, defaulting only when absent."""

    if not path.is_file():
        return SupervisorState()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"supervisor state is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise SupervisorError("supervisor state must be an object")
    return SupervisorState.from_mapping(value)


def _write_state(path: Path, state: SupervisorState) -> None:
    """Atomically persist supervisor crash and alert state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(path, canonical_json_bytes(state.to_dict()) + b"\n")


def _driver_state_status(runtime_root: Path) -> Optional[str]:
    """Return the disposable driver status when readable."""

    status = _driver_state_field(runtime_root, "status")
    return status if isinstance(status, str) else None


def _driver_state_field(runtime_root: Path, field: str) -> Optional[object]:
    """Return one field of the disposable driver cursor when readable.

    Parameters
    ----------
    runtime_root:
        Driver runtime root holding ``driver-state.json``.
    field:
        Field name to read.

    Returns
    -------
    object | None
        Recorded value, or ``None`` when the cursor is absent, unreadable, not an
        object, or does not carry the field. Absence is never substituted with a value.
    """

    path = runtime_root / "driver-state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    return value.get(field)


def _declared_stop(returncode: int) -> Optional[tuple[str, str]]:
    """Return the declared-stop event and summary template for one driver exit.

    Parameters
    ----------
    returncode:
        Driver exit status.

    Returns
    -------
    tuple[str, str] | None
        Event identity and summary template, or ``None`` when the exit is not one the
        driver described.
    """

    for status, event, template in DECLARED_STOPS:
        if returncode == status:
            return event, template
    return None


def _driver_lock_path(repo_root: Path) -> Path:
    """Return the driver ownership lockfile the ``EXIT_LOCKED`` refusal names.

    Parameters
    ----------
    repo_root:
        Campaign repository root.

    Returns
    -------
    pathlib.Path
        Driver lock path.
    """

    return repo_root / ".crawl-local" / "locks" / "driver.lock"


def _lock_holder_summary(path: Path) -> str:
    """Render the lock holder identity sitting in the lockfile the refusal names.

    Parameters
    ----------
    path:
        Driver lockfile path.

    Returns
    -------
    str
        ``pid``/``run_id``/``created_at`` rendering, with each absent field shown as
        ``unknown``. OPS-06/OPS-08: the owner payload was always on disk and the
        eternal spinner never printed it, so an operator could not tell a legitimate
        handoff from a dead holder.
    """

    fields: dict[str, object] = {"pid": UNMEASURED, "run_id": UNMEASURED, "created_at": UNMEASURED}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"unreadable at {path}: " + " ".join(f"{key}={item}" for key, item in fields.items())
    if isinstance(value, Mapping):
        for key in fields:
            recorded = value.get(key)
            if isinstance(recorded, (int, str)) and not isinstance(recorded, bool):
                fields[key] = recorded
    return " ".join(f"{key}={item}" for key, item in fields.items())


def _lock_backoff_seconds(refusals: int) -> float:
    """Return the bounded exponential driver-lock backoff.

    Parameters
    ----------
    refusals:
        Consecutive ``EXIT_LOCKED`` refusals, counting from one.

    Returns
    -------
    float
        Delay before the next attempt. The first refusal keeps the historical 30s
        handoff pause; later ones grow to the shared restart ceiling so a stuck lock
        is not polled hot for hours.
    """

    return float(
        min(MAX_RESTART_BACKOFF_SECONDS, LOCK_BACKOFF_SECONDS * (2 ** max(0, refusals - 1)))
    )


def _models_ledger_root(config: SupervisorConfig) -> Path:
    """Return the canonical append-only model ledger directory.

    Parameters
    ----------
    config:
        Supervisor configuration.

    Returns
    -------
    pathlib.Path
        Canonical model ledger directory. The supervisor deliberately does not load
        the mode-0600 campaign config to answer this, so a campaign whose records live
        elsewhere presents no witness here rather than a wrong one.
    """

    return config.repo_root / "menagerie" / "crawler" / "records" / "models"


def _operational_ledger_path(config: SupervisorConfig) -> Path:
    """Return the canonical append-only operational event ledger.

    Parameters
    ----------
    config:
        Supervisor configuration.

    Returns
    -------
    pathlib.Path
        Canonical operational event ledger path.
    """

    return _models_ledger_root(config).parent / "operational" / "events.jsonl"


def _file_digest(path: Path) -> Optional[str]:
    """Return a content digest for one durable file, or ``None`` when absent.

    Parameters
    ----------
    path:
        File to digest.

    Returns
    -------
    str | None
        Hex digest, or ``None`` when the file is absent or unreadable.
    """

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _ledger_extents(root: Path) -> tuple[tuple[str, int], ...]:
    """Return deterministic ``(name, size)`` extents for one append-only ledger tree.

    Parameters
    ----------
    root:
        Ledger directory.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Sorted shard extents, empty when the directory is absent. Sizes are the
        cheapest honest witness that an append-only ledger grew.
    """

    try:
        shards = sorted(root.glob("*.jsonl"))
    except OSError:
        return ()
    extents: list[tuple[str, int]] = []
    for shard in shards:
        try:
            extents.append((shard.name, shard.stat().st_size))
        except OSError:
            continue
    return tuple(extents)


def _newest_operational_event_instant(path: Path) -> Optional[datetime]:
    """Return the newest recorded operational event instant, or ``None``.

    Parameters
    ----------
    path:
        Append-only operational event ledger.

    Returns
    -------
    datetime.datetime | None
        Aware UTC instant of the ledger's last record, or ``None`` when the ledger is
        absent, empty, unreadable, or carries no parseable ``created_at``.
    """

    line = _last_jsonl_line(path)
    if line is None:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    created_at = value.get("created_at") if isinstance(value, Mapping) else None
    if not isinstance(created_at, str):
        return None
    try:
        return _parse_timestamp(created_at)
    except ValueError:
        return None


def _last_jsonl_line(path: Path, *, tail_bytes: int = 65536) -> Optional[str]:
    """Return the last non-empty line of one append-only ledger.

    Parameters
    ----------
    path:
        Ledger path.
    tail_bytes:
        Bounded tail read, so a multi-gigabyte ledger costs one seek. A record longer
        than this window reads as no record rather than as a truncated one.

    Returns
    -------
    str | None
        Last non-empty line, or ``None`` when absent, unreadable, or empty.
    """

    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - tail_bytes))
            chunk = handle.read()
    except OSError:
        return None
    lines = [line for line in chunk.split(b"\n") if line.strip()]
    if not lines:
        return None
    try:
        return lines[-1].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _restart_backoff_seconds(crash_count: int) -> float:
    """Return the bounded exponential restart delay."""

    return float(min(MAX_RESTART_BACKOFF_SECONDS, 30 * (2 ** max(0, crash_count - 1))))


def _parse_timestamp(value: str) -> datetime:
    """Parse one required aware RFC 3339 timestamp."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    """Format one aware instant as canonical UTC."""

    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover - operator entry point
    try:
        raise SystemExit(main())
    except (OSError, SupervisorError, ValueError) as exc:
        print(f"crawler supervisor error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR) from exc
