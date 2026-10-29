"""Typed contracts and configuration for the crawler driver.

This module contains data-only scheduler contracts and lock ownership mechanics.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, TypeAlias

from menagerie.crawler.artifact_transactions import (
    StagedArtifact,
)
from menagerie.crawler.author_dispatch import (
    AuthorBackoffSignal,
    AuthorResult,
    DeferRecommendation,
    ProposedAuthorResult,
)
from menagerie.crawler.authority import (
    AuthorityContext,
    EnvironmentAuthorityCache,
    EnvironmentAuthorityV1,
    EnvironmentExternalTarget,
    ShutdownInterruptionFact,
    WorkerLease,
)
from menagerie.crawler.checker_dispatch import (
    CheckerBackoffSignal,
)
from menagerie.crawler.checkpoint import (
    canonical_operational_ledger_path,
)
from menagerie.crawler.constants import (
    CLAUDE_CLI_VERSION,
    DEFAULT_AUTHOR_WAVE_CONCURRENCY,
    DEFAULT_CHECKER_WAVE_CONCURRENCY,
    DEFAULT_NOTIFY_COMMAND,
    DEFAULT_PROGRESS_MILESTONES,
    DEFAULT_REVIEW_CHECKPOINT_AT,
    FAILURE_REASON_CODES,
    HOST_CAPABILITY_DEFERRAL_STATUS_CODE,
    MAX_AUTHOR_WAVE_CONCURRENCY,
    MAX_CHECKER_WAVE_CONCURRENCY,
    SCHEDULER_MODES,
    TIER_CAMPAIGN_IDS,
    InvocationOrigin,
)
from menagerie.crawler.campaign_config import ONLY_STATUS_VOCABULARY
from menagerie.crawler.envs import (
    EnvironmentIntent,
)
from menagerie.crawler.env_lifecycle import (
    ProbeResult,
)
from menagerie.crawler.identity import (
    durable_fsync,
    stable_hash,
)
from menagerie.crawler.intake import (
    IntakeItem,
    IntakeSnapshot,
)
from menagerie.crawler.invocation_identity import (
    LEGACY_RUN_ID,
    LegacyRunIdentityError,
    mint_default_run_id,
)
from menagerie.crawler.models import JsonObject, LedgerPaths
from menagerie.crawler.recordio import (
    JsonlLedger,
)
from menagerie.crawler.reducer import (
    default_ledger_paths,
)
from menagerie.crawler.routing import (
    IntentRoute,
)
from menagerie.crawler.wakeup import OperationalContext


class DriverError(RuntimeError):
    """Base class for typed driver failures."""


class DriverLockError(DriverError):
    """Raised when another live driver owns the campaign lock."""


class DriverIntegrationError(DriverError):
    """Raised when an injected lane returns incomplete or contradictory facts."""


class LegStartRefused(DriverIntegrationError):
    """A typed leg-start refusal carrying its closed StopClass (W6/PD-11).

    Subclasses :class:`DriverIntegrationError` so every existing handler and
    exit-code mapping is unchanged; the ``stop_class`` field is what lets the
    leg-outcome classifier type the refusal (operator stop vs host fault)
    instead of reading it as an uncontained fault.
    """

    def __init__(self, stop_class: str, message: str) -> None:
        super().__init__(message)
        self.stop_class = stop_class


class VariantRecipeUnsupported(DriverIntegrationError):
    """Raised when a family recipe has no closed mechanical sibling selector."""


class HostCapabilityPreflightError(DriverIntegrationError):
    """Raised when every member of a leg's target set is host-capability deferred.

    A leg with nothing it can possibly run is not a leg. Step-100 launched eight of
    them (~23-28h wall) against a target set whose blocker had already been printed
    about 200 times in a ``recheck_hint``; each one authored, gated and re-drove
    settled members until the no-progress tripwire halted it. Refusing at start, with
    the needs-vs-actual census in the message, is the whole remedy: it costs one line
    and it cannot be mistaken for progress.
    """


class GateBatchUnusableError(DriverIntegrationError):
    """Raised when a checker gate is structurally unusable for EVERY batch member.

    This is the distinct, named whole-batch condition. It is deliberately NOT the
    by-product of one bad member: a single item whose identities went stale raises
    ``StaleGateBindingError`` and belongs to that item alone. Only a gate that
    cannot be read as a per-item result at all -- no item list, no routable body --
    genuinely costs every model in the batch.
    """


class StaleGateBindingError(DriverIntegrationError):
    """Raised when named gate items no longer bind their current proposal.

    A stale binding is a fact about ONE model: the checker answered a question
    about a proposal generation the driver no longer holds. Escalating it to the
    whole batch discards up to twenty innocent models per bad member, and the same
    bad member recurs on the retry. ``stale_ids`` names exactly the models whose
    bindings failed so the caller can terminalize them and let their siblings
    proceed.

    Parameters
    ----------
    stale_ids:
        Stable IDs whose gate item is absent, mis-lineaged, or identity-stale.
    detail:
        Human-readable diagnostic naming the same IDs.
    """

    def __init__(self, stale_ids: Sequence[str], detail: str) -> None:
        """Bind the exact stale model IDs to their shared diagnostic."""

        super().__init__(detail)
        self.stale_ids: tuple[str, ...] = tuple(stale_ids)


class PublicationAuthorizationRefused(DriverError):
    """The reducer refused to authorize one model's artifact BEFORE any commit.

    Raised when ``reducer.authorize_publication`` refuses with a typed exactness
    error (``ArtifactBindingError``/``ReductionError``) -- a stale dependency
    vector, a claim-set disagreement, a gate-lineage mismatch. Nothing durable
    happened: no authorization event was appended and no public byte was written,
    so this is a fact about ONE model's authority state, never the campaign's.
    Rung 11 (2026-08-07): a resume reloaded an author-result cache superseded by
    a later same-work staging; ``authorization dependency vector is stale`` for
    that single model propagated to the CLI and aborted the campaign, stranding
    three siblings. Callers contain it model-scoped; post-commit publication
    failures keep their own types and their own resume protocol.

    Parameters
    ----------
    cause:
        The reducer's typed refusal, preserved for the operational record.
    """

    def __init__(self, cause: BaseException) -> None:
        """Carry the reducer's exact refusal."""

        super().__init__(str(cause))
        self.cause = cause


class AuthorBlockedPrerequisite(DriverError):
    """One author's typed ``BLOCKED`` verdict, as the cause of its own attempt.

    This is deliberately NOT an operational failure and is never raised: it is
    the honest cause carried on the attempt record that a checked ``BLOCKED``
    recommendation produces. A ``failed:<stage>`` terminal must be witnessed by
    a failed attempt at that same stage, and the witness for this arm is the
    author's own named prerequisite -- not a crash, a timeout, or an engine
    fault. Naming the cause explicitly keeps that attempt from having to borrow
    an unrelated exception type and misreport what happened.
    """


class AuthorRepairTerminal(DriverError):
    """One bounded repair generation answered with a typed terminal recommendation.

    This is CONTROL FLOW, not a failure. A repair generation re-runs the author
    against the checker's findings, and the author is entitled to the same three
    typed terminal arms it may return on its first call: ``DEFER``, ``SKIP``, or
    ``BLOCKED``. Re-reading the sources to repair one field is precisely when an
    author discovers that the model cannot be authored at all -- pilot ``m5888``
    was asked to repair ``external_metadata.availability.predecessors`` and came
    back with six grounded excerpts showing the library default is not the
    published architecture.

    Before this class the repair lane had no path for that answer: it raised
    ``DriverIntegrationError``, which the surrounding handler reads as a
    ``protocol-violation`` and terminalizes ``failed:runner``. A well-formed,
    evidence-bound verdict was recorded as the author breaking its contract, and
    the recommendation -- its reason code, prerequisites and evidence -- was
    discarded. Because the agentic authoring stage runs exactly ONCE per model
    across a permanent catalog, that is a permanently wrong record, not a retry.

    Deliberately NOT a ``DriverIntegrationError``: it must not be classified as
    an infrastructure fault (which would retry the author) nor as a lane contract
    breach (which would blame the author). It carries the staged artifact so the
    caller can hand it to the same terminal-disposition gate the first-call arm
    uses.

    Parameters
    ----------
    artifact:
        Staged, identity-bound artifact whose ``author_result`` is the typed
        terminal recommendation.
    gate_kind:
        Repair gate that requested the generation, for the diagnostic only.
    """

    def __init__(self, artifact: "AuthorArtifact", gate_kind: str) -> None:
        """Bind the staged terminal artifact to the repair gate that asked for it."""

        super().__init__(
            f"author answered the bounded {gate_kind} repair with a typed terminal "
            f"recommendation ({type(artifact.author_result).__name__})"
        )
        self.artifact = artifact
        self.gate_kind = gate_kind


class RetryableOperatorError(DriverIntegrationError):
    """Raised when an operator lane fails transiently and must be retried.

    Risk R8: an unrecognized operator failure must never become a permanent model
    failure. This class is the typed, non-string-matched membership test used by
    ``_is_infrastructure_error``.
    """

    #: Executor-authored session failure class, when a structured notice supplied one.
    failure_class: Optional[str] = None
    #: Machine-readable basis for ``failure_class``, when the notice supplied one.
    failure_class_basis: Optional[str] = None
    #: Bounded author content-refusal quote, when the notice supplied one. Externally
    #: controlled text carried as data for the canonical row's additive
    #: ``error.refusal_excerpt`` field; never classification or pause authority.
    refusal_excerpt: Optional[str] = None
    #: Exact checker request digest when a machine status sidecar supplied one.
    request_sha256: Optional[str] = None
    #: Closed raw local infrastructure outcome, when one paid guard session
    #: attributed this failure to machine infrastructure rather than the model.
    paid_session_local_outcome: Optional[str] = None
    #: Sentinel marking ``paid_session_local_outcome`` as infrastructure-scoped
    #: (identity-compared against ``_PAID_SESSION_INFRASTRUCTURE_TOKEN``), never
    #: model-scoped.
    _paid_session_infrastructure_token: Optional[object] = None


class AuthorOutageError(RetryableOperatorError):
    """One model's share of a GLOBAL infrastructure condition, not a model failure.

    The distinguishing property of this class is that the cause is shared: a dead
    research provider and an exhausted forge budget each fail every model that
    follows, so the individual failure stays retryable while the author-wave
    coordinator counts consecutive instances across distinct models and promotes
    only a sustained streak to a campaign pause.

    Subclasses exist so the membership test is typed rather than string-matched,
    and so each outage keeps its own identity, threshold, and remedy: naming the
    wrong cause in a durable record is the defect this whole path guards against.
    """

    #: Model whose attempt observed the outage.
    stable_id: str
    #: Exact bounded diagnostic supplied by the trusted operator protocol.
    detail: str

    def __init__(self, message: str, stable_id: str, detail: str) -> None:
        """Attach the affected model and bounded operator detail."""

        super().__init__(message)
        self.stable_id = stable_id
        self.detail = detail


class ResearchToolsUnavailableError(AuthorOutageError):
    """One model's exact fail-loud research-tool guard outcome.

    The individual outcome stays retryable. The author-wave coordinator counts
    consecutive instances across distinct models and promotes only a sustained
    outage to a campaign pause.
    """

    def __init__(self, stable_id: str, detail: str) -> None:
        """Attach the affected model and bounded operator detail.

        Parameters
        ----------
        stable_id:
            Model whose research session could not reach its required tools.
        detail:
            Exact bounded diagnostic supplied by the trusted operator protocol.
        """

        super().__init__(
            f"research tools unavailable for {stable_id}: {detail}",
            stable_id,
            detail,
        )


class ForgeRateLimitedError(AuthorOutageError):
    """One model's source broker being thrown out by the forge, not a bad reference.

    The individual outcome stays retryable, exactly like
    :class:`ResearchToolsUnavailableError`: the author named a good reference and
    we were simply not allowed to dereference it. The author-wave coordinator
    counts consecutive instances across distinct models and promotes only a
    sustained limit to a campaign pause, because a forge budget is global -- once
    it is gone every later model hits the same wall until it resets.
    """

    def __init__(self, stable_id: str, detail: str) -> None:
        """Attach the affected model and bounded operator detail.

        Parameters
        ----------
        stable_id:
            Model whose source broker was throttled by the forge.
        detail:
            Exact bounded diagnostic supplied by the trusted operator protocol.
        """

        super().__init__(f"forge rate limited for {stable_id}: {detail}", stable_id, detail)


class AuthorQueueStalled(RetryableOperatorError):
    """Raised when the managing author session stops servicing the queue.

    Risk R6: the managing session is a single point of failure. A dead session is
    stalled infrastructure, not a model that failed to be authored.
    """


class AuthorSessionRetryExhausted(RetryableOperatorError):
    """One model's author sessions died without publishing until the retry budget spent.

    Rung 7 (2026-08-05, m7362): a repair-round ``claude -p`` died in 0.6s with rc=1,
    zero tokens, and an empty stderr tail -- twice, 0.85s apart, with no backoff --
    and the resulting ``RetryableOperatorError`` unwound the campaign, stranding five
    models that held published, gated work. The session transcripts held the cause
    ("Not logged in"): a transient host-side credential blip that a zero-backoff
    immediate retry structurally cannot survive and that says nothing about the model.

    This subclass marks the RETRY BUDGET SPENT boundary for exactly the
    no-publication author-session class (``session-crashed`` / ``wall-exceeded`` /
    ``no-result-output`` and spawn failures). It deliberately remains a
    ``RetryableOperatorError``: any handler not consulting the lane-health boundary
    still propagates it campaign-scoped, which is the fail-safe LOUD side. Handlers
    that do consult the boundary may terminalize the single model instead -- but only
    with a contemporaneous same-lane publication witness and passing host checks,
    because a session that died without publishing is otherwise indistinguishable
    from a broken host (BATON trap 3).

    Parameters
    ----------
    stable_id:
        Model whose author sessions died without publishing.
    attempts:
        Total sessions attempted before the budget spent.
    backoff_seconds:
        Waits observed between attempts, so the durable record can prove the
        failures were not sub-second echoes of one instant.
    detail:
        Bounded diagnostic from the final attempt.
    """

    def __init__(
        self,
        stable_id: str,
        *,
        attempts: int,
        backoff_seconds: tuple[float, ...],
        detail: str,
    ) -> None:
        """Attach the affected model and the spent retry ladder's shape."""

        super().__init__(
            f"author sessions for {stable_id} died without publishing "
            f"{attempts} times (backoff {list(backoff_seconds)}): {detail}"
        )
        self.stable_id = stable_id
        self.attempts = attempts
        self.backoff_seconds = backoff_seconds
        self.detail = detail


class AuthorCampaignTeardown(BaseException):
    """Machine-owned control flow for a session killed by its campaign halt.

    This is deliberately outside :class:`Exception`: model-local failure arms
    and retry ladders must never absorb the campaign's own teardown as author
    behavior.  The paid-session settlement boundary still catches
    :class:`BaseException`, records the exact machine outcome, and re-raises.

    Parameters
    ----------
    trigger:
        Driver-owned halt trigger that initiated teardown.
    teardown_record:
        Structured parent teardown record naming the target and signal.
    stderr_tail:
        Bounded, quarantined session diagnostic retained for forensics only.
    """

    def __init__(
        self,
        *,
        trigger: str,
        teardown_record: Mapping[str, Any],
        stderr_tail: str,
    ) -> None:
        """Attach parent-owned teardown authority and bounded diagnostics."""

        super().__init__(f"author session stopped by campaign teardown: {trigger}")
        self.trigger = trigger
        self.teardown_record = dict(teardown_record)
        self.stderr_tail = stderr_tail


class AuthorEffortCapExceeded(DriverIntegrationError):
    """Raised when an author session exceeds its declared effort grant.

    ``PLAN.md`` LP-13.2 makes cap exhaustion ``failed:<actual-stage>`` with a
    stage-valid effort reason: a permanent, model-local outcome rather than a
    retryable operator fault.

    The *actual stage* is carried here because only the raise site knows it. Every
    cap exhaustion used to be recorded as ``failed:source``, which asserts that the
    model's source could not be resolved -- routinely false. A session that blew its
    grant after the controlled fetch froze its manifest had its source resolved,
    fetched, and read; what ran out was budget at the author stage.
    Recording that as a source failure would bury models with perfectly good sources
    under a status that says none exists.

    Parameters
    ----------
    args:
        Standard exception arguments.
    stage:
        Closed :data:`~menagerie.crawler.constants.FAILURE_REASON_CODES` stage that
        was actually in flight. Defaults to ``"source"``, which is correct for a cap
        hit while the author is still naming sources.
    dimension:
        Closed author-effort dimension when ``stage="author"``.
    """

    def __init__(
        self,
        *args: object,
        stage: str = "source",
        dimension: Optional[str] = None,
    ) -> None:
        """Attach the closed failure stage the cap exhaustion actually occurred in."""

        super().__init__(*args)
        reason_code = (
            f"effort-exhausted:{dimension}"
            if stage == "author" and dimension is not None
            else "effort-cap-exhausted"
        )
        if reason_code not in FAILURE_REASON_CODES.get(stage, frozenset()):
            raise ValueError(f"effort-cap exhaustion cannot be attributed to stage {stage!r}")
        self.stage = stage
        self.dimension = dimension
        self.reason_code = reason_code


class DriverPaused(DriverError):
    """Raised internally to unwind one environment after a clean campaign pause."""


class AuthorBackoffError(DriverError):
    """Carries a typed author rate/quota pause out of the author lane.

    Raised instead of returning an artifact so the driver's blanket
    blanket model-local failure arm cannot convert Claude usage
    exhaustion into a permanent model failure.
    """

    def __init__(self, signal: AuthorBackoffSignal) -> None:
        """Attach the typed pause signal.

        Parameters
        ----------
        signal:
            Provider pause carrying reason, reset evidence, and provider.
        """

        super().__init__(f"author lane paused: {signal.reason.value}")
        self.signal = signal


class AuthorUsagePause(DriverError):
    """Unwinds one authoring wave after a recorded author usage pause.

    Distinct from :class:`DriverPaused`, which the run loop maps to the review
    checkpoint. This carries the already-recorded usage-pause reason back to the
    scheduler's ``paused:usage-limit`` return path.
    """

    def __init__(self, reason: str) -> None:
        """Attach the recorded pause reason.

        Parameters
        ----------
        reason:
            Value of the recorded :class:`AuthorPauseReason`.
        """

        super().__init__(reason)
        self.reason = reason


class DriverShutdown(BaseException):
    """Typed internal control flow that cannot become an ordinary model failure."""

    def __init__(self, fact: ShutdownInterruptionFact) -> None:
        """Attach the operational-only interruption fact.

        Parameters
        ----------
        fact:
            Parent-owned shutdown observation, never an attempt or model fact.
        """

        super().__init__(fact.admission_boundary)
        self.fact = fact


#: BL-8 (a): the CLOSED set of exception classes the scheduler's per-member
#: blast-radius containment must always re-raise instead of absorbing into a
#: member quarantine. Two families: campaign-level control flow (pauses,
#: shutdown, retryable infrastructure, provider backoff -- these describe the
#: CAMPAIGN, so parking one member on them would silently convert a pause into
#: a member fault), and malformed-PROGRAM errors (mirrors the driver's
#: ``_PROGRAMMING_ERRORS`` doctrine: a `NameError` recurs identically for every
#: member, so absorbing it converts a fail-fast bug into a campaign-wide silent
#: quarantine sweep). ``KeyError``/``IndexError``/``TypeError``/``ValueError``
#: are deliberately EXCLUDED: raised on a member's own data they ARE facts
#: about that member and belong in its quarantine.
MEMBER_CONTAINMENT_PASSTHROUGH: tuple[type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    AuthorCampaignTeardown,
    DriverShutdown,
    DriverPaused,
    AuthorBackoffError,
    AuthorUsagePause,
    RetryableOperatorError,
    NameError,  # also covers UnboundLocalError
    AttributeError,
    ImportError,
    RecursionError,
    MemoryError,
    SystemError,
)


@dataclass(frozen=True)
class DriverPaths:
    """Canonical and disposable paths used by one driver invocation."""

    runtime_root: Path
    intake_root: Path
    ledgers: LedgerPaths

    @property
    def lock_path(self) -> Path:
        """Return the process-level single-writer lock path."""

        return self.runtime_root / "locks" / "driver.lock"

    @property
    def operational_ledger(self) -> Path:
        """Return the append-only operational event ledger path."""

        return canonical_operational_ledger_path(self.ledgers.models)

    @property
    def paid_session_ledger(self) -> Path:
        """Return the chain-verified paid-session money authority path."""

        from menagerie.crawler.checkpoint import canonical_records_root

        return canonical_records_root(self.ledgers.models) / "operational" / "spend.jsonl"

    @property
    def state_database(self) -> Path:
        """Return the rebuildable SQLite state path."""

        return self.runtime_root / "state.sqlite"

    @property
    def driver_state(self) -> Path:
        """Return the disposable scheduler cursor path."""

        return self.runtime_root / "driver-state.json"

    @property
    def reports_root(self) -> Path:
        """Return the runtime report directory."""

        return self.runtime_root / "reports"

    @property
    def work_root(self) -> Path:
        """Return the durable local work-envelope directory."""

        return self.runtime_root / "work"

    @property
    def environments_root(self) -> Path:
        """Return the disposable environment prefix root."""

        return self.runtime_root / "envs"

    @property
    def requeue_grants(self) -> Path:
        """Return the append-only human requeue grant ledger path."""

        return self.runtime_root / "requeue-grants.jsonl"

    @property
    def worker_lock(self) -> Path:
        """Return the child-held single-execution kernel lock."""

        return self.runtime_root / "locks" / "worker.lock"

    @property
    def worker_lease(self) -> Path:
        """Return the fsynced local worker lease record."""

        return self.runtime_root / "locks" / "worker-lease.json"

    @property
    def wakeup_root(self) -> Path:
        """Return the disposable recurring wake projection root."""

        return self.runtime_root / "wakeup"


#: D-54 blast-radius guard: largest --redrive-targets list acceptable inline.
#: A hand-typed CSV above this arrives ONLY as a digested manifest bound to
#: the intake snapshot (:mod:`menagerie.crawler.redrive_manifest`). 25 keeps
#: routine surgical re-drives one flag while forcing wave-scale lists (the
#: ~69-member machine-artifact class) through the durable, reviewable artifact.
REDRIVE_INLINE_TARGET_CAP = 25


@dataclass(frozen=True)
class DriverConfig:
    """Deterministic scheduler configuration."""

    target: str = "osx-arm64"
    phase: Optional[str] = None
    #: D-22/DX-23 run_id axis rotation: every invocation carries its OWN minted
    #: identity. The retired shared constant ``crawler-run`` is refused for new
    #: configs; historical rows written under it stay readable (era rule).
    run_id: str = field(default_factory=mint_default_run_id)
    machine_id: str = platform.node() or "unknown-machine"
    review_checkpoint_at: Optional[int] = DEFAULT_REVIEW_CHECKPOINT_AT
    progress_milestones: tuple[int, ...] = DEFAULT_PROGRESS_MILESTONES
    notify_command: Optional[str] = DEFAULT_NOTIFY_COMMAND
    author_model: str = "claude-fable-5-1"
    author_version: str = "current"
    claude_cli_version: str = CLAUDE_CLI_VERSION
    #: The campaign's declared checker binding. Every production config is
    #: built by the CLI from the live ``CampaignSpec`` projection (the Fable
    #: primary since the 2026-09-01 seat swap; a tripwire test pins spec ==
    #: seat table). The default is the pre-swap label so a lane constructed
    #: WITHOUT a campaign binding (the in-process test rigs, which stamp this
    #: value on their fake gates) is an UNSEATED primary under the conflict
    #: ledger's pairing rule and keeps the historical same-identity K=2;
    #: production sessions never read this field for their model -- the
    #: command lane seals ``constants.CHECKER_SEATS``.
    checker_model: str = "gpt-5.6-sol"
    checker_version: str = "current"
    only_status: Optional[str] = None
    #: Operator scope cut (2026-08-15): schedule ONLY intake members with no
    #: model record at all, plus explicitly granted re-drives. Settled heads
    #: (awards AND terminals) are never re-opened -- no gate readmission, no
    #: staleness re-drive. The finishing leg that motivated it spent six hours
    #: re-executing ~35 settled models while five never-authored members
    #: starved behind them.
    only_unsettled: bool = False
    #: RC-2 named re-check mode: the targets are the recheck census's
    #: ``needs_recheck`` members -- heads recorded before a measured SOUNDNESS
    #: rotation (or an unmeasured move) on one of their policy axes -- derived
    #: at leg start from the raw ledger heads plus
    #: ``records/control/rotations.jsonl``; the config carries no target list.
    recheck: bool = False
    #: Declared closed leg purpose (:mod:`menagerie.crawler.leg_purpose`), or
    #: ``None`` for an ordinary leg. ``licensing_close`` buys the pre-launch
    #: certificate kind on a successor rung (shape 4) at the price of a
    #: driver-enforced total capability forfeit: the WS-1 plan seals EMPTY,
    #: the terminal-review backlog sweep is skipped, every paid admission
    #: refuses typed, and the close refuses any forfeit violation. The value
    #: is content-addressed into the durable leg-open row and immutable per
    #: generation; the declaration itself licenses nothing.
    leg_purpose: Optional[str] = None
    #: Operator acknowledgment of the leg-start award-holds census (AW-1/AW-2):
    #: a launch with projection-stale runs awards or live undrained
    #: award-supersession parks REFUSES unless this is set, so a policy
    #: rotation can never silently funnel settled awards into the re-drive
    #: treadmill again. The census event is appended either way.
    acknowledge_award_holds: bool = False
    #: D-47: the digest binding the award-holds acknowledgment to the EXACT
    #: census it acknowledges (census_ack.award_holds_ack_digest). A nonempty
    #: census with acknowledge_award_holds set refuses unless this matches:
    #: the blanket acknowledgment is retired.
    award_holds_digest: Optional[str] = None
    #: WS-1 clause 4: named re-drive mode. The enumerated stable IDs ARE the
    #: admission authority for non-``runs`` raw heads (S2-opus correction 1:
    #: without a named lane, non-review-held failure terminals are permanently
    #: unschedulable). Never a default; never reopens an award.
    redrive_targets: tuple[str, ...] = ()
    #: D-54 blast-radius guard: the durable operator rationale for a named
    #: re-drive. REQUIRED whenever ``redrive_targets`` is nonempty -- every
    #: named target buys a fresh work generation, and the "why" is part of the
    #: durable record (leg config event), never tribal knowledge.
    redrive_reason: Optional[str] = None
    #: D-54 blast-radius guard: absolute path of the self-digested re-drive
    #: manifest (:mod:`menagerie.crawler.redrive_manifest`). REQUIRED above
    #: ``REDRIVE_INLINE_TARGET_CAP`` inline targets; the driver re-validates
    #: the manifest against the ACTIVE intake snapshot at leg start.
    redrive_manifest_path: Optional[Path] = None
    #: WS-1 clause 4: named gate catch-up mode. Enumerated pending-metadata
    #: ``runs`` records re-enter for their catch-up gate round only.
    gate_catchup_targets: tuple[str, ...] = ()
    #: WS-1 clause 7: explicit operator acknowledgment that this ordinary leg
    #: may start while settled-but-dependency-stale heads exist (they are
    #: EXCLUDED and censused, never silently re-driven). Without it an
    #: ordinary leg refuses to start over a nonzero stale census.
    ack_stale_excluded: bool = False
    #: D-47: the digest binding the stale-excluded acknowledgment to the
    #: exact dependency-stale settled set (census_ack.stale_excluded_ack_digest).
    stale_excluded_digest: Optional[str] = None
    #: WS-1 clause 7: the launching config's selection continuation identity,
    #: persisted into the wake callback argv and machine-plumbed back on
    #: resume. The driver recomputes its own identity and refuses a mismatch:
    #: a wake must replay the COMPLETE launching config, byte-for-byte.
    wake_continuation_identity: Optional[str] = None
    #: The validated --environment-readiness-receipt path this production leg
    #: launched with. Part of the COMPLETE launching config (WS-1 clause 7):
    #: the CLI readiness gate refuses run/resume without it, so a wake
    #: callback that drops the flag crash-loops its supervisor at the door
    #: (step-200 leg 10, 2026-08-21: three flat restarts -> circuit open).
    environment_readiness_receipt_path: Optional[Path] = None
    campaign_config_path: Optional[Path] = None
    #: Frozen partitioner TIER campaign this run belongs to (``c1-mech`` ... ``c4-native``),
    #: or ``None`` when the run is not bound to one. This is the run's author-tier
    #: identity. It is NOT the per-item repair scope produced by
    #: :func:`_campaign_id_for_item`, which is ``campaign-<stable_id>`` and is per model.
    campaign_id: Optional[str] = None
    #: EN-2 explicit operator exemption: intents that MAY admit members while
    #: their lock family is absent (deliberately lockless native lanes). The
    #: admission door re-derives lockless-ness from the LIVE registry -- an
    #: entry naming an intent that is actually locked refuses as stale -- and
    #: never trusts a tool-written field for it (producer-input rule).
    allow_lockless_intents: tuple[str, ...] = ()
    author_queue_root: Optional[Path] = None
    #: Bounded number of author sessions one wave may keep in flight at once. Only the
    #: author *sessions* overlap: every ledger append, artifact publication, and terminal
    #: routing decision stays on the single canonical writer thread, in work order. The
    #: right value differs by lane -- an in-session queue pool is cheaper to widen than a
    #: `claude -p` subprocess fan-out -- so it is configurable rather than derived. ``1``
    #: restores the historical fully serial lane.
    author_concurrency: int = DEFAULT_AUTHOR_WAVE_CONCURRENCY
    #: Bounded number of checker sessions one gate round may keep in flight at once.
    #: The checker analogue of ``author_concurrency`` and the same discipline: only
    #: the *sessions* overlap. Envelope identity, gate normalization, routing, and
    #: every ledger append stay on the single canonical writer, in ladder order.
    #: ``1`` restores the historical fully serial checker lane.
    checker_concurrency: int = DEFAULT_CHECKER_WAVE_CONCURRENCY
    #: D-63 (PM-1): which channel decided each width, from the CLOSED source
    #: vocabulary (explicit-flag > explicit-env > frozen-config > default).
    #: Stamped into the leg-config attestation so an override that applied
    #: says so durably -- the 200a ledger claimed an override "in effect"
    #: that nothing consumed.
    author_concurrency_source: str = "default"
    checker_concurrency_source: str = "default"
    #: Leg-level scheduling shape. ``waves`` is the historical intake-wide phase
    #: barrier; ``pipelined`` is the M3 per-model scheduler (execute-first triage,
    #: env-intent turns, cap-width gate batching after execution). Records are
    #: scheduler-agnostic: both modes read and write the same canonical ledgers,
    #: so rollback is the same binary with ``waves``.
    scheduler: str = "waves"
    run_repair_max: int = 2
    invocation_origin: InvocationOrigin = InvocationOrigin.ORDINARY_RUN
    wake_episode_id: Optional[str] = None
    #: W6: the supervised restart ladder's attested ordinal (0 for the first
    #: invocation of a supervision episode); recorded on the leg_started row.
    invocation_ordinal: Optional[int] = None
    #: PM-5 (W6): the operator's rationale funding ONE acknowledged retry
    #: past a tripped repeat-cause fuse. Never persisted, never a
    #: continuation flag -- an acknowledgment is one leg's operator act.
    acknowledge_repeat_cause: Optional[str] = None
    #: Overhaul RC-1 (RULING D.1): ``leg`` is the historical sealed leg (run/
    #: resume, kept for one release); ``crawl`` is the continuous-crawl
    #: invocation -- in-process preflight, persisted order + limit, the
    #: leg_open row as the invocation record, typed exits, no close.
    invocation_shape: str = "leg"
    #: ``crawl --limit N``: the order prefix admitted to selection (``None`` is
    #: the whole order). Recorded as a ``crawl-limit`` event when set or raised.
    crawl_limit: Optional[int] = None
    #: The recorded seed a first-use composition of the campaign order uses;
    #: inert once the order exists (an order is never re-derived).
    crawl_order_seed: int = 20260901

    def __post_init__(self) -> None:
        """Validate checkpoint and milestone configuration."""

        if self.invocation_shape not in {"leg", "crawl"}:
            raise ValueError(
                f"invocation_shape must be 'leg' or 'crawl', not {self.invocation_shape!r}"
            )
        if self.crawl_limit is not None and self.crawl_limit < 1:
            raise ValueError(f"crawl_limit must be positive, not {self.crawl_limit}")
        if self.invocation_shape == "crawl" and self.scheduler != "pipelined":
            raise ValueError("the crawl invocation runs the pipelined scheduler only")

        if self.run_id == LEGACY_RUN_ID:
            raise LegacyRunIdentityError(
                "run_id 'crawler-run' is the retired shared constant: every "
                "invocation must carry its own minted identity (historical rows "
                "keep it; new rows never do)"
            )
        if not self.run_id.strip():
            raise ValueError("run_id cannot be empty")
        if self.phase not in {None, "pytorch", "native-tail"}:
            raise ValueError("phase must be pytorch, native-tail, or omitted")
        if self.only_status not in {None, *ONLY_STATUS_VOCABULARY}:
            raise ValueError("only_status must select one of the closed deferred statuses")
        if (
            self.only_status is not None
            and self.only_status != HOST_CAPABILITY_DEFERRAL_STATUS_CODE
            and self.target != "linux-x86_64-cuda"
        ):
            # The PLATFORM handoff is Linux-only by construction. A host-capability
            # deferral is not: its remedy is more RAM, and the machine that has it may
            # well be a bigger arm64 Mac. Reserving the whole flag to the CUDA target
            # left the memory cohort with no operator lane at all.
            raise ValueError("only_status is reserved for the Linux deferred handoff")
        selection_scopes = [
            self.only_unsettled,
            self.only_status is not None,
            bool(self.redrive_targets),
            bool(self.gate_catchup_targets),
            self.recheck,
        ]
        if sum(1 for selected in selection_scopes if selected) > 1:
            # WS-1 clause 4: a leg runs under exactly ONE named selection
            # mode; composing scopes would reintroduce the ambiguity the one
            # predicate exists to kill.
            raise ValueError(
                "only one selection mode may be set per leg: only-unsettled, "
                "only-status, redrive targets, gate-catchup targets, or recheck"
            )
        if self.leg_purpose is not None:
            from menagerie.crawler.leg_purpose import (
                LegPurposeError,
                validate_leg_purpose,
            )

            try:
                validate_leg_purpose(self.leg_purpose)
            except LegPurposeError as exc:
                raise ValueError(str(exc)) from exc
            if any(selection_scopes):
                # The purpose IS a total selection forfeit: composing it with
                # any named admission mode is a mis-declared production leg,
                # refused at the door rather than discovered at close.
                raise ValueError(
                    "leg_purpose licensing_close forfeits all admission; it "
                    "cannot be combined with only-unsettled, only-status, "
                    "redrive targets, or gate-catchup targets"
                )
        for targets, flag in (
            (self.redrive_targets, "redrive_targets"),
            (self.gate_catchup_targets, "gate_catchup_targets"),
        ):
            if any(not isinstance(target, str) or not target for target in targets):
                raise ValueError(f"{flag} must be nonempty stable IDs")
            if len(set(targets)) != len(targets):
                raise ValueError(f"{flag} must be unique stable IDs")
        # D-54 blast-radius guard: a named re-drive is an operator act whose
        # every target buys a fresh work generation. Reason required; inline
        # lists capped; above the cap the list arrives ONLY as a digested
        # manifest bound to the intake snapshot (validated at leg start).
        if self.redrive_targets and not (self.redrive_reason or "").strip():
            raise ValueError(
                "redrive_targets require --redrive-reason: a named re-drive is "
                "an operator act with a blast radius, and its rationale is part "
                "of the durable leg record"
            )
        if self.redrive_reason is not None and not self.redrive_targets:
            raise ValueError("redrive_reason without redrive_targets names no act")
        if self.redrive_manifest_path is not None and not self.redrive_targets:
            raise ValueError(
                "redrive_manifest_path without redrive targets: the launch "
                "surface must load the manifest's targets into redrive_targets"
            )
        if self.redrive_manifest_path is not None and not self.redrive_manifest_path.is_absolute():
            raise ValueError("redrive_manifest_path must be absolute")
        if (
            len(self.redrive_targets) > REDRIVE_INLINE_TARGET_CAP
            and self.redrive_manifest_path is None
        ):
            raise ValueError(
                f"{len(self.redrive_targets)} re-drive targets exceed the "
                f"inline cap ({REDRIVE_INLINE_TARGET_CAP}). Above the cap the "
                "target list must arrive as a digested manifest "
                "(--redrive-manifest) bound to the intake snapshot, never as "
                "a hand-typed CSV"
            )
        if self.review_checkpoint_at is not None and self.review_checkpoint_at < 0:
            raise ValueError("review_checkpoint_at cannot be negative")
        if any(value < 1 for value in self.progress_milestones):
            raise ValueError("progress milestones must be positive")
        if len(set(self.progress_milestones)) != len(self.progress_milestones):
            raise ValueError("progress milestones must be unique")
        if self.run_repair_max < 0:
            raise ValueError("run_repair_max cannot be negative")
        if self.claude_cli_version != CLAUDE_CLI_VERSION:
            raise ValueError(
                "claude_cli_version must match the reviewed campaign pin "
                f"{CLAUDE_CLI_VERSION!r}, not {self.claude_cli_version!r}"
            )
        if self.campaign_config_path is not None and not self.campaign_config_path.is_absolute():
            raise ValueError("campaign_config_path must be absolute")
        if self.campaign_id is not None and not self.campaign_id:
            raise ValueError("campaign_id cannot be empty")
        if self.campaign_id is not None and self.campaign_id not in TIER_CAMPAIGN_IDS:
            # A per-item repair scope reaching this field would be published to the author
            # queue as the run's tier and would select the wrong frozen author model.
            raise ValueError(
                f"campaign_id must name a frozen tier campaign {sorted(TIER_CAMPAIGN_IDS)}, "
                f"not {self.campaign_id!r}"
            )
        if self.author_queue_root is not None and not self.author_queue_root.is_absolute():
            raise ValueError("author_queue_root must be absolute")
        if not 1 <= self.author_concurrency <= MAX_AUTHOR_WAVE_CONCURRENCY:
            raise ValueError(
                "author_concurrency must be between 1 and "
                f"{MAX_AUTHOR_WAVE_CONCURRENCY}, not {self.author_concurrency}"
            )
        if not 1 <= self.checker_concurrency <= MAX_CHECKER_WAVE_CONCURRENCY:
            raise ValueError(
                "checker_concurrency must be between 1 and "
                f"{MAX_CHECKER_WAVE_CONCURRENCY}, not {self.checker_concurrency}"
            )
        for source in (self.author_concurrency_source, self.checker_concurrency_source):
            if source not in {"explicit-flag", "explicit-env", "frozen-config", "default"}:
                # Closed provenance vocabulary (D-63): a free-text source would
                # let a lying spelling reach the leg-config attestation.
                raise ValueError(
                    f"concurrency source must be one of the closed vocabulary, not {source!r}"
                )
        if self.scheduler not in SCHEDULER_MODES:
            # The dispatch test is `== "pipelined"`, so an unvalidated typo would
            # silently run the OTHER scheduler with no record of the intent.
            raise ValueError(
                f"scheduler must name one of {list(SCHEDULER_MODES)}, not {self.scheduler!r}"
            )
        if not isinstance(self.invocation_origin, InvocationOrigin):
            raise ValueError("invocation_origin must be a closed InvocationOrigin")
        if (
            self.invocation_origin is InvocationOrigin.WAKE_CALLBACK
            and self.wake_episode_id is None
        ):
            raise ValueError("wake callbacks require an exact wake episode ID")
        if (
            self.invocation_origin is not InvocationOrigin.WAKE_CALLBACK
            and self.wake_episode_id is not None
        ):
            raise ValueError("only wake callbacks may carry a wake episode ID")
        if self.invocation_ordinal is not None and (
            not isinstance(self.invocation_ordinal, int) or self.invocation_ordinal < 0
        ):
            raise ValueError("invocation_ordinal must be a non-negative int or omitted")
        if (
            self.invocation_origin is InvocationOrigin.SUPERVISED_RESTART
            and self.invocation_ordinal is None
        ):
            raise ValueError("a supervised restart must attest its ordinal")


@dataclass(frozen=True)
class WorkItem:
    """One intake row paired with its deterministic environment route."""

    intake: IntakeItem
    route: IntentRoute
    explicit_grants: tuple[str, ...] = ()
    requeue_work_id: Optional[str] = None
    requeue_active: bool = False
    discovery_source_url: Optional[str] = None
    refresh_work_id: Optional[str] = None
    #: Requeue-granted author-session extension for the ACTIVE granted
    #: generation (docket T1-13, allowance half): the consumed grant's
    #: ``attempts``, previously recorded on the consumption event and never
    #: read again. ``None`` for organic work and spent-grant history.
    granted_attempts: Optional[int] = None

    @property
    def stable_id(self) -> str:
        """Return the durable model identifier."""

        return self.intake.stable_id

    @property
    def family_representative_id(self) -> str:
        """Return the explicitly designated family representative ID."""

        return self.intake.family_representative_id or self.stable_id

    @property
    def is_family_variant(self) -> bool:
        """Return whether this item is a non-representative family size variant."""

        return (
            self.intake.variant_scope == "family"
            and self.family_representative_id != self.stable_id
        )

    @property
    def active_work_id(self) -> str:
        """Return the exact scheduled authority generation for this item."""

        return self.requeue_work_id or self.refresh_work_id or f"work-{self.stable_id}"


def bound_campaign_id(config: DriverConfig) -> str:
    """Return the run's frozen tier campaign for durable row emission.

    Every valve/park row emission used to spell ``str(config.campaign_id)``,
    which converted a legal unbound ``None`` into the illegal-but-unchecked
    string ``"None"`` at the ledger boundary -- all 62 pilot valve rows froze
    ``campaign_id="None"`` into their ``row_sha256`` (docket T2-12), permanently
    unattributable to any tier campaign. ``DriverConfig.__post_init__`` already
    refuses the string; this is the matching refusal for the absence.

    Parameters
    ----------
    config:
        Active driver configuration.

    Returns
    -------
    str
        The bound tier campaign ID, verbatim.

    Raises
    ------
    DriverIntegrationError
        When the run carries no campaign binding. Durable rows must name a
        real campaign or refuse loudly, never stringify the absence.
    """

    if config.campaign_id is None:
        raise DriverIntegrationError(
            "durable row emission requires a bound tier campaign_id; this run is not bound to one"
        )
    return config.campaign_id


def _campaign_id_for_item(item: WorkItem) -> str:
    """Return the bounded repair campaign rooted at the active work generation.

    This is a per-ITEM authority lineage (``campaign-<stable_id>``, or
    ``campaign-<work_id>`` for a requeue/refresh) -- the value the author envelopes carry
    as ``campaign_id`` and the ledgers record as ``campaign_root_work_id``. It is
    unrelated to :attr:`DriverConfig.campaign_id`, the run's frozen partitioner TIER
    campaign, and it must never be used to select an author model or prompt.
    """

    if item.requeue_work_id is not None or item.refresh_work_id is not None:
        return f"campaign-{item.active_work_id}"
    return f"campaign-{item.stable_id}"


def _intake_discovery_urls(snapshot: IntakeSnapshot) -> dict[str, str]:
    """Recover exact public lead URLs from immutable intake source rows.

    Parameters
    ----------
    snapshot:
        Verified immutable intake snapshot.

    Returns
    -------
    dict[str, str]
        Stable model IDs mapped to retained HTTP(S) discovery URLs.
    """

    filenames = {
        "master_catalog": "master_catalog.jsonl",
        "deferred": "deferred.jsonl",
    }
    rows_by_source: dict[str, dict[str, Mapping[str, Any]]] = {}
    for discovery_source, filename in filenames.items():
        path = snapshot.root / "sources" / filename
        if not path.is_file():
            continue
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise DriverIntegrationError(
                f"verified intake source is unreadable: {filename}"
            ) from exc
        if any(not isinstance(row, Mapping) for row in rows):
            raise DriverIntegrationError(
                f"verified intake source contains a non-object row: {filename}"
            )
        rows_by_source[discovery_source] = {
            stable_hash(row): row for row in rows if isinstance(row, Mapping)
        }
    recovered: dict[str, str] = {}
    for item in snapshot.items:
        row = rows_by_source.get(item.discovery_source, {}).get(item.legacy_row_sha256)
        if row is None:
            continue
        url = row.get("source_url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            recovered[item.stable_id] = url
    return recovered


@dataclass(frozen=True)
class RecordedIdentityBasis:
    """The identity world one finalized artifact was accepted under.

    Carried by rehydrated artifacts so identity revalidation runs at the
    RECORDED basis (the transaction's own dependency-vector pins) instead of
    re-imposing the live prompt files on history -- the rung-11/rung-12
    history-vs-live family. Reuse eligibility remains solely the reducer's
    dependency-currency decision; this basis only proves the recorded
    artifact's internal consistency.
    """

    author_prompt_sha256: str
    checker_prompt_sha256: str


@dataclass(frozen=True)
class AuthorArtifact:
    """Typed v3 author outcome plus private custody and exact request artifacts."""

    author_result: AuthorResult
    source_manifest: JsonObject
    model_dir: Path
    staged: Optional[StagedArtifact] = None
    canonical_code_root: Optional[Path] = None
    template_source_revision: Optional[str] = None
    recorded_identity_basis: Optional[RecordedIdentityBasis] = None
    #: The model the executor ACTUALLY launched (its ``--model`` at spawn), read
    #: from the lane-verified publication or queue receipt. This is the record
    #: provenance authority when present -- the stamped ``proposal.author.model``
    #: is only the campaign configuration's claim, and the two disagreed silently
    #: for every leg between 2026-08-05 and AM1 (C-AUTHOR-IDENTITY). ``None``
    #: means no attestation exists (history, cache rehydration, fixture lanes),
    #: in which case provenance keeps the stamped claim unrewritten.
    launched_author_model: Optional[str] = None

    @property
    def proposal(self) -> JsonObject:
        """Return the proposed arm or reject terminal-outcome misuse."""

        if isinstance(self.author_result, ProposedAuthorResult):
            return self.author_result.proposal
        if (
            isinstance(self.author_result, DeferRecommendation)
            and self.author_result.handoff_execution is not None
        ):
            return self.author_result.handoff_execution.proposal
        raise DriverIntegrationError("terminal author result has no executable proposal")

    @property
    def campaign_root_work_id(self) -> str:
        """Return the exact v3 campaign lineage identity."""

        return self.author_result.binding.campaign_id


@dataclass(frozen=True)
class ActivatedHandoffArtifact(AuthorArtifact):
    """Durably reconstructed executable authority for a target-host deferral resume."""

    handoff_sha256: str = ""


@dataclass(frozen=True)
class EnvironmentBinding:
    """Verified lifecycle facts and optional live sealed-prefix authority."""

    prefix: Path
    python_executable: Path
    family: str
    target: str
    env_generation: str
    lock_sha256: str
    resolved_export_sha256: str
    packages_manifest_sha256: str
    python_version: str
    compiler_identity: str
    sdk_identity: str
    authority_epoch: Optional[str] = None
    base_environment_generation: Optional[str] = None
    environment_content_sha256: Optional[str] = None
    environment_authority_id: Optional[str] = None
    selected_interpreter_relative_path: Optional[str] = None
    selected_interpreter_digest: Optional[str] = None
    external_escape_records: tuple[EnvironmentExternalTarget, ...] = ()
    environment_authority: Optional[EnvironmentAuthorityV1] = None
    environment_authority_cache: Optional[EnvironmentAuthorityCache] = None


UsageBackoffSignal: TypeAlias = CheckerBackoffSignal | AuthorBackoffSignal
"""Either lane's typed provider pause signal, routed through one pause path."""


@dataclass(frozen=True)
class CheckerOutcome:
    """Either one immutable gate or a typed provider pause, plus parent spend."""

    gate: Optional[JsonObject] = None
    backoff: Optional[CheckerBackoffSignal] = None
    session_id: Optional[str] = None
    wall_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        """Require exactly one checker outcome arm."""

        if (self.gate is None) == (self.backoff is None):
            raise ValueError("checker outcome requires exactly one gate or backoff signal")


@dataclass(frozen=True)
class DriverResult:
    """Observable disposition of one scheduler invocation."""

    status: str
    terminal_models: int
    models_reduced: int
    paused_reason: Optional[str]
    shutdown_interruption: Optional[ShutdownInterruptionFact] = None


class AuthorLane(Protocol):
    """Injectable author-session boundary."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Return one validated, immutable author artifact."""

        ...


class CheckerLane(Protocol):
    """Injectable independent metadata and fidelity checker boundary."""

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Check one fresh metadata batch."""

        ...

    def check_fidelity(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Check one fresh per-model fidelity envelope."""

        ...

    def check_terminal(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Check one staged terminal recommendation."""

        ...


class ForwardLane(Protocol):
    """Injectable isolated-worker supervisor boundary."""

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
        *,
        worker_lock_path: Optional[Path] = None,
        worker_lease_path: Optional[Path] = None,
        run_id: str = "direct-forward",
        shutdown_event: Optional[threading.Event] = None,
        lifecycle_event: Optional[Callable[[str, str, WorkerLease], None]] = None,
        attempt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        attempt_resolver: Optional[Callable[[int, str], Optional[Mapping[str, Any]]]] = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Return complete attempt records for every required cold mode run."""

        ...


class EnvironmentLane(Protocol):
    """Injectable one-at-a-time exact environment lifecycle boundary."""

    def run(
        self,
        intent: EnvironmentIntent,
        *,
        use: Callable[[Path, tuple[ProbeResult, ...]], None],
    ) -> object:
        """Create, probe, use, and tear down one environment."""

        ...


class Notifier(Protocol):
    """Injectable best-effort JMT notification boundary."""

    def notify(
        self,
        summary: str,
        *,
        idempotency_key: str,
        notification_class: Optional[str] = None,
    ) -> bool:
        """Send one ASCII summary, returning whether delivery succeeded.

        ``notification_class`` names the caller-derived closed
        :class:`~menagerie.crawler.notification_policy.NotificationClass`
        value; the transport-owning implementation applies the class policy,
        and ``None`` always attempts delivery.
        """

        ...


class UsagePauseScheduler(Protocol):
    """Injectable idempotent reset-wakeup scheduling boundary."""

    def schedule(
        self,
        signal: UsageBackoffSignal,
        operational: JsonlLedger,
        context: OperationalContext,
        created_at: str,
        reset_at: str,
        reset_observation: str,
    ) -> None:
        """Record the pause and schedule its exact reset wakeup."""

        ...


BoundaryHook = Callable[[str, str], None]

Clock = Callable[[], str]

MonotonicClock = Callable[[], float]


@dataclass(frozen=True)
class DriverDependencies:
    """All live external effects required by the deterministic driver."""

    author: AuthorLane
    checker: CheckerLane
    forward: ForwardLane
    environments: EnvironmentLane
    notifier: Notifier
    clock: Clock
    boundary_hook: BoundaryHook = lambda _boundary, _stable_id: None
    usage_pause_scheduler: Optional[UsagePauseScheduler] = None
    wakeup_installer: Optional[Callable[[Any], None]] = None
    wakeup_verifier: Optional[Callable[[Any], bool]] = None
    wakeup_deactivator: Optional[Callable[[Any], None]] = None
    #: Injectable wait used between bounded infrastructure retries. ``None`` means
    #: the driver waits on its own shutdown event (real time, interruptible by
    #: SIGTERM); tests inject a recorder so the backoff ladder is provable without
    #: sleeping.
    sleeper: Optional[Callable[[float], None]] = None
    #: Provider/broker status query keyed by the durable idempotency key.  A
    #: restart with an unresolved reservation refuses paid admission when this
    #: capability is absent; ambiguity is never treated as zero or retried.
    provider_reconciler: Optional[Callable[[str], Any]] = None
    #: Injectable monotonic clock read by every driver-side duration and
    #: deadline measurement: retry-ladder backoff deadlines, pool pacing,
    #: paid-session wall spans, and the scheduler's readiness windows.
    #: Virtual-time rehearsals inject a controlled clock here so hours of
    #: scheduling run in milliseconds of wall.  Waits that bound REAL child
    #: processes or kernel locks (guard communicate/kill grace, custody-lease
    #: reaps) deliberately stay on ``time.monotonic``: virtual time cannot
    #: make a real process exit sooner.
    monotonic: MonotonicClock = time.monotonic


class DriverLock:
    """Process-level advisory lock held across every mutable driver action."""

    def __init__(self, path: Path, owner: Mapping[str, Any]) -> None:
        """Configure the lock path and owner metadata."""

        self.path = path
        self.owner = dict(owner)
        self._handle: Optional[Any] = None

    def __enter__(self) -> "DriverLock":
        """Acquire the nonblocking kernel lock and publish owner metadata."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise DriverLockError(f"another driver owns {self.path}") from exc
        handle.seek(0)
        handle.truncate()
        json.dump(self.owner, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        durable_fsync(handle.fileno())
        self._handle = handle
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release the kernel lock."""

        del exc_type, exc_value, traceback
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None

    def held(self) -> bool:
        """Return whether this process still verifiably holds the lock.

        A cheap mechanical fact for the author-lane host checks: the handle is
        open, its file still exists at the lock path, and the open descriptor
        still refers to that same inode (an unlinked-and-recreated lock file
        means another writer may hold a lock this handle no longer guards).

        Returns
        -------
        bool
            Whether the acquired lock is still in force.
        """

        if self._handle is None:
            return False
        try:
            on_disk = os.stat(self.path)
            via_handle = os.fstat(self._handle.fileno())
        except OSError:
            return False
        return (on_disk.st_dev, on_disk.st_ino) == (via_handle.st_dev, via_handle.st_ino)


def default_driver_paths(repo_root: Path, intake_root: Path) -> DriverPaths:
    """Return conventional Slice-F runtime and canonical record paths."""

    return DriverPaths(
        runtime_root=repo_root / ".crawl-local",
        intake_root=intake_root,
        ledgers=default_ledger_paths(repo_root / "menagerie" / "crawler" / "records"),
    )
