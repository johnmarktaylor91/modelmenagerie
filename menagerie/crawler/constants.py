"""Closed crawler vocabularies and fixed contract values."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from enum import Enum
from typing import Any, Mapping

MODEL_SCHEMA_VERSION = "menagerie.crawler.model.v2"
ATTEMPT_SCHEMA_VERSION = "menagerie.crawler.attempt.v2"
GATE_SCHEMA_VERSION = "menagerie.crawler.gate.v2"
AUTHOR_PROPOSAL_SCHEMA_VERSION = "menagerie.crawler.author-proposal.v2"
OPERATIONAL_EVENT_SCHEMA_VERSION = "menagerie.crawler.operational-event.v1"
EXECUTION_READ_MANIFEST_VERSION_V2 = "menagerie.crawler.execution-read-manifest.v2"
EXECUTION_READ_MANIFEST_VERSION_V3 = "menagerie.crawler.execution-read-manifest.v3"
ENVIRONMENT_AUTHORITY_VERSION_V1 = "menagerie.crawler.environment-authority.v1"
ENVIRONMENT_CONTENT_MANIFEST_VERSION_V1 = "menagerie.crawler.environment-content-manifest.v1"
ENVIRONMENT_GENERATION_VERSION_V2 = "menagerie.crawler.environment-generation.v2"
DRIVER_SHUTDOWN_STATUS = "interrupted:shutdown"

# Round-14 contracts are additive during the interface freeze. Existing producers
# continue to emit v2 until the authority kernel switches atomically; v3 writers
# use these explicit current-version discriminators and may read immutable v2 history.
MODEL_SCHEMA_VERSION_V3 = "menagerie.crawler.model.v3"
ATTEMPT_SCHEMA_VERSION_V3 = "menagerie.crawler.attempt.v3"
GATE_SCHEMA_VERSION_V3 = "menagerie.crawler.gate.v3"
AUTHOR_PROPOSAL_SCHEMA_VERSION_V3 = "menagerie.crawler.author-proposal.v3"
AUTHOR_RESULT_SCHEMA_VERSION_V3 = "menagerie.crawler.author-result.v3"
AUTHOR_RESULT_SCHEMA_VERSION = "menagerie.crawler.author-result.v4"
PROMOTION_SCHEMA_VERSION = "menagerie.crawler.promotion.v1"
SOURCE_DISCOVERY_SCHEMA_VERSION = "menagerie.crawler.source-discovery.v1"
# The requeue-grant ledger was the ONE hand-appended canonical ledger with no
# schema identity at all (docket T2-12): rows self-described only through their
# field shapes. Newly minted grant and void rows stamp these discriminators;
# legacy rows without one stay readable forever, and a row claiming a FOREIGN
# identity refuses typed. Neither value enters the grant-identity preimage.
REQUEUE_GRANT_SCHEMA_VERSION_V1 = "menagerie.crawler.requeue-grant.v1"
# W31: v2 = v1 + created_at + adjudication_key + effective_session_ceiling.
# All three ride OUTSIDE the grant-identity preimage, exactly like the v1
# schema_version stamp did, so no existing grant hash moved.
REQUEUE_GRANT_SCHEMA_VERSION = "menagerie.crawler.requeue-grant.v2"
# The CLOSED monotone requeue-grant schema ladder, oldest first. Readers accept
# every rung (plus the pre-ladder unstamped era); writers mint only the newest.
REQUEUE_GRANT_SCHEMA_LADDER = (
    REQUEUE_GRANT_SCHEMA_VERSION_V1,
    REQUEUE_GRANT_SCHEMA_VERSION,
)
REQUEUE_GRANT_VOID_SCHEMA_VERSION = "menagerie.crawler.requeue-grant-void.v1"
ARTIFACT_EVENT_SCHEMA_VERSION = "menagerie.crawler.artifact-event.v1"
TEXT_CORPUS_INDEX_SCHEMA_VERSION = "menagerie.crawler.text-corpus-index.v1"

CURRENT_SCHEMA_VERSIONS = frozenset(
    {
        MODEL_SCHEMA_VERSION_V3,
        ATTEMPT_SCHEMA_VERSION_V3,
        GATE_SCHEMA_VERSION_V3,
        AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
        AUTHOR_RESULT_SCHEMA_VERSION,
        ARTIFACT_EVENT_SCHEMA_VERSION,
        OPERATIONAL_EVENT_SCHEMA_VERSION,
    }
)

LEGACY_UNTRUSTED_SCHEMA_VERSIONS = frozenset(
    {
        MODEL_SCHEMA_VERSION,
        ATTEMPT_SCHEMA_VERSION,
        GATE_SCHEMA_VERSION,
        AUTHOR_PROPOSAL_SCHEMA_VERSION,
        AUTHOR_RESULT_SCHEMA_VERSION_V3,
    }
)

AUTHOR_PROMPT_NAME = "claude_crawler_author_v2"
CHECKER_PROMPT_NAME = "codex_accuracy_checker_v2"

# BL-11: the author lane's confinement and result classification depend on the
# Claude CLI's concrete JSON/result internals. A weeks-long campaign therefore
# pins one reviewed CLI release instead of recording the content-free "current".
CLAUDE_CLI_VERSION = "2.1.257"

# W32: the checker lane's codex CLI rides the same pin discipline -- one
# reviewed release, verified per leg before any paid reserve, bumpable only
# with a toolchain re-attestation (menagerie/crawler/toolchain_attestation.json).
CODEX_CLI_VERSION = "0.144.5"

# Seat swap (2026-09-01): the Anthropic checker seat runs through the Claude
# CLI under its OWN pin (`MENAGERIE_CHECKER_CLAUDE_BIN`, toolchain lane
# `checker-anthropic`), separate from the author harness pin above because the
# two lanes bump independently: `claude-fable-5-1` needs >= 2.1.251 (the exact
# author pin 2.1.239 exits 78 on it), and the bench that seated Fable ran on
# 2.1.257. Same re-attestation discipline as the other two pins.
CHECKER_CLAUDE_CLI_VERSION = "2.1.257"

# OFFLINE-PINNING (step-100 triage row 9, m10759): the CLOSED list of known
# phone-home distributions mapped to their documented offline switch
# assignments. This is the ONE owner of that knowledge: ``build_safe_environment``
# sets every assignment in the sandboxed child environment, and proposal
# validation refuses a recipe naming a listed distribution whose switch the
# execution environment does not guarantee. Keys are canonical (PEP 503)
# distribution names. Growing the list is a deliberate, reviewed change: add the
# distribution here with its library-documented switch and value (ultralytics
# accepts YOLO_OFFLINE=true on both its old string-compare and current
# env_bool implementations), and the policy/proposal consumers pick it up
# without further edits.
PHONE_HOME_OFFLINE_SWITCHES: Mapping[str, Mapping[str, str]] = {
    "ultralytics": {"YOLO_OFFLINE": "true"},
    "wandb": {"WANDB_MODE": "offline", "WANDB_DISABLED": "true"},
    "huggingface-hub": {"HF_HUB_OFFLINE": "1"},
    "comet-ml": {"COMET_DISABLE_AUTO_LOGGING": "1"},
}

# Package-relative paths of the two shipped components whose content identities are
# derived from ZERO arguments. They live here, not beside either consumer, because four
# independent seams derive them -- the gate scaffold in ``checker_dispatch``, the
# ``AuthorityContext``, the gate proof loader, and the reducer dependency vector -- and a
# path spelled twice is a path that can drift.
AUTHOR_RESULT_SCHEMA_COMPONENT = "schemas/author-result-v4.schema.json"
AUTHOR_DISPATCHER_COMPONENT = "author_dispatch.py"

# WIDTH 1 (2026-08-29, Opus adjudication 5.3 rule 1 FIRED on the Arm A matrix):
# batched metadata adjudication is verdict-INDETERMINATE at width 5 -- under one
# model/prompt epoch, 8 of 25 item-cells flipped their acceptance bit across 3
# same-epoch replays (whole-batch bless mood swings), while the width-1 arm
# flipped 0 of 18 on the same goldens under the repaired prompt. Width is the
# mechanism, so production runs single-item metadata gates. Token headroom is a
# non-issue at width 1 (one item projects to 2--2.5k output tokens against the
# observed 21,571 ceiling; the rung-10 35-checks/item telemetry still governs).
# The measured width-1 session p95 keeps the 200-model rung inside the signed
# 70.0h wall ceiling at the 0.8 margin (Arm A: p95 486s -> 36.7h vs 56.0h).
METADATA_BATCH_MIN = 1
METADATA_BATCH_MAX = 1
METADATA_FINAL_TAIL_MIN = 1
STDIO_TAIL_MAX_CHARS = 1_500
STABLE_ID_DIGEST_CHARS = 20
DEFAULT_FORWARD_TIMEOUT_SECONDS = 300
MAX_FORWARD_TIMEOUT_SECONDS = 1_800
DEFAULT_NOTIFY_TIMEOUT_SECONDS = 5
DEFAULT_REVIEW_CHECKPOINT_MODELS = 1_000
DEFAULT_PROGRESS_NOTIFICATION_MILESTONES = (
    900,
    950,
    1_000,
    2_000,
    3_000,
    5_000,
    10_000,
    15_000,
    20_000,
)

# `PLAN.md` LP-13.2 author-session effort ceiling. The pool enforces the tool-call
# and wall budgets because it is the only boundary that observes Agent-tool events;
# the lane enforces the fetch-target count, which it observes directly, and audits
# the pool's declared consumption against the same grant.
AUTHOR_MAX_TOOL_CALLS = 30
AUTHOR_MAX_FETCH_TARGETS = 20
AUTHOR_SESSION_WALL_SECONDS = 30 * 60
# Outer stall guard. A managing session that dies must look like a stalled queue
# (retryable infrastructure), never a failed model.
AUTHOR_QUEUE_STALL_SECONDS = 45 * 60
AUTHOR_QUEUE_POLL_SECONDS = 2.0

# ---------------------------------------------------------------------------
# The author wall budget, single-sourced (SEAM_REDESIGN 3.7)
#
# Wall grants are PER CAMPAIGN, not a global constant: c3-classics carries the
# no-prior-code classics whose mean session is ~25 min with a long tail, and
# truncating that tail both loses the models and corrupts the very p95 the
# month's go/no-go depends on. Everything downstream -- the grant published to
# the session, the executor's own kill, and the lane's outer stall bound --
# derives from `resolve_author_wall_seconds` so no lower level can silently
# impose a stricter number of its own.
# ---------------------------------------------------------------------------

#: Per-campaign wall grants (seconds). Absent campaigns take the default grant.
AUTHOR_CAMPAIGN_WALL_SECONDS: dict[str, float] = {"c3-classics": 60.0 * 60.0}

#: The executor SIGKILLs its own `claude -p` child at grant x this factor. The
#: brief carries the deadline itself, so a session watching its clock lands a
#: typed BLOCKED-with-partial inside the grant and this only catches the ones
#: that do not.
AUTHOR_WALL_EXTERNAL_KILL_FACTOR = 1.10

#: Worst-case number of grant-sized `claude -p` sessions ONE executor invocation
#: may legitimately run. Stage 2 is the worst case: the primary run (1.0), one
#: cold rerun after a provider resume failure (1.0), and at most one typed
#: supplementary broker round at up to the full grant (1.0). The supplement
#: round used to be granted a bare `wall_seconds() / 2`, a constant with no
#: measured basis (F51 DECIDED_PLAN item 8; crawl-speed ruling R4): 276 of 513
#: production supplement rounds (54%) were killed at that ~990 s wall while the
#: successful ones ran p50 594 s / p90 945 s, crowding the ceiling. The round
#: now spends the attempt's REMAINING wall, floored at
#: `author_executor._SUPPLEMENT_ROUND_MIN_WALL_SECONDS`, and every continuation
#: is still capped by this licensed budget. The lane's bound must clear this,
#: or a stage 2 that legitimately took the cold-rerun path is killed from
#: outside while still inside its budget. `test_author_wall_budget.py` pins this
#: against the executor's actual session call sites.
AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET = 3.0

#: Slack for the work an invocation does OUTSIDE its `claude -p` children --
#: broker fetches, hashing, attempt-record writes, publication.
AUTHOR_LANE_WALL_MARGIN_SECONDS = 120.0

#: Environment variable carrying the resolved grant to the executor subprocess.
#: The driver PUBLISHES it from the campaign's authoritative grant rather than
#: trusting the operator to set it consistently by hand; a pre-set value that
#: disagrees is a startup failure, never a silent winner.
AUTHOR_WALL_SECONDS_ENV = "MENAGERIE_AUTHOR_WALL_SECONDS"
CAMPAIGN_TEARDOWN_RECORD_ENV = "MENAGERIE_CAMPAIGN_TEARDOWN_RECORD"
CAMPAIGN_TEARDOWN_RECORD_VERSION = "menagerie.crawler.campaign-teardown.v1"


def resolve_author_wall_seconds(
    campaign_id: "str | None", override: "float | None" = None
) -> float:
    """Return the authoritative wall grant for one campaign, in seconds.

    This is the ONLY place a wall grant is chosen. An explicit operator
    override wins; otherwise the campaign's own grant; otherwise the default.

    Parameters
    ----------
    campaign_id:
        Tier campaign identity, or ``None`` outside a tier campaign.
    override:
        Explicit operator grant, when one was configured.

    Returns
    -------
    float
        Strictly positive wall grant in seconds.

    Raises
    ------
    ValueError
        If an override is not a finite, strictly positive number of seconds.
    """

    if override is not None:
        value = float(override)
        if not value > 0.0 or value != value or value in (float("inf"), float("-inf")):
            raise ValueError(
                f"author wall grant must be a finite positive number of seconds, not {override!r}"
            )
        return value
    return float(
        AUTHOR_CAMPAIGN_WALL_SECONDS.get(campaign_id or "", float(AUTHOR_SESSION_WALL_SECONDS))
    )


def author_lane_wall_bound(grant_seconds: float) -> float:
    """Return the lane's outer stall bound for a given per-session grant.

    The lane bound is a STALL GUARD, not a budget. The budget is the grant, and
    the executor enforces it per session; the lane only exists to stop a wedged
    wrapper blocking the driver forever. It must therefore be strictly greater
    than every inner limit, or it preempts the typed, recoverable outcomes the
    inner layers produce -- which is exactly the silent 30-minute truncation
    this derivation replaced.

    Parameters
    ----------
    grant_seconds:
        Authoritative per-session wall grant.

    Returns
    -------
    float
        Bound covering one invocation's worst-case session budget plus the
        non-session work around it.
    """

    return (
        float(grant_seconds)
        * AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET
        * AUTHOR_WALL_EXTERNAL_KILL_FACTOR
        + AUTHOR_LANE_WALL_MARGIN_SECONDS
    )


# Bounded author-session fan-out per wave. PM-Q3 (decided with D-63, 2026-08-24):
# the INTENDED 200b widths are exactly these two defaults -- 12 author sessions and
# 24 checker sessions, the widths BL-11 measured and 200a actually ran (the
# configured-18 override was inert, so 18 was never a proven width). Any increase
# beyond this proven configuration is gated on AL-9's rehearsal-gated ramp
# (plan non-goal: no unrehearsed width increases), and with D-63 an explicit
# override now genuinely WINS and is attested per leg with its provenance
# channel, so a future ramped width is a recorded fact rather than a shell flag.
# The right value genuinely differs per lane and per host, so it is configurable;
# this is the defensible, measured default.
DEFAULT_AUTHOR_WAVE_CONCURRENCY = 12
# Hard ceiling. Past this the bound stops being a bound: a wave would fan out further
# than any provider tier or host can service, and the failure mode is mass quota
# exhaustion rather than throughput.
MAX_AUTHOR_WAVE_CONCURRENCY = 32

# Checker-lane analogue of the author wave bound (docket OD4). Checker sessions are
# heavier per minute than author sessions -- they gate awards -- and the provider quota,
# not the host, is the real ceiling. BL-11's measured campaign widths are 12 author
# sessions and 24 checker sessions; ``1`` restores either historical serial lane.
DEFAULT_CHECKER_WAVE_CONCURRENCY = 24
MAX_CHECKER_WAVE_CONCURRENCY = 32

# Consecutive checker-lane parks -- distinct models, with no successfully routed gate in
# between -- that promote per-model parking to lane-scoped weather. The streak must
# prove TWO things at once before a lane-wide verdict: (a) at least one more park than
# a single poisoned batch can produce by itself (METADATA_BATCH_MAX + 1 -- the rung-10
# fault shape parks at most its own members, a legitimate outcome, so the next
# consecutive park is necessarily evidence from a second, independent batch), and (b)
# enough DISTINCT models that model-specific poison is implausible -- the historical
# six-model evidence bar. At the width-5 era the two coincided (5 + 1 == 6); at the
# calibrated width-1 config the batch-independence arm alone would collapse to 2 and
# read any two adjacent poisoned models as lane weather, so the six-model mass keeps
# the floor. Grinding past the limit buys a failed session ladder per model for a lane
# that is demonstrably not answering anyone.
CHECKER_LANE_PARK_STREAK_LIMIT = max(6, METADATA_BATCH_MAX + 1)

# BL-10 (b): the ABSOLUTE per-lineage metadata gate round ceiling. The chargeable
# repair cap bounds author incorrigibility only -- a checker that disjointly
# alternates complaint sets is never chargeable after round one (pilot m8650 ran
# 10 rounds / 9 rejections against a cap of 2, charging 1), so chargeability alone
# leaves the lineage UNBOUNDED. Past this many total persisted rounds a still
# unaccepted lineage is checker disclosure debt, never author incorrigibility: it
# PARKS on the checker-lane valve (``checker-disclosure-debt``) for human review
# and is never minted as a model-blaming ``failed:*`` terminal. Six is the B-1
# recommendation: 1 initial + 2 chargeable repairs + 3 rounds of free first
# disclosures is already tail behavior (pilot mean 2.07 rounds/lineage).
METADATA_GATE_ROUND_CEILING = 6

# BL-10 (f): re-parks of the SAME model under an UNCHANGED checker adjudication
# identity (model + version + prompt) before the park row holds durably. The first
# park plus one fresh-leg retry distinguishes a stochastic lane flake from an
# item-deterministic failure; past that, re-attempting every leg re-buys the full
# halving ladder (~4 sessions) forever -- the pilot's six checker-lane parks were
# all re-purchasable on every later run. The row releases the moment the lane's
# adjudication identity changes (causal release), or by operator action.
CHECKER_LANE_PARK_RECIDIVISM_LIMIT = 2

# BL-10 (g): checker lane wall/spend breakers. The checker lane had NO analog of
# the author lane's wall constants: one 5-item batch could legally spend multiple
# unbounded hours across the halving ladder, and the campaign had no ceiling at
# all -- 51.15h of checker wall bought 67 models on the pilot, 28% of it rendering
# no verdict. Both are BREAKERS, not throttles: per-model exhaustion PARKS the
# model on the checker-lane valve (never blames it); per-leg exhaustion raises the
# loud retryable lane-weather stop so the campaign pauses resumably instead of
# grinding a broken lane. Model budget: ~3x the pilot's 0.76 h/model straight-line
# mean, comfortably above any honest ladder and far below a runaway.
#
# Leg wall budget (D-2 restoration, 2026-08-24). A reservation holds its FULL
# ceiling against this fuse until it settles, so a lane at width W carries up
# to ``W x reservation_ceiling`` of reserved-but-unconsumed wall with ZERO real
# spend -- deliberate conservatism, and the fuse must be sized for it. The
# checker reservation ceiling is 6336.3s (kill 6273.3 = (5583+120) x 1.10, plus
# 3s backoff slack and 60s wall grace), so the previous 30h (108,000s) value --
# sized when the lane's only real consumer was ~2 concurrent sessions -- was
# crossable by holds ALONE at ~17 concurrent against a configured width of 24
# (max 32): a checker lane busy enough wedges itself with no runaway anywhere.
# The value is now the lane's maximum reserved hold (32 x 6336.3 = 202,761.6s)
# plus the ORIGINAL bound unchanged in meaning: strictly more REAL settled
# checker wall in ONE process leg than the entire 72-model pilot consumed
# lifetime (28.5h -> the 30h bound) is a runaway leg, whatever the cause. Under
# a maximal hold the fuse still trips at exactly that historical real-wall
# bound -- a restoration of the breaker's semantics, not a loosening.
# ``test_lane_fuse_domination.py`` recomputes the domination from the live
# reservation ceilings on every run; a ceiling or width raise without a fuse
# review fails there. This constant is also the author lane's tiny-leg wall
# FLOOR (``author_leg_runaway_budget``), where the same domination rule holds
# against the author ceiling (32 x 5583 = 178,656s).
CHECKER_MODEL_WALL_BUDGET_SECONDS = 2.5 * 60 * 60
CHECKER_LEG_WALL_BUDGET_SECONDS = 310_800.0
CHECKER_LEG_SESSION_BUDGET = 2_000

# ---------------------------------------------------------------------------
# The per-session checker wall, single-sourced (leg-12 incident, 2026-08-21)
#
# The 02:59Z leg-12 incident: the campaign's first sol metadata_batch session
# ran 4.5 HOURS with no output then exited, its custody stayed bound, and the
# writer waited hours. Author sessions have a published wall grant plus a 10%
# external kill; the checker lane's only bound was derived per dispatch from
# the PROVISIONAL operator deadline -- no named wall, no kill factor, and a
# wall-killed session settled as a chargeable generic failure.
#
# Evidence for the number: measured sol terminal_disposition checker sessions
# complete in ~2-4 minutes, and a metadata_batch (5 items) is the same shape at
# batch width. 34 minutes is a ~10x generous multiple of that observed session
# time -- and it is the SMALLEST round figure that still contains the operator
# wrapper's own granted attempt ladder (``OPERATOR_DEADLINE_SECONDS`` = 3
# attempts x 600s + 3s backoff + 180s dispatch slack = 1983s ~= 33.05 min).
# The wall may never sit below that ladder: the wrapper self-polices to the
# deadline the lane publishes and lands typed retryable outcomes INSIDE it, so
# an external kill below the ladder would truncate a typed in-ladder retry
# from outside -- the exact silent-truncation defect the author lane's
# constructor guard refuses. ``test_checker_session_wall_ladder_is_strictly_
# ordered`` pins wall >= ladder, so replacing the provisional 600s attempt
# grant with a measured value tightens this wall in the same reviewed change.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The custody deadline reaper grace (legs 8 + 12, 2026-08-21)
#
# A paid custody binding whose subtree died without settling used to stay
# bound until the NEXT leg start -- and mid-leg, every consumer of the lease
# (the wave admission relay, the outcome writer) waited on it forever. The
# reaper force-settles a binding only when BOTH kernel oracles prove the
# custodian dead (flock FREE and process group EMPTY) AND the durable
# reservation deadline plus this grace has passed. The grace exists for the
# healthy race: a dispatch thread that just killed its session at the wall
# settles within its own bounded teardown (30s SIGTERM grace + 5s SIGKILL
# reap + a 60s-bounded provider reconcile query); five minutes is ~3x that
# worst case, so a reaped lease is one whose owning thread is provably wedged
# or gone, never one that is about to settle normally.
# ---------------------------------------------------------------------------

CUSTODY_REAPER_GRACE_SECONDS = 300.0

#: Author-leg attempt ceiling used ONLY to size the leg's runaway fuse: the
#: base 3-attempt allowance plus one bounded-effort grant of headroom. A model
#: with more granted attempts still buys them (the fuse is leg-level, not the
#: per-model valve); the fuse only has to sit ABOVE every honest leg.
AUTHOR_LEG_FUSE_ATTEMPT_CEILING = 4


def author_leg_runaway_budget(
    member_count: int, wall_grant_seconds: float
) -> "tuple[int, float]":
    """Return the author leg's runaway fuse ``(sessions, wall_seconds)``.

    The 2026-08-20 step-200 leg halted at 29.84h of accumulated author wall
    against ``CHECKER_LEG_WALL_BUDGET_SECONDS`` -- a fuse borrowed from the
    checker lane and sized for ITS leg shape. An author leg at concurrency 12
    accumulates 12 wall-hours per clock-hour, so the borrowed 30h fuse is a
    ~2.5-clock-hour trip on a PERFECTLY HEALTHY step-200 leg, and the
    2026-08-20 session-abort storm merely reached it first.

    The honest leg-level fuse is the leg's own theoretical maximum spend:
    every member exhausting its ceiling of attempts, every attempt buying its
    full complement of guard sessions, every session running to its external
    kill wall. Anything beyond that is provably runaway, whatever the cause.
    The checker-lane constants remain the FLOOR so a tiny leg keeps a
    meaningful fuse.

    Parameters
    ----------
    member_count:
        Members of the leg's frozen intake snapshot.
    wall_grant_seconds:
        The campaign's per-session wall grant
        (:func:`resolve_author_wall_seconds`).

    Returns
    -------
    tuple[int, float]
        ``(runaway_sessions, runaway_wall_seconds)`` for the author lane's
        :class:`~menagerie.crawler.paid_session_lifecycle.BudgetPolicy`.
    """

    derived_sessions = (
        int(member_count)
        * AUTHOR_LEG_FUSE_ATTEMPT_CEILING
        * AUTHOR_PAID_SESSIONS_PER_ATTEMPT
    )
    sessions = max(CHECKER_LEG_SESSION_BUDGET, derived_sessions)
    # The wall derives from the LEG's own session count, not the floored one:
    # flooring first would balloon a tiny leg's wall fuse to the big-leg shape
    # and the floor below would never bind.
    wall_seconds = max(
        float(CHECKER_LEG_WALL_BUDGET_SECONDS),
        float(derived_sessions)
        * float(wall_grant_seconds)
        * AUTHOR_WALL_EXTERNAL_KILL_FACTOR,
    )
    return sessions, wall_seconds

# A billing failure has no reset window: money does not refill on a schedule, so the
# recheck exists only to catch an operator refill. Six hours keeps the wake loop from
# buying one failed author session per hour forever (the guessed one-hour quota reset)
# while still noticing a refill the same day.
AUTHOR_BILLING_RECHECK_SECONDS = 6 * 60 * 60

# Three different models failing the exact research-tool guard in one wave is an
# infrastructure outage, not independent model trouble. One or two remain retryable:
# they can be transient namespace/session faults, while three supplies cross-model
# evidence and still fits inside the default author wave width of four.
RESEARCH_TOOL_OUTAGE_THRESHOLD = 3

# A forge rate limit is global, not per-model: once the hourly budget is gone,
# every subsequent ref resolution fails until it resets. The same cross-model
# evidence bar as the research-tool outage applies, for the same reason -- one
# throttled target can be a single odd host, three consecutive distinct models
# cannot -- and the individual failures stay retryable in the meantime. Grinding
# 28,482 models into an hour-long wall is the failure economics this prevents.
FORGE_RATE_LIMIT_OUTAGE_THRESHOLD = 3

# Closed usage-limit provider vocabulary. The author lane pauses on `anthropic`;
# the checker lane pauses on the provider of the SEATED checker model it was
# adjudicating under (`checker_provider_for_model`: `anthropic` for the Fable
# primary seat, `openai` for the sol confirmation seat); research-tool outages
# are not usage limits.
USAGE_LIMIT_PROVIDERS = frozenset({"anthropic", "openai"})

CHECKER_PROVIDER_ANTHROPIC = "anthropic"
CHECKER_PROVIDER_OPENAI = "openai"

#: The closed POSITIVE vocabulary of Anthropic-adjudicated checker models. The
#: checker provider is DERIVED from the seated model everywhere an identity is
#: sealed (envelope builder, authority context, wrapper preflight, backoff
#: signal) -- never declared by a caller and never read back from a result.
#: Every checker identity recorded before the 2026-09-01 seat swap carried
#: `openai`, so `openai` stays the derivation for every label outside this
#: set: no historical identity moves, and an unseated label still cannot buy
#: a session (the wrapper refuses any model not seated for its gate kind).
ANTHROPIC_CHECKER_MODELS = frozenset({"claude-fable-5-1"})


def checker_provider_for_model(model: str) -> str:
    """Return the closed usage-limit provider that adjudicates one checker model.

    Parameters
    ----------
    model:
        Exact checker model label.

    Returns
    -------
    str
        ``"anthropic"`` for a member of :data:`ANTHROPIC_CHECKER_MODELS`, else
        ``"openai"`` (the era default every pre-swap identity carries).
    """

    if model in ANTHROPIC_CHECKER_MODELS:
        return CHECKER_PROVIDER_ANTHROPIC
    return CHECKER_PROVIDER_OPENAI

# Every provider identity the scheduled-recheck layer may wake. A sustained
# research-tool outage or forge rate limit shares the reset/recheck mechanism
# without misclassifying itself as a usage limit. `forge` is deliberately its own
# identity rather than folded into the author provider: throttling by GitHub is
# not the LLM provider's quota, and saying so would be the same wrong-party
# defect this pause exists alongside.
SCHEDULED_RECHECK_PROVIDERS = frozenset({*USAGE_LIMIT_PROVIDERS, "research-tools", "forge"})

# The four frozen TIER campaigns the partitioner emits, each bound to its frozen
# author model. This is deliberately NOT the same concept as a *repair* campaign
# (`campaign-<stable_id>` / `campaign-<work_id>`), which is the driver's per-item
# authority lineage. A tier campaign is a property of the whole campaign run: its
# `author_model_identity` is frozen for the run, so a model a sonnet campaign finds
# genuinely hard is emitted as a typed BLOCKED recommendation and requeued into the
# opus campaign, never escalated in place. Mixing the two identities up would author
# a model with the wrong tier and corrupt the frozen identity for the whole run, so
# every boundary that selects an author tier validates against this closed set.
# `menagerie.crawler.partitioner.CAMPAIGN_SPECS` carries the same binding for the
# partitioner's own purposes; the two are asserted to agree in the test suite.
#
# c1-mech is bound to claude-fable-5-1 per the standing directive (JMT 2026-08-11:
# smartest Claudes for the whole crawl, through ~2026-12; JMT 2026-09-01: every
# crawler-role fable-5 pin moves to 5.1 on release, CLI 2.1.251+). claude-opus-5 is the
# DESIGNATED FALLBACK TIER if Fable throughput/529 weather binds on long legs
# (MASTER_DOCKET Appendix 1, call 1); switching a campaign to it is a frozen-
# identity change and warms on a fresh rung, never mid-run.
TIER_CAMPAIGN_AUTHOR_MODELS: dict[str, str] = {
    "c1-mech": "claude-fable-5-1",
    "c2-disco": "claude-sonnet",
    "c3-classics": "claude-opus-5",
    "c4-native": "claude-sonnet",
}
TIER_CAMPAIGN_IDS = frozenset(TIER_CAMPAIGN_AUTHOR_MODELS)

# Closed per-campaign author-model HISTORY, oldest binding first: every model
# that has ever been a campaign's frozen author binding. Historical records are
# append-only and stay unrewritten, so a directed mid-campaign switch (c1-mech
# sonnet -> fable, 2026-08-11; fable -> fable-5.1, 2026-09-01) leaves rows whose
# provenance truthfully names the
# PRIOR binding; validators that check record labels accept exactly this closed
# set -- never one model, never an open vocabulary -- and the census interprets
# which rows predate the switch. The current binding is always the last entry.
TIER_CAMPAIGN_AUTHOR_MODEL_HISTORY: dict[str, tuple[str, ...]] = {
    "c1-mech": ("claude-sonnet", "claude-fable-5", "claude-fable-5-1"),
    "c2-disco": ("claude-sonnet",),
    "c3-classics": ("claude-opus-5",),
    "c4-native": ("claude-sonnet",),
}

# Closed per-campaign CHECKER-model history, exactly the author-history shape and
# for exactly the author-history reason (BL-3). The B-05 sole-checker flip
# (`f08e8bdb`, JMT 2026-08-14 Terra retirement) rotated the frozen checker
# binding from `gpt-5.6-terra` to `gpt-5.6-sol` mid-campaign. The pilot's 298
# append-only model revisions -- and its frozen `campaigns.json`, written
# 2026-08-11 -- truthfully name the PRIOR binding, and rewriting append-only
# history would be worse than reading it. Before this history existed the flip
# made three areas refuse at once: every catalog entry point tracebacked at
# manifest load, `campaign_merge` refused all 10 live c1 models (including m4334
# and m5273, two of the three award-eating models), and a rename hard-refused
# every operator command including `status`.
#
# The set stays CLOSED -- an unregistered rename still fails closed, which is the
# property the refusal existed for -- and the current binding is always the last
# entry, so `history[-1] == CAMPAIGN_SPECS[i].checker_model` is an asserted
# invariant rather than a convention. c3-classics never ran under Terra.
#
# Seat swap (JMT 2026-09-01, measurement-gated; SEAT_RULING.md in the
# tl-checkerbench lane): every campaign's PRIMARY checker rotates from
# `gpt-5.6-sol` to `claude-fable-5-1` (item 32 / J6). Era-honest exactly like
# the Terra->sol flip: every award recorded under sol stays an award under its
# recorded identity; the row appended here is what lets a sol-labelled record
# keep validating. sol stays seated as the cross-model K=2 CONFIRMATION seat
# (`CHECKER_SEATS`), so it is a live identity, not a retired one.
TIER_CAMPAIGN_CHECKER_MODEL_HISTORY: dict[str, tuple[str, ...]] = {
    "c1-mech": ("gpt-5.6-terra", "gpt-5.6-sol", "claude-fable-5-1"),
    "c2-disco": ("gpt-5.6-terra", "gpt-5.6-sol", "claude-fable-5-1"),
    "c3-classics": ("gpt-5.6-sol", "claude-fable-5-1"),
    "c4-native": ("gpt-5.6-terra", "gpt-5.6-sol", "claude-fable-5-1"),
}

# The campaign manifest has two KINDS of field and conflating them is what BL-3
# was. A BINDING field addresses frozen bytes: it is verified against those bytes
# and never rewritten. A PROJECTION field is a read-time echo of the mutable
# in-code `CampaignSpec`: comparing it to a frozen file turns every spec rotation
# into a total refusal of artifacts that are perfectly intact. Projections are
# therefore resolved LIVE and never compared for equality; the identity-bearing
# two are additionally constrained to their campaign's closed declared history so
# an unregistered label still refuses.
#
# Every field of `CampaignSpec` other than `campaign_id` must appear in exactly
# one of these sets -- a new spec field that is classified in neither is a
# tripwire failure, not a silently unvalidated field.
CAMPAIGN_SPEC_BINDING_FIELDS: tuple[str, ...] = (
    "row_count",
    "partition_path",
    "partition_sha256",
    "intake_path",
    "intake_snapshot_id",
    "intake_snapshot_sha256",
)
CAMPAIGN_SPEC_PROJECTION_FIELDS: tuple[str, ...] = (
    "workload_class",
    "phase",
    "author_model",
    "checker_model",
    "review_checkpoint_at",
)

#: The projections that name an adjudicating authority. These are the ones a
#: rotation can silently re-attribute, so they are history-constrained on load
#: and durably attested per leg.
CAMPAIGN_SPEC_IDENTITY_PROJECTIONS: tuple[str, ...] = ("author_model", "checker_model")

# Every projection that steers adjudication must be durably attested by the
# leg that ran under it (B-09 `leg-config`), because the manifest's own copy is
# a mutable echo: after a rotation the manifest alone cannot say which model
# adjudicated which rung. `workload_class` is deliberately absent -- it is a
# human label consumed by nothing but manifest emission and validation, so
# `campaign_id` recovers it exactly; adding it would attest a fact no
# adjudication reads.
CAMPAIGN_SPEC_LEG_ATTESTED_PROJECTIONS: tuple[str, ...] = (
    "author_model",
    "author_version",
    "checker_model",
    "checker_version",
    "campaign_id",
    "phase",
    "review_checkpoint_at",
)

# Closed vocabulary of NON-VERSIONED identity literals. A sole authority cannot
# be identified over a multi-week run by an alias plus a word: `checker_version=
# "current"` names 344 lifetime gate rows and every one of the pilot's 298 model
# revisions, so a provider-side model update, a Codex CLI upgrade, or a prompt
# edit is indistinguishable from no change at all, and the W4 reuse key's version
# component is vacuous. Any load-bearing identity field carrying one of these is
# an unpinned identity; the census tripwire refuses a NEW one and the committed
# inventory keeps the surviving ones visible with their owning cluster.
NON_VERSIONED_IDENTITY_LITERALS = frozenset(
    {"current", "latest", "stable", "head", "newest", "default", "unknown"}
)

#: Environment variable naming the tier campaign this operator process serves.
TIER_CAMPAIGN_ENV = "MENAGERIE_CAMPAIGN_ID"


class StrEnum(str, Enum):
    """String-valued enum compatible with all supported Python versions."""


class InvocationOrigin(StrEnum):
    """Closed origin of one driver invocation for wake-episode transitions."""

    ORDINARY_RUN = "ordinary-run"
    MANUAL_RESUME = "manual-resume"
    WAKE_CALLBACK = "wake-callback"
    #: W6: the bounded supervised restart ladder's attested continuation --
    #: the supervisor relaunched the driver after a restartable typed exit,
    #: with the ordinal recorded on the leg_started row and the leg-config.
    SUPERVISED_RESTART = "supervised-restart"


class AuthoredMetadataState(StrEnum):
    """State of source-read fields in a model revision."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    FAILED = "failed"


class StatusKind(StrEnum):
    """Closed public terminal status kinds."""

    RUNS = "runs"
    DEFERRED = "deferred"
    SKIPPED = "skipped"
    FAILED = "failed"
    #: W13: a checker-adjudicated review hold. The member's own claim was
    #: REFUSED (or could not be verified) by the terminal-disposition review,
    #: which is neither "we broke" (``failed``) nor a capability fact
    #: (``deferred``) nor an epistemic world-claim (``skipped``). Splitting it
    #: out ends the R24 conflation where a disposition rejection published as
    #: ``failed:accuracy-gate`` with NO rendered accuracy verdict.
    BLOCKED = "blocked"


class FailureStage(StrEnum):
    """Closed failure stages from the canonical plan."""

    INTAKE = "intake"
    SOURCE = "source"
    FETCH = "fetch"
    AUTHOR = "author"
    EVIDENCE = "evidence"
    ACCURACY_GATE = "accuracy-gate"
    ENVIRONMENT = "environment"
    IMPORT = "import"
    CONSTRUCTOR = "constructor"
    INPUT = "input"
    FORWARD = "forward"
    FIDELITY = "fidelity"
    RESOURCE = "resource"
    POLICY = "policy"
    SANDBOX_UNAVAILABLE = "policy"
    RUNNER = "runner"


class SourceRung(StrEnum):
    """Ordered source-resolution ladder.

    Every member is a *selected* rung: a positive claim that source resolution
    reached this step of the ladder. ``SKIP`` in particular asserts a checked
    conclusion that no faithful source path exists, and carries a separately
    certified epistemic predicate in ``authority``. Nothing outside the ladder
    belongs in this enum, which is why "no rung was selected" is the distinct
    ``NO_RUNG_SELECTED`` sentinel below rather than a sixth member: keeping the
    enum a pure ladder is what makes rung ordering, author-proposal parsing, and
    the R5 predicate reject the sentinel automatically.
    """

    LIBRARY = "R1_LIBRARY"
    VENDOR = "R2_VENDOR"
    PORT = "R3_PORT"
    REIMPLEMENT = "R4_REIMPLEMENT"
    SKIP = "R5_SKIP"


NO_RUNG_SELECTED = "NO_RUNG_SELECTED"
"""Explicit sentinel for a record where no source rung was ever selected.

Author work can end before the ladder is walked at all -- effort-cap exhaustion, a
crashed or wedged author session, a lane failure. That outcome is a *budget or
process* fact, not a *source* verdict, and recording it as ``R5_SKIP`` would assert
a checked conclusion that no faithful source path exists when none was ever
checked. Consumers count ``source_resolution.rung`` unconditionally, so the
structured field -- not the surrounding narrative -- is what every report reads.
The sentinel makes "no rung was selected" representable and countable as itself.

It is never author-selectable and never satisfies an R5 predicate; see
``SOURCE_RESOLUTION_RUNG_VALUES`` and ``NO_RUNG_SELECTED_STATUS_KINDS``.
"""

SOURCE_RESOLUTION_RUNG_VALUES: frozenset[str] = frozenset(
    {member.value for member in SourceRung} | {NO_RUNG_SELECTED}
)
"""Closed vocabulary admitted by ``source_resolution.rung`` and attempted-rung lists."""


class GateKind(StrEnum):
    """Checker envelope kinds."""

    METADATA_BATCH = "metadata_batch"
    FIDELITY = "fidelity"
    TERMINAL_DISPOSITION = "terminal_disposition"


class CheckerSeatRole(StrEnum):
    """The two roles a seated checker model plays on one gate kind."""

    PRIMARY = "primary"
    CONFIRMATION = "confirmation"


@dataclass(frozen=True)
class CheckerSeat:
    """One seated checker model for one ``(gate kind, role)``.

    Parameters
    ----------
    model:
        Exact checker model label the wrapper pins the session to.
    provider:
        Derived usage-limit provider (:func:`checker_provider_for_model`).
    confirmation_sample_rate:
        For the CONFIRMATION role: the deterministic fraction of primary
        accepts that buy a blind cross-model confirmation (``1.0`` = every
        accept). Primaries carry ``1.0`` by construction (a primary session is
        never sampled away).
    """

    model: str
    provider: str
    confirmation_sample_rate: float = 1.0


#: RULING checker row (2026-09-01): terminal-disposition K=2 is DEMOTED to a
#: 10 percent deterministic sample of primary accepts; the sample measures the
#: label-difference rate the bench saw on refusals (SEAT_RULING.md).
TERMINAL_CONFIRMATION_SAMPLE_RATE = 0.10

_FABLE_PRIMARY = CheckerSeat("claude-fable-5-1", checker_provider_for_model("claude-fable-5-1"))
_SOL_CONFIRMATION = CheckerSeat("gpt-5.6-sol", checker_provider_for_model("gpt-5.6-sol"))

#: The seat table -- item 32's per-(kind, role) seats, the ONE owner of which
#: model adjudicates which gate kind in which role. Fable 5.1 is the PRIMARY on
#: all three kinds while MAX-POWER lasts; gpt-5.6-sol is the cross-model K=2
#: CONFIRMATION seat on metadata and fidelity accepts (every accept) and on a
#: 10 percent sample of terminal dispositions. Closed and exhaustive over
#: ``GateKind x CheckerSeatRole``; the wrapper refuses any envelope whose model
#: is seated in neither role for its kind.
CHECKER_SEATS: dict[tuple[GateKind, CheckerSeatRole], CheckerSeat] = {
    (GateKind.METADATA_BATCH, CheckerSeatRole.PRIMARY): _FABLE_PRIMARY,
    (GateKind.METADATA_BATCH, CheckerSeatRole.CONFIRMATION): _SOL_CONFIRMATION,
    (GateKind.FIDELITY, CheckerSeatRole.PRIMARY): _FABLE_PRIMARY,
    (GateKind.FIDELITY, CheckerSeatRole.CONFIRMATION): _SOL_CONFIRMATION,
    (GateKind.TERMINAL_DISPOSITION, CheckerSeatRole.PRIMARY): _FABLE_PRIMARY,
    (GateKind.TERMINAL_DISPOSITION, CheckerSeatRole.CONFIRMATION): CheckerSeat(
        _SOL_CONFIRMATION.model,
        _SOL_CONFIRMATION.provider,
        confirmation_sample_rate=TERMINAL_CONFIRMATION_SAMPLE_RATE,
    ),
}


def checker_seat(gate_kind: "GateKind | str", role: "CheckerSeatRole | str") -> CheckerSeat:
    """Return the seated checker for one ``(gate kind, role)``, fail-closed.

    Parameters
    ----------
    gate_kind:
        Metadata, fidelity, or terminal-disposition gate.
    role:
        Primary or confirmation.

    Raises
    ------
    KeyError
        Never for a closed-vocabulary pair; the table is exhaustive.
    """

    kind = gate_kind if isinstance(gate_kind, GateKind) else GateKind(gate_kind)
    seat_role = role if isinstance(role, CheckerSeatRole) else CheckerSeatRole(role)
    return CHECKER_SEATS[(kind, seat_role)]


def seated_checker_models(gate_kind: "GateKind | str") -> frozenset[str]:
    """Return every model seated (in any role) on one gate kind."""

    kind = gate_kind if isinstance(gate_kind, GateKind) else GateKind(gate_kind)
    return frozenset(
        seat.model for (seated_kind, _role), seat in CHECKER_SEATS.items() if seated_kind is kind
    )


def checker_confirmation_partner(
    gate_kind: "GateKind | str", primary_model: str
) -> "CheckerSeat | None":
    """Return the cross-model K=2 partner of one primary model, or ``None``.

    ``None`` means ``primary_model`` is not the seated PRIMARY for this kind
    (a test lane's ``codex`` label, or a legacy sol-primary record): its K=2
    partner is then the SAME identity, exactly the pre-swap duplicate-accept
    rule, so historical pairs keep validating and test lanes keep their shape.
    A seated primary's partner is the seated CONFIRMATION seat, and the pair
    MUST be cross-model: two Fable sessions are never a lawful K=2 pair.
    """

    kind = gate_kind if isinstance(gate_kind, GateKind) else GateKind(gate_kind)
    if CHECKER_SEATS[(kind, CheckerSeatRole.PRIMARY)].model != primary_model:
        return None
    return CHECKER_SEATS[(kind, CheckerSeatRole.CONFIRMATION)]


def terminal_confirmation_sampled(
    stable_id: str, author_result_sha256: str, *, rate: float
) -> bool:
    """Decide deterministically whether one accept buys its blind confirmation.

    Pure function of the adjudicated fact (member + author result content) and
    the seated rate: replays and resumes re-derive the same answer, so no
    persisted flag steers it and the census can recompute who was sampled
    from the gates ledger alone. ``rate >= 1.0`` always samples; ``rate <= 0``
    never does.
    """

    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(
        f"terminal-confirmation-sample\x00{stable_id}\x00{author_result_sha256}".encode(
            "utf-8"
        )
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return draw < rate


# AG-6: the closed per-lane ABSOLUTE round-ceiling inventory. Every gate lane
# gets the landed metadata mechanism's shape -- pre-purchase check, park as
# typed checker disclosure debt, never a model-blaming ``failed:*`` terminal --
# because chargeability bounds author incorrigibility only: a checker that
# keeps disclosing fresh complaints is never chargeable and holds ANY lane
# open forever (200a: m4257 burned 12 all-rejected terminal-disposition
# rounds; m1359/m5445 ran 7 fidelity rounds). Six mirrors the B-1 derivation
# on every lane (1 initial + 2 chargeable repairs + 3 free first disclosures
# is already tail behavior). The mapping is CLOSED over GateKind: a new gate
# kind cannot ship uncapped (closed-inventory test), and an unknown kind
# looked up here fails closed with a KeyError.
GATE_ROUND_CEILINGS: Mapping[str, int] = {
    GateKind.METADATA_BATCH.value: METADATA_GATE_ROUND_CEILING,
    GateKind.FIDELITY.value: 6,
    GateKind.TERMINAL_DISPOSITION.value: 6,
}


class AccuracyVerdict(StrEnum):
    """Closed metadata/integrity checker verdicts."""

    ACCURATE = "accurate"
    INACCURATE = "inaccurate"
    CANNOT_VERIFY = "cannot-verify"


class FidelityVerdict(StrEnum):
    """Closed fidelity checker verdicts."""

    MATCH = "match"
    MINOR_DRIFT = "minor-drift"
    MAJOR_DRIFT = "major-drift"
    SLOP = "slop"
    CANNOT_VERIFY = "cannot-verify"


class RetrievalStatus(StrEnum):
    """Closed controlled-fetch retrieval outcomes."""

    FETCHED = "fetched"
    ALREADY_PRESENT = "already-present"


class GateRoute(StrEnum):
    """Closed deterministic routes after a checker verdict."""

    ACCEPT = "accept"
    REQUEUE_NEXT_BATCH = "requeue-next-batch"
    HUMAN_FAIL = "human-fail"
    BLOCK_FIDELITY = "block-fidelity"


class CheckerPauseReason(StrEnum):
    """Closed checker responses that require a scheduler pause."""

    RATE_LIMIT = "rate-limit"
    QUOTA_EXHAUSTED = "quota-exhausted"
    # The checker provider threw the operator out for WHO it is, not for how much it
    # asked: an expired OAuth login, a revoked key. Distinct from both quota members
    # because the remedy is re-authentication, never waiting -- and because parking
    # models over it is wrong-blame (T4-T3: an expired login classified PERMANENT
    # parked every touched model as `permanent-contract-rejection`).
    AUTH_FAILURE = "auth-failure"


class AuthorPauseReason(StrEnum):
    """Closed author responses that require a scheduler pause.

    The author-side analogue of :class:`CheckerPauseReason`. Anthropic usage
    exhaustion, a sustained cross-model research-provider outage, and a sustained
    forge rate limit are campaign pauses, never model failures.
    """

    RATE_LIMIT = "rate-limit"
    QUOTA_EXHAUSTED = "quota-exhausted"
    # An exhausted credit balance, not a usage window. `QUOTA_EXHAUSTED` implies a
    # reset instant the scheduler can wake on; billing has none -- the guessed
    # now-plus-one-hour reset never clears on its own, so the distinct reason buys
    # the sparse operator recheck instead of an hourly futile wake.
    BILLING_FAILURE = "billing-failure"
    RESEARCH_TOOLS_UNAVAILABLE = "research-tools-unavailable"
    # The source forge threw the broker out. Distinct from `RATE_LIMIT`, which is
    # the author *provider* throttling us: the two have different resets,
    # different remedies, and blaming one for the other is a false record.
    FORGE_RATE_LIMITED = "forge-rate-limited"


class RunMode(StrEnum):
    """Meaningful model runtime modes."""

    TRAIN = "train"
    EVAL = "eval"


class AttemptResult(StrEnum):
    """Immutable attempt outcomes."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OBSERVED = "observed"


class EnvironmentPhase(StrEnum):
    """Ordered environment execution phases."""

    PYTORCH = "pytorch"
    NATIVE_TAIL = "native-tail"


class PlatformRequirement(StrEnum):
    """Platform capabilities that can support an evidenced deferral.

    ``BACKEND_PARITY`` is the capability "an environment lane exists for this model's
    framework on this platform". It is a member so backend deferrals reuse the one
    evidence record every other capability deferral uses, and it is deliberately NOT
    mintable through ``routing.defer_for_platform``: its proof is a machine-derived
    re-derivation of the framework capability registry, so it has exactly one mint point
    (``routing.backend_parity_deferral``). See
    :data:`BACKEND_PARITY_DEFERRAL_STATUS_CODE`.
    """

    CUDA = "cuda"
    X86 = "x86"
    BACKEND_PARITY = "backend-parity"


#: Closed native-framework markers searched in a roster row's zoo and name. Any
#: hit routes the row to :attr:`EnvironmentPhase.NATIVE_TAIL`; everything else is
#: PyTorch work.
NATIVE_FRAMEWORK_MARKERS: tuple[str, ...] = ("tensorflow", "keras", "jax", "flax", "paddle")

#: Canonical framework spellings this derivation may return, mirroring
#: :class:`menagerie.crawler.backend.Framework`. Mirrored rather than imported because
#: ``constants`` must stay at the bottom of the import graph; the dispatch-arm census
#: asserts the two are identical, so drift is a test failure.
CANONICAL_FRAMEWORK_VALUES: tuple[str, ...] = (
    "pytorch",
    "tensorflow",
    "jax",
    "paddle",
    "mlx",
    "tinygrad",
)

#: Marker spelling -> canonical framework. ``keras`` is TensorFlow and ``flax`` is JAX:
#: the shipped derivation returned the MARKER as if it were a framework, so ``"keras"``
#: reached routing (which accepted it), ``standard_inputs._convert_array`` (which raised
#: on it) and ``worker._seed_frameworks`` (which silently seeded nothing) -- one string,
#: three different behaviours (AREA_BACKENDS_opus B1).
_FRAMEWORK_MARKER_ALIASES: tuple[tuple[str, str], ...] = (
    ("tensorflow", "tensorflow"),
    ("tf-keras", "tensorflow"),
    ("keras", "tensorflow"),
    ("tf", "tensorflow"),
    ("paddlepaddle", "paddle"),
    ("paddle", "paddle"),
    ("flax", "jax"),
    ("jax", "jax"),
    ("pytorch", "pytorch"),
    ("torch", "pytorch"),
)
# ``mlx`` and ``tinygrad`` are deliberately ABSENT from the marker table even though they
# are members of the closed framework vocabulary. This table is the driver's STAGING prior:
# `route_model` is called over every intake row inside the leg, with no handler above it, so
# a marker that resolves to a framework with no environment lane would turn one roster row
# (the census finds exactly one ``mlx`` row and zero ``tinygrad`` rows) into a typed refusal
# that kills the whole leg -- strictly worse than the misroute it replaces. Those frameworks
# refuse where a refusal is HANDLED: at the proposal boundary, once the author has read the
# source and declares `run_framework`, which is also the only point where the evidence is
# better than a substring. See `routing.backend_parity_deferral` and the docket's owed
# source-probe wiring.

#: Marker aliases ordered by SPECIFICITY, longest first.
#:
#: The shipped derivation took the first hit in a five-element tuple, so a row whose text
#: contained both ``keras`` and ``tensorflow`` resolved to whichever literal came first --
#: arbitrary precedence over the same string (TRACKB_opus F6). Longest-alias-wins is
#: deterministic and reads the more specific claim.
_MARKER_ALIASES_BY_SPECIFICITY: tuple[tuple[str, str], ...] = tuple(
    sorted(_FRAMEWORK_MARKER_ALIASES, key=lambda item: (-len(item[0]), item[0]))
)

#: ``tf`` fires only as a delimited token. As a bare substring it carried a measured
#: near-total false-positive rate over both corpora (EN-1 census, 2026-08-24): word-
#: internal hits like ``efficientformer``, ``QuotientFourier`` (m4714), ``JetFormer``,
#: ``splatfacto``, ``sortformer``, and ``swiftformer`` are all PyTorch models, and the
#: driver routed them (and m10502) toward the lockless native tail.
_TF_TOKEN_PATTERN = re.compile(r"(?<![a-z0-9])tf(?![a-z0-9])")

#: Positive torch-zoo veto: zoos that are PyTorch-native by construction, so NO name
#: marker may route their rows to the native tail. timm and effdet use ``tf_*`` naming
#: for "ported from TensorFlow weights / SAME padding" -- delimited tokens a marker
#: rule cannot distinguish from a genuine framework claim, which is exactly how m10502
#: (``tf_efficientnet_b5``, zoo timm) died at a lockless env turn after a full paid
#: author pipeline. Membership is CLOSED and evidence-backed by the EN-1 pre-flip
#: census over master_catalog (8,533 rows) and crawl_roster (28,482 rows): every entry
#: either still produced token-level native hits (timm, effdet, NVlabs/stylegan -- the
#: stylegan rows' recipes import torch.nn) or is a torch-only library zoo the census
#: caught in the substring class (escnn, nerfstudio, neuralop, mm-lab family).
#: Widening this set is a reviewed change with fresh census evidence, never a hotfix.
TORCH_NATIVE_ZOO_VETO: frozenset[str] = frozenset(
    {
        "timm",
        "effdet",
        "escnn",
        "nerfstudio",
        "neuralop",
        "mmpretrain",
        "mmsegmentation",
        "mmsegmentation-configs",
        "open-mmlab/mmsegmentation",
        "github:nvlabs/stylegan",
    }
)


def _marker_framework(text: str) -> str | None:
    """Return the most specific canonical framework named in one roster string."""

    for alias, framework in _MARKER_ALIASES_BY_SPECIFICITY:
        if alias == "tf":
            if _TF_TOKEN_PATTERN.search(text) is not None:
                return framework
        elif alias in text:
            return framework
    return None


def framework_from_markers(zoo: str, name: str) -> str:
    """Derive the framework PRIOR for one roster row, zoo declaration first.

    This lives in the import-light constants module deliberately. It is the ONE
    framework derivation shared by the driver's phase scoping and by the
    read-only rung staging tools, and a read-only records composition must not
    have to import the driver (and through it the reducer and numpy) to ask
    which phase a roster row belongs to.

    It is a PRIOR, not a fact. The evidence is two roster strings, read before anyone
    fetches the source, and it carries a measured 13% false-positive rate: 8 of the 63
    rows the shipped version sent to the native tail are provably PyTorch
    (``paddleseg_*``/``paddledet_*`` reimplementations, 41 of 42 importing torch), and
    their own ``zoo`` field said so -- ``unregistered-classics-pytorch``. Two rules follow,
    and both are about not letting the weakest evidence outrank better evidence sitting
    right next to it:

    1. **The row's own zoo declaration wins.** A zoo naming a framework is a curated
       provenance field; a framework name inside a MODEL name is frequently just the
       architecture being reimplemented (PaddleSeg-in-PyTorch is a real and common thing).
    2. **Specificity, not tuple order, decides** among competing markers, and the result is
       always a CANONICAL framework spelling so only one spelling per framework can reach
       an identity preimage.

    What this function still cannot do is see a native marker that appears ONLY in the
    source URL (24 live rows) or a framework with no marker at all (~120 rows). Those need
    the source probe, which is why every consumer that decides EXECUTION must carry a
    :class:`menagerie.crawler.backend.FrameworkFact` basis rather than this string alone.

    Parameters
    ----------
    zoo, name:
        Immutable roster identity fields of one intake row.

    Returns
    -------
    str
        A canonical framework spelling; ``"pytorch"`` when nothing names a framework.
    """

    zoo_text = str(zoo).strip().lower()
    if zoo_text in TORCH_NATIVE_ZOO_VETO:
        # Rule 1 in its strongest form: the zoo is torch-native by construction, so a
        # framework name inside a MODEL name is provenance trivia, never a route.
        return "pytorch"
    declared = _marker_framework(zoo_text)
    if declared is not None:
        return declared
    named = _marker_framework(str(name).strip().lower())
    if named is not None:
        return named
    return "pytorch"


class ProducerVocabularyError(ValueError):
    """Raised when a producer-written string outside a closed vocabulary reaches a sink.

    The C-22 class: any producer-supplied string that steers a dispatch or enters an
    identity is refused typed at the boundary, never coerced. See
    :data:`RECIPE_TYPES` and :data:`RECIPE_REVISION_PATTERN`.
    """


RECIPE_TYPES: frozenset[str] = frozenset(
    {
        "declarative-library",
        "typed-adapter",
        "port",
        "reimplementation",
        "none",
    }
)
"""Closed recipe kinds. Author-gated, dispatched on, AND hashed into execution identity.

``recipe.load_recipe`` branches on this string and the execution identity preimage
contains it, so it is a C-22 member by both tests. The schema enum was already closed;
this is the Python-side authority that makes the closure executable, so the producer-field
census can prove the refusal rather than read a schema and hope.
"""

RECIPE_REVISION_PATTERN = r"^sha256:[0-9a-f]{64}$"
"""Closed grammar for ``recipe_revision``, another author-gated identity input.

A revision cannot be an enum, so its closure is a PATTERN -- which is a closed vocabulary
in exactly the sense C-22 requires: the set of accepted strings is fixed and a reviewed
change is needed to widen it. Anything that is not a content hash cannot enter the
execution identity.
"""


def normalize_recipe_type(value: object) -> str:
    """Return a validated recipe type, refusing anything outside the closed set.

    Parameters
    ----------
    value:
        Producer-written recipe kind.

    Returns
    -------
    str
        The value unchanged.

    Raises
    ------
    ProducerVocabularyError
        For any spelling outside :data:`RECIPE_TYPES`.
    """

    text = value if isinstance(value, str) else ""
    if text not in RECIPE_TYPES:
        raise ProducerVocabularyError(
            f"recipe_type {value!r} is outside the closed vocabulary {sorted(RECIPE_TYPES)}"
        )
    return text


def normalize_recipe_revision(value: object) -> str:
    """Return a validated recipe revision, refusing anything but a content hash.

    Parameters
    ----------
    value:
        Producer-written recipe revision.

    Returns
    -------
    str
        The value unchanged.

    Raises
    ------
    ProducerVocabularyError
        When the value does not match :data:`RECIPE_REVISION_PATTERN`.
    """

    if not isinstance(value, str) or re.fullmatch(RECIPE_REVISION_PATTERN, value) is None:
        raise ProducerVocabularyError(
            f"recipe_revision {value!r} must match {RECIPE_REVISION_PATTERN}"
        )
    return value


SKIPPED_STATUS_CODES = frozenset(
    {
        "skipped:insufficient-description",
        "skipped:no-description",
        "skipped:not-a-real-NN",
    }
)

ACCESS_BARRIER_REJECTION_CLASS = "access-barrier"
"""The one candidate-rejection class that asserts a locator was WITHHELD, not absent."""

ACCESS_BLOCKED_REASON_CODE = "needs-source-access"
"""BLOCKED reason routing an access-barred model onto its own terminal.

The arm goes through BLOCKED, exactly like ``needs-higher-tier``, rather than through a
``DeferRecommendation``: a platform defer mandates a ``handoff_execution``, and there is
nothing to hand off when the blocker is that we could not read the material at all.
"""

BLOCKED_PREREQUISITE_REASON_CODE = "blocked-prerequisite"
"""Record reason for a checked BLOCKED verdict whose author reason is off-vocabulary.

The author-result blocked arm's ``reason_code`` is an open checker-adjudicated string
("something a frozen source can witness"), while this record vocabulary is closed. The
total crossing in ``driver._blocked_terminal`` previously collapsed every unrecordable
blocked reason onto ``failed:author``/``malformed-result`` -- asserting an authoring
defect for a verdict the checker adjudicated ACCURATE end to end (pilot ``m1777``:
BLOCKED at stage ``environment``, ``missing-runtime-dependency`` minkowskiengine,
terminalized as ``malformed-result``). This member keeps the crossing total AND honest:
the terminal stays ``failed:<the author's own named stage>`` with a reason that says
exactly what happened -- a checked prerequisite blocked the stage -- while the author's
own reason string and prerequisite IDs survive verbatim on the attempt record's typed
``AuthorBlockedPrerequisite`` cause and the terminal detail. It is a member of every
stage the blocked arm's schema can name (source, fetch, author, evidence, environment,
policy, runner), and of no other stage.

ERA RULE (AL-6, JMT-signed 2026-08-24): this bare member is READ-SIDE ONLY. Historical
rows minted before the stage-suffixed family below stay valid forever, but every NEW
mint spells the stage-suffixed member from
:data:`BLOCKED_PREREQUISITE_STAGE_REASON_CODES`, so the ``(status_code, reason_code)``
pair recovers the blocking stage on every new-era row. A new bare-member mint is a
regression the ``blocked_prerequisite_injectivity_census`` discloses.
"""

#: The closed stage enum of the author-result blocked arm (author-result-v3/v4
#: ``blocked_payload.stage``). Every member is also a :data:`FAILURE_REASON_CODES`
#: stage; the reverse is deliberately false (``import``, ``forward``, ... cannot be
#: named by a BLOCKED verdict).
BLOCKED_ARM_STAGES: tuple[str, ...] = (
    "source",
    "fetch",
    "author",
    "evidence",
    "environment",
    "policy",
    "runner",
)

#: AL-6 stage-suffixed reason family for NEW blocked-prerequisite mints.
#:
#: One member per blocked-arm stage, ``blocked-prerequisite:<stage>``, and each member
#: belongs to exactly TWO stage vocabularies: its own stage (the plain
#: ``failed:<stage>`` arm) and the ``environment-adjacent`` routing stage below -- so
#: the blocking stage is recoverable from the reason suffix even when the status code
#: names the adjacency class rather than the stage itself.
BLOCKED_PREREQUISITE_STAGE_REASON_CODES: dict[str, str] = {
    stage: f"{BLOCKED_PREREQUISITE_REASON_CODE}:{stage}" for stage in BLOCKED_ARM_STAGES
}

ENVIRONMENT_ADJACENT_FAILURE_STAGE = "environment-adjacent"
"""AL-6 routing stage for runtime-dependency-class blocked prerequisites.

A checked BLOCKED verdict whose typed prerequisite IDs name installable runtime
dependencies (pilot: 19 of the 20 reclassified members, e.g. ``m1777``'s
``runtime-dependency-minkowskiengine``) states a fact about the ENVIRONMENT
ECOSYSTEM -- the dependency is not installable here TODAY -- not a defect at the
stage the author happened to be standing in. Recording it as
``failed:environment`` (or ``failed:author``) pooled recoverable ecosystem gaps
with genuine per-stage defects in every triage table. The stage the author named
survives in the reason suffix, so nothing is lost and the pair stays injective.
"""

ENVIRONMENT_ADJACENT_STATUS_CODE = f"failed:{ENVIRONMENT_ADJACENT_FAILURE_STAGE}"
"""Closed terminal status code for the runtime-dependency blocked class (AL-6).

ERA NOTE: introduced 2026-08-24 with the stage-suffixed reason family; no row
minted before that date carries it. It is ``failed:``, not ``deferred:``, on
purpose: unlike CUDA/x86/tier/access there is no NAMED capability whose arrival
re-opens the head -- an installable dependency may become installable at any
lock refresh, which is exactly the operator-adjudicated ``tools/requeue`` path
every ``failed:*`` terminal already rides (the stage is grantable because it is
a :data:`FAILURE_REASON_CODES` key).
"""

ACCESS_BLOCKED_STATUS_CODE = "deferred:needs-source-access"
"""Terminal for a model whose material exists and could not be read.

Before this code there was no honest terminal state for a paper behind a paywall. A
paywall yields no bytes to quote, so ``skipped:insufficient-description`` was unreachable
(it requires a literal retained excerpt), and the model collapsed into
``skipped:no-description`` -- defined as "no descriptive text after bounded search". That
is a FALSE statement recorded as terminal, for a model that is in fact recoverable with
institutional access, on a campaign that runs once.

It is ``deferred:``, not ``failed:fetch/access-denied``. ``deferred:*`` already means
exactly "terminal for this campaign, recoverable given a named capability"
(``needs-cuda``, ``needs-x86``, ``needs-opus-tier``), and access is such a capability.
Filing it under ``failed:`` would record a WORLD-FACT as "our pipeline broke", inviting
retry triage to sweep models where nothing is retryable -- fixing one conflation by
creating another.

It also RAISES the bar rather than lowering it: a paywalled model can no longer be
discharged as a cheap ``skipped:no-description``.
"""

PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE = "deferred:needs-cuda-prerequisites"
"""Terminal for a checked BLOCKED verdict whose prerequisites are all CUDA-bound.

The pilot's environment-stage BLOCKED verdicts named their missing runtime
dependencies verbatim, and a set consisting ENTIRELY of the closed
CUDA-platform class (``environment_coverage.PLATFORM_CLASS_DISTRIBUTIONS``:
triton, flash-attn, spconv, knn-cuda, cuda-runtime) states a capability fact --
"this model needs a CUDA host" -- not a pipeline failure. Terminalizing that
class as ``failed:environment`` invites retry triage on a host that can never
satisfy it (EN-6; the known-19 census is the evidence corpus).

It follows the ``deferred:needs-<capability>`` shape but is deliberately NOT
``deferred:needs-cuda``: that code is proved by the frozen platform-handoff
rule (a checker-adjudicated DEFER with a mandatory ``handoff_execution``), and
one status code must keep exactly one proof rule. This code's proof is the
accepted-BLOCKED capability rule below, exactly like ``needs-opus-tier`` and
``needs-source-access`` -- membership in
:data:`CAPABILITY_DEFERRAL_STATUS_CODES` is what routes it there. The
aggregation that decides it is closed and replay-derivable from the verbatim
prerequisite tokens plus the committed platform class
(``environment_coverage.aggregate_prerequisite_tokens``).
"""

CAPABILITY_DEFERRAL_STATUS_CODES = frozenset(
    {
        "deferred:needs-opus-tier",
        ACCESS_BLOCKED_STATUS_CODE,
        PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE,
    }
)
"""Deferrals proved by an accepted ``BLOCKED`` verdict rather than a platform handoff.

Every member names a real, located model this campaign cannot author and the capability
that would recover it. Their PROOF is identical -- an accepted BLOCKED terminal
disposition over a frozen manifest -- so they share one reducer rule; only the
capability named differs. The frozen platform rule (``needs-cuda`` / ``needs-x86``)
is untouched.

``deferred:needs-more-memory`` is deliberately NOT here. It names a capability like
these two, but its proof is machine-derived rather than checker-adjudicated, so it has
its own rule (:data:`HOST_CAPABILITY_DEFERRAL_STATUS_CODES`). Adding it to this set
would claim a BLOCKED adjudication that no host-capacity refusal has, which is exactly
the fabricated proof ``host_capacity`` refuses to mint.
"""

HOST_CAPABILITY_DEFERRAL_STATUS_CODE = "deferred:needs-more-memory"
"""The one durable spelling for "this HOST cannot instantiate this model".

It is the string already written in every host-capacity valve row, the key the
completion partition buckets on, and the value pinned by the frozen rung-9b/rung-10
accounting fixtures. Two audit syntheses each proposed a NEW name for the same fact
(``deferred:host-capability``, ``deferred:needs-host-memory``); minting either would
fork the ledger vocabulary and silently empty the buckets every existing consumer
reads. The remedy is to promote THIS code to a model terminal, not to rename it.

The canonical definition lives in ``host_capacity.CAPACITY_DEFERRAL_DISPOSITION``,
which imports this constant so the two can never drift.
"""

HOST_CAPABILITY_DEFERRAL_STATUS_CODES = frozenset({HOST_CAPABILITY_DEFERRAL_STATUS_CODE})
"""Deferrals proved by a MACHINE-derived, replay-recomputed capability measurement.

The proof is not a gate: it is the host's own physical-memory figure plus the model's
own declared-configuration parameter bound, recomputed by the reducer on every replay.
That makes the head machine-scoped by construction -- the identical record re-derived on
a bigger host does not refuse, which is precisely the re-openability the disposition
promises. Kept as a set (not a bare code) because the next machine-derived capability
refusal -- disk, accelerator memory -- joins this rule rather than copying it.
"""

BACKEND_PARITY_DEFERRAL_STATUS_CODE = "deferred:needs-backend-parity"
"""The terminal for "this BACKEND cannot execute this model here", which did not exist.

Before this code the closed terminal vocabulary ended at ``failed:{FailureStage}``, and
``FailureStage`` has no capture or replay member, so an honest backend gap had nowhere to
land. It landed as ``failed:forward``. m7960 (paddle) is the shipped instance: both
attempts died at ``stage: "policy"`` with ``reason_code: "checkpoint-read"`` and a
``SandboxDenialObservation``, every receipt stage ``false`` -- and the model head reads
``failed:forward``, a *forward* stage that demonstrably never started, published as a model
defect. ``BACKEND_POLICY_SOL.md`` specified this record and warned in terms: "Do not turn
backend incompleteness into ``failed:forward``, ``needs-environment-coverage``, or an
epistemic skip."

Why ``deferred:``, in the exact terms the sibling codes already use: ``deferred:*`` means
"terminal for this campaign, recoverable given a named capability", and a backend lane is
such a capability. Filing it under ``failed:`` records a MACHINERY fact as a MODEL defect,
and because a terminal is settled, the row would never be requeued when the backend lifts
-- a permanent hole in the catalog with a misleading cause. The exposed set is the 1,047-row
``c4-native`` partition plus the paddle rows inside ``c1-mech``.

Its proof follows the :data:`HOST_CAPABILITY_DEFERRAL_STATUS_CODE` template exactly, and
that choice is load-bearing: the proof is MACHINE-derived and recomputed on every replay
(the framework's own capability row plus the environment registry's intents), never a
checker adjudication and never a producer-written flag. That makes the head
lane-scoped by construction -- the identical record re-derived once an intent exists does
not refuse, which is the re-openability the disposition promises. Nothing a producer writes
may steer it.
"""

BACKEND_PARITY_DEFERRAL_STATUS_CODES = frozenset({BACKEND_PARITY_DEFERRAL_STATUS_CODE})
"""Deferrals proved by a machine-derived BACKEND-capability derivation.

Kept as a set, like the host-capability rule it mirrors, because the next backend-scoped
refusal -- a conformance-digest regression, a preview withdrawn on a platform -- joins this
rule rather than copying it.
"""

DECLARED_CONFIG_CAPACITY_ESTIMATE_BASIS = "declared-config-layer-stack-lower-bound"
"""The static parameter-bound provenance that may mint a host-capability terminal.

A refusal derived from the authored recipe's own declared layer stack is a fact about
the MODEL and the HOST, decidable without running anything, and re-decidable anywhere.
"""

WORKER_OBSERVED_CAPACITY_ESTIMATE_ORIGIN = "worker-observed-rss"
"""The measured-peak provenance that may NEVER mint a host-capability terminal.

An observed-RSS refusal says a run on THIS host, in THIS environment, at THIS moment
climbed past the enforced cap. m7637 carries four such rows across generations -- and 21
attempts, 5 gates, 13 artifacts and a ``failed:environment`` head. Terminalizing that
class would have frozen a recoverable member out of three later generations, so the
observed origin stays withheld-not-settled: durable row, no head.
"""

LINK_EVIDENCED_DISCOVERY_STATUS_CODES = frozenset(
    {
        "skipped:no-description",
        "skipped:insufficient-description",
    }
)
"""Typed-discovery terminals whose claim is only falsifiable against named locators.

Both codes assert something about the WORLD -- "no descriptive text exists after a
bounded search", "what exists is too vague to specify a forward pass" -- and the record
is checkable only if it names the locators that were actually looked at.

Before this, ``search_report`` had to carry non-empty ``queries``, ``places_checked`` and
``languages_checked``, but ``links_checked`` was unconstrained and
``search_evidence.candidate_links`` carries no ``minItems``. So the CHEAPEST valid skip
listed zero candidate locators -- and zero candidates makes
``discovery._probe_discovery_candidates`` return ``None``, so no machine probe receipt
could physically exist to contradict it. An author that honestly listed the paywalled DOI
it hit did strictly more work and invited strictly more checking than one that listed
nothing: an incentive pointing exactly the wrong way on a campaign that runs once.

``skipped:not-a-real-NN`` is deliberately excluded: concluding that a roster name denotes
no neural network at all can legitimately rest on searches that surfaced nothing worth
retaining. ``deferred:*`` codes are excluded too -- they assert no world-fact and hand the
search on unfinished -- except where the deferral's whole content IS a locator that could
not be read, which supplies its own stricter evidence invariant.
"""

TERMINAL_STATUS_CODES = frozenset(
    {
        "runs",
        "deferred:needs-cuda",
        "deferred:needs-x86",
        *CAPABILITY_DEFERRAL_STATUS_CODES,
        *HOST_CAPABILITY_DEFERRAL_STATUS_CODES,
        *BACKEND_PARITY_DEFERRAL_STATUS_CODES,
        *SKIPPED_STATUS_CODES,
        *(f"failed:{stage.value}" for stage in FailureStage),
        # AL-6: a routing stage, deliberately NOT a FailureStage member -- no
        # retry ladder, no per-stage repair loop ever runs "at" it; it exists
        # only as a terminal spelling for the runtime-dependency blocked class.
        ENVIRONMENT_ADJACENT_STATUS_CODE,
        # W13: the disposition-review hold (the literal spelling; the named
        # constant BLOCKED_DISPOSITION_REVIEW_STATUS_CODE is defined below and
        # a test pins the two equal).
        "blocked:disposition-review",
    }
)

NO_RUNG_SELECTED_STATUS_KINDS: frozenset[str] = frozenset(
    {StatusKind.FAILED.value, StatusKind.DEFERRED.value, StatusKind.BLOCKED.value}
)
"""Status kinds under which ``NO_RUNG_SELECTED`` is a coherent record fact.

A ``runs`` record necessarily selected a rung to be runnable, and every
``skipped:*`` code is an epistemic claim that must carry a checked ``R5_SKIP``
with its certified predicate. The sentinel is therefore only valid where work
genuinely stopped short: ``failed:*`` and ``deferred:*``. Enforcing this makes
the tripwire stricter than it was -- previously a cap-exhausted record and a
checked no-source conclusion were indistinguishable in the structured field.
"""

TERMINAL_DISPOSITION_REJECTED_REASON_CODE = "terminal-disposition-rejected"
"""The checker adjudicated one advisory terminal claim and refused it ON THE MERITS.

The author published a SKIP/DEFER/BLOCKED recommendation; the checker read the frozen
sources it names, and concluded the closed terminal predicate is not established. That is
a real, evidenced campaign outcome and it had no honest terminal before this code.

It is NOT the author's ``skipped:*``/``deferred:*`` claim, which is the very thing just
refused -- recording one would enter a rejected claim into the catalog as fact. It is not
``failed:runner``, which asserts the lane never obtained a verdict when it obtained a
decisive one. It is not ``failed:author``, whose members all describe a session that
crashed, exhausted, or produced something unrecordable, when this session completed and
published a well-formed, expressible result. It is not ``failed:evidence``, which asserts a
specific DEFECT in the evidence pack: the pilot's own rejections verify every excerpt and
hash and still refuse the predicate, so an evidence-stage failure did not occur.
"""

TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE = "terminal-disposition-unverifiable"
"""The checker could not adjudicate one advisory terminal claim either way.

The weaker sibling of :data:`TERMINAL_DISPOSITION_REJECTED_REASON_CODE`, and deliberately
distinct from it: "refused on the merits" and "could not determine" are exactly the
distinction the closed verdict vocabularies already draw everywhere else
(``AccuracyVerdict.INACCURATE`` vs ``CANNOT_VERIFY``), and collapsing them here would
reintroduce the conflation this pair exists to remove.
"""

CHECKER_SESSION_EXHAUSTION_REASON_CODE = "checker-session-exhaustion"
"""The checker lane's cumulative session/wall budget for one work generation is spent.

Minted only from the closed machine-classified QUOTA/budget-exhaustion class (a
``PaidSessionBudgetExceeded`` QUOTA refusal remapped to ``CheckerWallBudgetExceeded``
at the checker admission site -- never message-text matching). Unlike every transient
checker-lane failure, generation-scoped budget exhaustion is PERMANENT for the
generation: the run-scoped checker-lane park's "requeue on the next run" promise is
false for it, so the parked model re-entered and re-parked on every future leg with
no reviewable head a requeue grant could bind (the pilot's five no-exit parks,
m3417/m4893/m5656/m5703/m7646, 2026-08-21). The honest outcome is a
``failed:accuracy-gate`` terminal carrying this reason with mandatory human review:
exactly as reviewable and grant-re-drivable as every other terminal. No sibling
reason fits -- the two ``*-cap-exhausted`` codes assert a bounded REPAIR cap the
checker adjudicated, ``effort-cap-exhausted`` is the author-lane effort grant, and
``checker-contract-invalid`` blames output the checker never produced.
"""

TERMINAL_DISPOSITION_FAILURE_REASONS: dict[str, str] = {
    "rejected": TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
    "cannot-verify": TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE,
}
"""Non-accepted terminal-disposition verdicts mapped to their honest reason codes.

``accepted`` is deliberately absent: it is not a failure at all, and a
``failed:accuracy-gate`` claim over an ACCEPTED disposition is exactly the unevidenced
claim the terminal-proof tripwire exists to refuse.
"""


def terminal_disposition_failure_reason(verdict: object) -> str:
    """Map one non-accepted terminal-disposition verdict onto its honest reason.

    The single crossing between the checker's verdict vocabulary and the record's, shared
    by the driver and the authority kernel so the two can never derive different reasons
    for the same gate. Like ``driver._blocked_terminal`` it is TOTAL: an unreadable verdict
    degrades to the weaker ``unverifiable`` claim -- which is true of an adjudication that
    cannot be read as a refusal on the merits -- rather than aborting. One odd verdict
    string on one model must never take the whole campaign down.

    Parameters
    ----------
    verdict:
        Checker ``terminal_disposition.verdict`` known not to be ``accepted``.

    Returns
    -------
    str
        Reason code valid for the ``accuracy-gate`` stage.
    """

    return TERMINAL_DISPOSITION_FAILURE_REASONS.get(
        str(verdict), TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE
    )


BLOCKED_DISPOSITION_REVIEW_STATUS_CODE = "blocked:disposition-review"
"""W13: the terminal for a checker-REFUSED (or unverifiable) advisory terminal claim.

The author published a SKIP/DEFER/BLOCKED recommendation; the terminal-disposition
checker read the frozen sources and refused the claim on the merits, or could not
adjudicate it either way. Before this code that outcome was minted
``failed:accuracy-gate`` -- a stage label asserting an ACCURACY adjudication of
authored metadata that never happened (the 200a census carries 27 such heads with
``accuracy_gate.verdict = None``; R24's paired-rung mislabel class). The split
restores both labels' meaning:

* ``failed:accuracy-gate`` requires a RENDERED adjudication -- the metadata-repair
  lineage's rejected verdicts (the ``*-cap-exhausted`` family) or the closed
  machine-witnessed ``checker-session-exhaustion`` -- and reduction refuses a fresh
  terminal claiming it with a terminal-disposition reason;
* ``blocked:disposition-review`` is the disposition-review hold: kind ``blocked``,
  no failure stage, reason from :data:`BLOCKED_DISPOSITION_REVIEW_REASON_CODES`,
  mandatory human review (the ``deferred-terminal`` grant sentinel binds it exactly
  like every other stage-less review-held head).

Era honesty (PD-13, forward-only): the 27 historical ``failed:accuracy-gate`` heads
stay readable and re-appendable by the review sweep under their ORIGINAL code; the
refusal fires only on FRESH claims, and the historical shape is disclosed by the
``accuracy_gate_disposition_mislabel`` census counter, never silently normal.
"""

BLOCKED_DISPOSITION_REVIEW_REASON_CODES = frozenset(
    {
        TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE,
    }
)
"""Closed reasons a ``blocked:disposition-review`` terminal may carry."""

DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS = frozenset(
    {
        "sha256:01f8054ec89d382d51a9864b1688b53ced0144c3708bbefa0f8fd52df22c2cdd",
        "sha256:02d443aa460d013f53001fd306cace527c3c96f27e35b5632daf7bea884bc74b",
        "sha256:06007248240d3764b3bf48f2639f74178360f25694aae121f4459361a2e5dda9",
        "sha256:0f67c631ffd95e3ee1d138263b636e83023bd52d9e1b268a922c243cf402c17f",
        "sha256:158964c539484ed99c882c1468ad3509a81537ab2c03bfbce5e657c96ee86109",
        "sha256:1b4eaaacd06ef649efd33b56c3083324ee10cf973bc5f7d379bf46e4c08d89df",
        "sha256:1ead9823a00e00821c580d5dc9cc7b0659ce5f9401bccf60a5eb5efd6f814af3",
        "sha256:24ac595b5b42a1b9314bdfe941516ece8c88e707d5a6d559dcfb4debc8485772",
        "sha256:270062c9105ca396a488aba3377b0a886297961aae5534e250ded0b185244696",
        "sha256:419d75e4ccfdb170957d087761d08e62d3abb72fe92140043cffa0e817514d47",
        "sha256:501b7b1f0acdeb4063827b21eee34c46be14034ac1167c2ca0a791f279b122db",
        "sha256:53865b5bb40ac1f2910e4d13d65d95f6961977d794dfe613377dcd51ef0d0328",
        "sha256:586f059681cb53789dc52ed775604b7b0ba8505acd1bc15707621421d9a97e45",
        "sha256:5efa651faf6ec9c8072539bbc35bf96c413e90d5498e976450eebc5cf2c10c18",
        "sha256:61e38475837ca87703be555c230ffaff2f2b4898ac9de792a3d7472d2404acc0",
        "sha256:639a3d26bb39932a35714580c14c88bf12f200d29b7b03d06c8a359d1a5e55d0",
        "sha256:646e1fbcd590bf502d0c99d252a51943929f361a586d31c4cb26bbb678cc095a",
        "sha256:676a8de49bd16e6d7db05b6ff900b7dd04fa768ef29479f7e1af64659fc5cbaa",
        "sha256:6a34ee07639397a6690cb2f34ed5136d0a0ed82005c42bfb745be231319f9479",
        "sha256:71b2d56c3fcadb3bf84969755e310176fc1a33984a2a46e17bbce1c6c34aa074",
        "sha256:71e2c3271f94065dec613df89ef0df12007fb85a7f85d8b9003be472b706d3d4",
        "sha256:7ab41260a9e5db098ebe041a989bfafd68334da0d7a94c4b45d02f4fe58da42f",
        "sha256:836a39d8d18b8887f0dbbb261e3ced2b6a49d5b4aeec33f8b7c1fcfc0fc9f575",
        "sha256:8a9b4bf78da6f61cb692fbe986298cbae091dee3602c7fa51ee5bc0e8c56a108",
        "sha256:8aefec1869c9175f8c87e9dfd6df200bb4a7712c33744ccf52ff86b326fd3799",
        "sha256:8f6b477aba0084a3ae575549fc8a0fd92a04f028c720718ddd87c3f9fd7750e9",
        "sha256:900c3452b862c997a22aba18598530b94c7cb65980794cb52355e9b39c7d8aa5",
        "sha256:93e3fe23d0a1def3a3142257b220416fbf57afc49e84959724e9ce4120d20d5c",
        "sha256:941f13905f65e416aef4f729e02edc3e476e80c5384950af001e0a3d3e1296e8",
        "sha256:96d629e5cc1db7b2c5b32f17a989dc238c55e78598b9846aa00986e196e96494",
        "sha256:9953f6f1f7f6a14612ae898c6c2613b0b49cdfaa2411184ad18522cdabd5c313",
        "sha256:a167192f186f893772e01df1e9d11c5637d43119b98456691875f43ed56a9331",
        "sha256:a81a02d6d54e3e3c7d995f44164911987e7ac40bf8938777098f553d7490fad7",
        "sha256:ab1f3b7c6ce3061c9926aa81a7f4892eb4d9a3c6efcedb5f4acd637dd3956d77",
        "sha256:abb866003c42aa3a61f525c52d63ae7787fcdfa1db09f0aba3861727ea554fb7",
        "sha256:afbbf34617e1ecd802aea33aadb57354037adaf514163839965b9d9c2c7e5c2a",
        "sha256:bf3bbfa37c683d74ddb30c4281c2c4abd42a9d394fc0f856d55fdb278373e008",
        "sha256:c34ca424881b987ff49184a6520ce044313c585f1975a7799d38abbf3b70a397",
        "sha256:c7f46107908db20e64d93677ee7abeaba1f0cfb486eeba65784c8b2845125be5",
        "sha256:cad240c3927e1e572c53ea30a55a86158fe5fa355331ba62867de6ad27b32ea5",
        "sha256:cbc718617a7b14a7c25f789b64c98453523c3c405e02587e4b8d7fbd83258fc0",
        "sha256:d0bcb0ab9c60d835590c1a413a15c3ebf8f300f1c305d671990a830cf5cc71a5",
        "sha256:dcdabda0d8d2cfe01292f1db1d7e69c8b6f9dd69ea13b11f980de26f07866832",
        "sha256:dceaaffd6dec467967b64de58cbeab8cf371fb39a0590626d6ab69bab491052d",
        "sha256:e7d0c886b203cff2e7a7ca819a3c40d5d71d3a5d31848a39085312bd6c5ed4a7",
        "sha256:e89e0c218314e09d844541beb637e7167bcb7cf5bed733fe685d7875aeeed782",
        "sha256:e974c8b4fbf2df16f4ec556c2bc0a17898c5734c93749009cc4acd3bcd0856aa",
        "sha256:eb516b91292ce1553d593350e43b670eb2e96d94a1675c36f4267712dfd2e1a0",
        "sha256:f2d66746cbbe81b93ca4243c6f7a6ba4979fa843b0057a854ce516cb0bd10edf",
        "sha256:f7f5f624e54d58a77ab605d2e798f97af1cb09b6a25424f23bbce085b0164fa4",
        "sha256:fbd9dbdfb7c41b3f6b5c082e0384cfe9b5875a46a8e1314d5a4aaa61b4597c16",
        "sha256:ff781f088789fe29c73d79ca9dc6df1bda4e6104208b57406dd2aa98e6ac88c6",
    }
)
"""The FROZEN realized-history cohort of the pre-W13 disposition mislabel (R24).

Exact ``record_revision`` content digests -- ``payload_hash`` over each persisted
row's canonical bytes, ``record_seq`` included -- of EVERY pilot-ledger row whose
status carries ``failed:accuracy-gate`` with a reason from
:data:`BLOCKED_DISPOSITION_REVIEW_REASON_CODES`. Semantic replay grants the W13
fresh-claim refusal's era passage ONLY to a resident row whose recomputed bytes
digest is a member of this set (the reducer recomputes the digest from the row's
bytes at decision time; nothing the row says about itself -- including its own
``record_revision`` field -- is consulted). Digests bind BYTES, producer-blind:
a stable id, a status shape, an era label, or a self-consistent recomputed hash
over NEW bytes names nothing here, so a current-era row written straight into
the ledger (honest next sequence, valid self-hash, current
``terminal_rule_identity``) can never vouch under era law.

Derivation (2026-08-27, read-only over the pilot campaign ledger
``records/models/current-shard.jsonl``, 583 rows / 258 heads): select every row
with ``status.code == "failed:accuracy-gate"`` and ``status.reason_code`` in the
closed disposition pair; recompute ``payload_hash`` over each row's persisted
bytes and cross-check it equals the row's recorded ``record_revision`` (52/52
matched); confirm ``accuracy_gate.verdict is None`` on all 52. Result: 52 rows
across 41 stable ids -- 31 current heads plus superseded continuity ancestors of
the same shape. The census sha256 over the sorted joined digests is
``57b755eb94c6e64affcd9b5150663decc77628204548cfc012eb00cfda34cece`` (pinned by
``test_era_frozen_cohort_is_exact``).

Widening rule (four-part waiver doctrine, "exact identity bindings"): a future
era boundary that legitimately mints more rows of a then-historical shape adds
their digests HERE in its own deliberated, reviewed change -- never by widening
a heuristic, never at runtime, never from producer input. Rows minted after W13
through the live head-continuity passage need no membership: their same-shape
parent satisfies the continuity arm on replay.
"""


def metadata_item_accepted(item: Mapping[str, Any]) -> bool:
    """Return whether one metadata gate item is an ACCEPTED adjudication.

    The ONE metadata acceptance predicate. Acceptance is the conjunction of all
    three component verdicts the checker scores independently: the top-level
    content ``verdict``, the ``integrity`` block that adjudicates the EVIDENCE
    itself (excerpt locators, source hashes, quote fidelity), and the
    ``rung_check`` that adjudicates the claimed ladder rung. Any one of them
    non-accurate is a rejection, and :func:`metadata_gate_failure_reason` names
    which.

    Spelled once here, beside the reason kernels, for the reason the reason
    kernels are: this predicate was restated at nine production sites and one of
    them disagreed. ``gates.route_metadata_gate`` -- the WRITE DOOR, whose
    ``canonical_write_allowed`` is the only one of the nine that authorizes
    publication -- omitted the integrity conjunct, so a content-accurate claim
    set over evidence the checker had disputed was published while the bounded
    repair census, the terminal-proof kernel, the reuse eligibility tripwire, and
    the canonical write-door acceptance all scored the same bytes a rejection.
    The disagreement was also an anti-substitution hazard: ``gate_reuse`` replays
    only non-accepted priors and a metadata replay APPENDS a marked row, so the
    split predicate let one item be both replayable and publishable, minting a
    second ledger row over accepted-item bytes that
    ``resolve_exact_gate_item_membership`` refuses.

    TOTAL and fail-closed like its reason-kernel siblings: an unreadable or
    absent component block is NOT an acceptance, and the complaint vocabulary
    (``checker_dispatch.metadata_complaint_fields``) discloses it under the same
    bare component name, so a repeat of the same dispute stays chargeable.

    Parameters
    ----------
    item:
        Canonical metadata gate item, persisted or candidate.

    Returns
    -------
    bool
        True only when content, integrity, and rung check are all accurate.
    """

    accurate = AccuracyVerdict.ACCURATE.value
    if item.get("verdict") != accurate:
        return False
    for component in ("integrity", "rung_check"):
        block = item.get(component)
        if not isinstance(block, Mapping) or block.get("verdict") != accurate:
            return False
    return True


def fidelity_item_accepted(item: Mapping[str, Any]) -> bool:
    """Return whether one fidelity gate item is an ACCEPTED adjudication.

    The fidelity sibling of :func:`metadata_item_accepted`: a content acceptance
    (``match`` or ``minor-drift``, the two five-way verdicts deliberately absent
    from :data:`FIDELITY_VERDICT_FAILURE_REASONS`) AND an accurate rung check.

    The fidelity lane has no integrity conjunct by contract -- the fidelity
    checker adjudicates executed behaviour against the recipe, and the evidence
    integrity of the underlying proposal was already adjudicated by the metadata
    lane whose acceptance is a precondition for dispatching fidelity at all. All
    fidelity sites already agreed on this predicate; it is centralized here so
    the pair cannot drift apart the way the metadata one did.

    Parameters
    ----------
    item:
        Canonical fidelity gate item, persisted or candidate.

    Returns
    -------
    bool
        True only for an accepted fidelity verdict over an accurate rung check.
    """

    fidelity = item.get("fidelity")
    if not isinstance(fidelity, Mapping) or fidelity.get("verdict") not in {
        "match",
        "minor-drift",
    }:
        return False
    rung_check = item.get("rung_check")
    return isinstance(rung_check, Mapping) and rung_check.get("verdict") == (
        AccuracyVerdict.ACCURATE.value
    )


FIDELITY_VERDICT_FAILURE_REASONS: dict[str, str] = {
    "major-drift": "major-drift-cap-exhausted",
    "slop": "slop-cap-exhausted",
    "cannot-verify": "cannot-verify-cap-exhausted",
}
"""Rejecting five-way fidelity verdicts mapped to their honest capped reasons.

``match`` and ``minor-drift`` are deliberately absent: they are content ACCEPTANCES, and
an item carrying one is rejected only by its rung check, whose own verdict then decides
the reason (see :func:`fidelity_gate_failure_reason`).
"""


def fidelity_gate_failure_reason(item: Mapping[str, Any]) -> str:
    """Map one REJECTED fidelity gate item onto its honest failure reason.

    The single crossing between the fidelity checker's verdict vocabulary and the
    record's, shared by ``gates.route_fidelity_gate``, the authority terminal-proof
    kernel, and the driver's fidelity terminalization candidate so no two sites can
    derive different reasons for the same gate item (the m1359 write-side incoherence:
    the router and the proof disagreed on a content-accepted item whose rung check came
    back ``cannot-verify``).

    A rejecting five-way verdict names its own capped reason. A content acceptance
    (``match``/``minor-drift``) was rejected by its rung check alone, and the rung
    verdict keeps the closed vocabularies' merits-vs-undetermined distinction:
    ``inaccurate`` is an identity dispute (never a slop/budget outcome), while
    ``cannot-verify`` could not be adjudicated either way. Like
    :func:`terminal_disposition_failure_reason` the mapping is TOTAL: anything
    unreadable degrades to the WEAKER could-not-determine claim rather than aborting.

    Parameters
    ----------
    item:
        Canonical fidelity gate item known not to be accepted.

    Returns
    -------
    str
        Reason code valid for the ``fidelity`` stage.
    """

    fidelity = item.get("fidelity")
    verdict = fidelity.get("verdict") if isinstance(fidelity, Mapping) else None
    reason = FIDELITY_VERDICT_FAILURE_REASONS.get(str(verdict))
    if reason is not None:
        return reason
    rung_check = item.get("rung_check")
    rung_verdict = rung_check.get("verdict") if isinstance(rung_check, Mapping) else None
    if verdict in {"match", "minor-drift"} and rung_verdict == "inaccurate":
        return "identity-mismatch"
    return "cannot-verify-cap-exhausted"


def metadata_gate_failure_reason(item: Mapping[str, Any]) -> str:
    """Map one REJECTED metadata gate item onto its honest failure reason.

    The metadata sibling of :func:`fidelity_gate_failure_reason`, shared by the
    authority terminal-proof kernel and the driver's accuracy-gate terminalization
    candidate. ``cannot-verify`` outranks the integrity check because an undetermined
    round asserts nothing about identity; a determined round whose integrity check is
    not accurate is an identity dispute. A content-accepted round (``accurate``) was
    rejected by its rung check alone, so the rung verdict decides -- the same arms the
    fidelity sibling has for ``match``/``minor-drift``: ``inaccurate`` is an identity
    dispute, anything else is a could-not-determine outcome, and neither is the merits
    cap ``inaccurate-cap-exhausted`` would claim. Everything else exhausted its repair
    cap on the merits. TOTAL like its siblings: an unreadable integrity block reads as
    the integrity dispute it is, an unreadable rung block as the undetermined round it
    is, never as an abort.

    Parameters
    ----------
    item:
        Canonical metadata gate item known not to be accepted.

    Returns
    -------
    str
        Reason code valid for the ``accuracy-gate`` stage.
    """

    if item.get("verdict") == "cannot-verify":
        return "cannot-verify-cap-exhausted"
    integrity = item.get("integrity")
    if not isinstance(integrity, Mapping) or integrity.get("verdict") != "accurate":
        return "identity-mismatch"
    if item.get("verdict") == "accurate":
        rung_check = item.get("rung_check")
        rung_verdict = rung_check.get("verdict") if isinstance(rung_check, Mapping) else None
        if rung_verdict == "inaccurate":
            return "identity-mismatch"
        return "cannot-verify-cap-exhausted"
    return "inaccurate-cap-exhausted"


GATE_MAX_REPAIRS: int = 2
"""Bounded chargeable repair allowance for one gate lineage.

The ONE authority behind every ``max_repairs=`` argument the driver passes to
the gate routers. A rejected round charges this allowance only under the
complaint-disclosure rule (see ``gates.route_metadata_gate``).
"""

GATE_CAP_CHARGEABLE_REJECTIONS: int = GATE_MAX_REPAIRS + 1
"""Chargeable rejected rounds that exhaust one bounded repair cap.

Derived, never restated: the initial adjudication plus the ``GATE_MAX_REPAIRS``
repair rounds it allows. The authority terminal-proof kernel proves a
``*-cap-exhausted`` claim against this threshold; the routers escalate to human
failure when a chargeable count EXCEEDS ``GATE_MAX_REPAIRS``. Both spellings
are the same bound -- keeping one derived from the other is what makes editing
the allowance a one-line change instead of a silent divergence between the
write side's 2 and the proof side's 3.
"""

AUTHOR_SESSIONS_MAX: int = GATE_MAX_REPAIRS + 1
"""Chargeable author sessions one work generation may buy without a grant.

The initial authoring plus the ``GATE_MAX_REPAIRS`` repair sessions the gate
lineage allows -- the ``author_sessions_max`` every budget block has always
advertised. Chargeability is DERIVED from append-only history (distinct staged
``author_result_id`` values per exact ``(stable_id, work_id)`` in the artifact
ledger, see ``menagerie.crawler.author_spend``), never from an in-memory
counter a rehydration resets: the pilot recorded 31 work generations over this
cap because the recorded field was a per-leg snapshot hardcoded to 0/1 (docket
C-AUTHOR-SPEND, T2-07). A consumed requeue grant EXTENDS the granted
generation's allowance by its ``attempts`` (docket T1-13, allowance half).
"""

AUTHOR_PAID_SESSIONS_PER_ATTEMPT: int = 2
"""Paid guard sessions one author attempt may buy: source-request plus author.

One author-lane attempt runs UP TO two provider agent sessions through the paid
guard (``_fetch_author_sources`` then ``_author_from_frozen_sources``), and every
guard session is its own paid-session reservation with its own single custody
bind and provider idempotency key (the 2026-08-20 step-200 leg bricked its spend
ledger by rebinding one reservation for the second session).  The ATTEMPT bound
is :data:`AUTHOR_SESSIONS_MAX` in ATTEMPT units, enforced durably by the
paid-ledger attempt-group projection (reservations of one attempt share a
replay-derived ``attempt_group`` identity; ``_admit_author_spend`` refuses a
NEW group past the allowance -- source-only failures included) plus the
artifact-ledger allowance valve (``menagerie.crawler.author_spend``).  This
factor scales only the paid-ledger QUOTA ceiling, which stays a true
raw-guard-session bound at the per-attempt factor.  The accounting fold
enforces the same per-attempt session bound structurally
(:data:`menagerie.crawler.paid_session_accounting.LANE_ATTEMPT_SESSION_BOUNDS`,
the money-truth vocabulary this constant must mirror -- pinned equal in
``tests/test_paid_session_retry_custody.py``): a reservation that would be a
third paid session of one author attempt group refuses at the fold, so joins
can no longer launder unbounded sessions into one disclosed attempt.
"""

EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT: int = AUTHOR_PAID_SESSIONS_PER_ATTEMPT + 2
"""Worst-case paid guard sessions ONE granted attempt can consume (MC-Q4).

The MC-Q4 ruling keeps ``attempts:N`` semantics untouched and DISCLOSES, at
issuance, the guard-session ceiling those semantics imply. One granted attempt
can buy up to :data:`AUTHOR_PAID_SESSIONS_PER_ATTEMPT` author-lane sessions
(source-request plus author) and, when its result reaches adjudication, up to
two per-model checker-lane gate sessions (``GateKind.FIDELITY`` and
``GateKind.TERMINAL_DISPOSITION``; ``METADATA_BATCH`` rides a shared batch and
is not attributable to one attempt) -- so an ``attempts:1`` grant is really a
ceiling of ~4 paid sessions, and the grant row plus the requeue tool both say
so. This constant is DISCLOSURE ONLY: no admission valve or budget projection
derives authority from it, and ``attempts`` stays the enforced unit.
"""

DEFERRED_TERMINAL_GRANT_STAGE: str = "deferred-terminal"
"""Typed requeue-grant stage for review-held null-stage deferral terminals.

A review-held ``deferred:*`` terminal records ``status.stage=null`` -- there is
no failure stage to re-run, only a platform/capability deferral to
re-adjudicate -- so no member of the closed ``FAILURE_REASON_CODES`` stage
vocabulary can ever equal it and the class was structurally un-grantable
(docket T2-05; m4256 was the first live instance). This sentinel is the ONE
additional spelling the grant vocabulary accepts, and it binds narrowly: a
``deferred-terminal`` grant consumes only against a head with
``status.kind == "deferred"`` and ``status.stage == null``, and no ordinary
failure stage ever binds such a head.
"""

HOST_CAPABILITY_ROUTING_CAPABILITY: str = "host-capability-routing"
"""Typed grant capability authorizing a host-capability re-run elsewhere."""

GRANT_CAPABILITIES: frozenset[str] = frozenset({HOST_CAPABILITY_ROUTING_CAPABILITY})
"""Closed vocabulary of typed capabilities a requeue grant may authorize.

A capability is the operator saying WHAT this grant is for, not merely that it exists.
WS-2 needs exactly one: re-opening a machine-scoped ``deferred:needs-more-memory`` head
requires a grant that NAMES the routing decision, because an ordinary requeue mints a new
work id and a new work id is not evidence that the host gained memory. The field is
inside the ``grant_id`` preimage (see ``grant_identity.tool_grant_identity``), so it
cannot be appended to an existing authorization after the fact.
"""


FAILURE_REASON_CODES: dict[str, frozenset[str]] = {
    "intake": frozenset(
        {
            "schema-invalid",
            "stable-id-conflict",
            "duplicate-revision-conflict",
            "migration-invariant",
        }
    ),
    "source": frozenset(
        {
            "identity-unresolved",
            "missing-mandatory-link",
            "source-model-mismatch",
            "source-target-invalid",
            "higher-rung-unresolved",
            # The author read real, fetched, hash-frozen sources and none of them
            # materially specifies this architecture. Distinct from every sibling
            # above: the identity resolved, a mandatory link exists, the sources
            # are the right model, and the ladder was walked. Recording it as any
            # of those would state a fact that is false. This is the canonical
            # BLOCKED source reason in the stage-2 author contract fixture.
            "missing-material-source",
            "effort-cap-exhausted",
            BLOCKED_PREREQUISITE_REASON_CODE,
            # AL-6 era rule: NEW blocked-prerequisite mints carry the
            # stage-suffixed member (here and in every blocked-arm stage
            # below); the bare member above stays read-valid for historical
            # rows only.
            BLOCKED_PREREQUISITE_STAGE_REASON_CODES["source"],
        }
    ),
    "fetch": frozenset(
        {
            "unreachable",
            "revision-missing",
            "hash-mismatch",
            "access-denied",
            "artifact-missing",
            "effort-cap-exhausted",
            BLOCKED_PREREQUISITE_REASON_CODE,
            BLOCKED_PREREQUISITE_STAGE_REASON_CODES["fetch"],
        }
    ),
    "author": frozenset(
        {
            "effort-exhausted:tool-calls",
            "effort-exhausted:fetch-targets",
            "effort-exhausted:wall-seconds",
            # The work generation staged its whole author-session allowance
            # (base AUTHOR_SESSIONS_MAX plus any granted extension), derived
            # from append-only artifact history. Distinct from the per-session
            # dimensions above: no single session overran anything; the
            # GENERATION's replay-derived spend is what ran out (T2-07/T1-13).
            "effort-exhausted:author-sessions",
            # The same session/attempt fuse tripped, but the MAJORITY of the
            # consumed sessions settled with a closed machine-classified
            # infrastructure outcome (`ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES`): a
            # sustained provider storm burned the allowance, not this model's
            # authoring. Replay-derived from settlement evidence only
            # (`author_spend.author_sessions_exhaustion_dimension`); uncertainty
            # fails toward the plain exhaustion reason above, never toward the
            # storm. Splitting the reason keeps triage tables from pooling
            # restorable storm damage with genuine effort exhaustion (the
            # step-200 storm burned ~40 models' allowances on infra noise
            # under this one label).
            "effort-exhausted:provider-storm",
            "wall-exceeded",
            # `session-crashed` means a session actually crashed. It is the
            # transport catch-all and nothing else may be folded into it: it is
            # a field the reducer counts, so every unrelated cause it absorbs
            # makes the campaign's own failure census wrong.
            "session-crashed",
            "research-tools-unavailable",
            "repair-exhausted",
            # The author escalated but there is no higher tier to escalate to,
            # so no promotion lineage can exist. A true author-stage blocker,
            # and NOT a crash.
            "needs-higher-tier",
            # The author published a well-formed result naming a reason the
            # closed record vocabulary cannot express. Previously this had no
            # member at all and collapsed onto `session-crashed`, reporting a
            # session that ran to completion as having died.
            "malformed-result",
            # The author session ran to completion and REFUSED the work on
            # content grounds instead of publishing. Previously this had no
            # member and collapsed onto `session-crashed` (a death that did
            # not occur) or `malformed-result` (bytes that were never
            # published), leaving the rung-200/500 content-refusal salvage
            # audit an undifferentiated cohort with the refusal prose only in
            # gitignored scratch. See `SESSION_FAILURE_CLASSES`.
            "content-refusal",
            BLOCKED_PREREQUISITE_REASON_CODE,
            BLOCKED_PREREQUISITE_STAGE_REASON_CODES["author"],
        }
    ),
    "evidence": frozenset(
        {
            "locator-missing",
            "excerpt-mismatch",
            "insufficient-detail",
            "coverage-incomplete",
            "search-incomplete",
            "effort-cap-exhausted",
            BLOCKED_PREREQUISITE_REASON_CODE,
            BLOCKED_PREREQUISITE_STAGE_REASON_CODES["evidence"],
        }
    ),
    "accuracy-gate": frozenset(
        {
            "inaccurate-cap-exhausted",
            "cannot-verify-cap-exhausted",
            "identity-mismatch",
            "checker-contract-invalid",
            "effort-cap-exhausted",
            # The checker lane's own generation-scoped session/wall budget ran out
            # before a verdict was delivered. A budget fact about THIS model's
            # adjudication, permanent for the generation, machine-classified from
            # the closed QUOTA exhaustion remap. See
            # `CHECKER_SESSION_EXHAUSTION_REASON_CODE`.
            CHECKER_SESSION_EXHAUSTION_REASON_CODE,
            # The checker adjudicated a SINGLE advisory terminal disposition -- a
            # SKIP/DEFER/BLOCKED recommendation -- and refused it. Every sibling above
            # describes the metadata_batch REPAIR loop and states something false here:
            # the two `*-cap-exhausted` codes assert a bounded repair cap when there is
            # one adjudication and nothing to repair, `identity-mismatch` asserts the
            # thing checked was not the thing authored, and `checker-contract-invalid`
            # blames the checker for an author claim the checker refused correctly.
            # See `TERMINAL_DISPOSITION_FAILURE_REASONS`.
            TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
            TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE,
        }
    ),
    "environment": frozenset(
        {
            "solve-failed",
            "lock-missing",
            "artifact-hash-mismatch",
            "build-failed",
            "probe-failed",
            "resolved-export-mismatch",
            "island-cap",
            "below-minimum-island-size",
            "effort-cap-exhausted",
            BLOCKED_PREREQUISITE_REASON_CODE,
            BLOCKED_PREREQUISITE_STAGE_REASON_CODES["environment"],
        }
    ),
    # AL-6 routing stage for the runtime-dependency blocked class: the status
    # code names the ecosystem-adjacency class, and the reason suffix -- any
    # blocked-arm stage -- preserves the stage the author actually named, so
    # the (status, reason) pair recovers the blocking stage on every row.
    ENVIRONMENT_ADJACENT_FAILURE_STAGE: frozenset(
        BLOCKED_PREREQUISITE_STAGE_REASON_CODES.values()
    ),
    "import": frozenset(
        {
            "module-missing",
            "symbol-missing",
            "abi-load-failed",
            "import-exception",
            "effort-cap-exhausted",
        }
    ),
    "constructor": frozenset(
        {
            "exception",
            "requires-checkpoint",
            "requires-weight-asset",
            "invalid-model-object",
            "effort-cap-exhausted",
        }
    ),
    "input": frozenset(
        {
            "contract-invalid",
            "source-invalid-shape",
            "generation-exception",
            "semantic-constraint",
            "effort-cap-exhausted",
        }
    ),
    "forward": frozenset(
        {
            "exception",
            "mode-run",
            "incomplete-receipt",
            "invalid-output-signature",
            "confirmation-mismatch",
            "effort-cap-exhausted",
        }
    ),
    "fidelity": frozenset(
        {
            "major-drift-cap-exhausted",
            "slop-cap-exhausted",
            "cannot-verify-cap-exhausted",
            "identity-mismatch",
            "effort-cap-exhausted",
        }
    ),
    "resource": frozenset(
        {
            "timeout",
            "oom",
            "disk-floor",
            "scratch-cap",
            "rss-cap",
            # DX-7/RE-7: a sibling mode slot deliberately not spawned because a
            # prior slot of the same work generation already observed a
            # deterministic resource kill (timeout or the RSS watermark) -- a
            # two-mode member buys exactly one timeout. Derived evidence, never
            # a decisive verdict on its own.
            "blocked-by-sibling",
            "effort-cap-exhausted",
        }
    ),
    "policy": frozenset(
        {
            "network-attempt",
            "checkpoint-read",
            "write-outside-scratch",
            "credentials-exposed",
            "torchlens-import",
            "opaque-code",
            "sandbox-unavailable-v1",
            "effort-cap-exhausted",
            # BL-12(e): host telemetry loss is a HOST fact with its own
            # reason. The lane added it to attempt-common.schema.json but
            # missed this python-side set, so the reducer refused every
            # worker attempt carrying it ("attempt error reason is not
            # allowed for its stage") and the env-turn quarantined the model
            # as a machinery fault every leg (m12646/m13258/m13279 loop).
            "telemetry-integrity",
            BLOCKED_PREREQUISITE_REASON_CODE,
            BLOCKED_PREREQUISITE_STAGE_REASON_CODES["policy"],
        }
    ),
    "runner": frozenset(
        {
            "native-crash",
            "signal",
            "missing-receipt",
            "protocol-violation",
            "ledger-corruption",
            "internal-error",
            # BL-12(f): worker result present but its v3 attestation
            # association is unclean -- machinery-owned, schema-listed,
            # missed here (same incident as telemetry-integrity above).
            "attestation-mismatch",
            # BL-12(b): a D-state worker root is a kernel/host fact
            # (driver_receipts WorkerUnreapableError arm). Schema-listed and
            # owner-mapped from the start, missed HERE -- the same one-sided
            # registration as the two rows above, so the first live D-state
            # worker would have quarantine-looped its model on every leg.
            # Pinned equal by tests/test_reason_vocabulary_cross_pin.py (DX-15).
            "worker-unreapable",
            # DX-2/RE-2: the exec/OS layer refused to start the worker at all
            # (exit 126 with no constructor evidence) -- a HOST bootstrap fact
            # about the routed interpreter, never a model verdict.
            "worker-bootstrap-failed",
            # DX-2/RE-2: the worker's own crawler-frame runtime failed to
            # import before any model code was reachable (the StrEnum/py3.10
            # class) -- a HOST runtime fact, never a model verdict.
            "worker-runtime-unavailable",
            "unclassified-supervised-failure",
            "effort-cap-exhausted",
            BLOCKED_PREREQUISITE_REASON_CODE,
            BLOCKED_PREREQUISITE_STAGE_REASON_CODES["runner"],
        }
    ),
}

#: Closed vocabulary for one failed author session's machine-derived cause label.
#:
#: ``author_executor.session_failure_class`` computes the label (structured harness
#: fields first, quarantined diagnostic text last), the executor's structured
#: ``author-session-failure`` notice carries it across the process boundary, and the
#: driver persists it on the canonical attempt row (``error.failure_class``). The set is
#: closed HERE and in the attempt schema's enum so the rung-200/500 salvage audit reads a
#: typed census, but the crossing stays TOTAL like ``_blocked_terminal``: an unrecognized
#: label folds to ``unclassified`` (with the claimed label preserved on the basis) rather
#: than making one odd string unrecordable.
#:
#: ``content-refusal`` is the T4-T4 member: a session that ran to completion and declined
#: the work in prose. It is a label derived from externally controlled text; like every
#: diagnostic-text label it carries NO pause or backoff authority.
SESSION_FAILURE_CLASSES = frozenset(
    {
        "auth-unavailable",
        "provider-overloaded",
        "content-refusal",
        "unclassified",
    }
)

#: Bound on the durable refusal excerpt persisted on a canonical attempt row.
#:
#: The excerpt is the deliberate, narrow exception to diagnostics redaction: refusal
#: prose that lives only in gitignored scratch does not survive host teardown, so the
#: salvage audit needs a canonical quote -- bounded, never the full transcript. The
#: attempt schema enforces the same bound as ``maxLength``, so an over-long excerpt is
#: unrecordable rather than silently truncated by a reader.
REFUSAL_EXCERPT_MAX_CHARS = 500

#: Reason codes that name an effort or budget outcome rather than a source verdict.
#:
#: ``PLAN.md`` LP-13.2: cap exhaustion is ``failed:<actual-stage>`` with a stage-valid
#: effort reason, not a skip or a convenience defer, and only
#: ``tools/requeue --reason ... --grant ...`` reissues the work with a larger grant. An
#: exhausted model is *unfinished*, not *unresolvable*.
#:
#: The ``effort-exhausted:`` family is derived rather than listed so a dimension added to
#: :data:`FAILURE_REASON_CODES` later is covered without touching this set. The explicit
#: rows are the remaining exhaustion spellings the vocabulary defines, plus
#: ``budget-exhausted`` -- not a vocabulary member at all, but a free-form spelling a live
#: author session actually produced, listed so the laundering refusal names the real
#: diagnosis instead of emitting a bare unknown-code message.
#:
#: This set is deliberately the ONLY closed thing about a BLOCKED reason. The reason
#: vocabulary at large is NOT closed with a hard refusal, because ``_blocked_terminal``
#: already closes it TOTALLY and gracefully: an unrecordable reason becomes
#: ``failed:author`` with ``malformed-result``. Refusing every unrecognized reason at the
#: parse boundary would reintroduce exactly the failure mode that mapping exists to
#: prevent -- one odd reason string on one model taking the whole campaign down. An
#: exhaustion claim is different in kind: it is not an unrecognized string, it is a FALSE
#: statement about why the model stopped, and a false statement must be unrepresentable.
EFFORT_EXHAUSTION_REASON_CODES: frozenset[str] = frozenset(
    {
        "effort-cap-exhausted",
        "repair-exhausted",
        "wall-exceeded",
        "budget-exhausted",
    }
    | {
        code
        for codes in FAILURE_REASON_CODES.values()
        for code in codes
        if code.startswith("effort-exhausted:")
    }
)

#: Stage-valid exhaustion reason used when routing a refused exhaustion claim.
#:
#: The claim is refused as an advisory arm, but the session really did exhaust, so it lands
#: on ``failed:<stage>`` and needs a reason that stage's vocabulary actually admits.
#: ``author`` carries the ``effort-exhausted:`` family instead of ``effort-cap-exhausted``,
#: so a single hardcoded reason would emit an invalid ``(stage, reason_code)`` pair.
EXHAUSTION_TERMINAL_REASON_BY_STAGE: dict[str, str] = {
    stage: next(
        code
        for code in ("effort-cap-exhausted", "effort-exhausted:wall-seconds", "wall-exceeded")
        if code in codes
    )
    for stage, codes in sorted(FAILURE_REASON_CODES.items())
    if codes & EFFORT_EXHAUSTION_REASON_CODES
}

WORKFLOW_STATES = frozenset(
    {
        "UNTRIAGED",
        "queued",
        "authoring",
        "awaiting-gate",
        "fidelity-pending",
        "environment-pending",
        "forward-observed-but-blocked",
        "paused:usage-limit",
    }
)

#: Closed leg-scheduling shapes. ``waves`` is the historical intake-wide phase
#: barrier; ``pipelined`` is the per-model execute-first scheduler. One authority
#: for the CLI choices, the environment default, ``DriverConfig`` validation, and
#: the cutover/rollback disclosure -- a mode name that is not in this tuple must
#: never reach the driver, because the dispatch test is ``== "pipelined"`` and a
#: typo would otherwise silently run the other scheduler.
SCHEDULER_MODES: tuple[str, ...] = ("waves", "pipelined")

#: Leg-scheduling shape used when nothing names one.
DEFAULT_SCHEDULER = "waves"


INPUT_KINDS = frozenset(
    {
        "standard-image",
        "standard-text",
        "standard-audio",
        "standard-video",
        "standard-tabular",
        "standard-pointcloud",
        "standard-integer-range",
        "random-fallback",
    }
)


class OperationalEventKind(StrEnum):
    """Closed operational-event kinds."""

    USAGE_PAUSE = "usage-pause"
    USAGE_PAUSE_OBSERVATION = "usage-pause-observation.v1"
    USAGE_RESUME = "usage-resume"
    WAKEUP = "wakeup"
    CHECKPOINT = "checkpoint"
    CAMPAIGN_HEALTH = "campaign-health"
    WAKE_NOOP_ALREADY_RUNNING = "wake-noop-already-running"
    CHECKPOINT_REVIEW = "checkpoint-review"
    REVIEW_SIGNOFF = "review-signoff"
    PROGRESS_NOTIFICATION = "progress-notification"
    NOTIFICATION_DELIVERY = "notification-delivery"
    REQUEUE_GRANT_CONSUMED = "requeue-grant-consumed"
    WORKER_LEASE_OPENED = "worker-lease-opened"
    WORKER_LEASE_STARTED = "worker-lease-started"
    WORKER_LEASE_CLOSED = "worker-lease-closed"
    WORKER_LEASE_REAPED = "worker-lease-reaped"
    ORPHAN_WORKER_RECOVERED = "orphan-worker-recovered"
    WAKEUP_INSTALLED = "wakeup-installed"
    WAKEUP_REPAIRED = "wakeup-repaired"
    WAKEUP_FIRED = "wakeup-fired"
    WAKEUP_DEACTIVATED = "wakeup-deactivated"
    WAKEUP_HEALTH_DEGRADED = "wakeup-health-degraded"
    WAKEUP_FAILED = "wakeup-failed"
    CAMPAIGN_COMPLETED = "campaign-completed"
    OPERATOR_CANCELLED = "operator-cancelled"
    WORKER_SHUTDOWN_INTERRUPTED = "worker-shutdown-interrupted"
    TERMINAL_UNRECORDABLE = "terminal-unrecordable"
    MODEL_LANE_FAILED = "model-lane-failed"
    SCHEDULER_CUTOVER = "scheduler-cutover"
    SCHEDULER_ROLLBACK = "scheduler-rollback"
    #: WS-1 clause 7: the leg-start selection census -- one event per leg
    #: naming the mode and the exact never-recorded / stale-excluded /
    #: parked / deferred / granted counts.
    WORK_SELECTION_CENSUS = "work-selection-census"
    #: WS-1 clause 4: the leg-start projection-staleness operator REPORT
    #: ({stale_count, reasons_histogram, sample_ids}) -- the single cheapest
    #: diagnostic nobody ran in three days.
    PROJECTION_STALENESS = "projection-staleness"
    #: WS-1 clause 6: a durable work generation was explicitly discharged
    #: (operator void, leg-exit abandon disclosure, or migration backfill).
    REQUEUE_GENERATION_DISCHARGED = "requeue-generation-discharged"
    #: B-09: the leg-start effective run configuration -- scheduler mode, lane
    #: widths, author/checker models, the selection scope, and the named lanes.
    #: Appended once per leg BEFORE any admission decision, so a leg that later
    #: refuses (unacknowledged holds, review pause) still attests what it ran as.
    #: At the audit base the config a leg ran under was recoverable only from an
    #: operator's shell history.
    LEG_CONFIG = "leg-config"
    #: D-20 (LC-7): the leg-start read-only close preview -- one event per leg
    #: naming whether the close WOULD refuse today and which members/axes it
    #: would refuse on, derived through the exact production boundary
    #: predicate (the DX-24 probe path). Report-only: it triggers no recovery
    #: writes and never replaces the final replay; 8-11h of 200a was lost to
    #: refusals first read at leg END.
    CLOSE_PREVIEW = "close-preview"
    #: W6/OB-M12: the leg-outcome mirror of the durable ``leg_ended`` row --
    #: exactly one per leg that attested a leg-config, carrying the closed
    #: exit class, the PM-5 cause fingerprint, and the terminal delta, so
    #: ``count(leg-config) == count(leg-outcome) + running`` is a one-ledger
    #: census (orphans closed ``orphaned_uncontained`` by the boot
    #: reconciler).
    LEG_OUTCOME = "leg-outcome"
    #: Overhaul RC-1 (RULING D.1 behavior 3): an in-process invocation
    #: preflight arm refused the ``crawl`` invocation before any spend. Typed,
    #: per-occurrence, and counted toward NO fuse -- a refusal buys nothing.
    INVOCATION_REFUSED = "invocation-refused"
    #: Overhaul RC-1 (behavior 5): the paced-start census emitted once the
    #: first 20 terminals under a new configuration identity exist
    #: (publication rate, accept rate, storm-fuse state, mean session wall).
    CONFIGURATION_CANARY = "configuration-canary"
    #: Overhaul RC-1 (behavior 2): the crawl limit was set or raised. The
    #: persisted order is never re-derived; widening is exactly this event.
    CRAWL_LIMIT = "crawl-limit"


class OperationalEventStatus(StrEnum):
    """Closed operational-event dispositions."""

    USAGE_PAUSED = "paused:usage-limit"
    USAGE_RESUMED = "resumed:usage-limit"
    WAKEUP_SCHEDULED = "wakeup-scheduled"
    WAKEUP_FIRED = "wakeup-fired"
    WAKE_NOOP_ALREADY_RUNNING = "wake-noop-already-running"
    CHECKPOINT_COMPLETE = "checkpoint-complete"
    CHECKPOINT_FAILED = "checkpoint-failed"
    HEALTHY = "healthy"
    #: Operator-attention disposition for award-custody events (AW-1): an award
    #: supersession -- parked, granted, or reminted -- is never ``healthy``
    #: (the three destroyed awards' events all read ``healthy``, 1 of 315
    #: identical lines).
    DEGRADED = "degraded"
    RUNNER_FAILED = "failed:runner"
    RETRYABLE_INFRASTRUCTURE = "retryable:infrastructure"
    CHECKPOINT_REVIEW_PAUSED = "paused:checkpoint-review"
    REVIEW_SIGNED_OFF = "resumed:checkpoint-review"
    PROGRESS_NOTIFIED = "progress-notified"
    PROGRESS_RECORDED = "progress-recorded"
    NOTIFICATION_DELIVERED = "notification-delivered"
    NOTIFICATION_FAILED = "notification-failed"
    REQUEUE_GRANT_CONSUMED = "requeue-grant-consumed"
    WORKER_LEASE_OPEN = "worker-lease-open"
    WORKER_LEASE_ACTIVE = "worker-lease-active"
    WORKER_LEASE_CLOSED = "worker-lease-closed"
    WORKER_LEASE_REAPED = "worker-lease-reaped"
    ORPHAN_WORKER_RECOVERED = "orphan-worker-recovered"
    WAKEUP_INSTALLED = "wakeup-installed"
    WAKEUP_REPAIRED = "wakeup-repaired"
    WAKEUP_DEACTIVATED = "wakeup-deactivated"
    WAKEUP_HEALTH_DEGRADED = "wakeup-health-degraded"
    WAKEUP_FAILED = "wakeup-failed"
    CAMPAIGN_COMPLETED = "campaign-completed"
    OPERATOR_CANCELLED = "operator-cancelled"
    TERMINATED = "paused:terminated"
    SHUTDOWN_INTERRUPTED = DRIVER_SHUTDOWN_STATUS
    MODEL_LANE_FAILED = "failed:model-lane"
    SCHEDULER_MODE_CHANGED = "scheduler-mode-changed"
    INVOCATION_REFUSED = "refused:invocation"
    """A ``crawl`` invocation refused at its in-process preflight (RC-1); no spend, no fuse."""
    HOST_CAPABILITY_LEG_REFUSED = "refused:host-capability-leg"
    """A leg refused to launch because NOTHING in its target set can run on this host.

    Distinct from ``healthy`` on purpose: the partial case (some members deferred) is a
    disclosure and stays healthy, while a fully host-deferred target set is a refusal --
    the leg does no work at all. Recording it as healthy is how eight step-100 legs
    looked like activity while chasing an unsatisfiable halt condition.
    """


# Slice-F scheduler configuration defaults.  The earlier names remain the
# compatibility surface used by Slice E's event builders.
DEFAULT_REVIEW_CHECKPOINT_AT = DEFAULT_REVIEW_CHECKPOINT_MODELS
DEFAULT_PROGRESS_MILESTONES = DEFAULT_PROGRESS_NOTIFICATION_MILESTONES
DEFAULT_NOTIFY_COMMAND: str | None = None
