"""Supervised headless author executor: the author lane with no managing session.

Invoked by the existing :class:`CommandAuthorLane` subprocess contract (argv +
one absolute request path; result at the envelope's required path; typed exit
codes ``0/64/75/76/78``), this program owns one model's complete author
lifecycle:

- **stage 1** runs as ``claude -p`` with the verified pinned research
  configuration (every flag is load-bearing; a session that cannot reach its
  tools must fail loudly, never degrade quietly);
- the machine-owned **source broker** turns the typed discovery result into
  manifest rows whose exact strings (commit SHAs, digests, citation metadata)
  are derived from receipts, never authored;
- **stage 2** runs as ``claude -p --resume <stage-1 session>`` for context
  retention without a managing session, with at most one typed supplementary
  broker round.

Every artifact of an attempt lives under its own attempt directory
(:mod:`menagerie.crawler.author_attempts`); the executor is the only publisher
to the envelope's required path, publication is bound to the attempt nonce the
executor opened, and the published digest is verified and printed as a receipt
the lane re-verifies. A killed executor leaves a durable attempt record; the
retry resumes the recorded provider session when the record proves stage 1
completed, and otherwise opens attempt N+1 with the prior attempt in the
feedback channel.

Pause classification authority is structured signals only: the ``claude -p``
JSON error fields decide quota exits, and this program's own diagnostics use
closed reason codes, never verbatim provider text, so a GitHub rate-limit
message in a dying author's output can never masquerade as an Anthropic quota
event.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import quote
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from menagerie.crawler.author_attempts import (
    AttemptHandle,
    latest_attempt,
    list_attempts,
    new_attempt,
    prior_attempts_summary,
)
from menagerie.crawler.author_dispatch import (
    AuthorDispatchError,
    AuthorEffortExhaustionClaim,
    AuthorEngineFaultError,
    # The lane's own claim grammar and production result parser, reused rather
    # than re-derived: the pre-publication gate must refuse and enumerate with
    # exactly the authority the driver will apply after publication, or its
    # feedback teaches the session a different validator than the one that
    # decides.
    _names_effort_exhaustion,
    _validate_author_result_mapping,
    author_identity_binding,
    checker_identity_binding,
    derive_terminal_evidence_pack,
    derive_terminal_license_disposition,
)
from menagerie.crawler.capability_probe import (
    CAPABILITY_PROBE_FORMAT,
    CapabilityProbeError,
    derive_challenge,
    validate_capability_evidence,
)
from menagerie.crawler.constants import (
    AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET,
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_RESULT_SCHEMA_VERSION,
    AUTHOR_WALL_EXTERNAL_KILL_FACTOR,
    AUTHOR_WALL_SECONDS_ENV,
    CAMPAIGN_TEARDOWN_RECORD_ENV,
    CAMPAIGN_TEARDOWN_RECORD_VERSION,
    CLAUDE_CLI_VERSION,
    REFUSAL_EXCERPT_MAX_CHARS,
    SOURCE_DISCOVERY_SCHEMA_VERSION,
    resolve_author_wall_seconds,
)
from menagerie.crawler.discovery import (
    DiscoveryError,
    FoundDiscovery,
    HigherTierDiscovery,
    NegativeDiscovery,
    RetryableToolFailureDiscovery,
    validate_source_discovery,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    hash_bytes,
    stable_hash,
    utc_now,
)
from menagerie.crawler.metadata import (
    MetadataValidationError,
    recompute_accepted_identities,
)
from menagerie.crawler.models import JsonObject, bounded_json_repr
from menagerie.crawler.paid_session_custody import (
    CUSTODY_FD_ENV,
    anonymous_credential,
    inherited_custody_fds,
)
from menagerie.crawler.schema import (
    MODEL_SCHEMA_VERSION_V3,
    PayloadValidationError,
    get_validator,
    validate_payload,
)
from menagerie.crawler.source_broker import (
    BROKER_DETERMINISTIC_OUTCOMES,
    OUTCOME_FETCHED,
    OUTCOME_PAPER_DERIVATION_ONLY,
    OUTCOME_PROBED,
    BrokerPack,
    SourceBrokerError,
    broker_source_pack,
    default_transport,
    write_broker_outputs,
)
from menagerie.crawler.terminal_evidence import MACHINE_OWNED_EXCERPT_FIELDS
from menagerie.crawler.worker_supervisor import (
    # The hardened teardown: it proves the group is still ours (an unreaped child
    # leading its own group plus an unchanged process-start token) before it
    # signals anything. Reused rather than re-derived; there must be exactly one
    # such routine, and a raw ``os.killpg(process.pid, ...)`` is not it.
    _kill_process_group as kill_process_group,
    ProcessGroupHandle,
    capture_process_group,
    signal_verified_process_group,
)

EXECUTOR_VERSION = "menagerie-author-executor 1.0.0"
RECEIPT_VERSION = "menagerie.crawler.author-executor-receipt.v1"
SUPPLEMENT_VERSION = "menagerie.crawler.author-supplement-request.v1"

# Operator protocol exit codes (identical to the lane's expectations).
EXIT_OK = 0
EXIT_PERMANENT = 64
EXIT_RETRYABLE = 75
EXIT_BACKOFF = 76
EXIT_UNAVAILABLE = 78

#: Parent-owned side channel written before a campaign teardown signal.  The
#: session output is never consulted to decide whether the campaign killed it.
_CAMPAIGN_TEARDOWN_SIGNAL = signal.SIGTERM
_CAMPAIGN_TEARDOWN_REQUESTED = threading.Event()
_SESSION_COMMUNICATE_POLL_SECONDS = 0.1

#: The verified pinned web-tool recipe. Every flag is load-bearing: without
#: ``--mcp-config`` Exa is absent, without ``--allowedTools`` every tool is
#: permission-blocked, and a name mismatch reads as unavailability -- all three
#: complete "successfully" while researching nothing.
EXA_MCP_URL = "https://mcp.exa.ai/mcp"
EXA_API_KEY_ENV = "EXA_API_KEY"
RESEARCH_TOOLS = (
    "WebSearch",
    "WebFetch",
    "mcp__exa__web_search_exa",
    "mcp__exa__web_fetch_exa",
    "ToolSearch",
)


def exa_mcp_config(api_key: Optional[str] = None) -> str:
    """Return the Exa MCP config, authenticated when a key is available.

    The anonymous endpoint is rate limited at roughly 1,400 searches per month --
    about 230 models -- so it was never viable for a 28,482-model campaign. It
    exhausted mid-run on 2026-07-28 and cost six of six first attempts to
    ``research-tools-unavailable`` before the cause was identified.

    An absent key falls back to the anonymous endpoint rather than raising: a
    machine without the key still runs, degraded exactly as before, and the
    ``research-tools-unavailable`` guard still fails loudly if the tools cannot be
    reached. A missing credential must not be the reason a campaign cannot start.

    The key is read from the environment at spawn time and written only to the
    mode-0600 ephemeral MCP file the child reads. The argv carries the file path,
    never the secret-bearing JSON, and the file is removed when the child exits.
    It is never written into campaign config, prompts, receipts, manifests or any
    durable artifact -- this repository is public and the crawler serializes a
    great deal.
    """

    key = api_key if api_key is not None else os.environ.get(EXA_API_KEY_ENV, "")
    key = key.strip()
    url = f"{EXA_MCP_URL}?exaApiKey={quote(key, safe='')}" if key else EXA_MCP_URL
    return json.dumps({"mcpServers": {"exa": {"type": "http", "url": url}}})


#: The identity calculator's module path, as a repository-relative POSIX path.
#: The author invokes it as ``<interpreter> <abs script path> --request ... --facts
#: ...``; the grant pins both halves, so the specifier names exactly one program.
IDENTITY_TOOL_MODULE = "menagerie/crawler/tools/author_identities.py"


def identity_tool_script() -> Path:
    """Return the calculator's absolute script path.

    Returns
    -------
    Path
        Absolute path to the shipped calculator, resolved from this module's own
        location so a worktree or an installed copy each name their own file.
    """

    return Path(__file__).resolve().parent / "tools" / "author_identities.py"


def identity_tool_command() -> str:
    """Return the exact command prefix the identity-calculator grant names.

    The interpreter is this process's own, so the author is handed the same
    environment the executor already runs the crawler in and cannot be pointed at
    a Python that lacks the dependencies.

    The script is named BY PATH rather than as ``-m <module>``, and that is not
    cosmetic. ``menagerie`` is not an installed distribution -- it resolves through
    the current working directory -- and the author session's working directory is
    its own attempt directory. ``-m`` therefore died with ``No module named
    'menagerie'`` under a live probe, AFTER the harness had granted the command.
    A path invocation plus the script's own import bootstrap works from any
    directory.

    Returns
    -------
    str
        ``"<interpreter> <absolute calculator path>"``.
    """

    return f"{sys.executable} {identity_tool_script()}"


def identity_tool_rule() -> str:
    """Return the single ``Bash`` permission specifier granting the calculator.

    Returns
    -------
    str
        A prefix specifier of the documented ``Bash(<prefix>:*)`` form.
    """

    return f"Bash({identity_tool_command()}:*)"


def stage_tool_rules(
    *,
    write_root: Path,
    read_roots: Sequence[Path],
) -> tuple[str, ...]:
    """Return the path-scoped ``--allowedTools`` rules for one session stage.

    **The per-attempt directory is the sole writable path.** Confinement is
    synchronous, not post-hoc: the only file surface is the harness
    ``Read``/``Write``/``Edit`` tools, and those are granted as **path-scoped
    permission specifiers** rather than bare tool
    names. In ``-p`` mode an unmatched permission auto-denies, so an
    absolute-path write outside the attempt directory — ``..`` traversal, a
    sibling live attempt's directory, the authority root — is denied by the
    harness before bytes change. The executor remains the only publisher to any
    lane-visible path: staged model code, source packs, results, and probe
    evidence all leave the attempt directory only through executor-owned
    publication.

    **The one command grant, and why it does not widen that.** Exactly one
    ``Bash`` rule is emitted: a prefix specifier naming the identity calculator
    module and nothing else. It is granted because the author lane's other
    option was to keep asking a language model to reproduce a six-deep nested
    SHA-256 over canonical JSON by hand, which failed on three fields in ten
    out of ten published proposals. The confinement bar is unchanged, on three
    counts: the rule names a fixed module rather than an interpreter or a bare
    ``Bash`` capability, so no other program is reachable through it; the tool
    only reads two files and writes its answer to stdout, so it adds no
    writable path; and per the documented permission model each sub-command of
    a compound invocation must match a rule on its own, so a chained escape has
    nothing to chain to. :func:`identity_tool_command` single-sources the
    invocation for the grant, the rendered brief, and the escape suite, which
    pins the Bash allowlist to that ONE rule rather than to "no Bash" — an exact
    allowlist is the stronger assertion, since it fails on an unexpected grant
    as well as on a missing one.

    The escape acceptance suite (Sol's six cases, adopted verbatim) lives in
    ``tests/test_author_executor_escapes.py``; if the harness rules do not
    deny every case before bytes change, the mechanism behind this ONE
    injection point moves down the ladder (write-only Seatbelt profile, then
    executor-owned file RPCs) without touching call sites.

    Two rule-form facts are load-bearing, both documented (Claude Code
    "Configure permissions", verified empirically against claude 2.1.220 in
    the escape suite's live layer):

    1. **Absolute paths take a double slash.** ``Tool(//opt/x/**)`` anchors
       at the filesystem root; ``Tool(/opt/x/**)`` anchors at the *settings
       source* (the session cwd for CLI-passed rules), so a single-slash
       "absolute" rule matches nothing and every file call auto-denies.
    2. **File-permission checks match only ``Read(path)`` and ``Edit(path)``
       rules.** An ``Edit`` rule covers all file-editing tools (Edit, Write,
       NotebookEdit); a ``Write(path)`` rule is accepted but never matched,
       so it grants nothing and draws a startup warning.

    Parameters
    ----------
    write_root:
        The attempt directory — the only tree the session may write.
    read_roots:
        Directories the session may read (the model's author root; for stage
        2 also the frozen prompt directory the envelope names). Read scoping
        is a deferred residual by design; write scoping is the bar.
    """

    def scoped(tool: str, root: Path) -> str:
        path = Path(root)
        if not path.is_absolute():
            raise ValueError(f"confinement rule roots must be absolute, got {path!r}")
        # str(path) begins with "/"; the extra "/" yields the documented
        # ``//`` filesystem-root anchor.
        return f"{tool}(/{path}/**)"

    rules: list[str] = [*RESEARCH_TOOLS, identity_tool_rule()]
    for root in dict.fromkeys(Path(r) for r in (*read_roots, write_root)):
        rules.append(scoped("Read", root))
    rules.append(scoped("Edit", write_root))
    return tuple(rules)


#: The external kill fires at grant +10%; the brief carries the deadline so a
#: session watching its clock can land a typed ``BLOCKED`` instead of a SIGKILL.
#: Both this factor and the per-campaign grants are single-sourced in
#: :mod:`menagerie.crawler.constants` so the driver's lane and this executor can
#: never disagree about one campaign's budget.
EXTERNAL_KILL_FACTOR = AUTHOR_WALL_EXTERNAL_KILL_FACTOR
_KILL_GRACE_SECONDS = 10.0

#: Structured harness signals that mean a provider usage pause. Free-text
#: marker scanning is deliberately absent: classification authority is the
#: parsed harness JSON only (SEAM_REDESIGN section 3.5). The one place a FAILED
#: session's own error text is read at all is :func:`session_failure_class`,
#: which only LABELS an already-retryable failure for the durable record and
#: the driver's retry-ladder detail. Weather backoff authority belongs to an
#: executor-authored provider-overload record with the machine-observed harness
#: shape; quarantined text labels remain diagnostic and never enter the pause
#: extractor's fields.
#:
#: Every value below is transcribed from the shipped harness rather than assumed.
#: The prior constants (``usage_limit``, ``usage_limit_reached``,
#: ``error_usage_limit``, ``error_rate_limit``, ``rate_limit`` as *subtypes*) were
#: written from assumption and appear nowhere in the harness: not in the Claude
#: Code 2.1.220 binary, and not in the ``@anthropic-ai/claude-agent-sdk`` 0.3.211
#: ``subtype`` union. Matching on them could never fire, so the campaign pause was
#: unreachable and a real limit would have failed every model generically.
#:
#: Sources for what replaced them:
#:   * ``TerminalReason`` union -- claude-agent-sdk 0.3.211 ``sdk.d.ts`` line 6759.
#:     ``SDKResultSuccess`` and ``SDKResultError`` both carry the optional
#:     ``terminal_reason`` field (``sdk.d.ts`` lines 4171-4218), and a real captured
#:     ``--output-format json`` result does emit it (``"terminal_reason":
#:     "completed"`` on a clean run), so it is a live wire field.
#:   * ``SDKRateLimitInfo`` -- ``sdk.d.ts`` lines 4149-4168: ``status`` is
#:     ``allowed | allowed_warning | rejected``, with ``resetsAt`` (unix seconds)
#:     and ``rateLimitType`` distinguishing ``five_hour`` from the ``seven_day``
#:     weekly family.
#:   * ``api_error_status`` -- ``SDKResultSuccess`` (``sdk.d.ts``), observed as
#:     ``"api_error_status": null`` in a real captured result. HTTP 429 is the
#:     provider's rate-limit status.
#:   * ``rate_limit_error`` / ``overloaded_error`` -- the raw Anthropic API error
#:     envelope; both literals ship in the 2.1.220 binary.
#:
#: The subtype union the harness really emits is ``success |
#: error_during_execution | error_max_turns | error_max_budget_usd |
#: error_max_structured_output_retries``. None of those names a usage limit on its
#: own -- ``error_during_execution`` is the *generic* failure subtype, so pausing
#: the whole campaign on it would turn every ordinary session crash into an outage.
#: Subtype is therefore deliberately NOT a pause authority; ``terminal_reason`` is.
_LIMIT_TERMINAL_REASONS = frozenset({"blocking_limit", "rapid_refill_breaker"})
_LIMIT_RATE_STATUS = "rejected"
_RATE_LIMITED_HTTP_STATUS = 429
_LIMIT_ERROR_TYPES = frozenset({"rate_limit_error", "overloaded_error"})

_PROMPT_ROOT = Path(__file__).with_name("prompts")
_EXECUTOR_PROMPTS = _PROMPT_ROOT / "executor"
_POOL_PROMPTS = _PROMPT_ROOT / "pool"
_SCHEMA_ROOT = Path(__file__).with_name("schemas")

_SOURCE_REQUEST_VERSION = "menagerie.crawler.author-source-request.v1"
_AUTHOR_ENVELOPE_PREFIX = "menagerie.crawler.author-envelope"


class AuthorExecutorError(RuntimeError):
    """Raised when a request cannot be served at all (typed unavailability).

    ``errors`` carries the COMPLETE enumerated defect list when the refusal
    found several violations at once. Rung 8 measured what first-error-only
    reporting costs: every two-attempt malformed model burned a full session to
    surface exactly one error with two to four latents behind it (W-8). The
    single message stays the exception text; consumers that can repair read
    the full list.
    """

    def __init__(self, message: str, *, errors: Optional[Sequence[str]] = None) -> None:
        super().__init__(message)
        self.errors: tuple[str, ...] = tuple(errors) if errors else (message,)


@dataclass(frozen=True)
class ExecutorConfig:
    """Frozen executor configuration, resolved once per invocation.

    Parameters
    ----------
    claude_command:
        Harness command prefix (``MENAGERIE_AUTHOR_CLAUDE_BIN``, shlex-split).
    author_model:
        Optional ``--model`` value (``MENAGERIE_AUTHOR_MODEL``).
    campaign_id:
        Campaign identity (``MENAGERIE_CAMPAIGN_ID`` or ``--campaign``).
    wall_seconds_override:
        Optional wall-grant override (``MENAGERIE_AUTHOR_WALL_SECONDS``).
    pause_after:
        Test-only crash-injection hook (``MENAGERIE_EXECUTOR_PAUSE_AFTER``):
        the executor sleeps after the named phase so a harness test can
        ``kill -9`` it at an exact durable-state boundary. Never set in
        production.
    """

    claude_command: tuple[str, ...]
    author_model: Optional[str]
    campaign_id: Optional[str]
    wall_seconds_override: Optional[float]
    pause_after: Optional[str]

    @classmethod
    def from_env(cls, *, campaign_id: Optional[str] = None) -> "ExecutorConfig":
        """Resolve the configuration and verify the pinned Claude CLI.

        Parameters
        ----------
        campaign_id:
            Optional explicit campaign identity.

        Returns
        -------
        ExecutorConfig
            Frozen invocation configuration.
        """

        # W32: the author harness is pinned by ABSOLUTE path. A bare or
        # relative name resolving through $PATH is exactly the unpinned hole
        # the toolchain-pin authority closes; absence refuses the same way.
        from menagerie.crawler.toolchain_pins import (
            AUTHOR_CLI_PIN_ENV,
            ToolchainPinError,
            resolve_cli_pin,
        )

        try:
            command = resolve_cli_pin(AUTHOR_CLI_PIN_ENV)
        except ToolchainPinError as exc:
            raise AuthorExecutorError(str(exc)) from exc
        if command is None:
            raise AuthorExecutorError(
                f"{AUTHOR_CLI_PIN_ENV} is unset: the author harness must be "
                "pinned by absolute executable path (W32)"
            )
        _verify_claude_cli_version(command)
        raw_wall = os.environ.get(AUTHOR_WALL_SECONDS_ENV)
        return cls(
            claude_command=command,
            author_model=os.environ.get("MENAGERIE_AUTHOR_MODEL") or None,
            campaign_id=campaign_id or os.environ.get("MENAGERIE_CAMPAIGN_ID") or None,
            wall_seconds_override=float(raw_wall) if raw_wall else None,
            pause_after=os.environ.get("MENAGERIE_EXECUTOR_PAUSE_AFTER") or None,
        )

    def wall_seconds(self) -> float:
        """Return the campaign wall grant in seconds.

        Resolution is delegated so this executor and the driver's author lane
        read one table. A driver-published override always names the exact grant
        the lane also sized its stall bound from.
        """

        return resolve_author_wall_seconds(self.campaign_id, self.wall_seconds_override)


def _verify_claude_cli_version(command: Sequence[str]) -> None:
    """Fail closed unless the configured harness is the reviewed CLI release.

    Parameters
    ----------
    command:
        Configured Claude command prefix.

    Raises
    ------
    AuthorExecutorError
        If the command cannot report, or does not report, the pinned version.
    """

    # W32: ONE verification authority. The toolchain-pin module owns the
    # probe (and the per-leg cache the driver's paid-admission sites use);
    # this wrapper only translates its typed refusal into the executor's own
    # error class.
    from menagerie.crawler.toolchain_pins import ToolchainPinError, verify_cli_version

    try:
        verify_cli_version(command, CLAUDE_CLI_VERSION)
    except ToolchainPinError as exc:
        raise AuthorExecutorError(str(exc)) from exc


def _fetched_source_warning_filters(existing: Optional[str]) -> str:
    """Compose ``PYTHONWARNINGS`` so fetched-source compile warnings stay quiet.

    Deliberately narrow: only ``SyntaxWarning`` -- the compile-time class a
    session triggers merely by READING fetched third-party code -- is
    suppressed. Runtime warning classes (deprecations, futures) stay live:
    they can be genuine signal about the code the author is writing.

    Parameters
    ----------
    existing:
        Inherited ``PYTHONWARNINGS`` value, or ``None``.

    Returns
    -------
    str
        The inherited filters with the suppression appended LAST, which is
        where CPython gives a filter precedence, so an operator's own filters
        survive but cannot re-arm the contamination class.
    """

    suppression = "ignore::SyntaxWarning"
    return f"{existing},{suppression}" if existing else suppression


@dataclass(frozen=True)
class SessionOutcome:
    """One ``claude -p`` round trip, machine-observed.

    Parameters
    ----------
    harness:
        Parsed harness JSON from stdout, or ``None`` when unparseable.
    returncode:
        Harness process exit status (``-1`` after an external kill).
    timed_out:
        Whether the external wall kill fired.
    wall_seconds:
        Executor-observed session wall time (excludes broker time).
    session_id:
        Harness-reported session identity when available.
    argv:
        Exact argv, recorded for the attempt record.
    stderr_tail:
        Bounded stderr tail, recorded but never used for classification.
    stdout_tail:
        Bounded raw stdout tail. Load-bearing exactly when ``harness`` is
        ``None``: a session that died emitting unparseable stdout used to leave
        NO trace of what it said (rung 7, m7362 -- the crash cause was in the
        discarded payload). Like ``stderr_tail`` it is evidence, never a
        classification authority.
    teardown_records:
        Machine-authored, bounded kill records for this session.
    campaign_teardown:
        Whether driver shutdown state plus a valid parent teardown record
        prove that this campaign killed the session itself.
    """

    harness: Optional[JsonObject]
    returncode: int
    timed_out: bool
    wall_seconds: float
    session_id: Optional[str]
    argv: tuple[str, ...]
    stderr_tail: str
    stdout_tail: str
    teardown_records: tuple[JsonObject, ...] = ()
    campaign_teardown: bool = False


def _custody_fds_for_paid_exposure() -> tuple[int, ...]:
    """Return the custody descriptors a paid provider spawn must also inherit.

    The provider credential and the paid-subtree custody lock are exposed to a
    descendant in ONE ``pass_fds``, so a process that can buy is always a process
    the driver's kernel liveness oracle can see.  If this executor was launched
    under paid custody but its descriptor did not survive, the credential is NOT
    exposed: a credential-bearing subtree the driver cannot observe is exactly
    the undisclosed orphan the custody invariant forbids.

    Returns
    -------
    tuple[int, ...]
        The inherited custody descriptors, possibly empty outside paid custody.

    Raises
    ------
    AuthorExecutorError
        When paid custody was declared but its descriptor is unusable.
    """

    descriptors = inherited_custody_fds()
    if descriptors or CUSTODY_FD_ENV not in os.environ:
        return descriptors
    raise AuthorExecutorError(
        "paid custody was declared but its subtree lock descriptor did not survive; "
        "refusing to expose provider credential bytes to an unobservable subtree"
    )


def _parent_teardown_record() -> Optional[JsonObject]:
    """Return a validated parent teardown record, if one was durably written.

    The record path arrives in the driver-controlled child environment and the
    driver writes the record before delivering SIGTERM.  Session stdout,
    stderr, and harness JSON are deliberately absent from this derivation.

    Returns
    -------
    dict[str, Any] | None
        Validated machine record, or ``None`` in the conservative direction.
    """

    raw_path = os.environ.get(CAMPAIGN_TEARDOWN_RECORD_ENV)
    if not raw_path:
        return None
    value = _read_json_file(Path(raw_path))
    if value is None:
        return None
    if value.get("version") != CAMPAIGN_TEARDOWN_RECORD_VERSION:
        return None
    if value.get("driver_shutdown_state") not in {
        "halt-teardown-requested",
        "driver-parent-death",
        "guard-deadline-expired",
        "driver-timeout-escalation",
    }:
        return None
    if value.get("signal") != _CAMPAIGN_TEARDOWN_SIGNAL:
        return None
    for key in ("target_pid", "target_pgid"):
        target = value.get(key)
        if isinstance(target, bool) or not isinstance(target, int) or target <= 1:
            return None
    for key in ("reason", "teardown_trigger"):
        field = value.get(key)
        if not isinstance(field, str) or not field.strip():
            return None
    return value


@contextmanager
def _campaign_teardown_signal_handler() -> Iterator[None]:
    """Convert executor SIGTERM into inspectable shutdown state.

    The handler only sets an event.  The ordinary session supervision loop
    reads the already-durable parent record, tears down the isolated Claude
    group, captures its streams, and records the attempt before exiting.

    Yields
    ------
    None
        Control while the executor-specific handler is installed.
    """

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    _CAMPAIGN_TEARDOWN_REQUESTED.clear()
    previous = signal.getsignal(_CAMPAIGN_TEARDOWN_SIGNAL)

    def request_teardown(signum: int, frame: Any) -> None:
        """Record the signal without performing unsafe handler-side teardown."""

        del frame
        if signum == _CAMPAIGN_TEARDOWN_SIGNAL:
            _CAMPAIGN_TEARDOWN_REQUESTED.set()

    signal.signal(_CAMPAIGN_TEARDOWN_SIGNAL, request_teardown)
    try:
        yield
    finally:
        signal.signal(_CAMPAIGN_TEARDOWN_SIGNAL, previous)
        _CAMPAIGN_TEARDOWN_REQUESTED.clear()


def run_claude_session(
    prompt: str,
    *,
    cwd: Path,
    wall_seconds: float,
    config: ExecutorConfig,
    write_root: Path,
    read_roots: Sequence[Path] = (),
    session_id: Optional[str] = None,
    resume: Optional[str] = None,
) -> SessionOutcome:
    """Run one headless harness session under the pinned recipe.

    This is the ONE spawn-call injection point for session confinement: the
    write mechanism (currently harness path-scoped permission rules, per
    :func:`stage_tool_rules`) is applied here and can be swapped for a
    write-only OS sandbox profile without touching call sites.

    Parameters
    ----------
    prompt:
        Complete rendered brief.
    cwd:
        Attempt scratch directory (the session's working directory).
    wall_seconds:
        Wall grant; the external kill fires at grant +10%.
    config:
        Frozen executor configuration.
    write_root:
        The attempt directory — the session's sole writable tree.
    read_roots:
        Additional readable roots for this stage.
    session_id:
        Pre-generated session identity (durable before launch) for stage 1.
    resume:
        Prior session to resume for stage 2 / supplement rounds.

    Returns
    -------
    SessionOutcome
        Machine-observed outcome; never raises for harness failures.
    """

    cwd.mkdir(parents=True, exist_ok=True)
    credential_payload = (exa_mcp_config() + "\n").encode("utf-8")
    custody_fds = _custody_fds_for_paid_exposure()
    with anonymous_credential(credential_payload, directory=cwd) as credential:
        argv: list[str] = [
            *config.claude_command,
            "-p",
            prompt,
            "--setting-sources",
            "",
            "--mcp-config",
            credential.path,
            "--allowedTools",
            *stage_tool_rules(write_root=write_root, read_roots=read_roots),
            "--output-format",
            "json",
        ]
        if config.author_model:
            argv.extend(["--model", config.author_model])
        if resume is not None:
            argv.extend(["--resume", resume])
        elif session_id is not None:
            argv.extend(["--session-id", session_id])
        started = time.monotonic()
        child_environment = os.environ.copy()
        # Claude otherwise writes resumable transcripts under the user's global
        # config tree, outside both campaign disk accounting and the reaper. The
        # attempt scratch tree is already campaign-owned and explicitly reapable.
        child_environment["CLAUDE_CONFIG_DIR"] = str(cwd / ".claude-config")
        child_environment.pop(EXA_API_KEY_ENV, None)
        # AL-25: isolate the result channel from fetched-source warnings. A
        # SyntaxWarning emitted while a session's python imported a fetched
        # third-party file landed inside a schema-validated identifier of the
        # published result (200a, spikingjelly docstring escape), so the
        # attempt burned on warning text the author never wrote. Every python
        # the session runs inherits this filter; the author is grounding on
        # that code, not maintaining it, so its compile-time warnings are
        # noise on every channel they can reach.
        child_environment["PYTHONWARNINGS"] = _fetched_source_warning_filters(
            child_environment.get("PYTHONWARNINGS")
        )
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # The credential descriptor is exposed in the SAME act as the custody
            # lock, never on its own: that is what makes "credential-bearing"
            # imply "observable by the driver's flock oracle" for every paid
            # descendant, including the grandchild that does the buying.
            pass_fds=(*custody_fds, credential.fd),
            # Every Claude session leads its OWN group, including under the
            # paid deadline guard.  A verified session teardown can therefore
            # never signal the guard, executor, or a sibling session.
            start_new_session=True,
        )
        # Bind the group identity while the spawn PID is still held by this child.
        group = capture_process_group(process)
        timed_out = False
        campaign_teardown = False
        teardown_records: list[JsonObject] = []
        stdout = ""
        stderr = ""
        deadline = started + wall_seconds * EXTERNAL_KILL_FACTOR
        while True:
            if _CAMPAIGN_TEARDOWN_REQUESTED.is_set():
                parent_record = _parent_teardown_record()
                campaign_teardown = bool(
                    parent_record is not None
                    and parent_record["driver_shutdown_state"]
                    in {"halt-teardown-requested", "driver-parent-death"}
                )
                if parent_record is not None:
                    teardown_records.append(
                        {"record_kind": "driver-teardown-intent", **parent_record}
                    )
                stdout, stderr = _kill_session(
                    group,
                    reason=(
                        str(parent_record["reason"])
                        if parent_record is not None
                        else "executor received SIGTERM without a valid parent record"
                    ),
                    teardown_trigger=(
                        str(parent_record["teardown_trigger"])
                        if parent_record is not None
                        else "executor-sigterm-unattributed"
                    ),
                    records=teardown_records,
                )
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                timed_out = True
                stdout, stderr = _kill_session(
                    group,
                    reason="author session exceeded its machine wall grant",
                    teardown_trigger="session-wall-timeout",
                    records=teardown_records,
                )
                break
            try:
                stdout, stderr = process.communicate(
                    timeout=min(_SESSION_COMMUNICATE_POLL_SECONDS, remaining)
                )
            except subprocess.TimeoutExpired:
                continue
            break
    wall = time.monotonic() - started
    harness = _parse_harness_json(stdout or "")
    reported = harness.get("session_id") if harness else None
    return SessionOutcome(
        harness=harness,
        returncode=process.returncode if process.returncode is not None else -1,
        timed_out=timed_out,
        wall_seconds=wall,
        session_id=str(reported) if reported else session_id,
        argv=tuple(argv),
        stderr_tail=(stderr or "")[-2000:],
        stdout_tail=(stdout or "")[-2000:],
        teardown_records=tuple(teardown_records),
        campaign_teardown=campaign_teardown,
    )


def _kill_session(
    group: ProcessGroupHandle,
    *,
    reason: str = "author session teardown",
    teardown_trigger: str = "direct-session-teardown",
    records: Optional[list[JsonObject]] = None,
) -> tuple[str, str]:
    """Terminate a session's verified process group: SIGTERM, grace, SIGKILL.

    Both phases go through the supervisor's one hardened teardown, which signals
    only a group this parent can still prove it owns. When that proof fails the
    group is left entirely alone and only the root child -- the one process whose
    identity is never in doubt while it is unreaped -- is signalled, so a hung
    session still cannot outlive its wall grant.

    Parameters
    ----------
    group:
        Spawn-time group identity from :func:`capture_process_group`.
    reason:
        Machine-owned explanation for this teardown.
    teardown_trigger:
        Closed caller boundary that initiated teardown.
    records:
        Optional diagnostic sink receiving one record per delivered signal.

    Returns
    -------
    tuple[str, str]
        Whatever the session managed to emit before it was torn down.
    """

    process = group.process
    term = kill_process_group(group, signal.SIGTERM)
    _append_session_teardown_record(records, term, signal.SIGTERM, reason, teardown_trigger)
    if not term.benign:
        process.terminate()
    try:
        return process.communicate(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        forced = kill_process_group(group, signal.SIGKILL)
        _append_session_teardown_record(records, forced, signal.SIGKILL, reason, teardown_trigger)
        if not forced.benign:
            process.kill()
        try:
            return process.communicate(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return "", ""


def _append_session_teardown_record(
    records: Optional[list[JsonObject]],
    teardown: Any,
    signal_number: int,
    reason: str,
    teardown_trigger: str,
) -> None:
    """Append one bounded machine-authored session kill record.

    Parameters
    ----------
    records:
        Mutable diagnostic sink, or ``None`` for compatibility callers.
    teardown:
        Supervisor :class:`ProcessGroupTeardown` result.
    signal_number:
        Signal the supervisor attempted to deliver.
    reason:
        Machine-owned teardown explanation.
    teardown_trigger:
        Caller boundary that initiated teardown.
    """

    if records is None:
        return
    records.append(
        {
            "record_kind": "session-process-group-teardown",
            "target_pid": teardown.pid,
            "target_pgid": teardown.pgid,
            "signal": signal_number,
            "reason": reason[:512],
            "teardown_trigger": teardown_trigger[:128],
            "outcome": teardown.outcome,
            "detail": teardown.detail[:512],
            "recorded_at": utc_now(),
        }
    )


def _parse_harness_json(stdout: str) -> Optional[JsonObject]:
    """Parse the harness JSON document from session stdout."""

    text = stdout.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def structured_limit_signal(harness: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return the structured usage-limit signal, if the harness declared one.

    Only parsed harness JSON fields decide a quota exit. Free text never does.
    """

    if not harness:
        return None
    terminal = str(harness.get("terminal_reason", ""))
    if terminal in _LIMIT_TERMINAL_REASONS:
        return terminal
    info = harness.get("rate_limit_info")
    if isinstance(info, Mapping) and str(info.get("status", "")) == _LIMIT_RATE_STATUS:
        # ``rateLimitType`` separates the five-hour window from the seven-day
        # weekly family; both pause identically but the kind is worth recording.
        kind = str(info.get("rateLimitType", "") or "unspecified")
        return f"rate_limit_rejected:{kind}"
    status = harness.get("api_error_status")
    if not isinstance(status, bool) and status == _RATE_LIMITED_HTTP_STATUS:
        return f"api_error_status:{_RATE_LIMITED_HTTP_STATUS}"
    error = harness.get("error")
    if isinstance(error, Mapping) and str(error.get("type", "")) in _LIMIT_ERROR_TYPES:
        return str(error["type"])
    return None


def structured_limit_reset_at(harness: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return the harness's own declared reset instant as an ISO-8601 UTC string.

    Reads ``rate_limit_info.resetsAt`` (unix seconds, claude-agent-sdk 0.3.211
    ``sdk.d.ts`` line 4153). This is the only authoritative reset the harness
    offers; when it is absent the driver falls back to a guessed ``now + 1h``,
    which is right for a five-hour window and badly wrong for a seven-day one.
    Fractional seconds are truncated so the emitted timestamp never contains a
    ``.``, which the driver's reset phrase parser treats as a sentence boundary.
    """

    if not harness:
        return None
    info = harness.get("rate_limit_info")
    if not isinstance(info, Mapping):
        return None
    raw = info.get("resetsAt")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    try:
        moment = datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None
    # RFC 3339 ``Z`` form, not ``+00:00``: the wake layer rejects any reset that
    # does not end in ``Z`` (``wakeup._parse_utc``), so emitting the offset form
    # here would turn a correctly detected pause into a scheduling crash.
    return moment.isoformat().replace("+00:00", "Z")


def _backoff_detail(limit: str, harness: Optional[Mapping[str, Any]]) -> str:
    """Announce a provider pause on stdout and return the operator detail line.

    The driver classifies an author backoff from the process's *structured
    stdout* (``_structured_author_error`` then ``parse_author_reset_at``), so the
    harness's own reset instant has to travel on that channel or it is lost: the
    typed exit alone yields a pause with no reset, and the driver then guesses one
    hour. Emitting it here upgrades the recorded pause from ``guessed`` to
    ``observed`` -- which for a seven-day weekly limit is the difference between
    one wake and ~168 futile ones.

    The notice is machine-built from the parsed harness JSON only; no free text
    from the session influences it.
    """

    detail = f"author provider pause (structured signal {limit})"
    message = f"author provider pause: usage limit ({limit})"
    reset_at = structured_limit_reset_at(harness)
    if reset_at is not None:
        detail = f"{detail} resets at {reset_at}"
        message = f"{message}, resets at {reset_at}"
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "author-provider-pause",
                "is_error": True,
                "message": f"{message}.",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return detail


#: Bounded number of externally-controlled failure-text characters retained.
_SESSION_FAILURE_TEXT_MAX_CHARS = 2000

#: Case-folded markers naming a host credential outage in a failed session's
#: own error text (observed verbatim: rung 7 m7362 died rc=1 in 0.6s with
#: "Not logged in · Please run /login" sitting in the discarded stdout).
_AUTH_FAILURE_MARKERS = ("not logged in", "please run /login", "invalid api key")

#: Case-folded markers naming transient provider overload (observed verbatim on
#: the clone-2 specimen: "API Error: Repeated 529 Overloaded"). Deliberately
#: NOT promoted to a pause: a 529 storm carries no reset timestamp and clears
#: in seconds -- the driver's bounded backoff ladder is its designed consumer,
#: while the STRUCTURED ``overloaded_error`` harness field keeps its existing
#: pause routing through :func:`structured_limit_signal` untouched.
_OVERLOAD_FAILURE_MARKERS = ("overloaded", "529")

#: Closed machine terminal reasons that are infrastructure without consulting
#: session text.  The shipped Claude Code 2.1.220 ``TerminalReason`` vocabulary
#: also contains content/effort outcomes (for example ``prompt_too_long``,
#: ``max_turns``, and ``malformed_tool_use_exhausted``); those deliberately stay
#: chargeable.  ``aborted_streaming`` is narrower still: it refunds only with
#: the supervisor's SIGTERM return code, pinning the 2026-08-20 storm shape.
_INFRASTRUCTURE_TERMINAL_REASONS = frozenset({"api_error"})
_SUPERVISOR_SIGTERM_RETURNCODES = frozenset({-signal.SIGTERM, 128 + signal.SIGTERM})

#: Case-folded markers naming an author CONTENT refusal in the session's own
#: final text: the session ran to completion, declined the work in prose, and
#: published nothing. Checked LAST among the text arms so a host-condition
#: marker (auth, overload) always outranks refusal-sounding remainder text,
#: and never checked against traceback-shaped crash output by construction --
#: a refusal is first-person prose, not an exception rendering. The label is
#: diagnostic only (T4-T4: it makes the refusal cohort durably distinguishable
#: for the rung-200/500 salvage audit); it carries no pause or retry
#: authority.
_REFUSAL_FAILURE_MARKERS = (
    "i can't help with",
    "i cannot help with",
    "i can't assist with",
    "i cannot assist with",
    "i'm not able to help",
    "i am not able to help",
    "i'm unable to help",
    "i am unable to help",
    "i won't be able to help",
    "i will not be able to help",
    "i must decline",
    "i have to decline",
    "i can't comply with",
    "i cannot comply with",
    "against my guidelines",
)


def _harness_error_text(harness: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return the harness result document's own error-bearing text, bounded.

    Reads the same fields ``author_dispatch._structured_author_error`` reads
    (``result``, ``message``, and the ``error`` object) but without requiring
    ``is_error``: a session that died with rc != 0 said whatever it said, and
    that text is evidence worth keeping either way.

    Parameters
    ----------
    harness:
        Parsed harness JSON, or ``None``.

    Returns
    -------
    str | None
        Joined error-bearing text, bounded, or ``None`` when the harness
        carried none.
    """

    if not harness:
        return None
    fields: list[str] = []
    for key in ("result", "message"):
        value = harness.get(key)
        if isinstance(value, str) and value.strip():
            fields.append(value.strip())
    error = harness.get("error")
    if isinstance(error, str) and error.strip():
        fields.append(error.strip())
    elif isinstance(error, Mapping):
        for key in ("type", "code", "message"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                fields.append(value.strip())
    if not fields:
        return None
    return "\n".join(fields)[:_SESSION_FAILURE_TEXT_MAX_CHARS]


def session_failure_class(outcome: SessionOutcome) -> str:
    """Label one failed session's cause for the durable record and retry detail.

    Structured harness fields decide first; the session's own error text is
    consulted last and only to LABEL. The companion
    :func:`session_failure_class_basis` records whether the label came from a
    structured field or from quarantined diagnostic text. The driver grants
    weather backoff authority only to an executor-authored failure notice whose
    machine-observed harness fields match the real provider-overloaded shape.

    Parameters
    ----------
    outcome:
        Machine-observed failed session round trip.

    Returns
    -------
    str
        One of :data:`~menagerie.crawler.constants.SESSION_FAILURE_CLASSES`:
        ``auth-unavailable``, ``provider-overloaded``, ``content-refusal``,
        or ``unclassified``.
    """

    harness = outcome.harness or {}
    status = harness.get("api_error_status")
    if not isinstance(status, bool) and status == 529:
        return "provider-overloaded"
    error = harness.get("error")
    error_type = str(error.get("type", "")) if isinstance(error, Mapping) else ""
    if error_type == "authentication_error":
        return "auth-unavailable"
    text = (_harness_error_text(harness) or outcome.stdout_tail or "").lower()
    if any(marker in text for marker in _AUTH_FAILURE_MARKERS):
        return "auth-unavailable"
    if any(marker in text for marker in _OVERLOAD_FAILURE_MARKERS):
        return "provider-overloaded"
    if any(marker in text for marker in _REFUSAL_FAILURE_MARKERS):
        return "content-refusal"
    return "unclassified"


def session_failure_class_basis(outcome: SessionOutcome) -> str:
    """Return the machine-readable basis for :func:`session_failure_class`.

    Parameters
    ----------
    outcome:
        Machine-observed failed session round trip.

    Returns
    -------
    str
        Stable basis label for the failure-class diagnostic.
    """

    harness = outcome.harness or {}
    status = harness.get("api_error_status")
    if not isinstance(status, bool) and status == 529:
        return "api_error_status:529"
    error = harness.get("error")
    error_type = str(error.get("type", "")) if isinstance(error, Mapping) else ""
    if error_type == "authentication_error":
        return "error.type:authentication_error"
    text = (_harness_error_text(harness) or outcome.stdout_tail or "").lower()
    if any(marker in text for marker in _AUTH_FAILURE_MARKERS):
        return "diagnostic-text:auth-marker"
    if any(marker in text for marker in _OVERLOAD_FAILURE_MARKERS):
        return "diagnostic-text:overload-marker"
    if any(marker in text for marker in _REFUSAL_FAILURE_MARKERS):
        return "diagnostic-text:refusal-marker"
    return "none"


def session_infrastructure_failure_basis(outcome: SessionOutcome) -> Optional[str]:
    """Return exact machine evidence for an attempt-refunding session death.

    The harness envelope and subprocess status are machine-derived.  Subtype,
    stdout/stderr, error text, and the diagnostic ``failure_class`` are never
    consulted, because each may contain producer-controlled content.  Missing,
    unknown, or contradictory fields return ``None`` so the attempt remains
    chargeable.

    Parameters
    ----------
    outcome:
        Machine-observed failed session round trip.

    Returns
    -------
    str | None
        Closed evidence basis, or ``None`` in the conservative direction.
    """

    harness = outcome.harness
    if not isinstance(harness, Mapping) or harness.get("is_error") is not True:
        return None
    terminal = harness.get("terminal_reason")
    if not isinstance(terminal, str) or outcome.returncode == 0:
        return None
    if (
        terminal == "aborted_streaming"
        and outcome.returncode in _SUPERVISOR_SIGTERM_RETURNCODES
    ):
        return "harness:is_error+terminal:aborted_streaming+returncode:sigterm"
    if terminal in _INFRASTRUCTURE_TERMINAL_REASONS:
        return f"harness:is_error+terminal:{terminal}+returncode:nonzero"
    return None


def session_refusal_excerpt(outcome: SessionOutcome) -> Optional[str]:
    """Return the bounded refusal quote for a content-refusal session.

    The excerpt is the head of the same text the classifier labeled from --
    the session's own final prose -- bounded to
    :data:`~menagerie.crawler.constants.REFUSAL_EXCERPT_MAX_CHARS` so a
    canonical record can carry the quote durably without ever carrying the
    full transcript (T4-T4: the full prose previously lived only in gitignored
    scratch that host teardown deletes).

    Parameters
    ----------
    outcome:
        Machine-observed failed session round trip.

    Returns
    -------
    str | None
        Bounded refusal quote, or ``None`` when the session is not a
        content refusal.
    """

    if session_failure_class(outcome) != "content-refusal":
        return None
    text = (_harness_error_text(outcome.harness) or outcome.stdout_tail or "").strip()
    return text[:REFUSAL_EXCERPT_MAX_CHARS] or None


def session_failure_evidence(outcome: SessionOutcome) -> JsonObject:
    """Return bounded, durable evidence for one failed session round trip.

    Rung 7's abort root cause was invisible for exactly this gap: the executor
    parsed the harness JSON for effort fields and DISCARDED the ``result``
    error text and raw stdout, and both observed crash causes ("Not logged in
    · Please run /login" on m7362; "API Error: Repeated 529 Overloaded" on the
    clone-2 specimen) were sitting in the discarded payload. This record makes
    the cause durable on the attempt and consumable by the driver's retry
    ladder.

    Quarantine rules: every ``*_quarantined`` value is externally controlled
    text (the session/provider wrote it). It is bounded, stored as data, and
    never used as classification authority beyond the diagnostic
    ``failure_class`` label.

    Parameters
    ----------
    outcome:
        Machine-observed failed session round trip.

    Returns
    -------
    dict[str, Any]
        Bounded failure evidence for the attempt record and failure notice.
    """

    harness = outcome.harness
    subtype = (harness or {}).get("subtype")
    terminal_reason = (harness or {}).get("terminal_reason")
    is_error = (harness or {}).get("is_error")
    evidence: JsonObject = {
        "classification": "campaign-teardown" if outcome.campaign_teardown else "author-failure",
        "returncode": outcome.returncode,
        "failure_class": session_failure_class(outcome),
        "failure_class_basis": session_failure_class_basis(outcome),
        "infrastructure_failure_basis": session_infrastructure_failure_basis(outcome),
        "harness_subtype": subtype if isinstance(subtype, str) else None,
        "harness_terminal_reason": (terminal_reason if isinstance(terminal_reason, str) else None),
        "harness_is_error": is_error if isinstance(is_error, bool) else None,
        "session_error_text_quarantined": _harness_error_text(harness),
        # Bounded refusal quote, or None for every non-refusal failure: one
        # evidence shape for every consumer. Like every *_quarantined value it
        # is externally controlled text stored as data, never authority.
        "refusal_excerpt_quarantined": session_refusal_excerpt(outcome),
        "stderr_tail_quarantined": (
            outcome.stderr_tail[-_SESSION_FAILURE_TEXT_MAX_CHARS:] or None
        ),
        "teardown_records": [dict(record) for record in outcome.teardown_records],
    }
    if harness is None:
        evidence["stdout_tail_quarantined"] = (
            outcome.stdout_tail[-_SESSION_FAILURE_TEXT_MAX_CHARS:] or None
        )
    return evidence


def _session_failure_notice(evidence: JsonObject) -> None:
    """Print the machine-built failure notice the driver's stdio tail records.

    ``classify_author_exit`` slices the executor's combined stdio into the
    bounded ``tail`` that becomes the ``RetryableOperatorError`` message -- and,
    when the retry budget spends, ``AuthorSessionRetryExhausted.detail``. Before
    this notice the tail carried only the generic status line, so the exhausted
    record could not name its own cause.

    The notice deliberately carries NO ``result``, ``message``, or ``error``
    key: ``author_dispatch._structured_author_error`` reads exactly those
    fields off any ``is_error`` stdout JSON when classifying a backoff pause,
    and a quarantined session text containing e.g. "overloaded" would otherwise
    promote an ordinary retryable crash into a campaign-wide usage pause. The
    quarantined text rides under its own key: visible to the retry ladder's
    bounded detail, invisible to pause authority.

    Parameters
    ----------
    evidence:
        Bounded failure evidence from :func:`session_failure_evidence`.
    """

    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "author-session-failure",
                **evidence,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def effort_from_session(outcome: SessionOutcome) -> JsonObject:
    """Extract the machine-observed effort record for one session.

    Every value is taken verbatim from the harness JSON or the executor's own
    clock; nothing is self-reported by the model.
    """

    harness = outcome.harness or {}
    usage = harness.get("usage")
    return {
        "wall_seconds_observed": round(outcome.wall_seconds, 3),
        "duration_ms": harness.get("duration_ms"),
        "duration_api_ms": harness.get("duration_api_ms"),
        "num_turns": harness.get("num_turns"),
        "total_cost_usd": harness.get("total_cost_usd"),
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
        "timed_out": outcome.timed_out,
        "returncode": outcome.returncode,
    }


# -- brief rendering -------------------------------------------------------


def _read_prompt(root: Path, name: str) -> str:
    """Read one prompt fragment, failing typed when it is missing."""

    try:
        return (root / name).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AuthorExecutorError(f"executor prompt {name} is unavailable: {exc}") from exc


def _campaign_fragment(campaign_id: Optional[str]) -> Optional[str]:
    """Return the campaign standards fragment, failing loudly on a bad id."""

    if campaign_id is None:
        return None
    path = _POOL_PROMPTS / f"campaign_{campaign_id}.md"
    if not path.is_file():
        raise AuthorExecutorError(f"campaign {campaign_id!r} has no standards fragment at {path}")
    return path.read_text(encoding="utf-8").strip()


def author_steering_surface_digest(campaign_id: Optional[str]) -> str:
    """AL-22: content identity of the prompt surfaces steering ONE attempt.

    The historical ``author_prompt_identity`` axis hashes exactly one file
    (``prompts/claude_crawler_author_v2.txt``) that the headless executor
    never renders, so the prompts that actually steered every 200a attempt
    (the executor stage briefs and the campaign fragment) were absent from
    every identity axis and every attempt record. This digest closes that
    gap: a per-file ``sha256`` map over exactly the steering surfaces,
    content-addressed as one identity, stamped on each attempt's stage
    records and registered as the DISCLOSED-NON-STALING rotation axis
    ``author_steering_surface_identity`` -- a prompt edit rotates it
    (disclosed, auditable) without staling any carried head.

    Deliberately file-resident surfaces only: the rendered brief embeds
    per-attempt facts (nonce, deadline) and is already durable verbatim in
    the attempt's recorded argv, so hashing it would make every attempt's
    digest unique and identity-free.

    Parameters
    ----------
    campaign_id:
        Active campaign whose standards fragment steers the attempt; the
        fragment is omitted from the surface when ``None``.

    Returns
    -------
    str
        ``sha256:`` content address over the named surface map.

    Raises
    ------
    AuthorExecutorError
        If a steering surface file is unreadable (the same typed refusal a
        render would hit).
    """

    from menagerie.crawler.identity import hash_bytes, stable_hash

    members: list[tuple[str, Path]] = [
        ("executor/stage1_discovery.md", _EXECUTOR_PROMPTS / "stage1_discovery.md"),
        ("executor/stage2_author.md", _EXECUTOR_PROMPTS / "stage2_author.md"),
        ("claude_crawler_author_v2.txt", _PROMPT_ROOT / "claude_crawler_author_v2.txt"),
    ]
    if campaign_id is not None:
        members.append(
            (f"pool/campaign_{campaign_id}.md", _POOL_PROMPTS / f"campaign_{campaign_id}.md")
        )
    surfaces: dict[str, str] = {}
    for name, path in members:
        try:
            surfaces[name] = hash_bytes(path.read_bytes())
        except OSError as exc:
            raise AuthorExecutorError(
                f"steering surface {name} is unavailable: {exc}"
            ) from exc
    return stable_hash(
        {
            "steering_surface_version": "menagerie-author-steering-surface-v1",
            "surfaces": surfaces,
        }
    )


#: Machine-owned rule appended to EVERY repair brief, under the verbatim checker
#: findings. A checker repair is free prose, and the two most natural phrasings of a
#: metadata repair -- "ground or remove the asserted US country value", "ground or drop
#: it" -- describe a first half that is legal and a second half that is not: emptying a
#: gated field WITHOUT a typed availability state is refused by the deterministic gate as
#: an ungrounded claim. In the 2026-08-05 twenty-model rung both models that received
#: that exact repair (m4334, m7362) complied literally, emptied ``country``, declared no
#: availability record, and terminalized as permanently dead records. Neither author was
#: wrong about the evidence; each was told to do something the next gate refuses. The
#: rule is stamped by the machine rather than requested of the checker, because relying
#: on the rejecting party to phrase the repair completely is what failed.
_REPAIR_STANDING_RULE = (
    "STANDING RULE ON EVERY REPAIR -- READ BEFORE ACTING ON THE FINDINGS ABOVE\n"
    "A repair that says to REMOVE, DROP, RETRACT, or NOT ASSERT a gated value is only "
    "half an instruction, and obeying only that half is refused. Emptying a gated field "
    "is legal ONLY together with a typed external_metadata.availability.<key> state "
    "(status not-found-after-search / none-exist / not-applicable, basis from the closed "
    "enum -- an absence you established by reading the frozen sources is "
    "search-exhausted) plus the bounded search_report. A gated field left bare-empty is "
    "an ungrounded claim and the proposal dies. So: ground the value, or empty it AND "
    "declare the absence. There is no third option, and 'the checker told me to remove "
    "it' is not one."
)


def _prior_attempts_section(author_root: Path) -> Optional[str]:
    """Render the WHAT WENT WRONG LAST TIME feedback block."""

    summaries = prior_attempts_summary(author_root)
    relevant = [
        summary
        for summary in summaries
        if summary.get("status") in ("failed", "superseded")
        or summary.get("failure_reason")
        or summary.get("checker_findings")
    ]
    if not relevant:
        return None
    lines = [
        "## WHAT WENT WRONG LAST TIME (binding feedback -- do not repeat it)",
        "",
    ]
    for summary in relevant:
        lines.append(
            f"- attempt {summary.get('attempt_number')} "
            f"({summary.get('kind')}, {summary.get('status')}): "
            f"stage={summary.get('failure_stage')!r} "
            f"reason={summary.get('failure_reason')!r}"
        )
        failure_detail = summary.get("failure_detail")
        if failure_detail:
            lines.append(
                f"  failure detail, verbatim: {json.dumps(failure_detail, sort_keys=True)}"
            )
        findings = summary.get("checker_findings")
        if findings:
            lines.append(f"  checker findings, verbatim: {json.dumps(findings)}")
    lines.extend(("", _REPAIR_STANDING_RULE))
    return "\n".join(lines)


def _deadline_iso(wall_seconds: float) -> str:
    """Return the wall deadline as an ISO instant the brief can carry."""

    deadline = datetime.now(timezone.utc) + timedelta(seconds=wall_seconds)
    return deadline.isoformat().replace("+00:00", "Z")


def _facts(lines: Sequence[str]) -> str:
    """Render the binding JOB FACTS block."""

    return "\n".join(["## JOB FACTS (binding)", "", *lines])


def render_stage1_brief(
    *,
    request_path: Path,
    request: Mapping[str, Any],
    attempt: AttemptHandle,
    config: ExecutorConfig,
    author_root: Path,
) -> str:
    """Render the complete stage-1 discovery brief."""

    wall = config.wall_seconds()
    deadline = _deadline_iso(wall)
    facts = _facts(
        [
            f"- stable_id: `{request.get('stable_id')}`",
            f"- work_id: `{request.get('work_id')}`",
            f"- campaign: `{config.campaign_id}`",
            f"- attempt: `{attempt.number}` nonce `{attempt.nonce}`",
            f"- REQUEST envelope to read first: `{request_path}`",
            f"- DISCOVERY output path, exact: `{attempt.paths.directory / 'discovery.json'}`",
            f"- max_sources: {request.get('max_sources')}",
            f"- attempt started: `{utc_now()}` -- the FULL {wall / 60:.0f}-minute grant is"
            " yours from this instant; prior attempts' time is not deducted",
            f"- wall deadline: `{deadline}` (external kill at +10%)",
            f"- clock, exact: `{identity_tool_command()} --clock --deadline {deadline}`"
            " -- the only granted time observation; never guess elapsed time",
        ]
    )
    sections = [facts, _read_prompt(_EXECUTOR_PROMPTS, "stage1_discovery.md")]
    campaign = _campaign_fragment(config.campaign_id)
    if campaign is not None:
        sections.append(campaign)
    feedback = _prior_attempts_section(author_root)
    if feedback is not None:
        sections.append(feedback)
    return "\n\n---\n\n".join(sections) + "\n"


def render_stage2_brief(
    *,
    request_path: Path,
    request: Mapping[str, Any],
    attempt: AttemptHandle,
    config: ExecutorConfig,
    author_root: Path,
    cold_start_discovery: Optional[str] = None,
) -> str:
    """Render the complete stage-2 authoring brief."""

    wall = config.wall_seconds()
    deadline = _deadline_iso(wall)
    facts = _facts(
        [
            f"- stable_id: `{request.get('stable_id')}`",
            f"- work_id: `{request.get('work_id')}`",
            f"- campaign: `{config.campaign_id}`",
            f"- attempt: `{attempt.number}` nonce `{attempt.nonce}`",
            f"- REQUEST envelope to read first: `{request_path}`",
            f"- RESULT output path, exact: `{attempt.paths.directory / 'result.json'}`",
            f"- STAGED MODEL dir, exact: `{attempt.paths.directory / 'model'}`",
            f"- envelope allowed_model_dir (executor-mirrored at publication): "
            f"`{request.get('allowed_model_dir')}`",
            f"- PROPOSAL schema, exact: `{_SCHEMA_ROOT / 'author-proposal-v3.schema.json'}`",
            f"- REFERENCED schema directory, exact: `{_SCHEMA_ROOT}`",
            # The one command the session may run. Rendered as the exact literal
            # the permission grant pins, so the brief and the allowlist cannot
            # drift into naming two different invocations.
            f"- identity calculator, exact: `{identity_tool_command()}`",
            f"- attempt started: `{utc_now()}` -- the FULL {wall / 60:.0f}-minute grant is"
            " yours from this instant; prior attempts' time is not deducted",
            f"- wall deadline: `{deadline}` (external kill at +10%)",
            f"- clock, exact: `{identity_tool_command()} --clock --deadline {deadline}`"
            " -- the only granted time observation; never guess elapsed time",
        ]
    )
    sections = [facts, _read_prompt(_EXECUTOR_PROMPTS, "stage2_author.md")]
    campaign = _campaign_fragment(config.campaign_id)
    if campaign is not None:
        sections.append(campaign)
    if cold_start_discovery is not None:
        sections.append(
            "## COLD START -- the stage-1 session was lost\n\n"
            "The recorded stage-1 discovery output follows verbatim:\n\n"
            "```json\n" + cold_start_discovery + "\n```"
        )
    feedback = _prior_attempts_section(author_root)
    if feedback is not None:
        sections.append(feedback)
    return "\n\n---\n\n".join(sections) + "\n"


# -- publication -----------------------------------------------------------


def _publish_bytes(
    local_path: Path,
    required_path: Path,
    *,
    nonce: str,
    kind: str,
    author_model: Optional[str],
    stable_id: Optional[str] = None,
    attempt_number: Optional[int] = None,
) -> str:
    """Atomically publish one executor-local artifact and print its receipt.

    The executor is the only publisher: sessions write only their own attempt
    (or probe) directory, and this function moves bytes across that boundary.
    The published bytes are re-read and digest-verified, and the nonce-bound
    receipt is printed on stdout for the lane to verify again.

    Returns
    -------
    str
        The published digest.
    """

    data = local_path.read_bytes()
    digest = hash_bytes(data)
    required_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = required_path.with_name(f".{required_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, required_path)
    finally:
        temporary.unlink(missing_ok=True)
    published = hash_bytes(required_path.read_bytes())
    if published != digest:
        raise AuthorExecutorError(
            f"published bytes at {required_path} do not match the attempt artifact"
        )
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "kind": kind,
        "stable_id": stable_id,
        "attempt_number": attempt_number,
        "attempt_nonce": nonce,
        "result_sha256": digest,
        "published_path": str(required_path),
        # The model this executor ACTUALLY launched (its resolved ``--model``),
        # attested spawn-side because no downstream layer can observe it. An
        # explicit ``None`` means the launch was unpinned and the harness
        # default ran -- unknown, said out loud, never the envelope's claim.
        "author_model": author_model,
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return digest


def _publish(
    attempt: AttemptHandle,
    local_path: Path,
    required_path: Path,
    *,
    kind: str,
    author_model: Optional[str],
) -> str:
    """Publish one attempt-local artifact, binding the attempt record to it."""

    digest = _publish_bytes(
        local_path,
        required_path,
        nonce=attempt.nonce,
        kind=kind,
        author_model=author_model,
        stable_id=str(attempt.record.get("stable_id") or "") or None,
        attempt_number=attempt.number,
    )
    attempt.update(
        published={
            "kind": kind,
            "path": str(required_path),
            "sha256": digest,
            "published_at": utc_now(),
        }
    )
    attempt.event("published", kind=kind, path=str(required_path), sha256=digest)
    return digest


def _mirror_staged_model(
    attempt: AttemptHandle, model_dir: Path, *, phase: str = "publication"
) -> None:
    """Mirror the attempt's staged ``model/`` tree into the envelope's model dir.

    Sessions never write the shared model directory (the attempt directory is
    the sole writable path); the executor performs this copy itself, so
    cross-attempt contamination of staged code is structurally impossible.

    The copy runs at TWO moments, disclosed by ``phase``: before every
    pre-publication validation pass (``"prepublication"``) and at publication
    (``"publication"``). The pre-publication mirror exists because the deep
    replay resolves ``implementation.code_path`` against the envelope's
    ``allowed_model_dir`` -- exactly as the driver will after publication --
    and that directory is otherwise only populated AT publication. Rung 9b
    (m11584/m5445/m8189) validated against the still-empty directory,
    manufactured a false ``cannot read staged code_path`` defect for sessions
    that HAD staged their code, and drove their one repair round into
    "fixing" a non-defect (an absolute ``code_path``) that then genuinely
    killed the result driver-side.
    """

    staged = attempt.paths.directory / "model"
    if not staged.is_dir():
        return

    if model_dir.is_symlink() or model_dir.is_file():
        model_dir.unlink()
    elif model_dir.is_dir():
        shutil.rmtree(model_dir)
    shutil.copytree(staged, model_dir)
    attempt.event("staged-model-mirrored", destination=str(model_dir), phase=phase)


def _pause_hook(config: ExecutorConfig, phase: str) -> None:
    """Test-only crash-injection point: sleep after a durable-state boundary."""

    if config.pause_after == phase:
        time.sleep(600.0)


def _fail(
    attempt: AttemptHandle,
    *,
    stage: str,
    reason: str,
    exit_code: int,
    detail: JsonObject | None = None,
) -> tuple[int, str]:
    """Record one typed attempt failure and return its exit."""

    attempt.update(
        status="failed",
        outcome={
            "kind": "failure",
            "failure_stage": stage,
            "failure_reason": reason,
            "detail": detail or {},
        },
    )
    attempt.event("failed", stage=stage, reason=reason)
    return exit_code, f"author executor {stage} failed: {reason} (attempt {attempt.nonce})"


# -- source-request (stage 1 + broker) -------------------------------------


def _discovery_envelope_from_author_payload(
    author_payload: Mapping[str, Any],
    request: Mapping[str, Any],
) -> JsonObject:
    """Wrap one author-owned discovery payload in machine-owned bindings.

    Parameters
    ----------
    author_payload:
        Arm-specific discovery judgment written by the research session.
    request:
        Trusted source-request envelope supplying stable and work identities.

    Returns
    -------
    dict[str, Any]
        Complete registered source-discovery envelope.
    """

    payload = deepcopy(dict(author_payload))
    arm = payload.get("arm")
    return {
        "schema_version": SOURCE_DISCOVERY_SCHEMA_VERSION,
        "stable_id": str(request.get("stable_id", "")),
        "work_id": str(request.get("work_id", "")),
        "arm": arm,
        "payload": payload,
    }


def _write_json_object(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write one canonical JSON object.

    Parameters
    ----------
    path:
        Destination path.
    value:
        JSON object to serialize.
    """

    data = canonical_json_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def serve_source_request(
    request_path: Path,
    request: Mapping[str, Any],
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Serve one stage-1 discovery round: session, broker, publication."""

    author_root = request_path.parent
    stable_id = str(request.get("stable_id", ""))
    required_output = Path(str(request.get("required_output_path", "")))
    if not stable_id or not str(required_output):
        return EXIT_PERMANENT, "source request lacks stable_id or required_output_path"
    max_sources = int(request.get("max_sources", 20) or 20)

    resumable = latest_attempt(
        author_root, stable_id=stable_id, statuses=frozenset({"stage1-complete"})
    )
    if resumable is not None:
        # Crash recovery: stage 1 already completed durably; replay the broker
        # from the recorded discovery output without re-running the session.
        resumable.event("resumed-broker-from-record")
        return _complete_source_discovery(
            resumable,
            request=request,
            required_output=required_output,
            max_sources=max_sources,
            config=config,
        )

    attempt = new_attempt(
        author_root,
        stable_id=stable_id,
        campaign_id=config.campaign_id,
        kind="source-request",
    )
    brief = render_stage1_brief(
        request_path=request_path,
        request=request,
        attempt=attempt,
        config=config,
        author_root=author_root,
    )
    session_id = str(uuid.uuid4())
    attempt.update(
        status="stage1-running",
        stage1={
            "session_id": session_id,
            "started_at": utc_now(),
            # AL-22: the exact prompt-surface identity this stage was
            # steered by, durable per attempt.
            "steering_surface_digest": author_steering_surface_digest(
                config.campaign_id
            ),
        },
    )
    outcome = run_claude_session(
        brief,
        cwd=attempt.paths.scratch,
        wall_seconds=config.wall_seconds(),
        config=config,
        write_root=attempt.paths.directory,
        read_roots=[author_root],
        session_id=session_id,
    )
    stage1 = {
        "session_id": outcome.session_id or session_id,
        "started_at": attempt.record["stage1"]["started_at"],
        "completed_at": utc_now(),
        "effort": effort_from_session(outcome),
        "argv": list(outcome.argv),
        "stderr_tail": outcome.stderr_tail,
        "teardown_records": [dict(record) for record in outcome.teardown_records],
    }
    if outcome.campaign_teardown:
        attempt.update(stage1=stage1)
        evidence = session_failure_evidence(outcome)
        _session_failure_notice(evidence)
        return _fail(
            attempt,
            stage="stage1",
            reason="campaign-teardown",
            exit_code=EXIT_RETRYABLE,
            detail=evidence,
        )
    if outcome.timed_out:
        attempt.update(stage1=stage1)
        return _fail(attempt, stage="stage1", reason="wall-exceeded", exit_code=EXIT_RETRYABLE)
    limit = structured_limit_signal(outcome.harness)
    if limit is not None:
        attempt.update(stage1=stage1)
        _fail(attempt, stage="stage1", reason="provider-usage-pause", exit_code=EXIT_BACKOFF)
        return EXIT_BACKOFF, _backoff_detail(limit, outcome.harness)
    if outcome.returncode != 0:
        attempt.update(stage1=stage1)
        evidence = session_failure_evidence(outcome)
        _session_failure_notice(evidence)
        return _fail(
            attempt,
            stage="stage1",
            reason="session-crashed",
            exit_code=EXIT_RETRYABLE,
            detail=evidence,
        )

    discovery_path = attempt.paths.directory / "discovery.json"
    authored_discovery = _read_json_file(discovery_path)
    if authored_discovery is None:
        attempt.update(stage1=stage1)
        if session_failure_class(outcome) == "content-refusal":
            # Same arm as stage 2's: a completed session that refused in
            # prose and published no discovery is a typed content refusal,
            # not a bare missing output.
            evidence = session_failure_evidence(outcome)
            _session_failure_notice(evidence)
            return _fail(
                attempt,
                stage="stage1",
                reason="content-refusal",
                exit_code=EXIT_RETRYABLE,
                detail=evidence,
            )
        return _fail(
            attempt, stage="stage1", reason="no-discovery-output", exit_code=EXIT_RETRYABLE
        )
    discovery_envelope = _discovery_envelope_from_author_payload(
        authored_discovery,
        request,
    )
    try:
        validate_source_discovery(
            discovery_envelope,
            stable_id=str(request.get("stable_id", "")),
            work_id=str(request.get("work_id", "")),
        )
    except DiscoveryError as exc:
        attempt.update(stage1=stage1)
        return _fail(
            attempt,
            stage="stage1",
            reason="discovery-contract-invalid",
            exit_code=EXIT_RETRYABLE,
            detail={"error": str(exc)},
        )
    _write_json_object(discovery_path, discovery_envelope)
    attempt.update(status="stage1-complete", stage1=stage1, discovery_path=str(discovery_path))
    _pause_hook(config, "stage1")
    return _complete_source_discovery(
        attempt,
        request=request,
        required_output=required_output,
        max_sources=max_sources,
        config=config,
    )


def _complete_source_discovery(
    attempt: AttemptHandle,
    *,
    request: Mapping[str, Any],
    required_output: Path,
    max_sources: int,
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Validate one recorded envelope and publish its governed machine product.

    Parameters
    ----------
    attempt:
        Attempt carrying the exact stage-1 discovery file.
    request:
        Machine source request supplying stable/work binding authority.
    required_output:
        Lane-visible publication path.
    max_sources:
        Fetch-target ceiling.
    config:
        Executor configuration supplying broker collaborators.

    Returns
    -------
    tuple[int, str]
        Operator exit and diagnostic.
    """

    raw = _read_json_file(attempt.paths.directory / "discovery.json")
    if raw is None:
        return _fail(
            attempt,
            stage="stage1",
            reason="no-discovery-output",
            exit_code=EXIT_RETRYABLE,
        )
    try:
        discovery = validate_source_discovery(
            raw,
            stable_id=str(request.get("stable_id", "")),
            work_id=str(request.get("work_id", "")),
        )
    except DiscoveryError as exc:
        return _fail(
            attempt,
            stage="stage1",
            reason="discovery-contract-invalid",
            exit_code=EXIT_RETRYABLE,
            detail={"error": str(exc)},
        )
    attempt.event("stage1-complete", arm=str(raw["arm"]))
    if isinstance(discovery, RetryableToolFailureDiscovery):
        return _fail(
            attempt,
            stage="stage1",
            reason="research-tools-unavailable",
            exit_code=EXIT_RETRYABLE,
            detail={
                "tool_name": discovery.tool_name,
                "tool_spelling": discovery.tool_spelling,
                "error": discovery.error,
            },
        )
    if isinstance(discovery, (NegativeDiscovery, HigherTierDiscovery)):
        local = attempt.paths.directory / "discovery.json"
        _publish(
            attempt,
            local,
            required_output,
            kind="source-discovery",
            author_model=config.author_model,
        )
        attempt.update(
            status="completed",
            outcome={"kind": "discovery-published", "arm": str(raw["arm"])},
        )
        return EXIT_OK, f"source discovery published for {request.get('stable_id')}"
    return _broker_and_publish(
        attempt,
        discovery=discovery,
        request=request,
        required_output=required_output,
        max_sources=max_sources,
        config=config,
    )


def _broker_and_publish(
    attempt: AttemptHandle,
    *,
    discovery: FoundDiscovery,
    request: Mapping[str, Any],
    required_output: Path,
    max_sources: int,
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Run the broker over a recorded FOUND discovery and publish the pack."""

    try:
        pack = broker_source_pack(
            [descriptor.to_mapping() for descriptor in discovery.descriptors],
            broker_dir=attempt.paths.broker,
            transport=default_transport(),
        )
    except SourceBrokerError as exc:
        return _fail(
            attempt,
            stage="broker",
            reason="descriptor-rejected",
            exit_code=EXIT_RETRYABLE,
            detail={"error": str(exc)},
        )
    write_broker_outputs(pack, attempt.paths.broker)
    attempt.update(
        broker={
            "targets": len(pack.outcomes),
            "total_bytes": pack.total_bytes,
            "outcomes": {item.source_id: item.outcome for item in pack.outcomes},
            "receipts_path": str(attempt.paths.broker / "receipts.json"),
        }
    )
    if pack.blocked_by_rate_limit():
        # The forge threw us out before it evaluated anything. Reporting this as
        # `primary-implementation-unfetchable` would say the implementation could
        # not be found, when nobody ever looked -- and it would send the retry to
        # repair a reference that was never wrong. Distinct reason so the driver
        # can pause on a sustained limit instead of grinding every remaining model
        # into the same wall for an hour.
        return _fail(
            attempt,
            stage="broker",
            reason="forge-rate-limited",
            exit_code=EXIT_RETRYABLE,
            detail={
                "outcomes": [item.to_dict() for item in pack.outcomes],
                "retry_after_seconds": pack.retry_after_seconds(),
                "rate_limit_reset_epoch": pack.rate_limit_reset_epoch(),
            },
        )
    if not pack.implementation_rows():
        signature = _deterministic_broker_signature(pack)
        prior = _prior_deterministic_broker_signature(attempt)
        if signature is not None and signature == prior:
            # AL-3: two consecutive attempts produced byte-identical
            # deterministic refusals. A third paid session would buy the same
            # bytes and the same refusal -- the 200a leg-27 chain retried
            # exactly this shape three times before exhausting the member and
            # feeding the no-publication streak. The permanent exit refuses
            # the next admission instead.
            return _fail(
                attempt,
                stage="broker",
                reason="primary-implementation-unfetchable-deterministic",
                exit_code=EXIT_PERMANENT,
                detail={
                    "outcomes": [item.to_dict() for item in pack.outcomes],
                    "deterministic_signature": signature,
                },
            )
        return _fail(
            attempt,
            stage="broker",
            reason="primary-implementation-unfetchable",
            exit_code=EXIT_RETRYABLE,
            detail={"outcomes": [item.to_dict() for item in pack.outcomes]},
        )
    rows = _capped_rows(pack, max_sources)
    local = attempt.paths.directory / "source-targets.json"
    payload = {
        **pack.to_dict(),
        "sources": rows,
        "discovery": discovery.raw_result,
        "discovery_sha256": stable_hash(discovery.raw_result),
    }
    local.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _pause_hook(config, "broker")
    _publish(
        attempt,
        local,
        required_output,
        kind="source-targets",
        author_model=config.author_model,
    )
    attempt.update(status="sources-published", outcome={"kind": "sources-published"})
    return EXIT_OK, f"source pack published for {request.get('stable_id')}"


def _broker_failure_signature_rows(
    outcomes: Sequence[Mapping[str, Any]],
) -> Optional[list[JsonObject]]:
    """Return the canonical failure rows of one all-deterministic broker pass.

    Parameters
    ----------
    outcomes:
        Per-target broker outcome dicts (live ``to_dict`` rows or the persisted
        ``receipts.json`` rows -- deliberately the same shape).

    Returns
    -------
    list[dict[str, Any]] | None
        Sorted signature rows when every failing target failed
        DETERMINISTICALLY (:data:`BROKER_DETERMINISTIC_OUTCOMES`), else
        ``None`` -- one transient failure (unreachable, rate-limited) means a
        retry could legitimately see different bytes.
    """

    non_failure = {OUTCOME_FETCHED, OUTCOME_PROBED, OUTCOME_PAPER_DERIVATION_ONLY}
    rows: list[JsonObject] = []
    for outcome in sorted(outcomes, key=lambda entry: str(entry.get("source_id"))):
        kind = str(outcome.get("outcome"))
        if kind in non_failure:
            continue
        if kind not in BROKER_DETERMINISTIC_OUTCOMES:
            return None
        receipt = outcome.get("resolver_receipt")
        rows.append(
            {
                "source_id": str(outcome.get("source_id")),
                "outcome": kind,
                "status": outcome.get("status"),
                "bytes_fetched": outcome.get("bytes_fetched"),
                "final_url": outcome.get("final_url"),
                "response_sha256": (
                    receipt.get("response_sha256") if isinstance(receipt, Mapping) else None
                ),
            }
        )
    return rows or None


def _deterministic_broker_signature(pack: BrokerPack) -> Optional[str]:
    """Return the pack's deterministic-failure fingerprint, if it has one."""

    rows = _broker_failure_signature_rows([item.to_dict() for item in pack.outcomes])
    return stable_hash(rows) if rows is not None else None


def _prior_deterministic_broker_signature(attempt: AttemptHandle) -> Optional[str]:
    """Recompute the immediately-prior attempt's deterministic broker fingerprint.

    Replay-derived on every call from the prior attempt's persisted broker
    ``receipts.json`` -- never from a stored signature a session could have
    influenced. Only the strictly consecutive prior attempt counts: an
    intervening attempt that failed differently (or got further) breaks the
    chain, exactly like the grant-loop refusal's last-two-attempts rule.

    Returns
    -------
    str | None
        The fingerprint, or ``None`` when there is no prior attempt, the prior
        attempt did not fail on an unfetchable primary implementation, or its
        receipts are unreadable (fail open to the RETRYABLE path: a missing
        receipt must never manufacture a permanent refusal).
    """

    author_root = attempt.paths.directory.parent.parent
    unfetchable = {
        "primary-implementation-unfetchable",
        "primary-implementation-unfetchable-deterministic",
    }
    for handle in reversed(list_attempts(author_root)):
        if handle.nonce == attempt.nonce:
            continue
        outcome = handle.record.get("outcome")
        if not isinstance(outcome, Mapping) or outcome.get("failure_reason") not in unfetchable:
            return None
        try:
            receipts = json.loads(
                (handle.paths.broker / "receipts.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        recorded = receipts.get("broker", {}).get("outcomes") if isinstance(receipts, Mapping) else None
        if not isinstance(recorded, list):
            return None
        rows = _broker_failure_signature_rows(
            [entry for entry in recorded if isinstance(entry, Mapping)]
        )
        return stable_hash(rows) if rows is not None else None
    return None


def _capped_rows(pack: BrokerPack, max_sources: int) -> list[JsonObject]:
    """Return implementation-first rows, capped at the lane's fetch grant."""

    implementation = pack.implementation_rows()
    rest = [row for row in pack.rows if row not in implementation]
    return (implementation + rest)[:max_sources]


# -- author (stage 2, resume, supplement) ----------------------------------


def _authored_excerpt_records(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return every authored excerpt-shaped record on one stage-2 payload.

    Parameters
    ----------
    payload:
        Authored stage-2 payload, before machine-owned fields are stamped on.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Declared evidence records plus the declared license record, if any.
    """

    records = payload.get("evidence_records")
    found = (
        [record for record in records if isinstance(record, Mapping)]
        if isinstance(records, list)
        else []
    )
    license_record = payload.get("license_record")
    if isinstance(license_record, Mapping):
        found.append(license_record)
    return tuple(found)


#: The closed ``proposal.author`` key set. The object takes no other key -- an
#: extra one such as ``actor`` is rejected by the proposal schema outright -- so
#: the stamp is an exact set comparison, never a merge.
_AUTHOR_BINDING_KEYS = frozenset({"provider", "model", "version", "prompt_sha256"})

#: Proposal keys the request envelope already fixes. Every one of them is a value
#: the executor is handed verbatim in ``expected_result``; asking the author to
#: transcribe it buys nothing and invites a plausible-looking wrong copy.
_PROPOSAL_BINDING_KEYS = (
    "campaign_id",
    "stable_id",
    "work_id",
    "intake_snapshot_id",
    "intake_snapshot_sha256",
    "intake_item_sha256",
    "source_manifest_identity",
    "dispatcher_identity",
)


def _required_proposal(value: object, field: str) -> Mapping[str, Any]:
    """Return one authored proposal object or fail typed.

    Parameters
    ----------
    value:
        Candidate authored proposal.
    field:
        Field name used in the refusal message.

    Returns
    -------
    Mapping[str, Any]
        The proposal object.

    Raises
    ------
    AuthorExecutorError
        If the value is not a JSON object.
    """

    if not isinstance(value, Mapping):
        raise AuthorExecutorError(f"stage-2 {field} must be an object")
    return value


def _refuse_conflicting_machine_owned(
    machine: Mapping[str, Any], supplied: Mapping[str, Any], *, prefix: str
) -> None:
    """Refuse an authored value that disagrees with the machine's own.

    Omission is free: a field the author was never able to observe is filled in
    below. What is refused is a CONFLICTING value, and the distinction is the
    whole point. Silently overwriting would erase the evidence that a wrong
    value was ever supplied -- exactly the failure the checker lane's
    ``apply_machine_owned_gate_fields`` was fixed for on 2026-07-29, where an
    unconditional overwrite let a proposal templated from a repository test
    fixture arrive carrying that fixture's placeholder identities and be
    laundered into a well-formed one.

    Parameters
    ----------
    machine, supplied:
        Machine-owned values and the authored object they are stamped onto.
    prefix:
        Dotted path prefix used in the refusal message.
    """

    for key, value in machine.items():
        if key in supplied and supplied[key] != value:
            raise AuthorExecutorError(
                f"stage-2 proposal supplies machine-owned {prefix}{key} as "
                f"{bounded_json_repr(supplied[key])}, but the machine holds "
                f"{bounded_json_repr(value)}"
            )


def _machine_proposal_id(stable_id: str, attempt_number: int, nonce: str) -> str:
    """Return the executor-owned immutable proposal identifier for one attempt.

    F51 DECIDED_PLAN item 13 (author R-4): ``proposal_id`` is a function of the
    attempt's own dispatch identity -- the stable id and the attempt nonce the
    executor already owns -- not the author's judgment. Requiring the author to
    transcribe it produced 40 of the ``result-contract-invalid`` rows on the
    pilot, each a schema ``'proposal_id' is a required property`` rejection that
    burned a 300 s repair round. The executor now stamps it, exactly as it
    already stamps ``author``, from the identity it holds. The spelling matches
    the value production authors were writing by hand.
    """

    return f"proposal-{stable_id}-attempt-{attempt_number:03d}-{nonce}"


def _stamp_machine_owned_proposal_fields(
    proposal: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    author_binding: Mapping[str, Any],
    proposal_id: str,
) -> JsonObject:
    """Stamp every proposal leaf the machine can derive from what it already holds.

    The author's authority is its JUDGMENT -- which rung, what the evidence
    says, which source, what the architecture is. It is not the machine's own
    request bindings, and it is not a digest: the ruling rule this lane
    converged on is *a locator the machine can verify by dereferencing may be
    model-supplied; an identity the machine can derive must be machine-derived.*
    A required field the machine could compute is both a fabrication invitation
    and a needless failure mode, and it had become the dominant one.

    Four classes move here, and only these four:

    * the eight request bindings plus ``schema_version``, all handed to the
      executor verbatim in ``expected_result``;
    * ``author``, the closed ``{provider, model, version, prompt_sha256}``
      object. It is a function of NOTHING the author knows -- four machine-held
      facts about the author's own dispatch -- so recomputing it catches
      nothing: a "lifted" value here would be the CORRECT value. Left authored,
      it was pure failure surface, and it fired: across ten published c1-mech
      proposals the author guessed ``version`` as the prompt FILE NAME
      (``claude_crawler_author_v2``) or as a model name (``claude-opus-5``),
      while the campaign holds ``current``. All ten refused at
      ``_validate_proposal_binding``'s ``stable_hash(author)`` comparison with
      ``model`` and ``prompt_sha256`` already correct. That comparison is
      unchanged and still runs; it is now satisfied by construction rather than
      by the author reproducing a fact it was never told. Stamped
      UNCONDITIONALLY, like ``proposal_sha256`` and unlike the request
      bindings: an author that supplies a value here is by definition guessing,
      so refusing the conflict would only rebuild the wall one layer up;
    * ``proposed_facts.modes.per_mode_run``, which names attempts that do not
      exist at proposal time, so the only honest value is the empty map.
      Its ownership ANNOTATION stays ``author-gated`` and that is correct:
      ``proposed_facts`` is wholly the author's claim surface by design --
      even ``evidence_identity`` and ``recipe_revision`` sit there
      author-gated, with the machine's copies at the top level -- and
      ``_AUTHOR_PROPOSAL_FACT_BLOCK_FIELDS`` encodes that invariant. The
      reducer-derived annotation on ``model-v3``'s ``$.modes.per_mode_run``
      describes the ACCEPTED RECORD, a different object; the two schemas do
      not disagree. Stamping an empty claim the author cannot make is a
      separate question from who the schema says owns the claim;
    * ``proposal_sha256``, the whole-object self-hash, derived last.

    What deliberately does NOT move, and why it matters:

    ``source_identity``, ``evidence_identity``, ``recipe_revision``,
    ``vet_identity``, and ``fidelity_identity`` all LOOK derivable -- each is a
    pure function of the declared facts -- and are not free to stamp.
    ``driver_admission`` recomputes all five and refuses a mismatch, which is
    the live catch for a proposal that lifted an identity from somewhere its
    own facts do not produce. Stamping them would make that comparison compare
    the machine against itself.

    They are the exact opposite case to ``author`` above, and the distinction is
    the whole design: ``author`` is a function of nothing the author knows, so
    recomputing it catches nothing and the machine must supply it; these five
    are functions of the author's OWN drafted facts, so recomputing them catches
    a real lie and the machine must NOT supply them. What the author was missing
    was never authority over the value -- it was the arithmetic. That is handed
    over as a calculator instead, ``menagerie.crawler.tools.author_identities``,
    which runs the very same :func:`recompute_accepted_identities` on facts the
    author supplies. The tripwire is untouched: identities still have to follow
    from the published facts, so a calculator fed fabricated facts returns an
    identity that the driver's recompute refuses just as loudly.

    ``proposed_facts.evidence.excerpts[].text_sha256`` looks like the purest
    case of all -- a self-hash of the sibling ``text``, and the terminal
    ``evidence_records`` channel does refuse it from an author outright. It is
    still NOT stamped here, because ``compute_evidence_identity`` projects it:
    filling it in for an author who omitted it silently changes the evidence
    identity the driver recomputes, so a proposal that was merely missing a
    digest would arrive as an identity mismatch instead. The two channels
    disagree for a reason, and closing that gap means moving
    ``evidence_identity`` too, not this leaf on its own.

    Parameters
    ----------
    proposal:
        Complete authored ``author-proposal.v3`` object.
    expected:
        Trusted ``expected_result`` bindings from the request envelope.
    author_binding:
        The exact four-key author preimage the envelope discloses, whose digest
        is ``expected_result.author_identity``.

    Returns
    -------
    dict[str, Any]
        Stamped proposal whose ``proposal_sha256`` binds the stamped body.

    Raises
    ------
    AuthorExecutorError
        If the proposal is not an object, or supplies a machine-owned value
        that conflicts with the one the machine holds.
    """

    if not isinstance(proposal, Mapping):
        raise AuthorExecutorError("stage-2 proposal must be one JSON object")
    stamped = deepcopy(dict(proposal))
    bindings: JsonObject = {
        "schema_version": AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
        **{key: expected[key] for key in _PROPOSAL_BINDING_KEYS if key in expected},
    }
    _refuse_conflicting_machine_owned(bindings, stamped, prefix="proposal.")
    stamped.update(bindings)

    # Unconditional overwrite, not a conflict check: see the ``author`` bullet
    # above. The executor must hold a real binding to stamp -- an absent or
    # malformed one is an engine fault, never a licence to leave the author's
    # guess in place.
    if not isinstance(author_binding, Mapping) or set(author_binding) != _AUTHOR_BINDING_KEYS:
        raise AuthorExecutorError(
            "author request lacks the closed identity_inputs.author binding "
            f"{sorted(_AUTHOR_BINDING_KEYS)!r}"
        )
    stamped["author"] = {key: author_binding[key] for key in sorted(author_binding)}

    # Item 13 (author R-4): proposal_id is the executor's, derived from the
    # attempt dispatch identity, and stamped unconditionally for the same
    # reason ``author`` is -- an authored value here is by definition a guess at
    # a machine fact, so overwriting it beats refusing the conflict.
    if not proposal_id:
        raise AuthorEngineFaultError("executor must supply a non-empty proposal_id to stamp")
    stamped["proposal_id"] = proposal_id

    facts = stamped.get("proposed_facts")
    if isinstance(facts, dict):
        _stamp_proposal_per_mode_run(facts)

    # Derived LAST and unconditionally, because it is the one field whose
    # authored value cannot be meaningfully compared: any legitimate machine
    # fill above invalidates it by construction, through no fault of the
    # author's judgment. Nothing is lost -- a self-hash only ever caught the
    # author's own arithmetic, and the driver still recomputes and compares it
    # after publication, which is where it can still catch a real edit.
    stamped.pop("proposal_sha256", None)
    stamped["proposal_sha256"] = stable_hash(stamped)
    return stamped


def _stamp_proposal_per_mode_run(facts: JsonObject) -> None:
    """Declare the empty proposal-time per-mode outcome map.

    Nothing has run when a proposal is authored, so the only honest value is
    the empty map. Filling it in is not permissiveness: a NON-empty authored
    value is a claim about attempts that do not exist, and it is refused here
    rather than carried forward as an ordinary fact.

    Parameters
    ----------
    facts:
        Mutable ``proposed_facts`` block.

    Raises
    ------
    AuthorExecutorError
        If the author claims per-mode run outcomes at proposal time.
    """

    modes = facts.get("modes")
    if not isinstance(modes, dict):
        return
    supplied = modes.get("per_mode_run")
    if isinstance(supplied, Mapping) and supplied:
        raise AuthorExecutorError(
            "stage-2 proposal claims modes.per_mode_run "
            f"{bounded_json_repr(dict(supplied))}; no attempt has run when a "
            "proposal is authored, so no per-mode outcome exists to report"
        )
    modes["per_mode_run"] = {}


def _author_result_from_author_payload(
    authored_result: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    proposal_id: Optional[str] = None,
) -> JsonObject:
    """Materialize a registered author result from the author's judgment.

    The author owns the result arm and its factual payload. The executor owns
    the request bindings, redundant discriminator, timestamps, and identities
    derived from the completed object.

    Parameters
    ----------
    authored_result:
        Exact ``kind`` plus arm-specific ``payload`` written by stage 2.
    request:
        Trusted author envelope supplying expected result bindings.

    Returns
    -------
    dict[str, Any]
        Complete schema-valid author-result envelope.

    Raises
    ------
    AuthorExecutorError
        If the authored shape is not exact or the materialized result fails
        the registered schema.
    """

    if set(authored_result) != {"kind", "payload"}:
        raise AuthorExecutorError("stage-2 output must contain exactly kind and payload")
    kind = authored_result.get("kind")
    authored_payload = authored_result.get("payload")
    if not isinstance(kind, str) or not isinstance(authored_payload, Mapping):
        raise AuthorExecutorError("stage-2 kind must be a string and payload must be an object")
    forbidden = {
        "arm",
        "evidence_identity",
        "license_identity",
        "recommendation_sha256",
        "search_report_identity",
    } & set(authored_payload)
    if forbidden:
        raise AuthorExecutorError(
            f"stage-2 payload carries machine-owned fields {sorted(forbidden)!r}"
        )
    # ``evidence_records``/``license_record`` are the declared home for excerpts the
    # author READ, so they are deliberately not forbidden here. What stays machine-owned
    # inside them is the digest: the author has no hashing primitive, is never asked for
    # one, and the machine recomputes it from the frozen bytes after dereference. Say so
    # legibly rather than letting it surface as an opaque additionalProperties rejection.
    for record in _authored_excerpt_records(authored_payload):
        owned = sorted(set(MACHINE_OWNED_EXCERPT_FIELDS) & set(record))
        if owned:
            raise AuthorExecutorError(
                f"stage-2 excerpt record carries machine-owned fields {owned!r}; quote the "
                "text and its locator, the executor derives every digest"
            )
    expected = request.get("expected_result")
    if not isinstance(expected, Mapping):
        raise AuthorExecutorError("author request lacks expected_result bindings")
    try:
        author_binding = author_identity_binding(request)
    except AuthorEngineFaultError as exc:
        # Re-typed at the executor boundary: this lane's callers branch on
        # AuthorExecutorError, and an undisclosed binding is exactly the typed
        # unavailability that class names.
        raise AuthorExecutorError(str(exc)) from exc

    # The executor-owned proposal identity. The caller with the live attempt
    # passes the attempt-scoped id (`_machine_proposal_id`); a caller without
    # one (a fixture round trip) gets a deterministic id derived from the
    # request bindings so the stamped proposal is still schema-valid.
    stamp_proposal_id = proposal_id or (
        f"proposal-{expected.get('stable_id', '')}-{expected.get('work_id', '')}"
    )

    payload: JsonObject = {"arm": kind, **deepcopy(dict(authored_payload))}
    if kind == "PROPOSED":
        payload["proposal"] = _stamp_machine_owned_proposal_fields(
            _required_proposal(payload.get("proposal"), "proposed payload proposal"),
            expected,
            author_binding=author_binding,
            proposal_id=stamp_proposal_id,
        )
    source_manifest = request.get("source_manifest")
    manifest_sources = (
        source_manifest.get("sources") if isinstance(source_manifest, Mapping) else None
    )
    manifest_source_ids = (
        [
            str(source["source_id"])
            for source in manifest_sources
            if isinstance(source, Mapping)
            and isinstance(source.get("source_id"), str)
            and source["source_id"]
        ]
        if isinstance(manifest_sources, list)
        else []
    )
    source_manifest_identity = str(expected.get("source_manifest_identity", ""))
    evidence_pack: JsonObject | None = None
    license_disposition: JsonObject | None = None
    if kind == "DEFER_RECOMMENDATION":
        handoff = payload.get("handoff_execution")
        if not isinstance(handoff, Mapping) or set(handoff) != {"proposal"}:
            raise AuthorExecutorError(
                "defer payload handoff_execution must contain exactly proposal"
            )
        proposal = _stamp_machine_owned_proposal_fields(
            _required_proposal(handoff.get("proposal"), "defer handoff proposal"),
            expected,
            author_binding=author_binding,
            proposal_id=stamp_proposal_id,
        )
        implementation = proposal.get("proposed_facts")
        implementation = (
            implementation.get("implementation") if isinstance(implementation, Mapping) else None
        )
        code_manifest = (
            implementation.get("code_manifest") if isinstance(implementation, Mapping) else None
        )
        if not isinstance(code_manifest, list):
            raise AuthorExecutorError("defer proposal must carry an implementation code_manifest")
        handoff_body: JsonObject = {
            "proposal": deepcopy(dict(proposal)),
            # Read off the STAMPED proposal, so the wrapper's copy and the
            # proposal's own digest cannot drift apart by construction.
            "proposal_sha256": str(proposal["proposal_sha256"]),
            "code_manifest_identity": stable_hash(code_manifest),
            "source_manifest_identity": str(expected.get("source_manifest_identity", "")),
        }
        handoff_body["handoff_sha256"] = stable_hash(handoff_body)
        payload["handoff_execution"] = handoff_body
        evidence_pack = derive_terminal_evidence_pack(
            source_ids=[str(value) for value in payload.get("source_ids", [])],
            evidence_ids=[str(value) for value in payload.get("evidence_ids", [])],
            predicate=f"needs-{payload.get('platform')}",
        )
        facts = proposal.get("proposed_facts")
        licenses = facts.get("licenses") if isinstance(facts, Mapping) else None
        if not isinstance(licenses, Mapping):
            raise AuthorExecutorError("defer proposal must carry exact license facts")
        license_disposition = derive_terminal_license_disposition(
            kind=kind,
            source_manifest_identity=source_manifest_identity,
            licenses=licenses,
        )
    elif kind == "SKIP_RECOMMENDATION":
        source_ids = [str(value) for value in payload.get("source_ids", [])]
        evidence_ids = [str(value) for value in payload.get("evidence_ids", [])]
        status_code = str(payload.get("status_code", ""))
        predicate = status_code.removeprefix("skipped:")
        evidence_pack = derive_terminal_evidence_pack(
            source_ids=source_ids,
            evidence_ids=evidence_ids,
            predicate=predicate,
        )
        license_disposition = derive_terminal_license_disposition(
            kind=kind,
            source_manifest_identity=source_manifest_identity,
        )
        payload["search_report_identity"] = stable_hash(
            {
                "status_code": status_code,
                "source_ids": source_ids,
                "evidence_ids": evidence_ids,
                "source_manifest_identity": source_manifest_identity,
            }
        )
    elif kind == "BLOCKED":
        evidence_ids = [str(value) for value in payload.get("evidence_ids", [])]
        evidence_pack = derive_terminal_evidence_pack(
            source_ids=manifest_source_ids,
            evidence_ids=evidence_ids,
            predicate="blocked-prerequisite",
        )
        license_disposition = derive_terminal_license_disposition(
            kind=kind,
            source_manifest_identity=source_manifest_identity,
        )
    if kind != "PROPOSED":
        if evidence_pack is None or license_disposition is None:
            raise AuthorExecutorError(f"unsupported stage-2 result kind {kind!r}")
        payload["evidence_identity"] = evidence_pack["evidence_identity"]
        payload["license_identity"] = stable_hash(license_disposition)
        payload["recommendation_sha256"] = stable_hash(payload)

    result_seed = {
        "kind": kind,
        "expected_result": deepcopy(dict(expected)),
        "payload": payload,
    }
    body: JsonObject = {
        **deepcopy(dict(expected)),
        "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
        "result_id": stable_hash(result_seed),
        "kind": kind,
        "created_at": utc_now(),
        "payload": payload,
    }
    body["result_sha256"] = stable_hash(body)
    try:
        validate_payload(body, AUTHOR_RESULT_SCHEMA_VERSION)
    except PayloadValidationError as exc:
        # Refusal authority is unchanged (validate_payload decides); the FULL
        # enumeration rides along so the one bounded in-session repair round can
        # fix every violation at once instead of surfacing them one death at a
        # time (rung-8 W-8).
        raise AuthorExecutorError(
            str(exc), errors=_enumerate_schema_violations(body) or [str(exc)]
        ) from exc
    return body


#: Reporting ceiling for one enumerated defect list. The repair brief must stay
#: readable; a result broken in fifty places is fed its first twenty-five plus a
#: truncation notice, not a wall of noise.
_MAX_ENUMERATED_ERRORS = 25


def _enumerate_schema_violations(body: Mapping[str, Any]) -> list[str]:
    """Enumerate EVERY registered-schema violation in one validator pass.

    This is a REPORTING channel, never a refusal authority:
    :func:`~menagerie.crawler.schema.validate_payload` still decides validity,
    and this enumeration exists so its refusal can be repaired in one round.
    ``validate_payload`` reports the single closest error by design; when it
    grows a full-enumeration mode of its own (schema-stream work), this helper
    is the seam to replace.

    Union arms are resolved the same way a reader would: for a ``oneOf`` the
    branch with the FEWEST violations is the branch the author was writing
    against, and its errors are the actionable ones -- the other branches fail
    wholesale on their ``arm`` const and would only bury the signal.

    Parameters
    ----------
    body:
        Complete candidate result envelope.

    Returns
    -------
    list[str]
        Deterministic ``$.path: message`` lines, deduplicated and bounded.
        Empty when enumeration itself is unavailable (the single
        ``validate_payload`` error then stands alone).
    """

    try:
        validator = get_validator(AUTHOR_RESULT_SCHEMA_VERSION)
        top = sorted(
            validator.iter_errors(dict(body)),
            key=lambda err: ([str(part) for part in err.absolute_path], err.message),
        )
    except Exception:  # pragma: no cover - enumeration is best-effort reporting
        return []
    lines: list[str] = []
    for error in top:
        lines.extend(_flatten_union_error(error, prefix=list(error.absolute_path)))
    deduplicated = list(dict.fromkeys(lines))
    if len(deduplicated) > _MAX_ENUMERATED_ERRORS:
        overflow = len(deduplicated) - _MAX_ENUMERATED_ERRORS
        deduplicated = deduplicated[:_MAX_ENUMERATED_ERRORS] + [
            f"... {overflow} further schema violations truncated"
        ]
    return deduplicated


def _flatten_union_error(error: Any, *, prefix: list[Any]) -> list[str]:
    """Flatten one ``jsonschema`` error tree into actionable per-leaf lines.

    Parameters
    ----------
    error:
        One ``ValidationError``, possibly carrying union ``context`` errors.
    prefix:
        The error's own instance path from the document root. ``context``
        errors chain their parents, so each sub-error's ``absolute_path`` is
        already root-anchored and replaces the prefix rather than extending it.

    Returns
    -------
    list[str]
        ``$.path: message`` lines for the most-matching union branch.
    """

    context = getattr(error, "context", None)
    if not context:
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in prefix
        )
        return [f"{path}: {error.message}"]
    branches: dict[str, list[Any]] = {}
    for sub in context:
        schema_path = list(sub.schema_path)
        key = str(schema_path[0]) if schema_path else "0"
        branches.setdefault(key, []).append(sub)

    def _branch_rank(item: tuple[str, list[Any]]) -> tuple[int, int, str]:
        key, subs = item
        # A branch whose discriminator (``arm``) const fails is the WRONG arm:
        # its remaining errors describe a payload the author never wrote, so it
        # is only ever selected when every branch mismatches the discriminator.
        wrong_arm = any(
            list(sub.absolute_path)[-1:] == ["arm"] and sub.validator == "const" for sub in subs
        )
        return (1 if wrong_arm else 0, len(subs), key)

    best = min(branches.items(), key=_branch_rank)[1]
    flattened: list[str] = []
    for sub in best:
        flattened.extend(_flatten_union_error(sub, prefix=list(sub.absolute_path)))
    return flattened


# -- pre-publication verification (rung-8 S2 + W-8) -------------------------
#
# Two measured failure classes converge here, both from the 2026-08-05 rung-8
# census. (1) Eight of twenty sessions published a typed BLOCKED/wall-exceeded
# with ``timed_out=false`` and 63-88% of their grant UNUSED -- an unverified
# self-report the machine then recorded as a trusted effort terminal. (2) Every
# malformed result burned a whole attempt per error, with two to four latent
# violations behind each visible one. The gate below refuses the first class
# against the executor's own clock and repairs the second in ONE bounded
# in-session round, inside the attempt's existing wall grant -- no budget
# constant moves.

#: A wall/effort exhaustion claim is recordable only when the executor's own
#: clock corroborates it: below this fraction of the grant the claim is refuted
#: by the machine's observation and refused (census S2, tripwire finding 2).
WALL_CLAIM_MIN_OBSERVED_FRACTION = 0.5

#: Floor under which an in-session continuation is not dispatched: a round with
#: less than this much wall cannot honestly repair or continue anything.
_CONTINUATION_MIN_WALL_SECONDS = 120.0

#: Bounded repair budget for one settle: materialization and deep-validation
#: rounds share it. The second round exists for the deep validator's
#: first-violation reporting (a round that fixed everything DISCLOSED to it may
#: legitimately surface a new defect class on re-validation -- rung 9b's
#: m5915 venue -> country and m9304 schema-key -> grounding-id sequences); a
#: round that returns with a disclosed defect unchanged ends the attempt
#: instead, so this budget can never fund flailing on the same complaint.
_MAX_VALIDATION_REPAIR_ROUNDS = 2


#: F51 DECIDED_PLAN item 8 (crawl-speed R4, author R-2): floor under the ONE
#: supplementary broker round. The round used to be granted a bare
#: ``config.wall_seconds() / 2`` -- a half-grant constant with no measured
#: basis that killed 276 of 513 production supplement rounds (54%) at ~990 s
#: while the successful rounds ran p50 594 s / p90 945 s, crowding the same
#: ceiling. The round now spends the attempt's REMAINING wall, floored here so
#: a supplement that follows a fast stage 2 still gets a full round, and capped
#: by the licensed invocation budget through :func:`_continuation_wall_seconds`
#: (``AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET``, raised 2.5 -> 3.0 in the same
#: change so the lane's stall bound still clears the worst case). The floor is
#: the historical half-grant of the default 30-minute grant, so no supplement
#: is ever funded BELOW what it used to get.
_SUPPLEMENT_ROUND_MIN_WALL_SECONDS = 900.0


#: AL-1: minimum funded wall for one pre-publication REPAIR round, drawn from
#: the invocation's already-licensed session budget (never a new budget: the
#: 2.5-grant ``AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET`` bound holds
#: exactly). 200a starved 38 of 67 pre-publication refusals at ``grant -
#: spent ~ 0`` -- 24 of the 113 failed members held a result one enumerated
#: defect from valid with no seconds left to fix it, while 1.5 grants of
#: licensed invocation headroom went unused. Calibrated 2026-08-24 from the
#: 264 recorded validation-repair rounds in the 200a records (durations: p50
#: 92s, p90 202s, p95 275s, p99 397s, max 544s): fundable at p95 per
#: B1-XV14's ruling, with margin below p99. Recalibrate by re-running the
#: duration scan over ``work/*/author/attempts/*/record.json`` continuations.
_REPAIR_ROUND_MIN_WALL_SECONDS = 300.0


def _continuation_wall_seconds(grant: float, spent: float, *, floor: float = 0.0) -> float:
    """Return the wall grant for one bounded in-session continuation round.

    A continuation spends the wall this attempt's own grant has left
    (``grant - spent``), raised to ``floor`` for funded REPAIR rounds (AL-1)
    -- and in every case capped by what the invocation's licensed session
    budget has left, so ``AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET`` stays
    exactly valid without any constant change: a repair round only ever
    consumes headroom the worst-case invocation (cold rerun + supplement) was
    already licensed to spend. With the default ``floor=0.0`` this reproduces
    the historical remaining-wall bound unchanged.

    Parameters
    ----------
    grant:
        The attempt's authoritative per-session wall grant.
    spent:
        Total session wall this invocation has already observed.
    floor:
        Minimum funded wall for this round kind (repair rounds pass
        :data:`_REPAIR_ROUND_MIN_WALL_SECONDS`; other continuations keep 0.0).

    Returns
    -------
    float
        Positive continuation grant, or ``0.0`` when neither the attempt's
        own wall nor the invocation's licensed budget can fund a useful round.
    """

    funded = max(float(grant) - float(spent), float(floor))
    budget_remaining = AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET * float(grant) - float(spent)
    wall = min(funded, budget_remaining)
    if wall < _CONTINUATION_MIN_WALL_SECONDS:
        return 0.0
    return wall


def _unverified_wall_claim(
    result: Mapping[str, Any], *, spent: float, grant: float
) -> Optional[JsonObject]:
    """Return the refusal record when a wall claim is refuted by the clock.

    The claim grammar is the lane's own (:func:`_names_effort_exhaustion`), so
    a creative spelling cannot dodge the check that its canonical spelling
    trips. The tripwire direction is fixed: an HONEST exhaustion claim (the
    session really consumed its grant) is never refused here -- it publishes
    and the driver routes it to the typed effort terminal exactly as before.

    Parameters
    ----------
    result:
        Materialized result envelope.
    spent:
        Executor-observed total session wall for this attempt.
    grant:
        The attempt's wall grant.

    Returns
    -------
    dict[str, Any] | None
        Machine-observed refusal evidence, or ``None`` when no wall claim is
        present or the claim is corroborated.
    """

    if result.get("kind") != "BLOCKED":
        return None
    payload = result.get("payload")
    reason = str(payload.get("reason_code", "")) if isinstance(payload, Mapping) else ""
    if not _names_effort_exhaustion(reason):
        return None
    if spent >= WALL_CLAIM_MIN_OBSERVED_FRACTION * grant:
        return None
    return {
        "reason_code": reason,
        "wall_seconds_observed": round(float(spent), 3),
        "wall_seconds_grant": round(float(grant), 3),
        "observed_fraction": round(float(spent) / float(grant), 3) if grant else 0.0,
    }


def _prepublication_validation_available(request: Mapping[str, Any]) -> bool:
    """Whether the request is a verified envelope the production parser accepts.

    The deep validator authenticates ``envelope_sha256`` before anything else,
    so running it against a transport stub would only manufacture noise. Every
    production envelope is self-hashed by ``build_author_envelope``; an
    unverifiable one downgrades the gate to publish-as-before, never to a
    refusal.
    """

    digest = request.get("envelope_sha256")
    if not isinstance(digest, str) or not digest:
        return False
    body = {key: value for key, value in request.items() if key != "envelope_sha256"}
    return stable_hash(body) == digest


def _supplement_grounding_rows(attempt: AttemptHandle) -> Optional[list[JsonObject]]:
    """Return the ONE granted supplementary pack's rows, grounded for reading.

    The broker pack rows carry ``expected_sha256`` (the broker's own digest of
    its fetch) but no CAS placement -- the driver promotes those bytes into the
    model's CAS only after the executor exits. For the pre-publication replay
    the same bytes are still sitting in the attempt's own broker evidence
    directory, so each row is pointed there through the in-process
    ``unpromoted_read_path`` channel (NOT ``cas_path``: recorded CAS paths are
    metadata the readers no longer dereference when a governing ``cas_root`` is
    supplied, while the unpromoted channel is honored precisely because it is
    constructed per-run and never persisted). This is fail-closed by
    construction: the evidence reader re-hashes whatever the path holds against
    ``content_sha256`` and refuses a mismatch, so a wrong or tampered blob can
    never ground an excerpt.

    Returns
    -------
    list[dict[str, Any]] | None
        Grounding-ready supplementary rows, or ``None`` when no supplement
        round was granted (the validator then falls back to the envelope's own
        manifest rows).
    """

    record = attempt.record.get("supplement")
    if not isinstance(record, Mapping):
        return None
    manifest_path = record.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        return None
    pack = _read_json_file(Path(manifest_path))
    rows = pack.get("sources") if isinstance(pack, Mapping) else None
    if not isinstance(rows, list):
        return None
    evidence_dir = attempt.paths.broker / "supplement" / "evidence"
    grounded: list[JsonObject] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        entry: JsonObject = dict(row)
        digest = str(entry.get("expected_sha256") or "")
        if digest.startswith("sha256:") and "content_sha256" not in entry:
            entry["content_sha256"] = digest
            candidate = evidence_dir / f"{digest.removeprefix('sha256:')[:16]}.bin"
            if candidate.is_file():
                entry["unpromoted_read_path"] = str(candidate)
        grounded.append(entry)
    return grounded


def _identity_mirror_errors(result: Mapping[str, Any], request: Mapping[str, Any]) -> list[str]:
    """Mirror the driver's staging identity checks over one drafted proposal.

    ``validate_author_result_mapping`` does not recompute the five accepted
    identities or the ``verified_hashes.source_manifest`` binding -- those
    refusals live in ``driver_admission._validate_artifact_identities`` at
    staging, after the result is published and the session is gone. Both are
    pure functions of what the executor already holds, so they are mirrored
    here as REPORTING for the one repair round. The driver's checks are
    untouched and stay authoritative; rung 8's m538 died exactly on the
    manifest binding, carrying a digest copied from a prior attempt whose
    sources had been re-frozen underneath it.
    """

    kind = result.get("kind")
    payload = result.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    if kind == "PROPOSED":
        proposal = payload.get("proposal")
    elif kind == "DEFER_RECOMMENDATION":
        handoff = payload.get("handoff_execution")
        proposal = handoff.get("proposal") if isinstance(handoff, Mapping) else None
    else:
        return []
    if not isinstance(proposal, Mapping):
        return []
    errors: list[str] = []
    verified = proposal.get("verified_hashes")
    if isinstance(verified, Mapping) and "source_manifest" in verified:
        expected = proposal.get("source_manifest_identity")
        if verified.get("source_manifest") != expected:
            errors.append(
                "verified_hashes.source_manifest does not bind the proposal source "
                f"manifest identity: declared {bounded_json_repr(verified.get('source_manifest'))}, "
                f"this envelope holds {bounded_json_repr(expected)} -- identities drafted "
                "in a prior attempt are STALE; recopy every machine value from THIS envelope"
            )
    facts = proposal.get("proposed_facts")
    if not isinstance(facts, Mapping):
        return errors
    try:
        checker = checker_identity_binding(request)
    except AuthorEngineFaultError:
        return errors
    inputs = request.get("identity_inputs")
    schema_version = MODEL_SCHEMA_VERSION_V3
    if isinstance(inputs, Mapping) and isinstance(inputs.get("model_schema_version"), str):
        schema_version = str(inputs["model_schema_version"])
    try:
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=checker["prompt_sha256"],
            checker_model=checker["model"],
            checker_version=checker["version"],
            schema_version=schema_version,
        )
    except MetadataValidationError as exc:
        errors.append(
            f"accepted identities cannot be derived from the declared proposed_facts: {exc}"
        )
        return errors
    claimed = {
        "source_identity": identities.source,
        "evidence_identity": identities.evidence,
        "recipe_revision": identities.recipe,
        "vet_identity": identities.vet,
        "fidelity_identity": identities.fidelity,
    }
    for field, value in claimed.items():
        if proposal.get(field) != value:
            errors.append(
                f"{field} does not follow from the declared proposed_facts: the proposal "
                f"carries {bounded_json_repr(proposal.get(field))}, the derivation yields "
                f"{bounded_json_repr(value)} -- rerun the identity calculator on the FINAL facts"
            )
    implementation = facts.get("implementation")
    if not isinstance(implementation, Mapping) or (
        implementation.get("recipe_revision") != identities.recipe
    ):
        errors.append(
            "embedded implementation.recipe_revision is stale for the derived recipe_revision"
        )
    evidence = facts.get("evidence")
    if not isinstance(evidence, Mapping) or (
        evidence.get("evidence_identity") != identities.evidence
    ):
        errors.append(
            "embedded evidence.evidence_identity is stale for the derived evidence_identity"
        )
    return errors


def _staged_model_listing(attempt: AttemptHandle) -> list[str]:
    """Return the attempt's staged ``model/`` tree as sorted relative paths.

    The listing rides the repair brief so a session reasoning about a staging
    or ``code_path`` defect sees exactly what IS staged, instead of guessing
    (rung-9b sessions, told a file could not be read, concluded the validator
    read a directory they could not write and "repaired" the path itself).
    """

    staged = attempt.paths.directory / "model"
    if not staged.is_dir():
        return []
    listing = sorted(str(path.relative_to(staged)) for path in staged.rglob("*") if path.is_file())
    return listing[:_MAX_ENUMERATED_ERRORS]


def _collect_prepublication_errors(
    materialized: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    attempt: AttemptHandle,
    author_root: Path,
) -> list[str]:
    """Replay the production result validator BEFORE publication.

    The replay must ground exactly like the driver's own post-publication
    validation, so the attempt's staged ``model/`` tree is mirrored into the
    envelope's ``allowed_model_dir`` first -- the same mirror publication
    performs -- and staged-code reads resolve against real bytes instead of a
    not-yet-populated directory (the rung-9b false-defect class). The driver
    re-runs the same validator after publication and stays the authority.
    NOTE (schema-stream seam): the mapping validator still raises on its FIRST
    deep violation; whatever list its exceptions learn to carry (an ``errors``
    attribute is consumed here already) will widen this enumeration without
    another executor change.

    Returns
    -------
    list[str]
        Deduplicated, bounded defect lines; empty when the result validates or
        when its only finding is a doctrine-routed effort claim.
    """

    model_dir = str(request.get("allowed_model_dir") or "")
    if model_dir:
        _mirror_staged_model(attempt, Path(model_dir), phase="prepublication")
    errors: list[str] = []
    try:
        _validate_author_result_mapping(
            materialized,
            request,
            cas_root=author_root / "source-cas",
            supplementary_sources=_supplement_grounding_rows(attempt),
        )
    except AuthorEffortExhaustionClaim:
        # An effort claim is doctrine-routed by the driver to a typed effort
        # terminal; whether it is HONEST is the wall gate's question, not a
        # content defect this round could repair.
        return []
    except AuthorDispatchError as exc:
        listed = getattr(exc, "errors", None)
        errors.extend(str(item) for item in (listed or [str(exc)]))
    except Exception as exc:
        # The gate is a REPORTING channel over a result the driver will
        # validate authoritatively anyway; an unexpected replay crash must
        # degrade to publish-as-before, never cost a publishable result. The
        # crash is still recorded, and the identity mirror below still runs.
        attempt.event(
            "prepublication-validation-unavailable",
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
    try:
        errors.extend(_identity_mirror_errors(materialized, request))
    except Exception as exc:  # pragma: no cover - same publish-preserving bar
        attempt.event(
            "prepublication-identity-mirror-unavailable",
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
    return list(dict.fromkeys(errors))[:_MAX_ENUMERATED_ERRORS]


def _render_validation_repair_brief(
    errors: Sequence[str],
    *,
    result_path: Path,
    wall_seconds: float,
    staged_files: Sequence[str] = (),
    enumeration_complete: bool = True,
    final_round: bool = False,
) -> str:
    """Render one bounded repair round's continuation brief.

    Two honesty rules learned from rung 9b, where 5 of 12 repair rounds
    "failed to land" and every one of those published a malformed terminal:

    * ``enumeration_complete`` distinguishes the schema path (every violation
      enumerated at once) from the deep-validation path, where the production
      validator stops at its FIRST violation -- the old brief claimed "the
      COMPLETE enumerated list" for both, so sessions that fixed everything
      disclosed were still burned by the next hidden error (m5915's venue ->
      country, m9304's schema key -> grounding id).
    * The brief states the true stakes: a persisted defect now refuses
      publication instead of silently publishing a result the driver is
      guaranteed to kill.
    """

    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(errors, start=1))
    if enumeration_complete:
        scope = [
            "The executor ran the production result validator over the result you wrote,",
            "BEFORE publication. It found the defects below -- the COMPLETE enumerated",
            "list as of this run, not merely the first error. Fix ALL of them in one",
            "pass by editing that result file in place, keeping the same inner",
            '`{"kind": ..., "payload": ...}` shape.',
        ]
    else:
        scope = [
            "The executor ran the production result validator over the result you wrote,",
            "BEFORE publication. It found the defects below. The deep validator stops",
            "at the FIRST violation it meets, so this list may not be exhaustive:",
            "fixing a listed defect can surface a deeper one on re-validation. Fix",
            "every listed defect AND re-check the rest of your result against the",
            "stage brief's rules in the same pass, editing the result file in place",
            'and keeping the same inner `{"kind": ..., "payload": ...}` shape.',
        ]
    if final_round:
        stakes = [
            "This is the FINAL repair round: any defect re-validation still finds",
            "ends this attempt. The executor will NOT publish an invalid result;",
            "the enumerated defects become the attempt's typed failure record.",
            "",
        ]
    else:
        stakes = [
            "If re-validation finds a defect disclosed here still present, this",
            "attempt ends immediately. The executor will NOT publish an invalid",
            "result; the enumerated defects become the attempt's typed failure",
            "record instead.",
            "",
        ]
    staged_section: list[str] = []
    if staged_files:
        staged_section = [
            "Your attempt's staged `model/` tree, exactly as the validator reads it",
            "(the executor mirrors it into the envelope's model directory before",
            "validating and again at publication -- NEVER write that directory",
            "yourself, and keep `code_path` repository-relative to this tree):",
            "",
            *(f"- `{name}`" for name in staged_files),
            "",
        ]
    return "\n".join(
        [
            "## PRE-PUBLICATION VALIDATION FAILED -- one bounded repair round",
            "",
            f"- RESULT output path, exact: `{result_path}`",
            "",
            *scope,
            "",
            numbered,
            "",
            *stakes,
            *staged_section,
            "Binding reminders:",
            "- Machine-owned fields stay omitted, exactly as the stage brief says.",
            "- Identities must follow from your corrected facts: after ANY fact edit,",
            "  re-run the identity calculator and recopy all five identities plus the",
            "  two embedded copies.",
            "- Identities and digests drafted in prior attempts are STALE; recopy every",
            "  machine value from THIS envelope and THIS manifest.",
            f"- Your wall for this round is {wall_seconds / 60:.1f} minutes.",
            "",
            "Then stop. Write nothing else.",
        ]
    )


def _render_wall_refusal_brief(
    claim: Mapping[str, Any], *, result_path: Path, wall_seconds: float
) -> str:
    """Render the refused-wall-claim continuation brief.

    The refusal is machine-built from the executor's own clock observations;
    no session text influences it.
    """

    observed = float(claim.get("wall_seconds_observed", 0.0))
    grant = float(claim.get("wall_seconds_grant", 0.0))
    unused = max(0.0, grant - observed)
    deadline = _deadline_iso(wall_seconds)
    return "\n".join(
        [
            "## WALL CLAIM REFUSED -- THE EXECUTOR'S CLOCK SAYS YOU HAVE TIME (binding)",
            "",
            f"- RESULT output path, exact: `{result_path}`",
            "",
            f"You published a `{claim.get('reason_code')}` claim, but the executor observed",
            f"only {observed:.0f}s of this attempt's {grant:.0f}s wall grant consumed --",
            f"about {unused / 60:.0f} minutes remain unused -- and no external timeout",
            "fired. A wall-exhaustion claim the machine's own clock refutes is refused,",
            "never recorded.",
            "",
            f"You now have {wall_seconds / 60:.0f} more minutes (deadline `{deadline}`).",
            "Observe the clock instead of guessing:",
            "",
            f"    {identity_tool_command()} --clock --deadline {deadline}",
            "",
            "Continue the authoring work NOW and overwrite your result file at",
            f"`{result_path}` with the real result: a complete proposal, or an honest",
            "terminal grounded in a real prerequisite. Publish `wall-exceeded` ONLY when",
            "a fresh clock observation shows the remaining budget genuinely cannot fit",
            "the remaining work.",
        ]
    )


def _stage2_continuation(
    attempt: AttemptHandle,
    config: ExecutorConfig,
    *,
    brief: str,
    wall_seconds: float,
    read_roots: Sequence[Path],
    resume: Optional[str],
    kind: str,
) -> SessionOutcome:
    """Run ONE bounded in-session continuation round and record it durably.

    This is the single spawn site for both continuation kinds, so the wall
    budget tripwire in ``test_author_wall_budget.py`` pins exactly one new
    session call whose grant is remaining-budget-bounded by
    :func:`_continuation_wall_seconds`.
    """

    attempt.event(
        "stage2-continuation-dispatched",
        kind=kind,
        wall_seconds=round(float(wall_seconds), 3),
    )
    continuation = run_claude_session(
        brief,
        cwd=attempt.paths.scratch,
        wall_seconds=wall_seconds,
        config=config,
        write_root=attempt.paths.directory,
        read_roots=read_roots,
        resume=resume,
    )
    rounds = list(attempt.record.get("continuations") or [])
    rounds.append(
        {
            "kind": kind,
            "completed_at": utc_now(),
            "effort": effort_from_session(continuation),
        }
    )
    attempt.update(continuations=rounds)
    return continuation


def _settle_result_prepublication(
    attempt: AttemptHandle,
    request: Mapping[str, Any],
    config: ExecutorConfig,
    *,
    author_root: Path,
    result_path: Path,
    read_roots: Sequence[Path],
    spent: float,
    resume_session: Optional[str],
) -> tuple[Optional[JsonObject], Optional[tuple[int, str]]]:
    """Settle one authored result through the bounded pre-publication gate.

    At most :data:`_MAX_VALIDATION_REPAIR_ROUNDS` repair rounds (materialization
    and deep validation share the budget) and one wall-claim refusal round run,
    each inside the attempt's own remaining wall. A wall claim the executor's
    clock refutes is refused outright -- recording it would launder an
    unverified self-report into a trusted terminal.

    A result the deep replay still refuses after its repair budget is NOT
    published (rung 9b: all five publish-with-residual results became
    ``failed:author/malformed-result`` terminals -- publishing a known-invalid
    result only spends a driver round-trip to reach the same truth). It becomes
    the typed ``result-contract-invalid`` failure with the full enumerated
    errors on the durable record, which the driver already classes as
    model-scoped authoring evidence with one bounded fresh-session retry, and
    which the next attempt's brief carries verbatim through
    ``prior_attempts``. The round-charging rule is disclosure-scoped: a round
    is charged only against defects DISCLOSED to it. A round that fixed
    everything it was told but surfaced NEW defects on re-validation landed --
    it earns the next bounded round (wall permitting); a round that returned
    with a disclosed defect unchanged failed -- the attempt ends immediately
    (a fresh session outperforms re-asking one that just demonstrated it
    cannot fix the complaint). Deep validation stays reporting-shaped in one
    direction only: when the replay itself is UNAVAILABLE (unverifiable
    envelope, replay crash) the gate degrades to publish-as-before and the
    driver's identical validation stays the sole authority.

    Returns
    -------
    tuple[dict[str, Any] | None, tuple[int, str] | None]
        The materialized result to publish, or the typed failure exit.
    """

    grant = config.wall_seconds()
    repair_rounds_used = 0
    disclosed_deep: frozenset[str] = frozenset()
    refusal_used = False
    # Item 13 (author R-4): the executor owns proposal_id, derived from this
    # attempt's own dispatch identity, so the author never has to transcribe it.
    proposal_id = _machine_proposal_id(
        str(request.get("stable_id", "")), attempt.number, attempt.nonce
    )

    def _materialize_current() -> tuple[Optional[JsonObject], Optional[tuple[str, list[str]]]]:
        parsed = _read_json_file(result_path)
        if parsed is None:
            return None, ("result-not-json", ["the result file is not one JSON document"])
        try:
            return _author_result_from_author_payload(parsed, request, proposal_id=proposal_id), None
        except AuthorExecutorError as exc:
            return None, ("result-contract-invalid", list(exc.errors))

    def _typed_failure(reason: str, errors: Sequence[str], **extra: Any) -> tuple[int, str]:
        detail: JsonObject = {"error": errors[0] if errors else reason, **extra}
        if len(errors) > 1:
            detail["errors"] = list(errors)
        return _fail(
            attempt,
            stage="stage2",
            reason=reason,
            exit_code=EXIT_RETRYABLE,
            detail=detail,
        )

    materialized, failure = _materialize_current()
    while True:
        if materialized is None:
            reason, errors = failure or ("result-contract-invalid", [])
            wall = (
                0.0
                if repair_rounds_used >= _MAX_VALIDATION_REPAIR_ROUNDS
                else _continuation_wall_seconds(
                    grant, spent, floor=_REPAIR_ROUND_MIN_WALL_SECONDS
                )
            )
            if wall <= 0.0:
                return None, _typed_failure(reason, errors)
            is_final_round = repair_rounds_used + 1 >= _MAX_VALIDATION_REPAIR_ROUNDS
            repair_rounds_used += 1
            continuation = _stage2_continuation(
                attempt,
                config,
                brief=_render_validation_repair_brief(
                    errors,
                    result_path=result_path,
                    wall_seconds=wall,
                    enumeration_complete=True,
                    final_round=is_final_round,
                ),
                wall_seconds=wall,
                read_roots=read_roots,
                resume=resume_session,
                kind="validation-repair",
            )
            spent += continuation.wall_seconds
            limit = structured_limit_signal(continuation.harness)
            if limit is not None:
                _fail(
                    attempt,
                    stage="stage2",
                    reason="provider-usage-pause",
                    exit_code=EXIT_BACKOFF,
                )
                return None, (EXIT_BACKOFF, _backoff_detail(limit, continuation.harness))
            if continuation.timed_out or continuation.returncode != 0:
                return None, _typed_failure(reason, errors, repair_round="session-failed")
            resume_session = continuation.session_id or resume_session
            materialized, failure = _materialize_current()
            if materialized is None:
                reason, errors = failure or ("result-contract-invalid", [])
                return None, _typed_failure(reason, errors, repair_round="did-not-repair")
            continue

        claim = _unverified_wall_claim(materialized, spent=spent, grant=grant)
        if claim is not None:
            wall = 0.0 if refusal_used else _continuation_wall_seconds(grant, spent)
            if wall <= 0.0:
                attempt.event("unverified-wall-claim-refused", **dict(claim))
                return None, _fail(
                    attempt,
                    stage="stage2",
                    reason="unverified-wall-claim",
                    exit_code=EXIT_RETRYABLE,
                    detail=dict(claim),
                )
            refusal_used = True
            continuation = _stage2_continuation(
                attempt,
                config,
                brief=_render_wall_refusal_brief(claim, result_path=result_path, wall_seconds=wall),
                wall_seconds=wall,
                read_roots=read_roots,
                resume=resume_session,
                kind="wall-claim-refusal",
            )
            spent += continuation.wall_seconds
            limit = structured_limit_signal(continuation.harness)
            if limit is not None:
                _fail(
                    attempt,
                    stage="stage2",
                    reason="provider-usage-pause",
                    exit_code=EXIT_BACKOFF,
                )
                return None, (EXIT_BACKOFF, _backoff_detail(limit, continuation.harness))
            if continuation.timed_out or continuation.returncode != 0:
                # The continuation itself consumed wall; the SAME claim may now
                # be corroborated (a round that ran to its own deadline spent
                # the time it was disputing). A timed-out session may still be
                # writing, so the file is NOT re-read.
                claim = _unverified_wall_claim(materialized, spent=spent, grant=grant)
                if claim is not None:
                    attempt.event("unverified-wall-claim-refused", **dict(claim))
                    return None, _fail(
                        attempt,
                        stage="stage2",
                        reason="unverified-wall-claim",
                        exit_code=EXIT_RETRYABLE,
                        detail={**claim, "refusal_round": "session-failed"},
                    )
                break
            resume_session = continuation.session_id or resume_session
            materialized, failure = _materialize_current()
            continue

        if not _prepublication_validation_available(request):
            break
        residual = _collect_prepublication_errors(
            materialized, request, attempt=attempt, author_root=author_root
        )
        if not residual:
            break
        persisted = sorted(disclosed_deep.intersection(residual))
        if persisted:
            # The last round returned with a defect it was explicitly told
            # about unchanged. Re-asking the same session the same thing is
            # flailing; the attempt fails typed and the NEXT attempt's brief
            # carries the full enumeration (rung-9b m11584/m5445/m8189 shape).
            attempt.event(
                "prepublication-validation-residual",
                count=len(residual),
                errors=residual[:5],
                disposition="publication-refused",
            )
            return None, _typed_failure(
                "result-contract-invalid",
                residual,
                repair_round="did-not-repair",
                persisted_errors=persisted[:5],
            )
        if repair_rounds_used >= _MAX_VALIDATION_REPAIR_ROUNDS:
            # Every disclosed defect was fixed, but the deep validator's
            # first-violation reporting surfaced yet another class and the
            # bounded budget is spent. Stop honestly: never publish invalid.
            attempt.event(
                "prepublication-validation-residual",
                count=len(residual),
                errors=residual[:5],
                disposition="publication-refused",
            )
            return None, _typed_failure(
                "result-contract-invalid", residual, repair_round="rounds-exhausted"
            )
        wall = _continuation_wall_seconds(grant, spent, floor=_REPAIR_ROUND_MIN_WALL_SECONDS)
        if wall <= 0.0:
            attempt.event(
                "prepublication-validation-unrepaired",
                count=len(residual),
                errors=residual[:5],
                disposition="publication-refused",
            )
            return None, _typed_failure(
                "result-contract-invalid", residual, repair_round="wall-exhausted"
            )
        is_final_round = repair_rounds_used + 1 >= _MAX_VALIDATION_REPAIR_ROUNDS
        repair_rounds_used += 1
        disclosed_deep = frozenset(residual)
        continuation = _stage2_continuation(
            attempt,
            config,
            brief=_render_validation_repair_brief(
                residual,
                result_path=result_path,
                wall_seconds=wall,
                staged_files=_staged_model_listing(attempt),
                enumeration_complete=False,
                final_round=is_final_round,
            ),
            wall_seconds=wall,
            read_roots=read_roots,
            resume=resume_session,
            kind="validation-repair",
        )
        spent += continuation.wall_seconds
        limit = structured_limit_signal(continuation.harness)
        if limit is not None:
            # The round may have finished its edit before the pause hit, and
            # re-validation is local and free: a now-clean result publishes.
            # A still-invalid one must NOT ride the old "publishable result
            # outranks the pause" arm -- publishing it would only buy a
            # guaranteed malformed-result terminal -- so the pause surfaces
            # as the same typed backoff the other continuation arms use.
            repaired, _repaired_failure = _materialize_current()
            if repaired is not None and not _collect_prepublication_errors(
                repaired, request, attempt=attempt, author_root=author_root
            ):
                materialized = repaired
                break
            _fail(
                attempt,
                stage="stage2",
                reason="provider-usage-pause",
                exit_code=EXIT_BACKOFF,
            )
            return None, (EXIT_BACKOFF, _backoff_detail(limit, continuation.harness))
        if continuation.timed_out or continuation.returncode != 0:
            # A timed-out session may still be writing; the file is NOT
            # re-read, and a known-invalid result is never published.
            return None, _typed_failure(
                "result-contract-invalid", residual, repair_round="session-failed"
            )
        resume_session = continuation.session_id or resume_session
        repaired, repaired_failure = _materialize_current()
        if repaired is None:
            broke_reason, broke_errors = repaired_failure or (
                "result-contract-invalid",
                [],
            )
            attempt.event(
                "prepublication-repair-broke-result",
                errors=broke_errors[:5],
            )
            return None, _typed_failure(
                broke_reason, broke_errors, repair_round="repair-broke-result"
            )
        materialized = repaired
        continue

    return materialized, None


def _record_author_model_attestation(
    attempt: AttemptHandle,
    request: Mapping[str, Any],
    config: ExecutorConfig,
) -> None:
    """Attest the launch configuration and cross-check the envelope's claim.

    The executor is the only layer that knows which ``--model`` actually runs,
    so the cross-check against the envelope's disclosed author identity lives
    here. A disagreement is a typed capture-defect event on the durable attempt
    record -- the 2026-08-05 configuration exactly (env launched fable, the
    campaign identity claimed sonnet, and 119 records inherited the claim). It
    is deliberately NOT a refusal: the session proceeds and the record layer
    resolves provenance with the ACTUAL model winning, so a config drift costs
    one typed event, never a silent lie and never a dead campaign.

    An unpinned launch (no ``--model``) runs the harness default, which nothing
    machine-side can name; that is its own typed event so it can never pose as
    an attested launch.
    """

    claimed: Optional[str] = None
    try:
        claimed = str(author_identity_binding(request)["model"])
    except AuthorEngineFaultError:
        # An undisclosed binding refuses later, at the result boundary that
        # owns that contract; the attestation records what it can.
        pass
    if config.author_model is None:
        attempt.event("author-model-unpinned", claimed_model=claimed)
        return
    if claimed is not None and claimed != config.author_model:
        attempt.event(
            "author-model-binding-mismatch",
            launched_model=config.author_model,
            claimed_model=claimed,
        )


def serve_author(
    request_path: Path,
    request: Mapping[str, Any],
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Serve one stage-2 authoring round on the retained stage-1 session."""

    author_root = request_path.parent
    stable_id = str(request.get("stable_id", ""))
    required_output = Path(str(request.get("required_output_path", "")))
    if not stable_id or not str(required_output):
        return EXIT_PERMANENT, "author envelope lacks stable_id or required_output_path"

    attempt, resume_session, cold_discovery = _stage2_attempt(author_root, stable_id, config)
    _record_author_model_attestation(attempt, request, config)
    brief = render_stage2_brief(
        request_path=request_path,
        request=request,
        attempt=attempt,
        config=config,
        author_root=author_root,
        cold_start_discovery=cold_discovery,
    )
    attempt.update(
        status="stage2-running",
        stage2={
            "started_at": utc_now(),
            "resumed_from": resume_session,
            "cold_start_reason": None if resume_session else "no-recoverable-stage1-session",
            # AL-22: the exact prompt-surface identity this stage was
            # steered by, durable per attempt.
            "steering_surface_digest": author_steering_surface_digest(
                config.campaign_id
            ),
        },
    )
    stage2_read_roots = _stage2_read_roots(author_root, request)
    outcome = run_claude_session(
        brief,
        cwd=attempt.paths.scratch,
        wall_seconds=config.wall_seconds(),
        config=config,
        write_root=attempt.paths.directory,
        read_roots=stage2_read_roots,
        resume=resume_session,
    )
    # Executor-observed session wall for this invocation: the machine-side
    # ledger the pre-publication gate verifies wall claims against and bounds
    # continuation rounds with. Never self-reported.
    session_wall_spent = outcome.wall_seconds
    if (
        resume_session is not None
        and outcome.returncode != 0
        and not outcome.timed_out
        and not outcome.campaign_teardown
    ):
        limit = structured_limit_signal(outcome.harness)
        if limit is None and not (attempt.paths.directory / "result.json").is_file():
            # Provider resume failure: record it and rerun once, cold, with the
            # recorded stage-1 discovery inlined (SEAM_REDESIGN 3.1 step 4).
            attempt.event(
                "resume-failed",
                returncode=outcome.returncode,
                resumed_from=resume_session,
                failure=session_failure_evidence(outcome),
            )
            stage2 = dict(attempt.record.get("stage2") or {})
            stage2["cold_start_reason"] = f"resume-exit-{outcome.returncode}"
            attempt.update(stage2=stage2)
            cold_brief = render_stage2_brief(
                request_path=request_path,
                request=request,
                attempt=attempt,
                config=config,
                author_root=author_root,
                cold_start_discovery=_recorded_discovery_text(attempt),
            )
            outcome = run_claude_session(
                cold_brief,
                cwd=attempt.paths.scratch,
                wall_seconds=config.wall_seconds(),
                config=config,
                write_root=attempt.paths.directory,
                read_roots=stage2_read_roots,
            )
            session_wall_spent += outcome.wall_seconds
    stage2 = dict(attempt.record.get("stage2") or {})
    stage2.update(
        {
            "session_id": outcome.session_id,
            "completed_at": utc_now(),
            "effort": effort_from_session(outcome),
            "argv": list(outcome.argv),
            # The spawn-resolved launch model, explicit rather than excavated
            # from argv: the durable half of the attestation the publication
            # receipt carries for the lane.
            "author_model": config.author_model,
            "stderr_tail": outcome.stderr_tail,
            "teardown_records": [dict(record) for record in outcome.teardown_records],
        }
    )
    attempt.update(stage2=stage2)
    if outcome.campaign_teardown:
        evidence = session_failure_evidence(outcome)
        _session_failure_notice(evidence)
        return _fail(
            attempt,
            stage="stage2",
            reason="campaign-teardown",
            exit_code=EXIT_RETRYABLE,
            detail=evidence,
        )
    if outcome.timed_out:
        # The provider session may still be live; the attempt is superseded so
        # any late output it writes is quarantined, never read as a result.
        attempt.event("stage2-timeout-superseding")
        attempt.mark_superseded(by_nonce="", reason="timeout")
        return (
            EXIT_RETRYABLE,
            f"author executor stage2 failed: wall-exceeded (attempt {attempt.nonce})",
        )
    limit = structured_limit_signal(outcome.harness)
    if limit is not None:
        _fail(attempt, stage="stage2", reason="provider-usage-pause", exit_code=EXIT_BACKOFF)
        return EXIT_BACKOFF, _backoff_detail(limit, outcome.harness)
    if outcome.returncode != 0:
        evidence = session_failure_evidence(outcome)
        _session_failure_notice(evidence)
        return _fail(
            attempt,
            stage="stage2",
            reason="session-crashed",
            exit_code=EXIT_RETRYABLE,
            detail=evidence,
        )
    _pause_hook(config, "stage2")

    result_path = attempt.paths.directory / "result.json"
    if not result_path.is_file():
        supplement_exit = _maybe_supplement_round(
            attempt, request_path, request, config, outcome, spent=session_wall_spent
        )
        if supplement_exit is not None:
            return supplement_exit
    if not result_path.is_file():
        if session_failure_class(outcome) == "content-refusal":
            # The session ran to completion, declined the work in prose, and
            # published nothing. `no-result-output` states less than we know:
            # the typed reason plus the structured notice make the refusal
            # durably distinguishable for the driver's canonical attempt row
            # (T4-T4) instead of leaving the prose only in gitignored scratch.
            evidence = session_failure_evidence(outcome)
            _session_failure_notice(evidence)
            return _fail(
                attempt,
                stage="stage2",
                reason="content-refusal",
                exit_code=EXIT_RETRYABLE,
                detail=evidence,
            )
        return _fail(attempt, stage="stage2", reason="no-result-output", exit_code=EXIT_RETRYABLE)
    resume_session_id = outcome.session_id
    supplement_record = attempt.record.get("supplement")
    if isinstance(supplement_record, Mapping):
        # The supplement round already ran inside this invocation: fold its
        # observed wall into the machine ledger, and resume the newest session.
        effort = supplement_record.get("effort")
        observed = effort.get("wall_seconds_observed") if isinstance(effort, Mapping) else None
        if isinstance(observed, (int, float)) and not isinstance(observed, bool):
            session_wall_spent += float(observed)
        if supplement_record.get("session_id"):
            resume_session_id = str(supplement_record["session_id"])
    materialized_result, typed_exit = _settle_result_prepublication(
        attempt,
        request,
        config,
        author_root=author_root,
        result_path=result_path,
        read_roots=stage2_read_roots,
        spent=session_wall_spent,
        resume_session=resume_session_id,
    )
    if typed_exit is not None:
        return typed_exit
    if materialized_result is None:  # pragma: no cover - structurally unreachable
        return _fail(
            attempt,
            stage="stage2",
            reason="result-contract-invalid",
            exit_code=EXIT_RETRYABLE,
        )
    _write_json_object(result_path, materialized_result)
    model_dir = str(request.get("allowed_model_dir") or "")
    if model_dir:
        _mirror_staged_model(attempt, Path(model_dir))
    _publish(
        attempt,
        result_path,
        required_output,
        kind="author-result",
        author_model=config.author_model,
    )
    attempt.update(status="completed", outcome={"kind": "result-published"})
    return EXIT_OK, f"author result published for {stable_id}"


def _stage2_attempt(
    author_root: Path,
    stable_id: str,
    config: ExecutorConfig,
) -> tuple[AttemptHandle, Optional[str], Optional[str]]:
    """Select or open the attempt stage 2 runs in.

    Returns
    -------
    tuple[AttemptHandle, str | None, str | None]
        The live attempt, the stage-1 session to resume (when recoverable),
        and the recorded discovery text for a cold start (when needed).
    """

    # ``stage1-complete`` is reachable when the executor died between
    # publishing the source pack and recording ``sources-published``; the lane
    # only dispatches the author round after consuming a published pack, so
    # both states mean "stage 1 is durable, resume its session in place".
    ready = latest_attempt(
        author_root,
        stable_id=stable_id,
        statuses=frozenset({"sources-published", "stage1-complete"}),
    )
    if ready is not None:
        return ready, _stage1_session(ready.record), None
    # No in-place attempt. A killed or timed-out stage 2 never reuses its
    # directory — the orphaned provider session may still write there — but
    # its recorded stage-1 session identity is still the cheapest context we
    # have, so the newest attempt that recorded one donates it. Only when no
    # attempt ever recorded a stage-1 session (provably unrecoverable) does
    # stage 2 run cold.
    donor: Optional[AttemptHandle] = None
    for handle in reversed(
        [
            candidate
            for candidate in list_attempts(author_root)
            if candidate.record.get("stable_id") == stable_id
        ]
    ):
        if _stage1_session(handle.record) is not None:
            donor = handle
            break
    fresh = new_attempt(
        author_root,
        stable_id=stable_id,
        campaign_id=config.campaign_id,
        kind="author",
        inherit_from=donor.record if donor is not None else None,
    )
    if donor is not None:
        return fresh, _stage1_session(donor.record), None
    return fresh, None, None


def _stage2_read_roots(author_root: Path, request: Mapping[str, Any]) -> list[Path]:
    """Return stage-2 readable roots: author root, prompt dir, and schemas.

    The envelope's ``allowed_model_dir`` is deliberately NOT writable by the
    session: staged code goes under the attempt's ``model/`` tree and the
    executor mirrors it into the envelope's model dir at publication, so the
    per-attempt directory stays the sole writable path. Schemas are readable so
    the author can follow the registered proposal topology instead of inventing
    a parallel result shape.
    """

    read_roots: list[Path] = [author_root, _SCHEMA_ROOT]
    prompt = request.get("prompt")
    if isinstance(prompt, Mapping) and prompt.get("path"):
        read_roots.append(Path(str(prompt["path"])).parent)
    return read_roots


def _stage1_session(record: Mapping[str, Any]) -> Optional[str]:
    """Return the recorded stage-1 session identity, walking inheritance."""

    stage1 = record.get("stage1")
    if isinstance(stage1, Mapping) and stage1.get("session_id"):
        return str(stage1["session_id"])
    inherited = record.get("inherited")
    if isinstance(inherited, Mapping):
        stage1 = inherited.get("stage1")
        if isinstance(stage1, Mapping) and stage1.get("session_id"):
            return str(stage1["session_id"])
    return None


def _recorded_discovery_text(attempt: AttemptHandle) -> Optional[str]:
    """Return the recorded discovery JSON text for cold-start inlining."""

    for candidate in (
        attempt.paths.directory / "discovery.json",
        Path(str((attempt.record.get("inherited") or {}).get("discovery_path") or "")),
    ):
        if candidate and candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except OSError:
                continue
    return None


def _supplement_request_from_author_payload(
    author_payload: Mapping[str, Any],
    request: Mapping[str, Any],
) -> JsonObject:
    """Wrap and validate one author-owned supplementary source request.

    Parameters
    ----------
    author_payload:
        Exact ``sources`` plus ``why`` object written by stage 2.
    request:
        Trusted author envelope supplying stable and work identities.

    Returns
    -------
    dict[str, Any]
        Machine-versioned supplement request whose descriptors passed the
        registered source-discovery schema.

    Raises
    ------
    AuthorExecutorError
        If the authored shape or any source descriptor is invalid.
    """

    if set(author_payload) != {"sources", "why"}:
        raise AuthorExecutorError("supplement request must contain exactly sources and why")
    sources = author_payload.get("sources")
    why = author_payload.get("why")
    if not isinstance(sources, list) or not sources:
        raise AuthorExecutorError("supplement sources must be a nonempty array")
    if not isinstance(why, str) or not why.strip():
        raise AuthorExecutorError("supplement why must be a nonempty string")
    discovery_envelope = _discovery_envelope_from_author_payload(
        {"arm": "FOUND", "sources": deepcopy(sources)},
        request,
    )
    try:
        discovery = validate_source_discovery(
            discovery_envelope,
            stable_id=str(request.get("stable_id", "")),
            work_id=str(request.get("work_id", "")),
        )
    except DiscoveryError as exc:
        raise AuthorExecutorError(str(exc)) from exc
    if not isinstance(discovery, FoundDiscovery):
        raise AuthorExecutorError("supplement sources did not produce a FOUND discovery")
    return {
        "supplement_version": SUPPLEMENT_VERSION,
        "sources": [descriptor.to_mapping() for descriptor in discovery.descriptors],
        "why": why,
    }


def _maybe_supplement_round(
    attempt: AttemptHandle,
    request_path: Path,
    request: Mapping[str, Any],
    config: ExecutorConfig,
    stage2_outcome: SessionOutcome,
    *,
    spent: float,
) -> Optional[tuple[int, str]]:
    """Grant at most one typed supplementary broker round, then resume once.

    Parameters
    ----------
    spent:
        Executor-observed session wall already consumed by this invocation, so
        the supplement round can spend the attempt's REMAINING wall (floored at
        :data:`_SUPPLEMENT_ROUND_MIN_WALL_SECONDS`) instead of a flat
        half-grant (F51 DECIDED_PLAN item 8; author R-2, crawl-speed R4).

    Returns
    -------
    tuple[int, str] | None
        A typed exit when the supplement round itself failed AND left no result
        on disk, else ``None`` (the caller re-checks for the result and settles
        it through the SAME pre-publication validator authority -- a wall kill
        or non-zero exit never discards a complete result the session already
        wrote, R-2b/2.8).
    """

    supplement_path = attempt.paths.directory / "supplement-request.json"
    scratch_supplement = attempt.paths.scratch / "supplement-request.json"
    if not supplement_path.is_file() and scratch_supplement.is_file():
        supplement_path = scratch_supplement
    if not supplement_path.is_file():
        return None
    if attempt.record.get("supplement") is not None:
        return _fail(
            attempt,
            stage="supplement",
            reason="supplement-round-already-consumed",
            exit_code=EXIT_RETRYABLE,
        )
    parsed = _read_json_file(supplement_path)
    if parsed is None:
        return _fail(
            attempt,
            stage="supplement",
            reason="supplement-request-untyped",
            exit_code=EXIT_RETRYABLE,
        )
    try:
        supplement = _supplement_request_from_author_payload(parsed, request)
    except AuthorExecutorError as exc:
        return _fail(
            attempt,
            stage="supplement",
            reason="supplement-request-untyped",
            exit_code=EXIT_RETRYABLE,
            detail={"error": str(exc)},
        )
    _write_json_object(supplement_path, supplement)
    attempt.update(status="supplement-running", supplement={"requested_at": utc_now()})
    try:
        pack = broker_source_pack(
            supplement["sources"],
            broker_dir=attempt.paths.broker / "supplement",
            transport=default_transport(),
        )
    except SourceBrokerError as exc:
        return _fail(
            attempt,
            stage="supplement",
            reason="descriptor-rejected",
            exit_code=EXIT_RETRYABLE,
            detail={"error": str(exc)},
        )
    write_broker_outputs(pack, attempt.paths.broker / "supplement")
    manifest_path = attempt.paths.directory / "supplement-manifest.json"
    manifest_path.write_text(
        json.dumps(pack.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    fetched_rows = len(pack.rows)
    if fetched_rows:
        outcome_lines = [
            "The broker fetched your supplementary request; per-target outcomes and",
            "receipts are in the manifest. Write the complete result now -- no further",
            "rounds are granted.",
        ]
    else:
        outcome_lines = [
            "The broker fetched NONE of your supplementary targets; the typed",
            "per-target refusals are in the manifest. Ground on the frozen manifest",
            "you already hold and write the complete result now -- no further rounds",
            "are granted.",
        ]
    continuation = "\n".join(
        [
            "## SUPPLEMENTARY SOURCE ROUND (the only one)",
            "",
            f"- supplementary manifest, exact: `{manifest_path}`",
            f"- RESULT output path, exact: `{attempt.paths.directory / 'result.json'}`",
            "",
            *outcome_lines,
        ]
    )
    supplement_wall = _continuation_wall_seconds(
        config.wall_seconds(), spent, floor=_SUPPLEMENT_ROUND_MIN_WALL_SECONDS
    )
    outcome = run_claude_session(
        continuation,
        cwd=attempt.paths.scratch,
        wall_seconds=supplement_wall,
        config=config,
        write_root=attempt.paths.directory,
        read_roots=_stage2_read_roots(request_path.parent, request),
        resume=stage2_outcome.session_id,
    )
    attempt.update(
        supplement={
            "requested_at": attempt.record["supplement"]["requested_at"],
            # The manifest is recorded for driver-side ingest ONLY when it
            # carries a fetched row. An all-failed round used to record its
            # empty pack anyway, and the driver's ingest tripwire then refused
            # the member's RESULT with `supplementary source pack carries no
            # fetched source` -- a machinery-owned death charged to the model
            # (m2045, 200a). The file itself stays on disk as evidence either
            # way, and the ingest check stays armed against untyped packs.
            "manifest_path": str(manifest_path) if fetched_rows else None,
            "fetched_source_rows": fetched_rows,
            "session_id": outcome.session_id,
            "effort": effort_from_session(outcome),
            "completed_at": utc_now(),
            "stderr_tail": outcome.stderr_tail,
            "teardown_records": [dict(record) for record in outcome.teardown_records],
        },
        status="stage2-running",
    )
    if outcome.timed_out or outcome.returncode != 0:
        # RESULT-ON-DISK WINS (R-2b/2.8): a wall kill or non-zero exit no
        # longer discards a complete result the session already wrote. The
        # author writes ``result.json`` atomically (tmp + rename), so a present
        # file is a whole JSON document, never a torn one; when it exists the
        # caller settles it through the identical pre-publication validator and
        # publishes or fails on ITS verdict -- nothing is loosened, the same
        # authority decides. Historically 24 of 293 production supplement wall
        # kills had a complete typed result on disk at kill time, and every one
        # was a silently lost publication (m8684: a 97 KB PROPOSED written 14 s
        # before SIGTERM). Only when NO result exists is the killed session the
        # terminal outcome.
        if (attempt.paths.directory / "result.json").is_file():
            return None
        evidence = session_failure_evidence(outcome)
        _session_failure_notice(evidence)
        return _fail(
            attempt,
            stage="supplement",
            reason="supplement-session-failed",
            exit_code=EXIT_RETRYABLE,
            detail=evidence,
        )
    return None


# -- capability probe ------------------------------------------------------


def serve_capability_probe(
    request_path: Path,
    request: Mapping[str, Any],
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Serve the doctor's live web-tools probe through the production path.

    The session researches and writes raw *evidence*; the executor turns it
    into the doctor-shaped *receipt* through
    :func:`menagerie.crawler.capability_probe.validate_capability_evidence` —
    the same proof the pool path applied — and publishes only a receipt that
    survived every check. The machine/model split is deliberate: the executor
    stamps only what it knows authoritatively (the top-level nonce it was
    handed, and ``completed_at`` from its own clock via ``now=``), while every
    ``exercised``/``receipt`` entry is *derived* from the session's validated
    per-tool evidence (nonce echo, live URLs, tool-shaped results, corroborated
    versions). The executor never asserts work it did not witness: evidence
    that fails validation publishes nothing, and the doctor's strict check
    fails — the correct outcome for an author path that cannot research.
    """

    nonce = str(request.get("nonce", ""))
    required_output = Path(str(request.get("required_output_path", "")))
    if not nonce or not str(required_output):
        return EXIT_PERMANENT, "capability probe lacks nonce or required_output_path"
    raw_requested = str(request.get("requested_at") or "").strip()
    try:
        requested_at = datetime.fromisoformat(
            raw_requested.removesuffix("Z") + "+00:00"
            if raw_requested.endswith("Z")
            else raw_requested
        )
    except ValueError:
        return EXIT_PERMANENT, "capability probe lacks a parseable requested_at"
    if requested_at.tzinfo is None:
        requested_at = requested_at.replace(tzinfo=timezone.utc)
    requested_at = requested_at.astimezone(timezone.utc)
    deadline = float(request.get("deadline_seconds", 120) or 120)
    challenge = derive_challenge(nonce)
    probe_root = request_path.parent / f"executor-probe-{nonce[:12]}"
    evidence_path = probe_root / "evidence.json"
    facts = _facts(
        [
            f"- probe nonce: `{nonce}`",
            f"- challenge package: `{challenge.package}`",
            f"- challenge metadata URL: `{challenge.metadata_url}`",
            f"- challenge project URL: `{challenge.project_url}`",
            f"- challenge_id: `{challenge.challenge_id}`",
            f"- REQUIRED output path, exact: `{evidence_path}`",
            f"- deadline: {deadline:g}s from dispatch",
        ]
    )
    brief = "\n\n---\n\n".join([facts, _read_prompt(_POOL_PROMPTS, "stage_capability_probe.md")])
    outcome = run_claude_session(
        brief,
        cwd=probe_root / "scratch",
        wall_seconds=deadline,
        config=config,
        write_root=probe_root,
        read_roots=[request_path.parent],
        session_id=str(uuid.uuid4()),
    )
    limit = structured_limit_signal(outcome.harness)
    if limit is not None:
        return EXIT_BACKOFF, _backoff_detail(limit, outcome.harness)
    if outcome.timed_out or outcome.returncode != 0:
        return EXIT_RETRYABLE, "capability probe session did not complete"
    if not evidence_path.is_file():
        return EXIT_RETRYABLE, "capability probe session published no evidence"
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return EXIT_RETRYABLE, f"capability probe evidence is unreadable: {exc}"
    if not isinstance(evidence, Mapping):
        return EXIT_RETRYABLE, "capability probe evidence is not an object"
    # The receipt is minted ONLY from evidence that survives every proof in
    # capability_probe: per-tool nonce echo, tool-shaped live results, digest
    # consistency, cross-tool version agreement, and the freshness window.
    # `now=` is the executor's own clock — the one fact the machine, not the
    # session, is the authority on — and becomes the receipt's completed_at.
    try:
        receipt = validate_capability_evidence(
            nonce=nonce,
            evidence=evidence,
            requested_at=requested_at,
            deadline_seconds=int(deadline),
            now=datetime.now(timezone.utc),
        )
    except CapabilityProbeError as exc:
        return EXIT_RETRYABLE, f"capability probe evidence is unproven: {exc}"
    receipt_path = probe_root / "receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    # The session writes only its probe root; the executor publishes the
    # validated receipt to the doctor's required path and prints the
    # nonce-bound publication receipt.
    _publish_bytes(
        receipt_path,
        required_output,
        nonce=nonce,
        kind="capability-probe",
        author_model=config.author_model,
    )
    return EXIT_OK, f"capability probe evidence proven for nonce {nonce[:12]}"


# -- CLI -------------------------------------------------------------------

_PARENT_DEADLINE_GUARD_MODE = "--parent-deadline-guard"
_PARENT_DEADLINE_GUARD_ENV = "MENAGERIE_PARENT_DEADLINE_GUARD"


def _write_guard_teardown_record(
    *,
    driver_shutdown_state: str,
    reason: str,
    teardown_trigger: str,
) -> None:
    """Durably name a guard-owned teardown before delivering SIGTERM.

    Parameters
    ----------
    driver_shutdown_state:
        Closed parent/guard state that caused teardown.
    reason:
        Machine-owned bounded explanation.
    teardown_trigger:
        Stable guard boundary name.
    """

    raw_path = os.environ.get(CAMPAIGN_TEARDOWN_RECORD_ENV)
    if not raw_path:
        return
    record = {
        "version": CAMPAIGN_TEARDOWN_RECORD_VERSION,
        "driver_shutdown_state": driver_shutdown_state,
        "target_pid": os.getpid(),
        "target_pgid": os.getpgrp(),
        "signal": signal.SIGTERM,
        "reason": reason[:512],
        "teardown_trigger": teardown_trigger[:128],
        "recorded_at": utc_now(),
        "outcome": "signal-pending",
    }
    try:
        _write_json_object(Path(raw_path), record)
    except OSError as exc:
        print(f"parent deadline guard could not persist teardown record: {exc}", file=sys.stderr)


def run_parent_deadline_guard(argv: Sequence[str]) -> int:
    """Run an operator command under a parent-owned death and deadline guard.

    Parameters
    ----------
    argv:
        ``parent_pid``, ``timeout_seconds``, then the exact non-shell operator
        command. These values are minted by the driver, never by the producer.

    Returns
    -------
    int
        The guarded operator's exit status. Parent death or deadline expiry
        SIGTERMs the verified guard group; the executor then tears down its
        separately isolated Claude group and records the session diagnostics.
    """

    if len(argv) < 3:
        print("parent deadline guard requires parent, timeout, and command", file=sys.stderr)
        return 64
    try:
        parent_pid = int(argv[0])
        timeout_seconds = float(argv[1])
    except ValueError:
        print("parent deadline guard received invalid parent or timeout", file=sys.stderr)
        return 64
    if parent_pid <= 1 or not timeout_seconds > 0.0:
        print("parent deadline guard requires positive trusted bounds", file=sys.stderr)
        return 64
    command = list(argv[2:])
    release_fd_raw = os.environ.get("MENAGERIE_PAID_RELEASE_FD")
    if release_fd_raw is not None:
        try:
            release_fd = int(release_fd_raw)
            released = os.read(release_fd, 1)
        except (OSError, TypeError, ValueError):
            released = b""
        if released != b"1":
            print("parent deadline guard was not durably released", file=sys.stderr)
            return 70
    child_environment = {
        **os.environ,
        _PARENT_DEADLINE_GUARD_ENV: "1",
    }
    try:
        process = subprocess.Popen(
            command,
            env=child_environment,
            start_new_session=False,
            # The custody flock descriptor, and ONLY it.  The whole paid subtree
            # inherits the one primitive whose release the driver can observe
            # for itself; there is deliberately no channel on which a wrapper
            # can assert anything about its own descendants.
            pass_fds=inherited_custody_fds(),
        )
    except OSError as exc:
        print(f"parent deadline guard could not start operator: {exc}", file=sys.stderr)
        return 127
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        parent_died = os.getppid() != parent_pid
        deadline_expired = time.monotonic() >= deadline
        if parent_died or deadline_expired:
            _write_guard_teardown_record(
                driver_shutdown_state=(
                    "driver-parent-death" if parent_died else "guard-deadline-expired"
                ),
                reason=(
                    "driver parent died while paid author guard was active"
                    if parent_died
                    else "paid author guard deadline expired"
                ),
                teardown_trigger=(
                    "driver-parent-death" if parent_died else "guard-deadline"
                ),
            )
            # Guard and executor share the driver-created paid group; Claude
            # leads a different verified group. SIGTERM lets the executor's
            # machine handler tear down that group and persist its attempt.
            if os.getpgrp() == os.getpid():
                signal_verified_process_group(
                    os.getpid(),
                    os.getpgrp(),
                    None,
                    sig=signal.SIGTERM,
                )
            # Unit-level/direct guard invocations do not own their ambient
            # process group.  Never signal a foreign group; bound the root-only
            # fallback and report the custody precondition failure as timeout.
            process.kill()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
            return 70
        time.sleep(0.02)
    return int(process.returncode or 0)


def classify_request(request: Mapping[str, Any]) -> str:
    """Return the request kind one envelope asks for.

    Mirrors the operator wrapper's closed classification.
    """

    if str(request.get("format", "")) == CAPABILITY_PROBE_FORMAT:
        return "capability-probe"
    version = str(request.get("envelope_version", ""))
    if version == _SOURCE_REQUEST_VERSION:
        return "source-request"
    if version.startswith(_AUTHOR_ENVELOPE_PREFIX):
        return "author"
    raise AuthorExecutorError(
        f"request is neither a capability probe nor an author envelope: "
        f"format={request.get('format')!r} envelope_version={version!r}"
    )


def _read_json_file(path: Path) -> Optional[JsonObject]:
    """Read one JSON object, returning ``None`` when absent or invalid."""

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def build_parser() -> argparse.ArgumentParser:
    """Build the executor CLI parser."""

    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.author_executor",
        description="Serve one author request headless via claude -p.",
    )
    parser.add_argument("--version", action="store_true", help="print the executor version")
    parser.add_argument("--campaign", default=None, help="campaign identity")
    parser.add_argument("request", nargs="?", default=None, help="absolute request path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the author executor.

    Returns
    -------
    int
        Operator protocol exit status.
    """

    arguments = list(argv) if argv is not None else list(sys.argv[1:])
    if arguments and arguments[0] == _PARENT_DEADLINE_GUARD_MODE:
        return run_parent_deadline_guard(arguments[1:])
    args = build_parser().parse_args(arguments)
    if args.version:
        print(EXECUTOR_VERSION)
        return EXIT_OK
    if args.request is None:
        print("author executor requires exactly one request path", file=sys.stderr)
        return EXIT_PERMANENT
    request_path = Path(args.request).expanduser().resolve()
    request = _read_json_file(request_path)
    if request is None:
        print(f"author request at {request_path} is not a JSON object", file=sys.stderr)
        return EXIT_PERMANENT
    try:
        config = ExecutorConfig.from_env(campaign_id=args.campaign or request.get("campaign_id"))
        kind = classify_request(request)
        with _campaign_teardown_signal_handler():
            if kind == "capability-probe":
                code, detail = serve_capability_probe(request_path, request, config)
            elif kind == "source-request":
                code, detail = serve_source_request(request_path, request, config)
            else:
                code, detail = serve_author(request_path, request, config)
    except AuthorExecutorError as exc:
        print(f"author executor error: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except OSError as exc:
        print(f"author executor io error: {exc}", file=sys.stderr)
        return EXIT_RETRYABLE
    if code != EXIT_OK:
        print(detail, file=sys.stderr)
    return code


def wrapper_command(*, campaign_id: Optional[str] = None) -> str:
    """Return the shell-quoted ``MENAGERIE_AUTHOR_COMMAND`` for this executor.

    Pair it with ``MENAGERIE_AUTHOR_REQUIRE_RECEIPT=1`` so the lane demands
    the executor's attempt-bound publication receipt on every round trip.
    """

    argv = [sys.executable, "-m", "menagerie.crawler.author_executor"]
    if campaign_id:
        argv.extend(["--campaign", campaign_id])
    return shlex.join(argv)


if __name__ == "__main__":  # pragma: no cover -- executor entry point
    raise SystemExit(main())
