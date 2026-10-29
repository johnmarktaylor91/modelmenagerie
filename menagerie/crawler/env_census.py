"""Closed census of every environment variable the crawler reads (W15/SD-4).

The wake/launch environment was a hand-me-down: 32 names were read as string
literals across the tree (plus a dozen more through module constants), only
six were persisted and restored through the campaign launch contract, and two
of the unpersisted ones could silently reshape a whole campaign --
``MENAGERIE_AUTHOR_CLAUDE_BIN`` (a bare ``claude`` resolved through an
unpersisted PATH turns every author attempt into infra deaths that read as a
storm) and ``EXA_API_KEY`` (absence at a wake became a phantom campaign-wide
pause, B3-XO-8). This module is the ONE classified vocabulary; the census
test walks the tree's actual ``os.environ``/``os.getenv`` reads and refuses
any name that is not classified here, so a new read cannot ship unclassified.

``WAKE_REQUIRED_SECRETS`` is the refuse-loudly arm: a production leg that
would need these refuses AT LEG OPEN, naming the exact variable, instead of
buying author sessions that die into a fake storm.
"""

from __future__ import annotations

import os

#: Names read as string LITERALS in menagerie/crawler (non-test) code, by
#: classification. The census test derives the observed set by AST and
#: compares; both directions of drift fail.
CLASSIFIED_ENVIRONMENT_READS: dict[str, str] = {
    # WS5 DX-70/UU-24: the in-worker numerics-context assert reads the OMP pin
    # the routed environment set for it (platform state, never config).
    "OMP_NUM_THREADS": "platform",
    # PD-8: the TCC launchd-probe payload records the launchd service marker
    # as context evidence in its receipt (platform state, never config).
    "XPC_SERVICE_NAME": "platform",
    # Campaign/launch configuration (the six launch-restored names live in
    # cli.LAUNCH_ENVIRONMENT_VARIABLES; the rest are read-through config).
    "MENAGERIE_ENVIRONMENT_STATE_ROOT": "campaign-config",
    # RC-1 crawl preflight: the lockless-intent exemptions (the retired
    # launch preflight's --allow-lockless spelled as a launch-environment
    # fact) and the environment lane's staging root, both read at invocation
    # open and recorded in the preflight arm census.
    "MENAGERIE_ALLOW_LOCKLESS": "campaign-config",
    "MENAGERIE_TMP_ROOT": "campaign-config",
    "MENAGERIE_AUTHOR_MODEL": "campaign-config",
    "MENAGERIE_PUBLIC_MIRROR": "campaign-config",
    "MENAGERIE_PRIVATE_MIRROR": "campaign-config",
    "MENAGERIE_ENVIRONMENT_COMMAND": "campaign-config",
    "MENAGERIE_AUTHOR_REQUIRE_RECEIPT": "campaign-config",
    "MENAGERIE_AUTHOR_QUEUE": "campaign-config",
    "MENAGERIE_AUTHOR_COMMAND": "campaign-config",
    "MENAGERIE_AUTHOR_CONCURRENCY": "campaign-config",
    "MENAGERIE_AUTHOR_POOL_OWNER": "campaign-config",
    "MENAGERIE_CHECKER_COMMAND": "campaign-config",
    "MENAGERIE_CHECKER_CONCURRENCY": "campaign-config",
    "MENAGERIE_SCHEDULER": "campaign-config",
    "MENAGERIE_CAMPAIGN_ID": "campaign-config",
    "MENAGERIE_PROVIDER_RECONCILE_COMMAND": "campaign-config",
    "MENAGERIE_NOTIFY_TRANSPORT": "campaign-config",
    # Crawler-internal runtime handoffs (parent -> child process contracts).
    "MENAGERIE_PARENT_DEADLINE_GUARD": "runtime-handoff",
    "MENAGERIE_PAID_RELEASE_FD": "runtime-handoff",
    "MENAGERIE_NOTIFICATION_RECEIPT_PATH": "runtime-handoff",
    "MENAGERIE_NOTIFICATION_IDEMPOTENCY_KEY": "runtime-handoff",
    "MENAGERIE_EXECUTOR_PAUSE_AFTER": "runtime-handoff",
    "MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION": "runtime-handoff",
    "MENAGERIE_CRAWLER_CHECKPOINT_CACHE_VERIFY": "runtime-handoff",
    "MENAGERIE_BROKER_FIXTURES": "test-fixture",
    # Release/CI gates.
    "MENAGERIE_RELEASE_GATE": "ci",
    "GITHUB_SHA": "ci",
    # RC-3 gate verdicts disclose (never key on) which real-environment
    # prefix family a gate run was armed against: the selected prefix, and
    # the ambient conda prefix the fixture falls back to when it is unset.
    "MENAGERIE_REAL_ENV_PREFIX": "ci",
    # Platform inheritance.
    "PATH": "platform",
    "CONDA_PREFIX": "platform",
    "LANG": "platform",
    "LC_ALL": "platform",
    # Secrets (never persisted; presence checked loudly, value never logged).
    "EXA_API_KEY": "secret",
}

#: Names read through module CONSTANTS (the census resolves constant-name
#: arguments to these and refuses an unresolvable or unclassified one).
CONSTANT_CARRIED_READS: dict[str, str] = {
    "MENAGERIE_POWER_WAKE_COMMAND": "campaign-config",
    "MENAGERIE_LAUNCHD_AGENT_ROOT": "campaign-config",
    # W32: the toolchain CLI pins, read through toolchain_pins.resolve_cli_pin
    # via the AUTHOR_CLI_PIN_ENV / CHECKER_CLI_PIN_ENV constants.
    "MENAGERIE_AUTHOR_CLAUDE_BIN": "campaign-config",
    "MENAGERIE_CHECKER_CODEX_BIN": "campaign-config",
    # Seat swap (2026-09-01): the Anthropic checker seat's Claude CLI pin, read
    # through toolchain_pins.CHECKER_CLAUDE_CLI_PIN_ENV (lane checker-anthropic).
    "MENAGERIE_CHECKER_CLAUDE_BIN": "campaign-config",
    # D-59: the machine-scoped scheduler ownership registry root override,
    # read through scheduler_ownership.SCHEDULER_OWNERSHIP_ROOT_ENV.
    "MENAGERIE_SCHEDULER_OWNERSHIP_ROOT": "campaign-config",
    "MENAGERIE_PAID_SUBTREE_LOCK_FD": "runtime-handoff",
    "MENAGERIE_WORKER_REQUEST_SHA256": "runtime-handoff",
    "MENAGERIE_EXECUTION_READ_MANIFEST_ID": "runtime-handoff",
    "MENAGERIE_PARENT_STANDARD_INPUT_ASSET": "runtime-handoff",
    "MENAGERIE_PARENT_ALLOWED_READ_PATHS": "runtime-handoff",
    "MENAGERIE_PARENT_COMPLETION_CHALLENGE": "runtime-handoff",
    "MENAGERIE_WORKER_GLOBAL_LOCK_FD": "runtime-handoff",
    "MENAGERIE_WORKER_LIFECYCLE_FD": "runtime-handoff",
    "MENAGERIE_WORKER_LOCK_FD": "runtime-handoff",
    "MENAGERIE_AUTHOR_WALL_SECONDS": "campaign-config",
    "MENAGERIE_CAMPAIGN_TEARDOWN_RECORD": "runtime-handoff",
    "MENAGERIE_CHECKER_CALIBRATION_RECEIPT": "campaign-config",
    "MENAGERIE_CHECKER_CALIBRATION_RUN": "campaign-config",
    # Seat swap (2026-09-01): the checker activation-matrix asset override, read
    # through checker_calibration.CHECKER_ACTIVATION_ENV; test rigs point it at a
    # missing file to exercise the receipt door.
    "MENAGERIE_CHECKER_ACTIVATION": "campaign-config",
    # W3 third mode: the shadow-rehearsal generation namespace, read through
    # checker_calibration.SHADOW_REHEARSAL_RUN_ENV (mode accessors, the env
    # snapshot attestation, and identity.stamp_shadow_rehearsal).
    "MENAGERIE_SHADOW_REHEARSAL_RUN": "campaign-config",
    "MENAGERIE_CHECKER_GOLDEN_CORPUS": "campaign-config",
    # Read through GITHUB_TOKEN_ENV_NAMES in source_broker (credential-mode
    # durability is DX-63/WS4 territory; census-classified only).
    "GH_TOKEN": "secret",
    "GITHUB_TOKEN": "secret",
}

#: Names the campaign-restore path WRITES into ``os.environ`` beyond the six
#: launch-restored ``cli.LAUNCH_ENVIRONMENT_VARIABLES``:
#: ``_restore_campaign_arguments`` exports the two mirror roots persisted-wins
#: ("the campaign's identity, not a tunable"), and consumers such as PD-8's
#: state-placement preflight arm read them AMBIENTLY by design. Every name a
#: production restore writes must appear here or in the launch vocabulary --
#: the write census (test_env_read_census) refuses any other production
#: ``os.environ`` write, and the test-session isolation fixture
#: (tests/conftest) snapshots exactly this union so a test driving the real
#: CLI factory cannot leak restore writes into later tests (the 200b presweep
#: gate lost nine preflight tests to a leaked ``/mirrors/private``).
CAMPAIGN_IDENTITY_ENVIRONMENT_WRITES: tuple[str, ...] = (
    "MENAGERIE_PUBLIC_MIRROR",
    "MENAGERIE_PRIVATE_MIRROR",
)

#: Secrets whose ABSENCE at a production leg open refuses loudly instead of
#: becoming a phantom campaign pause (B3-XO-8). GH_TOKEN/GITHUB_TOKEN are
#: excluded deliberately: an anonymous GitHub fallback exists and its
#: credential-mode durability is DX-63 (WS4).
WAKE_REQUIRED_SECRETS: tuple[str, ...] = ("EXA_API_KEY",)


def missing_wake_secrets() -> tuple[str, ...]:
    """Return every wake-required secret absent (or blank) in this process."""

    return tuple(
        name for name in WAKE_REQUIRED_SECRETS if not os.environ.get(name, "").strip()
    )
