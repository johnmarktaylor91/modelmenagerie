"""W32 (+OC-16, PM-13, AL-21): the toolchain pin authority.

The campaign's session economics ride two external CLIs (the Claude author
harness, the codex checker) and one tier->author-model table. 200a resolved
the author binary by bare name, had no codex pin at all, verified nothing per
leg before money moved, and had no machinery making a pin bump or a model-
tier change (F-O23 rider: a mid-rung Opus fallback changes session-shape
economics) a REGISTERED event.

This module is the ONE owner of toolchain-pin resolution and verification:

- :func:`resolve_cli_pin` — a declared pin must be an ABSOLUTE path to an
  existing file (venv-symlink semantics via ``normalize_executable``); a bare
  name refuses typed. An UNDECLARED pin resolves to ``None``: the launch
  preflight's DX-62 cli-versions arm refuses pinless production launches, so
  absence is a launch-blocked disclosure, not a silent fallback.
- :func:`require_verified_toolchain` — the cached per-leg verification the
  paid admission sites consult BEFORE any money row is minted. One
  ``--version`` probe per (leg cache key, pin, expected version); drift
  refuses typed.
- :func:`toolchain_identity` — the ``toolchain_identity`` rotation-axis
  value: a content hash over {CLI versions, tier->author-model table}, so a
  pin bump or tier change is a registerable rotation, never silent drift.
- :func:`verify_toolchain_attestation` — the committed attestation binds
  {CLI versions, the pinned Claude version's captured harness-envelope
  fixture, the per-tier derived storm fuse tables}. Staging re-derives and
  refuses a mismatch (:func:`menagerie.crawler.rung_staging.
  assert_toolchain_attestation_current`), so a pin bump without storm-
  envelope re-attestation, or a tier change without fuse re-derivation,
  cannot stage a rung.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    CHECKER_CLAUDE_CLI_VERSION,
    CLAUDE_CLI_VERSION,
    CODEX_CLI_VERSION,
    TIER_CAMPAIGN_AUTHOR_MODELS,
)
from menagerie.crawler.executable_paths import normalize_executable
from menagerie.crawler.identity import stable_hash

__all__ = [
    "AUTHOR_CLI_PIN_ENV",
    "CHECKER_CLAUDE_CLI_PIN_ENV",
    "CHECKER_CLI_PIN_ENV",
    "CHECKER_PROVIDER_TOOLCHAIN_LANES",
    "TOOLCHAIN_ATTESTATION_PATH",
    "TOOLCHAIN_ATTESTATION_SCHEMA_VERSION",
    "ToolchainPinError",
    "clear_toolchain_verification_cache",
    "expected_cli_version",
    "require_verified_toolchain",
    "checker_toolchain_lane",
    "resolve_cli_pin",
    "toolchain_identity",
    "verify_cli_version",
    "verify_toolchain_attestation",
]

AUTHOR_CLI_PIN_ENV = "MENAGERIE_AUTHOR_CLAUDE_BIN"
CHECKER_CLI_PIN_ENV = "MENAGERIE_CHECKER_CODEX_BIN"
#: Seat swap (2026-09-01): the Anthropic checker seat's Claude CLI pin. Its own
#: lane (``checker-anthropic``) rather than the author pin, because the checker
#: seat needs a release the author harness pin does not (>= 2.1.251 for 5.1).
CHECKER_CLAUDE_CLI_PIN_ENV = "MENAGERIE_CHECKER_CLAUDE_BIN"

TOOLCHAIN_ATTESTATION_SCHEMA_VERSION = "menagerie.crawler.toolchain-attestation.v1"
TOOLCHAIN_ATTESTATION_PATH = Path(__file__).with_name("toolchain_attestation.json")

_LANE_PINS: Mapping[str, tuple[str, str]] = {
    "author": (AUTHOR_CLI_PIN_ENV, CLAUDE_CLI_VERSION),
    "checker": (CHECKER_CLI_PIN_ENV, CODEX_CLI_VERSION),
    "checker-anthropic": (CHECKER_CLAUDE_CLI_PIN_ENV, CHECKER_CLAUDE_CLI_VERSION),
}

#: The toolchain lane that verifies one checker provider's CLI pin before a
#: paid checker reserve (``CHECKER_PROVIDER_TOOLCHAIN_LANES[provider]``).
CHECKER_PROVIDER_TOOLCHAIN_LANES: Mapping[str, str] = {
    "openai": "checker",
    "anthropic": "checker-anthropic",
}


def checker_toolchain_lane(provider: str) -> str:
    """Return the toolchain lane owning one checker provider's CLI pin."""

    try:
        return CHECKER_PROVIDER_TOOLCHAIN_LANES[provider]
    except KeyError:
        raise ToolchainPinError(f"no checker toolchain lane for provider {provider!r}") from None

_VERSION_RE = re.compile(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)")

#: Per-process verification memo: (cache_key, command, expected) -> report.
#: The driver process lives the whole leg, so keying on the leg's
#: (campaign_id, generation_id) makes this exactly the per-leg cache.
_VERIFICATION_CACHE: dict[tuple, dict] = {}


class ToolchainPinError(RuntimeError):
    """Raised on pin-resolution, verification, or attestation refusals."""


def clear_toolchain_verification_cache() -> None:
    """Drop every cached verification (tests, explicit re-probe)."""

    _VERIFICATION_CACHE.clear()


def resolve_cli_pin(
    env_name: str, *, env: Optional[Mapping[str, str]] = None
) -> Optional[tuple[str, ...]]:
    """Resolve one declared CLI pin to an absolute-path command, or ``None``.

    Raises
    ------
    ToolchainPinError
        If the pin is declared but its command head is not an absolute path
        to an existing file. A bare name is exactly the unpinned hole W32
        closes — it refuses instead of resolving through ``$PATH``.
    """

    raw = (env if env is not None else os.environ).get(env_name)
    if raw is None or not raw.strip():
        return None
    command = tuple(shlex.split(raw))
    head = command[0]
    if not os.path.isabs(head):
        raise ToolchainPinError(
            f"{env_name} must pin an absolute executable path (W32); "
            f"got the bare/relative name {head!r}"
        )
    normalized = normalize_executable(head)
    if normalized is None:
        raise ToolchainPinError(f"{env_name} pins {head!r}, which does not exist")
    return (str(normalized), *command[1:])


def expected_cli_version(lane: str) -> str:
    """Return the reviewed pinned version for one paid lane."""

    try:
        return _LANE_PINS[lane][1]
    except KeyError:
        raise ToolchainPinError(f"unknown toolchain lane {lane!r}") from None


def verify_cli_version(
    command: Sequence[str],
    expected_version: str,
    *,
    runner=subprocess.run,
) -> str:
    """Probe ``<command> --version`` and fail closed off the pinned release.

    Returns
    -------
    str
        The observed semver.

    Raises
    ------
    ToolchainPinError
        If the probe cannot run, reports nothing parseable, or reports any
        version other than the pinned one.
    """

    try:
        completed = runner(
            (*command, "--version"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolchainPinError(
            f"unable to verify pinned CLI {command[0]!r}: {exc}"
        ) from exc
    observed = _VERSION_RE.search(completed.stdout or "")
    if completed.returncode != 0 or observed is None or observed.group(1) != expected_version:
        raise ToolchainPinError(
            f"CLI version mismatch for {command[0]!r}: required {expected_version}, "
            f"observed {observed.group(1) if observed is not None else 'unparseable'}"
        )
    return observed.group(1)


def require_verified_toolchain(
    lane: str,
    *,
    cache_key: tuple,
    env: Optional[Mapping[str, str]] = None,
    runner=subprocess.run,
) -> dict:
    """Verify one lane's declared CLI pin, cached per leg, before money moves.

    An UNDECLARED pin returns a disclosed ``{"pinned": False}`` report — the
    launch preflight owns refusing pinless production launches; this site
    owns catching drift between leg open and every subsequent buy.

    Raises
    ------
    ToolchainPinError
        On a declared pin that is non-absolute, missing, or off-version.
    """

    command = resolve_cli_pin(_LANE_PINS[lane][0], env=env) if lane in _LANE_PINS else None
    if lane not in _LANE_PINS:
        raise ToolchainPinError(f"unknown toolchain lane {lane!r}")
    if command is None:
        return {"lane": lane, "pinned": False}
    expected = expected_cli_version(lane)
    memo_key = (cache_key, command, expected)
    cached = _VERIFICATION_CACHE.get(memo_key)
    if cached is not None:
        return {**cached, "cached": True}
    observed = verify_cli_version(command, expected, runner=runner)
    report = {
        "lane": lane,
        "pinned": True,
        "command": list(command),
        "observed_version": observed,
        "cached": False,
    }
    _VERIFICATION_CACHE[memo_key] = report
    return dict(report)


def toolchain_identity() -> str:
    """Return the ``toolchain_identity`` rotation-axis value.

    A content hash over the reviewed CLI versions and the tier->author-model
    table: bumping a pin or changing a tier's model MOVES this identity, so
    the rotation census reads the change as a stale head until a rotation
    registration row is appended — the same discipline every other identity
    axis rides.
    """

    # Seat swap (2026-09-01): the checker seat table and its Claude CLI pin
    # join the preimage, so this axis is the CARRIER of the `checker_model`
    # projection until RC-2's `rotations.jsonl` names checker_model as its own
    # measured axis. The swap is therefore one registered refinement row here
    # (measured by the bench's golden replay, SEAT_RULING.md), never a silent
    # drift. `tier_checker_models` is the per-campaign live binding; the seat
    # table is the per-kind primary/confirmation pair the wrapper enforces.
    from menagerie.crawler.constants import CHECKER_SEATS
    from menagerie.crawler.partitioner import CAMPAIGN_SPECS

    return stable_hash(
        {
            "claude_cli_version": CLAUDE_CLI_VERSION,
            "codex_cli_version": CODEX_CLI_VERSION,
            "checker_claude_cli_version": CHECKER_CLAUDE_CLI_VERSION,
            "tier_author_models": dict(TIER_CAMPAIGN_AUTHOR_MODELS),
            "tier_checker_models": {
                spec.campaign_id: spec.checker_model for spec in CAMPAIGN_SPECS
            },
            "checker_seats": {
                f"{kind.value}/{role.value}": {
                    "model": seat.model,
                    "provider": seat.provider,
                    "confirmation_sample_rate": seat.confirmation_sample_rate,
                }
                for (kind, role), seat in sorted(
                    CHECKER_SEATS.items(), key=lambda entry: (entry[0][0].value, entry[0][1].value)
                )
            },
        }
    )


def _envelope_fixture_path() -> Path:
    from menagerie.crawler.tools.author_stream_probe import fixture_path

    return fixture_path(CLAUDE_CLI_VERSION)


def verify_toolchain_attestation() -> list[str]:
    """Re-derive the toolchain attestation and return every mismatch.

    Returns
    -------
    list[str]
        Empty when the committed attestation matches this tree: CLI version
        pins, the captured harness-envelope fixture for the pinned Claude
        release, and every tier's re-derived storm fuse table digest.
    """

    from menagerie.crawler.storm_fuse import derive_fuse_table

    mismatches: list[str] = []
    try:
        payload = json.loads(TOOLCHAIN_ATTESTATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"toolchain attestation unreadable: {exc}"]
    if payload.get("schema_version") != TOOLCHAIN_ATTESTATION_SCHEMA_VERSION:
        mismatches.append(
            f"attestation schema {payload.get('schema_version')!r} is not "
            f"{TOOLCHAIN_ATTESTATION_SCHEMA_VERSION}"
        )
    if payload.get("claude_cli_version") != CLAUDE_CLI_VERSION:
        mismatches.append(
            f"claude pin {CLAUDE_CLI_VERSION} vs attested {payload.get('claude_cli_version')}"
        )
    if payload.get("codex_cli_version") != CODEX_CLI_VERSION:
        mismatches.append(
            f"codex pin {CODEX_CLI_VERSION} vs attested {payload.get('codex_cli_version')}"
        )
    if payload.get("checker_claude_cli_version") != CHECKER_CLAUDE_CLI_VERSION:
        mismatches.append(
            f"checker claude pin {CHECKER_CLAUDE_CLI_VERSION} vs attested "
            f"{payload.get('checker_claude_cli_version')}"
        )
    if payload.get("toolchain_identity") != toolchain_identity():
        mismatches.append(
            f"toolchain identity re-derives to {toolchain_identity()}; attested "
            f"{payload.get('toolchain_identity')} -- the seat table, a tier model, or a "
            "pin moved without re-attestation"
        )
    envelope = _envelope_fixture_path()
    if payload.get("harness_envelope_fixture") != envelope.name:
        mismatches.append(
            f"harness envelope fixture {envelope.name} vs attested "
            f"{payload.get('harness_envelope_fixture')}"
        )
    if not envelope.is_file():
        mismatches.append(f"harness envelope fixture missing: {envelope}")
    attested_tables = payload.get("fuse_tables")
    if not isinstance(attested_tables, dict):
        return [*mismatches, "attestation lacks fuse_tables"]
    tiers = dict(TIER_CAMPAIGN_AUTHOR_MODELS)
    for campaign_id in sorted(set(tiers) | set(attested_tables)):
        if campaign_id not in tiers:
            mismatches.append(f"attested fuse table for unknown tier {campaign_id}")
            continue
        derived = derive_fuse_table(campaign_id)
        attested = attested_tables.get(campaign_id)
        if not isinstance(attested, dict) or attested.get("digest") != derived.digest:
            mismatches.append(
                f"fuse table for {campaign_id} re-derives to {derived.digest} "
                f"(model {derived.author_model}); attested "
                f"{attested.get('digest') if isinstance(attested, dict) else attested!r} "
                "— re-attest the storm envelope for the changed pin/tier"
            )
    return mismatches


def mint_toolchain_attestation() -> dict:
    """Return the attestation payload for THIS tree (maintenance helper).

    A pin bump or tier change re-runs this in the SAME reviewed change (with
    a fresh harness-envelope capture for a Claude bump) and commits the
    result; :func:`verify_toolchain_attestation` and staging then re-derive
    against it.
    """

    from menagerie.crawler.storm_fuse import derive_fuse_table

    return {
        "schema_version": TOOLCHAIN_ATTESTATION_SCHEMA_VERSION,
        "claude_cli_version": CLAUDE_CLI_VERSION,
        "codex_cli_version": CODEX_CLI_VERSION,
        "checker_claude_cli_version": CHECKER_CLAUDE_CLI_VERSION,
        "harness_envelope_fixture": _envelope_fixture_path().name,
        "toolchain_identity": toolchain_identity(),
        "fuse_tables": {
            campaign_id: {
                "author_model": derive_fuse_table(campaign_id).author_model,
                "wall_seconds": derive_fuse_table(campaign_id).wall_seconds,
                "distinct_models": derive_fuse_table(campaign_id).distinct_models,
                "window_seconds": derive_fuse_table(campaign_id).window_seconds,
                "digest": derive_fuse_table(campaign_id).digest,
            }
            for campaign_id in sorted(TIER_CAMPAIGN_AUTHOR_MODELS)
        },
    }


if __name__ == "__main__":
    print(json.dumps(mint_toolchain_attestation(), indent=2, sort_keys=True))
