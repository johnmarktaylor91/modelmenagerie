"""W32 (+OC-16, PM-13, AL-21): toolchain version as a rotation axis.

The author lane pins ONE reviewed Claude CLI release but resolved the binary
by bare name; codex had no pin at all; nothing verified either CLI per leg
before money moved, and a pin bump or an author-model TIER change (the
F-O23 rider: a mid-rung Opus fallback changes session-shape economics)
could land without re-attesting the storm envelope or re-deriving the fuse
table. This module is the tripwire set:

- pins resolve to ABSOLUTE paths or refuse typed;
- per-leg cached verification runs BEFORE the paid reserve and fails closed
  on version drift;
- ``toolchain_identity`` is a registered rotation axis, so a pin bump or
  tier change is a registerable rotation, never a silent drift;
- the committed toolchain attestation binds {CLI versions, harness-envelope
  fixture, per-tier derived fuse tables}; staging re-derives and refuses a
  mismatch.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from menagerie.crawler.constants import (
    CLAUDE_CLI_VERSION,
    CODEX_CLI_VERSION,
    TIER_CAMPAIGN_AUTHOR_MODELS,
)
from menagerie.crawler.rotation_registry import ROTATION_AXES
from menagerie.crawler.storm_fuse import (
    CROSS_MODEL_STORM_DISTINCT_MODELS,
    CROSS_MODEL_STORM_WINDOW_SECONDS,
    derive_fuse_table,
)
from menagerie.crawler.toolchain_pins import (
    AUTHOR_CLI_PIN_ENV,
    CHECKER_CLI_PIN_ENV,
    TOOLCHAIN_ATTESTATION_PATH,
    ToolchainPinError,
    clear_toolchain_verification_cache,
    require_verified_toolchain,
    resolve_cli_pin,
    toolchain_identity,
    verify_toolchain_attestation,
)

pytestmark = pytest.mark.smoke


def _fake_cli(tmp_path: Path, name: str, version: str) -> Path:
    script = tmp_path / name
    script.write_text(
        f"#!{sys.executable}\nprint('{name} {version}')\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_resolve_cli_pin_refuses_bare_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AUTHOR_CLI_PIN_ENV, "claude")
    with pytest.raises(ToolchainPinError, match="absolute"):
        resolve_cli_pin(AUTHOR_CLI_PIN_ENV)


def test_resolve_cli_pin_refuses_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(AUTHOR_CLI_PIN_ENV, str(tmp_path / "not-there"))
    with pytest.raises(ToolchainPinError, match="does not exist"):
        resolve_cli_pin(AUTHOR_CLI_PIN_ENV)


def test_resolve_cli_pin_absent_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(AUTHOR_CLI_PIN_ENV, raising=False)
    assert resolve_cli_pin(AUTHOR_CLI_PIN_ENV) is None


def test_require_verified_toolchain_verifies_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_toolchain_verification_cache()
    cli = _fake_cli(tmp_path, "fake-claude", CLAUDE_CLI_VERSION)
    monkeypatch.setenv(AUTHOR_CLI_PIN_ENV, str(cli))
    first = require_verified_toolchain("author", cache_key=("camp", "gen-1"))
    assert first["pinned"] is True
    assert first["observed_version"] == CLAUDE_CLI_VERSION
    assert first["cached"] is False
    # Same leg: cached, no re-run even if the binary breaks meanwhile.
    cli.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    second = require_verified_toolchain("author", cache_key=("camp", "gen-1"))
    assert second["cached"] is True
    # A NEW leg re-verifies and now fails closed.
    with pytest.raises(ToolchainPinError, match="version"):
        require_verified_toolchain("author", cache_key=("camp", "gen-2"))
    clear_toolchain_verification_cache()


def test_require_verified_toolchain_version_mismatch_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_toolchain_verification_cache()
    cli = _fake_cli(tmp_path, "fake-codex", "0.0.1")
    monkeypatch.setenv(CHECKER_CLI_PIN_ENV, str(cli))
    with pytest.raises(ToolchainPinError, match=CODEX_CLI_VERSION):
        require_verified_toolchain("checker", cache_key=("camp", "gen-1"))
    clear_toolchain_verification_cache()


def test_require_verified_toolchain_unpinned_is_disclosed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_toolchain_verification_cache()
    monkeypatch.delenv(CHECKER_CLI_PIN_ENV, raising=False)
    report = require_verified_toolchain("checker", cache_key=("camp", "gen-1"))
    assert report["pinned"] is False


def test_toolchain_is_a_rotation_axis() -> None:
    assert "toolchain_identity" in ROTATION_AXES
    identity = toolchain_identity()
    assert identity.startswith("sha256:")
    # The identity covers CLI versions AND the tier->author-model table (the
    # F-O23 rider): any change is a registerable rotation.
    assert identity == toolchain_identity()


def test_toolchain_identity_moves_with_a_tier_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = toolchain_identity()
    mutated = dict(TIER_CAMPAIGN_AUTHOR_MODELS)
    mutated["c1-mech"] = "claude-opus-5"
    monkeypatch.setattr(
        "menagerie.crawler.toolchain_pins.TIER_CAMPAIGN_AUTHOR_MODELS", mutated
    )
    assert toolchain_identity() != baseline


def test_fuse_table_is_derived_per_tier() -> None:
    for campaign_id in TIER_CAMPAIGN_AUTHOR_MODELS:
        table = derive_fuse_table(campaign_id)
        assert table.campaign_id == campaign_id
        assert table.author_model == TIER_CAMPAIGN_AUTHOR_MODELS[campaign_id]
        assert table.distinct_models == CROSS_MODEL_STORM_DISTINCT_MODELS
        # The window is wall-bound (composes with D-4): never narrower than
        # the constant, and always covering at least three tier session walls.
        assert table.window_seconds >= CROSS_MODEL_STORM_WINDOW_SECONDS
        assert table.window_seconds >= 3.0 * table.wall_seconds
        assert table.digest.startswith("sha256:")
    # Outside a tier campaign the constant-shaped default applies.
    default = derive_fuse_table(None)
    assert default.window_seconds == CROSS_MODEL_STORM_WINDOW_SECONDS


def test_committed_attestation_matches_this_tree() -> None:
    """The pin-bump / tier-change tripwire: attestation and constants move together."""

    assert verify_toolchain_attestation() == []
    payload = json.loads(TOOLCHAIN_ATTESTATION_PATH.read_text(encoding="utf-8"))
    assert payload["claude_cli_version"] == CLAUDE_CLI_VERSION
    assert payload["codex_cli_version"] == CODEX_CLI_VERSION
    assert set(payload["fuse_tables"]) == set(TIER_CAMPAIGN_AUTHOR_MODELS)


def test_attestation_catches_a_silent_tier_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutated = dict(TIER_CAMPAIGN_AUTHOR_MODELS)
    mutated["c3-classics"] = "claude-sonnet"
    monkeypatch.setattr(
        "menagerie.crawler.toolchain_pins.TIER_CAMPAIGN_AUTHOR_MODELS", mutated
    )
    monkeypatch.setattr("menagerie.crawler.storm_fuse.TIER_CAMPAIGN_AUTHOR_MODELS", mutated)
    mismatches = verify_toolchain_attestation()
    assert any("c3-classics" in row for row in mismatches)


def test_staging_refuses_a_stale_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stage_rung re-derives the toolchain attestation and refuses drift."""

    from menagerie.crawler.rung_staging import assert_toolchain_attestation_current

    # Green on this tree.
    assert_toolchain_attestation_current("c1-mech")
    mutated = dict(TIER_CAMPAIGN_AUTHOR_MODELS)
    mutated["c1-mech"] = "claude-opus-5"
    monkeypatch.setattr(
        "menagerie.crawler.toolchain_pins.TIER_CAMPAIGN_AUTHOR_MODELS", mutated
    )
    monkeypatch.setattr("menagerie.crawler.storm_fuse.TIER_CAMPAIGN_AUTHOR_MODELS", mutated)
    with pytest.raises(ToolchainPinError, match="re-attest"):
        assert_toolchain_attestation_current("c1-mech")


def test_author_executor_refuses_a_bare_name_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from menagerie.crawler.author_executor import AuthorExecutorError, ExecutorConfig

    monkeypatch.setenv(AUTHOR_CLI_PIN_ENV, "claude")
    with pytest.raises(AuthorExecutorError, match="absolute"):
        ExecutorConfig.from_env()


def test_codex_argv_head_honors_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from menagerie.crawler.operator_checker import build_codex_argv

    cli = _fake_cli(tmp_path, "fake-codex", CODEX_CLI_VERSION)
    monkeypatch.setenv(CHECKER_CLI_PIN_ENV, str(cli))
    argv = build_codex_argv(
        workdir=tmp_path,
        model="gpt-5.6-sol",
        schema_path=tmp_path / "schema.json",
        last_message_path=tmp_path / "last.json",
        prompt="p",
    )
    assert argv[0] == str(cli)
    monkeypatch.delenv(CHECKER_CLI_PIN_ENV)
    argv = build_codex_argv(
        workdir=tmp_path,
        model="gpt-5.6-sol",
        schema_path=tmp_path / "schema.json",
        last_message_path=tmp_path / "last.json",
        prompt="p",
    )
    assert argv[0] == "codex"
