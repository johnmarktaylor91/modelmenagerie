"""D-63 (PM-1): the explicit-wins concurrency override actually WINS.

The 200a shape this file retires: ``--author-concurrency 18`` and
``MENAGERIE_AUTHOR_CONCURRENCY=18`` on a resume were both INERT.
``_apply_campaign_config`` clobbered the parsed value with the frozen one
BEFORE the explicit-wins resolver ran, the env name had no runtime consumer
(only the argparse default, already evaluated), and the divergence disclosure
then asserted an override was "in effect" that nothing consumed -- legs 23-43
ran width 12 under a ledger claiming 18 (risk register R6: disclosure lies).

The contract pinned here:

- precedence is EXPLICIT-WINS with a closed source vocabulary:
  explicit flag > explicit environment > frozen campaign config > default;
- the resolved source of each width is stamped for the leg-config attestation
  (``args.concurrency_sources``), so "which channel won" is durable;
- a flag-channel divergence is disclosed with the value that actually applied
  (mirroring the ``--tmp-root`` precedent), never a value nothing consumed;
- absent any explicit source the frozen value still wins (the historical
  resume behavior is preserved bit-for-bit).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from menagerie.crawler import cli
from menagerie.crawler.campaign_config import CampaignConfig, LaunchEnvironment
from menagerie.crawler.constants import (
    DEFAULT_AUTHOR_WAVE_CONCURRENCY,
    DEFAULT_CHECKER_WAVE_CONCURRENCY,
)


def _launch_environment(**overrides: object) -> LaunchEnvironment:
    """Return one fully populated launch environment contract."""

    base: dict[str, object] = {
        "environment_state_root": Path("/records/cas"),
        "tmp_root": Path("/scratch/tmp"),
        "author_model": "claude-fable-5-1",
        "require_author_receipt": True,
    }
    base.update(overrides)
    return LaunchEnvironment(**base)  # type: ignore[arg-type]


def _config(**overrides: object) -> CampaignConfig:
    """Return one resolved campaign config over absolute placeholder paths."""

    base: dict[str, object] = {
        "repo_root": Path("/repo"),
        "intake_root": Path("/repo/intake/step-200"),
        "target": "osx-arm64",
        "run_id": "run-one",
        "author_queue_root": None,
        "author_command": ("/bin/echo", "author"),
        "checker_command": ("/bin/echo", "checker"),
        "environment_command": ("/bin/echo", "environment"),
        "notify_command": None,
        "public_mirror": Path("/mirrors/public"),
        "private_mirror": Path("/mirrors/private"),
        "review_checkpoint_at": None,
        "progress_milestones": (10,),
        "phase": None,
        "only_status": None,
        "author_wall_seconds": 3600.0,
        "author_concurrency": 3,
        "checker_concurrency": 2,
        "provider_reconcile_command": ("/bin/echo", "reconcile"),
        "launch_environment": _launch_environment(),
    }
    base.update(overrides)
    return CampaignConfig(**base)  # type: ignore[arg-type]


def _args(**overrides: object) -> argparse.Namespace:
    """Return a parsed-namespace stand-in for a resume invocation.

    ``author_concurrency``/``checker_concurrency`` default to ``None`` exactly
    like the real parser after D-63: ``None`` means "the operator passed no
    flag", so the sentinel is the honesty mechanism itself.
    """

    namespace = argparse.Namespace(
        command="resume",
        intake=None,
        target=None,
        run_id=None,
        author_queue=None,
        author_command=None,
        checker_command=None,
        environment_command=None,
        provider_reconcile_command=None,
        notify_command=None,
        review_checkpoint_at=None,
        progress_milestones=(),
        phase=None,
        only_status=None,
        author_wall_seconds=None,
        author_concurrency=None,
        checker_concurrency=None,
        tmp_root=None,
    )
    for name, value in overrides.items():
        setattr(namespace, name, value)
    return namespace


@pytest.fixture()
def bare_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate the launchd wake: every contract variable absent."""

    for name in cli.LAUNCH_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.smoke
def test_explicit_flag_wins_over_the_frozen_concurrency(
    bare_environment: None,
) -> None:
    """The exact 200a failure: the flag must displace the frozen width."""

    args = _args(author_concurrency=18, checker_concurrency=None)

    cli._apply_campaign_config(args, _config())

    assert args.author_concurrency == 18
    assert args.checker_concurrency == 2
    assert args.concurrency_sources == {
        "author_concurrency": "explicit-flag",
        "checker_concurrency": "frozen-config",
    }


@pytest.mark.smoke
def test_explicit_env_wins_over_the_frozen_concurrency(
    bare_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env channel must reach the runtime consumer, not just argparse."""

    monkeypatch.setenv("MENAGERIE_AUTHOR_CONCURRENCY", "18")
    args = _args()

    divergences = cli._apply_campaign_config(args, _config())

    assert args.author_concurrency == 18
    assert args.concurrency_sources["author_concurrency"] == "explicit-env"
    # The env-channel divergence is still disclosed by the launch-environment
    # restore, now describing a value that actually applied.
    names = {record["name"] for record in divergences}
    assert "MENAGERIE_AUTHOR_CONCURRENCY" in names


def test_the_frozen_value_wins_only_absent_any_explicit_source(
    bare_environment: None,
) -> None:
    """No explicit source: the historical resume behavior is preserved."""

    args = _args()

    divergences = cli._apply_campaign_config(args, _config())

    assert args.author_concurrency == 3
    assert args.checker_concurrency == 2
    assert args.concurrency_sources == {
        "author_concurrency": "frozen-config",
        "checker_concurrency": "frozen-config",
    }
    assert divergences == ()


def test_the_flag_beats_the_env_which_beats_the_frozen_value(
    bare_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full precedence order in one shot."""

    monkeypatch.setenv("MENAGERIE_AUTHOR_CONCURRENCY", "20")
    args = _args(author_concurrency=18)

    cli._apply_campaign_config(args, _config())

    assert args.author_concurrency == 18
    assert args.concurrency_sources["author_concurrency"] == "explicit-flag"


def test_a_malformed_env_width_is_refused_not_ignored(
    bare_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silent fallback would hide a misconfigured width for a month."""

    monkeypatch.setenv("MENAGERIE_AUTHOR_CONCURRENCY", "twelve")

    with pytest.raises(ValueError, match="MENAGERIE_AUTHOR_CONCURRENCY"):
        cli._apply_campaign_config(_args(), _config())


@pytest.mark.smoke
def test_the_flag_divergence_is_disclosed_with_the_applied_value(
    bare_environment: None,
) -> None:
    """The disclosure must describe what ran, never a value nothing consumed."""

    args = _args(author_concurrency=18)

    divergences = cli._apply_campaign_config(args, _config())

    flag_records = [
        record for record in divergences if record["name"] == "--author-concurrency"
    ]
    assert flag_records == [
        {"name": "--author-concurrency", "persisted": "3", "explicit": "18"}
    ]
    # The applied value and the disclosed explicit value are the SAME number:
    # that identity is the whole point of D-63.
    assert str(args.author_concurrency) == flag_records[0]["explicit"]


def test_an_identical_explicit_flag_is_not_a_divergence(
    bare_environment: None,
) -> None:
    """A wake replaying the effective width via the continuation argv is quiet."""

    args = _args(author_concurrency=3, checker_concurrency=2)

    divergences = cli._apply_campaign_config(args, _config())

    assert args.author_concurrency == 3
    assert divergences == ()
    assert args.concurrency_sources["author_concurrency"] == "explicit-flag"


def test_fresh_launch_resolution_orders_flag_env_default(
    bare_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fresh-launch path resolves the SAME order with no frozen config."""

    args = _args(command="run", author_concurrency=7)
    cli._resolve_effective_concurrency(args, None)
    assert args.author_concurrency == 7
    assert args.checker_concurrency == DEFAULT_CHECKER_WAVE_CONCURRENCY
    assert args.concurrency_sources == {
        "author_concurrency": "explicit-flag",
        "checker_concurrency": "default",
    }

    monkeypatch.setenv("MENAGERIE_CHECKER_CONCURRENCY", "9")
    env_args = _args(command="run")
    cli._resolve_effective_concurrency(env_args, None)
    assert env_args.author_concurrency == DEFAULT_AUTHOR_WAVE_CONCURRENCY
    assert env_args.checker_concurrency == 9
    assert env_args.concurrency_sources == {
        "author_concurrency": "default",
        "checker_concurrency": "explicit-env",
    }


def test_the_source_vocabulary_is_closed(bare_environment: None) -> None:
    """Every stamped source is a member of the closed vocabulary."""

    assert cli.CONCURRENCY_SOURCES == (
        "explicit-flag",
        "explicit-env",
        "frozen-config",
        "default",
    )
    args = _args(author_concurrency=18)
    cli._apply_campaign_config(args, _config())
    for source in args.concurrency_sources.values():
        assert source in cli.CONCURRENCY_SOURCES
