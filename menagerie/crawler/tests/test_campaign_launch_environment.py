"""The launch environment contract survives a bare launchd resume environment.

A usage-pause resume fires through launchd, which hands the process a bare
environment. Seeding it with ``launchctl setenv`` is fragile and reboot-lost, so
the contract is frozen into the campaign config beside the lane commands and
restored on resume -- with an explicit resume-time value always winning, and the
divergence disclosed rather than silently absorbed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from menagerie.crawler import cli
from menagerie.crawler.campaign_config import (
    CAMPAIGN_CONFIG_FORMAT,
    CampaignConfig,
    LaunchEnvironment,
    load_campaign_config,
    write_campaign_config,
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
    """Return a parsed-namespace stand-in for a resume invocation."""

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
        # D-63: None is the real parser default -- "no flag passed". An
        # explicit width here would read as an operator override and win.
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


# --- persistence ---


@pytest.mark.smoke
def test_launch_environment_round_trips_through_the_persisted_config(
    tmp_path: Path,
) -> None:
    """The contract is durable, mode-0600, and reloads byte-identically."""

    path = tmp_path / "campaign.json"
    config = _config()

    write_campaign_config(path, config)
    loaded = load_campaign_config(path)

    assert loaded.launch_environment == config.launch_environment
    assert json.loads(path.read_text())["format"] == CAMPAIGN_CONFIG_FORMAT


@pytest.mark.smoke
def test_absent_optional_members_round_trip_as_null(tmp_path: Path) -> None:
    """An unset state root, tmp root, or model persists as null, not as ''."""

    path = tmp_path / "campaign.json"
    config = _config(
        launch_environment=_launch_environment(
            environment_state_root=None,
            tmp_root=None,
            author_model=None,
            require_author_receipt=False,
        )
    )

    write_campaign_config(path, config)
    loaded = load_campaign_config(path)

    assert loaded.launch_environment == config.launch_environment
    persisted = json.loads(path.read_text())["launch_environment"]
    assert persisted["environment_state_root"] is None
    assert persisted["require_author_receipt"] is False


@pytest.mark.smoke
def test_legacy_v5_config_without_the_contract_still_loads(tmp_path: Path) -> None:
    """A campaign frozen before this field resumes; it just restores nothing."""

    path = tmp_path / "campaign-v5.json"
    payload = _config().to_dict()
    del payload["launch_environment"]
    payload["format"] = "menagerie.crawler.campaign-config.v5"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)

    loaded = load_campaign_config(path)

    assert loaded.launch_environment == LaunchEnvironment(None, None, None, False)


@pytest.mark.smoke
def test_a_relative_persisted_path_refuses(tmp_path: Path) -> None:
    """Clean-environment safety means absolute paths only, validated on load."""

    path = tmp_path / "campaign-bad.json"
    payload = _config().to_dict()
    payload["launch_environment"]["tmp_root"] = "relative/tmp"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="tmp root"):
        load_campaign_config(path)


@pytest.mark.smoke
def test_the_contract_field_set_is_exact(tmp_path: Path) -> None:
    """An extra or missing member refuses instead of being silently ignored."""

    path = tmp_path / "campaign-extra.json"
    payload = _config().to_dict()
    payload["launch_environment"]["surprise"] = "value"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="launch environment"):
        load_campaign_config(path)


# --- restore into a bare wake environment ---


@pytest.mark.smoke
def test_bare_resume_restores_the_whole_contract(
    bare_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This is LB-2's actual fix: no launchctl setenv, no reboot loss."""

    monkeypatch.setattr(cli.os, "environ", dict())
    args = _args()

    divergences = cli._apply_campaign_config(args, _config())

    assert cli.os.environ["MENAGERIE_ENVIRONMENT_STATE_ROOT"] == "/records/cas"
    assert cli.os.environ["MENAGERIE_AUTHOR_MODEL"] == "claude-fable-5-1"
    assert cli.os.environ["MENAGERIE_AUTHOR_REQUIRE_RECEIPT"] == "1"
    assert args.tmp_root == Path("/scratch/tmp")
    assert divergences == ()


@pytest.mark.smoke
def test_concurrency_and_reconcile_command_reach_the_environment(
    bare_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both are already first-class config fields; the env is derived from them.

    They are NOT duplicated into the contract object: two persisted copies of
    one value is how a resume silently reverts half of a tuned launch.
    """

    monkeypatch.setattr(cli.os, "environ", dict())

    cli._apply_campaign_config(_args(), _config())

    assert cli.os.environ["MENAGERIE_AUTHOR_CONCURRENCY"] == "3"
    assert cli.os.environ["MENAGERIE_CHECKER_CONCURRENCY"] == "2"
    assert cli.os.environ["MENAGERIE_PROVIDER_RECONCILE_COMMAND"] == "/bin/echo reconcile"


@pytest.mark.smoke
def test_receipt_flag_off_restores_as_absent_not_as_zero(
    bare_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is read as ``== "1"``, so a frozen-off campaign must stay unset."""

    monkeypatch.setattr(cli.os, "environ", dict())

    cli._apply_campaign_config(
        _args(), _config(launch_environment=_launch_environment(require_author_receipt=False))
    )

    assert "MENAGERIE_AUTHOR_REQUIRE_RECEIPT" not in cli.os.environ


# --- precedence: an explicit resume-time value wins, and is disclosed ---


@pytest.mark.smoke
def test_explicit_environment_wins_over_the_persisted_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator override at resume time is honored, not overwritten."""

    monkeypatch.setattr(
        cli.os, "environ", {"MENAGERIE_AUTHOR_MODEL": "claude-opus-5"}
    )

    divergences = cli._apply_campaign_config(_args(), _config())

    assert cli.os.environ["MENAGERIE_AUTHOR_MODEL"] == "claude-opus-5"
    diverged = {record["name"]: record for record in divergences}
    assert diverged["MENAGERIE_AUTHOR_MODEL"]["persisted"] == "claude-fable-5-1"
    assert diverged["MENAGERIE_AUTHOR_MODEL"]["explicit"] == "claude-opus-5"


@pytest.mark.smoke
def test_an_explicit_tmp_root_flag_wins_and_is_disclosed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tmp root's override channel is its flag, not an environment name."""

    monkeypatch.setattr(cli.os, "environ", dict())
    args = _args(tmp_root=Path("/other/tmp"))

    divergences = cli._apply_campaign_config(args, _config())

    assert args.tmp_root == Path("/other/tmp")
    assert [record["name"] for record in divergences] == ["--tmp-root"]


@pytest.mark.smoke
def test_an_identical_explicit_value_is_not_a_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a real disagreement is disclosed; agreement is silent."""

    monkeypatch.setattr(
        cli.os, "environ", {"MENAGERIE_AUTHOR_MODEL": "claude-fable-5-1"}
    )

    assert cli._apply_campaign_config(_args(), _config()) == ()


@pytest.mark.smoke
def test_divergence_becomes_one_operational_event() -> None:
    """The override is disclosed in an operational event, never only in a log."""

    divergences = (
        {
            "name": "MENAGERIE_AUTHOR_MODEL",
            "persisted": "claude-fable-5-1",
            "explicit": "claude-opus-5",
        },
    )

    event = cli._launch_environment_divergence_event(
        divergences, run_id="run-one", machine_id="host-one"
    )

    assert event["details"]["disposition"] == "launch-environment-override"
    assert event["details"]["overrides"] == list(divergences)
    assert event["run_id"] == "run-one"


@pytest.mark.smoke
def test_the_restore_runs_before_the_environment_readiness_check() -> None:
    """The ordering LB-2 actually turns on.

    The readiness receipt is validated against
    ``MENAGERIE_ENVIRONMENT_STATE_ROOT``. If the restore ran later -- in the
    driver factory, where the rest of the contract is consumed -- a bare
    launchd wake would refuse on the missing variable before ever reading the
    config that holds it, and persisting the state root would buy nothing.
    """

    import inspect

    source = inspect.getsource(cli.main)
    restore_at = source.index("_load_or_persist_campaign_config(args)")
    check_at = source.index("_require_environment_readiness_receipt(args)")

    assert restore_at < check_at


@pytest.mark.smoke
def test_the_resolution_is_memoized_so_divergence_is_disclosed_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second pass must not observe the environment the first pass restored.

    Without the memo the pre-pass restores the value, and the factory's own
    call then compares the persisted value against the value it just wrote,
    finds them equal, and reports no override -- so the disclosure disappears
    exactly when an operator override is in effect.
    """

    monkeypatch.setattr(cli.os, "environ", {"MENAGERIE_AUTHOR_MODEL": "claude-opus-5"})
    config_path = tmp_path / "campaign.json"
    write_campaign_config(config_path, _config(repo_root=tmp_path))
    args = _args(repo_root=tmp_path, campaign_config=config_path)

    first = cli._load_or_persist_campaign_config(args)
    second = cli._load_or_persist_campaign_config(args)

    assert first is second
    assert [record["name"] for record in first[1]] == ["MENAGERIE_AUTHOR_MODEL"]


@pytest.mark.smoke
def test_the_contract_vocabulary_is_closed() -> None:
    """The restored environment names are enumerated, never computed."""

    assert cli.LAUNCH_ENVIRONMENT_VARIABLES == (
        "MENAGERIE_AUTHOR_CONCURRENCY",
        "MENAGERIE_AUTHOR_MODEL",
        "MENAGERIE_AUTHOR_REQUIRE_RECEIPT",
        "MENAGERIE_CHECKER_CONCURRENCY",
        "MENAGERIE_ENVIRONMENT_STATE_ROOT",
        "MENAGERIE_PROVIDER_RECONCILE_COMMAND",
    )
