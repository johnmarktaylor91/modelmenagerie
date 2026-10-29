"""``run`` never silently re-freezes an existing campaign config.

D-57 (OC-6): the pilot re-froze its campaign config from the ambient
interactive shell four times without any refusal or record. A fresh ``run``
over an existing frozen config is now a typed refusal unless the operator
passes ``--refreeze-campaign-config``, and an exercised override is disclosed
as a durable operational event carrying both config digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from menagerie.crawler import cli
from menagerie.crawler.campaign_config import default_campaign_config_path
from menagerie.crawler.cli import OperatorOutageError


def _run_args(repo_root: Path, intake_root: Path, **overrides: object) -> argparse.Namespace:
    """Return a parsed-namespace stand-in for one fresh ``run`` launch."""

    namespace = argparse.Namespace(
        command="run",
        repo_root=repo_root,
        campaign_config=None,
        intake=intake_root,
        target="osx-arm64",
        run_id="run-one",
        author_queue=None,
        author_command="/bin/echo author",
        checker_command="/bin/echo checker",
        environment_command="/bin/echo environment",
        provider_reconcile_command="/bin/echo reconcile",
        notify_command=None,
        review_checkpoint_at=None,
        progress_milestones=(10,),
        phase=None,
        only_status=None,
        author_wall_seconds=None,
        author_concurrency=1,
        checker_concurrency=1,
        tmp_root=None,
        refreeze_campaign_config=False,
    )
    for name, value in overrides.items():
        setattr(namespace, name, value)
    return namespace


@pytest.fixture()
def launch_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Provide mirrors + a bare contract environment for fresh launches."""

    for name in cli.LAUNCH_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    public = tmp_path / "mirrors" / "public"
    private = tmp_path / "mirrors" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    monkeypatch.setenv("MENAGERIE_PUBLIC_MIRROR", str(public))
    monkeypatch.setenv("MENAGERIE_PRIVATE_MIRROR", str(private))
    repo_root = tmp_path / "repo"
    intake_root = repo_root / "intake" / "step-200"
    intake_root.mkdir(parents=True)
    return repo_root, intake_root


@pytest.mark.smoke
def test_run_over_an_existing_frozen_config_refuses(
    launch_environment: tuple[Path, Path],
) -> None:
    """The second ambient ``run`` is a typed refusal, not a silent refreeze."""

    repo_root, intake_root = launch_environment
    first_path, _ = cli._resolve_campaign_config(_run_args(repo_root, intake_root))
    assert first_path.is_file()
    frozen_bytes = first_path.read_bytes()

    with pytest.raises(OperatorOutageError, match="refreeze-campaign-config"):
        cli._resolve_campaign_config(
            _run_args(repo_root, intake_root, run_id="run-two")
        )
    assert first_path.read_bytes() == frozen_bytes


@pytest.mark.smoke
def test_explicit_refreeze_overrides_and_carries_both_digests(
    launch_environment: tuple[Path, Path],
) -> None:
    """--refreeze-campaign-config refreezes and mints the disclosure record."""

    repo_root, intake_root = launch_environment
    first_path, _ = cli._resolve_campaign_config(_run_args(repo_root, intake_root))
    superseded = hashlib.sha256(first_path.read_bytes()).hexdigest()

    second_path, records = cli._resolve_campaign_config(
        _run_args(
            repo_root, intake_root, run_id="run-two", refreeze_campaign_config=True
        )
    )

    assert second_path == first_path
    assert json.loads(second_path.read_text())["run_id"] == "run-two"
    refreezes = [
        record
        for record in records
        if record.get("kind") == cli._CAMPAIGN_CONFIG_REFREEZE_KIND
    ]
    assert len(refreezes) == 1
    record = refreezes[0]
    assert record["config_path"] == str(second_path)
    assert record["superseded_sha256"] == superseded
    assert (
        record["refrozen_sha256"]
        == hashlib.sha256(second_path.read_bytes()).hexdigest()
    )
    assert record["superseded_sha256"] != record["refrozen_sha256"]


@pytest.mark.smoke
def test_fresh_launch_without_an_existing_config_freezes_normally(
    launch_environment: tuple[Path, Path],
) -> None:
    """No pre-existing config means no refusal and no refreeze record."""

    repo_root, intake_root = launch_environment
    path, records = cli._resolve_campaign_config(_run_args(repo_root, intake_root))

    assert path == default_campaign_config_path(repo_root.resolve(), intake_root.resolve())
    assert path.is_file()
    assert [
        record
        for record in records
        if record.get("kind") == cli._CAMPAIGN_CONFIG_REFREEZE_KIND
    ] == []


@pytest.mark.smoke
def test_refreeze_record_discloses_as_its_own_operational_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exercised override lands in the operational ledger, typed."""

    appended: list[dict] = []

    @contextmanager
    def _fake_ledger(_paths: object) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(append=appended.append)

    monkeypatch.setattr(cli, "_launch_environment_ledger", _fake_ledger)
    record = {
        "kind": cli._CAMPAIGN_CONFIG_REFREEZE_KIND,
        "config_path": "/repo/.crawl-local/campaign-configs/step-200-abc.json",
        "superseded_sha256": "a" * 64,
        "refrozen_sha256": "b" * 64,
    }
    cli._disclose_launch_environment_divergence(
        SimpleNamespace(),
        SimpleNamespace(run_id="run-two", machine_id="machine-one"),
        (record,),
    )

    assert len(appended) == 1
    details = appended[0]["details"]
    assert details["disposition"] == "campaign-config-refreeze"
    assert details["refreeze"] == record


@pytest.mark.smoke
def test_run_parser_accepts_the_refreeze_flag() -> None:
    """The override spelling exists on the launch surface and defaults off."""

    parser = cli.build_parser()
    parsed = parser.parse_args(
        ["run", "--intake", "/tmp/intake", "--refreeze-campaign-config"]
    )
    assert parsed.refreeze_campaign_config is True
    parsed_default = parser.parse_args(["run", "--intake", "/tmp/intake"])
    assert parsed_default.refreeze_campaign_config is False
