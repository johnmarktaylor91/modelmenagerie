"""RC-1 behaviors 1 and 6: the ``crawl`` CLI shape and the supervisor's crawl shape."""

from __future__ import annotations

import argparse
import json
import plistlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler import cli, supervisor
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.tests.crawl_support import (
    CAMPAIGN,
    build_crawl_repo,
    crawl_paths,
    plenty_of_disk,
    records_root_of,
    root_snapshot,
)


@pytest.mark.smoke
def test_crawl_parser_accepts_the_spec_shape() -> None:
    """``crawl --records-root R --campaign C [--limit N] [--author-width W]``."""

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "crawl",
            "--records-root",
            "/tmp/records",
            "--campaign",
            "c1-mech",
            "--limit",
            "230",
            "--author-width",
            "12",
        ]
    )
    assert args.command == "crawl"
    assert args.records_root == Path("/tmp/records")
    assert args.campaign == "c1-mech"
    assert args.limit == 230
    assert args.author_width == 12
    minimal = parser.parse_args(["crawl", "--records-root", "/tmp/r", "--campaign", "c2-disco"])
    assert minimal.limit is None and minimal.author_width is None
    with pytest.raises(SystemExit):
        parser.parse_args(["crawl", "--records-root", "/tmp/r", "--campaign", "not-a-campaign"])


@pytest.mark.smoke
def test_crawl_command_derives_the_leg_flags_and_configures_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The factory sees the root intake and the driver runs in the crawl shape."""

    from menagerie.crawler.tests.test_slice_f_driver import _driver

    repo_root = build_crawl_repo(tmp_path, members=3, layout="production")
    monkeypatch.setattr(CrawlerDriver, "_crawl_disk_free", staticmethod(plenty_of_disk))
    monkeypatch.delenv("MENAGERIE_ENVIRONMENT_STATE_ROOT", raising=False)
    seen: dict[str, Any] = {}

    def factory(namespace: argparse.Namespace) -> CrawlerDriver:
        seen["intake"] = namespace.intake
        seen["scheduler"] = namespace.scheduler
        seen["review"] = namespace.review_checkpoint_at
        seen["author_concurrency"] = namespace.author_concurrency
        snapshot = root_snapshot(repo_root)
        driver = _driver(
            tmp_path,
            snapshot,
            paths_override=crawl_paths(repo_root, snapshot),
            scheduler="pipelined",
            author_concurrency=int(namespace.author_concurrency),
        )
        seen["driver"] = driver
        return driver

    code = cli.main(
        [
            "--repo-root",
            str(repo_root),
            "crawl",
            "--records-root",
            str(records_root_of(repo_root)),
            "--campaign",
            CAMPAIGN,
            "--limit",
            "2",
            "--author-width",
            "3",
        ],
        driver_factory=factory,
    )
    assert code == cli.EXIT_OK
    snapshot = root_snapshot(repo_root)
    assert seen["intake"] == records_root_of(repo_root) / f"intake/{snapshot.snapshot_id}"
    assert seen["scheduler"] == "pipelined"
    assert seen["review"] == 1000  # c1-mech's frozen spec
    assert seen["author_concurrency"] == 3
    driver = seen["driver"]
    assert driver.config.invocation_shape == "crawl"
    assert driver.config.crawl_limit == 2
    state = json.loads(driver.paths.driver_state.read_text())
    assert state["status"] == "work_exhausted"


@pytest.mark.smoke
def test_crawl_command_refuses_a_records_root_outside_the_repository(tmp_path: Path) -> None:
    """RC-1 supports the canonical layout; anything else is a typed operator outage."""

    repo_root = build_crawl_repo(tmp_path, members=2, green_verdicts=False, layout="production")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    code = cli.main(
        [
            "--repo-root",
            str(repo_root),
            "crawl",
            "--records-root",
            str(elsewhere),
            "--campaign",
            CAMPAIGN,
        ],
        driver_factory=lambda _args: pytest.fail("factory must not run"),
    )
    assert code == cli.EXIT_OPERATOR_OUTAGE


@pytest.mark.smoke
def test_typed_invocation_refusal_exits_operator_outage_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """closure-green missing -> one JSON refusal line on stderr, typed exit code."""

    from menagerie.crawler.tests.test_slice_f_driver import _driver

    repo_root = build_crawl_repo(tmp_path, members=2, green_verdicts=False, layout="production")
    monkeypatch.setattr(CrawlerDriver, "_crawl_disk_free", staticmethod(plenty_of_disk))
    monkeypatch.delenv("MENAGERIE_ENVIRONMENT_STATE_ROOT", raising=False)

    def factory(_namespace: argparse.Namespace) -> CrawlerDriver:
        snapshot = root_snapshot(repo_root)
        return _driver(
            tmp_path, snapshot, paths_override=crawl_paths(repo_root, snapshot), scheduler="pipelined"
        )

    code = cli.main(
        [
            "--repo-root",
            str(repo_root),
            "crawl",
            "--records-root",
            str(records_root_of(repo_root)),
            "--campaign",
            CAMPAIGN,
        ],
        driver_factory=factory,
    )
    assert code == cli.EXIT_OPERATOR_OUTAGE
    err = capsys.readouterr().err.strip().splitlines()
    payload = json.loads(err[-1])
    assert payload["status"] == "invocation-refused"
    assert payload["arm"] == "closure-green"
    assert "Traceback" not in "\n".join(err)


# ---------------------------------------------------------------------------
# Behavior 6: the supervisor's crawl shape.
# ---------------------------------------------------------------------------


def _crawl_supervisor_config(tmp_path: Path) -> supervisor.SupervisorConfig:
    return supervisor.SupervisorConfig(
        campaign_id=CAMPAIGN,
        repo_root=tmp_path / "repo",
        campaign_config_path=tmp_path / "repo" / ".crawl-local" / "campaign-configs",
        author_queue_root=None,
        runtime_root=tmp_path / "repo" / ".crawl-local" / "supervisor" / CAMPAIGN,
        author_concurrency=12,
        selection_argv=(
            "--records-root",
            str(tmp_path / "repo" / "menagerie" / "crawler" / "records"),
            "--campaign",
            CAMPAIGN,
            "--limit",
            "230",
            "--author-width",
            "12",
            "--target",
            "osx-arm64",
        ),
        driver_shape="crawl",
        records_root=tmp_path / "repo" / "menagerie" / "crawler" / "records",
        crawl_limit=230,
        target="osx-arm64",
    )


@pytest.mark.smoke
def test_supervisor_driver_command_gains_the_crawl_shape(tmp_path: Path) -> None:
    """The supervised argv is ``python -m menagerie.crawler crawl ...`` verbatim."""

    config = _crawl_supervisor_config(tmp_path)
    command = supervisor._driver_command(config)
    assert command[1:3] == ("-m", "menagerie.crawler")
    assert command[3:5] == ("--repo-root", str(tmp_path / "repo"))
    assert command[5] == "crawl"
    assert "--records-root" in command and "--campaign" in command
    assert command[command.index("--limit") + 1] == "230"
    assert "resume" not in command and "--campaign-config" not in command
    restarted = supervisor._driver_command(config, restart_ordinal=2)
    assert restarted[-2:] == ("--supervised-restart-ordinal", "2")
    # The leg shape is untouched for one release.
    legacy = replace(config, driver_shape="resume", selection_argv=())
    assert "resume" in supervisor._driver_command(legacy)


@pytest.mark.smoke
def test_supervisor_crawl_parser_and_plist_render(tmp_path: Path) -> None:
    """``supervisor crawl --render-launchd`` writes the keep-alive plist for the shape."""

    repo_root = build_crawl_repo(tmp_path, members=2, green_verdicts=False, layout="production")
    output = tmp_path / "crawl.plist"
    code = supervisor.main(
        [
            "crawl",
            "--campaign-id",
            CAMPAIGN,
            "--repo-root",
            str(repo_root),
            "--records-root",
            str(records_root_of(repo_root)),
            "--limit",
            "230",
            "--author-width",
            "12",
            "--render-launchd",
            str(output),
        ]
    )
    assert code == supervisor.EXIT_OK
    payload = plistlib.loads(output.read_bytes())
    arguments = payload["ProgramArguments"]
    assert arguments[3] == "crawl"
    assert "--records-root" in arguments and "--limit" in arguments
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["Label"] == f"org.menagerie.crawler.{CAMPAIGN}"


@pytest.mark.smoke
def test_supervisor_settles_work_exhausted_as_a_described_stop(tmp_path: Path) -> None:
    """A ``work_exhausted`` driver state at exit 0 stops the supervisor cleanly."""

    config = _crawl_supervisor_config(tmp_path)
    runtime = config.repo_root / ".crawl-local"
    runtime.mkdir(parents=True)
    (runtime / "driver-state.json").write_text(json.dumps({"status": "work_exhausted"}))
    config.runtime_root.mkdir(parents=True)
    delivered: list[str] = []

    class Notifier:
        def notify(self, summary: str, **_: Any) -> None:
            delivered.append(summary)

    sup = supervisor.CrawlerSupervisor(config, Notifier())  # type: ignore[arg-type]
    sup._notify_until_delivered = lambda event, summary: delivered.append(event)  # type: ignore[method-assign]
    assert sup._settle_exit_zero() == supervisor.EXIT_OK
    assert delivered == ["work-exhausted"]
