"""TC-21: scheduled-CI reds text once per new red, recover once, never re-spam."""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.tools.scheduled_ci_watch import (
    SCHEDULED_WORKFLOWS,
    check_workflows,
    render_launchd_plist,
)

pytestmark = pytest.mark.smoke


def _fetcher(table):
    def fetch(workflow, _repo_root):
        return table.get(workflow)

    return fetch


def _run(run_id, conclusion):
    return {
        "databaseId": run_id,
        "conclusion": conclusion,
        "displayTitle": "scheduled",
        "url": f"https://example.invalid/runs/{run_id}",
    }


class TestRedRouting:
    def test_new_red_texts_once_and_only_once(self, tmp_path: Path) -> None:
        state = tmp_path / "state.json"
        table = {w: _run(1, "success") for w in SCHEDULED_WORKFLOWS}
        table["nightly.yml"] = _run(7, "failure")
        sent: list[str] = []
        red = check_workflows(
            state_path=state, repo_root=tmp_path, fetch=_fetcher(table), send=sent.append
        )
        assert red == 1
        assert len(sent) == 1 and "nightly.yml" in sent[0] and "failure" in sent[0]
        sent.clear()
        red = check_workflows(
            state_path=state, repo_root=tmp_path, fetch=_fetcher(table), send=sent.append
        )
        assert red == 1
        assert sent == [], "a standing red must not re-text every poll"

    def test_a_new_red_run_texts_again(self, tmp_path: Path) -> None:
        state = tmp_path / "state.json"
        table = {w: _run(1, "success") for w in SCHEDULED_WORKFLOWS}
        table["weekly.yml"] = _run(7, "failure")
        sent: list[str] = []
        check_workflows(
            state_path=state, repo_root=tmp_path, fetch=_fetcher(table), send=sent.append
        )
        table["weekly.yml"] = _run(8, "failure")
        check_workflows(
            state_path=state, repo_root=tmp_path, fetch=_fetcher(table), send=sent.append
        )
        assert len(sent) == 2

    def test_recovery_texts_once(self, tmp_path: Path) -> None:
        state = tmp_path / "state.json"
        table = {w: _run(1, "success") for w in SCHEDULED_WORKFLOWS}
        table["latest-canary.yml"] = _run(7, "failure")
        sent: list[str] = []
        check_workflows(
            state_path=state, repo_root=tmp_path, fetch=_fetcher(table), send=sent.append
        )
        table["latest-canary.yml"] = _run(8, "success")
        check_workflows(
            state_path=state, repo_root=tmp_path, fetch=_fetcher(table), send=sent.append
        )
        assert any("recovered" in message for message in sent)
        sent.clear()
        check_workflows(
            state_path=state, repo_root=tmp_path, fetch=_fetcher(table), send=sent.append
        )
        assert sent == []

    def test_unwatchable_watcher_discloses_itself(self, tmp_path: Path) -> None:
        state = tmp_path / "state.json"
        table = {w: _run(1, "success") for w in SCHEDULED_WORKFLOWS}
        table.pop("nightly.yml")
        sent: list[str] = []
        check_workflows(
            state_path=state, repo_root=tmp_path, fetch=_fetcher(table), send=sent.append
        )
        assert sent == []
        check_workflows(
            state_path=state, repo_root=tmp_path, fetch=_fetcher(table), send=sent.append
        )
        assert any("UNWATCHED" in message for message in sent)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "SCHEDULED_WORKFLOWS names TorchLens's nightly, weekly and latest-canary"
            " workflows; the menagerie repo has no scheduled CI yet (JMT 2026-10-01)"
        ),
    )
    def test_watched_set_matches_the_scheduled_workflows_on_disk(self) -> None:
        workflows_dir = Path(__file__).resolve().parents[3] / ".github" / "workflows"
        scheduled = {
            path.name
            for path in workflows_dir.glob("*.yml")
            if "schedule:" in path.read_text(encoding="utf-8")
        }
        assert scheduled == set(SCHEDULED_WORKFLOWS), (
            "a scheduled workflow is not under watch (or a watched one is"
            " gone); update SCHEDULED_WORKFLOWS in the same reviewed change"
        )

    def test_plist_polls_and_sends(self, tmp_path: Path) -> None:
        import plistlib

        payload = plistlib.loads(render_launchd_plist(1800, tmp_path).encode("utf-8"))
        assert payload["StartInterval"] == 1800
        assert "--send" in payload["ProgramArguments"]
