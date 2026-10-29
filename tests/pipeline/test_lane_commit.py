"""The tick's ``lanes`` step: what lanes wrote reaches a commit once no live lane holds it.

The step runs on a real git checkout in a temporary folder; ``git add`` plus ``git commit --
<paths>`` stands in for ``git-commit-paths`` (named paths only), and the leases are a fake set.
``test_tick`` checks that every pass runs the step.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import DataLayout, append_jsonl, lane_commit, tick, write_json
from menagerie.pipeline.records import StoreIndexLine


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout


def _git_commit(repo: Path, paths: Sequence[str], message: str, deadline: float) -> None:
    for argv in (["add", "--", *paths], ["commit", "-qm", message, "--", *paths]):
        done = subprocess.run(["git", "-C", str(repo), *argv], capture_output=True, text=True,
                              check=False)
        if done.returncode:
            raise OSError(f"git {argv[0]} rc {done.returncode}: {done.stderr[-300:]}")


def _line(n: int) -> StoreIndexLine:
    return StoreIndexLine(sha256=f"{n:064x}", bytes=n, mime="application/json",
                          fetched="2026-10-20T00:00:00Z", license_class="third-party")


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """A real git checkout with one committed ledger line."""

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = Tick Test\n\temail = tick@example.invalid\n"
                   "[init]\n\tdefaultBranch = main\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    data = DataLayout(root.resolve() / "menagerie" / "data")
    append_jsonl([_line(0)], data.ledger("2026-10"))
    _git(root, "add", "menagerie")
    _git(root, "commit", "-qm", "base")
    return data


def _lane_writes(data: DataLayout, family: str, model_id: str) -> list[Path]:
    """What a finished 2a and 2b lane leaves: a packet, an adapter, a model record."""

    write_json(_line(1), data.packet(family))  # any record stands in for the packet
    data.adapter(family).parent.mkdir(parents=True, exist_ok=True)
    data.adapter(family).write_text("def build():\n    return None\n", encoding="utf-8")
    record = data.model(model_id)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"id": model_id, "family": family}) + "\n", encoding="utf-8")
    return [data.packet(family), data.adapter(family), record]


def _pass(data: DataLayout, leased: set[str], state: dict[str, Any]) -> tick.Pass:
    """Run the ``lanes`` step as a pass does; its result is ``results["lanes"]``."""

    repo = data.root.parents[1]
    deps = tick.Deps(repo=lambda d: repo, lane_keys=lambda d: set(leased), commit=_git_commit)
    p = tick.Pass(mode="collect", dry_run=False, data=data, deps=deps, cfg=tick.config(),
                  state=state, started=datetime.now(timezone.utc), deadline=float("inf"))
    p.results["lanes"] = tick._step_lanes(p)
    return p


def _committed(repo: Path, path: Path) -> bool:
    rel = path.relative_to(repo).as_posix()
    return bool(_git(repo, "ls-tree", "HEAD", "--", rel)) and not _git(repo, "status",
                                                                         "--porcelain", "--", rel)


def test_a_finished_lanes_results_reach_a_commit_and_a_live_lanes_wait(checkout: DataLayout) -> None:
    repo = checkout.root.parents[1]
    state: dict[str, Any] = {}
    done = _lane_writes(checkout, "f-done", "m1")
    live = _lane_writes(checkout, "f-live", "m2")
    append_jsonl([_line(2)], checkout.ledger("2026-10"))  # a lane's give-up row
    bad = checkout.packet("f-bad")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text('{"family": "f-bad"', encoding="utf-8")  # cut off mid-write

    p = _pass(checkout, leased={"f-live"}, state=state)
    out = p.results["lanes"]

    assert all(_committed(repo, p) for p in [*done, checkout.ledger("2026-10")])
    assert not any(_committed(repo, p) for p in [*live, bad])
    assert out["committed"] == 4 and out["held_by_lane"] == 3 and out["holding_keys"] == ["f-live"]
    assert list(out["unreadable"]) == [bad.relative_to(repo).as_posix()]
    assert [k for k, t in p.messages if "lane result files stay uncommitted" in t] == ["lane-commit"]
    assert "tick committed lane results" in _git(repo, "log", "-1", "--format=%s")

    out = _pass(checkout, leased=set(), state=state).results["lanes"]  # the lane has ended
    assert all(_committed(repo, p) for p in live) and out["committed"] == 3
    assert not _committed(repo, bad)


def test_the_switch_turns_the_step_off(checkout: DataLayout) -> None:
    (tick._tick_dir()).mkdir(parents=True, exist_ok=True)
    (tick._tick_dir() / "config.json").write_text(json.dumps({"commit_lanes": False}))
    _lane_writes(checkout, "f-done", "m1")
    out = _pass(checkout, leased=set(), state={}).results["lanes"]
    assert out == {"skipped": "commit_lanes is off"}


@pytest.mark.parametrize(("rel", "parsed", "keys"), [
    ("research/f-/f-a.json", {}, ("f-a",)),
    ("adapters/f-a--m7.py", None, ("f-a",)),
    ("adapters/f-a.py", None, ("f-a",)),
    ("models/1/m1.json", {"id": "m1", "family": "f-a"}, ("f-a", "m1")),
    ("crawl_roster.jsonl", None, ()),
])
def test_each_file_belongs_to_its_family(rel: str, parsed: Any, keys: tuple[str, ...]) -> None:
    assert lane_commit.keys_of(rel, parsed) == keys


def _files(tmp_path: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = tmp_path / "d" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if name.endswith(".json") else "\n")
    return tmp_path


def test_a_refused_batch_is_halved_until_the_refused_file_stands_alone(tmp_path: Path) -> None:
    names = [f"research/f-{n}.json" for n in range(6)]
    repo = _files(tmp_path, names)
    batches: list[list[str]] = []

    def commit(repo: Path, paths: Sequence[str], message: str, deadline: float) -> None:
        if "d/research/f-3.json" in paths:
            raise OSError("COMMIT BLOCKED: public-content guard")
        batches.append(list(paths))

    state: dict[str, Any] = {}
    digest = {"d/research/f-3.json": "h1"}
    out = lane_commit.step(repo, "d", [f"d/{n}" for n in names], set(), state, commit=commit,
                           digest=lambda r, p: digest.get(p), deadline=float("inf"))
    assert out["committed"] == 5 and out["refused"] == ["d/research/f-3.json"]
    assert state["lane_refused"] == {"d/research/f-3.json": "h1"}
    assert sorted(p for b in batches for p in b) == [f"d/{n}" for n in names if "f-3" not in n]

    again = lane_commit.step(repo, "d", ["d/research/f-3.json"], set(), state, commit=commit,
                             digest=lambda r, p: digest.get(p), deadline=float("inf"))
    assert again["ready"] == 0 and again["refused"] == ["d/research/f-3.json"]  # unchanged: skipped
    digest["d/research/f-3.json"] = "h2"
    third = lane_commit.step(repo, "d", ["d/research/f-3.json"], set(), state, commit=commit,
                             digest=lambda r, p: digest.get(p), deadline=float("inf"))
    assert third["ready"] == 1  # changed since: tried again


def test_a_held_index_lock_stops_the_pass_and_refuses_nothing(tmp_path: Path) -> None:
    repo = _files(tmp_path, ["research/f-a.json"])

    def commit(repo: Path, paths: Sequence[str], message: str, deadline: float) -> None:
        raise OSError("git-commit-paths rc 1: index.lock exists")

    state: dict[str, Any] = {}
    with pytest.raises(OSError, match="index.lock"):
        lane_commit.step(repo, "d", ["d/research/f-a.json"], set(), state, commit=commit,
                         digest=lambda r, p: "h", deadline=float("inf"))
    assert not state.get("lane_refused")


def test_unfinished_jsonl_bytecode_and_deleted_files_are_never_committed(tmp_path: Path) -> None:
    repo = _files(tmp_path, ["cite/2026-10.jsonl", "adapters/__pycache__/f-a.cpython-312.pyc"])
    (repo / "d" / "cite" / "2026-10.jsonl").write_text('{"a": 1}\n{"b": 2}')
    found = lane_commit.plan(repo, "d", ["d/cite/2026-10.jsonl", "d/models/1/m9.json",
                                         "d/adapters/__pycache__/f-a.cpython-312.pyc"],
                             set(), {}, lambda r, p: None)
    assert found.ready == [] and found.gone == ["d/models/1/m9.json"]
    assert found.unreadable == {"d/cite/2026-10.jsonl": "its last line is unfinished"}
