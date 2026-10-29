"""The tick commits exactly what it wrote, on a real temporary git checkout."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from menagerie.pipeline import append_jsonl, tick_commit, write_json, written
from menagerie.pipeline.records import StoreIndexLine

BASE = '{"n":1}\n'
OTHER = '{"other":1}\n'


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout


def line(n: int) -> StoreIndexLine:
    return StoreIndexLine(sha256=f"{n:064x}", bytes=n, mime="application/json",
                          fetched="2026-10-10T00:00:00Z", license_class="third-party")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = Tick Test\n\temail = tick@example.invalid\n"
                   "[init]\n\tdefaultBranch = main\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repo"
    (root / "d").mkdir(parents=True)
    git(root, "init", "-q")
    (root / "d" / "index.jsonl").write_text(BASE)
    (root / "d" / "p.json").write_text('{"v":1}\n')
    git(root, "add", "d/index.jsonl", "d/p.json")
    git(root, "commit", "-qm", "base")
    return root.resolve()


def _commit(repo: Path, owned: tick_commit.Owned) -> tick_commit.Outcome:
    return tick_commit.commit_owned(repo, owned, "data: tick test", time.monotonic() + 60)


def test_own_lines_are_committed_beside_another_writers_uncommitted_lines(repo: Path) -> None:
    index = repo / "d" / "index.jsonl"
    with index.open("a") as handle:
        handle.write(OTHER)  # another writer's line, left uncommitted
    with written.recording() as rec:
        append_jsonl([line(1), line(2)], index)
    ours = index.read_text()[len(BASE + OTHER):]
    out = _commit(repo, tick_commit.Owned.from_writes(rec, repo))
    assert out.committed == ["d/index.jsonl"] and not out.held and not out.dropped
    assert git(repo, "show", "HEAD:d/index.jsonl") == BASE + ours
    assert index.read_text() == BASE + OTHER + ours  # the working tree is untouched
    assert git(repo, "diff", "--cached") == ""  # the shared index follows the commit
    assert [x for x in git(repo, "diff", "-U0").splitlines() if x.startswith("+{")] == ["+" + OTHER.strip()]
    again = _commit(repo, tick_commit.Owned.from_writes(rec, repo))
    assert again.commit is None and again.done == ["d/index.jsonl"]


def test_a_rewritten_file_is_held_while_another_writers_edit_is_inside_then_committed(
        repo: Path) -> None:
    packet = repo / "d" / "p.json"
    packet.write_text('{"v":2}\n')  # another writer's uncommitted edit
    with written.recording() as rec:
        write_json(line(3), packet)  # the tick rewrites the file on top of it
    out = _commit(repo, tick_commit.Owned.from_writes(rec, repo))
    assert out.commit is None and "before the tick wrote it" in out.held["d/p.json"]
    assert git(repo, "show", "HEAD:d/p.json") == '{"v":1}\n' and out.pending.files
    blob = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input='{"v":2}\n',
                          check=True, capture_output=True, text=True).stdout.strip()
    git(repo, "update-index", "--cacheinfo", f"100644,{blob},d/p.json")
    git(repo, "commit", "-qm", "the other writer commits its own edit")
    later = _commit(repo, out.pending)
    assert later.committed == ["d/p.json"] and git(repo, "show", "HEAD:d/p.json") == packet.read_text()
    assert git(repo, "status", "--porcelain") == ""


def test_a_clean_file_is_committed_and_one_changed_since_is_dropped(repo: Path) -> None:
    packet = repo / "d" / "p.json"
    with written.recording() as rec:
        write_json(line(4), packet)
    owned = tick_commit.Owned.from_writes(rec, repo)
    packet.write_text('{"v":9}\n')  # someone else changes it after the tick wrote it
    out = _commit(repo, owned)
    assert out.commit is None and "after the tick wrote it" in out.dropped["d/p.json"]
    assert not out.pending
    git(repo, "checkout", "--", "d/p.json")  # the other edit is undone; the tick writes again
    with written.recording() as rec:
        write_json(line(4), packet)
    out = _commit(repo, tick_commit.Owned.from_writes(rec, repo))
    assert out.committed == ["d/p.json"] and git(repo, "status", "--porcelain") == ""


def test_rewritten_committed_lines_hold_and_removed_own_lines_drop(repo: Path) -> None:
    index = repo / "d" / "index.jsonl"
    with written.recording() as rec:
        append_jsonl([line(5)], index)
    owned = tick_commit.Owned.from_writes(rec, repo)
    ours = index.read_text()[len(BASE):]
    index.write_text('{"n":0}\n' + ours)  # another writer rewrote a committed line
    out = _commit(repo, owned)
    assert out.commit is None and "committed lines" in out.held["d/index.jsonl"]
    assert out.pending.lines == owned.lines
    index.write_text(BASE)  # and now the tick's line is gone
    out = _commit(repo, owned)
    assert "no longer in the file" in out.dropped["d/index.jsonl"] and not out.pending


def test_a_commit_landing_meanwhile_is_kept(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = repo / "d" / "index.jsonl"
    with written.recording() as rec:
        append_jsonl([line(6)], index)
    real, raced = tick_commit._build, []

    def racing(*args: object) -> str | None:
        built = real(*args)  # type: ignore[arg-type]
        if not raced:
            raced.append(1)
            (repo / "other.txt").write_text("x\n")
            git(repo, "add", "other.txt")
            git(repo, "commit", "-qm", "another session")
        return built

    monkeypatch.setattr(tick_commit, "_build", racing)
    out = _commit(repo, tick_commit.Owned.from_writes(rec, repo))
    assert out.commit and git(repo, "log", "-1", "--format=%s", "HEAD~1").strip() == "another session"
    assert git(repo, "show", "HEAD:other.txt") == "x\n"
    assert git(repo, "show", "HEAD:d/index.jsonl") == index.read_text()


def test_hooks_run_a_refusing_pre_commit_stops_the_commit(repo: Path) -> None:
    index = repo / "d" / "index.jsonl"
    with written.recording() as rec:
        append_jsonl([line(7)], index)
    hook = repo / ".git" / "hooks" / "commit-msg"
    hook.write_text('#!/bin/sh\necho "checked" >> "$1"\n')
    hook.chmod(0o755)
    out = _commit(repo, tick_commit.Owned.from_writes(rec, repo))
    assert out.commit and "checked" in git(repo, "log", "-1", "--format=%B")
    with written.recording() as rec:
        append_jsonl([line(8)], index)
    refuse = repo / ".git" / "hooks" / "pre-commit"
    refuse.write_text("#!/bin/sh\necho no >&2\nexit 1\n")
    refuse.chmod(0o755)
    head = git(repo, "rev-parse", "HEAD")
    with pytest.raises(OSError, match="pre-commit"):
        _commit(repo, tick_commit.Owned.from_writes(rec, repo))
    assert git(repo, "rev-parse", "HEAD") == head


def test_owned_survives_the_state_round_trip_and_merges_without_repeats() -> None:
    a = tick_commit.Owned({"x.jsonl": ["1\n"]}, {"p.json": [None, "h1"]})
    b = tick_commit.Owned({"x.jsonl": ["1\n", "2\n"]}, {"p.json": ["h1", "h2"]})
    merged = tick_commit.Owned.from_mapping(a.to_mapping()).merged(b)
    assert merged.lines == {"x.jsonl": ["1\n", "2\n"]} and merged.files == {"p.json": [None, "h2"]}
    assert tick_commit.Owned().to_mapping() == {} and not tick_commit.Owned.from_mapping(None)


def test_a_second_partial_commit_works_past_another_writers_line(repo: Path) -> None:
    index = repo / "d" / "index.jsonl"
    with index.open("a") as handle:
        handle.write(OTHER)
    with written.recording() as first:
        append_jsonl([line(10)], index)
    assert _commit(repo, tick_commit.Owned.from_writes(first, repo)).commit
    with written.recording() as second:  # the working tree is now base, other, tick 1, tick 2
        append_jsonl([line(11)], index)
    out = _commit(repo, tick_commit.Owned.from_writes(second, repo))
    assert out.committed == ["d/index.jsonl"] and not out.held
    head = git(repo, "show", "HEAD:d/index.jsonl")
    assert OTHER not in head and head.count("\n") == 3


def test_a_file_changed_between_two_tick_writes_is_held(repo: Path) -> None:
    packet = repo / "d" / "p.json"
    with written.recording() as rec:
        write_json(line(12), packet)
        packet.write_text(packet.read_text() + " ")  # another writer edits between the two writes
        write_json(line(13), packet)
    out = _commit(repo, tick_commit.Owned.from_writes(rec, repo))
    assert out.commit is None and "d/p.json" in out.held
    merged = tick_commit.Owned(files={"p": ["h0", "h1"]}).merged(tick_commit.Owned(files={"p": ["h9", "h2"]}))
    assert merged.files["p"] == [written.BROKEN, "h2"]


def test_a_held_index_lock_is_waited_out_never_removed(repo: Path) -> None:
    index = repo / "d" / "index.jsonl"
    with written.recording() as rec:
        append_jsonl([line(14)], index)
    lock = repo / ".git" / "index.lock"
    lock.write_text("")
    head = git(repo, "rev-parse", "HEAD")
    with pytest.raises(OSError, match="index.lock"):
        tick_commit.commit_owned(repo, tick_commit.Owned.from_writes(rec, repo), "data: t",
                                 time.monotonic() + 3)
    assert git(repo, "rev-parse", "HEAD") == head and lock.exists()
    lock.unlink()
    assert _commit(repo, tick_commit.Owned.from_writes(rec, repo)).commit


def test_a_path_someone_staged_is_held_until_it_is_no_longer_staged(repo: Path) -> None:
    index = repo / "d" / "index.jsonl"
    with index.open("a") as handle:
        handle.write(OTHER)
    git(repo, "add", "d/index.jsonl")  # another writer stages base plus its line
    with written.recording() as rec:
        append_jsonl([line(15)], index)
    owned = tick_commit.Owned.from_writes(rec, repo)
    head = git(repo, "rev-parse", "HEAD")
    out = _commit(repo, owned)
    assert out.commit is None and "staged" in out.held["d/index.jsonl"] and out.pending.lines == owned.lines
    assert git(repo, "rev-parse", "HEAD") == head
    git(repo, "reset", "-q", "--", "d/index.jsonl")
    out = _commit(repo, out.pending)
    assert out.committed == ["d/index.jsonl"] and OTHER not in git(repo, "show", "HEAD:d/index.jsonl")
    assert git(repo, "diff", "--cached") == ""
