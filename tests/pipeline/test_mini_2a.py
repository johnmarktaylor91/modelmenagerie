"""A 2a lane on a worker: only the 2a verbs, this lane's families, and the blobs its fetches print."""

from pathlib import Path

import pytest

from menagerie.pipeline import mini_2a, mini_lane

FAMILIES = ("f-one", "f-two")


def _authorize(args: list[str], tmp_path: Path) -> mini_2a.Plan:
    (tmp_path / "drafts").mkdir(exist_ok=True)
    return mini_2a.authorize(args, FAMILIES, tmp_path / "drafts", str(tmp_path))


def test_the_2a_verbs_run_with_their_own_options_and_this_lanes_families(tmp_path: Path) -> None:
    assert _authorize(["fetch", "paper", "2401.00001"], tmp_path).argv == ["fetch", "paper", "2401.00001"]
    plan = _authorize(["fetch", "code", "https://github.com/a/b", "--rev=abc", "--family", "f-two"],
                      tmp_path)
    assert plan.argv == ["fetch", "code", "https://github.com/a/b", "--rev", "abc", "--family", "f-two"]
    assert plan.fetch and not plan.gh and plan.draft is None
    assert _authorize(["family", "seed", "m1", "m22"], tmp_path).argv == ["family", "seed", "m1", "m22"]
    assert _authorize(["research", "second-search", "f-one"], tmp_path).argv[-1] == "f-one"
    plan = _authorize(["packet", "write", "f-one", "drafts/f-one.json"], tmp_path)
    assert plan.argv == ["packet", "write", "f-one", str((tmp_path / "drafts/f-one.json").resolve())]
    assert plan.draft == plan.argv[-1] and not plan.fetch
    plan = _authorize(["gh", "search", "code", "class", "ResNet", "--limit", "5", "--json=path"], tmp_path)
    assert plan.gh and plan.argv == ["gh", "search", "code", "class", "ResNet", "--limit", "5",
                                     "--json", "path"]


@pytest.mark.parametrize(("args", "why"), [
    (["try", "m1"], "verbs here"),
    (["store", "put", "x"], "verbs here"),
    (["fetch", "page", "https://x.org"], "needs --family"),
    (["fetch", "hf", "a/b", "--family", "f-other"], "not one of this lane's families"),
    (["fetch", "paper", "x", "--family", "f-one"], "does not take --family"),
    (["fetch", "paper", "--rev"], "does not take --rev"),
    (["fetch", "paper", "a", "b"], "takes 1 argument"),
    (["family", "seed", "f-one"], "roster ids"),
    (["packet", "write", "f-other", "drafts/x.json"], "not one of this lane's families"),
    (["packet", "write", "f-one", "../escape.json"], "drafts folder"),
    (["packet", "write", "f-one", "drafts/notes.txt"], "drafts folder"),
    (["research", "second-search", "f-one", "--family=f-one"], "does not take --family"),
    (["gh", "api", "repos/a/b"], "gh search repos|code"),
    (["gh", "search", "issues", "x"], "gh search repos|code"),
    (["gh", "search", "repos", "x", "--web"], "does not take --web"),
    (["gh", "search", "repos", "--limit", "5"], "needs a query"),
])
def test_anything_else_is_refused(args: list[str], why: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=why):
        _authorize(args, tmp_path)


def test_a_symlinked_draft_cannot_leave_the_drafts_folder(tmp_path: Path) -> None:
    (tmp_path / "drafts").mkdir()
    (tmp_path / "outside.json").write_text("{}")
    (tmp_path / "drafts/link.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="drafts folder"):
        _authorize(["packet", "write", "f-one", "drafts/link.json"], tmp_path)


def test_fetches_send_the_blobs_they_printed_within_the_caps(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    store = tmp_path / "store"
    shas = ["a" * 64, "b" * 64, "c" * 64]
    for sha, size in zip(shas, (10, 30, 5)):
        (store / "sha256" / sha[:2]).mkdir(parents=True, exist_ok=True)
        (store / "sha256" / sha[:2] / sha).write_bytes(b"x" * size)
    monkeypatch.setattr(mini_2a, "BLOB_MAX", 20)
    text = f"{shas[0]} page\n{shas[1]} pdf\n{shas[0]} again\n{'d' * 64} missing\n{shas[2]}\n{'e' * 65}"
    assert mini_2a.blobs(text, str(store)) == [f"sha256/aa/{shas[0]}", f"sha256/cc/{shas[2]}"]
    assert mini_2a.blobs(text, None) == []


def test_the_switch_is_off_unless_true_and_the_lane_folder_is_a_brief_folder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    assert not mini_2a.enabled({}) and not mini_2a.enabled({"mini_2a": "yes"})
    assert mini_2a.enabled({"mini_2a": True})
    brief = tmp_path / "briefs/2a-tick-1/BRIEF.md"
    mapped = mini_2a.mapping("/w/.mini-lane", brief, "cd /repo && python3 -m menagerie.pipeline", "/st")
    assert mapped == {str(brief.parent): "/w/.mini-lane/work/2a-tick-1",
                      "cd /repo && python3 -m menagerie.pipeline": "/w/.mini-lane/work/2a-tick-1/menagerie",
                      "/st": "/w/.mini-lane/store"}
    with pytest.raises(ValueError, match="briefs work root"):
        mini_2a.folder_of(tmp_path / "2b/f-one/BRIEF.md")


def test_a_2a_lane_keeps_its_web_tools_and_gets_a_gh_wrapper(tmp_path: Path) -> None:
    spec = {"runtime": ["claude"], "label": "2a-mini-x", "account": "a", "model": "m", "effort": "high"}
    assert "WebFetch,WebSearch,mcp__executor" in mini_lane.agent_argv(spec)
    argv = mini_lane.agent_argv({**spec, "web": True})
    assert argv[argv.index("--disallowedTools") + 1] == "mcp__executor"
    path = mini_lane.write_2a_wrappers(tmp_path, "2a-tick-1")
    assert path == tmp_path / "bin"
    assert (tmp_path / "bin/gh").read_text().rstrip().endswith("'2a-tick-1' gh \"$@\"")
    assert (tmp_path / "work/2a-tick-1/menagerie").read_text().rstrip().endswith("'2a-tick-1' \"$@\"")


def test_finish_returns_a_2a_lanes_drafts(tmp_path: Path) -> None:
    root, out = tmp_path / ".mini-lane", tmp_path / "out"
    (root / "work/2a-tick-1/drafts").mkdir(parents=True)
    (root / "work/2a-tick-1/drafts/f-one.json").write_text("{}")
    (root / "work/2a-tick-1/BRIEF.result.md").write_text("f-one packet written\n")
    out.mkdir()
    spec = {"kind": "2a", "label": "2a-mini-x", "families": ["f-one"],
            "brief": "work/2a-tick-1/BRIEF.md", "result": "work/2a-tick-1/BRIEF.result.md"}
    assert mini_lane.finish(root, out, spec, 0)
    assert (out / "drafts/f-one.json").read_text() == "{}"
    assert (out / "result.md").read_text() == "f-one packet written\n"
