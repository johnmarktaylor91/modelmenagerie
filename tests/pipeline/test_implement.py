"""The 2b verbs on the hub side: drafts, budgets, give-ups, done, src, the brief and the bridge.

Torch-free: runs go through a spool served by a stand-in thread, and the bridge talks to a fake
``ssh`` that runs each command locally, so the real forwarding code is exercised without a host.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, implement, ledger, read_json, write_json
from menagerie.pipeline.records import (
    GiveUp, Modes, ModelRecord, Packet, RecordError, RunRequest, RunResult,
)

FIXTURES = Path(__file__).with_name("fixtures")


def test_done_refuses_a_known_standin_revision(data: DataLayout) -> None:
    original = read_json(ModelRecord, data.model('m8839'))
    typed = replace(original, give_up=GiveUp(cls='standin:kwargs', reason='shrunken width',
                                            at='2026-10-08T00:00:00Z'))
    write_json(typed, data.model(typed.id))
    made = implement.Draft(original.recipe, None, original.source_case)
    with pytest.raises(RecordError, match='standin:kwargs'):
        implement._guard_standin(original.id, made, data)

FAMILY = "f-toy"
ADAPTER = '''"""A toy adapter."""
import torch

META = {"source_case": "reimplement"}
VARIANTS = {"m9001": {"width": 8}, "m9002": {"width": 16}, "m9003": {"width": 32}}


def build(cfg):
    return torch.nn.Linear(4, cfg["width"])
'''


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """A data root with the toy family (three members) and resnet18, and a private work root."""

    layout = DataLayout(tmp_path / "data")
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_LANE", "lane-a")
    monkeypatch.delenv("MENAGERIE_BRIDGE_SPOOL", raising=False)
    resnet = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    write_json(resnet, layout.model("m8839"))
    for n in ("m9001", "m9002", "m9003"):
        write_json(replace(resnet, id=n, name=f"toy-{n}", zoo="paper", family=FAMILY,
                           source_case="reimplement", recipe=None), layout.model(n))
    packet = read_json(Packet, FIXTURES / "f-resnet.packet.json")
    members = tuple(replace(packet.members[0], id=n) for n in ("m9001", "m9002", "m9003"))
    write_json(replace(packet, family=FAMILY, members=members), layout.packet(FAMILY))
    return layout


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict]]:
    seen: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(implement, "_event", lambda family, verb, **f: seen.append((family, verb, f)))
    return seen


def _draft_files(folder: Path, adapter: str = ADAPTER) -> Path:
    (folder / "adapters").mkdir(parents=True, exist_ok=True)
    (folder / "adapters" / f"{FAMILY}.py").write_text(adapter)
    recipe = folder / "recipe.json"
    recipe.write_text(json.dumps({
        "kind": "adapter", "env": "torch-main", "backend": "torch",
        "adapter": f"adapters/{FAMILY}.py", "config": {},
        "input": {"preset": "randn", "kind": "random-fallback", "shape": [1, 4]},
        "source_case": "reimplement"}))
    return recipe


def _result(request: RunRequest, outcome: str = "pass", cls: str | None = None,
            msg: str | None = None) -> RunResult:
    from menagerie.pipeline.records import LedgerRow

    row = LedgerRow(id=request.id, stage="forward", at="2026-10-04T00:00:00Z", host="worker",
                    env="torch-main@00000000", recipe_rev=request.recipe.rev, outcome=outcome,
                    cls=cls, mode="eval")
    return RunResult(request=request.request, id=request.id, mode="forward", outcome=outcome,
                     started="2026-10-04T00:00:00Z", ended="2026-10-04T00:00:02Z", host="worker",
                     cls=cls, msg=msg, output_shapes=((1, 8),) if outcome == "pass" else (),
                     modes=Modes(eval="pass", train="pass", divergence="none")
                     if outcome == "pass" else None, rows=(row,))


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Callable]]:
    """A stand-in runner serving ``tmp_path/spool``; push answer functions onto the list."""

    spool = tmp_path / "spool"
    monkeypatch.setenv("MENAGERIE_BRIDGE_SPOOL", str(spool))
    answers: list[Callable[[RunRequest], RunResult]] = []
    stop = threading.Event()

    def serve() -> None:
        for name in ("new", "done"):
            (spool / name).mkdir(parents=True, exist_ok=True)
        while not stop.is_set():
            (spool / "serving").touch()
            for path in sorted((spool / "new").glob("*.json")):
                request = read_json(RunRequest, path)
                path.unlink()
                answer = answers.pop(0) if answers else (lambda r: _result(r))
                write_json(answer(request), spool / "done" / path.name)
            time.sleep(0.05)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield answers
    stop.set()
    thread.join(timeout=5)


def _fail(cls: str, msg: str) -> Callable[[RunRequest], RunResult]:
    return lambda r: _result(r, "fail", cls, msg)


def test_draft_revision_follows_member_config_and_extras(data: DataLayout, tmp_path: Path) -> None:
    recipe = _draft_files(tmp_path / "lane")
    one = implement.draft(recipe, family=FAMILY, member="m9001")
    again = implement.draft(recipe, family=FAMILY, member="m9001")
    two = implement.draft(recipe, family=FAMILY, member="m9002")
    extra = implement.draft(recipe, family=FAMILY, member="m9001", extras=["tabulate==0.9.0"])
    assert one.recipe.rev == again.recipe.rev and one.recipe.config == {"width": 8}
    assert two.recipe.config == {"width": 16} and two.recipe.rev != one.recipe.rev
    assert extra.recipe.extras == ("tabulate==0.9.0",) and extra.recipe.rev != one.recipe.rev
    assert one.recipe.adapter == f"adapters/{FAMILY}.py" and one.adapter_src == ADAPTER
    assert one.source_case == "reimplement"
    with pytest.raises(RecordError, match="literal"):
        implement.variants("VARIANTS = dict(a=1)\n")


def test_try_through_the_spool_logs_and_charges(data: DataLayout, tmp_path: Path,
                                                served: list, events: list) -> None:
    recipe = _draft_files(tmp_path / "lane")
    served.append(lambda r: replace(_result(r, "infra", "infra:no_result", "died"), rows=()))
    result = implement.try_("m9001", recipe=recipe, data=data)
    assert result.outcome == "pass" and result.output_shapes == ((1, 8),)
    tries = implement._tries(FAMILY)
    assert [(t.charged, t.result.outcome) for t in tries] == [(True, "pass")]
    request_file = next((tmp_path / "spool" / "done").glob("*-r1.json"))
    assert request_file.is_file()  # the infra result was retried under a new name
    assert [e[1] for e in events] == ["try"] and events[0][2]["charged"] is True
    assert events[0][2]["class"] is None and events[0][2]["outcome"] == "pass"


def test_try_on_the_hub_goes_through_the_bridge_spool(data: DataLayout, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MENAGERIE_ENVS", raising=False)
    assert implement._bridge_spool(None) == tmp_path / "work" / "hubspool"
    with pytest.raises(RecordError, match="never here"):
        implement.run_all([implement._request("m9001", implement.draft(
            _draft_files(tmp_path / "lane"), family=FAMILY, member="m9001"))], spool=None)
    monkeypatch.setenv("MENAGERIE_ENVS", str(tmp_path / "envs"))
    assert implement._bridge_spool(None) is None  # a worker runs the attempt itself
    assert implement._bridge_spool(tmp_path / "s") == tmp_path / "s"


def test_third_identical_failure_says_give_up_then_refuses(data: DataLayout, tmp_path: Path,
                                                           served: list, events: list) -> None:
    recipe = _draft_files(tmp_path / "lane")
    for line in (12, 13, 14):  # the line number differs; the surface is the same
        served.append(_fail("construct_error", f"NameError: name 'x' is not defined (line {line})"))
    first = implement.try_("m9001", recipe=recipe, data=data)
    second = implement.try_("m9001", recipe=recipe, data=data)
    third = implement.try_("m9001", recipe=recipe, data=data)
    assert "give up" not in (first.msg or "") and "give up" not in (second.msg or "")
    assert third.cls == "construct_error" and third.msg.startswith("give up: 3 identical")
    refused = implement.try_("m9001", recipe=recipe, data=data)
    assert refused.cls == "budget" and refused.msg.startswith("give up")
    assert len(implement._tries(FAMILY)) == 3


def test_try_budgets_per_member_and_family_minutes(data: DataLayout, tmp_path: Path,
                                                   served: list, events: list) -> None:
    recipe = _draft_files(tmp_path / "lane")
    for n in range(10):
        served.append(_fail("forward_error", f"different error {'abcdefghij'[n]}"))
        assert implement.try_("m9001", recipe=recipe, data=data).cls == "forward_error"
    assert implement.try_("m9001", recipe=recipe, data=data).cls == "budget"
    for n in range(3):
        served.append(_fail("forward_error", f"later member {'xyz'[n]}"))
        implement.try_("m9001", recipe=recipe, member="m9002", data=data)
    assert "m9002 has used its 3 tries" in implement.try_(
        "m9001", recipe=recipe, member="m9002", data=data).msg
    index = implement._lane_dir() / "src" / FAMILY / "index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"at": "2026-01-01T00:00:00Z", "sources": []}))
    assert "45 minutes" in implement.try_("m9001", recipe=recipe, member="m9003", data=data).msg


def test_giveup_research_gap_needs_its_fields(data: DataLayout, tmp_path: Path,
                                              events: list) -> None:
    with pytest.raises(RecordError, match="command"):
        implement.giveup("m9001", "research_gap:config", "no config", data=data)
    with pytest.raises(RecordError, match="not a 2b give-up class"):
        implement.giveup("m9001", "fabrication", "x", data=data)
    reason = ("command: menagerie try m9001 --recipe r.json; missing: the hidden size of the "
              "decoder; searched: s1, s2")
    record = implement.giveup("m9001", "research_gap:hidden-size", reason, data=data)
    assert record.give_up.missing == "the hidden size of the decoder"
    assert record.give_up.searched == ("s1", "s2")
    assert record.give_up.by == "agent"  # the lanes' verb: an agent has tried this row
    assert read_json(ModelRecord, data.model("m9001")).give_up == record.give_up
    rows = [json.loads(line) for p in (data.root / "ledger").glob("*.jsonl")
            for line in p.read_text().splitlines()]
    assert [(r["outcome"], r["class"]) for r in rows] == [("giveup", "research_gap:hidden-size")]
    lane = tmp_path / "work" / "2b" / FAMILY
    homed = (f"command: {lane}/menagerie try m9001 --recipe {lane}/recipe.json; missing: a "
             f"config under {lane}; searched: s1")
    record = implement.giveup("m9001", "research_gap:config", homed, data=data)
    assert record.give_up.command == "menagerie try m9001 --recipe recipe.json"
    assert record.give_up.missing == "a config under ." and str(tmp_path) not in record.give_up.reason
    assert str(tmp_path) not in "".join(p.read_text() for p in (data.root / "ledger").glob("*"))


def test_giveup_types_a_degenerate_forward_no_init_change_repairs(data: DataLayout,
                                                                  events: list) -> None:
    with pytest.raises(RecordError, match="degenerate_forward:constant"):
        implement.giveup("m9001", "degenerate_forward:silent", "x", data=data)
    record = implement.giveup("m9001", "degenerate_forward:constant",
                              "LIF layers silent at random weights; only pretrained weights fire",
                              data=data)
    assert record.give_up.cls == "degenerate_forward:constant" and record.give_up.by == "agent"


def test_done_writes_records_adapter_ledger_and_runs_variants(data: DataLayout, tmp_path: Path,
                                                              served: list, events: list) -> None:
    recipe = _draft_files(tmp_path / "lane")
    with pytest.raises(RecordError, match="no passing try"):
        implement.done(recipe, data=data)
    implement.try_("m9001", recipe=recipe, data=data)
    served.append(lambda r: _result(r))  # m9002 passes
    served.append(_fail("forward_error", "RuntimeError: shape mismatch"))  # m9003 fails
    failures = implement.done(recipe, data=data)
    assert [r.id for r in failures] == ["m9003"]
    first = read_json(ModelRecord, data.model("m9001"))
    second = read_json(ModelRecord, data.model("m9002"))
    assert first.recipe.modes == Modes(eval="pass", train="pass", divergence="none")
    assert first.recipe.config == {"width": 8} and second.recipe.config == {"width": 16}
    assert read_json(ModelRecord, data.model("m9003")).recipe is None
    assert data.adapter(FAMILY).read_text() == ADAPTER
    rows = [json.loads(line) for p in (data.root / "ledger").glob("*.jsonl")
            for line in p.read_text().splitlines()]
    assert sorted(r["id"] for r in rows) == ["m9001", "m9002"]
    (tmp_path / "lane" / "adapters" / f"{FAMILY}.py").write_text(ADAPTER + "\n# edited\n")
    with pytest.raises(RecordError, match="no passing try"):
        implement.done(recipe, data=data)


def test_done_reads_only_its_familys_tries_for_a_draft_in_the_family_folder(
        data: DataLayout, tmp_path: Path, served: list, events: list,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.paths import work_dir

    recipe = _draft_files(work_dir("2b") / FAMILY)
    implement.try_("m9001", recipe=recipe, data=data)
    other = work_dir("lanes") / "lane-b" / "tries" / "f-other"
    other.mkdir(parents=True)
    passed = json.loads(next((work_dir("lanes") / "lane-a" / "tries" / FAMILY).glob("*.json")).read_text())
    (other / "m1-forward.json").write_text(json.dumps({**passed, "family": "f-other", "member": "m1"}))
    drafted = []
    real = implement.draft
    monkeypatch.setattr(implement, "draft", lambda path, **kw: drafted.append(kw["family"]) or real(path, **kw))
    attempt, _ = implement._passing(recipe)
    assert attempt.member == "m9001" and drafted == [FAMILY]
    (tmp_path / "lane").mkdir()
    elsewhere = shutil.copy(recipe, tmp_path / "lane" / "recipe.json")
    shutil.copytree(recipe.parent / "adapters", tmp_path / "lane" / "adapters")
    drafted.clear()
    implement._passing(Path(elsewhere))
    assert sorted(drafted) == sorted([FAMILY, "f-other"])


def test_done_names_the_members_outside_variants_it_did_not_run(
        data: DataLayout, tmp_path: Path, served: list, events: list,
        capsys: pytest.CaptureFixture[str]) -> None:
    """A library recipe has no VARIANTS: ``done`` used to print ``[]`` with the other members
    still owed, the lane ended there, and the tick launched the family again every pass."""

    only = ADAPTER.replace(', "m9002": {"width": 16}, "m9003": {"width": 32}', "")
    recipe = _draft_files(tmp_path / "lane", only)
    implement.try_("m9001", recipe=recipe, data=data)
    scripted = GiveUp(cls="standin:no_published_config", reason="unverified",
                      at="2026-10-09T20:24:24Z", by="script")
    third = read_json(ModelRecord, data.model("m9003"))
    write_json(replace(third, give_up=scripted), data.model("m9003"))
    left = implement.done(recipe, data=data)
    assert [r.id for r in left] == ["m9002", "m9003"]
    assert "m9002: not in VARIANTS, so done did not run it" in capsys.readouterr().err
    assert events[-1][2]["unrun"] == ["m9002", "m9003"]
    write_json(replace(third, give_up=replace(scripted, by="agent")), data.model("m9003"))
    assert [r.id for r in implement.done(recipe, data=data)] == ["m9002"]
    line = implement._members_line(["m9001", "m9002", "m9003"], tmp_path, data)
    assert line.startswith("Members: m9001, m9002, m9003. Try `m9002` first with `try m9002")
    assert line.endswith("leave their recipes as they are: m9001, m9003. Still owed: m9002.")


def _traced(data: DataLayout, member: str, outcome: str = "pass") -> None:
    from menagerie.pipeline.records import LedgerRow

    rev = read_json(ModelRecord, data.model(member)).recipe.rev
    ledger.append([LedgerRow(id=member, stage="trace", at="2026-10-21T08:00:00Z", host="worker",
                             env="torch-main", recipe_rev=rev, outcome=outcome, mode="eval",
                             capture="full", tl="2.36.0",
                             cls=None if outcome == "pass" else "torchlens_gap:replay")],
                  data=data)


def test_a_lane_leaves_a_validated_member_and_its_adapter_alone(
        data: DataLayout, tmp_path: Path, served: list, events: list,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Rung 3 lanes re-implemented rows that had validated at rung 0 (m8839 resnet18, m3638
    bert): each new recipe retired a passing trace until it traced again."""

    recipe = _draft_files(tmp_path / "lane")
    implement.try_("m9001", recipe=recipe, data=data)
    assert implement.done(recipe, data=data) == []
    _traced(data, "m9001")
    _traced(data, "m9002", outcome="fail")  # a failed trace does not protect a member
    assert sorted(implement.settled(["m9001", "m9002", "m9003"], data)) == ["m9001"]
    with pytest.raises(RecordError, match="m9001 is validated"):
        implement.try_("m9001", recipe=recipe, data=data)
    with pytest.raises(RecordError, match="m9001 is validated"):
        implement.giveup("m9001", "budget", "out of time", data=data)
    line = implement._members_line(["m9001", "m9002", "m9003"], tmp_path, data)
    assert "refuse them: m9001." in line and "Try `m9002` first" in line
    # The same adapter: done re-runs m9003 but leaves validated m9001 out of its sweep.
    implement.try_("m9002", recipe=recipe, member="m9002", data=data)
    capsys.readouterr()
    assert implement.done(recipe, data=data) == []
    assert "m9001: validated" in capsys.readouterr().err
    assert events[-1][2]["kept"] == ["m9001"] and "m9001" not in events[-1][2]["accepted"]
    # A changed adapter that m9001 runs is refused even as a family rewrite.
    edited = _draft_files(tmp_path / "lane", ADAPTER + "\n# edited\n")
    implement.try_("m9003", recipe=edited, member="m9003", data=data)
    with pytest.raises(RecordError, match="run by validated m9001"):
        implement.done(edited, data=data, rewrite_family=True)


def test_a_substitute_recipe_that_traced_is_not_settled(data: DataLayout, tmp_path: Path,
                                                        served: list, events: list) -> None:
    """m1778 (f-minkowski-res16unet34c): a dense stand-in passed its trace, so the lane was told
    to leave it alone while drive still owed it one real attempt, and the family stayed spent."""

    recipe = _draft_files(tmp_path / "lane")
    implement.try_("m9001", recipe=recipe, data=data)
    assert implement.done(recipe, data=data) == []
    _traced(data, "m9001")
    assert list(implement.settled(["m9001"], data)) == ["m9001"]
    path = data.model("m9001")
    record = read_json(ModelRecord, path)
    write_json(replace(record, give_up=GiveUp(cls="standin:dense", reason="dense stand-in",
                                              at="2026-10-06T03:59:14Z")), path)
    assert implement.settled(["m9001"], data) == {}


def test_src_materializes_sources_with_an_index(data: DataLayout, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch,
                                                events: list) -> None:
    from menagerie.pipeline import store

    pulled: list[str] = []

    def materialize(sha: str, dest: Path, *, store: StoreLayout) -> Path:
        pulled.append(sha)
        Path(dest).write_text(sha)
        return Path(dest)

    monkeypatch.setattr(store, "materialize", materialize)
    rows = implement.src(FAMILY, data=data, store=StoreLayout(tmp_path / "store"))
    by_sid = {r["sid"]: r for r in rows}
    assert by_sid["s1"]["path"].endswith("s1") and by_sid["s3"]["path"].endswith("s3.txt")
    assert by_sid["s2"]["path"] is None and by_sid["s2"]["symbol"] == "resnet18"
    index = json.loads((implement._lane_dir() / "src" / FAMILY / "index.json").read_text())
    assert [s["sid"] for s in index["sources"]] == ["s1", "s2", "s3"]
    implement.src(FAMILY, data=data, store=StoreLayout(tmp_path / "store"))
    assert len(pulled) == 2  # already materialized sources are not pulled again


def _put(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(dest.stem)
    return dest


def test_brief_holds_packet_inventory_budgets_and_prompt(data: DataLayout, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch,
                                                         events: list) -> None:
    from menagerie.pipeline import store

    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(store, "materialize", lambda sha, dest, *, store: _put(Path(dest)))
    for name in implement.DECLARATION:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RecordError, match="runner's declaration"):
        implement.brief(FAMILY, data=data)
    monkeypatch.setenv("MENAGERIE_CPUS", "3")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "20")
    path = implement.brief(FAMILY, data=data)
    assert sorted(p.name for p in implement._cache(FAMILY).iterdir()) == ["s1", "s3.txt"]
    monkeypatch.setattr(store, "materialize", lambda *a, **k: pytest.fail("the lane has no network"))
    rows = implement.src(FAMILY, data=data, store=StoreLayout(tmp_path / "store"))
    assert Path(rows[0]["path"]).is_symlink() and Path(rows[0]["path"]).read_text() == "s1"
    text = path.read_text()
    assert '"family": "f-toy"' in text and "torch-main@" in text and "10 tries" in text
    assert "## Give-up classes" in text and "## Native code" in text
    assert "`unverified` has not passed" in text and "`env_unresolved`" in text
    wrapper = path.with_name("menagerie")
    assert os.access(wrapper, os.X_OK) and str(data.root) in wrapper.read_text()
    exports = wrapper.read_text()
    assert "MENAGERIE_CPUS='3'" in exports and "MENAGERIE_MEMORY_GB='20'" in exports
    assert f"MENAGERIE_BRIDGE_SPOOL='{tmp_path / 'work' / 'hubspool'}'" in exports
    prompt = implement.PROMPT.read_text()
    assert len(prompt.splitlines()) < 150
    vendors = re.compile(r"(?i)\b(claude|anthropic|openai|codex|gpt-\d|gemini|opus|sonnet|fable)\b")
    assert not vendors.search(prompt)


FAKE_SSH = """#!/bin/sh
# Stand-in for ssh: a master (-M) just holds its control socket; a command runs locally.
master=0; ctl=""
while [ $# -gt 1 ]; do
  case "$1" in
    -M) master=1 ;;
    -S) shift; ctl="$1" ;;
    -o) shift ;;
    -N) ;;
    *) break ;;
  esac
  shift
done
if [ $master = 1 ]; then touch "$ctl"; exec sleep 600; fi
shift
cd "$FAKE_REMOTE_HOME" && exec sh -c "$1"
"""


def test_bridge_forwards_requests_and_results(data: DataLayout, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch,
                                              events: list) -> None:
    bin_dir, remote_home = tmp_path / "bin", tmp_path / "remote"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text(FAKE_SSH)
    (bin_dir / "ssh").chmod(stat.S_IRWXU)
    remote = remote_home / "scratch" / "menagerie" / "spool"
    (remote / "done").mkdir(parents=True)
    hub = tmp_path / "hubspool"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_REMOTE_HOME":
           str(remote_home), "MENAGERIE_RUNNER_HOST": "worker", "MENAGERIE_BRIDGE_SPOOL": str(hub),
           "PYTHONPATH": str(Path(__file__).parents[2])}
    proc = subprocess.Popen([sys.executable, "-m", "menagerie.pipeline", "bridge", "serve"],
                            env=env, stderr=subprocess.PIPE)
    try:
        monkeypatch.setenv("MENAGERIE_BRIDGE_SPOOL", str(hub))
        recipe = _draft_files(tmp_path / "lane")
        answered = threading.Thread(target=_answer_remote, args=(remote,), daemon=True)
        answered.start()
        result = implement.try_("m9001", recipe=recipe, data=data)
        assert result.outcome == "pass", result
        assert not list((hub / "claimed").glob("*.json"))
        forwarded = read_json(RunRequest, next((remote / "new").glob("*.json.seen")))
        assert forwarded.adapter_src == ADAPTER and forwarded.recipe.config == {"width": 8}
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    assert not (hub / ".ssh-control").exists()


def _answer_remote(remote: Path) -> None:
    """The remote runner: heartbeat, answer one request, keep it for inspection."""

    deadline = time.time() + 60
    while time.time() < deadline:
        (remote / "serving").touch()
        for path in (remote / "new").glob("*.json") if (remote / "new").is_dir() else ():
            request = read_json(RunRequest, path)
            write_json(_result(request), remote / "done" / path.name)
            shutil.move(path, path.with_name(path.name + ".seen"))
            return
        time.sleep(0.1)


# --- in the sandbox on the worker: real attempts through the runner ------------------------

ENVS = Path(os.environ.get("MENAGERIE_ENVS", "/nonexistent"))


def _built() -> bool:
    from menagerie.pipeline import envs

    try:
        envs.python("torch-main", root=ENVS)
    except FileNotFoundError:
        return False
    return shutil.which("bwrap") is not None


sandboxed = pytest.mark.skipif(not _built(), reason="needs MENAGERIE_ENVS with torch-main and bwrap")


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_CPUS", "2")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "8")


def _adapter_try(data: DataLayout, folder: Path, adapter: str, **recipe: object) -> RunResult:
    path = _draft_files(folder, adapter)
    raw = json.loads(path.read_text())
    path.write_text(json.dumps({**raw, **recipe}))
    return implement.try_("m9001", recipe=path, data=data)


@sandboxed
def test_library_recipe_passes_with_modes_and_boundary(data: DataLayout, worker_env: None,
                                                       events: list) -> None:
    started = time.monotonic()
    result = implement.try_("m8839", data=data)
    assert result.outcome == "pass", result.tail
    assert result.output_shapes == ((1, 1000),) and result.params == 11689512
    assert result.modes.train == "pass" and result.input_boundary == "real"
    assert time.monotonic() - started < 60


EXAMPLE_ADAPTER = ADAPTER + '''

def example_input(cfg):
    return {"input": torch.zeros(2, 4) + cfg["width"]}
'''


@sandboxed
def test_example_input_gets_the_member_config(data: DataLayout, tmp_path: Path, worker_env: None,
                                              events: list) -> None:
    example = {"preset": "example:menagerie_adapter:example_input", "kind": "random-fallback"}
    result = _adapter_try(data, tmp_path / "lane", EXAMPLE_ADAPTER, input=example)
    assert result.outcome == "pass", result.tail
    assert result.output_shapes == ((2, 8),)


@pytest.mark.parametrize("env", ["detectron2", "mmlab"])
def test_draft_takes_a_pixi_environment(data: DataLayout, tmp_path: Path, env: str) -> None:
    recipe = _draft_files(tmp_path / "lane")
    recipe.write_text(json.dumps({**json.loads(recipe.read_text()), "env": env}))
    made = implement.draft(recipe, family=FAMILY, member="m9001")
    assert made.recipe.env == env and re.fullmatch(r"[0-9a-f]{12}", made.recipe.rev)
    assert implement.draft(recipe, family=FAMILY, member="m9001").recipe.rev == made.recipe.rev


@sandboxed
def test_draft_adapter_runs_from_its_text(data: DataLayout, tmp_path: Path, worker_env: None,
                                          events: list) -> None:
    result = _adapter_try(data, tmp_path / "lane", ADAPTER)
    assert result.outcome == "pass", result.tail
    assert result.output_shapes == ((1, 8),) and result.modes.divergence == "none"
    broken = ADAPTER.replace("torch.nn.Linear(4", "torch.nn.Linear(undefined_name")
    failed = _adapter_try(data, tmp_path / "lane2", broken)
    assert failed.cls == "construct_error" and "undefined_name" in failed.msg
    assert any("undefined_name" in line for line in failed.tail)


VENDOR_PKG = '''import torch


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)

    def forward(self, x):
        return self.conv(x).mean((2, 3))
'''
VENDOR_ADAPTER = '''import socket

from toyvendor import Tiny

META = {"source_case": "vendor"}
VARIANTS = {"m9001": {}}


def build(cfg):
    if cfg.get("socket"):
        socket.create_connection(("1.1.1.1", 443), timeout=5)
    if cfg.get("write"):
        from menagerie.pipeline import harvest
        harvest.vendor_dir("toyvendor", "new.py").write_text("x = 1")
    return Tiny()
'''


@pytest.fixture
def vendor_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A saved code tree in a store; ``store.materialize`` copies it (the research lane unpacks)."""

    from menagerie.pipeline import store, tree_sha256

    tree = tmp_path / "tree" / "toyrepo-abc123" / "toyvendor"
    tree.mkdir(parents=True)
    (tree / "__init__.py").write_text(VENDOR_PKG)
    sha = tree_sha256(tmp_path / "tree")

    def materialize(sha256: str, dest: Path, *, store: StoreLayout) -> Path:
        shutil.copytree(tmp_path / "tree", dest)
        return dest

    monkeypatch.setattr(store, "materialize", materialize)
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    return sha


@sandboxed
def test_vendor_tree_is_importable_read_only_and_offline(data: DataLayout, tmp_path: Path,
                                                         worker_env: None, vendor_sha: str,
                                                         events: list) -> None:
    image = {"preset": "randn", "kind": "random-fallback", "shape": [1, 3, 16, 16]}
    passed = _adapter_try(data, tmp_path / "a", VENDOR_ADAPTER, vendor=vendor_sha, input=image)
    assert passed.outcome == "pass", passed.tail
    assert passed.output_shapes == ((1, 4),)
    socket_try = VENDOR_ADAPTER.replace('{"m9001": {}}', '{"m9001": {"socket": 1}}')
    offline = _adapter_try(data, tmp_path / "b", socket_try, vendor=vendor_sha, input=image)
    assert offline.cls == "construct_error" and "OSError" in offline.msg, offline.msg
    write_try = VENDOR_ADAPTER.replace('{"m9001": {}}', '{"m9001": {"write": 1}}')
    read_only = _adapter_try(data, tmp_path / "c", write_try, vendor=vendor_sha, input=image)
    assert read_only.cls == "construct_error" and "Read-only" in read_only.msg, read_only.msg


@sandboxed
def test_sdist_only_extra_is_refused_typed(data: DataLayout, tmp_path: Path, worker_env: None,
                                           events: list) -> None:
    result = _adapter_try(data, tmp_path / "lane", ADAPTER, extras=["docopt==0.6.2"])
    assert result.outcome == "fail" and result.cls == "missing_dependency:sdist", result.msg


CUSTOM_CONFIG = '''from transformers import PretrainedConfig


class ToyConfig(PretrainedConfig):
    model_type = "toy-custom"

    def __init__(self, hidden=8, **kwargs):
        self.hidden = hidden
        super().__init__(**kwargs)
'''
CUSTOM_MODEL = '''import torch
from transformers import PreTrainedModel

from .configuration_toy import ToyConfig


class ToyModel(PreTrainedModel):
    config_class = ToyConfig

    def __init__(self, config):
        super().__init__(config)
        self.proj = torch.nn.Linear(4, config.hidden)
        self.post_init()

    def forward(self, x):
        return self.proj(x)
'''
HUB_ADAPTER = '''from menagerie.pipeline import harvest

META = {"source_case": "library"}
VARIANTS = {"m9001": {"hidden": 6}}


def build(cfg):
    return harvest.hub_model(harvest.vendor_dir(), **cfg)
'''


@sandboxed
def test_hub_custom_code_builds_offline_from_a_snapshot(data: DataLayout, tmp_path: Path,
                                                        worker_env: None,
                                                        monkeypatch: pytest.MonkeyPatch,
                                                        events: list) -> None:
    from menagerie.pipeline import store, tree_sha256

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "configuration_toy.py").write_text(CUSTOM_CONFIG)
    (snap / "modeling_toy.py").write_text(CUSTOM_MODEL)
    (snap / "config.json").write_text(json.dumps({
        "model_type": "toy-custom", "hidden": 8, "auto_map": {
            "AutoConfig": "other/repo--configuration_toy.ToyConfig",
            "AutoModel": "modeling_toy.ToyModel"}}))
    monkeypatch.setattr(store, "materialize",
                        lambda sha, dest, *, store: Path(shutil.copytree(snap, dest)))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    result = _adapter_try(data, tmp_path / "lane", HUB_ADAPTER, vendor=tree_sha256(snap))
    assert result.outcome == "pass", result.tail
    assert result.output_shapes == ((1, 6),)


COMPILED_ADAPTER = '''from torch.utils.cpp_extension import load_inline

META = {"source_case": "vendor"}


def build(cfg):
    load_inline(name="broken_ext", cpp_sources="int f( { return 0; }", functions=["f"])
'''


@sandboxed
def test_failed_extension_build_in_the_sandbox_is_compiled_ext(data: DataLayout, tmp_path: Path,
                                                               worker_env: None,
                                                               events: list) -> None:
    result = _adapter_try(data, tmp_path / "lane", COMPILED_ADAPTER)
    assert result.cls == "compiled_ext", (result.cls, result.msg)


SLOW_ADAPTER = ADAPTER.replace("def build(cfg):\n", "def build(cfg):\n    import time\n"
                               "    time.sleep(15)\n")


@sandboxed
def test_runner_restart_mid_request_costs_no_try(data: DataLayout, tmp_path: Path,
                                                 worker_env: None,
                                                 monkeypatch: pytest.MonkeyPatch,
                                                 events: list) -> None:
    spool = tmp_path / "spool"
    monkeypatch.setenv("MENAGERIE_BRIDGE_SPOOL", str(spool))
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])}
    serve = [sys.executable, "-m", "menagerie.pipeline", "runner", "serve", "--spool", str(spool)]
    first, second = subprocess.Popen(serve, env=env), None
    out: list[RunResult] = []
    thread = threading.Thread(target=lambda: out.append(_adapter_try(
        data, tmp_path / "lane", SLOW_ADAPTER)), daemon=True)
    thread.start()
    try:
        deadline = time.time() + 60
        while not list(spool.glob("attempts/*/attempt.log")) and time.time() < deadline:
            time.sleep(0.2)
        time.sleep(2)
        first.kill()
        first.wait()
        (attempt,) = spool.glob("attempts/*")
        assert list(spool.glob("claimed/*.json"))
        second = subprocess.Popen(serve, env=env)
        thread.join(timeout=120)
    finally:
        for proc in (first, second):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait()
    (result,) = out
    assert result.outcome == "pass", result.tail
    infra = read_json(RunResult, attempt / "infra-1.json")
    assert infra.outcome == "infra" and infra.cls == "infra:restart"
    tries = implement._tries(FAMILY)
    assert [(t.charged, t.result.outcome) for t in tries] == [(True, "pass")]


def test_launch_records_the_plan_tier(data: DataLayout, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive

    seen: list[tuple] = []
    (tmp_path / "BRIEF.md").write_text("the brief\n")
    monkeypatch.setattr(implement, "brief", lambda family, *, data: tmp_path / "BRIEF.md")
    monkeypatch.setattr(drive, "launch", lambda kind, role, path, **kw: seen.append(
        (kind, role, kw["keys"], kw["web"])) or "run-2b")
    monkeypatch.setattr(drive, "event", lambda lane, family, verb, **f: seen.append(
        (lane, family, verb, f)))
    assert implement.launch(FAMILY, role="model-default", data=data, tier="native") == "run-2b"
    assert seen == [("2b", "model-default", (FAMILY,), False),
                    ("run-2b", FAMILY, "tier", {"tier": "native"})]


# --- rung 2: shared adapters (fix 16), repair briefs (fix 4), lane wrappers (fix 3), labs --------


def _member(data: DataLayout, member: str) -> None:
    """Add a new member to the toy family: a roster-less model record and a packet member."""

    record = read_json(ModelRecord, data.model("m9003"))
    write_json(replace(record, id=member, name=f"toy-{member}", recipe=None), data.model(member))
    packet = read_json(Packet, data.packet(FAMILY))
    added = replace(packet.members[0], id=member)
    write_json(replace(packet, members=(*packet.members, added)), data.packet(FAMILY))


def _own_draft(folder: Path, name: str, adapter: str) -> Path:
    recipe = _draft_files(folder, adapter)
    (folder / name).write_text(adapter)
    raw = json.loads(recipe.read_text())
    recipe.write_text(json.dumps({**raw, "adapter": name}))
    return recipe


def test_done_keeps_a_shared_adapter_unless_the_family_is_rewritten(
        data: DataLayout, tmp_path: Path, served: list, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _draft_files(tmp_path / "lane-a")
    implement.try_("m9001", recipe=first, data=data)
    assert implement.done(first, data=data) == []
    (tmp_path / "lane-a" / "adapters" / f"{FAMILY}.py").write_text(ADAPTER + "# same lane\n")
    implement.try_("m9001", recipe=first, data=data)
    assert implement.done(first, data=data) == []  # the lane's own members: allowed
    shared = data.adapter(FAMILY).read_text()
    _member(data, "m9004")
    monkeypatch.setenv("MENAGERIE_LANE", "lane-b")
    grown = ADAPTER.replace('"m9003": {"width": 32}}', '"m9003": {"width": 32}, "m9004": {"width": 64}}')
    edit = _draft_files(tmp_path / "lane-b", grown)
    implement.try_("m9004", recipe=edit, member="m9004", data=data)
    with pytest.raises(RecordError, match="is run by m9001, m9002, m9003"):
        implement.done(edit, data=data)
    assert data.adapter(FAMILY).read_text() == shared
    own = ('"""m9004 alone."""\nimport torch\n\nVARIANTS = {"m9004": {"width": 64}}\n\n\n'
           + ADAPTER.split("\n\n\n", 1)[1])
    alone = _own_draft(tmp_path / "lane-c", f"adapters/{FAMILY}--m9004.py", own)
    implement.try_("m9004", recipe=alone, member="m9004", data=data)
    assert implement.done(alone, data=data) == []
    record = read_json(ModelRecord, data.model("m9004"))
    assert record.recipe.adapter == f"adapters/{FAMILY}--m9004.py"
    assert (data.root / record.recipe.adapter).read_text() == own
    assert data.adapter(FAMILY).read_text() == shared
    dropped = _draft_files(tmp_path / "lane-d", ADAPTER.replace(', "m9003": {"width": 32}', ""))
    implement.try_("m9001", recipe=dropped, data=data)
    with pytest.raises(RecordError, match="m9003 are missing from the new VARIANTS"):
        implement.done(dropped, data=data, rewrite_family=True)
    before = read_json(ModelRecord, data.model("m9002")).recipe.rev
    implement.try_("m9004", recipe=edit, member="m9004", data=data)
    assert implement.done(edit, data=data, rewrite_family=True) == []
    assert read_json(ModelRecord, data.model("m9002")).recipe.rev != before
    lines = (tmp_path / "work" / "lanes" / "lane-b" / "events.jsonl").read_text().splitlines()
    (done,) = [e for e in map(json.loads, lines) if e["verb"] == "done" and e["rewrote"]]
    assert done["rewrote"] == ["m9001", "m9002", "m9003"]
    assert sorted(done["accepted"]) == ["m9001", "m9002", "m9003", "m9004"]


FINDINGS = "f" * 64
REVIEW = {"id": "m9001", "reviewer": "model-design-review", "verdict": "fabrication",
          "summary": "The gate is invented.", "components": [
              {"name": "encoder", "finding": "match", "source": "s", "ours": "o"},
              {"name": "gate", "finding": "invented", "source": "none", "ours": "a sigmoid gate"},
          ]}


def _reviewed(data: DataLayout, verdict: str) -> None:
    from menagerie.pipeline import ledger
    from menagerie.pipeline.records import LedgerRow

    resnet = read_json(ModelRecord, data.model("m8839"))
    record = replace(read_json(ModelRecord, data.model("m9001")), recipe=resnet.recipe)
    write_json(record, data.model("m9001"))
    ledger.append([LedgerRow(id="m9001", stage="review", at="2026-10-05T10:00:00Z", host="hub",
                             env="review:model-design-review", recipe_rev=record.recipe.rev,
                             outcome="pass", verdict=verdict, findings=FINDINGS,
                             reviewer="model-design-review")], data=data)


def test_a_repair_brief_carries_the_fabrication_findings_and_stays_on_its_lab(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list) -> None:
    from menagerie.pipeline import admission, drive, store

    def materialize(sha: str, dest: Path, *, store: StoreLayout) -> Path:
        if sha != FINDINGS:
            return _put(Path(dest))
        Path(dest).write_text(json.dumps(REVIEW))
        return Path(dest)

    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("MENAGERIE_CPUS", "3")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "20")
    monkeypatch.setattr(store, "materialize", materialize)
    assert "## Repair round" not in implement.brief(FAMILY, data=data).read_text()
    tried: list[str] = []

    def refuse(kind: str, role: str, brief: Path, **kw: object) -> str:
        tried.append(role)
        raise admission.Refused("no_admission", f"{role} full")

    monkeypatch.setattr(drive, "launch", refuse)
    assert implement.launch(FAMILY, role="model-default", data=data) == f"queued:2b-{FAMILY}"
    assert tried == ["model-default", "model-implementation", admission.THIRD_LAB_IMPLEMENT]
    _reviewed(data, "fabrication")
    text = implement.brief(FAMILY, data=data).read_text()
    assert "## Repair round (one)" in text and "The gate is invented." in text
    assert "- gate [invented]: source: none; ours: a sigmoid gate" in text
    assert "encoder [match]" not in text
    tried.clear()
    implement.launch(FAMILY, role="model-default", data=data)
    assert tried == ["model-default"]  # a repair round is the author's lab's
    (step,) = drive.queued()
    assert step["roles"] == ["model-default"]


def test_review_lanes_get_the_same_verb_wrapper(data: DataLayout, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import review, store

    monkeypatch.delenv("MENAGERIE_STORE", raising=False)
    with pytest.raises(RecordError, match="MENAGERIE_STORE"):
        implement.lane_wrapper(tmp_path / "lane", data=data)
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    wrapper = implement.lane_wrapper(tmp_path / "lane", data=data)
    text = wrapper.read_text()
    assert f"MENAGERIE_STORE='{tmp_path / 'store'}'" in text and sys.executable in text
    bare = {"PATH": "/nonexistent", "MENAGERIE_WORK_ROOT": str(tmp_path / "work")}
    done = subprocess.run([str(wrapper), "next", "m8839"], env=bare, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1])[0]["id"] == "m8839"
    write_json(read_json(Packet, FIXTURES / "f-resnet.packet.json"), data.packet("f-resnet"))
    monkeypatch.setattr(store, "materialize", lambda sha, dest, *, store: _put(Path(dest)))
    briefs: list[Path] = []
    monkeypatch.setattr(review.drive, "launch",
                        lambda kind, role, brief, **kw: briefs.append(brief) or "run-r")
    monkeypatch.setattr(review.calibration, "counts", lambda role, prompt, *, data: True)
    assert review.launch(["m8839"], data=data) == ["run-r"]
    lane_wrapper = briefs[0].with_name("menagerie")
    assert f"`{lane_wrapper} review record <case>" in briefs[0].read_text()
    assert os.access(lane_wrapper, os.X_OK)


def test_giveup_oversized_keeps_the_draft_for_the_cluster(data: DataLayout, tmp_path: Path,
                                                          events: list) -> None:
    """Fix 5: an oversized give-up keeps the lane's draft with placement axon-cpu (the give-up
    stays), so stage 3 has a recipe to capture structure from; needs_gpu keeps it for the GPU
    window; the draft's adapter becomes the family's, never over a different one."""

    with pytest.raises(RecordError, match="--recipe"):
        implement.giveup("m9001", "oversized", "memory_cap at 56 GiB", data=data)
    recipe = _draft_files(tmp_path / "lane")
    record = implement.giveup("m9001", "oversized", "memory_cap at 56 GiB", data=data,
                              recipe=recipe)
    assert record.give_up.cls == "oversized"
    assert record.recipe.placement == "axon-cpu" and record.recipe.config == {"width": 8}
    assert record.recipe.rev == implement.draft(recipe, family=FAMILY, member="m9001").recipe.rev
    assert read_json(ModelRecord, data.model("m9001")) == record
    assert data.adapter(FAMILY).read_text() == ADAPTER
    gpu = implement.giveup("m9002", "needs_gpu", "the kernel is CUDA only", data=data,
                           recipe=recipe)
    assert gpu.recipe.placement == "gpu-window" and gpu.recipe.config == {"width": 16}
    other = _draft_files(tmp_path / "other", ADAPTER.replace("Linear(4", "Linear(5"))
    with pytest.raises(RecordError, match="different adapter"):
        implement.giveup("m9003", "oversized", "too big", data=data, recipe=other)
    assert read_json(ModelRecord, data.model("m9003")).give_up is None

    plain = implement.giveup("m9003", "compiled_ext", "no fallback", data=data, recipe=recipe)
    assert plain.recipe is None  # other classes record no recipe


def test_giveup_standin_dense_keeps_the_recipe_and_refuses_other_details(data: DataLayout,
                                                                         events: list) -> None:
    """A passing recipe built on a hand-written equivalent of a library is typed after the
    fact; the record keeps its recipe (the export refuses it by the give-up)."""

    before = read_json(ModelRecord, data.model("m9001"))
    with pytest.raises(RecordError, match="needs one of"):
        implement.giveup("m9001", "standin:handmade", "not a known detail", data=data)
    record = implement.giveup("m9001", "standin:dense", "dense stand-in for a sparse library",
                              data=data)
    assert record.give_up.cls == "standin:dense" and record.recipe == before.recipe
    (row,) = [r for r in ledger.read_all(data=data) if r.id == "m9001" and r.cls == "standin:dense"]
    # The row types the accepted recipe's revision, never a later try's.
    assert row.recipe_rev == (before.recipe.rev if before.recipe else "0" * 12)


@pytest.fixture
def batch_data(data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    from menagerie.pipeline import store

    monkeypatch.setenv("MENAGERIE_CPUS", "3")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "20")
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(store, "materialize", lambda sha, dest, *, store: _put(Path(dest)))
    packet = read_json(Packet, data.packet(FAMILY))
    write_json(replace(packet, family="f-other"), data.packet("f-other"))
    return data


def test_batch_one_brief_is_original_bytes(batch_data: DataLayout, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = implement.PROMPT
    original = tmp_path / "original.md"
    original.write_text(prompt.read_text().split("\n## Batch\n", 1)[0])
    monkeypatch.setattr(implement, "PROMPT", original)
    expected = implement.brief(FAMILY, data=batch_data).read_bytes()
    wrapper = implement.brief(FAMILY, data=batch_data).with_name("menagerie").read_bytes()
    monkeypatch.setattr(implement, "PROMPT", prompt)
    path = implement.batch_brief([FAMILY], label="unused", data=batch_data)
    assert path.read_bytes() == expected
    assert path.with_name("menagerie").read_bytes() == wrapper


def test_batch_has_sections_leases_all_keys_and_preserves_timeout(
        batch_data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive

    calls = []
    monkeypatch.setattr(drive, "launch_first", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or "run-batch")
    assert implement.launch_batch([FAMILY, "f-other"], role="model-implementation",
                                  data=batch_data) == "run-batch"
    args, kwargs = calls[0]
    assert args[0] == "2b" and kwargs["keys"] == (FAMILY, "f-other")
    assert kwargs["on_start"] == [{"family": f, "verb": "batch:assigned"}
                                   for f in (FAMILY, "f-other")]
    assert kwargs["label"].startswith("2b-batch-") and kwargs["web"] is False
    text = args[2].read_text()
    assert text.count("## Packet") == 2 and text.count("## Sources (") == 2
    assert text.count("# 2b lane: implement") == 2
    assert "180 minutes" in text and "45 minutes per family" in text
    assert "10-minute" in text and "less than 20 minutes" in text
    assert drive.LANE_MINUTES["2b"] == 180 and implement.FAMILY_MINUTES == 45
    assert '"$1" = defer' in (implement.work_dir("2b") / FAMILY / "menagerie").read_text()


def test_single_batch_uses_original_launcher(batch_data: DataLayout,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(implement, "launch", lambda family, **kw:
                        calls.append((family, kw)) or "run-single")
    assert implement.launch_batch([FAMILY], role="r", data=batch_data) == "run-single"
    assert calls == [(FAMILY, {"role": "r", "data": batch_data, "queue": True})]


@pytest.mark.parametrize("families", [[], [FAMILY, FAMILY], ["../bad"],
                                      [f"f-{i}" for i in range(7)]])
def test_batch_refuses_invalid_keys_before_launch(data: DataLayout, families: list[str]) -> None:
    with pytest.raises(RecordError):
        implement.launch_batch(families, role="r", data=data)


def test_batch_groups_use_same_case_and_stable_order(batch_data: DataLayout) -> None:
    packet = read_json(Packet, batch_data.packet(FAMILY))
    write_json(replace(packet, family="f-paper", code_order=()), batch_data.packet("f-paper"))
    assert implement.batch_groups([FAMILY, "f-paper", "f-other"], 5, data=batch_data) == [
        [FAMILY, "f-other"], ["f-paper"]]
    with pytest.raises(RecordError, match="source case"):
        implement.launch_batch([FAMILY, "f-paper"], role="r", data=batch_data)
    with pytest.raises(RecordError, match="families_per"):
        implement.batch_groups([FAMILY], 3, data=batch_data)


def test_deferred_family_keeps_draft_returns_to_2b_and_is_never_typed(
        data: DataLayout) -> None:
    from menagerie.pipeline import drive

    folder = implement.work_dir("2b") / FAMILY
    recipe = _draft_files(folder)
    before = recipe.read_bytes()
    records = {m: data.model(m).read_bytes() for m in ("m9001", "m9002", "m9003")}
    implement.defer(FAMILY, reason="lane reserve reached", data=data)
    assert recipe.read_bytes() == before
    assert records == {m: data.model(m).read_bytes() for m in records}
    assert ledger.read_all(data=data) == []
    steps = drive.next_steps(list(records), data=data)
    assert all(s["step"] == "2b" and "2b:deferred" in s["why"] for s in steps)
    events = list(drive._read_lines(implement._lane_dir() / "events.jsonl"))
    assert events[-1]["verb"] == "2b:deferred"
    assert events[-1]["pending"] == list(records)
    assert drive._give_up([], events) == ""


def test_deferred_family_is_retried_alone(batch_data: DataLayout) -> None:
    implement.defer(FAMILY, reason="lane reserve reached", data=batch_data)
    assert implement.batch_groups(["f-other", FAMILY], 5, data=batch_data) == [
        [FAMILY], ["f-other"]]


def test_batch_defer_wrapper_rejects_missing_reason_cleanly(batch_data: DataLayout) -> None:
    implement.batch_brief([FAMILY, "f-other"], label="wrapper-test", data=batch_data)
    wrapper = implement.work_dir("2b") / FAMILY / "menagerie"
    result = subprocess.run([str(wrapper), "defer", FAMILY], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 2 and "reason" in result.stderr
    assert "Traceback" not in result.stderr


def test_batch_measures_split_family_event_minutes(data: DataLayout,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive

    events = [
        {"family": FAMILY, "verb": "src", "at": "2026-10-09T00:01:00Z"},
        {"family": FAMILY, "verb": "done", "at": "2026-10-09T00:06:00Z"},
        {"family": "f-other", "verb": "src", "at": "2026-10-09T00:04:00Z"},
        {"family": "f-other", "verb": "2b:deferred", "at": "2026-10-09T00:12:00Z"},
    ]
    lane = drive._Lane({"run_id": "r", "state": "DONE"},
                       {"kind": "2b", "keys": [FAMILY, "f-other"]}, events, [], [])
    monkeypatch.setattr(drive, "_runs_under", lambda run: [lane.run])
    monkeypatch.setattr(drive, "_lane", lambda run: lane)
    out = drive.measures("r", data=data)
    rows = {r["family"]: r for r in out["families"]}
    assert rows[FAMILY]["min_2b"] == 5 and rows["f-other"]["min_2b"] == 8
    assert rows["f-other"]["giveup"] == "" and rows["f-other"]["deferred"] == 1


def test_batch_refuses_unadmitted_trial_without_queuing(batch_data: DataLayout,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive

    calls = []
    monkeypatch.setattr(drive, "launch_first", lambda *args, **kwargs:
                        calls.append(kwargs) or "r")
    implement.launch_batch([FAMILY, "f-other"], role="r", data=batch_data, queue=False)
    assert calls[0]["queue"] is False


def test_batch_unstarted_family_minutes_are_unknown(data: DataLayout) -> None:
    from menagerie.pipeline import drive

    lane = drive._Lane({}, {"kind": "2b", "keys": [FAMILY, "f-other"]},
                       [{"family": FAMILY, "verb": "tier", "tier": "library",
                         "at": "2026-10-09T00:01:00Z"}], None, [])
    assert drive._minutes([lane], FAMILY)["min_2b"] is None


def test_batch_assignment_does_not_expand_family_token_windows() -> None:
    from menagerie.pipeline import drive
    events = [
        {"at": "2026-01-01T00:00:00Z", "family": f, "verb": "batch:assigned"}
        for f in (FAMILY, "f-other")]
    events += [{"at": f"2026-01-01T00:{minute}:00Z", "family": f, "verb": verb}
               for f, minute, verb in ((FAMILY, "01", "src"), (FAMILY, "05", "done"),
                                       ("f-other", "06", "src"), ("f-other", "10", "done"))]
    usage = [(drive._parse("2026-01-01T00:03:00Z"), 100, 20, 10),
             (drive._parse("2026-01-01T00:08:00Z"), 200, 30, 40)]
    lane = drive._Lane({}, {"kind": "2b", "keys": [FAMILY, "f-other"]}, events, usage, [])
    assert drive._family_usage(lane, FAMILY) == usage[:1]
    assert drive._family_usage(lane, "f-other") == usage[1:]
