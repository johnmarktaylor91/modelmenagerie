"""Mini recipe lanes: the worker never reaches the hub; the hub stages, answers verbs and pulls."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from menagerie.paths import work_dir
from menagerie.pipeline import admission, mini_lane, mini_relay, mini_verbs

LABEL = "2b-mini-test"
ACCOUNT = "test-account"
LANE_FILE = Path(mini_lane.__file__).resolve()


class LocalRemote(mini_relay.Remote):
    """The worker as a local folder: commands run with HOME set to it, rsync copies locally."""

    def __init__(self, home: Path) -> None:
        self.home, self.alias, self.lock = home, "worker-a-ssh", threading.Lock()

    def run(self, command: str, *, data: bytes | None = None) -> bytes:
        return subprocess.run(["sh", "-c", command], cwd=self.home, input=data, check=True,
                              env={**os.environ, "HOME": str(self.home)}, capture_output=True,
                              timeout=60).stdout

    def target(self, path: str) -> str:
        return path

    def rsync(self, options: Any, source: str, dest: str) -> None:
        subprocess.run(["rsync", *options, source, dest], check=True, capture_output=True, timeout=60)

    def close(self) -> None:
        pass


needs_rsync = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")


def _script(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


# --- the worker side ------------------------------------------------------------------------


def test_verb_without_an_answer_is_retryable(tmp_path: Path) -> None:
    assert mini_lane.verb(tmp_path, "f-one", ["src", "f-one"], wait=0.2) == 75
    request, = (tmp_path / "spool/new").glob("*.json")
    body = json.loads(request.read_text())
    assert body["family"] == "f-one" and body["args"] == ["src", "f-one"]
    assert mini_relay.NAME.fullmatch(request.stem)


def test_verb_prints_the_hub_answer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def answer() -> None:
        while not list((tmp_path / "spool/new").glob("*.json")):
            time.sleep(0.05)
        name = next((tmp_path / "spool/new").glob("*.json")).name
        (tmp_path / "spool/done").mkdir(parents=True)
        (tmp_path / "spool/done" / name).write_text('{"rc": 3, "stdout": "out\\n", "stderr": ""}')

    threading.Thread(target=answer).start()
    assert mini_lane.verb(tmp_path, "f-one", ["done", "recipe.json"], wait=10) == 3
    assert capsys.readouterr().out == "out\n"


def _drafts(folder: Path) -> None:
    for name in ("recipe.json", "adapters/f-one.py", "sources/code.py", "tries/1.json",
                 "weights.pt", "menagerie", "BRIEF.md", "deferred.json"):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text("x")


def test_finish_pulls_small_drafts_only(tmp_path: Path) -> None:
    _drafts(tmp_path / ".mini-lane/work/f-one")
    spec = {"label": LABEL, "families": ["f-one"], "result": "work/f-one/BRIEF.result.md"}
    out = tmp_path / "outputs/lane"
    assert mini_lane.finish(tmp_path / ".mini-lane", out, spec, 0) is False
    kept = sorted(str(p.relative_to(out / "families/f-one")) for p in out.rglob("*") if p.is_file()
                  and "families" in p.parts)
    assert kept == ["adapters/f-one.py", "recipe.json"]
    assert json.loads((out / "lane.json").read_text())["skipped"] == {"f-one": ["weights.pt"]}


def test_finish_leaves_out_the_sandbox_s_empty_stub_files(tmp_path: Path) -> None:
    work = tmp_path / ".mini-lane/work/f-one"
    _drafts(work)
    for name in (".env", ".env.local", "package.json", "yarn.lock", "scripts", ".bashrc", ".vscode"):
        (work / name).write_bytes(b"")
    (work / "adapters/__init__.py").write_bytes(b"")  # an empty file the agent wrote is a draft
    spec = {"label": LABEL, "families": ["f-one"], "result": "work/f-one/BRIEF.result.md"}
    out = tmp_path / "outputs/lane"
    mini_lane.finish(tmp_path / ".mini-lane", out, spec, 0)
    kept = sorted(str(p.relative_to(out / "families/f-one")) for p in out.rglob("*") if p.is_file()
                  and "families" in p.parts)
    assert kept == ["adapters/__init__.py", "adapters/f-one.py", "recipe.json"]


def test_snapshot_copies_the_same_drafts_and_bounds_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / ".mini-lane"
    _drafts(root / "work/f-one")
    assert mini_lane.snapshot(root, "1-0123abcd", "f-one")["files"] == 2
    assert sorted(p.name for p in (root / "snap/1-0123abcd/f-one").rglob("*") if p.is_file()) == [
        "f-one.py", "recipe.json"]
    assert "refused" in mini_lane.snapshot(root, "../x", "f-one")
    monkeypatch.setattr(mini_lane, "SNAP_MAX_FILES", 1)
    assert "too large" in mini_lane.snapshot(root, "2-0123abcd", "f-one")["refused"]


def test_runtime_placeholders_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = {"label": LABEL, "account": ACCOUNT, "model": "m", "effort": "high",
            "runtime": ["runtime", "--job", "{job}"], "identity": ["probe", "{account}"]}
    assert mini_lane.agent_argv(spec)[:3] == ["runtime", "--job", LABEL]
    assert mini_lane.agent_argv({**spec, "nice": 10})[:4] == ["nice", "-n", "10", "runtime"]
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: seen.append(argv) or
                        subprocess.CompletedProcess(argv, 0, '{"harness":"claude","account":"other"}', ""))
    with pytest.raises(OSError, match="identity"):
        mini_lane.runtime_env(spec)
    assert seen == [["probe", ACCOUNT]]


# --- what the hub runs ----------------------------------------------------------------------


@pytest.fixture
def scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> mini_verbs.Scope:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    for family in ("f-one", "f-two"):
        (tmp_path / "work/2b" / family / "adapters").mkdir(parents=True)
    monkeypatch.setattr(mini_verbs, "members_of", lambda family, data: {
        "f-one": {"m1", "m2"}, "f-two": {"t1"}}[family])
    from menagerie.pipeline import DataLayout

    return mini_verbs.scope_for(["f-one", "f-two"], tmp_path / "work/briefs/b/BRIEF.md",
                                DataLayout(tmp_path / "data"))


def test_only_the_lanes_verbs_targets_and_options_run(scope: mini_verbs.Scope, tmp_path: Path) -> None:
    base = scope.folders[0]
    plan = mini_verbs.authorize(["try", "m2", "--recipe", "recipe.json", "--with", "timm==1.0"],
                                scope, base)
    assert plan.argv == ["try", "m2", "--recipe", f"{base}/recipe.json", "--with", "timm==1.0"]
    assert plan.family == "f-one"
    assert mini_verbs.authorize(["giveup", "t1", "oversized", "too big /opt"], scope,
                                base).family == "f-two"
    assert mini_verbs.authorize(["done", f"{scope.folders[1]}/recipe.json"], scope,
                                base).family == "f-two"
    for refused in (["bridge", "serve"], ["try", "m1", "--spool", "/tmp/x"], ["try", "other"],
                    ["giveup", "m1", "Bad Class", "r"], ["defer", "f-three", "r"],
                    ["try", "m1", "--member", "other"], ["try", "m1", "--recipe=/etc/passwd"],
                    ["done", "../../leases/x.json"], ["src"], ["try", "m1", "--with", "$(id)"],
                    # only the target family's drafts are brought in, so only they may be named
                    ["try", "t1", "--recipe", "recipe.json"]):
        with pytest.raises(ValueError):
            mini_verbs.authorize(refused, scope, base)


def test_symlinks_and_recipe_adapters_stay_inside_the_lane(scope: mini_verbs.Scope, tmp_path: Path) -> None:
    folder = Path(scope.folders[0])
    (tmp_path / "secret.txt").write_text("hub secret")
    (folder / "escape.json").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(ValueError, match="outside"):
        mini_verbs.authorize(["try", "m1", "--recipe", "escape.json"], scope, str(folder))
    (folder / "recipe.json").write_text(json.dumps({"kind": "adapter", "adapter": str(tmp_path / "secret.txt")}))
    plan = mini_verbs.authorize(["try", "m1", "--recipe", "recipe.json"], scope, str(folder))
    with pytest.raises(ValueError, match="outside"):
        mini_verbs.check_recipes(plan, scope)
    (folder / "adapters/f-one.py").write_text('VARIANTS = {"m1": {}, "someone-else": {}}\n')
    (folder / "recipe.json").write_text(json.dumps({"kind": "adapter", "adapter": "adapters/f-one.py"}))
    mini_verbs.check_recipes(plan, scope)  # try reads the adapter but never names variants
    with pytest.raises(ValueError, match="someone-else"):
        mini_verbs.check_recipes(mini_verbs.authorize(["done", "recipe.json"], scope, str(folder)),
                                 scope)
    # A variant of another leased family is refused too: only the locked family's drafts are in.
    (folder / "adapters/f-one.py").write_text('VARIANTS = {"m1": {}, "t1": {}}\n')
    with pytest.raises(ValueError, match="t1"):
        mini_verbs.check_recipes(mini_verbs.authorize(["done", "recipe.json"], scope, str(folder)),
                                 scope)
    (folder / "adapters/f-one.py").write_text('VARIANTS = {"m1": {}, "someone-else": {}}\n')
    # Without an adapter field the family's default adapter is the one checked.
    (folder / "recipe.json").write_text(json.dumps({"kind": "adapter"}))
    with pytest.raises(ValueError, match="someone-else"):
        mini_verbs.check_recipes(mini_verbs.authorize(["done", "recipe.json"], scope, str(folder)),
                                 scope)


def test_done_accepts_only_the_locked_familys_passing_try(scope: mini_verbs.Scope,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from menagerie.pipeline import implement

    folder = Path(scope.folder("f-one"))
    (folder / "recipe.json").write_text(json.dumps({"kind": "library", "module": "timm"}))
    plan = mini_verbs.authorize(["done", "recipe.json"], scope, str(folder))
    for family, member, ok in (("f-one", "m1", True), ("f-two", "t1", False), ("other", "x", False)):
        monkeypatch.setattr(implement, "_passing", lambda path, f=family, m=member: (
            SimpleNamespace(family=f, member=m), None))
        if ok:
            mini_verbs.check_recipes(plan, scope)
        else:
            with pytest.raises(ValueError, match="outside f-one"):
                mini_verbs.check_recipes(plan, scope)


def test_paths_translate_longest_first() -> None:
    assert mini_relay.translate("/hub/a/recipe.json /hub/a-long/file", {
        "/hub/a": "/local/a", "/hub/a-long": "/local/long"}) == "/local/a/recipe.json /local/long/file"


def test_absolute_worker_adapter_becomes_relative_on_the_hub(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()
    recipe = {"kind": "adapter", "adapter": "/w/work/f-one/adapters/model.py", "config": {"x": "/w"}}
    (hub / "recipe.json").write_text(json.dumps(recipe))
    mini_relay.keep_adapter_paths(hub, {str(hub): "/w/work/f-one"})
    kept = json.loads((hub / "recipe.json").read_text())
    assert kept["adapter"] == "adapters/model.py"
    assert kept["config"] == recipe["config"]


def test_scan_plan_reads_pull_outputs_dry_run_lines() -> None:
    text = ("pull-outputs: 2 files, 9 bytes, project=menagerie dest=/hub/outputs/j\n"
            "  .mini-lane/snap/1-0123abcd/f-one/recipe.json          4  0123456789abcdef\n"
            "  .mini-lane/snap/1-0123abcd/f-one/my notes.txt         5  fedcba9876543210\n"
            "(dry run; nothing copied)\n")
    assert mini_relay.scan_plan(text) == {
        ".mini-lane/snap/1-0123abcd/f-one/recipe.json": (4, "0123456789abcdef"),
        ".mini-lane/snap/1-0123abcd/f-one/my notes.txt": (5, "fedcba9876543210")}


# --- one lane, end to end -------------------------------------------------------------------


FAKE_PULL = r'''#!{python}
import hashlib, pathlib, re, shutil, sys
host, job, *rest = sys.argv[1:]
assert host == "worker-a"
tree = pathlib.Path({home!r}) / ".orca-run/worktrees/menagerie" / job
dest = {outputs!r} + "/" + job
if "--dry-run" in rest:
    glob = rest[rest.index("--pull") + 1]
    files = sorted(p for p in tree.glob(glob.removesuffix("**") + "**/*") if p.is_file())
    if any(b"LEAKME" in p.read_bytes() for p in files):
        print("LEAK SCAN BLOCKED", file=sys.stderr)
        sys.exit(3)
    print(f"pull-outputs: {{len(files)}} files, 0 bytes, project=menagerie dest={{dest}}")
    for p in files:
        print(f"  {{p.relative_to(tree)}}  {{p.stat().st_size}}  {{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}}")
    sys.exit(0)
shutil.copytree(tree / "outputs/lane", dest + "/outputs/lane")
print("pull-outputs: 4 files, 10 bytes, project=menagerie dest=" + dest)
'''

FAKE_AGENT = r'''
import json, os, re, subprocess, sys
prompt = sys.stdin.read()
work = re.search(r"Work folder: `(\S+)`", prompt)[1]
wrapper = re.search(r"Commands: `(\S+) <verb>", prompt)[1]
result = re.search(r"Result file: `(\S+)`", prompt)[1]
def verb(*args):
    done = subprocess.run([wrapper, *args], capture_output=True, text=True)
    return done.returncode, done.stdout, done.stderr
notes = {}
rc, out, err = verb("src", "f-one")
notes["src"] = open(out.strip()).read()
os.makedirs(work + "/pkg/menagerie", exist_ok=True)
open(work + "/pkg/menagerie/__init__.py", "w").write("raise SystemExit('planted import ran')\n")
open(work + "/adapters.py", "w").write("# adapter\n")
json.dump({"kind": "adapter", "adapter": work + "/adapters.py"}, open(work + "/recipe.json", "w"))
open(work + "/notes.txt", "w").write("LEAKME token-shaped text\n")
notes["leak"] = verb("try", "m1", "--recipe", work + "/recipe.json")
os.remove(work + "/notes.txt")
os.chdir(work + "/pkg")
notes["try"] = verb("try", "m1", "--recipe", work + "/recipe.json")
notes["refused"] = verb("bridge", "serve")[0]
open(work + "/deferred.json", "w").write("worker overwrite")
json.dump(notes, open(work + "/notes.json", "w"))
open(result, "w").write("m1 pass; recipe " + work + "/recipe.json\n")
print("agent finished")
'''


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A hub work root with one family, a worker home with the job's checkout and one lease."""
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(mini_relay, "STATUS_EVERY_S", 0.0)
    monkeypatch.setattr(mini_verbs, "members_of", lambda family, data: {"m1"})
    family = tmp_path / "work/2b/f-one"
    (family / "sources").mkdir(parents=True)
    (family / "sources/code.py").write_text("saved model source")
    (family / "deferred.json").write_text("hub marker")
    (family / "BRIEF.md").write_text(
        f"# 2b lane: implement f-one\n\nWork folder: `{family}` (drafts). Result file: "
        f"`{family / 'BRIEF.result.md'}`, written last. Commands: `{family / 'menagerie'} <verb> ...`.\n")
    hub_log = tmp_path / "hub-verbs.jsonl"
    _script(family / "menagerie", f"""#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
with open({str(hub_log)!r}, "a") as log:
    log.write(json.dumps({{"lane": os.environ.get("MENAGERIE_LANE"), "cwd": os.getcwd(), "args": args,
                          "safe": os.environ.get("PYTHONSAFEPATH")}}) + "\\n")
if args[0] == "src":
    dest = pathlib.Path({str(tmp_path / 'work/lanes')!r}) / os.environ["MENAGERIE_LANE"] / "src/f-one"
    dest.mkdir(parents=True, exist_ok=True)
    link = dest / "code.py"
    if not link.exists():
        link.symlink_to({str(family / 'sources/code.py')!r})
    print(link)
elif args[0] == "try":
    recipe = json.loads(pathlib.Path(args[args.index("--recipe") + 1]).read_text())
    print("pass", recipe["adapter"])
""")
    home = tmp_path / "worker"
    worktree = home / mini_relay.WORKTREE.format(label=LABEL)
    (worktree / "menagerie/pipeline").mkdir(parents=True)
    shutil.copy(LANE_FILE, worktree / "menagerie/pipeline/mini_lane.py")  # the job's checkout
    outputs = tmp_path / "outputs"
    fake_pull = _script(tmp_path / "pull-outputs", FAKE_PULL.format(
        python=sys.executable, home=str(home), outputs=str(outputs)))
    fake_run_on = _script(tmp_path / "run-on", "#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(admission, "_tool", lambda name: str(
        {"pull-outputs": fake_pull, "run-on": fake_run_on}[name]))
    lease = {"label": LABEL, "run": "run-1", "kind": "2b", "keys": ["f-one"], "placement": "worker-a",
             "account": ACCOUNT, "mini": {"ssh_alias": "worker-a-ssh", "deliver": None},
             "brief": str(family / "BRIEF.md"), "result": str(family / "BRIEF.result.md"),
             "started": "2026-10-12T00:00:00Z", "expires": "2099-01-01T00:00:00Z"}
    leases = tmp_path / "work/leases"
    leases.mkdir(parents=True)
    (leases / f"{LABEL}.json").write_text(json.dumps(lease))
    return {"tmp": tmp_path, "family": family, "home": home, "worktree": worktree, "outputs": outputs,
            "lease": lease, "lease_path": leases / f"{LABEL}.json", "hub_log": hub_log}


def _run_job(lane: dict[str, Any], agent: Path) -> dict[str, Any]:
    spec = {"label": LABEL, "account": ACCOUNT, "model": "m", "effort": "high",
            "runtime": [sys.executable, str(agent)],
            "identity": [sys.executable, "-c", f"print('{{\"harness\":\"claude\",\"account\":\"{ACCOUNT}\"}}')"],
            "families": ["f-one"], "brief": "work/f-one/BRIEF.md",
            "result": "work/f-one/BRIEF.result.md", "stage_timeout_s": 60, "lane_timeout_s": 120}
    job: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: job.update(rc=mini_lane.run(spec, cwd=lane["worktree"])))
    worker.start()
    return {"thread": worker, "job": job}


def _drive(relay: mini_relay.Relay, until: Any, seconds: float = 90) -> None:
    deadline = time.monotonic() + seconds
    while not until() and time.monotonic() < deadline:
        relay.once()
        time.sleep(0.1)


@needs_rsync
def test_fake_runtime_lane_lands_through_a_pull(lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    family = lane["family"]
    state = {"run": "RUNNING"}
    monkeypatch.setattr(admission, "_status", lambda run: {"run": {"state": state["run"]}})
    agent = lane["tmp"] / "agent.py"
    agent.write_text(FAKE_AGENT)
    started = _run_job(lane, agent)
    relay = mini_relay.Relay(connect=lambda alias, control: LocalRemote(lane["home"]))
    _drive(relay, lambda: not started["thread"].is_alive())
    started["thread"].join(timeout=1)
    assert started["job"].get("rc") == 0, (lane["worktree"] / "outputs/lane/agent.log").read_text()

    # The job has ended but nothing is ingested yet: the lease keeps the family.
    state["run"] = "TERMINAL"
    assert admission._live(lane["lease_path"], lane["lease"]) == "INGESTING"
    _drive(relay, lambda: mini_relay.ingested(LABEL))
    assert json.loads((mini_relay.state_dir(LABEL) / "ingested.json").read_text())["status"] == "ingested"
    assert admission._live(lane["lease_path"], lane["lease"]) is None
    assert not lane["lease_path"].exists()

    notes = json.loads((family / "notes.json").read_text())
    assert notes["src"] == "saved model source"  # src's hub path resolved on the worker
    assert notes["leak"][0] == 2 and "leak scan" in notes["leak"][2]
    assert notes["try"][0] == 0 and notes["try"][1].startswith("pass ")
    assert notes["refused"] == 2  # only the 2b verbs run on the hub
    assert json.loads((family / "recipe.json").read_text())["adapter"] == "adapters.py"
    assert not (family / "notes.txt").exists()  # the blocked draft never reached the hub
    assert (family / "deferred.json").read_text() == "hub marker"
    assert (family / "BRIEF.result.md").read_text() == f"m1 pass; recipe {family}/recipe.json\n"
    calls = [json.loads(line) for line in lane["hub_log"].read_text().splitlines()]
    assert [c["args"][0] for c in calls] == ["src", "try"]
    assert {c["lane"] for c in calls} == {"run-1"}
    assert {c["safe"] for c in calls} == {"1"}
    assert {c["cwd"] for c in calls} == {str(mini_relay.state_dir(LABEL) / "cwd")}
    assert calls[1]["args"][-1] == os.path.realpath(family / "recipe.json")
    assert (lane["outputs"] / LABEL / "outputs/lane/lane.json").is_file()
    assert not list((lane["worktree"] / ".mini-lane/snap").glob("*"))


def _staged_lane(lane: dict[str, Any]) -> tuple[mini_relay.Lane, Path]:
    root = lane["worktree"].resolve() / ".mini-lane"
    for box in ("new", "claimed", "done"):
        (root / "spool" / box).mkdir(parents=True)
    mini_relay.state_dir(LABEL).mkdir(parents=True)
    (mini_relay.state_dir(LABEL) / "staged.json").write_text(json.dumps({
        "root": str(root), "worktree": str(lane["worktree"].resolve()), "paths": {}, "phase": "ready"}))
    return mini_relay.Lane(lane["lease"], LocalRemote(lane["home"])), root


@needs_rsync
def test_relay_restart_reruns_unstarted_verbs_and_flags_started_ones(lane: dict[str, Any]) -> None:
    one, root = _staged_lane(lane)
    started, waiting = "1-0123abcd", "2-0123abcd"
    for name in (started, waiting):
        (root / f"spool/claimed/{name}.json").write_text(json.dumps(
            {"id": name, "family": "f-one", "args": ["bridge", "serve"], "cwd": "/"}))
    (mini_relay.state_dir(LABEL) / "answers").mkdir()
    (mini_relay.state_dir(LABEL) / f"answers/{started}.started").touch()
    (root / "spool/new/bad;name.json").write_text("{}")
    one.poll()
    for thread in one.inflight.values():
        thread.join(timeout=10)
    first = json.loads((root / f"spool/done/{started}.json").read_text())
    assert first["rc"] == 75 and "restarted during this verb" in first["stderr"]
    second = json.loads((root / f"spool/done/{waiting}.json").read_text())
    assert second["rc"] == 2 and "verbs here" in second["stderr"]
    assert [p.name for p in (root / "spool/new").iterdir()] == ["bad;name.json"]  # ignored, never run
    assert not list((root / "spool/claimed").iterdir())


@needs_rsync
def test_undelivered_answers_are_kept_and_sent_later(lane: dict[str, Any]) -> None:
    one, root = _staged_lane(lane)
    name = "3-0123abcd"
    (root / f"spool/claimed/{name}.json").write_text("{}")
    one._put = lambda path, body: (_ for _ in ()).throw(OSError("link down"))  # type: ignore[method-assign]
    one.answer(name, json.dumps({"id": name, "family": "f-one", "args": ["src", "nope"]}).encode())
    assert (mini_relay.state_dir(LABEL) / f"answers/{name}.json").is_file()
    assert not (root / f"spool/done/{name}.json").exists()
    del one._put  # the link is back
    one.redeliver()
    assert json.loads((root / f"spool/done/{name}.json").read_text())["rc"] == 2


@needs_rsync
def test_failed_login_delivery_aborts_before_ready(lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    lease = {**lane["lease"], "mini": {"ssh_alias": "worker-a-ssh", "deliver": ["false", "{job}"]}}
    one = mini_relay.Lane(lease, LocalRemote(lane["home"]))
    one.stage()
    root = lane["worktree"].resolve() / ".mini-lane"
    assert (root / "ABORT").is_file() and not (root / "READY").exists()
    assert one.staged["phase"] == "aborted"
    assert (root / "work/f-one/sources/code.py").read_text() == "saved model source"
    assert not (root / "work/f-one/menagerie").exists() and not (root / "work/f-one/deferred.json").exists()
    assert str(lane["family"]) not in (root / "work/f-one/BRIEF.md").read_text()
    with pytest.raises(OSError, match="aborted"):
        mini_lane.wait_ready(root, timeout=1)


@needs_rsync
def test_a_lane_that_ends_asks_the_tick_for_a_refill(lane: dict[str, Any]) -> None:
    one, _ = _staged_lane(lane)
    want = work_dir("tick") / mini_relay.REFILL_WANT
    assert not want.exists()
    one._ended({"status": "pull-failed", "at": mini_relay.now()})
    assert want.is_file() and mini_relay.ingested(one.label)


def test_a_late_ingest_never_overwrites_a_newer_lease(lane: dict[str, Any]) -> None:
    one, _ = _staged_lane(lane)
    lane["lease_path"].write_text(json.dumps({**lane["lease"], "run": "run-2"}))
    pulled = lane["tmp"] / "pulled"
    (pulled / "families/f-one").mkdir(parents=True)
    (pulled / "families/f-one/recipe.json").write_text("late draft")
    (pulled / "result.md").write_text("late result")
    assert one.keep(pulled).startswith("late")
    assert not (lane["family"] / "recipe.json").exists()
    assert not (lane["family"] / "BRIEF.result.md").exists()


# --- a describe lane: no verbs, the batch folder in, out/ and the result back ---------------------


DESCRIBE_LABEL = "describe-mini-test"
DESCRIBE_AGENT = r'''
import json, os, re, sys
prompt = sys.stdin.read()
folder = re.search(r"Family folders: `(\S+)/families/<family>/`", prompt)[1]
result = re.search(r"result file is `(\S+)`", prompt)[1]
paper = open(folder + "/families/f-one/source/s1-paper.txt").read()
earlier = open(folder + "/out/f-one.json").read() if os.path.exists(folder + "/out/f-one.json") else ""
json.dump({"family": "f-one", "paper": paper, "earlier": earlier}, open(folder + "/out/f-one.json", "w"))
json.dump({"family": "f-other"}, open(folder + "/out/f-other.json", "w"))
os.makedirs(os.path.dirname(folder) + "/../spool/new", exist_ok=True)
open(os.path.dirname(folder) + "/../spool/new/1-0123abcd.json", "w").write(json.dumps(
    {"id": "1-0123abcd", "family": "f-one", "args": ["bridge", "serve"], "cwd": "/"}))
open(result, "w").write("done; drafts in " + folder + "/out\n")
'''


@pytest.fixture
def describe_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A hub describe batch folder, a worker home with the job's checkout and one describe lease."""
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(mini_relay, "STATUS_EVERY_S", 0.0)
    folder = tmp_path / "work/describe/abcd1234"
    (folder / "families/f-one/source").mkdir(parents=True)
    (folder / "families/f-one/source/s1-paper.txt").write_text("We present a residual framework.")
    (folder / "families/f-one/facts.json").write_text("{}")
    (folder / "out").mkdir()
    (folder / "out/f-one.json").write_text('{"earlier": true}')  # a return lane's earlier draft
    (folder / "trial-values.json").write_text("{}")  # never staged
    (folder / "BRIEF.md").write_text(
        f"# Describe lane\n\nFamily folders: `{folder}/families/<family>/`. Write `{folder}/out/<family>.json` "
        f"for each family below; the result file is `{folder}/BRIEF.result.md`.\n")
    home = tmp_path / "worker"
    worktree = home / mini_relay.WORKTREE.format(label=DESCRIBE_LABEL)
    (worktree / "menagerie/pipeline").mkdir(parents=True)
    shutil.copy(LANE_FILE, worktree / "menagerie/pipeline/mini_lane.py")
    outputs = tmp_path / "outputs"
    fake_pull = _script(tmp_path / "pull-outputs", FAKE_PULL.format(
        python=sys.executable, home=str(home), outputs=str(outputs)))
    fake_run_on = _script(tmp_path / "run-on", "#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(admission, "_tool", lambda name: str(
        {"pull-outputs": fake_pull, "run-on": fake_run_on}[name]))
    lease = {"label": DESCRIBE_LABEL, "run": "run-d", "kind": "describe", "keys": ["f-one"],
             "placement": "worker-a", "account": ACCOUNT, "mini": {"ssh_alias": "worker-a-ssh", "deliver": None},
             "brief": str(folder / "BRIEF.md"), "result": str(folder / "BRIEF.result.md"),
             "started": "2026-10-29T00:00:00Z", "expires": "2099-01-01T00:00:00Z"}
    leases = tmp_path / "work/leases"
    leases.mkdir(parents=True)
    (leases / f"{DESCRIBE_LABEL}.json").write_text(json.dumps(lease))
    return {"tmp": tmp_path, "folder": folder, "home": home, "worktree": worktree, "lease": lease,
            "lease_path": leases / f"{DESCRIBE_LABEL}.json"}


@needs_rsync
def test_describe_lane_stages_its_batch_and_ingests_only_its_drafts(describe_lane: dict[str, Any],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    folder, worktree = describe_lane["folder"], describe_lane["worktree"]
    assert mini_relay.layout(["f-one"], folder / "BRIEF-r2.md") == (
        "work/abcd1234/BRIEF-r2.md", "work/abcd1234/BRIEF-r2.result.md")
    state = {"run": "RUNNING"}
    monkeypatch.setattr(admission, "_status", lambda run: {"run": {"state": state["run"]}})
    agent = describe_lane["tmp"] / "agent.py"
    agent.write_text(DESCRIBE_AGENT)
    spec = {"label": DESCRIBE_LABEL, "account": ACCOUNT, "model": "m", "effort": "high", "kind": "describe",
            "runtime": [sys.executable, str(agent)],
            "identity": [sys.executable, "-c", f"print('{{\"harness\":\"claude\",\"account\":\"{ACCOUNT}\"}}')"],
            "families": ["f-one"], "brief": "work/abcd1234/BRIEF.md", "result": "work/abcd1234/BRIEF.result.md",
            "stage_timeout_s": 60, "lane_timeout_s": 120}
    job: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: job.update(rc=mini_lane.run(spec, cwd=worktree)))
    worker.start()
    relay = mini_relay.Relay(connect=lambda alias, control: LocalRemote(describe_lane["home"]))
    _drive(relay, lambda: not worker.is_alive())
    worker.join(timeout=1)
    assert job.get("rc") == 0, (worktree / "outputs/lane/agent.log").read_text()
    root = worktree.resolve() / ".mini-lane"
    assert not (root / "work/abcd1234/trial-values.json").exists()
    assert not (root / "work/f-one").exists()  # no verb wrappers
    assert (root / "spool/new/1-0123abcd.json").is_file()  # a describe lane's spool is never read
    state["run"] = "TERMINAL"
    assert admission._live(describe_lane["lease_path"], describe_lane["lease"]) == "INGESTING"
    _drive(relay, lambda: mini_relay.ingested(DESCRIBE_LABEL))
    assert json.loads((mini_relay.state_dir(DESCRIBE_LABEL) / "ingested.json").read_text())["status"] == "ingested"
    draft = json.loads((folder / "out/f-one.json").read_text())
    assert draft["paper"] == "We present a residual framework." and draft["earlier"] == '{"earlier": true}'
    assert not (folder / "out/f-other.json").exists()  # only the lease's families come back
    assert (folder / "BRIEF.result.md").read_text() == f"done; drafts in {folder}/out\n"
    assert admission._live(describe_lane["lease_path"], describe_lane["lease"]) is None
