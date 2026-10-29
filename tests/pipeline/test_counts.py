"""Counts built on the worker (speed-up lane sp-hubload): the file, the hub's reader, the launcher."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from menagerie.pipeline import (
    DataLayout,
    append_jsonl,
    counts,
    drive,
    publish,
    read_json,
    read_jsonl,
    tick_full,
    write_json,
)
from menagerie.pipeline.records import ModelRecord, Packet, dumps, parse_ledger_line

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).parents[2]
TL = "2.35.2"
IDS = ["m3631", "m8839"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "core.hooksPath=/dev/null", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo: Path) -> str:
    _git(repo, "add", "menagerie")
    _git(repo, "commit", "-q", "-m", "data")
    return _git(repo, "rev-parse", "HEAD")


def _status_files(repo: Path) -> None:
    data = repo / "menagerie" / "data"
    (data / "sweeps").mkdir(parents=True)
    (data / "sweeps" / "index.jsonl").write_text(json.dumps(
        {"sweep": "S001", "axes": {"venue": {"state": "open", "frontier": "2026-07-04"}}}) + "\n")
    (data / "master_catalog.jsonl").write_text('{"name": "a"}\n')
    (data / "catalog").mkdir()
    (data / "catalog" / "plan.tsv").write_text("id\twave\nm8839\tC1\n", encoding="utf-8")
    site = repo / "site" / "site-data"
    site.mkdir(parents=True)
    (site / "export_manifest.json").write_text(json.dumps(
        {"counts": {"models": 7}, "export_version": "phase0-preview"}))


@pytest.fixture()
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """resnet18's record, packet, roster rows and a forward row, committed in a scratch repo."""

    monkeypatch.setattr(publish, "torchlens_pin", lambda: TL)
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    repo = tmp_path / "repo"
    layout = DataLayout(repo / "menagerie" / "data")
    layout.roster.parent.mkdir(parents=True)
    layout.roster.write_text((FIXTURES / "roster.jsonl").read_text(encoding="utf-8"))
    write_json(read_json(ModelRecord, FIXTURES / "m8839.model.json"), layout.model("m8839"))
    write_json(read_json(Packet, FIXTURES / "f-resnet.packet.json"), layout.packet("f-resnet"))
    trace, _ = read_jsonl(parse_ledger_line, FIXTURES / "m8839.e2e.ledger.jsonl")
    forward = dataclasses.replace(trace, stage="forward", capture=None, trace=None, n_ops=None,
                                  graph_hash=None, validation_state=None)
    layout.ledger("2026-10").parent.mkdir(parents=True)
    layout.ledger("2026-10").write_text(dumps(forward) + "\n", encoding="utf-8")
    _status_files(repo)
    _git(repo, "init", "-q")
    _commit(repo)
    return layout


def _built(data: DataLayout, tmp_path: Path) -> Path:
    out = tmp_path / "out"
    counts.build(data=data, out=out, commit=_git(data.root, "rev-parse", "HEAD"))
    return out / counts.NEXT_STEPS


def _direct(data: DataLayout, ids: list[str] = IDS) -> dict[str, str]:
    return {s["id"]: s["step"] for s in drive.next_steps(ids, data=data)}


def test_build_writes_steps_family_keys_and_status(data: DataLayout, tmp_path: Path) -> None:
    path = _built(data, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["schema"] == counts.SCHEMA and payload["tl"] == TL
    assert payload["commit"] == _git(data.root, "rev-parse", "HEAD")
    assert payload["steps"] == {"m3631": ["2a", None],
                                "m8839": ["trace", counts._family_key("f-resnet")]}
    assert "resnet" not in path.read_text()  # family names stay out of the pulled file
    status = json.loads((path.parent / counts.STATUS).read_text())
    assert status["counts"]["roster"]["rows"] == 2 and status["counts"]["moving"]["traced"] == 0


def test_the_file_gives_the_in_process_plan_on_the_same_commit(data: DataLayout,
                                                              tmp_path: Path) -> None:
    path = _built(data, tmp_path)
    assert counts.steps_for(IDS, data=data, path=path) == _direct(data)
    assert counts.plan_equal(IDS, data=data, path=path) == {
        "ids": 2, "file_used": True, "differ": [], "equal": True}


def test_rows_changed_since_the_commit_are_recomputed(data: DataLayout, tmp_path: Path) -> None:
    path = _built(data, tmp_path)
    trace, _ = read_jsonl(parse_ledger_line, FIXTURES / "m8839.e2e.ledger.jsonl")
    append_jsonl([trace], data.ledger("2026-10"))  # uncommitted: the hub's working tree
    assert counts.steps_for(IDS, data=data, path=path) == _direct(data) == {
        "m3631": "2a", "m8839": "render"}
    _commit(data.root.parents[1])  # and once the collector commits it
    assert counts.steps_for(IDS, data=data, path=path) == _direct(data)
    packet = read_json(Packet, data.packet("f-resnet"))
    m3631 = dataclasses.replace(packet.members[0], id="m3631")
    write_json(dataclasses.replace(packet, members=(*packet.members, m3631)), data.packet("f-resnet"))
    assert counts.steps_for(IDS, data=data, path=path) == _direct(data)
    assert _direct(data)["m3631"] == "2b"  # a packet now lists it
    record = read_json(ModelRecord, FIXTURES / "m3631.model.json")
    write_json(record, data.model("m3631"))  # a new, untracked record
    assert counts.steps_for(IDS, data=data, path=path) == _direct(data)
    assert counts.steps_for(["m3631", "m9999"], data=data, path=path) == _direct(
        data, ["m3631", "m9999"])  # an id the file never listed


def test_a_repair_a_lane_ended_is_done_although_the_file_says_2b(data: DataLayout,
                                                               tmp_path: Path) -> None:
    from menagerie.paths import work_dir
    from menagerie.pipeline.records import Amendment

    trace, _ = read_jsonl(parse_ledger_line, FIXTURES / "m8839.e2e.ledger.jsonl")
    append_jsonl([dataclasses.replace(trace, outcome="fail", cls="torchlens_gap:replay")],
                 data.ledger("2026-10"))
    packet = read_json(Packet, data.packet("f-resnet"))
    note = Amendment(at="2026-10-28T01:00:00Z", why="repair: m8839; warm the BatchNorms",
                     evidence=(), changed=("repair",), by="Crawler")
    write_json(dataclasses.replace(packet, amendments=(note,)), data.packet("f-resnet"))
    _commit(data.root.parents[1])
    path = _built(data, tmp_path)
    assert counts.steps_for(IDS, data=data, path=path)["m8839"] == "2b"
    result = work_dir("2b") / "f-resnet" / "BRIEF.result.md"
    result.parent.mkdir(parents=True)
    result.write_text("the repair does not apply\n", encoding="utf-8")  # no committed file moves
    assert counts.steps_for(IDS, data=data, path=path) == _direct(data)
    assert _direct(data)["m8839"] == "done"


def test_a_stale_missing_or_foreign_file_falls_back(data: DataLayout, tmp_path: Path) -> None:
    path = _built(data, tmp_path)
    payload = json.loads(path.read_text())
    assert counts.steps_for(IDS, data=data, path=tmp_path / "none.json") is None
    old = datetime.now(timezone.utc) - timedelta(minutes=31)
    path.write_text(json.dumps({**payload, "built_at": old.strftime("%Y-%m-%dT%H:%M:%SZ")}))
    assert counts.steps_for(IDS, data=data, path=path) is None
    for change in ({"tl": "2.35.3"}, {"commit": None}, {"commit": "0" * 40}, {"schema": "x"}):
        path.write_text(json.dumps({**payload, **change}))
        assert counts.steps_for(IDS, data=data, path=path) is None, change
    path.write_text("{not json")
    assert counts.steps_for(IDS, data=data, path=path) is None
    path.write_text(json.dumps(payload))
    data.roster.write_text(data.roster.read_text() + "\n")  # a roster change: fall back
    assert counts.steps_for(IDS, data=data, path=path) is None


def test_the_tick_reads_the_file_only_when_switched_on(data: DataLayout, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(counts, "steps_for", lambda ids, data: seen.append("file") or {"m8839": "x"})
    monkeypatch.delenv("MENAGERIE_TICK_COUNTS", raising=False)
    assert tick_full._steps(["m8839"], data, {"counts_file": False}) == {"m8839": "trace"}
    assert seen == []
    assert tick_full._steps(["m8839"], data, {"counts_file": True}) == {"m8839": "x"}
    monkeypatch.setenv("MENAGERIE_TICK_COUNTS", "1")
    assert tick_full._steps(["m8839"], data, {}) == {"m8839": "x"}
    monkeypatch.setattr(counts, "steps_for", lambda ids, data: None)  # stale: fall back
    assert tick_full._steps(["m8839"], data, {}) == {"m8839": "trace"}


def _fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!{sys.executable}\nimport json, os, shutil, sys\nfrom pathlib import Path\n"
                    f"args = sys.argv[1:]\nopen(os.environ['CALLS'], 'a').write(json.dumps(['{name}', *args]) + '\\n')\n"
                    + body)
    path.chmod(0o755)


def test_the_worker_job_round_trips_into_the_tick(data: DataLayout, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """``counts build --worker``: submit at HEAD, wait, pull, move, install, reap; the tick reads it."""

    built = _built(data, tmp_path)
    repo = data.root.parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "run-on", "if args[0] == 'submit': print('job: ' + args[args.index('--job') + 1])\n"
                             "if args[0] == 'status': print(args[1] + ' worker DONE rc=0')\n")
    _fake(bin_dir, "pull-outputs", f"""
job = args[1]
pulled = Path.cwd() / 'outputs' / job
if '--pull' in args:
    (pulled / 'outputs' / 'counts').mkdir(parents=True)
    shutil.copy({str(built)!r}, pulled / 'outputs' / 'counts' / 'next-steps.json')
if '--move-to' in args:
    shutil.move(str(pulled), args[args.index('--move-to') + 1])
""")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.setattr(counts.time, "sleep", lambda s: None)
    result = counts.build(data=data, worker=True)
    assert result["state"] == "DONE" and result["commit"] == _git(repo, "rev-parse", "HEAD")
    assert counts.installed().is_file() and counts.STATUS not in result["installed"]
    calls = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    submit = next(c for c in calls if c[:2] == ["run-on", "submit"])
    assert submit[submit.index("--ref") + 1] == result["commit"]
    assert submit[-2:] == ["--commit", result["commit"]]
    assert ["run-on", "reap", "--host", "worker", "--job", result["job"], "--apply"] in calls
    assert counts.steps_for(IDS, data=data) == _direct(data)
    assert tick_full._steps(IDS, data, {"counts_file": True}) == _direct(data)


def test_a_refused_worker_job_installs_nothing(data: DataLayout, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "run-on", "print('run-on: x waiting: NO_CAPACITY: the worker is full')\n"
                             "sys.exit(75)\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CALLS", str(tmp_path / "calls.jsonl"))
    result = counts.build(data=data, worker=True)
    assert result["state"] == "refused" and "NO_CAPACITY" in result["why"]
    assert not counts.installed().exists()


def test_data_status_json_renders_the_same_markdown() -> None:
    spec = importlib.util.spec_from_file_location("data_status", REPO / "scripts" / "data_status.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "data_status.py"), "--json"],
                          capture_output=True, text=True, check=True)
    assert module.render_counts(json.loads(done.stdout)) == module.render(REPO)
