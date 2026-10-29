"""The driver on real rung 0 fixtures: next steps, events, leases, launch, measures and collect.

External tools (``role``, admission, ``fleet-run``, ``git-commit-paths``) are small fakes on
``PATH`` that record their arguments; nothing here starts a lane or needs the network.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from menagerie.pipeline import (
    DataLayout, admission, append_jsonl, drive, publish, read_json, read_jsonl, write_json,
)
from menagerie.pipeline.records import (
    Amendment,
    GiveUp,
    LedgerRow,
    ModelRecord,
    Packet,
    RenderRow,
    Source,
    dumps,
    parse_ledger_line,
)

FIXTURES = Path(__file__).parent / "fixtures"
TL = "2.35.2"


@pytest.fixture()
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """resnet18's real record, packet, roster row and e2e trace and render rows."""

    monkeypatch.setattr(publish, "torchlens_pin", lambda: TL)
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    layout = DataLayout(tmp_path / "data")
    layout.roster.parent.mkdir(parents=True)
    layout.roster.write_text((FIXTURES / "roster.jsonl").read_text(encoding="utf-8"))
    write_json(read_json(ModelRecord, FIXTURES / "m8839.model.json"), layout.model("m8839"))
    write_json(read_json(Packet, FIXTURES / "f-resnet.packet.json"), layout.packet("f-resnet"))
    return layout


def _rows() -> tuple[LedgerRow, RenderRow]:
    trace, render = read_jsonl(parse_ledger_line, FIXTURES / "m8839.e2e.ledger.jsonl")
    return trace, render


def _forward(trace: LedgerRow, **kw: object) -> LedgerRow:
    return dataclasses.replace(trace, stage="forward", capture=None, trace=None, n_ops=None,
                               graph_hash=None, validation_state=None, **kw)


def _ledger(data: DataLayout, *rows: object) -> None:
    path = data.ledger("2026-10")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")


def _record(data: DataLayout, **kw: object) -> ModelRecord:
    record = dataclasses.replace(read_json(ModelRecord, data.model("m8839")), **kw)
    write_json(record, data.model("m8839"))
    return record


def _step(data: DataLayout, model_id: str = "m8839") -> str:
    (row,) = drive.next_steps([model_id], data=data)
    return row["step"]


def test_the_run_steps_follow_the_ledger(data: DataLayout) -> None:
    trace, render = _rows()
    assert _step(data) == "forward"
    _ledger(data, _forward(trace))
    assert _step(data) == "trace"
    _ledger(data, _forward(trace), trace)
    assert _step(data) == "render"
    _ledger(data, _forward(trace), trace, render)
    assert drive.next_steps(["m8839"], data=data)[0] == {
        "id": "m8839", "step": "done", "why": "library: traced and rendered"}


def test_a_new_recipe_or_pin_retires_the_rows(data: DataLayout,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    trace, render = _rows()
    _ledger(data, _forward(trace), trace, render)
    monkeypatch.setattr(publish, "torchlens_pin", lambda: "2.35.3")
    assert _step(data) == "trace"
    record = read_json(ModelRecord, data.model("m8839"))
    _record(data, recipe=dataclasses.replace(record.recipe, rev="0123456789ab"))
    assert _step(data) == "forward"


def test_failures_end_typed_or_take_a_structure_only_capture(data: DataLayout) -> None:
    trace, render = _rows()
    _ledger(data, _forward(trace, outcome="fail", cls="missing_dependency:mmcv"))
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == (
        "forward failed: missing_dependency:mmcv")
    capped = _forward(trace, outcome="fail", cls="memory_cap")
    _ledger(data, capped)
    assert _step(data) == "trace"
    structure = dataclasses.replace(trace, capture="structure_only", trace=None,
                                    summary=render.trace, validation_state="structure_only",
                                    params_source=None)
    full_fail = dataclasses.replace(trace, outcome="fail", cls="memory_cap", trace=None,
                                    validation_state=None)
    _ledger(data, capped, full_fail, structure, render)
    assert _step(data) == "done"
    _ledger(data, _forward(trace), dataclasses.replace(trace, outcome="fail",
                                                       cls="torchlens_gap:replay"))
    assert _step(data) == "done"


def test_a_row_traced_without_a_forward_row_moves_past_the_forward(data: DataLayout) -> None:
    trace, render = _rows()
    _ledger(data, trace)  # traced before forward rows were written: stage3 skips it as traced
    assert _step(data) == "render"
    _ledger(data, trace, render)
    assert _step(data) == "done"
    _ledger(data, dataclasses.replace(trace, outcome="fail", cls="torchlens_gap:replay"))
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == "trace failed: torchlens_gap:replay"


def _amend(data: DataLayout, at: str, why: str) -> None:
    packet = read_json(Packet, data.packet("f-resnet"))
    note = Amendment(at=at, why=why, evidence=(), changed=("repair",), by="Crawler")
    write_json(dataclasses.replace(packet, amendments=(*packet.amendments, note)),
               data.packet("f-resnet"))


def test_a_repair_amendment_sends_a_failed_row_back_to_2b_until_a_lane_tries_it(
        data: DataLayout) -> None:
    from menagerie.paths import work_dir
    from menagerie.pipeline import implement

    trace, render = _rows()
    failed = dataclasses.replace(trace, outcome="fail", cls="torchlens_gap:replay")
    _ledger(data, _forward(trace), failed)
    _amend(data, "2026-10-01T00:00:00Z", "repair: m8839; before the failure: no repair")
    _amend(data, "2026-10-28T00:00:00Z", "repair: m1; another member's repair")
    assert _step(data) == "done"
    packet = read_json(Packet, data.packet("f-resnet"))
    assert implement.repair_request_section(packet, data) == ""
    _amend(data, "2026-10-28T01:00:00Z", "repair: m8839; warm the BatchNorms (m14958)")
    (row,) = drive.next_steps(["m8839"], data=data)
    assert row["step"] == "2b" and row["why"].startswith(
        "packet amended for a repair at 2026-10-28T01:00:00Z (trace failed: torchlens_gap:replay)")
    section = implement.repair_request_section(read_json(Packet, data.packet("f-resnet")), data)
    assert "## Repair request" in section and "m8839" in section
    assert "warm the BatchNorms (m14958)" in section and "another member" not in section
    _ledger(data, _forward(trace), trace, render)  # a passing trace is never sent back
    assert _step(data) == "done"
    _ledger(data, _forward(trace), failed)
    result = work_dir("2b") / "f-resnet" / "BRIEF.result.md"
    result.parent.mkdir(parents=True)
    result.write_text("nothing to repair\n", encoding="utf-8")  # a lane tried it after the amendment
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == "trace failed: torchlens_gap:replay"


def test_a_cluster_recipe_skips_the_worker_forward(data: DataLayout) -> None:
    record = read_json(ModelRecord, data.model("m8839"))
    _record(data, recipe=dataclasses.replace(record.recipe, placement="axon-cpu"))
    assert _step(data) == "trace"


def test_authored_recipes_wait_for_a_review(data: DataLayout) -> None:
    trace, render = _rows()
    review = dataclasses.replace(_forward(trace), stage="review", mode=None, verdict="faithful")
    _ledger(data, _forward(trace), trace, render)
    _record(data, source_case="classic")
    assert _step(data) == "review"
    _ledger(data, _forward(trace), trace, render, review)
    assert _step(data) == "done"
    _ledger(data, _forward(trace), trace, render, dataclasses.replace(review,
                                                                      verdict="fabrication"))
    assert _step(data) == "2b"
    from menagerie.pipeline.review import PROVISIONAL

    first = dataclasses.replace(review, verdict="fabrication", msg=f"prompt p; {PROVISIONAL}",
                                reviewer="model-design-review")
    _ledger(data, _forward(trace), trace, render, first)
    assert _step(data) == "review"  # a provisional seat's fabrication waits for a second review
    again = dataclasses.replace(first, at="2026-12-30T00:00:00Z")
    _ledger(data, _forward(trace), trace, render, first, again)
    assert _step(data) == "review"  # the same seat again is not an independent second review
    second = dataclasses.replace(first, at="2026-12-31T00:00:00Z", reviewer="model-default")
    _ledger(data, _forward(trace), trace, render, first, again, second)
    assert _step(data) == "2b"


def test_a_review_with_no_saved_source_goes_to_2a_first(data: DataLayout) -> None:
    trace, render = _rows()
    _record(data, source_case="classic")
    packet = read_json(Packet, data.packet("f-resnet"))
    local = Source(sid="s1", role="code", url="menagerie/classics/x.py", official=False)
    stub = dataclasses.replace(packet, by="harvester@menagerie", sources=(local,), amendments=(),
                               code_order=())
    write_json(stub, data.packet("f-resnet"))
    _ledger(data, _forward(trace), trace, render)
    assert _step(data) == "2a"
    unjudged = dataclasses.replace(_forward(trace), stage="review", mode=None,
                                   verdict="cannot_judge", reviewer="model-design-review")
    _ledger(data, _forward(trace), trace, render, unjudged)
    assert _step(data) == "2a"  # a cannot_judge with nothing to judge by is not settled either
    write_json(dataclasses.replace(stub, by="2a-tick-x"), data.packet("f-resnet"))
    assert _step(data) == "done"  # 2a researched it and found no source: cannot_judge settles


def test_a_cannot_judge_is_reviewed_again_once_the_packet_gains_sources(data: DataLayout) -> None:
    trace, render = _rows()
    _record(data, source_case="classic")
    unjudged = dataclasses.replace(_forward(trace), stage="review", mode=None,
                                   verdict="cannot_judge", reviewer="model-design-review")
    _ledger(data, _forward(trace), trace, render, unjudged)
    assert _step(data) == "done"
    packet = read_json(Packet, data.packet("f-resnet"))
    later = Amendment(at="2999-01-01T00:00:00Z", why="paper fetched", evidence=(),
                      changed=("sources.s9",), by="research-complete")
    write_json(dataclasses.replace(packet, amendments=(*packet.amendments, later)),
               data.packet("f-resnet"))
    assert _step(data) == "review"


def test_vendor_code_is_reviewed_unless_official(data: DataLayout) -> None:
    trace, render = _rows()
    _ledger(data, _forward(trace), trace, render)
    record = _record(data, source_case="vendor")
    packet = read_json(Packet, data.packet("f-resnet"))
    code = Source(sid="s9", role="code", url="https://github.com/x/y", official=False)
    write_json(dataclasses.replace(packet, sources=(*packet.sources, code)),
               data.packet("f-resnet"))
    _record(data, recipe=dataclasses.replace(record.recipe, sources=("s9",)))
    assert _step(data) == "review"
    write_json(dataclasses.replace(packet, sources=(*packet.sources,
                                                    dataclasses.replace(code, official=True))),
               data.packet("f-resnet"))
    assert _step(data) == "done"


def test_research_steps_come_from_packets_and_give_ups(data: DataLayout) -> None:
    data.model("m8839").unlink()
    assert _step(data) == "2b"  # the packet lists m8839 as a member
    assert _step(data, "m3631") == "2a"
    record = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    gap = GiveUp(cls="research_gap:hf-config", reason="no config", at="2026-10-04T07:40:56Z")
    write_json(dataclasses.replace(record, recipe=None, give_up=gap), data.model("m8839"))
    assert _step(data) == "2a"
    packet = read_json(Packet, data.packet("f-resnet"))
    other = Amendment(at="2026-10-04T08:00:00Z", why="the library's paper link",
                      evidence=("s2",), changed=("sources.s2",), by="integration")
    write_json(dataclasses.replace(packet, amendments=(other,)), data.packet("f-resnet"))
    assert _step(data) == "2a"
    answer = dataclasses.replace(other, why="research_gap:hf-config: the Hub config.json saved")
    write_json(dataclasses.replace(packet, amendments=(other, answer)), data.packet("f-resnet"))
    assert _step(data) == "2b"
    again = dataclasses.replace(gap, cls="research_gap:hf-code", at="2026-10-04T09:00:00Z")
    write_json(dataclasses.replace(record, recipe=None, give_up=again), data.model("m8839"))
    assert _step(data) == "2a"  # no earlier give-up of this model in the ledger: a first gap
    append_jsonl([LedgerRow(id="m8839", stage="forward", at=gap.at, host="hub", env="torch-main",
                            recipe_rev="0" * 12, outcome="giveup", cls=gap.cls)],
                 data.ledger("2026-10"))
    assert drive.next_steps(["m8839"], data=data)[0] == {
        "id": "m8839", "step": "done",
        "why": "gave up: research_gap:hf-code after the amendment's one 2b retry"}
    write_json(dataclasses.replace(record, recipe=None, give_up=dataclasses.replace(
        gap, cls="compiled_ext:mamba", by="agent")), data.model("m8839"))
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == "gave up: compiled_ext:mamba"


def test_a_row_its_packet_dropped_goes_to_2a_not_to_its_old_familys_lane(
        data: DataLayout) -> None:
    """m155 kept ``family: f-autoencoder-vae`` after 2a dropped it from that packet; the family's
    2b lane, which works only on the packet's members, was launched for it every tick pass."""

    record = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    scripted = GiveUp(cls="needs_adapter:inputs", reason="unsupported", at="2026-10-09T20:28:53Z",
                      by="script")
    write_json(dataclasses.replace(record, recipe=None, give_up=scripted), data.model("m8839"))
    assert _step(data) == "2b"  # a member owed one agent attempt
    packet = read_json(Packet, data.packet("f-resnet"))
    moved = (dataclasses.replace(packet.members[0], id="m1"),)
    write_json(dataclasses.replace(packet, members=moved), data.packet("f-resnet"))
    assert drive.next_steps(["m8839"], data=data)[0] == {
        "id": "m8839", "step": "2a",
        "why": "not a member of packet f-resnet: 2a gives it a family first"}
    write_json(dataclasses.replace(record, recipe=None, give_up=dataclasses.replace(
        scripted, by="agent")), data.model("m8839"))
    assert _step(data) == "done"  # a settled row stays settled wherever its packet lists it


def test_roster_state_and_default_ids(data: DataLayout) -> None:
    lines = data.roster.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1]) | {"state": "duplicate_of:m8839"}
    data.roster.write_text("\n".join([lines[0], json.dumps(row)]) + "\n", encoding="utf-8")
    steps = drive.next_steps(data=data)
    assert [s["id"] for s in steps] == ["m3631", "m8839"]
    assert steps[0] == {"id": "m3631", "step": "done", "why": "roster: duplicate_of:m8839"}


def test_events_append_whole_lines(data: DataLayout, tmp_path: Path) -> None:
    drive.event("run-1", "f-resnet", "fetch code", url="https://github.com/x/y")
    drive.event("run-1", "f-resnet", "try", ok=True)
    lines = (tmp_path / "work" / "lanes" / "run-1" / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["verb"] for line in lines] == ["fetch code", "try"]
    assert json.loads(lines[0])["url"] == "https://github.com/x/y"
    with pytest.raises(ValueError):
        drive.event("../up", "f-resnet", "try")


def _fake(bin_dir: Path, name: str, body: str) -> Path:
    """A fake tool: records its argv to ``<name>.calls`` and runs ``body`` (Python)."""

    path = bin_dir / name
    path.write_text(f"#!{sys.executable}\nimport json, sys\n"
                    f"open({str(bin_dir / (name + '.calls'))!r}, 'a').write("
                    "json.dumps(sys.argv[1:]) + '\\n')\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture()
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake fleet tools: a desk root with role, admission, pause and fleet-run."""

    desk = tmp_path / "desk"
    (desk / "bin").mkdir(parents=True)
    (desk / "tools").mkdir()
    plans = {"model-default": {"harness": "claude", "model": "m-claude", "effort": "high"},
             "model-implementation": {"harness": "codex", "model": "m-codex", "effort": "max"}}
    _fake(desk / "bin", "role", f"print(json.dumps({plans!r}[sys.argv[2]]))")
    _fake(desk / "bin", "burst-admission", "")
    _fake(desk / "bin", "fleet-run", "import os\nif sys.argv[1] == 'status':\n"
          "    print(json.dumps({'run': {'state': os.environ.get('FLEET_STATE', 'RUNNING')}}))\n"
          "else:\n    print('started run_id=run-9 pid=1')")
    _fake(desk / "tools", "pause.py", "")
    _fake(desk / "tools", "chatgpt_usage.py",
          "import os\nsys.exit(75 if os.environ.get('CODEX_FULL') else 0)")
    _fake(desk / "tools", "lane_admission.py", "print('CLAUDE_ACCOUNT=acct-a')")
    monkeypatch.setenv("FLEET_RUN_DESK_ROOT", str(desk))
    monkeypatch.setenv("FLEET_RUN_ADAPTER", "/adapter")
    monkeypatch.setenv("MENAGERIE_CLAUDE_ACCOUNT", "pool")  # not the checkout's own marker
    monkeypatch.setenv("PATH", f"{desk / 'bin'}:{os.environ['PATH']}")
    _fake(desk / "bin", "account-usage", "print('acct-a: 12% of the week')")
    _fake(desk / "bin", "agy-quota", "print('gemini-subscription: 2% used')")
    monkeypatch.setattr(admission, "_GAUGES", {})  # each test reads its own fake gauges
    for module in (drive, admission):
        monkeypatch.setattr(module, "_sh", lambda argv, **kw: subprocess.run(
            [sys.executable, *argv[1:]] if argv[0] == "/usr/bin/python3" else list(argv),
            capture_output=True, text=True, timeout=60, **kw))
    return desk


def _calls(path: Path) -> list[list[str]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_launch_leases_and_records_what_ran(data: DataLayout, tools: Path, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    brief = tmp_path / "BRIEF.md"
    brief.write_text("write the result\n")
    monkeypatch.setenv("MENAGERIE_DATA", str(data.root))
    run = drive.launch("2b", "model-implementation", brief, keys=["f-resnet"], parent="p-1",
                       label="smoke")
    assert run == "run-9"
    (start,) = [c for c in _calls(tools / "bin" / "fleet-run.calls") if c[0] == "start"]
    assert start[start.index("--parent") + 1] == "p-1"
    assert start[start.index("--artifact") + 1] == str(tmp_path / "BRIEF.result.md")
    tail = start[start.index("/adapter"):]
    assert tail[:3] == ["/adapter", "codex", str(brief)]
    assert "model_reasoning_effort=xhigh" in tail and "workspace-write" in tail
    assert "sandbox_workspace_write.network_access=false" in tail
    assert tail[tail.index(str(data.root)) - 1] == "--add-dir"
    lease = json.loads((tmp_path / "work" / "leases" / "smoke.json").read_text())
    assert (lease["harness"], lease["model"], lease["keys"]) == ("codex", "m-codex", ["f-resnet"])
    events = (tmp_path / "work" / "lanes" / "run-9" / "events.jsonl").read_text().splitlines()
    assert json.loads(events[0])["verb"] == "launch"
    with pytest.raises(drive.Refused) as held:
        drive.launch("2b", "model-default", brief, keys=["f-resnet"], label="other")
    assert held.value.result["result"] == "lease_held"
    monkeypatch.setenv("CODEX_FULL", "1")
    with pytest.raises(drive.Refused) as refused:
        drive.launch("2b", "model-implementation", brief, keys=["f-x"], label="third")
    assert refused.value.result["result"] == "no_admission"
    monkeypatch.setenv("FLEET_STATE", "TERMINAL")  # the lane ended: its keys are free again
    drive.launch("2b", "model-default", brief, keys=["f-resnet"], label="after")
    starts = [c for c in _calls(tools / "bin" / "fleet-run.calls") if c[0] == "start"]
    assert len(starts) == 2
    assert sorted(p.stem for p in (tmp_path / "work" / "leases").glob("*.json")) == ["after"]


def test_a_claude_lane_runs_on_the_admitted_account_without_web(data: DataLayout, tools: Path,
                                                                 tmp_path: Path) -> None:
    brief = tmp_path / "BRIEF.md"
    brief.write_text("x\n")
    drive.launch("2a", "model-default", brief, label="claude-lane")
    (start,) = _calls(tools / "bin" / "fleet-run.calls")
    assert "CLAUDE_ACCOUNT=acct-a" in start and "CLAUDE_CONFIG_DIR" in start
    assert start[start.index("--disallowedTools") + 1] == drive.WEB_TOOLS
    assert _calls(tools / "bin" / "burst-admission.calls") == [["continue", "--account",
                                                                "acct-a"]]


def test_lanes_releases_ended_and_expired_leases(data: DataLayout, tools: Path,
                                                 tmp_path: Path) -> None:
    _fake(tools / "bin", "fleet-run", "state = 'TERMINAL' if sys.argv[3] == 'r-end' else 'RUNNING'\n"
          "print(json.dumps({'run': {'state': state}}))")
    leases = tmp_path / "work" / "leases"
    leases.mkdir(parents=True)
    for label, run, expires in (("live", "r-live", "2999-01-01T00:00:00Z"),
                                ("ended", "r-end", "2999-01-01T00:00:00Z"),
                                ("old", "r-old", "2000-01-01T00:00:00Z")):
        (leases / f"{label}.json").write_text(json.dumps(
            {"label": label, "run": run, "keys": [], "started": "2026-10-04T00:00:00Z",
             "expires": expires}))
    assert [lane["label"] for lane in drive.lanes(data=data)] == ["live"]
    assert sorted(p.stem for p in leases.glob("*.json")) == ["live"]


def _jsonl(path: Path, *entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


def test_measures_from_events_and_transcripts(data: DataLayout, tools: Path, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runs = tmp_path / "runs"
    claude_log, codex_log = runs / "a" / "log.out", runs / "b" / "log.out"
    _jsonl(claude_log, {"type": "system", "subtype": "init", "session_id": "s-1"})
    _jsonl(codex_log, {"type": "thread.started", "thread_id": "t-1"})
    for name, env in (("a", {}), ("b", {"CODEX_HOME": str(tmp_path / "codex")})):
        (runs / name / "spec.json").write_text(json.dumps({"argv": [], "environment": env}))
    listed = [
        {"run_id": "top", "parent_run_id": None, "kind": "claude", "started_at": 0,
         "ended_at": 600, "state": "TERMINAL", "spec_path": str(runs / "a" / "spec.json"),
         "log_path": str(claude_log), "label": "research"},
        {"run_id": "kid", "parent_run_id": "top", "kind": "codex", "started_at": 0,
         "ended_at": 1200, "state": "TERMINAL", "spec_path": str(runs / "b" / "spec.json"),
         "log_path": str(codex_log), "label": "implement"},
    ]
    _fake(tools / "bin", "fleet-run", f"print(json.dumps({listed!r}))")

    def usage(at: str, mid: str, n: int) -> dict:
        return {"type": "assistant", "timestamp": at, "message": {"id": mid, "usage": {
            "input_tokens": n, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1,
            "output_tokens": 2}, "content": []}}

    _jsonl(tmp_path / "home" / ".claude" / "projects" / "p" / "s-1.jsonl",
           usage("2026-10-04T10:01:00Z", "a", 10), usage("2026-10-04T10:01:00Z", "a", 10),
           usage("2026-10-04T10:06:00Z", "b", 20), usage("2026-10-04T10:20:00Z", "c", 40))
    _jsonl(tmp_path / "codex" / "sessions" / "2026" / "10" / "04" / "rollout-x-t-1.jsonl",
           {"type": "token_usage_record", "timestamp": "2026-10-04T11:00:00Z", "payload": {
               "response_id": "r", "usage": {"input_tokens": 50, "cached_input_tokens": 30,
                                             "output_tokens": 5}}},
           {"type": "response_item", "timestamp": "2026-10-04T11:01:00Z", "payload": {
               "type": "custom_tool_call", "input": "exec_command({cmd:\"curl -sI x\"})"}})
    lanes_dir = tmp_path / "work" / "lanes"
    _jsonl(lanes_dir / "top" / "events.jsonl",
           {"at": "2026-10-04T10:00:00Z", "family": "", "verb": "launch", "kind": "2a",
            "role": "model-default", "harness": "claude", "model": "m-claude"},
           {"at": "2026-10-04T10:00:00Z", "family": "f-resnet", "verb": "fetch code"},
           {"at": "2026-10-04T10:02:00Z", "family": "f-resnet", "verb": "packet write"},
           {"at": "2026-10-04T10:05:00Z", "family": "f-beit", "verb": "fetch hf", "tier": "hub"},
           {"at": "2026-10-04T10:10:00Z", "family": "f-beit", "verb": "packet write"})
    _jsonl(lanes_dir / "kid" / "events.jsonl",
           {"at": "2026-10-04T11:00:00Z", "family": "", "verb": "launch", "kind": "2b",
            "harness": "codex"},
           {"at": "2026-10-04T11:00:00Z", "family": "f-resnet", "verb": "try"},
           {"at": "2026-10-04T11:30:00Z", "family": "f-resnet", "verb": "try"})
    found = drive.measures("top", data=data)
    lanes = {row["run"]: row for row in found["lanes"]}
    assert lanes["top"]["wall_min"] == 10.0
    assert (lanes["top"]["tokens_in"], lanes["top"]["tokens_out"]) == (70, 6)
    assert (lanes["kid"]["tokens_in"], lanes["kid"]["tokens_cached"]) == (20, 30)
    assert lanes["kid"]["network"] == ['exec_command({cmd:"curl -sI x"})']
    families = {row["family"]: row for row in found["families"]}
    resnet, beit = families["f-resnet"], families["f-beit"]
    assert (resnet["tier"], resnet["min_2a"], resnet["min_2b"]) == ("library", 2.0, 30.0)
    assert (resnet["fetches"], resnet["tries"], resnet["network"]) == (1, 2, True)
    assert resnet["tokens_in"] == 10 + 20  # its 2a window plus the whole one-family 2b lane
    assert (beit["tier"], beit["tokens_in"], beit["network"]) == ("hub", 20, False)
    assert sorted(Path(f).name for f in found["files"]) == ["lanes.tsv", "measures.md",
                                                            "measures.tsv"]
    assert "| library | 1 |" in (lanes_dir / "top" / "measures.md").read_text()


def test_measures_says_unknown_without_a_transcript_or_run(data: DataLayout, tools: Path,
                                                            tmp_path: Path) -> None:
    listed = [{"run_id": "r", "parent_run_id": None, "kind": "claude", "started_at": 0,
               "ended_at": 60, "state": "TERMINAL"}]
    _fake(tools / "bin", "fleet-run", f"print(json.dumps({listed!r}))")
    (row,) = drive.measures("r", data=data)["lanes"]
    assert (row["wall_min"], row["tokens_in"], row["model"]) == (1.0, "unknown", "unknown")
    assert drive.measures("nope", data=data)["lanes"] == []


def test_collect_lists_changed_data_files_and_commits_only_when_asked(
        tmp_path: Path, tools: Path) -> None:
    repo = tmp_path / "repo"
    data = DataLayout(repo / "menagerie" / "data")
    data.root.mkdir(parents=True)
    (data.root / "kept.json").write_text("{}\n")
    (repo / "README").write_text("x\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    (data.root / "kept.json").write_text('{"a": 1}\n')
    (data.root / "ledger").mkdir()
    (data.root / "ledger" / "2026-10.jsonl").write_text("{}\n")
    (data.root / "ledger" / ".2026-10.jsonl.tmp").write_text("{}\n")
    (repo / "README").write_text("y\n")
    _fake(tools / "bin", "git-commit-paths", "")
    expected = ["menagerie/data/kept.json", "menagerie/data/ledger/2026-10.jsonl"]
    assert sorted(drive.collect(data=data)) == expected
    assert not (tools / "bin" / "git-commit-paths.calls").exists()
    drive.collect(data=data, commit=True)
    (call,) = _calls(tools / "bin" / "git-commit-paths.calls")
    assert call[:2] == ["-C", str(repo)] and sorted(call[call.index("--") + 1:]) == expected
    assert call[call.index("-m") + 1] == "data: collect 2 files (kept.json 1, ledger 1)"


def test_renames_name_only_their_new_path() -> None:
    assert drive._changed("R  new.json\0old.json\0 M a.json\0?? b.json\0") == [
        "new.json", "a.json", "b.json"]


# --- rung 2: the project account, the refusal rules, Google lanes, gauges, the check-in -------


def test_claude_lanes_start_on_the_project_account(tools: Path, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MENAGERIE_CLAUDE_ACCOUNT")
    _fake(tools / "tools", "lane_admission.py",
          "import os\nif os.environ.get('CLAUDE_FULL'):\n    sys.exit(75)\n"
          "print('CLAUDE_ACCOUNT=' + os.environ['CLAUDE_ACCOUNT'])")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    assert admission._admit("claude", "model-default", checkout)[1] == "pool"
    (checkout / ".claude-account").write_text("org\n")
    words, account = admission._admit("claude", "model-default", checkout)
    assert account == "org" and "CLAUDE_ACCOUNT=org" in words
    monkeypatch.setenv("MENAGERIE_CLAUDE_ACCOUNT", "pool")
    assert admission._admit("claude", "model-default", checkout)[1] == "pool"
    monkeypatch.delenv("MENAGERIE_CLAUDE_ACCOUNT")
    assert _calls(tools / "bin" / "burst-admission.calls")[-1] == ["continue", "--account", "pool"]
    monkeypatch.setenv("CLAUDE_FULL", "1")  # the gate and burst check hold for any account
    with pytest.raises(drive.Refused) as refused:
        admission._admit("claude", "model-default", checkout)
    assert refused.value.result["account"] == "org"


def _events(tmp_path: Path, run: str) -> list[dict]:
    path = tmp_path / "work" / "lanes" / run / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_refused_lab_hands_the_lane_on_and_none_queues_it(
        data: DataLayout, tools: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(tools / "tools", "lane_admission.py",
          "import os\nif os.environ.get('CLAUDE_FULL'):\n    sys.exit(75)\n"
          "print('CLAUDE_ACCOUNT=acct-a')")
    brief = tmp_path / "BRIEF.md"
    brief.write_text("x\n")
    roles = ("model-implementation", "model-default")
    monkeypatch.setenv("CODEX_FULL", "1")
    tier = [{"family": "f-resnet", "verb": "tier", "tier": "library"}]
    assert drive.launch_first("2a", roles, brief, label="moved", on_start=tier) == "run-9"
    (start,) = _calls(tools / "bin" / "fleet-run.calls")
    assert start[start.index("/adapter") + 1] == "claude"
    verbs = [(e["verb"], e.get("role"), e.get("tier")) for e in _events(tmp_path, "run-9")]
    assert verbs == [("launch", "model-default", None), ("refused", "model-implementation", None),
                     ("tier", None, "library")]
    monkeypatch.setenv("CLAUDE_FULL", "1")
    assert drive.launch_first("2a", roles, brief, label="later", on_start=tier) == "queued:later"
    (step,) = drive.queued()
    assert step["trigger"] == "any of model-implementation, model-default admitted"
    assert drive.requeue() == [{"label": "later", "run": "queued:later"}]
    assert drive.queued()[0]["queued"] == step["queued"]  # a requeue keeps its place
    monkeypatch.delenv("CLAUDE_FULL")
    monkeypatch.setenv("FLEET_STATE", "TERMINAL")
    assert drive.requeue() == [{"label": "later", "run": "run-9"}]
    assert drive.queued() == []
    with pytest.raises(drive.Refused):
        monkeypatch.setenv("CLAUDE_FULL", "1")
        drive.launch_first("2a", roles, brief, label="strict", queue=False)


def test_google_lanes_review_read_only_and_implement_only_last(
        data: DataLayout, tools: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plans = {"model-review-third-lab-plan": {"harness": "antigravity", "model": "g-pro",
                                             "effort": "high", "edits": False},
             "model-implementation-third-lab": {"harness": "antigravity", "model": "g-pro",
                                                "effort": "high", "edits": True}}
    _fake(tools / "bin", "role", f"print(json.dumps({plans!r}[sys.argv[2]]))")
    agy = _fake(tools / "bin", "agy-bg.sh", "print('FLEET_RUN_STARTED run_id=agy-1')")
    monkeypatch.setenv("MENAGERIE_AGY_BG", str(agy))
    folder = tmp_path / "review-lane"
    (folder / "out").mkdir(parents=True)
    brief = folder / "BRIEF.md"
    brief.write_text(f"Write `{folder}/out/c01.json`, then `{folder}/BRIEF.result.md`.\n")
    with pytest.raises(drive.RecordError, match="only after both main labs"):
        drive.launch("2b", "model-implementation-third-lab", brief, label="impl")
    with pytest.raises(drive.RecordError, match="does not run on the Google seat"):
        drive.launch("2a", "model-review-third-lab-plan", brief, label="research")
    assert drive.launch("review", "model-review-third-lab-plan", brief, keys=["m1"],
                        parent="p-2", label="gemini-review") == "agy-1"
    (call,) = _calls(tools / "bin" / "agy-bg.sh.calls")
    assert call[call.index("--role") + 1] == "model-review-third-lab-plan"
    assert call[call.index("--cd") + 1] == str(folder)
    assert "This seat cannot write files" in Path(call[1]).read_text()
    lease = json.loads((tmp_path / "work" / "leases" / "gemini-review.json").read_text())
    answer = Path(lease["answer"])
    answer.write_text("Done.\n" + json.dumps({"files": {
        f"{folder}/out/c01.json": {"id": "c01", "verdict": "minor"},
        f"{tmp_path}/outside.json": {"no": "write"}}}))
    monkeypatch.setenv("FLEET_STATE", "TERMINAL")
    assert drive.lanes(data=data) == []
    assert json.loads((folder / "out" / "c01.json").read_text())["verdict"] == "minor"
    assert (folder / "BRIEF.result.md").read_text() == "wrote c01.json\n"
    assert not (tmp_path / "outside.json").exists()
    launch = _events(tmp_path, "agy-1")[0]
    assert launch["harness"] == "antigravity" and launch["account"] == "gemini-plan"
    assert launch["gauges"]["gemini"] == "gemini-subscription: 2% used"


def test_google_lanes_need_the_fresh_gemini_gate_and_can_answer_in_blocks(
        data: DataLayout, tools: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plans = {"model-review-third-lab-plan": {"harness": "antigravity", "model": "g-pro",
                                             "effort": "high", "edits": False}}
    _fake(tools / "bin", "role", f"print(json.dumps({plans!r}[sys.argv[2]]))")
    _fake(tools / "bin", "agy-quota",
          "import os\nsys.exit(75 if os.environ.get('AGY_FULL') else 0)")
    agy = _fake(tools / "bin", "agy-bg.sh", "print('FLEET_RUN_STARTED run_id=agy-2')")
    monkeypatch.setenv("MENAGERIE_AGY_BG", str(agy))
    folder = tmp_path / "review-lane"
    folder.mkdir()
    brief = folder / "BRIEF-b1.md"
    brief.write_text("Review c01 and c02.\n")
    monkeypatch.setenv("AGY_FULL", "1")
    with pytest.raises(drive.Refused, match="no_admission"):
        drive.launch("review", "model-review-third-lab-plan", brief, label="g-1",
                     answer="blocks", ids=["c01", "c02"])
    gates = _calls(tools / "bin" / "agy-quota.calls")
    assert gates == [["gate", "--require-fresh"]]
    assert not (tools / "bin" / "agy-bg.sh.calls").exists()
    monkeypatch.delenv("AGY_FULL")
    with pytest.raises(drive.RecordError, match="blocks without ids"):
        drive.launch("review", "model-review-third-lab-plan", brief, label="g-1", answer="blocks")
    assert drive.launch("review", "model-review-third-lab-plan", brief, label="g-1",
                        lab="antigravity", answer="blocks", ids=["c01", "c02"]) == "agy-2"
    (call,) = _calls(tools / "bin" / "agy-bg.sh.calls")
    prompt = Path(call[1]).read_text()
    assert "one block per item" in prompt and '{"files"' not in prompt
    lease = json.loads((tmp_path / "work" / "leases" / "g-1.json").read_text())
    assert (lease["answer_kind"], lease["ids"]) == ("blocks", ["c01", "c02"])
    Path(lease["answer"]).write_text("```json\n" + json.dumps(
        {"id": "c01", "verdict": "minor"}) + "\n```\n```json\n" + json.dumps(
        {"id": "c99", "verdict": "faithful"}) + "\n```\nc01 minor\n")
    monkeypatch.setenv("FLEET_STATE", "TERMINAL")
    assert drive.lanes(data=data) == []
    assert json.loads((folder / "out" / "c01.json").read_text())["verdict"] == "minor"
    assert not (folder / "out" / "c99.json").exists()
    assert (folder / "BRIEF-b1.result.md").read_text() == "c01 minor\nc02 missing\n"


def test_one_gauge_reading_serves_the_launches_of_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    asked = []
    clock = [1000.0]
    monkeypatch.setattr(admission, "_GAUGES", {})
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(admission, "_sh", lambda argv, **kw: asked.append(argv[-2]) or
                        subprocess.CompletedProcess(argv, 0, "acct-a: 12% of the week\n", ""))
    first = admission.gauges("acct-a")
    assert admission.gauges("acct-a") == first and len(asked) == 3  # one reading per gauge
    clock[0] += admission.GAUGES_FRESH_S
    admission.gauges("acct-a")
    assert len(asked) == 6  # a stale reading is read again


def test_a_launch_records_the_three_quota_gauges(data: DataLayout, tools: Path,
                                                 tmp_path: Path) -> None:
    brief = tmp_path / "BRIEF.md"
    brief.write_text("x\n")
    drive.launch("2a", "model-default", brief, label="gauged")
    (launch,) = _events(tmp_path, "run-9")
    assert launch["gauges"]["claude"] == "acct-a: 12% of the week"
    assert set(launch["gauges"]) == {"claude", "codex", "gemini"}


def test_checkin_counts_the_rung_rows_without_fleet_paths(data: DataLayout, tools: Path,
                                                         tmp_path: Path) -> None:
    trace, render = _rows()
    _ledger(data, _forward(trace), trace, render)
    rows = data.root / "catalog" / "r2-rows.tsv"
    rows.parent.mkdir(parents=True)
    rows.write_text("id\tname\tzoo\tvariant\tcase\tcounted\n"
                    "m8839\tresnet18\ttimm\t\tlibrary\t1\n")
    (rows.parent / "plan.tsv").write_text("id\tgroup\twave\nm8839\tA\tC1\n")
    _fake(tools / "bin", "fleet-run", "print('[]')")
    out = drive.checkin("run-r2", data=data, out=tmp_path / "rungs" / "r2-checkin.md")
    text = out.read_text()
    assert "| library | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 |" in text
    assert "| A | C1 | 1 | 0 | 1 | 0 | 0 |" in text  # traced; the fixture is not validated
    assert "`f-resnet`" in text and "No lanes under run `run-r2`" in text
    assert str(tmp_path) not in text and "/home/" not in text


def test_a_project_account_outside_the_pool_keeps_claude_roles_on_claude(
        tools: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Background routing moved model-default to Codex because the pool is short.
    _fake(tools / "bin", "role", "claude = {'harness': 'claude', 'model': 'm-claude'}\n"
          "codex = {'harness': 'codex', 'model': 'm-codex'}\n"
          "print(json.dumps(codex if '--background' in sys.argv else claude))")
    assert admission._resolve("model-default", "pool")["harness"] == "codex"
    plan = admission._resolve("model-default", "org")
    assert (plan["harness"], plan["routing"]["reason"]) == ("claude", "project account org")


def test_a_claude_only_placement_keeps_the_roles_own_seat(
        tools: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Background routing moved the role to Codex, which a Claude-only worker cannot run.
    _fake(tools / "bin", "role", "claude = {'harness': 'claude', 'model': 'm-claude'}\n"
          "codex = {'harness': 'codex', 'model': 'm-codex'}\n"
          "print(json.dumps(codex if '--background' in sys.argv else claude))")
    plan = admission._resolve("model-default", "pool", harness="claude")
    assert (plan["harness"], plan["routing"]["reason"]) == ("claude", "placement runs claude only")


def test_a_claude_only_placement_gets_the_routed_seat_when_the_role_has_no_claude_one(
        tools: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(tools / "bin", "role",
          "print(json.dumps({'harness': 'codex', 'model': 'm-codex'}))")
    assert admission._resolve("model-implementation", "pool", harness="claude")["harness"] == "codex"



def test_a_held_google_seat_is_refused_and_the_switch_is_read_from_the_tick_config(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    assert admission.third_lab_held() is False
    config = tmp_path / "work" / "tick" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"third_lab_hold": True}))
    assert admission.third_lab_held() is True
    for role in (admission.THIRD_LAB_IMPLEMENT, admission.THIRD_LAB_REVIEW):
        with pytest.raises(admission.Refused) as refused:
            admission._resolve(role)
        assert refused.value.result["result"] == "no_admission"
    config.write_text(json.dumps({"third_lab_hold": "yes"}))
    with pytest.raises(ValueError, match="third_lab_hold"):
        admission.third_lab_held()

def test_codex_review_lanes_take_the_gate_ceiling_from_the_environment(
        tools: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = tools / "tools" / "chatgpt_usage.py.calls"
    admission._admit("codex", "model-design-review", tmp_path, "review")
    monkeypatch.setenv("MENAGERIE_CODEX_MAX_PCT", "88")
    admission._admit("codex", "model-design-review", tmp_path, "review")
    admission._admit("codex", "model-implementation", tmp_path, "2b")  # review lanes only
    assert [c[1:] for c in _calls(calls)[-3:]] == [
        ["--require-fresh"], ["--require-fresh", "--max-pct", "88"], ["--require-fresh"]]


def test_describe_lanes_take_a_lower_claude_ceiling_never_a_higher_one(
        monkeypatch: pytest.MonkeyPatch) -> None:
    assert admission._claude_ceiling("describe") == admission.UNATTENDED_MAX_PCT
    monkeypatch.setenv("MENAGERIE_DESCRIBE_MAX_PCT", "60")
    assert admission._claude_ceiling("describe") == 60
    assert admission._claude_ceiling("2a") == admission._claude_ceiling("2b") == 70
    monkeypatch.setenv("MENAGERIE_DESCRIBE_MAX_PCT", "90")
    assert admission._claude_ceiling("describe") == 70


@pytest.mark.parametrize(("cls", "placement", "host"), [
    ("needs_gpu", "gpu-window", "cluster"), ("oversized", "axon-cpu", "cluster")])
def test_a_kept_draft_that_passes_where_it_runs_flows_on(data: DataLayout, cls: str,
                                                         placement: str, host: str) -> None:
    trace, render = _rows()
    record = read_json(ModelRecord, data.model("m8839"))
    give = GiveUp(cls=cls, reason="the draft runs elsewhere", at="2026-10-05T23:00:00Z")
    _record(data, recipe=dataclasses.replace(record.recipe, placement=placement), give_up=give)
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == f"gave up: {cls}"
    failed = dataclasses.replace(trace, host=host, outcome="fail", cls="capture_error",
                                 trace=None, validation_state=None)
    _ledger(data, failed)
    assert _step(data) == "done"  # a failed attempt leaves the give-up typed
    passed = dataclasses.replace(trace, host=host)
    _ledger(data, failed, passed)
    assert _step(data) == "render"  # no worker forward first: the attempt ran its own
    _ledger(data, failed, passed, render)
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == "library: traced and rendered"
    from menagerie.pipeline import checkin

    model = checkin._Model({"id": "m8839"}, drive._load(data), "done")
    assert "typed" not in model.stages() and model.failure() is None  # counted traced, not typed
    other = GiveUp(cls="missing_dependency:x", reason="r", at=give.at)
    _record(data, give_up=other)  # only a give-up that keeps its draft flows on
    assert _step(data) == "done"
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == "gave up: missing_dependency:x"


def test_a_render_that_cannot_finish_is_typed(data: DataLayout) -> None:
    """Two failed renders of a revision's traces end the row typed ``render``; so does a passing
    trace that saved nothing to draw; a later passing render settles it instead."""

    from menagerie.pipeline import checkin
    from menagerie.pipeline.records import RenderSkip

    trace, render = _rows()
    skip = (RenderSkip(kind="svg", why="render:timeout", variant="unrolled-none"),)
    failed = dataclasses.replace(render, outcome="fail", renders=(), skipped=skip)
    _ledger(data, _forward(trace), trace, failed)
    assert _step(data) == "render"  # one failure: the next render may draw it
    again = dataclasses.replace(failed, at="2026-10-06T03:26:03Z")
    _ledger(data, _forward(trace), trace, failed, again)
    (row,) = drive.next_steps(["m8839"], data=data)
    assert row["step"] == "done" and "2 renders failed" in row["why"]
    assert "render:timeout (unrolled-none)" in row["why"]
    model = checkin._Model({"id": "m8839"}, drive._load(data), "done")
    assert model.failure() == "render"
    _ledger(data, _forward(trace), trace, failed, again, render)
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == "library: traced and rendered"
    assert checkin._Model({"id": "m8839"}, drive._load(data), "done").failure() is None

    nothing = dataclasses.replace(trace, trace=None, summary=None)
    _ledger(data, _forward(trace), nothing)
    assert _step(data) == "done"
    assert checkin._Model({"id": "m8839"}, drive._load(data), "done").failure() == "render"


def test_type_render_records_the_evidence_at_the_recipe_revision(data: DataLayout) -> None:
    trace, _ = _rows()
    _ledger(data, _forward(trace), trace)
    why = "render mode could not load the save: TorchLensIOError ... (attempt m8839-render-x)"
    typed = drive.type_render("m8839", why, data=data)
    assert typed.give_up is not None and (typed.give_up.cls, typed.give_up.reason) == (
        "render", why)
    (row,) = drive.next_steps(["m8839"], data=data)  # the give-up settles the row
    assert (row["step"], row["why"]) == ("done", "gave up: render")
    last = read_jsonl(parse_ledger_line, data.ledger(typed.give_up.at[:7]))[-1]
    assert (last.outcome, last.cls, last.recipe_rev) == ("giveup", "render", typed.recipe.rev)


def test_a_gpu_window_recipe_skips_the_worker_forward(data: DataLayout) -> None:
    record = read_json(ModelRecord, data.model("m8839"))
    _record(data, recipe=dataclasses.replace(record.recipe, placement="gpu-window"))
    assert _step(data) == "trace"


# --- rung 3: the account move, unknown quota and the lab guard (BUILD-PLAN r3 item 4) ---------


def test_the_environment_account_wins_over_the_marker(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".claude-account").write_text("org\n")
    monkeypatch.setenv("MENAGERIE_CLAUDE_ACCOUNT", "pool")  # the tick unit sets it
    assert admission.claude_account(tmp_path) == "pool"
    monkeypatch.setenv("MENAGERIE_CLAUDE_ACCOUNT", " ")
    assert admission.claude_account(tmp_path) == "org"
    monkeypatch.delenv("MENAGERIE_CLAUDE_ACCOUNT")
    assert admission.claude_account(tmp_path) == "org"
    assert admission.claude_account(tmp_path / "elsewhere") == "pool"


def test_unknown_quota_is_refused(tools: Path, tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(tools / "tools", "lane_admission.py",
          "import os\nif sys.argv[1] == 'gate' and os.environ.get('QUOTA_UNKNOWN'):\n"
          "    print('acct-a: no fresh reading (unknown)', file=sys.stderr)\n    sys.exit(75)\n"
          "print('CLAUDE_ACCOUNT=acct-a')")
    assert admission._admit("claude", "model-default", tmp_path)[1] == "acct-a"
    gate = _calls(tools / "tools" / "lane_admission.py.calls")[-1]
    assert gate == ["gate", "acct-a", "--max-pct", "70", "--require-fresh"]
    bursts = len(_calls(tools / "bin" / "burst-admission.calls"))
    monkeypatch.setenv("QUOTA_UNKNOWN", "1")
    with pytest.raises(drive.Refused, match="no fresh reading") as refused:
        admission._admit("claude", "model-default", tmp_path)
    assert refused.value.result["account"] == "acct-a"
    assert len(_calls(tools / "bin" / "burst-admission.calls")) == bursts  # before the burst


def test_a_lab_tied_step_refuses_the_other_lab_and_queues(
        data: DataLayout, tools: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    brief = tmp_path / "BRIEF.md"
    brief.write_text("x\n")
    moved = {"model-default": {"harness": "codex", "model": "m-codex", "effort": "high",
                               "routing": {"reason": "pool short"}}}
    _fake(tools / "bin", "role", f"print(json.dumps({moved!r}[sys.argv[2]]))")
    with pytest.raises(drive.Refused, match="resolves to codex") as refused:
        drive.launch("review", "model-default", brief, label="prose", lab="claude")
    assert refused.value.result["result"] == "lab_mismatch"
    assert not (tools / "tools" / "chatgpt_usage.py.calls").exists()  # no quota read
    assert drive.launch_first("review", ("model-default",), brief, label="tied",
                              lab="claude") == "queued:tied"
    (step,) = drive.queued()
    assert step["lab"] == "claude" and step["refusals"][0]["result"] == "lab_mismatch"
    assert drive.requeue() == [{"label": "tied", "run": "queued:tied"}]
    assert drive.launch("review", "model-default", brief, label="free") == "run-9"
    with pytest.raises(drive.RecordError, match="lab 'anthropic'"):
        drive.launch("review", "model-default", brief, label="bad", lab="anthropic")
    _fake(tools / "bin", "role", "print(json.dumps({'harness': 'claude', 'model': 'm'}))")
    assert drive.requeue() == [{"label": "tied", "run": "run-9"}]


SCRIPTED = ("june_code:flagged", "standin:no_published_config", "standin:kwargs",
            "needs_adapter", "needs_adapter:inputs", "construct_error", "missing_dependency",
            "missing_dependency:sdist", "needs_env:mmlab")


@pytest.mark.parametrize("cls", SCRIPTED)
def test_a_scripted_give_up_owes_the_row_one_agent_attempt(data: DataLayout, cls: str) -> None:
    """Coverage rule (decisions.md 2026-10-08 9:56 PM PT): a scripted verdict never settles a
    row; the agent's own give-up does."""

    give = GiveUp(cls=cls, reason="typed by a script", at="2026-10-09T10:00:00Z", by="script")
    _record(data, recipe=None, give_up=give)
    assert drive.next_steps(["m8839"], data=data)[0] == {
        "id": "m8839", "step": "2b", "why": f"scripted route typed {cls}: owed one agent attempt"}
    _record(data, recipe=None, give_up=dataclasses.replace(give, by=None))
    assert _step(data) == "2b"  # a give-up from before `by`, with no agent's ledger row
    _record(data, recipe=None, give_up=dataclasses.replace(give, by="agent"))
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == f"gave up: {cls}"


def test_a_harvester_packet_with_no_source_goes_to_2a_first(data: DataLayout) -> None:
    """A June-catalog stub packet (no source, ``harvester@menagerie``) is not research: 2b on it
    has nothing to read, so the owed attempt and a recipe-less row start at 2a; a 2a write
    (its own ``by``) ends that, even with no source; a recipe still settles the research steps."""

    packet = read_json(Packet, data.packet("f-resnet"))
    stub = dataclasses.replace(packet, by="harvester@menagerie", sources=(), code_order=(),
                               citation=None, claims=())
    write_json(stub, data.packet("f-resnet"))
    give = GiveUp(cls="standin:no_published_config", reason="typed by a script",
                  at="2026-10-09T10:00:00Z", by="script")
    _record(data, recipe=None, give_up=give)
    assert drive.next_steps(["m8839"], data=data)[0] == {
        "id": "m8839", "step": "2a",
        "why": "scripted route typed standin:no_published_config: owed one agent attempt; "
               "packet f-resnet has no sources: 2a research first"}
    _record(data, recipe=None, give_up=None)
    assert drive.next_steps(["m8839"], data=data)[0] == {
        "id": "m8839", "step": "2a", "why": "packet f-resnet has no sources: 2a research first"}
    _record(data, recipe=None, give_up=dataclasses.replace(give, by="agent"))
    assert _step(data) == "done"  # an agent's own give-up still settles the row
    write_json(dataclasses.replace(stub, by="2a@prompt-v2 model-default (claude m)"),
               data.packet("f-resnet"))
    _record(data, recipe=None, give_up=give)
    assert _step(data) == "2b"  # 2a has written the packet: its verdict stands
    write_json(packet, data.packet("f-resnet"))
    assert _step(data) == "2b"  # a harvester packet with a library source is ready for 2b


def test_a_give_up_from_before_the_author_field_reads_the_ledger(data: DataLayout) -> None:
    give = GiveUp(cls="missing_dependency", reason="mmcv-full has no wheels",
                  at="2026-10-04T19:10:18Z")
    record = _record(data, recipe=None, give_up=give)
    assert not drive.agent_tried(record)
    other = LedgerRow(id="m8839", stage="forward", at="2026-10-04T19:00:00Z", host="worker",
                      env="torch-main", recipe_rev="0" * 12, outcome="giveup",
                      cls="missing_dependency")
    _ledger(data, other)  # a different give-up time: not this give-up's row
    assert _step(data) == "2b"
    _ledger(data, other, dataclasses.replace(other, at=give.at))  # what implement.giveup appended
    assert drive.agent_tried(record, [dataclasses.replace(other, at=give.at)])
    assert drive.next_steps(["m8839"], data=data)[0]["why"] == "gave up: missing_dependency"
    _record(data, recipe=None, give_up=dataclasses.replace(give, by="script"))
    assert _step(data) == "2b"  # an explicit author wins over the ledger


@pytest.mark.parametrize(("cls", "step"), [
    ("not_found", "2a"), ("inaccessible", "2a"), ("standin:compact", "2b")])
def test_an_unknown_source_owes_research_not_a_recipe(data: DataLayout, cls: str,
                                                      step: str) -> None:
    _record(data, recipe=None, give_up=GiveUp(cls=cls, reason="r", at="2026-10-09T10:00:00Z",
                                              by="script"))
    assert _step(data) == step


@pytest.mark.parametrize(("cls", "with_recipe", "step"), [
    ("render", True, "done"),             # traced: drawing it is not implementing it
    ("memory_cap", True, "done"),         # a run of a recipe that is there
    ("standin:dense", True, "2b"),        # a recipe that is a substitute: no faithful one yet
    ("june_code:flagged", True, "2b"),
    ("not_architecture", False, "done"),
    ("needs_env:no-such-env", False, "done"),  # waits for its environment, then reopens
])
def test_what_a_scripted_give_up_does_not_reopen(data: DataLayout, cls: str, with_recipe: bool,
                                                step: str) -> None:
    record = read_json(ModelRecord, data.model("m8839"))
    give = GiveUp(cls=cls, reason="r", at="2026-10-09T10:00:00Z", by="script")
    _record(data, recipe=record.recipe if with_recipe else None, give_up=give)
    (row,) = drive.next_steps(["m8839"], data=data)
    assert row["step"] == step, row


def test_scripted_writers_sign_their_give_ups(data: DataLayout) -> None:
    trace, render = _rows()
    _ledger(data, _forward(trace), trace)
    typed = drive.type_render("m8839", "two renders failed", data=data)
    assert typed.give_up.by == "script"
    assert read_json(ModelRecord, data.model("m8839")).give_up.by == "script"


def test_the_stand_in_sweeps_ledger_row_is_not_an_agents(data: DataLayout) -> None:
    """``standin.sweep`` (a script) appended a give-up ledger row like ``implement.giveup``'s."""

    record = read_json(ModelRecord, data.model("m8839"))
    give = GiveUp(cls="standin:dense", reason="stand-in: recipe revision 965234c715ef was typed",
                  at="2026-10-05T10:00:00Z")
    _record(data, recipe=record.recipe, give_up=give)
    row = LedgerRow(id="m8839", stage="forward", at=give.at, host="worker", env="torch-main",
                    recipe_rev="0" * 12, outcome="giveup", cls="standin:dense")
    _ledger(data, row)
    assert _step(data) == "2b"
    trace, _ = _rows()
    _ledger(data, row, _forward(trace), trace)  # the stand-in traced: still not the model
    assert _step(data) == "2b"
    _record(data, recipe=record.recipe, give_up=dataclasses.replace(give, reason="an agent's"))
    assert _step(data) == "done"


def test_an_unknown_source_answered_by_2a_goes_to_2b(data: DataLayout) -> None:
    """One 2a pass per scripted unknown-source give-up: once 2a amends the packet, 2b tries."""

    give = GiveUp(cls="not_found", reason="r", at="2026-10-09T10:00:00Z", by="script")
    _record(data, recipe=None, give_up=give)
    packet = read_json(Packet, data.packet("f-resnet"))
    before = Amendment(at="2026-10-09T09:00:00Z", why="earlier", evidence=("s2",),
                       changed=("sources.s2",), by="integration")
    write_json(dataclasses.replace(packet, amendments=(before,)), data.packet("f-resnet"))
    assert _step(data) == "2a"
    after = dataclasses.replace(before, at="2026-10-09T11:00:00Z", why="the code found")
    write_json(dataclasses.replace(packet, amendments=(before, after)), data.packet("f-resnet"))
    assert _step(data) == "2b"


def _age(path: Path, seconds: float = 3600) -> None:
    """Back-date a file past the memo's racy window (a fresh file is always re-read)."""

    when = int((datetime.now(timezone.utc).timestamp() - seconds) * 1e9)
    os.utime(path, ns=(when, when))


def test_the_load_memo_rereads_only_a_changed_shard(data: DataLayout,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    trace, render = _rows()
    _ledger(data, _forward(trace))
    june = data.ledger("2026-06")
    june.write_text(dumps(dataclasses.replace(render, id="m3631")) + "\n", encoding="utf-8")
    for path in (june, data.ledger("2026-10"), data.roster):
        _age(path)
    parsed: list[str] = []
    real = drive._ledger_line
    monkeypatch.setattr(drive, "_ledger_line", lambda raw: parsed.append(raw["stage"]) or real(raw))
    assert _step(data) == "trace"
    first = len(parsed)
    assert _step(data) == "trace" and len(parsed) == first  # nothing changed: nothing re-read
    append_jsonl([trace], data.ledger("2026-10"))
    _age(data.ledger("2026-10"), 1800)
    assert _step(data) == "render" and parsed[first:] == [trace.stage]  # the append, alone
    assert drive._load(data).rows["m3631"] == [dataclasses.replace(render, id="m3631")]


def test_the_memo_sees_a_new_or_rewritten_record(data: DataLayout) -> None:
    record = read_json(ModelRecord, data.model("m8839"))
    data.model("m8839").unlink()
    assert _step(data) == "2b"
    write_json(record, data.model("m8839"))
    _age(data.model("m8839"))  # an old-looking new file still counts: its stamp changed
    assert _step(data) == "forward"
    write_json(dataclasses.replace(record, recipe=None), data.model("m8839"))
    assert _step(data) == "2b"


def test_an_appended_file_is_parsed_only_past_what_was_read(tmp_path: Path) -> None:
    path, built = tmp_path / "x.jsonl", []
    memo = drive._Memo()

    def read() -> list:
        return memo.appended(path, lambda raw: built.append(raw["n"]) or raw["n"])

    assert read() == [] and built == []  # a missing file has no rows
    path.write_bytes(b'{"n": 1}\n{"n": 2}\n{"n": 3')  # the last line is still being written
    assert read() == [1, 2] and built == [1, 2]
    with path.open("ab") as handle:
        handle.write(b'}\n{"n": 4}')
    assert read() == [1, 2, 3, 4] and built == [1, 2, 3, 4]  # whole lines kept; 4 read again
    assert read() == [1, 2, 3, 4] and built == [1, 2, 3, 4, 4]
    path.write_bytes(b'{"n": 9}\n{"n": 2}\n{"n": 3}\n{"n": 5}\n')  # rewritten, not appended
    assert read() == [9, 2, 3, 5]
    path.unlink()
    path.write_bytes(b'{"n": 7}\n')  # a new file at the same path
    assert read() == [7]


def test_a_file_read_inside_the_racy_window_is_read_again(tmp_path: Path) -> None:
    path, reads = tmp_path / "x.json", []
    path.write_text("1")
    memo = drive._Memo()
    for _ in range(2):
        memo.get(path, lambda p: reads.append(p.read_text()) or p.read_text())
    assert len(reads) == 2  # fresh: never trusted
    _age(path)
    for _ in range(2):
        memo.get(path, lambda p: reads.append(p.read_text()) or p.read_text())
    assert len(reads) == 3  # old and unchanged: read once more, then served


def test_a_filtered_load_equals_the_full_one_for_its_ids(data: DataLayout) -> None:
    trace, render = _rows()
    _ledger(data, _forward(trace), trace, dataclasses.replace(render, id="m3631"))
    full, some = drive._load(data), drive._load(data, ids={"m8839"})
    assert some.rows == {"m8839": full.rows["m8839"]} and list(some.roster) == ["m8839"]
    assert drive._step("m8839", some) == drive._step("m8839", full)
    assert drive._load(data, ids=set()).rows == {}
