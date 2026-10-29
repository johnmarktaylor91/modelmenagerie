"""``stage3``: queue from a lane's terminal event, serve through a runner spool, collect on main.

Torch-free: a stand-in thread serves the runner spool and answers each request, so the jobs the
hub queues, the attempts the worker asks for and the lines collect appends are checked as the
real runner would see them. The spool is local (no ``MENAGERIE_RUNNER_HOST``), so ``Spool``'s
shell commands run here exactly as they would over ssh.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.pipeline import (
    DataLayout, StoreLayout, envs, flow, read_json, recipe_revision, stage3, write_json,
)
from menagerie.pipeline.records import (
    LedgerRow, Member, ModelRecord, Modes, Packet, RecordError, RunRequest, RunResult, from_dict,
    to_dict,
)

FIXTURES = Path(__file__).with_name("fixtures")
ADAPTER = '''"""A toy adapter."""
import torch


def build(cfg):
    return torch.nn.Linear(4, 8)
'''
AT = "2026-10-05T00:00:00Z"
Answer = Callable[[RunRequest], RunResult]


def _current(record: ModelRecord, data: DataLayout) -> ModelRecord:
    """The record with its recipe revision recomputed from today's adapter and lock."""

    assert record.recipe is not None
    adapter = (data.root / record.recipe.adapter).read_bytes() if record.recipe.adapter else None
    rev = recipe_revision(to_dict(record.recipe), adapter,
                          envs.lock_path(record.recipe.env).read_bytes())
    return replace(record, recipe=replace(record.recipe, rev=rev))


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """resnet18's library record (m8839) and an adapter member (m9001) of family f-resnet."""

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("MENAGERIE_CPUS", "2")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "16")
    monkeypatch.delenv("MENAGERIE_RUNNER_HOST", raising=False)
    monkeypatch.delenv("MENAGERIE_STORE_REMOTE", raising=False)
    layout = DataLayout(tmp_path / "data")
    layout.adapter("f-resnet").parent.mkdir(parents=True, exist_ok=True)
    layout.adapter("f-resnet").write_text(ADAPTER, encoding="utf-8")
    resnet = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    assert resnet.recipe is not None
    write_json(_current(resnet, layout), layout.model("m8839"))
    recipe = replace(resnet.recipe, kind="adapter", adapter="adapters/f-resnet.py", call=None)
    toy = _current(replace(resnet, id="m9001", name="toy", recipe=recipe), layout)
    write_json(toy, layout.model("m9001"))
    packet = read_json(Packet, FIXTURES / "f-resnet.packet.json")
    write_json(replace(packet, members=(*packet.members, Member(id="m9001"))),
               layout.packet("f-resnet"))
    return layout


def _answer(request: RunRequest, *, peak_mb: int = 1024, trace_cls: str | None = None) -> RunResult:
    """Forward passes with a measured peak; a trace passes unless ``trace_cls`` is given."""

    fail = request.mode == "trace" and trace_cls is not None
    stage = "forward" if request.mode == "forward" else "trace"
    capture = None if stage == "forward" else (
        "structure_only" if request.mode == "trace-structure-only" else "full")
    row = LedgerRow(id=request.id, stage=stage, at=AT, host="worker", env="torch-main@00000000",
                    recipe_rev=request.recipe.rev, outcome="fail" if fail else "pass",
                    cls=trace_cls if fail else None, mode=request.model_mode, capture=capture,
                    validation_state="structure_only" if capture == "structure_only" else None,
                    forward_peak_rss_mb=peak_mb if stage == "forward" else None,
                    msg=request.request)
    return RunResult(request=request.request, id=request.id, mode=request.mode,
                     outcome=row.outcome, started=AT, ended=AT, host="worker", cls=row.cls,
                     modes=Modes(eval="pass", train="pass", divergence="none")
                     if stage == "forward" else None, rows=(row,))


@pytest.fixture
def runner_spool(tmp_path: Path) -> Iterator[tuple[Path, list[RunRequest], dict[str, Answer]]]:
    """A stand-in runner on ``tmp_path/spool``: yields the spool, the requests it saw and a slot
    holding its answer function."""

    path, seen, answer = tmp_path / "spool", [], {"fn": _answer}
    stop = threading.Event()

    def serve() -> None:
        for name in ("new", "done", "attempts"):
            (path / name).mkdir(parents=True, exist_ok=True)
        while not stop.is_set():
            (path / "serving").touch()
            for item in sorted((path / "new").glob("*.json")):
                request = read_json(RunRequest, item)
                item.unlink()
                seen.append(request)
                write_json(answer["fn"](request), path / "done" / item.name)
            time.sleep(0.02)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield path, seen, answer
    stop.set()
    thread.join(timeout=5)


def _terminal(monkeypatch: pytest.MonkeyPatch, state: str = "TERMINAL") -> None:
    monkeypatch.setattr(stage3, "_lane_state", lambda lane: state)


def test_lane_state_reads_fleet_run_through_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lane's state comes from fleet-run's status, which r2-lanes moved into ``admission``."""

    from menagerie.pipeline import admission

    seen = []
    monkeypatch.setattr(admission, "_status",
                        lambda run: seen.append(run) or {"run": {"state": "TERMINAL"}})
    assert stage3._lane_state("01a10000-0000-7000-8000-000000000000") == "TERMINAL"
    monkeypatch.setattr(admission, "_status", lambda run: None)
    assert stage3._lane_state("01a10000-0000-7000-8000-000000000001") == "unknown"
    assert seen == ["01a10000-0000-7000-8000-000000000000"]


def _lane_event(lane: str, family: str, verb: str, **fields: object) -> None:
    from menagerie.pipeline import drive

    drive.event(lane, family, verb, id="m8839", **fields)


# --- flow sizing (fix 10) ---------------------------------------------------------------------


def _forward(peak_mb: int | None) -> RunResult:
    request = RunRequest(request="m8839-forward-eval-x-0000", id="m8839", mode="forward",
                         recipe=read_json(ModelRecord, FIXTURES / "m8839.model.json").recipe,
                         caps=flow.Caps(rss_mb=1024, timeout_s=60, threads=1))
    return replace(_answer(request, peak_mb=peak_mb or 0),
                   rows=() if peak_mb is None else _answer(request, peak_mb=peak_mb).rows)


@pytest.mark.parametrize(("peak_gib", "share", "rss_gib", "placement"), [
    (1, 56, 8.0, None),         # the floor
    (2, 56, 10.5, None),        # 5.25 times the forward
    (6, 56, 31.5, None),        # the single-worker slot: still this worker
    (6, 16, None, "single-worker"),  # beyond this declaration
    (10, 56, None, "axon-cpu"),  # 52.5 GiB: over the 50 GiB line
    (20, 56, None, "axon-reviewed"),
])
def test_the_trace_cap_is_sized_from_the_forwards_peak(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch, peak_gib: float, share: int,
        rss_gib: float | None, placement: str | None) -> None:
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", str(share))
    record = read_json(ModelRecord, data.model("m8839"))
    plan = flow.trace_plan(record, _forward(int(peak_gib * 1024)))
    assert plan.placement == placement
    assert (plan.caps.rss_mb / 1024 if plan.caps else None) == rss_gib


def test_the_projection_uses_the_forwards_process_tree_peak(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "56")
    record = read_json(ModelRecord, data.model("m8839"))
    forward = _forward(1024)  # the worker's own forward peak: 1 GiB
    tree = replace(forward.rows[0], peak_rss_mb=4 * 1024)  # the runner's tree peak: 4 GiB
    plan = flow.trace_plan(record, replace(forward, rows=(tree,)))
    assert plan.caps is not None and plan.caps.rss_mb == int(4 * flow.TRACE_PEAK_RATIO * 1024)


def test_a_requests_own_cap_and_timeout_replace_the_projection(data: DataLayout) -> None:
    record = read_json(ModelRecord, data.model("m8839"))
    plan = flow.trace_plan(record, _forward(1024), timeout_s=900, memory_gib=12)
    assert plan.caps is not None and (plan.caps.rss_mb, plan.caps.timeout_s) == (12 * 1024, 900)
    assert flow.trace_plan(record, _forward(None)).caps is None  # unknown peak: runner defaults


def _run(data: DataLayout, spool: Path, sid: str = "m9001") -> flow.Outcome:
    record = read_json(ModelRecord, data.model(sid))
    return flow.run_model(record, caps=None, store=StoreLayout.from_env(), data=data,
                          attempts=spool, spool=spool)


def test_a_projection_beyond_the_worker_is_typed_then_captured_structure_only(
        data: DataLayout, runner_spool: tuple[Path, list[RunRequest], dict[str, Answer]]) -> None:
    path, seen, answer = runner_spool
    answer["fn"] = lambda r: _answer(r, peak_mb=6 * 1024)  # 31.5 GiB projected on a 16 GiB share
    outcome = _run(data, path)
    assert [q.mode for q in seen] == ["forward", "trace-structure-only"]
    typed = [r for r in outcome.rows if isinstance(r, LedgerRow) and r.capture == "full"]
    assert [(r.outcome, r.cls) for r in typed] == [("fail", "oversized:projected")]
    assert "placement single-worker" in (typed[0].msg or "")
    assert outcome.placement == "single-worker"


@pytest.mark.parametrize(("cls", "modes"), [
    ("memory_cap", ["forward", "trace", "trace", "trace-structure-only"]),
    ("timeout", ["forward", "trace", "trace-structure-only"]),
    ("capture_error", ["forward", "trace"]),
])
def test_a_failed_trace_gets_the_retry_and_structure_only_section_5_gives_it(
        data: DataLayout, runner_spool: tuple[Path, list[RunRequest], dict[str, Answer]],
        cls: str, modes: list[str]) -> None:
    path, seen, answer = runner_spool
    answer["fn"] = lambda r: _answer(r, trace_cls=cls)
    _run(data, path)
    assert [q.mode for q in seen] == modes
    traces = [q for q in seen if q.mode == "trace"]
    assert traces[0].caps.rss_mb == 8 * 1024  # the floor: 5.25 GiB projected
    if cls == "memory_cap":
        assert traces[1].caps.rss_mb == 16 * 1024


@pytest.mark.parametrize(("cls", "modes"), [
    ("memory_cap", ["forward", "trace-structure-only"]),
    ("construct_error", ["forward"]),
])
def test_a_failed_forward_goes_to_structure_only_or_stops(
        data: DataLayout, runner_spool: tuple[Path, list[RunRequest], dict[str, Answer]],
        cls: str, modes: list[str]) -> None:
    path, seen, answer = runner_spool

    def failing(request: RunRequest) -> RunResult:
        result = _answer(request)
        if request.mode != "forward":
            return result
        row = replace(result.rows[0], outcome="fail", cls=cls)
        return replace(result, outcome="fail", cls=cls, rows=(row,))

    answer["fn"] = failing
    outcome = _run(data, path)
    assert [q.mode for q in seen] == modes
    # A forward capped at 1 GiB projects a 5.25 GiB trace: past nothing but this worker's slot.
    assert outcome.placement == ("single-worker" if cls == "memory_cap" else None)


def _measured(data: DataLayout, sid: str, *, forward_mb: int, trace_mb: int,
              trace_cls: str | None = None, at: str = AT) -> None:
    from menagerie.pipeline import ledger

    record = read_json(ModelRecord, data.model(sid))
    assert record.recipe is not None
    base = {"id": sid, "at": at, "host": "worker", "env": "torch-main@00000000",
            "recipe_rev": "0" * 12, "mode": "eval"}  # an older revision still counts
    ledger.append([
        LedgerRow(stage="forward", outcome="pass", forward_peak_rss_mb=forward_mb, **base),
        LedgerRow(stage="trace", capture="full", outcome="fail" if trace_cls else "pass",
                  cls=trace_cls, peak_rss_mb=trace_mb, **base),
    ], data=data)


def test_sizing_prefers_a_measured_peak_then_the_familys_ratio_then_a_step_past_memory_cap(
        data: DataLayout) -> None:
    _measured(data, "m8839", forward_mb=1024, trace_mb=10 * 1024)  # f-resnet: ratio 10
    peaks, families = stage3.Peaks.read(data), stage3._families(data)
    own = stage3.sizing(read_json(ModelRecord, data.model("m8839")), peaks, families)
    assert own.measured_gib == 12.5 and own.project(1.0) == 12.5  # 10 GiB x the headroom
    sibling = stage3.sizing(read_json(ModelRecord, data.model("m9001")), peaks, families)
    assert sibling.ratio == 10.0 and sibling.project(2.0) == 20.0  # not 5.25 x 2
    _measured(data, "m9001", forward_mb=2048, trace_mb=41 * 1024, trace_cls="memory_cap")
    capped = stage3.sizing(read_json(ModelRecord, data.model("m9001")), stage3.Peaks.read(data),
                           families)
    assert capped.ratio == 20.5 and capped.at_least_gib == 61.5  # 1.5 x the capped 41 GiB
    assert capped.project(2.0) == 61.5
    assert stage3.placement("m9001", data=data)["placement"] == "axon-reviewed"
    assert stage3.placement("m8839", data=data)["placement"] == "worker"


def test_a_newer_memory_cap_outranks_an_older_smaller_passing_peak(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import runner

    # m9885 on 2026-10-22: passed at 3.4 GiB earlier, then memory_cap at 16.4 GiB (B's night cap)
    _measured(data, "m8839", forward_mb=3641, trace_mb=3483)
    _measured(data, "m8839", forward_mb=3641, trace_mb=16440, trace_cls="memory_cap",
              at="2026-10-22T06:07:00Z")
    record, families = read_json(ModelRecord, data.model("m8839")), stage3._families(data)
    sized = stage3.sizing(record, stage3.Peaks.read(data), families)
    assert sized.measured_gib is None and sized.at_least_gib == 24.08  # 1.5 x 16.05 GiB
    assert stage3.placement("m8839", data=data)["placement"] == "single-worker"
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "16")  # B's night size: typed beyond, not retried
    assert flow.trace_plan(record, None, sizing=sized).placement == "single-worker"
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "36")  # B's day size runs it under a 24 GiB cap
    day = flow.trace_plan(record, None, sizing=sized)
    assert day.placement is None and day.caps is not None and day.caps.rss_mb >= 24 * 1024
    # a pass after the cap (the row ran at a size that fits) brings the measured peak back
    _measured(data, "m8839", forward_mb=3641, trace_mb=17000, at="2026-10-22T09:00:00Z")
    again = stage3.sizing(record, stage3.Peaks.read(data), families)
    assert again.measured_gib == round(17000 / 1024 * runner.PEAK_HEADROOM, 2)


def test_an_older_memory_cap_leaves_a_later_passing_peak_in_charge(data: DataLayout) -> None:
    _measured(data, "m8839", forward_mb=1024, trace_mb=9 * 1024, trace_cls="memory_cap",
              at="2026-10-01T00:00:00Z")
    _measured(data, "m8839", forward_mb=1024, trace_mb=4 * 1024)
    sized = stage3.sizing(read_json(ModelRecord, data.model("m8839")), stage3.Peaks.read(data),
                          stage3._families(data))
    assert sized.measured_gib == 5.0 and sized.at_least_gib is None


def test_a_measured_peak_that_fits_brings_a_cluster_placement_back_to_the_worker(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive

    monkeypatch.setattr(drive, "lanes", lambda data: [])
    record = read_json(ModelRecord, data.model("m8839"))
    assert record.recipe is not None
    write_json(replace(record, recipe=replace(record.recipe, placement="axon-cpu")),
               data.model("m8839"))
    _measured(data, "m8839", forward_mb=4096, trace_mb=26 * 1024)
    (name,) = stage3.queue(None, data=data, ids=["m8839"], spool=tmp_path / "spool")
    job = json.loads((tmp_path / "spool" / "stage3" / "new" / f"{name}.json").read_text())
    assert job["record"]["recipe"]["placement"] == "single-worker"
    assert job["sizing"]["measured_gib"] == 32.5


def test_the_trace_plan_follows_the_sizing_it_is_given(data: DataLayout) -> None:
    record = read_json(ModelRecord, data.model("m8839"))
    plan = flow.trace_plan(record, _forward(1024), sizing=flow.Sizing(ratio=12.0))
    assert plan.caps is not None and plan.caps.rss_mb == 12 * 1024
    beyond = flow.trace_plan(record, _forward(1024), sizing=flow.Sizing(at_least_gib=61.0))
    assert beyond.placement == "axon-reviewed" and beyond.caps is None


# --- queue (hub) ------------------------------------------------------------------------------


def test_queue_waits_for_the_lanes_terminal_event(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    _terminal(monkeypatch, "RUNNING")
    _lane_event("run-2b-1", "f-resnet", "done")
    with pytest.raises(RecordError, match="terminal event"):
        stage3.queue("run-2b-1", data=data, spool=data.root.parent / "spool")


def test_queue_sends_a_terminal_lanes_members_pinned_at_their_revision(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _terminal(monkeypatch)
    _lane_event("run-2b-1", "f-resnet", "done")
    spool = tmp_path / "spool"
    names = stage3.queue("run-2b-1", data=data, spool=spool)
    assert sorted(n.split("-")[0] for n in names) == ["m8839", "m9001"]
    job = json.loads((spool / "stage3" / "new" / f"{names[1]}.json").read_text())
    record = read_json(ModelRecord, data.model("m9001"))
    assert (job["id"], job["rev"], job["lane"]) == ("m9001", record.recipe.rev, "run-2b-1")
    assert job["adapter_src"] == ADAPTER and from_dict(ModelRecord, job["record"]) == record
    events = (tmp_path / "work" / "lanes" / "run-2b-1" / "events.jsonl").read_text()
    assert events.count('"verb": "stage3"') == 2
    # Sent and not collected: queueing again skips both; a lane's non-member is refused.
    assert stage3.queue("run-2b-1", data=data, spool=spool) == []
    with pytest.raises(RecordError, match="not members"):
        stage3.queue("run-2b-1", data=data, ids=["m1"], spool=spool)


def test_queue_takes_the_members_a_done_event_accepted_or_rewrote(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _terminal(monkeypatch)
    _lane_event("run-2b-3", "f-resnet", "done", accepted=["m9001"], rewrote=[])
    names = stage3.queue("run-2b-3", data=data, spool=tmp_path / "spool")
    assert [n.split("-")[0] for n in names] == ["m9001"]


def test_queue_skips_a_stale_revision_and_refuses_ids_a_live_lane_holds(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    from menagerie.pipeline import drive

    data.adapter("f-resnet").write_text(ADAPTER + "# edited by a later lane\n", encoding="utf-8")
    monkeypatch.setattr(drive, "lanes", lambda data: [])
    spool = tmp_path / "spool"
    names = stage3.queue(None, data=data, ids=["m9001", "m8839"], spool=spool)
    assert [n.split("-")[0] for n in names] == ["m8839"]
    assert "m9001: not queued: recipe rev" in capsys.readouterr().err
    monkeypatch.setattr(drive, "lanes", lambda data: [{"run": "run-2b-2", "keys": ["f-resnet"]}])
    with pytest.raises(RecordError, match="live lanes hold"):
        stage3.queue(None, data=data, ids=["m8839"], spool=spool)


def test_queue_skips_a_member_traced_at_its_revision_unless_again(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    from menagerie.pipeline import drive, ledger
    from menagerie.pipeline.publish import torchlens_pin

    monkeypatch.setattr(drive, "lanes", lambda data: [])
    record = read_json(ModelRecord, data.model("m8839"))
    assert record.recipe is not None
    row = LedgerRow(id="m8839", stage="trace", at=AT, host="worker", env="torch-main@00000000",
                    recipe_rev=record.recipe.rev, outcome="fail", cls="capture_error",
                    mode="eval", capture="full", tl=torchlens_pin())
    ledger.append([row], data=data)
    spool = tmp_path / "spool"
    assert stage3.queue(None, data=data, ids=["m8839"], spool=spool) == []
    assert "m8839: not queued: current" in capsys.readouterr().err
    assert len(stage3.queue(None, data=data, ids=["m8839"], spool=spool, again=True)) == 1


# --- serve (worker) and collect (hub) ---------------------------------------------------------


def _queue_both(data: DataLayout, spool: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from menagerie.pipeline import drive

    monkeypatch.setattr(drive, "lanes", lambda data: [])
    return stage3.queue(None, data=data, ids=["m9001", "m8839"], spool=spool)


def test_serve_runs_each_job_through_the_runner_and_collect_appends_once(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        runner_spool: tuple[Path, list[RunRequest], dict[str, Answer]],
        capsys: pytest.CaptureFixture[str]) -> None:
    path, seen, _ = runner_spool
    names = _queue_both(data, path, monkeypatch)
    stage3.serve(spool=path, workers=2, once=True)
    assert {q.id for q in seen} == {"m9001", "m8839"}
    assert all(q.adapter_src == ADAPTER for q in seen if q.id == "m9001")
    done = json.loads((path / "stage3" / "done" / f"{names[0]}.json").read_text())
    assert done["outcome"] == "done" and len(done["ledger"]) == 2  # forward and trace
    assert not list((path / "attempts").iterdir())  # attempts reaped into the job folder
    main = DataLayout(tmp_path / "main")
    write_json(read_json(ModelRecord, data.model("m9001")), main.model("m9001"))
    index = {"sha256": "a" * 64, "bytes": 1, "mime": "application/json", "fetched": AT,
             "license_class": "derived"}
    main.store_index("2026-10").parent.mkdir(parents=True, exist_ok=True)
    main.store_index("2026-10").write_text(json.dumps(index) + "\n")
    extra = {**done, "store_index": [json.dumps(index), json.dumps({**index, "sha256": "b" * 64})]}
    (path / "stage3" / "done" / f"{names[0]}.json").write_text(json.dumps(extra))
    capsys.readouterr()
    assert sorted(stage3.collect(data=main, spool=path)) == ["m8839", "m9001"]
    lines = capsys.readouterr().out.splitlines()
    by_id = {json.loads(line).get("id"): json.loads(line) for line in lines}
    assert by_id["m9001"]["stale"] is False and by_id["m8839"]["stale"] is True  # no record
    assert json.loads(lines[-1])["added"] == {"ledger/2026-10": 4, "store-index/2026-10": 1}
    assert not list((path / "stage3" / "done").glob("*.json"))
    assert len(list((path / "stage3" / "collected").glob("*.json"))) == 2
    assert stage3.collect(data=main, spool=path) == []  # nothing left, nothing twice


def test_a_drained_spool_claims_no_new_job_until_the_drain_ends(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import disk_guard, runner, runner_job

    path = tmp_path / "spool"
    _queue_both(data, path, monkeypatch)
    monkeypatch.setattr(runner, "serving", lambda spool: True)
    monkeypatch.setattr(disk_guard, "claim_ok", lambda root: True)
    monkeypatch.setattr(stage3, "_serve_one", lambda *args: None)
    dirs, seen, running = stage3._dirs(path), {}, {}
    store = SimpleNamespace(root=tmp_path / "store")
    (path / runner_job.DRAIN).touch()
    stage3._start(dirs, seen, running, workers=2, spool=path, store=store)
    assert not running and len(list(dirs["new"].glob("*.json"))) == 2
    (path / runner_job.DRAIN).unlink()
    stage3._start(dirs, seen, running, workers=2, spool=path, store=store)
    assert len(running) == 2 and not list(dirs["new"].glob("*.json"))


def test_serve_refuses_a_superseded_job_and_sets_aside_an_unreadable_one(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        runner_spool: tuple[Path, list[RunRequest], dict[str, Answer]]) -> None:
    path, seen, _ = runner_spool
    first = _queue_both(data, path, monkeypatch)
    old = path / "stage3" / "new" / f"{first[0]}.json"
    job = json.loads(old.read_text())
    newer = {**job, "rev": "0" * 12, "queued": "2026-12-01T00:00:00Z",
             "name": f"m9001-{'0' * 12}-20261201t000000z-0000"}
    newer["record"]["recipe"]["rev"] = "0" * 12
    (path / "stage3" / "new" / f"{newer['name']}.json").write_text(json.dumps(newer))
    (path / "stage3" / "new" / "junk.json").write_text("{}")
    stage3.serve(spool=path, workers=1, once=True)
    refused = json.loads((path / "stage3" / "done" / f"{first[0]}.json").read_text())
    assert refused["outcome"] == "refused" and newer["name"] in refused["why"]
    assert (path / "stage3" / "rejected" / "junk.json").is_file()
    assert sorted({q.id for q in seen}) == ["m8839", "m9001"]


def test_a_job_moved_to_the_head_still_supersedes_the_older_revision() -> None:
    """A head move rewrites ``queued`` to an early time; supersession follows the name's stamp."""

    old = {"name": "m9001-aaaaaaaaaaaa-20261005t154556z-d811", "id": "m9001",
           "rev": "aaaaaaaaaaaa", "queued": "2026-10-05T15:45:56Z"}
    new = {"name": "m9001-bbbbbbbbbbbb-20261005t190153z-ac21", "id": "m9001",
           "rev": "bbbbbbbbbbbb", "queued": "2026-10-05T14:00:00Z"}  # moved to the head
    jobs = [new, old]
    assert stage3._superseded(old, jobs) == new["name"]
    assert stage3._superseded(new, jobs) is None
    same_rev = {**new, "name": "m9001-bbbbbbbbbbbb-20261006t000000z-0001"}
    assert stage3._superseded(new, [new, same_rev]) is None  # a GPU follow-up never supersedes


@pytest.mark.parametrize(("at", "open_"), [
    ("2026-10-08T08:00:00-07:00", False),
    ("2026-10-08T21:59:59-07:00", False),
    ("2026-10-08T22:00:00-07:00", True),
    ("2026-10-09T07:59:59-07:00", True),
    ("2026-10-10T12:00:00-07:00", False),  # Weekend quiet hours apply too.
    ("2026-12-01T07:59:59-08:00", True),
    ("2026-12-01T08:00:00-08:00", False),  # Follows daylight saving.
])
def test_the_gpu_window_follows_jmt_zone_every_day(at: str, open_: bool, tmp_path, monkeypatch) -> None:
    from datetime import datetime

    from menagerie.pipeline import worker

    setting = tmp_path / "zone.json"
    setting.write_text(json.dumps({"zone": "America/Los_Angeles"}))
    monkeypatch.setenv("JMT_TZ_FILE", str(setting))
    assert stage3.gpu_window_open is worker.in_gpu_window  # one definition, the worker's
    assert stage3.gpu_window_open(datetime.fromisoformat(at)) is open_


def _gpu_member(data: DataLayout) -> None:
    record = read_json(ModelRecord, data.model("m9001"))
    assert record.recipe is not None
    write_json(replace(record, recipe=replace(record.recipe, placement="gpu-window")),
               data.model("m9001"))


def test_a_gpu_member_is_captured_now_and_its_gpu_attempt_held_for_the_window(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch,
        runner_spool: tuple[Path, list[RunRequest], dict[str, Answer]]) -> None:
    from menagerie.pipeline import drive

    path, seen, _ = runner_spool
    _gpu_member(data)
    monkeypatch.setattr(drive, "lanes", lambda data: [])
    monkeypatch.setattr(stage3, "gpu_window_open", lambda at=None: False)
    (first,) = stage3.queue(None, data=data, ids=["m9001"], spool=path)
    stage3.serve(spool=path, workers=1, once=True)  # returns: only the held job is left
    assert [q.mode for q in seen] == ["trace-structure-only"]
    done = json.loads((path / "stage3" / "done" / f"{first}.json").read_text())
    held = path / "stage3" / "new" / f"{done['gpu_held_as']}.json"
    assert json.loads(held.read_text())["part"] == "gpu"
    monkeypatch.setattr(stage3, "gpu_window_open", lambda at=None: True)
    stage3.serve(spool=path, workers=1, once=True)
    assert [q.mode for q in seen] == ["trace-structure-only", "gpu"]
    assert not held.exists()


def test_a_gpu_attempt_refused_outside_the_window_is_held_not_retried(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch,
        runner_spool: tuple[Path, list[RunRequest], dict[str, Answer]]) -> None:
    from menagerie.pipeline import drive

    path, seen, answer = runner_spool
    _gpu_member(data)

    def refused(request: RunRequest) -> RunResult:
        result = _answer(request)
        if request.mode != "gpu":
            return result
        row = replace(result.rows[0], outcome="infra", cls="infra:gpu_window")
        return replace(result, outcome="infra", cls="infra:gpu_window", rows=(row,))

    answer["fn"] = refused
    monkeypatch.setattr(drive, "lanes", lambda data: [])
    monkeypatch.setattr(stage3, "gpu_window_open", lambda at=None: True)  # the clocks disagree
    (first,) = stage3.queue(None, data=data, ids=["m9001"], spool=path)
    monkeypatch.setattr(stage3, "_held", lambda job: job.get("part") == "gpu")
    stage3.serve(spool=path, workers=1, once=True)
    done = json.loads((path / "stage3" / "done" / f"{first}.json").read_text())
    assert [q.mode for q in seen] == ["trace-structure-only", "gpu"]  # one attempt, no retry
    assert done["gpu_held_as"] and len(done["ledger"]) == 1  # the refusal is not a ledger row


def test_accept_records_the_forwards_measured_peak_as_memory_gib(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import implement

    record = read_json(ModelRecord, data.model("m9001"))
    assert record.recipe is not None
    made = implement.Draft(replace(record.recipe, memory_gib=30.0), ADAPTER, "port")
    request = RunRequest(request="m9001-forward-eval-x-0000", id="m9001", mode="forward",
                         recipe=made.recipe, caps=flow.Caps(rss_mb=1024, timeout_s=60, threads=1))
    implement._accept("m9001", made, _answer(request, peak_mb=3 * 1024), data)
    accepted = read_json(ModelRecord, data.model("m9001")).recipe
    assert accepted is not None and accepted.memory_gib == 3.0 and accepted.rev == record.recipe.rev


def test_the_cli_queues_committed_ids_through_the_verb(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import cli, drive

    monkeypatch.setattr(drive, "lanes", lambda data: [])
    monkeypatch.setenv("MENAGERIE_DATA", str(data.root))
    spool = tmp_path / "spool"
    cli.main(["stage3", "queue", "--committed", "m8839", "--spool", str(spool),
              "--timeout-s", "900"])
    jobs = [json.loads(p.read_text()) for p in (spool / "stage3" / "new").glob("*.json")]
    assert [(j["id"], j["timeout_s"]) for j in jobs] == [("m8839", 900)]


def test_per_row_floor_is_four_gib_with_the_switch_and_eight_without(data: DataLayout) -> None:
    record = read_json(ModelRecord, data.model("m9001"))
    families = stage3._families(data)
    empty = stage3.sizing(record, stage3.Peaks({}, {}, {}), families, per_row_floor=True)
    assert empty.floor_gib == 4 and empty.project(1.0) == flow.TRACE_PEAK_RATIO  # 5.25 x forward
    _measured(data, "m8839", forward_mb=1024, trace_mb=2 * 1024)
    peaks = stage3.Peaks.read(data)
    assert stage3.sizing(record, peaks, families).floor_gib == 8
    plan = stage3.sizing(record, peaks, families, per_row_floor=True)
    assert plan.floor_gib == 4 and plan.project(1.0) == 2
    _measured(data, "m9001", forward_mb=1024, trace_mb=3 * 1024, trace_cls="memory_cap")
    plan = stage3.sizing(record, stage3.Peaks.read(data), families, per_row_floor=True)
    assert plan.at_least_gib == 4.5 and plan.project(1.0) == 4.5


def test_g_queue_pins_local_placement_and_night_gate(data: DataLayout, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive, runner_job, tick

    monkeypatch.setattr(drive, "lanes", lambda data: [])
    cfg = {"big_runner": True}
    monkeypatch.setattr(tick, "config", lambda: cfg)
    _measured(data, "m8839", forward_mb=1024, trace_mb=32 * 1024)
    assert stage3.placement("m8839", data=data)["spool"].endswith("spool-G")
    spool = tmp_path / "spool-G"
    (name,) = stage3.queue(None, data=data, ids=["m8839"], spool=spool, again=True)
    path = spool / "stage3" / "new" / f"{name}.json"
    job = json.loads(path.read_text())
    assert job["record"]["recipe"]["placement"] == "single-worker"
    assert job["sizing"]["big_runner"] is True
    monkeypatch.setattr(runner_job, "night_open", lambda: False)
    assert stage3._held(stage3._read_job(path))
    monkeypatch.setattr(runner_job, "night_open", lambda: True)
    assert not stage3._held(stage3._read_job(path))
    cfg["big_runner"] = False
    with pytest.raises(ValueError, match="requires big_runner"):
        stage3.queue(None, data=data, ids=["m9001"], spool=spool)


def test_g_spool_refuses_rows_past_its_40_gib_declaration(data: DataLayout, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive, tick, tick_full

    monkeypatch.setattr(drive, "lanes", lambda data: [])
    monkeypatch.setattr(tick, "config", lambda: {"big_runner": True})
    _measured(data, "m8839", forward_mb=1024, trace_mb=36 * 1024)  # projected 45 GiB
    assert stage3.placement("m8839", data=data)["spool"] is None
    monkeypatch.setattr(tick_full, "_structure_ids", lambda data: {"m8839"})
    assert tick_full._big_candidates(data, ["m8839"]) == []
    with pytest.raises(ValueError, match=r"projection in \(18, 40\] GiB"):
        stage3.queue(None, data=data, ids=["m8839"], spool=tmp_path / "spool-G", again=True)


def test_tick_reserves_big_rows_without_touching_g_when_off(data: DataLayout, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive, runner_job, tick_full

    path = tmp_path / "worklist.txt"
    path.write_text("m8839\n")
    cfg = {"worklist": str(path), "queue_spool": "spool-B", "queue_depth": 40, "big_runner": False}
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [{"id": sid, "step": "trace"} for sid in ids])
    monkeypatch.setattr(tick_full, "_not_pending", lambda ids, data: list(ids))
    seen = []
    monkeypatch.setattr(tick_full, "_waiting", lambda spool: seen.append(spool) or 0)
    _measured(data, "m8839", forward_mb=1024, trace_mb=32 * 1024)
    off = tick_full.stage3_queue(data, cfg, dry_run=True)
    assert off["would_queue"] == ["m8839"] and seen == ["spool-B"]
    cfg["big_runner"] = True
    monkeypatch.setattr(runner_job, "night_open", lambda: False)
    day = tick_full.stage3_queue(data, cfg, dry_run=True)
    assert not day["would_queue"] and day["big"]["reserved"] == ["m8839"]
    assert day["big"]["waiting"] == 0  # by day too, so last night's unclaimed jobs keep G on
    assert seen == ["spool-B", "spool-B", "scratch/menagerie/spool-G"]
    monkeypatch.setattr(runner_job, "night_open", lambda: True)
    night = tick_full.stage3_queue(data, cfg, dry_run=True)
    assert night["big"]["would_queue"] == ["m8839"] and not night["would_queue"]


def test_default_queue_payload_is_compatible_with_live_old_runners(data: DataLayout, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive, tick

    monkeypatch.setattr(drive, "lanes", lambda data: [])
    monkeypatch.setattr(tick, "config", lambda: {"big_runner": False, "per_row_floor": False})
    spool = tmp_path / "spool-B"
    (name,) = stage3.queue(None, data=data, ids=["m8839"], spool=spool)
    job = json.loads((spool / "stage3" / "new" / f"{name}.json").read_text())
    assert set(job["sizing"]) == {"ratio", "at_least_gib", "measured_gib", "basis"}


def test_automatic_g_attempt_is_not_repeated_at_same_recipe(data: DataLayout, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import tick_full

    _measured(data, "m8839", forward_mb=1024, trace_mb=32 * 1024)
    record = read_json(ModelRecord, data.model("m8839"))
    monkeypatch.setattr(stage3, "_local", lambda part: tmp_path)
    (tmp_path / "prior.json").write_text(json.dumps({"id": record.id, "rev": record.recipe.rev,
                                                   "sizing": {"big_runner": True}}))
    assert tick_full._big_candidates(data, [record.id]) == []


def test_g_spool_failure_keeps_normal_queue_result(data: DataLayout, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive, runner_job, tick_full

    path = tmp_path / "worklist.txt"
    path.write_text("m8839\n")
    cfg = {"worklist": str(path), "queue_spool": "spool-B", "queue_depth": 40, "big_runner": True}
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [{"id": sid, "step": "trace"} for sid in ids])
    monkeypatch.setattr(tick_full, "_not_pending", lambda ids, data: list(ids))
    monkeypatch.setattr(runner_job, "night_open", lambda: True)
    def waiting(spool: str) -> int:
        if spool.endswith("spool-G"):
            raise OSError("unavailable")
        return 0
    monkeypatch.setattr(tick_full, "_waiting", waiting)
    result = tick_full.stage3_queue(data, cfg, dry_run=True)
    assert result["would_queue"] == ["m8839"] and result["big"]["held"] == "unavailable"


def test_corrupt_g_metadata_keeps_regular_work_available(data: DataLayout, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive, tick_full

    path = tmp_path / "worklist.txt"
    path.write_text("m8839\n")
    cfg = {"worklist": str(path), "queue_spool": "spool-B", "queue_depth": 40, "big_runner": True}
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [{"id": sid, "step": "trace"} for sid in ids])
    monkeypatch.setattr(tick_full, "_not_pending", lambda ids, data: list(ids))
    monkeypatch.setattr(tick_full, "_waiting", lambda spool: 0)
    def corrupt() -> set[tuple[str, str]]:
        raise ValueError("corrupt request")
    monkeypatch.setattr(tick_full, "_big_sent", corrupt)
    result = tick_full.stage3_queue(data, cfg, dry_run=True)
    assert result["would_queue"] == ["m8839"] and result["big"]["held"] == "corrupt request"


def test_g_candidates_respect_stopped_cohorts(data: DataLayout, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import tick_full

    path = tmp_path / "worklist.txt"
    path.write_text("m8839 r4-github\n")
    cfg = {"worklist": str(path), "stopped_cohorts": ["r4-github"]}
    _measured(data, "m8839", forward_mb=1024, trace_mb=32 * 1024)
    monkeypatch.setattr(tick_full, "_structure_ids", lambda data: {"m8839"})
    assert tick_full._big_candidates(data, ["m8839"], cfg=cfg) == []


@pytest.mark.parametrize(("trace_only", "again"), [(False, False), (True, False), (True, True)])
def test_queue_reuses_forward_at_the_current_recipe(
        trace_only: bool, again: bool,
        data: DataLayout, tmp_path: Path,
        runner_spool: tuple[Path, list[RunRequest], dict[str, Answer]]) -> None:
    from menagerie.pipeline import ledger

    path, seen, answer = runner_spool
    record = read_json(ModelRecord, data.model("m8839"))
    row = LedgerRow(id=record.id, stage="forward", at=AT, host="worker",
                    env="torch-main@00000000", recipe_rev=record.recipe.rev,
                    outcome="pass", peak_rss_mb=128, mode="eval")
    ledger.append([row], data=data)
    jobs = stage3.queue(None, data=data, ids=["m8839"], spool=path,
                        trace_only=trace_only, again=again)
    assert len(jobs) == 1
    stage3.serve(spool=path, once=True)
    assert seen and not any(r.mode == "forward" for r in seen)
    assert any(r.mode == "trace" for r in seen)


def test_trace_only_queue_does_not_create_a_missing_forward(
        data: DataLayout, tmp_path: Path,
        runner_spool: tuple[Path, list[RunRequest], dict[str, Answer]]) -> None:
    path, seen, answer = runner_spool
    assert stage3.queue(None, data=data, ids=["m8839"], spool=path, trace_only=True)
    stage3.serve(spool=path, once=True)
    assert seen and not any(r.mode == "forward" for r in seen)
    assert any(r.mode == "trace" for r in seen)



def test_trace_only_jobs_are_fenced_from_pre_repin_consumers(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive, tick

    monkeypatch.setattr(drive, "lanes", lambda data: [])
    monkeypatch.setattr(tick, "config", lambda: {"big_runner": False, "per_row_floor": False})
    spool = tmp_path / "spool-B"
    (name,) = stage3.queue(None, data=data, ids=["m8839"], spool=spool, trace_only=True)
    path = spool / "stage3" / "new" / f"{name}.json"
    job = json.loads(path.read_text())
    assert job["schema"] == stage3.TRACE_ONLY_SCHEMA and job["trace_only"] is True
    assert stage3._read_job(path)["id"] == "m8839"
    # A consumer from before the repin accepts only stage3-job/1, so it rejects this job.
    assert job["schema"] != "stage3-job/1"
    for schema, trace_only in ((stage3.JOB_SCHEMA, True), (stage3.TRACE_ONLY_SCHEMA, False)):
        path.write_text(json.dumps({**job, "schema": schema, "trace_only": trace_only}))
        with pytest.raises(RecordError, match="not a stage-3 job"):
            stage3._read_job(path)

def test_training_forward_cannot_supply_eval_evidence(data: DataLayout) -> None:
    record = read_json(ModelRecord, data.model("m8839"))
    request = RunRequest(request="m8839-forward-eval-reused-0000", id=record.id,
                         mode="forward", recipe=record.recipe,
                         caps=flow.Caps(rss_mb=1024, timeout_s=60, threads=1))
    result = _answer(request)
    result = replace(result, rows=(replace(result.rows[0], mode="train"),))
    with pytest.raises(RecordError, match="eval-mode"):
        stage3._checked_forward({"settled_forward": to_dict(result)}, record)
