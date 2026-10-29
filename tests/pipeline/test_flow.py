"""Stage 3 through a runner spool: adapter recipes carry their committed adapter's text.

Torch-free: a stand-in thread serves the spool and answers each request, so the requests flow
builds are checked as the runner would receive them.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, flow, read_json, write_json
from menagerie.pipeline.records import LedgerRow, ModelRecord, Modes, RunRequest, RunResult

FIXTURES = Path(__file__).with_name("fixtures")
ADAPTER = '''"""A toy adapter."""
import torch


def build(cfg):
    return torch.nn.Linear(4, 8)
'''


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """resnet18's library record, and the same model as an adapter recipe (m9001)."""

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_CPUS", "2")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "8")
    layout = DataLayout(tmp_path / "data")
    resnet = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    write_json(resnet, layout.model("m8839"))
    layout.adapter("f-toy").parent.mkdir(parents=True, exist_ok=True)
    layout.adapter("f-toy").write_text(ADAPTER, encoding="utf-8")
    assert resnet.recipe is not None
    recipe = replace(resnet.recipe, kind="adapter", adapter="adapters/f-toy.py", call=None)
    write_json(replace(resnet, id="m9001", name="toy", recipe=recipe), layout.model("m9001"))
    return layout


def _answer(request: RunRequest) -> RunResult:
    """Forward passes; a trace fails, so no render follows."""

    outcome = "pass" if request.mode == "forward" else "fail"
    row = LedgerRow(id=request.id, stage=request.mode.split("-")[0], at="2026-10-04T00:00:00Z",
                    host="worker", env="torch-main@00000000", recipe_rev=request.recipe.rev,
                    outcome=outcome, cls=None if outcome == "pass" else "construct_error",
                    mode=request.model_mode)
    return RunResult(request=request.request, id=request.id, mode=request.mode, outcome=outcome,
                     started="2026-10-04T00:00:00Z", ended="2026-10-04T00:00:02Z", host="worker",
                     cls=row.cls, output_shapes=((1, 8),) if outcome == "pass" else (),
                     modes=Modes(eval="pass", train="pass", divergence="none")
                     if outcome == "pass" else None, rows=(row,))


@pytest.fixture
def spool(tmp_path: Path) -> Iterator[tuple[Path, list[RunRequest]]]:
    """A stand-in runner serving ``tmp_path/spool``; yields the spool and the requests it saw."""

    path = tmp_path / "spool"
    seen: list[RunRequest] = []
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
                write_json(sys.modules[__name__]._answer(request), path / "done" / item.name)
            time.sleep(0.05)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield path, seen
    stop.set()
    thread.join(timeout=5)


def test_adapter_src_reads_the_committed_adapter_only_for_adapter_recipes(
        data: DataLayout) -> None:
    assert flow.adapter_src(read_json(ModelRecord, data.model("m9001")), data) == ADAPTER
    assert flow.adapter_src(read_json(ModelRecord, data.model("m8839")), data) is None


def test_stage3_through_a_spool_carries_the_adapter_text(
        data: DataLayout, tmp_path: Path, spool: tuple[Path, list[RunRequest]]) -> None:
    path, seen = spool
    out = tmp_path / "e2e.jsonl"
    table = flow.run(["m9001", "m8839"], caps=None, data=data, store=StoreLayout(tmp_path / "s"),
                     attempts=tmp_path / "unused", out=out, spool=path)
    by_id = {r.id: [q for q in seen if q.id == r.id] for r in seen}
    assert [q.mode for q in by_id["m9001"]] == ["forward", "trace"]
    assert all(q.adapter_src == ADAPTER for q in by_id["m9001"])
    assert all(q.adapter_src is None for q in by_id["m8839"])
    assert [row["forward"] for row in table] == ["pass", "pass"]
    assert not (tmp_path / "unused").exists()  # attempts live in the spool
    ledger = sorted(data.root.glob("ledger/*.jsonl"))
    assert ledger and sum(1 for _ in ledger[0].open()) == 4


def test_a_memory_capped_trace_is_retried_with_the_whole_declaration(
        data: DataLayout, tmp_path: Path, spool: tuple[Path, list[RunRequest]],
        monkeypatch: pytest.MonkeyPatch) -> None:
    path, seen = spool
    answer = _answer

    def capped(request: RunRequest) -> RunResult:
        result = answer(request)
        if request.mode == "trace" and request.caps.rss_mb < 8 * 1024:
            return replace(result, cls="memory_cap")
        return result

    monkeypatch.setattr(sys.modules[__name__], "_answer", capped)
    flow.run(["m9001"], caps=None, data=data, store=StoreLayout(tmp_path / "s"),
             attempts=tmp_path / "unused", out=tmp_path / "e2e.jsonl", spool=path)
    traces = [q for q in seen if q.mode.startswith("trace")]
    assert [q.mode for q in traces] == ["trace", "trace"]  # no structure-only after the retry passed
    assert traces[0].caps.rss_mb < 8 * 1024 == traces[1].caps.rss_mb


def test_opt_in_big_tier_and_low_floor(data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "40")
    record = read_json(ModelRecord, data.model("m8839"))
    assert flow.placement_for(40, big_runner=True) == "single-worker"
    assert flow.placement_for(45, big_runner=True) == "axon-cpu"  # past G, the cluster (2026-10-28)
    assert flow.placement_for(45) == "single-worker"  # without the big runner, the whole worker
    assert flow.placement_for(55, big_runner=True) == "axon-cpu"  # past G's 40 GiB
    assert flow.placement_for(90, big_runner=True) == "axon-reviewed"
    plan = flow.trace_plan(record, None, memory_gib=40, sizing=flow.Sizing(big_runner=True))
    assert plan.placement is None and plan.caps.rss_mb == 40 * 1024
    plan = flow.trace_plan(record, None, memory_gib=2, sizing=flow.Sizing(floor_gib=3))
    assert plan.caps.rss_mb == 3 * 1024
    default = flow.trace_plan(record, None, memory_gib=2)
    assert default.caps.rss_mb == 8 * 1024
    assert flow.trace_plan(record, None, memory_gib=60).placement is not None
    assert flow.trace_plan(record, None, memory_gib=44, sizing=flow.Sizing(big_runner=True)).placement \
        is not None


def test_a_low_cap_that_breaks_retries_once_straight_into_the_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "36")
    monkeypatch.setenv("RUNNER_WORKERS", "8")
    caps = []

    def go(mode, **kw):
        cap = kw["caps"].rss_mb
        caps.append(cap)
        return SimpleNamespace(cls="memory_cap" if cap < 8 * 1024 else None)

    result = flow._full_trace(go, first=flow.Caps(rss_mb=4 * 1024, timeout_s=90, threads=1),
                              retry=True)
    assert caps == [4 * 1024, int(flow.WORKER_GIB * 1024)]  # one retry, in the 18 GiB slot
    assert len(result) == 2 and result[-1].cls is None


def test_train_trace_memory_cap_is_retried_once_at_the_whole_declaration(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "40")
    calls = []

    def go(mode, **kw):
        calls.append((kw.get("model_mode"), kw["caps"].rss_mb if kw.get("caps") else None))
        return SimpleNamespace(cls="memory_cap" if kw["caps"].rss_mb < 40 * 1024 else None)

    sized = flow.Caps(rss_mb=35 * 1024, timeout_s=90, threads=1)
    result = flow._train_trace(go, sized, retry=True)
    assert calls == [("train", 35 * 1024), ("train", 40 * 1024)] and result[-1].cls is None
    calls.clear()
    assert len(flow._train_trace(go, sized, retry=False)) == 1  # explicit caps: never retried
    calls.clear()
    whole = flow.Caps(rss_mb=40 * 1024, timeout_s=90, threads=1)
    assert len(flow._train_trace(go, whole, retry=True)) == 1  # already at the whole declaration


def test_a_memory_cap_retry_takes_a_worker_slot_on_a_runner_with_several_workers(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "36")
    monkeypatch.setenv("RUNNER_WORKERS", "4")
    calls = []

    def go(mode, **kw):
        calls.append((kw.get("model_mode"), kw["caps"].rss_mb))
        return SimpleNamespace(cls="memory_cap" if kw["caps"].rss_mb < 18 * 1024 else None)

    eight = flow.Caps(rss_mb=8 * 1024, timeout_s=90, threads=1)
    assert flow._full_trace(go, first=eight, retry=True)[-1].cls is None
    assert calls == [(None, 8 * 1024), (None, 18 * 1024)]  # a slot, not the whole 36 GiB
    calls.clear()
    assert flow._train_trace(go, flow.Caps(rss_mb=16 * 1024, timeout_s=90, threads=1), retry=True)
    assert calls == [("train", 16 * 1024), ("train", 18 * 1024)]
    calls.clear()
    eighteen = flow.Caps(rss_mb=18 * 1024, timeout_s=90, threads=1)
    monkeypatch.setattr(flow, "_memory_cap", lambda result: True)
    assert len(flow._full_trace(go, first=eighteen, retry=True)) == 1  # broke at a slot: G's row
    monkeypatch.setenv("RUNNER_WORKERS", "1")
    assert flow._retry_caps(eight).rss_mb == 36 * 1024  # one worker: the whole declaration


def test_g_forward_uses_its_single_worker_declaration(data: DataLayout, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "40")
    monkeypatch.setenv("MENAGERIE_CPUS", "6")
    seen = []

    def answer(request, outcome, **kw):
        seen.append(request)
        result = _answer(request)
        outcome.results.append(result)
        return result

    monkeypatch.setattr(flow, "_run", answer)
    flow.run_model(read_json(ModelRecord, data.model("m8839")), caps=None,
                   data=data, store=StoreLayout(tmp_path / "store"), attempts=tmp_path / "attempts",
                   memory_gib=40, sizing=flow.Sizing(big_runner=True))
    assert seen[0].mode == "forward" and seen[0].caps.rss_mb == 40 * 1024
    assert seen[0].caps.threads == 6
    assert seen[1].mode == "trace" and seen[1].caps.rss_mb == 40 * 1024


# --- render staging: the portable save, else the trace save ---------------------------------

def _saved_trace(attempt: Path, *, portable: bool, on_disk: bool = True) -> RunResult:
    """A passing eval trace attempt whose files list ``trace.tlspec`` and, when ``portable``,
    ``render.tlspec`` (written on disk unless ``on_disk`` is false)."""

    from menagerie.pipeline import tree_sha256
    from menagerie.pipeline.records import OutputFile

    files = []
    for kind, name in (("tlspec", "trace.tlspec"), ("tlspec-portable", "render.tlspec")):
        if kind == "tlspec-portable" and not portable:
            continue
        save = attempt / name
        (save / "blobs").mkdir(parents=True)
        (save / "manifest.json").write_text(f'{{"save_level": "portable", "name": "{name}"}}')
        files.append(OutputFile(kind=kind, path=name, sha256=tree_sha256(save), bytes=1))
        if not on_disk and kind == "tlspec-portable":
            import shutil

            shutil.rmtree(save)
    result = RunResult(request=attempt.name, id="m8839", mode="trace", outcome="pass",
                       started="2026-10-28T00:00:00Z", ended="2026-10-28T00:00:01Z",
                       host="worker", files=tuple(files))
    write_json(result, attempt / "result.json")
    return result


def _staged(attempts: Path, request: str) -> dict[str, str]:
    import json

    return {p.name: json.loads((p / "manifest.json").read_text())["name"]
            for p in (attempts / request).glob("*.tlspec")}


@pytest.mark.parametrize(("portable", "on_disk", "staged"), [
    (True, True, {"render.tlspec": "render.tlspec"}),  # a runnable save and its portable save
    (False, True, {"trace.tlspec": "trace.tlspec"}),  # a refused save: trace.tlspec is portable
    (True, False, {"trace.tlspec": "trace.tlspec"}),  # the portable save is gone from disk
])
def test_render_request_stages_the_portable_save_else_the_trace_save(
        data: DataLayout, tmp_path: Path, portable: bool, on_disk: bool,
        staged: dict[str, str]) -> None:
    attempt = tmp_path / "attempts" / "m8839-trace-eval-20261028T000000-abcd"
    trace = _saved_trace(attempt, portable=portable, on_disk=on_disk)
    record = read_json(ModelRecord, data.model("m8839"))
    request = flow.render_request(record, trace, attempt, caps=None, attempts=tmp_path / "r")
    key = next(f.sha256 for f in trace.files if f.kind == "tlspec")
    assert request.trace == key and request.mode == "render" and request.model_mode == "eval"
    assert _staged(tmp_path / "r", request.request) == staged


def test_render_request_without_any_save_on_disk_leaves_render_mode_the_store_blob(
        data: DataLayout, tmp_path: Path) -> None:
    import shutil

    attempt = tmp_path / "attempts" / "m8839-trace-eval-20261028T000000-abcd"
    trace = _saved_trace(attempt, portable=False)
    shutil.rmtree(attempt / "trace.tlspec")
    record = read_json(ModelRecord, data.model("m8839"))
    request = flow.render_request(record, trace, attempt, caps=None, attempts=tmp_path / "r")
    assert request.trace == trace.files[0].sha256 and _staged(tmp_path / "r", request.request) == {}


def test_rerender_of_a_trace_without_a_portable_save_stages_its_trace_save(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The axon render job (``axon_feed render`` through ``flow.rerender``) on a collected
    attempt whose runnable save was refused: keyed on the ledger ``trace`` sha, drawn from
    ``trace.tlspec``."""

    from menagerie.pipeline.records import Caps

    root = tmp_path / "back" / "attempts"
    trace = _saved_trace(root / "m8839-trace-eval-20261028T000000-abcd", portable=False)
    seen: list[RunRequest] = []

    def run(request: RunRequest, outcome: flow.Outcome, **kw: object) -> None:
        seen.append(request)
        assert _staged(tmp_path / "work", request.request) == {"trace.tlspec": "trace.tlspec"}

    monkeypatch.setattr(flow, "_run", run)
    rows = flow.rerender(["m8839"], [root], caps=Caps(rss_mb=1024, timeout_s=60), data=data,
                         store=StoreLayout(tmp_path / "store"), attempts=tmp_path / "work")
    assert rows == [] and [r.trace for r in seen] == [trace.files[0].sha256]


# --- per-row pledges for forwards and renders (runner B with per_row_floor) -----------------

def _row_sizing() -> flow.Sizing:
    return flow.Sizing(floor_gib=flow.ROW_FLOOR_GIB)


def test_forward_and_render_pledges_follow_the_row_with_a_floor_and_the_flat_pledge_as_ceiling(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import runner

    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "36")
    monkeypatch.setenv("MENAGERIE_CPUS", "6")
    record = read_json(ModelRecord, data.model("m8839"))
    assert flow._row_caps(record, "render", 6.0, flow.Sizing()) is None  # flat 8 GiB floor: B off
    assert flow._row_caps(record, "render", None, _row_sizing()) is None  # nothing measured
    small = flow._row_caps(record, "render", 1.0, _row_sizing())
    assert small is not None and small.rss_mb == 4 * 1024  # the floor
    mid = flow._row_caps(record, "forward", 6.0, _row_sizing())
    assert mid is not None and mid.rss_mb == int(6.0 * runner.PEAK_HEADROOM * 1024)
    big = flow._row_caps(record, "render", 30.0, _row_sizing())
    assert big is not None and big.rss_mb == 12 * 1024  # never above the flat 36 / 3


def test_a_render_pledge_that_breaks_retries_once_in_the_slot(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "36")
    monkeypatch.setenv("RUNNER_WORKERS", "8")
    record = read_json(ModelRecord, data.model("m8839"))
    row = LedgerRow(id="m8839", stage="trace", at="2026-10-28T00:00:00Z", host="worker",
                    env="torch-main@00000000", recipe_rev=record.recipe.rev, outcome="pass",
                    peak_rss_mb=2 * 1024)
    trace = SimpleNamespace(request="t", rows=(row,))
    caps = []

    def draw(_trace, cap):
        caps.append(cap.rss_mb if cap else None)
        return SimpleNamespace(cls="memory_cap" if cap.rss_mb < 8 * 1024 else None)

    flow._render(record, trace, draw, caps=None, sizing=_row_sizing())
    assert caps == [4 * 1024, int(flow.WORKER_GIB * 1024)]
    caps.clear()
    flow._render(record, trace, lambda t, c: caps.append(c) or None, caps=None, sizing=flow.Sizing())
    assert caps == [None]  # off B: the flat pledge, no retry


def test_a_forward_pledge_that_breaks_retries_once_in_the_slot(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "36")
    monkeypatch.setenv("RUNNER_WORKERS", "8")
    record = read_json(ModelRecord, data.model("m8839"))
    caps = []

    def go(mode, **kw):
        cap = kw.get("caps")
        caps.append(cap.rss_mb if cap else None)
        return SimpleNamespace(cls="memory_cap" if cap and cap.rss_mb < 8 * 1024 else None)

    measured = flow.Sizing(measured_gib=2.5, floor_gib=flow.ROW_FLOOR_GIB)
    assert flow._forward(record, go, measured).cls is None
    assert caps == [4 * 1024, int(flow.WORKER_GIB * 1024)]
    caps.clear()
    flow._forward(record, go, flow.Sizing(floor_gib=flow.ROW_FLOOR_GIB))  # nothing measured
    assert caps == [None]
