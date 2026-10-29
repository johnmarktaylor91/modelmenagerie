"""The in-attempt disk watchdog: the output cap, the free-space floor, render staging, and flow
stopping after an over-cap trace. Torch-free; the supervise test runs a plain Python child."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    attempt_watch,
    disk_guard,
    flow,
    read_json,
    runner,
    write_json,
)
from menagerie.pipeline.records import Caps, LedgerRow, ModelRecord, Modes, RunRequest, RunResult

FIXTURES = Path(__file__).with_name("fixtures")
GIB = disk_guard.GIB


def _statvfs(free_gib: float, total_gib: float = 458.0):
    def read(_path: str) -> SimpleNamespace:
        return SimpleNamespace(f_bavail=int(free_gib * GIB) // 4096,
                               f_blocks=int(total_gib * GIB) // 4096, f_frsize=4096)
    return read


def _request(mode: str = "trace") -> RunRequest:
    record = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    return runner.request_for(record, mode, caps=Caps(rss_mb=4096, timeout_s=300, threads=1))


def test_a_child_writing_past_the_cap_is_killed(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(attempt_watch.MAX_ENV, str(2 / 1024))  # 2 MiB
    monkeypatch.setattr(attempt_watch, "POLL_S", 0.1)
    out = tmp_path / "trace.tlspec"
    out.mkdir()
    script = (f"import os, time\nf = open({str(out / 'blob')!r}, 'wb')\nwhile True:\n"
              "    f.write(b'x' * 2**20); f.flush(); os.fsync(f.fileno()); time.sleep(0.02)\n")
    argv = [sys.executable, "-c", script]
    how, _code, _peak = runner._supervise(argv, tmp_path / "attempt.log",
                                          Caps(rss_mb=4096, timeout_s=60, threads=1),
                                          dict(os.environ), tmp_path)
    assert how == attempt_watch.OUTPUT_CAP


def test_an_over_cap_attempt_is_typed_oversized_and_keeps_only_small_files(
        tmp_path: Path) -> None:
    for name in ("trace.tlspec", "render.tlspec", "trace.tlspec.tmp.abc"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "blob").write_bytes(b"x" * 1024)
    (tmp_path / "attempt.log").write_text("log\n")
    write_json(_request(), tmp_path / "request.json")
    result = runner._watchdog_result(_request(), "2026-10-18T10:00:00Z",
                                     attempt_watch.OUTPUT_CAP, tmp_path, 100)
    assert result.outcome == "fail" and result.cls == "oversized:output"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["attempt.log", "request.json"]


def test_disk_low_is_infra(tmp_path: Path) -> None:
    result = runner._watchdog_result(_request(), "2026-10-18T10:00:00Z",
                                     attempt_watch.DISK_LOW, tmp_path, 100)
    assert result.outcome == "infra" and result.cls == "infra:disk_low"


def test_free_space_under_the_floor_plus_margin_breaches_only_in_a_runner_job(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(disk_guard.PAUSE_ENV, raising=False)
    assert attempt_watch.breach(tmp_path, statvfs=_statvfs(30)) is None
    monkeypatch.setenv(disk_guard.PAUSE_ENV, "56")
    assert attempt_watch.breach(tmp_path, statvfs=_statvfs(48)) == attempt_watch.DISK_LOW
    assert attempt_watch.breach(tmp_path, statvfs=_statvfs(60)) is None


def test_the_floor_is_the_larger_of_the_config_floor_and_a_tenth_of_the_disk() -> None:
    assert disk_guard.free_and_floor("/", statvfs=_statvfs(80, 200))[1] == pytest.approx(45.7)
    assert disk_guard.free_and_floor("/", statvfs=_statvfs(80, 1000))[1] == pytest.approx(100)


def test_the_cap_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(attempt_watch.MAX_ENV, raising=False)
    assert attempt_watch.max_bytes() == 8 * GIB
    monkeypatch.setenv(attempt_watch.MAX_ENV, "0")
    assert attempt_watch.max_bytes() is None


class _Stop(Exception):
    """Ends ``runner.serve``'s loop after one pass."""


def _stop(_seconds: float) -> None:
    raise _Stop


def test_runner_serve_claims_nothing_below_the_pause_line(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = []
    monkeypatch.setattr(disk_guard, "claim_ok", lambda path: False)
    monkeypatch.setattr(runner, "_start_waiting", lambda *a, **k: started.append(1))
    monkeypatch.setattr(runner.time, "sleep", _stop)
    monkeypatch.setenv("MENAGERIE_CPUS", "2")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "8")
    with pytest.raises(_Stop):
        runner.serve(spool=tmp_path / "spool")
    assert started == []


def test_render_staging_hardlinks_the_portable_save(tmp_path: Path) -> None:
    save = tmp_path / "trace" / "render.tlspec"
    (save / "blobs").mkdir(parents=True)
    (save / "blobs" / "0001.safetensors").write_bytes(b"x" * 4096)
    staged = tmp_path / "render" / "render.tlspec"
    flow.stage_render(save, staged)
    source, copy = save / "blobs" / "0001.safetensors", staged / "blobs" / "0001.safetensors"
    assert copy.read_bytes() == source.read_bytes()
    assert copy.stat().st_ino == source.stat().st_ino


def _answer(request: RunRequest) -> RunResult:
    """The forward passes with a structural divergence; the full eval trace is over the cap."""

    over = request.mode == "trace"
    cls = "oversized:output" if over else None
    row = LedgerRow(id=request.id, stage=request.mode.split("-")[0], at="2026-10-18T00:00:00Z",
                    host="worker", env="torch-main@00000000", recipe_rev=request.recipe.rev,
                    outcome="fail" if over else "pass", cls=cls, mode=request.model_mode)
    return RunResult(request=request.request, id=request.id, mode=request.mode,
                     outcome="fail" if over else "pass", started="2026-10-18T00:00:00Z",
                     ended="2026-10-18T00:00:02Z", host="worker", cls=cls,
                     output_shapes=() if over else ((1, 8),),
                     modes=Modes(eval="pass", train="pass", divergence="structural"),
                     rows=(row,))


def test_an_over_cap_eval_trace_gets_structure_only_and_no_train_trace(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_CPUS", "2")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "8")
    data = DataLayout(tmp_path / "data")
    record = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    write_json(record, data.model("m8839"))
    seen: list[tuple[str, str | None]] = []

    def answer(request, outcome, **_kw):
        seen.append((request.mode, request.model_mode))
        result = _answer(request)
        outcome.results.append(result)
        return result

    monkeypatch.setattr(flow, "_run", answer)
    flow.run_model(replace(record), caps=None, data=data, store=StoreLayout(tmp_path / "store"),
                   attempts=tmp_path / "attempts")
    modes = [mode for mode, _ in seen]
    assert modes[:3] == ["forward", "trace", "trace-structure-only"]
    assert ("trace", "train") not in seen
