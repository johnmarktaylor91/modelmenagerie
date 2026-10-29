"""The runner on the worker: real sandboxed attempts in the built ``torch-main``.

Needs ``MENAGERIE_ENVS`` with ``torch-main`` built and ``bwrap`` on the path; the caps tests
need neither. ``runner_probe.py`` stands in for the worker module (owned by the worker lane), so
these attempts run a real timm resnet18 forward and force each way an attempt can end.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.pipeline import envs, read_json, write_json
from menagerie.pipeline import runner
from menagerie.pipeline.records import Caps, ModelRecord, RunRequest, RunResult

FIXTURES = Path(__file__).with_name("fixtures")
ROOT = Path(os.environ.get("MENAGERIE_ENVS", "/nonexistent"))


def _built() -> bool:
    try:
        envs.python("torch-main", root=ROOT)
    except FileNotFoundError:
        return False
    return shutil.which("bwrap") is not None


sandboxed = pytest.mark.skipif(not _built(), reason="needs MENAGERIE_ENVS with torch-main built and bwrap")


@pytest.fixture
def probe_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "WORKER", (str(Path(__file__).with_name("runner_probe.py")),))


def _resnet18() -> ModelRecord:
    return read_json(ModelRecord, FIXTURES / "m8839.model.json")


def _request(call: str, *, mode: str = "forward", caps: Caps | None = None,
             kwargs: dict | None = None, extras: tuple[str, ...] = ()) -> RunRequest:
    record = _resnet18()
    recipe = replace(record.recipe, call=call, args=(), kwargs=kwargs or {})
    caps = caps or Caps(rss_mb=4096, timeout_s=300, threads=2)
    return runner.request_for(replace(record, recipe=recipe), mode, caps=caps, extras=extras)


def test_default_caps_follow_the_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_CPUS", "6")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "12")
    assert runner.default_caps("trace") == Caps(rss_mb=4096, timeout_s=1800, threads=2)
    record = _resnet18()
    big = replace(record, recipe=replace(record.recipe, memory_gib=8.0))
    huge = replace(record, recipe=replace(record.recipe, memory_gib=60.0))
    assert runner.request_for(big, "trace").caps.rss_mb == 10240
    assert runner.request_for(huge, "trace").caps.rss_mb == 12288
    request = runner.request_for(record, "forward")
    assert request.request.startswith("m8839-forward-eval-") and request.recipe == record.recipe


@sandboxed
def test_resnet18_passes_in_the_sandbox(tmp_path: Path, probe_worker: None) -> None:
    request = runner.request_for(_resnet18(), "forward",
                                 caps=Caps(rss_mb=4096, timeout_s=300, threads=2))
    result = runner.attempt(request, attempt_dir=tmp_path / "a")
    assert result.outcome == "pass", result.tail
    assert result.output_shapes == ((1, 1000),) and result.params == 11689512
    (row,) = result.rows
    assert row.stage == "forward" and row.outcome == "pass" and row.fw == "torch 2.6.0+cpu"
    assert row.peak_rss_mb and row.peak_rss_mb > 100  # the process tree's peak, which the cap sees
    (log,) = [f for f in result.files if f.kind == "log"]
    assert row.evidence == log.sha256


@sandboxed
def test_sandbox_hides_network_home_and_protects_env(tmp_path: Path, probe_worker: None) -> None:
    request = _request("probe.isolation", kwargs={"home": str(Path.home())}, extras=("tabulate",))
    result = runner.attempt(request, attempt_dir=tmp_path / "a")
    assert result.outcome == "pass", result.tail
    assert result.msg == "isolated"


@sandboxed
def test_forced_oversize_is_memory_cap(tmp_path: Path, probe_worker: None) -> None:
    request = _request("probe.oversize", mode="trace",
                       caps=Caps(rss_mb=1024, timeout_s=300, threads=1))
    result = runner.attempt(request, attempt_dir=tmp_path / "a")
    assert (result.outcome, result.cls) == ("fail", "memory_cap"), result.msg
    (row,) = result.rows
    assert row.stage == "trace" and row.capture == "full" and row.peak_rss_mb >= 1024
    assert row.env == envs.lock_id("torch-main")
    assert row.tl == envs.package_version("torch-main", "torchlens")  # the locked pin


@sandboxed
def test_forced_hang_is_timeout(tmp_path: Path, probe_worker: None) -> None:
    clock = time.monotonic()
    request = _request("probe.hang", caps=Caps(rss_mb=4096, timeout_s=5, threads=1))
    result = runner.attempt(request, attempt_dir=tmp_path / "a")
    assert (result.outcome, result.cls) == ("fail", "timeout"), result.msg
    assert time.monotonic() - clock < 60
    assert result.rows[0].stage == "forward"


@sandboxed
def test_death_without_result_is_infra(tmp_path: Path, probe_worker: None) -> None:
    request = _request("probe.die", mode="trace-structure-only")
    result = runner.attempt(request, attempt_dir=tmp_path / "a")
    assert (result.outcome, result.cls) == ("infra", "infra:no_result"), result.msg
    (row,) = result.rows
    assert row.outcome == "infra" and row.capture == "structure_only"


@sandboxed
def test_spool_serves_requeues_and_rejects(tmp_path: Path, probe_worker: None) -> None:
    spool = tmp_path / "spool"
    passing = runner.request_for(_resnet18(), "forward",
                                 caps=Caps(rss_mb=4096, timeout_s=300, threads=2))
    dying = _request("probe.die")
    stale = _request("probe.die")
    runner.submit(passing, spool=spool)
    runner.submit(dying, spool=spool)
    write_json(stale, spool / "claimed" / f"{stale.request}.json")  # left by a dead server
    (spool / "new" / "garbage.json").write_text("{}")
    runner.serve(spool=spool, workers=2, once=True)
    done = {r.request: read_json(RunResult, spool / "done" / f"{r.request}.json")
            for r in (passing, dying, stale)}
    assert done[passing.request].outcome == "pass"
    assert done[dying.request].cls == done[stale.request].cls == "infra:no_result"
    assert not list((spool / "claimed").iterdir()) and not list((spool / "new").iterdir())
    assert (spool / "rejected" / "garbage.json").is_file()
    assert runner.serving(spool)


def test_serve_backfills_small_requests_until_a_big_one_has_waited(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that does not fit is passed by smaller ones, but only for ``BACKFILL_S``."""

    monkeypatch.setenv("MENAGERIE_CPUS", "3")
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "16")
    started: list[str] = []

    def serve_one(request: RunRequest, dirs: dict[str, Path]) -> None:
        started.append(request.request)
        time.sleep(0.5)
        write_json(runner._made(request, "2026-10-04T00:00:00Z", "infra", "infra:test", "",
                                dirs["attempts"] / request.request), dirs["done"] / f"{request.request}.json")

    monkeypatch.setattr(runner, "_serve_one", serve_one)

    def queue(spool: Path, name: str, mb: int, age: float) -> None:
        request = replace(_request("probe.die"), request=name,
                          caps=Caps(rss_mb=mb, timeout_s=60, threads=1))
        path = runner.submit(request, spool=spool).parent.parent / "new" / f"{name}.json"
        os.utime(path, (time.time() - age, time.time() - age))

    fresh, old = tmp_path / "fresh", tmp_path / "old"
    for spool, age in ((fresh, 0.0), (old, runner.BACKFILL_S + 60)):
        started.clear()
        queue(spool, "a-running", 9 * 1024, age + 3)
        queue(spool, "b-big", 9 * 1024, age + 2)
        queue(spool, "c-small", 4 * 1024, age + 1)
        runner.serve(spool=spool, once=True)
        assert sorted(started) == ["a-running", "b-big", "c-small"]
        order = started.copy()
        if spool == fresh:
            assert order.index("c-small") < order.index("b-big")  # backfilled
        else:
            assert order == ["a-running", "b-big", "c-small"]  # the big one waited too long


def test_unsandboxed_mode_is_for_slurm_jobs_only(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """The cluster batch's mode: no bwrap, the worker started in its attempt directory, the
    private /tmp moved into it and the job's toolchain passed through; refused outside SLURM."""

    request = runner.request_for(_resnet18(), "forward",
                                 caps=Caps(rss_mb=4096, timeout_s=300, threads=2))
    assert runner._sandbox_argv(tmp_path, Path("/env/bin/python"), [])[0] == "bwrap"
    monkeypatch.setenv("MENAGERIE_SANDBOX", "none")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="SLURM"):
        runner._sandbox_argv(tmp_path, Path("/env/bin/python"), [])
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda")
    argv = runner._sandbox_argv(tmp_path, Path("/env/bin/python"), [])
    assert argv[:2] == ["/bin/sh", "-c"] and "bwrap" not in argv
    assert argv[3:5] == [str(tmp_path), "/env/bin/python"] and argv[-1] == str(tmp_path)
    env = runner._sandbox_env(request, tmp_path)
    private = tmp_path / "tmp"
    assert env["HOME"] == env["TMPDIR"] == str(private) and private.is_dir()
    assert env["HF_HOME"] == str(private / "hf") and env["CUDA_HOME"] == "/opt/cuda"
    assert env["HF_HUB_OFFLINE"] == "1" and env["MENAGERIE_MEMORY_GB"] == "4.000"


@pytest.mark.skipif(not ROOT.joinpath(envs.lock_id("torch-main").replace("@", "-")).is_dir(),
                    reason="needs MENAGERIE_ENVS with torch-main built")
def test_resnet18_passes_unsandboxed(tmp_path: Path, probe_worker: None,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_SANDBOX", "none")
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    request = runner.request_for(_resnet18(), "forward",
                                 caps=Caps(rss_mb=4096, timeout_s=300, threads=2))
    result = runner.attempt(request, attempt_dir=tmp_path / "a")
    assert result.outcome == "pass", result.tail
    assert result.output_shapes == ((1, 1000),)


def test_a_vendor_folder_named_like_a_stdlib_module_never_replaces_it(tmp_path: Path) -> None:
    """The vendor root reaches the worker as ``MENAGERIE_VENDOR``, never ``PYTHONPATH``; the worker
    puts it ahead of the code root once started, so the tree's own modules import and a top-level
    ``importlib`` copy (f-uiec2net-uice2-net) does not break the interpreter's start."""

    import subprocess
    import sys

    tree = tmp_path / "vendor" / "repo-rev"
    (tree / "importlib").mkdir(parents=True)
    (tree / "importlib" / "__init__.py").write_text("raise RuntimeError('vendored importlib loaded')\n")
    (tree / "vendored_mod.py").write_text("NAME = 'vendored'\n")
    (tmp_path / "vendor" / runner.VENDOR_MARKER).write_text("sha\n")
    request = runner.request_for(_resnet18(), "forward", caps=Caps(rss_mb=4096, timeout_s=300, threads=2))
    env = runner._sandbox_env(request, tmp_path, tmp_path / "vendor")
    assert env["PYTHONPATH"] == str(runner.CODE_ROOT) and env["MENAGERIE_VENDOR"] == str(tree)
    probe = ("import importlib, os, sys\n"
             "from menagerie.pipeline.worker import vendor_on_path\n"
             "vendor_on_path(sys.path, os.environ)\n"
             "import vendored_mod\n"
             "assert sys.path.index(os.environ['MENAGERIE_VENDOR']) < sys.path.index(os.environ['CODE'])\n"
             "print(vendored_mod.NAME, importlib.__file__, os.environ['PYTHONPATH'])\n")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120,
                          env={"PATH": os.environ["PATH"], "PYTHONPATH": env["PYTHONPATH"],
                               "MENAGERIE_VENDOR": env["MENAGERIE_VENDOR"], "CODE": str(runner.CODE_ROOT)})
    assert done.returncode == 0, done.stderr
    name, stdlib, children = done.stdout.split()
    assert name == "vendored" and str(tree) not in stdlib
    assert children.split(os.pathsep) == [str(tree), str(runner.CODE_ROOT)]
    # as PYTHONPATH (the old way) the copy replaces the stdlib's before any code runs
    old = subprocess.run([sys.executable, "-m", "json.tool", "--help"], capture_output=True, text=True,
                         timeout=120, env={"PATH": os.environ["PATH"], "PYTHONPATH": str(tree)})
    assert "vendored importlib loaded" in old.stderr or old.returncode != 0
