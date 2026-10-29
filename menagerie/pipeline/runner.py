"""The runner: one sandboxed attempt per request, served from a spool on the worker.

Owned by the environment and runner lane. Each attempt runs ``python -m menagerie.pipeline
worker ATTEMPT_DIR`` in the recipe's environment (``MENAGERIE_ENVS``) inside bubblewrap with
the network unshared, the environment, code, data and store read-only, a private ``/tmp`` and
HOME, an empty process environment plus the offline switches, and the request's caps: the
process tree's RSS is sampled and killed above ``rss_mb`` (``memory_cap``), and its peak is
written into each row's ``peak_rss_mb`` when above the worker's own reading, the attempt is killed
after ``timeout_s`` (``timeout``), and BLAS and torch threads are set to ``threads``. The worker
writes ``result.json`` last; an attempt that ends without it is an ``infra`` result, never a
model failure. Caps come from ``host_budget`` (the job's declaration), never ``os.cpu_count()``.

The spool is a directory: ``new/<request>.json`` waits, ``claimed/`` is in flight,
``attempts/<request>/`` holds the attempt's files and ``done/<request>.json`` the final result;
an unreadable request file is moved to ``rejected/``. Forward requests (an agent's ``try``) start
ahead of trace and render. A server that dies leaves its claims behind; the next ``serve`` records
each as an ``infra`` result in its attempt directory (never a try) and puts it back in ``new/``.

A request's ``adapter_src`` is written into the attempt as ``adapter.py`` and loaded from there
(``MENAGERIE_ADAPTER``), so a draft runs without a commit. A recipe's ``vendor`` snapshot is
unpacked once from the store (``MENAGERIE_STORE``) under ``work_dir("vendor")/<sha>/`` and bound
read-only; ``MENAGERIE_VENDOR`` names its root, which the worker puts first on ``sys.path`` once the
interpreter has started (``worker.vendor_on_path``), so a tree's folder named like a stdlib module
never replaces the stdlib's at start-up.
The recipe's and the request's extras are installed together into one overlay, wheels only.
"""

from __future__ import annotations

import fcntl
import glob
import hashlib
import os
import secrets
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from menagerie import host_budget
from menagerie.paths import work_dir
from menagerie.pipeline import (
    StoreLayout, attempt_watch, disk_guard, envs, now, read_json, write_json,
)
from menagerie.pipeline.records import (
    Caps, LedgerRow, ModelMode, ModelRecord, OutputFile, RecordError, RunMode, RunRequest,
    RunResult,
)

if TYPE_CHECKING:  # psutil is imported where an attempt is supervised, so the hub never needs it
    import psutil

DEFAULT_WORKERS = 3
TIMEOUT_S: dict[str, int] = {
    "forward": 600, "trace": 1800, "trace-structure-only": 600, "render": 1800, "gpu": 2400,
}
PEAK_HEADROOM = 1.25  # a recipe's measured peak times this is its RSS cap
POLL_S = 0.2
HEARTBEAT_STALE_S = 60
BACKFILL_S = 600  # a request that does not fit is passed by smaller ones for at most this long
TAIL_LINES = 40
LOG = "attempt.log"
ADAPTER = "adapter.py"
VENDOR_MARKER = ".menagerie-vendor"
CODE_ROOT = Path(__file__).resolve().parents[2]
WORKER: tuple[str, ...] = ("-m", "menagerie.pipeline", "worker")
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "YOLO_OFFLINE": "true", "WANDB_MODE": "offline"}
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "MENAGERIE_CPUS")
CUDA_HOME = "/usr/local/cuda"  # the host's CUDA toolkit (under /usr, so already bound read-only)
# Read-only system paths inside the sandbox; /etc is limited to what loaders and fonts need.
SYSTEM_RO = ("/usr", "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/fonts",
             "/etc/alternatives", "/etc/ssl", "/etc/localtime", "/etc/passwd", "/etc/group")


def host_label() -> str:
    """Return the generic host label written into rows (machine names never enter the repo)."""

    return os.environ.get("MENAGERIE_HOST", "worker")


def default_caps(mode: RunMode) -> Caps:
    """Return the caps for a mode, sized from ``host_budget`` (never ``os.cpu_count()``)."""

    cpus = host_budget.declared_cpus()
    workers = min(DEFAULT_WORKERS, cpus)
    rss_mb = int(host_budget.declared_memory_gb() * 1024 / workers)
    return Caps(rss_mb=rss_mb, timeout_s=TIMEOUT_S[mode],
                threads=host_budget.threads_per_worker(cpus, workers))


def request_for(record: ModelRecord, mode: RunMode, *, caps: Caps | None = None,
                intervention_ready: bool = True, model_mode: ModelMode = "eval",
                trace: str | None = None, extras: Sequence[str] = (),
                adapter_src: str | None = None) -> RunRequest:
    """Build the request for one model record; ``trace`` names the ``.tlspec`` for render mode.

    ``adapter_src`` carries an adapter recipe's file text, so the worker needs no data checkout.
    """

    if record.recipe is None:
        raise ValueError(f"{record.id} has no recipe to run")
    if mode != "render" and record.give_up and (
            record.give_up.cls.startswith("standin:") or
            record.give_up.cls == "june_code:flagged"):
        raise RecordError(f"{record.id}: {record.give_up.cls}: refuses a substitute or flagged recipe")
    if caps is None:
        caps = default_caps(mode)
        if record.recipe.memory_gib:
            budget_mb = int(host_budget.declared_memory_gb() * 1024)
            wanted = int(record.recipe.memory_gib * 1024 * PEAK_HEADROOM)
            caps = replace(caps, rss_mb=min(budget_mb, max(caps.rss_mb, wanted)))
    stamp = now().replace("-", "").replace(":", "").lower()
    name = f"{record.id}-{mode}-{model_mode}-{stamp}-{secrets.token_hex(2)}"
    return RunRequest(request=name, id=record.id, mode=mode, recipe=record.recipe, caps=caps,
                      intervention_ready=intervention_ready, model_mode=model_mode, trace=trace,
                      extras=tuple(extras), adapter_src=adapter_src)


def attempt(request: RunRequest, *, attempt_dir: Path) -> RunResult:
    """Run one sandboxed attempt in ``attempt_dir`` and return its result (``infra`` if none)."""

    attempt_dir = Path(attempt_dir).resolve()
    attempt_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("result.json", LOG):  # a retried attempt never reads its predecessor's files
        (attempt_dir / stale).unlink(missing_ok=True)
    write_json(request, attempt_dir / "request.json")
    if request.adapter_src is not None:
        (attempt_dir / ADAPTER).write_text(request.adapter_src, encoding="utf-8")
    started = now()
    try:
        interpreter, extra_ro = _interpreter(request)
    except FileNotFoundError as exc:
        return _made(request, started, "infra", "infra:env_missing", str(exc), attempt_dir)
    except envs.ExtrasError as exc:
        return _made(request, started, "fail", exc.cls, str(exc), attempt_dir)
    try:
        vendor = _vendor(request.recipe.vendor)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:  # RecordError is a ValueError
        msg = f"vendor {request.recipe.vendor} not unpacked: {exc}"[:300]
        return _made(request, started, "infra", "infra:vendor", msg, attempt_dir)
    argv = _sandbox_argv(attempt_dir, interpreter, [*extra_ro, *([vendor] if vendor else [])],
                         gpu=request.mode == "gpu")
    env = _sandbox_env(request, attempt_dir, vendor)
    how, code, peak_mb = _supervise(argv, attempt_dir / LOG, request.caps, env, attempt_dir)
    if how in (attempt_watch.OUTPUT_CAP, attempt_watch.DISK_LOW):
        return _watchdog_result(request, started, how, attempt_dir, peak_mb)
    result_path = attempt_dir / "result.json"
    if result_path.is_file():
        try:
            return _with_evidence(read_json(RunResult, result_path), attempt_dir, peak_mb)
        except (RecordError, ValueError) as exc:
            return _made(request, started, "infra", "infra:bad_result", str(exc)[:300],
                         attempt_dir, peak_mb)
    if how == "memory_cap":
        msg = f"process tree RSS reached {peak_mb} MB, over the {request.caps.rss_mb} MB cap"
        return _made(request, started, "fail", _cls(request, "memory_cap"), msg, attempt_dir,
                     peak_mb)
    if how == "timeout":
        msg = f"no result after {request.caps.timeout_s} s"
        return _made(request, started, "fail", _cls(request, "timeout"), msg, attempt_dir, peak_mb)
    msg = f"worker exited with code {code} without writing result.json"
    return _made(request, started, "infra", "infra:no_result", msg, attempt_dir, peak_mb)


def _watchdog_result(request: RunRequest, started: str, how: str, attempt_dir: Path,
                     peak_mb: int) -> RunResult:
    """The result of an attempt the disk watchdog killed; its saves are deleted first."""

    attempt_watch.shed(attempt_dir)
    (attempt_dir / "result.json").unlink(missing_ok=True)
    if how == attempt_watch.DISK_LOW:
        msg = "worker free disk fell under the fleet floor plus margin while the attempt ran"
        return _made(request, started, "infra", "infra:disk_low", msg, attempt_dir, peak_mb)
    cap = attempt_watch.max_bytes() or 0
    msg = (f"the attempt wrote over {cap / disk_guard.GIB:g} GiB "
           f"({attempt_watch.MAX_ENV}); killed and its saves deleted")
    return _made(request, started, "fail", _cls(request, "oversized:output"), msg, attempt_dir,
                 peak_mb)


def not_run(request: RunRequest, cls: str, msg: str) -> RunResult:
    """The typed failure of a request decided without an attempt (a projection beyond the worker)."""

    return _made(request, now(), "fail", cls, msg, work_dir("attempts") / request.request)


def run(request: RunRequest, *, spool: Path | None = None) -> RunResult:
    """Run a request now, or through the spool served by ``serve`` when ``spool`` is given.

    Through a spool it waits for the result; it gives up only after the heartbeat has been stale
    for ``HEARTBEAT_STALE_S``, so a runner that is starting or restarting loses no request.
    """

    if spool is None:
        return attempt(request, attempt_dir=work_dir("attempts") / request.request)
    done = submit(request, spool=spool)
    gone_since = None  # a runner starting or restarting has a stale heartbeat for a moment
    while not done.is_file():
        if serving(spool):
            gone_since = None
        elif gone_since is None:
            gone_since = time.monotonic()
        elif time.monotonic() - gone_since > HEARTBEAT_STALE_S:
            raise RuntimeError(f"no runner is serving {spool}; start `runner serve`")
        time.sleep(1)
    return read_json(RunResult, done)


def submit(request: RunRequest, *, spool: Path) -> Path:
    """Queue a request on the spool; return the path its final result will appear at."""

    dirs = _spool(spool)
    write_json(request, dirs["new"] / f"{request.request}.json")
    return dirs["done"] / f"{request.request}.json"


def serving(spool: Path) -> bool:
    """Return whether a server has touched the spool's heartbeat recently."""

    beat = Path(spool) / "serving"
    return beat.is_file() and time.time() - beat.stat().st_mtime < HEARTBEAT_STALE_S


def serve(*, spool: Path, workers: int | None = None, once: bool = False) -> None:
    """Serve the spool: run waiting requests with ``workers`` workers; ``once`` drains and exits.

    Requests start forward passes first, then oldest first, while a worker is free and the RSS caps of the running attempts
    plus the next one fit in the declared memory; a request whose cap exceeds the whole
    declaration is clamped to it and runs alone. A request that does not fit yet is passed by
    smaller ones behind it (backfill) until it has waited ``BACKFILL_S``; then nothing new
    starts until it fits.
    """

    cpus = host_budget.declared_cpus()
    workers = workers or min(DEFAULT_WORKERS, cpus)
    budget_mb = int(host_budget.declared_memory_gb() * 1024)
    threads = host_budget.threads_per_worker(cpus, workers)
    dirs = _spool(spool)
    requeue(dirs)
    running: dict[str, tuple[threading.Thread, int]] = {}
    modes: dict[str, bool] = {}
    while True:
        (Path(spool) / "serving").touch()
        for name in [n for n, (t, _) in running.items() if not t.is_alive()]:
            (dirs["claimed"] / f"{name}.json").unlink(missing_ok=True)
            del running[name]
        if disk_guard.claim_ok(dirs["attempts"]):  # below the job's pause line, claim nothing
            _start_waiting(dirs, modes, running, workers=workers, budget_mb=budget_mb,
                           threads=threads)
        if once and not running and not any(dirs["new"].glob("*.json")):
            return
        time.sleep(POLL_S * 5)


def _start_waiting(dirs: dict[str, Path], modes: dict[str, bool],
                   running: dict[str, tuple[threading.Thread, int]], *, workers: int,
                   budget_mb: int, threads: int) -> None:
    """Claim and start waiting requests while a worker is free and their caps fit."""

    for path in _waiting(dirs["new"], modes):
        try:
            request = _fit(read_json(RunRequest, path), budget_mb, threads)
        except (RecordError, ValueError):  # unreadable: set aside, never run, never retried
            os.replace(path, dirs["rejected"] / path.name)
            continue
        if len(running) >= workers:
            return
        if running and sum(mb for _, mb in running.values()) + request.caps.rss_mb > budget_mb:
            if _waited(path) > BACKFILL_S:
                return  # waited long enough: let the running attempts drain for it
            continue  # backfill: a smaller request behind it may fit now
        os.replace(path, dirs["claimed"] / path.name)
        worker = threading.Thread(target=_serve_one, args=(request, dirs), daemon=True)
        worker.start()
        running[request.request] = (worker, request.caps.rss_mb)


def _waited(path: Path) -> float:
    """Seconds since a waiting request was written (its file's mtime; 0 if it is gone)."""

    try:
        return time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def requeue(dirs: dict[str, Path]) -> list[str]:
    """Put a dead server's claims back in ``new/``, each recorded as an ``infra`` result.

    The record sits in the request's attempt directory (``infra-<n>.json``), never in ``done/``,
    so the caller waiting on ``done/`` gets the rerun's result and no try is spent on the restart.
    """

    names = []
    for stale in sorted(dirs["claimed"].glob("*.json")):
        try:
            request = read_json(RunRequest, stale)
        except (RecordError, ValueError):
            os.replace(stale, dirs["rejected"] / stale.name)
            continue
        attempt_dir = dirs["attempts"] / request.request
        attempt_dir.mkdir(parents=True, exist_ok=True)
        n = len(list(attempt_dir.glob("infra-*.json"))) + 1
        msg = "the runner restarted while this attempt ran; requeued"
        write_json(_made(request, now(), "infra", "infra:restart", msg, attempt_dir),
                   attempt_dir / f"infra-{n}.json")
        os.replace(stale, dirs["new"] / stale.name)
        names.append(request.request)
    return names


def _waiting(new: Path, forward: dict[str, bool]) -> list[Path]:
    """Waiting requests, forward passes first, then oldest first; ``forward`` caches the mode."""

    def key(path: Path) -> tuple[bool, float, str]:
        if path.name not in forward:
            try:
                forward[path.name] = read_json(RunRequest, path).mode == "forward"
            except (RecordError, ValueError, OSError):
                forward[path.name] = True  # rejected at once by the caller
        return (not forward[path.name], path.stat().st_mtime, path.name)

    listing, paths = list(new.glob("*.json")), []
    for name in set(forward) - {path.name for path in listing}:
        del forward[name]
    for path in listing:
        try:
            paths.append((key(path), path))
        except FileNotFoundError:  # claimed by now
            continue
    return [path for _, path in sorted(paths)]


def _serve_one(request: RunRequest, dirs: dict[str, Path]) -> None:
    attempt_dir = dirs["attempts"] / request.request
    try:
        result = attempt(request, attempt_dir=attempt_dir)
    except Exception as exc:  # the runner's own fault: never lose the request, never blame it
        result = _made(request, now(), "infra", "infra:runner", repr(exc)[:300], attempt_dir)
    write_json(result, dirs["done"] / f"{request.request}.json")


def _fit(request: RunRequest, budget_mb: int, threads: int) -> RunRequest:
    caps = replace(request.caps, rss_mb=min(request.caps.rss_mb, budget_mb),
                   threads=max(1, min(request.caps.threads, threads)))
    return replace(request, caps=caps)


def _spool(spool: Path) -> dict[str, Path]:
    dirs = {name: Path(spool) / name for name in ("new", "claimed", "attempts", "done", "rejected")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _envs_root() -> Path:
    root = os.environ.get("MENAGERIE_ENVS")
    if not root:
        raise FileNotFoundError("MENAGERIE_ENVS is not set, so no environment can be found")
    return Path(root)


def _interpreter(request: RunRequest) -> tuple[Path, list[Path]]:
    """Return the interpreter for the request and the trees it needs bound read-only."""

    env, root = request.recipe.env, _envs_root()
    extras = sorted({*request.recipe.extras, *request.extras})
    if extras:
        chosen = envs.overlay(env, extras, root=root)
    else:
        chosen = envs.python(env, root=root)
    trees, link = [chosen.parents[1]], chosen
    while link.is_symlink():  # the venv's interpreter link and uv's minor-version directory link
        link = link.parent / os.readlink(link)
        trees.append(link.parents[1])
    trees.append(Path(os.path.realpath(chosen)).parents[1])
    return chosen, trees


def unsandboxed() -> bool:
    """Whether attempts run without bubblewrap: only inside a SLURM job that asks for it.

    The cluster has no user namespaces, so its batch job sets ``MENAGERIE_SANDBOX=none`` and the
    SLURM allocation is the boundary (``cluster.run``). Anywhere else the setting is refused.
    """

    if os.environ.get("MENAGERIE_SANDBOX") != "none":
        return False
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("MENAGERIE_SANDBOX=none is honoured only inside a SLURM job")
    return True


def _sandbox_argv(attempt_dir: Path, interpreter: Path, extra_ro: Sequence[Path], *,
                  gpu: bool = False) -> list[str]:
    if unsandboxed():  # the worker starts in its attempt directory, as bwrap's --chdir does
        return ["/bin/sh", "-c", 'cd "$0" && exec "$@"', str(attempt_dir), str(interpreter),
                *WORKER, str(attempt_dir)]
    argv = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/etc"]
    if gpu:  # only a `gpu` request sees the NVIDIA devices (the worker's GPU window mode)
        for device in sorted(glob.glob("/dev/nvidia*")):
            argv += ["--dev-bind", device, device]
    for name in ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"):
        if os.path.islink(name):
            argv += ["--symlink", os.readlink(name), name]
        elif os.path.isdir(name):
            argv += ["--ro-bind", name, name]
    data = [os.environ.get(k) for k in ("MENAGERIE_DATA", "MENAGERIE_STORE")]
    for path in [*SYSTEM_RO, *map(str, extra_ro), str(CODE_ROOT), *filter(None, data)]:
        if os.path.exists(path):
            argv += ["--ro-bind", path, path]
    argv += ["--bind", str(attempt_dir), str(attempt_dir), "--chdir", str(attempt_dir)]
    return [*argv, str(interpreter), *WORKER, str(attempt_dir)]


def _sandbox_env(request: RunRequest, attempt_dir: Path,
                 vendor: Path | None = None) -> dict[str, str]:
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8", "TMPDIR": "/tmp",
           "PYTHONPATH": str(CODE_ROOT), "PYTHONDONTWRITEBYTECODE": "1",
           "XDG_CACHE_HOME": "/tmp/cache", "MPLCONFIGDIR": "/tmp/mpl", "HF_HOME": "/tmp/hf",
           "TORCH_HOME": "/tmp/torch", "YOLO_CONFIG_DIR": "/tmp/yolo",
           "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions",
           "MENAGERIE_HOST": host_label(), "MENAGERIE_ATTEMPT": str(attempt_dir),
           "MENAGERIE_MEMORY_GB": f"{request.caps.rss_mb / 1024:.3f}",
           **OFFLINE, **{k: str(request.caps.threads) for k in THREAD_VARS}}
    for key in ("MENAGERIE_DATA", "MENAGERIE_STORE"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    if request.adapter_src is not None:
        env["MENAGERIE_ADAPTER"] = str(attempt_dir / ADAPTER)
    if request.mode == "gpu" and os.path.isfile(f"{CUDA_HOME}/bin/nvcc"):
        # torch's extension builder finds nvcc here; a failed build is the recipe's compiled_ext.
        env["CUDA_HOME"], env["PATH"] = CUDA_HOME, f"{CUDA_HOME}/bin:{env['PATH']}"
    if vendor:
        env["MENAGERIE_VENDOR"] = str(_vendor_root(vendor))
    return _unsandboxed_env(env, attempt_dir) if unsandboxed() else env


# The job's toolchain (a gpu batch's CUDA and compiler modules) reaches an unsandboxed attempt.
TOOLCHAIN_VARS = ("CUDA_HOME", "CC", "CXX", "TORCH_CUDA_ARCH_LIST", "LD_LIBRARY_PATH")


def _unsandboxed_env(env: dict[str, str], attempt_dir: Path) -> dict[str, str]:
    """The sandbox's environment with its private ``/tmp`` moved into the attempt directory."""

    private = attempt_dir / "tmp"
    private.mkdir(parents=True, exist_ok=True)
    moved = {k: str(private) + v[4:] if v == "/tmp" or v.startswith("/tmp/") else v
             for k, v in env.items()}
    moved["PATH"] = os.environ.get("PATH", moved["PATH"])
    moved.update({k: os.environ[k] for k in TOOLCHAIN_VARS if os.environ.get(k)})
    return moved


def _vendor(sha: str | None) -> Path | None:
    """Unpack a vendor snapshot once from the store; return its directory (``None`` without)."""

    if sha is None:
        return None
    target = work_dir("vendor") / sha
    if (target / VENDOR_MARKER).is_file():
        (target / VENDOR_MARKER).touch()  # last use, read by `reap_vendor`
        return target
    from menagerie.pipeline import store

    with _locked(work_dir("vendor") / f".{sha}.lock"):
        if not (target / VENDOR_MARKER).is_file():
            shutil.rmtree(target, ignore_errors=True)
            store.materialize(sha, target, store=StoreLayout.from_env())
            target.chmod(target.stat().st_mode | 0o200)
            (target / VENDOR_MARKER).write_text(sha + "\n")
    return target


def reap_vendor(*, idle_s: float) -> list[str]:
    """Delete unpacked vendor snapshots unused for ``idle_s`` seconds; return their shas.

    Each use touches the snapshot's marker, and an attempt runs at most ``TIMEOUT_S["gpu"]``, so
    an idle time well over that never removes a tree a running attempt has bound. The store keeps
    every snapshot, so a reaped one is unpacked again on its next use.
    """

    root, reaped = work_dir("vendor"), []
    for marker in sorted(root.glob(f"*/{VENDOR_MARKER}")) if root.is_dir() else ():
        sha = marker.parent.name
        with _locked(root / f".{sha}.lock"):
            if marker.is_file() and time.time() - marker.stat().st_mtime > idle_s:
                shutil.rmtree(marker.parent, ignore_errors=True)
                if not marker.parent.exists():
                    reaped.append(sha)
    return reaped


def _vendor_root(vendor: Path) -> Path:
    """The snapshot's top directory: a tarball's single ``<repo>-<rev>/`` folder, if that is all."""

    entries = [p for p in vendor.iterdir() if p.name != VENDOR_MARKER]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else vendor


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _supervise(argv: list[str], log: Path, caps: Caps, env: dict[str, str],
               attempt_dir: Path | None = None) -> tuple[str, int, int]:
    """Run the sandbox; return how it ended (exit, memory_cap, timeout, or the disk watchdog's
    output_cap or disk_low, checked every ``attempt_watch.POLL_S`` in ``attempt_dir``), its code
    and peak MB."""

    import psutil

    with log.open("wb") as out:
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        tree, peak_mb = psutil.Process(proc.pid), 0
        deadline = time.monotonic() + caps.timeout_s
        next_disk = time.monotonic() + attempt_watch.POLL_S
        while proc.poll() is None:
            rss_mb = _tree_rss_mb(tree)
            peak_mb = max(peak_mb, rss_mb)
            how = "memory_cap" if rss_mb > caps.rss_mb else (
                "timeout" if time.monotonic() > deadline else None)
            if how is None and attempt_dir is not None and time.monotonic() >= next_disk:
                how, next_disk = attempt_watch.breach(attempt_dir), next_disk + attempt_watch.POLL_S
            if how is not None:
                _kill(proc, tree)
                return how, proc.returncode, peak_mb
            time.sleep(POLL_S)
    return "exit", proc.returncode, peak_mb


def _tree_rss_mb(tree: psutil.Process) -> int:
    import psutil

    total = 0
    try:
        members = [tree, *tree.children(recursive=True)]
    except psutil.Error:
        return 0
    for member in members:
        try:
            total += member.memory_info().rss
        except psutil.Error:
            continue
    return total // (1024 * 1024)


def _kill(proc: subprocess.Popen[bytes], tree: psutil.Process) -> None:
    import psutil

    try:
        members = tree.children(recursive=True)
    except psutil.Error:
        members = []
    for target in (proc.pid, *(m.pid for m in members)):
        try:
            os.kill(target, signal.SIGKILL)
        except ProcessLookupError:
            continue
    proc.wait()


def _cls(request: RunRequest, head: str) -> str:
    return f"render:{head}" if request.mode == "render" else head


def _log_file(attempt_dir: Path) -> OutputFile | None:
    log = attempt_dir / LOG
    if not log.is_file():
        return None
    data = log.read_bytes()
    return OutputFile(kind="log", path=LOG, sha256=hashlib.sha256(data).hexdigest(),
                      bytes=len(data))


def _with_evidence(result: RunResult, attempt_dir: Path, peak_mb: int | None = None) -> RunResult:
    """Point the worker's rows at the attempt log, which only the runner sees whole, and give each
    row the process tree's peak RSS when it is above the worker's own (the cap is on the tree)."""

    def seen(row: Any) -> Any:
        if isinstance(row, LedgerRow) and peak_mb and peak_mb > (row.peak_rss_mb or 0):
            row = replace(row, peak_rss_mb=peak_mb)
        return row if row.evidence or log is None else replace(row, evidence=log.sha256)

    log = _log_file(attempt_dir)
    rows = tuple(seen(row) for row in result.rows)
    return replace(result, rows=rows, files=(*result.files, *([log] if log else [])))


def _made(request: RunRequest, started: str, outcome: str, cls: str, msg: str,
          attempt_dir: Path, peak_mb: int | None = None) -> RunResult:
    """Build the result for an attempt that ended without a usable ``result.json``."""

    log = _log_file(attempt_dir)
    tail: tuple[str, ...] = ()
    if log is not None:
        lines = (attempt_dir / LOG).read_text(errors="replace").splitlines()
        tail = tuple(lines[-TAIL_LINES:])
    rows = ()
    if request.mode != "render":  # render rows need the renderer's revision; the caller adds them
        rows = (_ledger_row(request, outcome, cls, msg, peak_mb, log),)
    return RunResult(request=request.request, id=request.id, mode=request.mode, outcome=outcome,
                     started=started, ended=now(), host=host_label(), cls=cls, msg=msg,
                     tail=tail, rows=rows, files=(log,) if log else ())


def _ledger_row(request: RunRequest, outcome: str, cls: str, msg: str, peak_mb: int | None,
                log: OutputFile | None) -> LedgerRow:
    recipe, structure = request.recipe, request.mode == "trace-structure-only"
    try:
        env, torch = envs.lock_id(recipe.env), envs.package_version(recipe.env, "torch")
        tl = envs.package_version(recipe.env, "torchlens")
    except FileNotFoundError:
        env, torch, tl = recipe.env, None, None
    return LedgerRow(
        id=request.id, stage="forward" if request.mode == "forward" else "trace", at=now(),
        host=host_label(), env=env, recipe_rev=recipe.rev, outcome=outcome,
        mode=request.model_mode, capture=None if request.mode == "forward" else
        ("structure_only" if structure else "full"), backend=recipe.backend, tl=tl,
        fw=f"torch {torch}" if torch else None, cls=cls, peak_rss_mb=peak_mb,
        validation_state="structure_only" if structure else None,
        evidence=log.sha256 if log else None, msg=msg)
