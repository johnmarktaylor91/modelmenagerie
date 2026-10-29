"""The hub-to-runner bridge: spool files forwarded to the runner's host over one ssh connection.

Split from ``implement`` at rung 2 (``implement.bridge`` and the ``bridge serve`` verb call
``serve``). Runs on the hub; imports neither torch nor psutil.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, StoreLayout, now, read_json, runner, write_json
from menagerie.pipeline.implement import _NAME, POLL_S, _infra
from menagerie.pipeline.records import RecordError, RunRequest

FAST_POLL_S = 5  # a fresh claim waits: its result reaches the waiting lane within about 5 s
IDLE_POLL_S = 20  # nothing waits: still refreshes `serving` 3 times per runner.HEARTBEAT_STALE_S
# A claim older than this no longer holds the fast rate (a try is capped at
# implement.TRY_TIMEOUT_S = 600 s; the rest allows for queueing); its result is still collected.
CLAIM_FRESH_S = 3600
# The bridge and the mini relay run for days and import pipeline code once, so a merged fix to
# what the relay runs in process (the verb checks) never took effect: done waited 20 to 110
# minutes on a 10-21 copy of ``implement._passing`` until a restart on 27 Oct. When a module it
# imported changes on disk, the bridge exits for systemd (Restart=always) to start it on the new
# code: once the relay has no verb in progress, or after STALE_MAX_S regardless (the relay's
# reconcile answers a verb cut off mid-run as uncertain, never reruns it).
STALE_CHECK_S = 60
STALE_MAX_S = 30 * 60
# One ssh per poll: the runner's heartbeat age, its clock, then each waiting request's result
# (names on stdin, already checked against implement._NAME) as name NUL body NUL; `true` keeps
# a last name with no result yet from failing the command.
POLL_COMMAND = ("cd {remote} 2>/dev/null && {{ stat -c %Y serving 2>/dev/null || echo 0; "
                "date +%s; while read -r n; do [ -f \"done/$n.json\" ] && printf '%s\\0' \"$n\" "
                "&& cat \"done/$n.json\" && printf '\\0'; done; true; }}")


class _Link:
    """One ssh ControlMaster connection to the runner's host; commands reuse it."""

    def __init__(self, host: str, control: Path) -> None:
        self.host, self.control = host, control
        self.master: subprocess.Popen[bytes] | None = None

    def ensure(self) -> None:
        if self.master is not None and self.master.poll() is None:
            return
        self.control.unlink(missing_ok=True)
        self.master = subprocess.Popen(
            # ControlPersist=no keeps the master in the foreground so poll() tracks it; a
            # ControlPersist from ~/.ssh/config forks it away and every run() leaks one.
            ["ssh", "-M", "-N", "-S", str(self.control), "-o", "ControlPersist=no",
             "-o", "ServerAliveInterval=15",
             "-o", "ServerAliveCountMax=3", "-o", "BatchMode=yes", self.host],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 30
        while not self.control.exists() and time.monotonic() < deadline:
            time.sleep(0.1)

    def run(self, command: str, *, data: bytes | None = None) -> bytes:
        self.ensure()
        done = subprocess.run(["ssh", "-S", str(self.control), "-o", "BatchMode=yes",
                               self.host, command], input=data, capture_output=True,
                              timeout=120, check=True)
        return done.stdout

    def close(self) -> None:
        if self.master is not None:
            self.master.terminate()
            self.master.wait(timeout=10)
        self.control.unlink(missing_ok=True)


def serve(*, spool: Path | None = None) -> None:
    """Serve the hub-to-runner bridge until stopped.

    Inputs: the local bridge spool (default ``MENAGERIE_BRIDGE_SPOOL``, else
    ``work_dir("hubspool")``); the runner's ssh host ``MENAGERIE_RUNNER_HOST`` and its spool
    ``MENAGERIE_RUNNER_SPOOL`` (default ``scratch/menagerie/spool`` under the remote home).
    Each request in ``new/`` is forwarded (after draining its ``vendor`` blob to the worker
    store, ``store.drain``) and moved to ``claimed/``; each result in the remote ``done/`` is
    copied back to ``done/``; ``serving`` is touched while the remote runner's heartbeat is
    fresh. Returns: nothing; runs until interrupted. Side effects: files in both spools only.
    """

    host = os.environ.get("MENAGERIE_RUNNER_HOST")
    if not host:
        raise RecordError("set MENAGERIE_RUNNER_HOST to the runner's ssh host")
    local = Path(spool or os.environ.get("MENAGERIE_BRIDGE_SPOOL") or work_dir("hubspool"))
    remote = os.environ.get("MENAGERIE_RUNNER_SPOOL", "scratch/menagerie/spool")
    for name in ("new", "claimed", "done", "rejected"):
        (local / name).mkdir(parents=True, exist_ok=True)
    link = _Link(host, local / ".ssh-control")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # a stop still closes the link
    _start_mini_relay()
    last = float("-inf")  # monotonic time of the last remote poll
    code = LoadedCode()
    try:
        while not code.restart_due():
            try:
                last = _bridge_once(link, local, remote, last)
            except (subprocess.SubprocessError, OSError) as exc:  # a dropped link: retry
                print(f"{now()} bridge: {exc}", file=sys.stderr, flush=True)
                time.sleep(5)
            time.sleep(POLL_S)
        print(f"{now()} bridge: code changed on disk ({', '.join(code.changed[:5])}); exiting "
              "so systemd starts the new code", file=sys.stderr, flush=True)
    finally:
        link.close()


class LoadedCode:
    """The Menagerie source files this process has imported, and whether one changed since."""

    def __init__(self, idle: Callable[[], bool] | None = None,
                 clock: Callable[[], float] = time.monotonic, root: Path | None = None) -> None:
        self.root = str(root or Path(__file__).resolve().parents[1])  # the ``menagerie`` package
        self.seen: dict[str, float] = {}
        self.changed: list[str] = []
        self.idle, self.clock = idle or _relay_idle, clock
        self.checked, self.stale_since = clock(), None
        self.check()  # the baseline: what is loaded now

    def check(self) -> list[str]:
        """Files whose modification time moved since first seen (a module first seen now is
        taken as loaded now)."""
        for module in list(sys.modules.values()):
            path = getattr(module, "__file__", None)
            if not isinstance(path, str) or not path.startswith(self.root):
                continue
            try:
                mtime = os.stat(path).st_mtime
            except OSError:
                mtime = -1.0
            if self.seen.setdefault(path, mtime) != mtime and path not in self.changed:
                self.changed.append(path)
        return self.changed

    def restart_due(self) -> bool:
        if self.clock() - self.checked < STALE_CHECK_S:
            return False
        self.checked = self.clock()
        if not self.check():
            return False
        self.stale_since = self.stale_since or self.clock()
        return self.idle() or self.clock() - self.stale_since >= STALE_MAX_S


def _relay_idle() -> bool:
    from menagerie.pipeline import mini_relay

    return mini_relay.idle()


def _start_mini_relay() -> None:
    """The mini lanes' hub relay runs beside the bridge; it idles while no mini lease is live."""

    from menagerie.pipeline import mini_relay

    threading.Thread(target=mini_relay.serve, name="mini-relay", daemon=True).start()


def _bridge_once(link: _Link, local: Path, remote: str, last: float) -> float:
    """Forward new requests now; poll the runner when its pace has passed since ``last``.

    The pace is ``FAST_POLL_S`` while a fresh claim waits, else ``IDLE_POLL_S``. Returns the
    monotonic time of the latest poll (``last`` when none ran this pass).
    """

    for path in sorted(local.glob("new/*.json")):
        _forward(link, path, local, remote)
    pace = FAST_POLL_S if _in_flight(local) else IDLE_POLL_S
    if time.monotonic() - last < pace:
        return last
    _poll(link, local, remote)
    return time.monotonic()


def _in_flight(local: Path) -> bool:
    cutoff = time.time() - CLAIM_FRESH_S  # a claim keeps its request's mtime (written to new/)
    return any(p.stat().st_mtime > cutoff for p in local.glob("claimed/*.json"))


def _poll(link: _Link, local: Path, remote: str) -> None:
    """Touch ``serving`` if the runner's heartbeat is fresh; settle every waiting result."""

    waiting = {p.stem for p in local.glob("claimed/*.json")}
    out = link.run(POLL_COMMAND.format(remote=remote),
                   data="".join(f"{name}\n" for name in sorted(waiting)).encode())
    head = out.split(b"\n", 2)
    if len(head) >= 2 and int(head[1]) - int(head[0]) < runner.HEARTBEAT_STALE_S:
        (local / "serving").touch()
    parts = head[2].split(b"\0") if len(head) == 3 else []
    for name, body in zip(parts[0::2], parts[1::2]):  # a trailing empty part is dropped
        if name.decode() in waiting:
            _settle_local(local, name.decode(), body)


def _settle_local(local: Path, name: str, body: bytes) -> None:
    tmp = local / "done" / f".{name}.part"
    tmp.write_bytes(body)
    os.replace(tmp, local / "done" / f"{name}.json")
    (local / "claimed" / f"{name}.json").unlink(missing_ok=True)


def _forward(link: _Link, path: Path, local: Path, remote: str) -> None:
    name = path.stem
    try:
        request = read_json(RunRequest, path)
        if not _NAME.fullmatch(name) or name != request.request:
            raise RecordError(f"request file {path.name} does not name its request")
    except (RecordError, ValueError):
        os.replace(path, local / "rejected" / path.name)
        return
    shas = (request.recipe.vendor, *(request.recipe.code or ()))
    try:
        for sha in shas:
            _drain(sha)
    except (OSError, ValueError, subprocess.SubprocessError, RecordError) as exc:
        result = _infra(request, f"store tree {sha} not drained: {exc}"[:300])
        write_json(replace(result, cls="infra:bridge"), local / "done" / f"{name}.json")
        path.unlink()
        return
    link.run(f"mkdir -p {remote}/new && cat > {remote}/new/.{name}.part && "
             f"mv {remote}/new/.{name}.part {remote}/new/{name}.json", data=path.read_bytes())
    os.replace(path, local / "claimed" / path.name)


def _drain(vendor: str | None) -> None:
    """Make sure a vendor blob is in the worker store; a blob not staged here is already there."""

    if vendor is None:
        return
    from menagerie.pipeline import store

    layout = StoreLayout.from_env()
    if layout.blob(vendor).exists():
        store.drain([vendor], store=layout, data=DataLayout.from_env())

