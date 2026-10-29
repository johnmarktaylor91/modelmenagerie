"""Hub side of the mini recipe lanes: stage inputs, answer verbs, pull and ingest results.

Every connection goes from the hub to the worker (the worker's registry ssh alias); the worker
never reaches the hub. For each live mini lease (``mini_placement``) the relay stages the brief
and family folders into the run-on job's checkout (``.mini-lane/``), runs the configured hub
``deliver`` hook if any, and only then writes ``READY``. It answers the verbs the lane queues
in ``.mini-lane/spool/new/``: a request is authorized by ``mini_verbs``, the drafts it uses
come to the hub only through a worker snapshot that ``pull-outputs --dry-run`` leak-scans and
whose hashes the copy must match, and the canonical hub wrapper runs from a trusted empty
folder. Answers are kept on the hub until delivered. When fleet-run reports the run ended the
relay pulls ``outputs/lane/`` with ``pull-outputs`` into the project's ``outputs/<job>/`` and,
while the lease is still this lane's, ingests the drafts and result; the lease is kept until
then (``admission._live``).

Started as a thread of the bridge service (``bridge.serve``); ``python -m
menagerie.pipeline.mini_relay`` runs it alone. One relay serves at a time. Imports neither
torch nor psutil.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, admission, now
from menagerie.pipeline.mini_lane import (
    NOT_DRAFTS,
    OUT,
    ROOT,
    SNAP_MAX_BYTES,
    SNAP_MAX_FILES,
    VERB_TIMEOUT_S,
    expand,
)

POLL_S = 2.0
STATUS_EVERY_S = 10.0
HEARTBEAT_STALE_S = 60
MAX_VERBS = 4  # concurrent verbs per lane
SPOOL_MAX = 256  # request files read per pass
OUTPUT_MAX = 200_000
REQUEST_MAX = 65_536
PULL_ATTEMPTS = 5
PULL_RETRY_S = 60.0
# In the tick folder: touched whenever a lane ends here (ingested, no result, late or pull failed),
# so the refill timer tops up the freed 2b slot before the next full tick (``tick`` refill mode).
REFILL_WANT = "refill.want"
DELIVER_TIMEOUT_S = 300
# run-on's job checkout, relative to the worker's home (legacy run-on, pull-outputs).
WORKTREE = ".orca-run/worktrees/menagerie/{label}"
SAFE_PATH = re.compile(r"/[A-Za-z0-9._/-]+")
NAME = re.compile(r"[0-9]+-[0-9a-f]{8}")
q = shlex.quote


def state_dir(label: str | None = None) -> Path:
    root = work_dir("minis")
    return root / label if label else root


def serving() -> bool:
    beat = state_dir() / "serving"
    return beat.is_file() and time.time() - beat.stat().st_mtime < HEARTBEAT_STALE_S


def ingested(label: str) -> bool:
    return (state_dir(label) / "ingested.json").is_file()


def _log(label: str, text: str) -> None:
    folder = state_dir(label)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "relay.log").open("a", encoding="utf-8") as log:
        log.write(f"{now()} {text}\n")


def _reply(rc: int, stderr: str = "", stdout: str = "") -> dict[str, Any]:
    return {"rc": rc, "stdout": stdout, "stderr": stderr}


# --- the path map ---------------------------------------------------------------------------


def layout(families: Sequence[str], brief: Path) -> tuple[str, str]:
    """The brief's and result's paths under the worker's ``.mini-lane/``."""
    folder = Path(brief).parent
    name = folder.name if folder.parent == work_dir("2b") and folder.name in families else "_batch"
    return f"work/{name}/{Path(brief).name}", f"work/{name}/BRIEF.result.md"


def paths(root: str, families: Sequence[str], brief: Path, run: str) -> dict[str, str]:
    """Hub path to worker path for every folder the lane's text names (``translate``)."""
    mapped = {}
    for family in families:
        worker = f"{root}/work/{family}"
        mapped[str(work_dir("2b") / family)] = worker
        mapped[str(work_dir("lanes") / run / "src" / family)] = worker + "/sources"
        mapped[str(work_dir("lanes") / run / "tries" / family)] = worker + "/tries"
    brief_rel, _ = layout(families, brief)
    mapped.setdefault(str(Path(brief).parent), f"{root}/{Path(brief_rel).parent}")
    return mapped


def translate(text: str, mapping: Mapping[str, str]) -> str:
    for source in sorted(mapping, key=len, reverse=True):
        text = text.replace(source, mapping[source])
    return text


def inverse(mapping: Mapping[str, str]) -> dict[str, str]:
    return {v: k for k, v in mapping.items()}


def keep_adapter_paths(folder: Path, mapping: Mapping[str, str]) -> None:
    """Draft adapter paths written on the worker become relative to the hub draft."""
    back = inverse(mapping)
    for path in folder.rglob("*.json"):
        if path.relative_to(folder).parts[0] in ("sources", "tries") or path.is_symlink():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # The canonical verb reports malformed drafts, unchanged.
        if not isinstance(raw, dict) or raw.get("kind") != "adapter" \
                or not isinstance(raw.get("adapter"), str):
            continue
        adapter = translate(raw["adapter"], back)
        if adapter != raw["adapter"]:
            kept = {**raw, "adapter": os.path.relpath(adapter, path.parent)}
            admission._write_atomic(path, json.dumps(kept) + "\n")


def _copy_drafts(source: Path, dest: Path, only: Sequence[Path] | None = None) -> None:
    for path in (only if only is not None else source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink() or not path.is_file() or relative.parts[0] in NOT_DRAFTS:
            continue
        (dest / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / relative)


def _draft_path(rel: str) -> bool:
    """A scanned draft name the hub will write: relative, inside the folder, not a hub file."""
    parts = Path(rel).parts
    return bool(parts) and SAFE_PATH.fullmatch("/" + rel) is not None and ".." not in parts \
        and parts[0] not in NOT_DRAFTS


def _plan_refusal(plan: Mapping[str, tuple[int, str]]) -> str | None:
    if any(not _draft_path(rel) for rel in plan):
        return "draft file names may use only letters, digits, '.', '_', '-' and '/'"
    if len(plan) > SNAP_MAX_FILES or sum(size for size, _ in plan.values()) > SNAP_MAX_BYTES:
        return "drafts too large for the hub"
    return None


def _write_bytes(path: Path, body: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)


def scan_plan(text: str) -> dict[str, tuple[int, str]]:
    """``pull-outputs --dry-run``'s plan lines: ``  <relpath> <bytes> <sha256[:16]>``."""
    plan = {}
    for line in text.splitlines():
        parts = line.strip().rsplit(None, 2)
        if line.startswith("  ") and len(parts) == 3 and parts[1].isdigit() \
                and re.fullmatch(r"[0-9a-f]{16}", parts[2]):
            plan[parts[0]] = (int(parts[1]), parts[2])
    return plan


# --- the worker connection ------------------------------------------------------------------


class Remote:
    """One worker over hub-to-worker ssh, sharing one ControlMaster connection."""

    def __init__(self, alias: str, control: Path) -> None:
        from menagerie.pipeline.bridge import _Link

        self.alias, self.link, self.lock = alias, _Link(alias, control), threading.Lock()

    def run(self, command: str, *, data: bytes | None = None) -> bytes:
        with self.lock:
            self.link.ensure()
        return self.link.run(command, data=data)

    def target(self, path: str) -> str:
        return f"{self.alias}:{path}"

    def rsync(self, options: Sequence[str], source: str, dest: str) -> None:
        with self.lock:
            self.link.ensure()
        ssh = shlex.join(["ssh", "-S", str(self.link.control), "-o", "BatchMode=yes"])
        subprocess.run(["rsync", "-e", ssh, *options, source, dest], capture_output=True,
                       timeout=600, check=True)

    def close(self) -> None:
        self.link.close()


def _q(path: str) -> str:
    if not SAFE_PATH.fullmatch(path):
        raise ValueError(f"unsafe worker path {path!r}")
    return q(path)


# Staging: the hub's wrapper, result and defer marker stay on the hub.
STAGE = ["-rtl", "--safe-links", "--exclude=/menagerie", "--exclude=/BRIEF.result.md",
         "--exclude=/deferred.json"]


class Lane:
    """The relay's view of one live mini lease. Phases: staged, delivered, ready (or aborted)."""

    def __init__(self, lease: Mapping[str, Any], remote: Remote) -> None:
        self.lease, self.remote = dict(lease), remote
        self.label, self.run, self.host = lease["label"], lease["run"], lease["placement"]
        self.families = list(lease["keys"])
        self.brief = Path(lease["brief"])
        self.expires = admission._parse(lease["expires"])
        self.family_locks = {f: threading.Lock() for f in self.families}
        self.deliver_lock = threading.Lock()  # one writer publishes answers at a time
        self.scan_lock = threading.Lock()  # pull-outputs stages one scan per job at a time
        self.inflight: dict[str, threading.Thread] = {}
        self.task: threading.Thread | None = None
        self.state, self.checked, self.next_pull, self.pulls = "unknown", 0.0, 0.0, 0
        self.scope = None
        self.staged = self._load("staged.json")

    def _load(self, name: str) -> dict[str, Any] | None:
        path = state_dir(self.label) / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _save(self, name: str, value: Mapping[str, Any]) -> None:
        state_dir(self.label).mkdir(parents=True, exist_ok=True)
        admission._write_atomic(state_dir(self.label) / name, json.dumps(value) + "\n")

    def busy(self) -> bool:
        return any(t.is_alive() for t in self.inflight.values()) or bool(
            self.task and self.task.is_alive())

    def step(self) -> None:
        """Start the lane's next piece of work off the relay loop, one at a time."""
        if ingested(self.label) or (self.task and self.task.is_alive()):
            return
        self.task = threading.Thread(target=self._advance, daemon=True)
        self.task.start()

    def _advance(self) -> None:
        try:
            self.inflight = {k: t for k, t in self.inflight.items() if t.is_alive()}
            if self.ended():
                if not self.inflight:
                    self.ingest()
            elif (self.staged or {}).get("phase") == "ready":
                self.poll()
            elif (self.staged or {}).get("phase") != "aborted":
                self.stage()
        except Exception as exc:  # noqa: BLE001 - retried on the next pass; the log keeps why.
            _log(self.label, f"step failed: {exc!r}"[:500])

    def ended(self) -> bool:
        if time.monotonic() - self.checked >= STATUS_EVERY_S:
            status = admission._status(self.run)
            self.state = status["run"]["state"] if status else "unknown"
            self.checked = time.monotonic()
        return self.state == "TERMINAL"

    # staging ---------------------------------------------------------------------------------

    def stage(self) -> None:
        """Inputs, then the login hook, then READY; each phase is kept for a relay restart."""
        if self.staged is None and not self._stage_inputs():
            return
        if self.staged["phase"] == "staged":
            deliver = self.lease.get("mini", {}).get("deliver")
            if deliver and not self._deliver(deliver):
                self._put(f"{self.staged['root']}/ABORT", b"runtime login delivery failed\n")
                self._phase("aborted")
                return
            self._phase("delivered")
        self._put(f"{self.staged['root']}/READY", b"staged\n")
        self._phase("ready")

    def _stage_inputs(self) -> bool:
        try:
            home = self.remote.run("cd " + q(WORKTREE.format(label=self.label)) + " && pwd -P")
        except subprocess.CalledProcessError:
            return False  # The job has not created its checkout yet.
        worktree = home.decode().strip()
        root = worktree + "/" + ROOT
        mapping = paths(root, self.families, self.brief, self.run)
        spool = " ".join(_q(f"{root}/spool/{box}") for box in ("new", "claimed", "done"))
        self.remote.run(f"mkdir -p {spool}")
        for family in self.families:
            self.remote.run(f"mkdir -p {_q(f'{root}/work/{family}')}")
            self.remote.rsync(STAGE, f"{work_dir('2b') / family}/",
                              self.remote.target(f"{root}/work/{family}/"))
        brief_rel, _ = layout(self.families, self.brief)
        self._put(f"{root}/{brief_rel}", translate(self.brief.read_text(encoding="utf-8"),
                                                   mapping).encode())
        # pull-outputs, which scans the drafts each verb brings in, needs the job's hub mirror.
        subprocess.run([admission._tool("run-on"), "fetch", self.label], capture_output=True,
                       text=True, timeout=180, check=False)
        self.staged = {"root": root, "worktree": worktree, "paths": mapping, "phase": "staged",
                       "at": now()}
        self._save("staged.json", self.staged)
        _log(self.label, f"staged {len(self.families)} families at {root}")
        return True

    def _deliver(self, argv: Sequence[str]) -> bool:
        """The configured hub hook that hands the runtime its login; it finishes before READY."""
        values = {"job": self.label, "host": self.host, "account": self.lease["account"]}
        try:
            done = subprocess.run(expand(argv, values), capture_output=True, text=True,
                                  timeout=DELIVER_TIMEOUT_S, check=False, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _log(self.label, f"deliver hook did not finish: {type(exc).__name__}")
            return False
        # Never log the hook's output: it may concern a credential.
        _log(self.label, f"deliver hook rc={done.returncode}")
        return done.returncode == 0

    def _phase(self, phase: str) -> None:
        self.staged = {**self.staged, "phase": phase}
        self._save("staged.json", self.staged)

    def _put(self, path: str, body: bytes) -> None:
        part = _q(f"{path}.{secrets.token_hex(4)}.part")
        self.remote.run(f"mkdir -p {_q(str(Path(path).parent))} && cat > {part} && "
                        f"mv {part} {_q(path)}", data=body)

    # verbs -----------------------------------------------------------------------------------

    def _box(self, box: str) -> list[str]:
        """Request names in one spool folder: regular files with a request name, at most
        ``SPOOL_MAX``; anything else there is ignored."""
        folder = _q(f"{self.staged['root']}/spool/{box}")
        # Only request names count toward the cap, so junk can never hide a real request.
        listed = self.remote.run(f"find {folder} -mindepth 1 -maxdepth 1 -type f -name '*.json' "
                                 f"| grep -E '/[0-9]+-[0-9a-f]{{8}}\\.json$' | head -n {SPOOL_MAX}"
                                 ).decode().splitlines()
        stems = (Path(entry).name[:-5] for entry in listed)
        return sorted(stem for stem in stems if NAME.fullmatch(stem))

    def _spool(self, box: str, name: str) -> str:
        return _q(f"{self.staged['root']}/spool/{box}/{name}.json")

    def _answers(self) -> Path:
        return state_dir(self.label) / "answers"

    def poll(self) -> None:
        self.reconcile()
        self.redeliver()
        for name in self._box("new"):
            if name in self.inflight or len(self.inflight) >= MAX_VERBS:
                continue
            self.remote.run(f"mv {self._spool('new', name)} {self._spool('claimed', name)}")
            if (self._answers() / f"{name}.json").is_file():
                continue  # Already answered; redeliver sends it.
            body = self.remote.run(f"head -c {REQUEST_MAX + 1} {self._spool('claimed', name)}")
            thread = threading.Thread(target=self.answer, args=(name, body), daemon=True)
            self.inflight[name] = thread
            thread.start()

    def reconcile(self) -> None:
        """Claimed verbs this relay is not running (after a restart or a failed read): one that
        never started runs again; one that started is answered as uncertain, never rerun; a
        kept answer is delivered."""
        claimed, done = set(self._box("claimed")), set(self._box("done"))
        for name in sorted(claimed - done - set(self.inflight)):
            if (self._answers() / f"{name}.json").is_file():
                continue
            if (self._answers() / f"{name}.started").is_file():
                self._record(name, _reply(75, "infra: the hub relay restarted during this verb; "
                                              "check its effect (tries, records) before retrying\n"))
            else:
                self.remote.run(f"mv {self._spool('claimed', name)} {self._spool('new', name)}")

    def redeliver(self) -> None:
        with self.deliver_lock:
            answers = sorted(self._answers().glob("*.json")) if self._answers().is_dir() else []
            for path in answers:
                if not path.with_suffix(".delivered").exists():
                    self._put(f"{self.staged['root']}/spool/done/{path.name}", path.read_bytes())
                    self.remote.run(f"rm -f {self._spool('claimed', path.stem)}")
                    path.with_suffix(".delivered").touch()

    def _record(self, name: str, reply: Mapping[str, Any]) -> None:
        mapping = self.staged["paths"]
        kept = {k: translate(v, mapping)[-OUTPUT_MAX:] if isinstance(v, str) else v
                for k, v in reply.items()}
        self._answers().mkdir(parents=True, exist_ok=True)
        admission._write_atomic(self._answers() / f"{name}.json", json.dumps(kept))

    def answer(self, name: str, body: bytes) -> None:
        try:
            reply = (self.serve_verb(name, json.loads(body)) if len(body) <= REQUEST_MAX
                     else _reply(2, "verb request too large\n"))
        except Exception as exc:  # noqa: BLE001 - the lane always gets an answer; the log keeps why.
            _log(self.label, f"verb {name} failed in the relay: {exc!r}"[:500])
            reply = _reply(75, f"infra: hub relay error ({type(exc).__name__})\n")
        self._record(name, reply)
        try:
            self.redeliver()
        except (subprocess.SubprocessError, OSError) as exc:  # retried on the next poll
            _log(self.label, f"verb {name} answer not delivered yet: {exc!r}"[:500])

    def _scope(self) -> Any:
        from menagerie.pipeline import mini_verbs

        if self.scope is None:
            self.scope = mini_verbs.scope_for(self.families, self.brief, DataLayout.from_env())
        return self.scope

    def serve_verb(self, name: str, request: Mapping[str, Any]) -> dict[str, Any]:
        """Authorize, bring the drafts in, run the canonical hub verb, send its evidence back."""
        from menagerie.pipeline import mini_verbs

        if not self._owned():
            return _reply(2, "this lane is past its lease window; finish with the result file\n")
        family, args = request.get("family"), request.get("args")
        if request.get("id") != name or family not in self.families or not isinstance(args, list) \
                or not args or not all(isinstance(a, str) and "\0" not in a for a in args):
            return _reply(2, "malformed verb request\n")
        mapping, scope = self.staged["paths"], self._scope()
        hub = [translate(a, inverse(mapping)) for a in args]
        try:
            plan = mini_verbs.authorize(hub, scope, mini_verbs.base_folder(
                request.get("cwd"), family, scope, mapping))
        except ValueError as exc:
            return _reply(2, f"{exc}\n")
        with self.family_locks[plan.family]:
            reply = self._locked(name, plan, scope)
        _log(self.label, f"verb {name} {plan.verb} {plan.family} rc={reply['rc']}")
        return reply

    def _locked(self, name: str, plan: Any, scope: Any) -> dict[str, Any]:
        """Under the family's lock: drafts in, checks, the verb, its kept outcome, evidence."""
        from menagerie.pipeline import mini_verbs

        refused = self.bring_in(name, plan.family) if self._owned() else "past the lease window"
        if refused:
            return _reply(2, refused + "\n")
        try:
            mini_verbs.check_recipes(plan, scope)
        except ValueError as exc:
            return _reply(2, f"{exc}\n")
        if not self._owned():
            return _reply(2, "this lane is past its lease window; finish with the result file\n")
        reply = self._run(name, plan)
        self._record(name, reply)  # the outcome is kept even if the evidence push fails
        try:
            self.push_evidence(plan.family)
        except (subprocess.SubprocessError, OSError) as exc:
            _log(self.label, f"verb {name} evidence not pushed: {exc!r}"[:500])
        return reply

    def _owned(self) -> bool:
        """The lease is still this lane's and a verb started now ends before it expires, so no
        replacement lane can exist while it runs (``admission._live`` waits for ingest)."""
        path = work_dir("leases") / f"{self.label}.json"
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        window = datetime.now(timezone.utc) + timedelta(seconds=VERB_TIMEOUT_S)
        return current.get("run") == self.run and window < self.expires

    def _run(self, name: str, plan: Any) -> dict[str, Any]:
        """The canonical wrapper from a trusted empty folder, so no draft can shadow an import."""
        cwd = state_dir(self.label) / "cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k != "FLEET_RUN_ID"}
        env.update(MENAGERIE_LANE=self.run, PYTHONSAFEPATH="1")
        self._answers().mkdir(parents=True, exist_ok=True)
        (self._answers() / f"{name}.started").touch()
        with subprocess.Popen([str(work_dir("2b") / plan.family / "menagerie"), *plan.argv],
                              cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              env=env, stdin=subprocess.DEVNULL, start_new_session=True) as child:
            try:
                stdout, stderr = child.communicate(timeout=VERB_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)  # the verb and everything it started
                child.communicate()
                return _reply(124, "infra: verb timed out on the hub\n")
        return _reply(child.returncode, stderr, stdout)

    def bring_in(self, name: str, family: str) -> str | None:
        """The family's drafts, through a worker snapshot the leak scan passed; or a refusal."""
        root, worktree = self.staged["root"], self.staged["worktree"]
        summary = json.loads(self.remote.run(
            f"cd {_q(worktree)} && python3 menagerie/pipeline/mini_lane.py snapshot "
            f"{_q(root)} {q(name)} {q(family)}"))
        if summary.get("refused") or not summary.get("files"):
            return summary.get("refused")
        try:
            return self._scanned_copy(name, family)
        finally:
            self.remote.run(f"rm -rf {_q(f'{root}/snap/{name}')}")

    def _scanned_copy(self, name: str, family: str) -> str | None:
        """Read exactly the files the scan listed, each at most its scanned size, keep them only
        when every hash matches what was scanned, and promote them only while the lease is this
        lane's. Nothing unscanned is written on the hub."""
        snap = f"{ROOT}/snap/{name}/{family}/"
        with self.scan_lock:
            scan = subprocess.run([admission._tool("pull-outputs"), self.host, self.label,
                                   "--pull", f"{ROOT}/snap/{name}/**", "--dry-run"],
                                  capture_output=True, text=True, timeout=300, check=False)
        if scan.returncode == 3:
            return "the leak scan blocked these drafts: remove the secret-shaped text and retry"
        if scan.returncode:
            raise OSError(f"draft scan failed rc={scan.returncode}")
        plan = {rel[len(snap):]: v for rel, v in scan_plan(scan.stdout).items()
                if rel.startswith(snap)}
        refused = _plan_refusal(plan)
        if refused:
            return refused
        got, base = {}, f"{self.staged['root']}/snap/{name}/{family}"
        for rel, (size, sha) in plan.items():
            body = self.remote.run(f"head -c {size + 1} {_q(f'{base}/{rel}')}")
            if len(body) != size or hashlib.sha256(body).hexdigest()[:16] != sha:
                return "drafts changed while the hub scanned them; retry"
            got[rel] = body
        return self._promote(family, got)

    def _promote(self, family: str, files: Mapping[str, bytes]) -> str | None:
        folder = work_dir("2b") / family
        real = os.path.realpath(folder)
        with admission._lease_lock():
            if not self._owned():
                return "this lane is past its lease window; finish with the result file"
            for rel, body in files.items():
                target = folder / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                parent = os.path.realpath(target.parent)
                if target.is_symlink() or not (parent == real or parent.startswith(real + "/")):
                    continue  # never write through a link out of the family's folder
                _write_bytes(target, body)
        keep_adapter_paths(folder, self.staged["paths"])
        return None

    def push_evidence(self, family: str) -> None:
        """The attempts and unpacked sources a verb wrote; drafts on the worker are untouched."""
        worker = f"{self.staged['root']}/work/{family}"
        lanes = work_dir("lanes") / self.run
        for kind, name, options in (("tries", "tries", ["-rt"]), ("src", "sources", ["-rtL"])):
            folder = lanes / kind / family
            if folder.is_dir():
                self.remote.rsync(options, f"{folder}/", self.remote.target(f"{worker}/{name}/"))

    # ingest ----------------------------------------------------------------------------------

    def ingest(self) -> None:
        """Pull ``outputs/lane/`` through pull-outputs, then keep its drafts and result."""
        if time.monotonic() < self.next_pull:
            return
        self.pulls += 1
        subprocess.run([admission._tool("run-on"), "fetch", self.label], capture_output=True,
                       text=True, timeout=180, check=False)
        pulled = subprocess.run([admission._tool("pull-outputs"), self.host, self.label],
                                capture_output=True, text=True, timeout=900, check=False)
        found = re.search(r"dest=(\S+)", pulled.stdout + pulled.stderr)
        lane = _pulled_lane(Path(found[1])) if found and pulled.returncode == 0 else None
        if lane is None:
            _log(self.label, f"pull {self.pulls} failed rc={pulled.returncode}")
            if self.pulls < PULL_ATTEMPTS:
                self.next_pull = time.monotonic() + PULL_RETRY_S
                return
            self._ended({"status": "pull-failed", "at": now()})
            return
        self._ended({"status": self.keep(lane), "pulled": str(lane), "at": now()})
        _log(self.label, f"ingested {lane}")

    def _ended(self, record: Mapping[str, Any]) -> None:
        """Record how the lane ended, then ask for a refill of its slot (``REFILL_WANT``)."""
        self._save("ingested.json", record)
        want = work_dir("tick") / REFILL_WANT
        want.parent.mkdir(parents=True, exist_ok=True)
        want.touch()

    def keep(self, lane: Path) -> str:
        """Drafts and result into the canonical folders, only while the lease is this lane's."""
        mapping = (self.staged or {}).get("paths", {})
        with admission._lease_lock() as leases:
            path = leases / f"{self.label}.json"
            current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if current.get("run") != self.run:
                return "late: the lease was released; results stay in the pulled folder"
            for family in self.families:
                source = lane / "families" / family
                if source.is_dir():
                    _copy_drafts(source, work_dir("2b") / family)
                    keep_adapter_paths(work_dir("2b") / family, mapping)
            result = lane / "result.md"
            if not result.is_file() or not result.stat().st_size:
                return "no-result"
            text = translate(result.read_text(encoding="utf-8"), inverse(mapping))
            admission._write_atomic(Path(self.lease["result"]), text)
        return "ingested"


def _pulled_lane(dest: Path) -> Path | None:
    for candidate in (dest / OUT, dest / Path(OUT).name):
        if (candidate / "lane.json").is_file():
            return candidate
    return None


# --- the loop -------------------------------------------------------------------------------


def mini_leases() -> list[dict[str, Any]]:
    """Live mini leases with a started run, read without the lease lock (read-only)."""
    found = []
    for path in sorted(work_dir("leases").glob("*.json")):
        try:
            lease = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if lease.get("mini") and lease.get("placement") and \
                not str(lease.get("run", "")).startswith("pending:"):
            found.append(lease)
    return found


class Relay:
    def __init__(self, connect: Callable[[str, Path], Remote] = Remote) -> None:
        self.connect, self.remotes, self.lanes = connect, {}, {}

    def remote(self, alias: str) -> Remote:
        if alias not in self.remotes:
            state_dir().mkdir(parents=True, exist_ok=True)
            self.remotes[alias] = self.connect(alias, state_dir() / f".ssh-{alias}")
        return self.remotes[alias]

    def once(self) -> None:
        state_dir().mkdir(parents=True, exist_ok=True)
        (state_dir() / "serving").touch()
        live = {lease["label"]: lease for lease in mini_leases()}
        for label, lease in live.items():
            try:
                if label not in self.lanes:
                    self.lanes[label] = Lane(lease, self.remote(lease["mini"]["ssh_alias"]))
                self.lanes[label].step()
            except Exception as exc:  # noqa: BLE001 - one lane's failure never stops the others.
                _log(label, f"step failed: {exc!r}"[:500])
        for label in [k for k, lane in self.lanes.items() if k not in live and not lane.busy()]:
            del self.lanes[label]

    def close(self) -> None:
        for remote in self.remotes.values():
            remote.close()


_serving: Relay | None = None  # the relay this process runs, for ``idle``


def idle() -> bool:
    """No lane of this process's relay has a verb or a step in progress (true when none runs)."""
    relay = _serving
    return relay is None or not any(lane.busy() for lane in list(relay.lanes.values()))


def serve(stop: threading.Event | None = None) -> None:
    """Run the relay until ``stop`` is set; a failed pass is logged and retried.

    One relay serves at a time (a lock in the state folder): a second one returns at once, so
    two relays never claim, or answer as uncertain, each other's verbs.
    """
    stop = stop or threading.Event()
    state_dir().mkdir(parents=True, exist_ok=True)
    lock = (state_dir() / "relay.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{now()} mini relay: another relay is serving", file=sys.stderr, flush=True)
        lock.close()
        return
    global _serving  # one relay per process (the lock above)
    relay = _serving = Relay()
    try:
        while not stop.is_set():
            try:
                relay.once()
            except Exception as exc:  # noqa: BLE001 - the relay outlives any one pass's failure.
                print(f"{now()} mini relay: {exc!r}", file=sys.stderr, flush=True)
            stop.wait(POLL_S)
    finally:
        relay.close()
        lock.close()


if __name__ == "__main__":
    serve()
