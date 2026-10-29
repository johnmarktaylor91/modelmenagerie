"""``counts build`` (speed-up lane sp-hubload, idea 1): next steps and status counts off the hub.

On the science-compute worker (``counts build``): ``drive.next_steps`` over the whole roster and
``scripts/data_status.py``'s counts, written as ``next-steps.json`` and ``status-counts.json``.
On the hub (``counts build --worker``): the same through ``run-on`` at the checkout's HEAD; the
two files are pulled back and installed in the work root's ``counts/`` folder.

The tick reads the installed file through ``steps_for`` (behind ``config.json``'s
``counts_file`` switch, default off): a file under ``MAX_AGE_S`` old, built at the hub's
TorchLens pin from a commit the hub checkout holds, with no roster or pipeline-code change since.
Ids whose own files changed between that commit and the hub's working tree (a model record, a
packet of their family, a ledger line naming them) are recomputed in-process from a filtered
load, so the result equals ``drive.next_steps`` on the hub's files. Anything else falls back:
``steps_for`` returns ``None`` and the caller runs ``drive.next_steps``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Collection, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, now

SCHEMA = "counts/1"
NEXT_STEPS, STATUS = "next-steps.json", "status-counts.json"
MAX_AGE_S = 30 * 60  # the tick ignores an older file
INPUTS = ("ledger", "models", "research")  # under the data root; the roster change falls back
WORKER_CPUS, WORKER_GIB = 1, 4
WAIT_S = 20 * 60  # ``--worker`` gives up on a job (left to the reaper) after this long
ADMIT_WAIT_S = 600  # the runners often hold the whole share; a slot frees as lane jobs end
ID = re.compile(rb'"id":\s*"([^"]+)"')


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True,
                          check=False, timeout=120)
    if done.returncode:
        raise OSError(f"git {args[0]}: rc {done.returncode}: {done.stderr.strip()[-200:]}")
    return done.stdout


def _family_key(family: str | None) -> str | None:
    """A family's short hash: the file names no family, so the pull's key-shaped-string scan
    never sees a long hyphenated name."""

    return hashlib.sha256(family.encode()).hexdigest()[:12] if family else None


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _status(data: DataLayout) -> dict[str, Any]:
    """``scripts/data_status.py --json``'s counts for the checkout holding ``data``."""

    script = Path(__file__).resolve().parents[2] / "scripts" / "data_status.py"
    spec = importlib.util.spec_from_file_location("data_status", script)
    if spec is None or spec.loader is None:
        return {"error": f"cannot load {script.name}"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.counts(Path(data.root).resolve().parents[1])
    except (OSError, KeyError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def build(*, data: DataLayout, out: Path | None = None, commit: str | None = None,
          worker: bool = False) -> dict[str, Any]:
    """Write the counts files under ``out``; return their paths and the commit they read.

    ``commit`` names the hub commit the job shipped (the worker's tree is a snapshot, so its
    own HEAD is not the hub's); ``worker`` runs this through ``run-on`` and installs the result.
    """

    if worker:
        return submit(data=data)
    from menagerie.pipeline import drive
    from menagerie.pipeline.publish import torchlens_pin

    out = Path(out or work_dir("counts"))
    steps = drive.next_steps(data=data)
    family = drive.families_of([s["id"] for s in steps], data=data)
    payload = {"schema": SCHEMA, "built_at": now(), "commit": commit, "tl": torchlens_pin(),
               "steps": {s["id"]: [s["step"], _family_key(family[s["id"]])] for s in steps}}
    _write(out / NEXT_STEPS, payload)
    _write(out / STATUS, {"schema": SCHEMA, "built_at": payload["built_at"], "commit": commit,
                          "counts": _status(data)})
    return {"next_steps": str(out / NEXT_STEPS), "status_counts": str(out / STATUS),
            "commit": commit, "rows": len(steps)}


# --- the hub's reader ------------------------------------------------------------------------


def installed(name: str = NEXT_STEPS) -> Path:
    return work_dir("counts") / name


def _fresh(path: Path, max_age_s: float) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        built = datetime.fromisoformat(payload["built_at"].replace("Z", "+00:00"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    age = (datetime.now(timezone.utc) - built).total_seconds()
    ok = payload.get("schema") == SCHEMA and payload.get("commit") and 0 <= age <= max_age_s
    return payload if ok else None


def _members(text: str | None) -> set[str]:
    if not text:
        return set()
    try:
        return {m["id"] for m in json.loads(text).get("members") or ()}
    except (ValueError, TypeError, KeyError, AttributeError):
        return set()


def _packet_ids(repo: Path, rel: str, commit: str, family: str,
                steps: dict[str, list[Any]]) -> set[str]:
    """Ids a packet change may move: its old and new members and ids filed under its family."""

    before = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"],
                            capture_output=True, text=True, check=False, timeout=60)
    path = repo / rel
    after = path.read_text(encoding="utf-8") if path.is_file() else None
    key = _family_key(family)
    return (_members(before.stdout if before.returncode == 0 else None) | _members(after)
            | {i for i, (_, fam) in steps.items() if fam == key})


def _ledger_ids(repo: Path, commit: str, ledger: str, untracked: Sequence[str]) -> set[str]:
    diff = _git(repo, "diff", "-U0", "--no-color", "--no-ext-diff", commit, "--", ledger)
    changed = [line for line in diff.encode().splitlines()
               if line[:1] in (b"+", b"-") and line[:3] not in (b"+++", b"---")]
    for rel in untracked:
        changed += (repo / rel).read_bytes().splitlines()
    return {m.decode() for line in changed for m in ID.findall(line)}  # every id a line names


def _shared(repo: Path) -> list[str]:
    """Inputs every step reads, as paths in ``repo``: the pipeline code, the TorchLens pin and
    the environment specs (``records.known_envs``). A new input of ``drive._step`` goes here or
    among ``touched``'s per-id paths."""

    from menagerie.pipeline.records import ENV_SPECS

    here = Path(__file__).resolve()
    paths = (here.parent, here.parents[2] / "pyproject.toml", Path(ENV_SPECS).resolve())
    return [p.relative_to(repo).as_posix() for p in paths if p.is_relative_to(repo)]


def touched(data: DataLayout, commit: str, steps: dict[str, list[Any]]) -> set[str] | None:
    """Ids whose files differ between ``commit`` and the working tree; ``None``: fall back
    (an unknown commit; a roster, pipeline-code, pin or environment change)."""

    root = Path(data.root).resolve()
    try:
        repo = Path(_git(root, "rev-parse", "--show-toplevel").strip())
        rel = root.relative_to(repo).as_posix()
        diff, new = _changed(repo, commit, [f"{rel}/crawl_roster.jsonl", *_shared(repo)])
        if diff or [n for n in new if "__pycache__/" not in n]:  # the checkout keeps bytecode
            return None
        diff, new = _changed(repo, commit, [f"{rel}/{p}" for p in INPUTS])
        ids = {i for name in diff + new for i in _ids_of(repo, rel, name, commit, steps)}
        ledger_new = [n for n in new if Path(n).relative_to(rel).parts[0] == "ledger"]
        return ids | _ledger_ids(repo, commit, f"{rel}/ledger", ledger_new)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _changed(repo: Path, commit: str, paths: Sequence[str]) -> tuple[list[str], list[str]]:
    """Files under ``paths`` that differ from ``commit`` in the working tree, and untracked ones."""

    return (_git(repo, "diff", "--name-only", "--no-renames", commit, "--", *paths).split(),
            _git(repo, "ls-files", "--others", "--exclude-standard", "--", *paths).split())


def _ids_of(repo: Path, rel: str, name: str, commit: str, steps: dict[str, list[Any]]) -> set[str]:
    """The ids a changed model record or packet may move (ledger shards: ``_ledger_ids``)."""

    part, stem = Path(name).relative_to(rel).parts[0], Path(name).stem
    if part == "models":
        return {stem}
    if part == "research":
        return _packet_ids(repo, name, commit, stem, steps)
    return set()


def steps_for(ids: Sequence[str], *, data: DataLayout, path: Path | None = None,
              max_age_s: float = MAX_AGE_S) -> dict[str, str] | None:
    """Each id's next step from the counts file, corrected for files changed since its commit.

    Returns ``None`` (the caller runs ``drive.next_steps``) when the file is missing, unreadable,
    older than ``max_age_s``, built at another TorchLens pin, or ``touched`` cannot say.
    """

    from menagerie.pipeline import drive
    from menagerie.pipeline.publish import torchlens_pin

    payload = _fresh(path or installed(), max_age_s)
    if payload is None or payload.get("tl") != torchlens_pin():
        return None
    steps = payload.get("steps") or {}
    changed = touched(data, payload["commit"], steps)
    if changed is None:
        return None
    # A cached 2b is recomputed too: a repair ends when its lane writes a result file in scratch
    # (``drive._repair_or``), which no commit shows.
    redo = {i for i in ids if i in changed or i not in steps or steps[i][0] == "2b"}
    files = drive._load(data, ids=redo)
    fresh = {i: drive._step(i, files)[0] for i in redo}
    return {i: fresh[i] if i in fresh else steps[i][0] for i in ids}


# --- the hub's launcher ----------------------------------------------------------------------


def _run(argv: Sequence[str], cwd: Path, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=cwd, capture_output=True, text=True, check=False,
                          timeout=timeout)


def _state(job: str, repo: Path) -> tuple[str, str]:
    """The job's host and state from ``run-on status`` (``<job> <host> <STATE> rc=N``)."""

    line = next((x for x in _run(["run-on", "status", job], repo, 60).stdout.splitlines()
                 if x.startswith(job)), "")
    parts = line.split()
    return (parts[1], parts[2]) if len(parts) > 2 else ("", "")


def _install(pulled: Path, job: str) -> dict[str, str]:
    """Copy the pulled files into ``counts/`` (atomic each), keep the job's copy under
    ``counts/jobs/`` and the three newest such copies."""

    dest = work_dir("counts")
    found = {name: next(iter(sorted(pulled.rglob(name))), None) for name in (NEXT_STEPS, STATUS)}
    if found[NEXT_STEPS] is None:
        raise OSError(f"counts job {job}: no {NEXT_STEPS} pulled under {pulled}")
    out = {}
    for name, src in found.items():
        if src is not None:
            _write(dest / name, json.loads(src.read_text(encoding="utf-8")))
            out[name] = str(dest / name)
    jobs = sorted((dest / "jobs").glob("menagerie-counts-*"))
    for old in jobs[:-3]:
        shutil.rmtree(old)
    return out


def submit(*, data: DataLayout, wait_s: float = WAIT_S) -> dict[str, Any]:
    """Run ``counts build`` on the worker at HEAD; pull and install its files; reap its copy."""

    repo = Path(_git(Path(data.root).resolve(), "rev-parse", "--show-toplevel").strip())
    sha = _git(repo, "rev-parse", "HEAD").strip()
    job = f"menagerie-counts-{datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%Sz')}"
    argv = ["run-on", "submit", "--project", "menagerie", "--os", "linux", "--no-fallback",
            "--mode", "batch", "--ref", sha, "--cpus", str(WORKER_CPUS), "--memory-gib",
            str(WORKER_GIB), "--pull", "outputs/counts/*.json", "--admit-wait", str(ADMIT_WAIT_S),
            "--job", job, "--", "env", f"MENAGERIE_CPUS={WORKER_CPUS}", f"MENAGERIE_MEMORY_GB={WORKER_GIB}",
            "uv", "run", "--no-project", "--python", "3.11", "python", "-m", "menagerie.pipeline",
            "counts", "build", "--out", "outputs/counts", "--commit", sha]  # stdlib, as the runner
    done = _run(argv, repo, ADMIT_WAIT_S + 300)
    text = done.stdout + done.stderr
    if not re.search(rf"^job: +{re.escape(job)}", text, re.MULTILINE):
        # admission said no (``waiting: NO_CAPACITY ...``, ``refused: ...``): nothing ran; the
        # installed file ages out and the tick falls back until a later run is admitted
        why = re.search(r"(?:waiting|refused): [A-Z_]+[^\n]{0,200}", text)
        if why:
            return {"job": job, "commit": sha, "state": "refused", "why": why.group(0), "installed": {}}
        raise OSError(f"counts submit {job}: rc {done.returncode}: {text[-300:]}")
    deadline = time.monotonic() + wait_s
    host, state = "", ""
    while time.monotonic() < deadline and state not in ("DONE", "FAILED"):
        time.sleep(15)
        host, state = _state(job, repo)
    if state != "DONE":
        return {"job": job, "commit": sha, "state": state or "unknown", "installed": {}}
    try:
        return {"job": job, "commit": sha, "state": state, "installed": _fetch(job, host, repo)}
    finally:  # run-on keeps a copy whose outputs were not pulled; the reaper timer settles it
        _run(["run-on", "reap", "--host", host, "--job", job, "--apply"], repo, 300)


def _fetch(job: str, host: str, repo: Path) -> dict[str, str]:
    """Pull the finished job's files to the hub, move them under ``counts/jobs/`` and install."""

    _run(["run-on", "fetch", job], repo, 120)
    keep = work_dir("counts") / "jobs" / job
    keep.parent.mkdir(parents=True, exist_ok=True)
    for argv in (["--pull", "outputs/counts/*.json"], ["--move-to", str(keep)]):
        done = _run(["pull-outputs", host, job, *argv], repo, 300)
        if done.returncode:
            raise OSError(f"pull-outputs {argv[0]} {job}: rc {done.returncode}: "
                          f"{done.stderr.strip()[-300:]}")
    return _install(keep, job)


def plan_equal(ids: Collection[str], *, data: DataLayout, path: Path | None = None) -> dict[str, Any]:
    """The must-hold check: ``steps_for`` against ``drive.next_steps`` on the same files."""

    from menagerie.pipeline import drive

    chosen = list(ids)
    from_file = steps_for(chosen, data=data, path=path)
    direct = {s["id"]: s["step"] for s in drive.next_steps(chosen, data=data)}
    differ = sorted(i for i in chosen if from_file is not None and from_file.get(i) != direct.get(i))
    return {"ids": len(chosen), "file_used": from_file is not None, "differ": differ[:50],
            "equal": from_file is not None and not differ}
