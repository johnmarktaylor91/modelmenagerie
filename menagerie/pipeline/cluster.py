"""AXON batches for the rows the worker cannot hold: giants and CUDA-only kernels.

Owned by the giants lane (rung 2, fix 5). A batch is one SLURM task: it builds the runner's
environment on the cluster from the committed lock (a static uv and uv-managed Python, which run
on the cluster's glibc 2.17), then runs each row's full attempt, and its rows come back as ledger
lines. Kinds (``KINDS``): ``cpu`` (8 CPUs, 64 GiB, 2 h), ``gpu`` (the same plus one GPU and the
toolchain in ``envs/axon/gpu.sh`` for building CUDA extensions) and ``reviewed`` (128 GiB, up
to four models, run one after another). The verbs:

- ``axon_batch`` (hub) writes ``work_dir("axon")/<batch>/``: ``batch.json``, a minimal data root
  (the rows' model records and adapters), the vendor snapshots the rows import (``store/``) and
  ``job.sh``, the script ``axon-submit`` runs. Nothing is submitted.
- ``submit`` (hub) copies the code at its commit, the pinned uv and the batch to the cluster and
  submits ``job.sh`` through ``axon-submit`` with the kind's allocation.
- ``run`` (inside the job) runs each row: forward, then the full trace at the job's whole
  declaration (a train trace on a structural divergence; the worker made the structure-only one),
  through the runner with no bubblewrap (the cluster has no user namespaces, so the allocation
  is the boundary, ``MENAGERIE_SANDBOX=none``). No renders: the cluster has no Graphviz, so the
  worker renders from the stored saves (``flow --rerender-from`` over the collected results). A
  row is started only while the allocation has ``ROW_MIN_S`` left; the rest are "not reached".
- ``collect`` (hub) brings a finished job back: its ledger and store-index lines are appended on
  main (lines already there are skipped), its blobs are relayed to the worker store
  (``MENAGERIE_STORE_REMOTE``) and verified there, and its attempt results are kept locally.

The hub side imports neither torch nor psutil. Machine names stay out of the repo: the cluster is
the ssh alias ``MENAGERIE_AXON_HOST`` (default ``axon``) and rows say ``host: cluster``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, StoreLayout, now, read_json, write_json, written
from menagerie.pipeline.records import ModelRecord, RecordError, RenderRow

BatchKind = Literal["cpu", "gpu", "reviewed"]


@dataclass(frozen=True)
class Allocation:
    """One batch kind's SLURM allocation and the placements its rows may carry."""

    cpus: int
    memory_gib: int
    minutes: int
    gpu: bool
    placements: tuple[str, ...]
    max_rows: int | None = None
    reviewed: str | None = None  # the registry's reviewed allocation (axon-submit --reviewed)


KINDS: dict[str, Allocation] = {
    "cpu": Allocation(8, 64, 120, False, ("axon-cpu",)),
    # The GPU window's rows fall back here when the worker's card refuses their kernels.
    "gpu": Allocation(8, 64, 120, True, ("axon-gpu", "gpu-window")),
    # JMT 2026-10-03 (decisions.md): one reviewed 128 GiB allocation per batch of giants; the
    # rows run one after another (`run`), never two at once (Crawler 2026-10-28: up to four).
    "reviewed": Allocation(8, 128, 120, False, ("axon-reviewed",), max_rows=4,
                           reviewed="giants-128g"),
}
HEADROOM_GIB = 4  # the runner's cap sits under SLURM's limit, so a giant ends `memory_cap`
ROW_MIN_S = 900  # a row starts only with this much of the allocation left
END_MARGIN_S = 300  # each attempt's timeout leaves this much for the batch to finish
FORWARD_TIMEOUT_S = 1200
TRACE_TIMEOUT_S = 3600
SPECS = Path(__file__).resolve().parents[2] / "envs" / "axon"
CODE_ROOT = Path(__file__).resolve().parents[2]
# The worker card is a GTX 2080 Ti; the partition's GTX 2080 nodes add nothing for its refusals.
GPU_EXCLUDE = "ax01,ax02"
MIME = {"tlspec": "application/x-tlspec", "tlspec-portable": "application/x-tlspec",
        "summary": "application/json"}


def host() -> str:
    """The ssh alias of the cluster's login node."""

    return os.environ.get("MENAGERIE_AXON_HOST", "axon")


def remote_root(home: str) -> str:
    """The project's folder on the cluster (``MENAGERIE_AXON_ROOT``, under the remote home)."""

    rel = os.environ.get("MENAGERIE_AXON_ROOT", "projects/menagerie")
    return rel if rel.startswith("/") else f"{home.rstrip('/')}/{rel}"


def _ssh(command: str, *, timeout: int = 60) -> str:
    """One light command on the login node (never compute there)."""

    done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host(),
                           command], capture_output=True, text=True, timeout=timeout)
    if done.returncode:
        raise RuntimeError(f"ssh {host()} failed ({done.returncode}): {done.stderr.strip()[:300]}")
    return done.stdout


def _remote_home() -> str:
    return _ssh('printf %s "$HOME"').strip()


def _commit() -> str:
    """The code commit a batch runs: the last commit that touched the code (its archive is HEAD's
    code, and data commits do not push another 80 MB tree to the cluster); the checkout must be
    clean under the code."""

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(CODE_ROOT), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "menagerie", "envs",
                ":(exclude)menagerie/data")
    if dirty:
        raise RecordError(f"the code under {CODE_ROOT} has uncommitted changes; the cluster runs "
                          "a commit:\n" + dirty[:500])
    return git("log", "-1", "--format=%H", "--", "menagerie", "envs", ":(exclude)menagerie/data")


# --- writing a batch -----------------------------------------------------------------------


def _records(kind: str, ids: Sequence[str], data: DataLayout, *,
             assign: bool = False) -> list[ModelRecord]:
    """The batch's records; ``assign`` gives each the kind's placement in the batch copy only (the
    feeder's rows keep their records on main as they are)."""

    alloc = KINDS[kind]
    if not ids:
        raise RecordError("a batch needs at least one stable id")
    if alloc.max_rows is not None and len(ids) > alloc.max_rows:
        raise RecordError(f"a {kind} batch holds at most {alloc.max_rows} rows; got {len(ids)}")
    records = []
    capped = _cluster_capped(data) if kind == "reviewed" else set()
    for sid in dict.fromkeys(ids):
        record = read_json(ModelRecord, data.model(sid))
        if assign and record.recipe is not None:
            record = replace(record, recipe=replace(record.recipe, placement=alloc.placements[0]))
        _check_placement(record, kind, escalated=sid in capped)
        records.append(record)
    envs = {r.recipe.env for r in records if r.recipe}
    if len(envs) != 1:
        raise RecordError(f"a batch builds one environment; these rows need {sorted(envs)}")
    return records


def _check_placement(record: ModelRecord, kind: str, *, escalated: bool) -> None:
    alloc = KINDS[kind]
    placement = record.recipe.placement if record.recipe else None
    if placement == "axon-cpu" and escalated:
        placement = "axon-reviewed"  # its cpu batch ran out of memory: escalated
    if placement not in alloc.placements:
        raise RecordError(f"{record.id}: placement {placement!r} is not one a {kind} batch runs "
                          f"({', '.join(alloc.placements)})")


def _cluster_capped(data: DataLayout) -> set[str]:
    """Rows whose full trace on the cluster ended ``memory_cap``: a reviewed batch may take them."""

    from menagerie.pipeline import ledger

    return {row.id for row in ledger.read_all(data=data)
            if getattr(row, "host", None) == "cluster" and getattr(row, "capture", None) == "full"
            and (getattr(row, "cls", None) or "").startswith("memory_cap")}


def _copy_inputs(records: Sequence[ModelRecord], data: DataLayout, batch: Path) -> list[str]:
    """Copy records and adapters into the batch's data root; return the store shas needed (each
    recipe's vendor snapshot and extra code trees)."""

    target = DataLayout(batch / "data")
    vendors = []
    for record in records:
        write_json(record, target.model(record.id))
        recipe = record.recipe
        if recipe is not None and recipe.adapter and recipe.kind == "adapter":
            dest = target.root / recipe.adapter
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data.root / recipe.adapter, dest)
        if recipe is not None and recipe.vendor:
            vendors.append(recipe.vendor)
        if recipe is not None and recipe.code:
            vendors.extend(recipe.code)
    return sorted(set(vendors))


def _stage_vendors(shas: Sequence[str], batch: Path) -> None:
    """Put each vendor snapshot into the batch store, from this store or the worker store."""

    from menagerie.pipeline import store as store_mod

    if not shas:
        return
    here = StoreLayout.from_env()
    staged = StoreLayout(batch / "store")
    for sha in shas:
        try:
            local = store_mod.get(sha, store=here)
        except FileNotFoundError:
            local = None
        if local is not None and local.is_file():
            staged.blob(sha).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, staged.blob(sha))
            continue
        remote = os.environ.get("MENAGERIE_STORE_REMOTE")
        if not remote:
            raise RecordError(f"vendor {sha} is not in {here.root} and MENAGERIE_STORE_REMOTE "
                              "is not set")
        store_mod.restore(sha, store=staged, source=remote)


def _build_line(env: str) -> str:
    """Build ``env`` from its committed lock on the cluster, once per lock (``envs/axon/build.sh``:
    the renderer's binary-only packages, ``NON_TRACE_ROOTS``, are left out, and the few locked
    packages with no glibc 2.17 wheel build from their locked sdists)."""

    from menagerie.pipeline import NON_TRACE_ROOTS, envs

    target = f"$MENAGERIE_ENVS/{envs.prefix(env, root=Path('.')).name}"
    return (f"bash envs/axon/build.sh {shlex.quote(env)} {target} {envs.lock_id(env)} "
            f"{' '.join(NON_TRACE_ROOTS)}")


def job_script(*, batch: str, kind: str, env: str, commit: str, root: str) -> str:
    """The sbatch body: build the environment, run the canary, run the batch, write DONE.

    ``axon-submit`` sets the account, partition, CPUs, memory, time, GPU and the fleet comment on
    the command line; the script sets only its name, log and (gpu) the nodes it avoids.
    """

    alloc = KINDS[kind]
    declared = alloc.memory_gib - HEADROOM_GIB
    q = shlex.quote
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=menagerie-{batch}",
        f"#SBATCH --output={root}/logs/{batch}-%A.log",
        f"#SBATCH --error={root}/logs/{batch}-%A.log",
        *([f"#SBATCH --exclude={GPU_EXCLUDE}"] if alloc.gpu else []),
        "# Written by `menagerie-pipeline axon batch`; submitted by `axon submit` through",
        "# axon-submit. Builds the environment from the committed lock, then runs the batch.",
        "set -euo pipefail",
        f"ROOT={q(root)}; BATCH=$ROOT/batches/{batch}; CODE=$ROOT/code/{commit}",
        "trap 'echo \"rc=$? $(date -u +%FT%TZ)\" > \"$BATCH/DONE\"' EXIT",
        "mkdir -p \"$ROOT/home\" \"$ROOT/cache\"",
        "export PATH=$ROOT/bin:/usr/bin:/bin HOME=$ROOT/home",
        "export UV_CACHE_DIR=$ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$ROOT/python UV_NO_CONFIG=1",
        "export MENAGERIE_ENVS=$ROOT/envs MENAGERIE_DATA=$BATCH/data MENAGERIE_STORE=$BATCH/store",
        "export MENAGERIE_WORK_ROOT=$BATCH/work MENAGERIE_HOST=cluster MENAGERIE_SANDBOX=none",
        "export MENAGERIE_STORE_PACK=0",  # the relay moves and verifies raw trees; the trim packs them
        f"export MENAGERIE_CPUS={alloc.cpus} MENAGERIE_MEMORY_GB={declared}",
        f"export MENAGERIE_DEADLINE=$(( $(date +%s) + {alloc.minutes * 60} ))",
        *(["source \"$CODE/envs/axon/gpu.sh\""] if alloc.gpu else []),
        "cd \"$CODE\"",
        "echo \"$(date -u +%FT%TZ) $(hostname -s) $(ldd --version | head -1) $(uv --version)\"",
        "BOOT=(uv run --no-project --python 3.11 python -m menagerie.pipeline)",
        _build_line(env),
        f"\"${{BOOT[@]}}\" envs canary {q(env)} --root \"$MENAGERIE_ENVS\"",
        f"PY=$(uv run --no-project --python 3.11 python -c \"from pathlib import Path; "
        f"from menagerie.pipeline import envs; print(envs.python({env!r}, "
        f"root=Path('$MENAGERIE_ENVS')))\")",
        "PYTHONPATH=$CODE \"$PY\" -m menagerie.pipeline axon run \"$BATCH\"",
    ]
    return "\n".join(lines) + "\n"


def axon_batch(kind: BatchKind, ids: Sequence[str], *, data: DataLayout,
               out: Path | None = None, home: str | None = None, assign: bool = False) -> Path:
    """Write one AXON batch for the given rows.

    Inputs: the batch kind, stable ids whose recipes carry a placement the kind runs (one
    environment per batch), the output folder (default: ``work_dir("axon")/<batch>``) and the
    cluster home (default: asked over ssh); ``assign`` places every row on the kind (the feeder).
    Returns: the batch script's path. Side effects: the
    batch folder with ``batch.json``, the data root, the vendor snapshots and ``job.sh``;
    nothing is submitted.
    """

    if kind not in KINDS:
        raise RecordError(f"kind {kind!r} is not one of {sorted(KINDS)}")
    records = _records(kind, ids, data, assign=assign)
    env = records[0].recipe.env  # type: ignore[union-attr]
    commit = _commit()
    stamp = now().replace("-", "").replace(":", "").lower()
    name = f"{kind}-{stamp}-{hashlib.sha256(' '.join(ids).encode()).hexdigest()[:4]}"
    batch = Path(out) if out is not None else work_dir("axon") / name
    batch.mkdir(parents=True, exist_ok=True)
    _stage_vendors(_copy_inputs(records, data, batch), batch)
    root = remote_root(home if home is not None else _remote_home())
    script = batch / "job.sh"
    script.write_text(job_script(batch=batch.name, kind=kind, env=env, commit=commit, root=root),
                      encoding="utf-8")
    meta = {"batch": batch.name, "kind": kind, "ids": [r.id for r in records], "env": env,
            "commit": commit, "root": root, "created": now(), "allocation": asdict(KINDS[kind]),
            "revisions": {r.id: r.recipe.rev for r in records if r.recipe}}
    (batch / "batch.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return script


# --- submitting ----------------------------------------------------------------------------


def _uv_binary() -> Path:
    """The pinned static uv (``envs/axon/uv.txt``: version and release sha256), fetched once."""

    version, digest = (SPECS / "uv.txt").read_text().split()[:2]
    target = work_dir("axon") / "bin" / f"uv-{version}"
    if target.is_file():
        return target
    url = (f"https://github.com/astral-sh/uv/releases/download/{version}/"
           "uv-x86_64-unknown-linux-musl.tar.gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    tarball = target.with_suffix(".tar.gz")
    subprocess.run(["curl", "-fsSL", "-o", str(tarball), url], check=True, timeout=300)
    actual = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if actual != digest:
        tarball.unlink()
        raise RecordError(f"uv {version} release hashes to {actual}, not the pinned {digest}")
    member = "uv-x86_64-unknown-linux-musl/uv"
    with target.open("wb") as handle:
        subprocess.run(["tar", "-xzOf", str(tarball), member], stdout=handle, check=True)
    target.chmod(0o755)
    tarball.unlink()
    return target


def _push_code(commit: str, root: str) -> None:
    """Unpack the commit's code (no data) under ``code/<commit>`` on the cluster, once."""

    code = f"{root}/code/{commit}"
    probe = _ssh(f"test -f {shlex.quote(code)}/.complete && echo yes || true").strip()
    if probe == "yes":
        return
    archive = subprocess.Popen(
        ["git", "-C", str(CODE_ROOT), "archive", commit, "--", "menagerie", "envs",
         ":(exclude)menagerie/data"], stdout=subprocess.PIPE)
    unpack = f"mkdir -p {shlex.quote(code)} && tar -x -C {shlex.quote(code)} && touch " \
             f"{shlex.quote(code)}/.complete"
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", host(), unpack], stdin=archive.stdout,
                          capture_output=True, text=True, timeout=1800)
    archive.wait()
    if done.returncode or archive.returncode:
        raise RuntimeError(f"code {commit[:12]} did not unpack on {host()}: {done.stderr[:300]}")


def _rsync(src: str, dest: str, *extra: str) -> None:
    subprocess.run(["rsync", "-a", *extra, src, dest], check=True, capture_output=True,
                   text=True, timeout=3600)


def submit(batch: Path, *, dry_run: bool = False) -> str:
    """Copy a written batch to the cluster and submit it through ``axon-submit``.

    Inputs: the batch folder (``axon_batch``'s output). ``dry_run`` copies everything and prints
    the admission plan without submitting. Returns: the SLURM job id (or the plan). Side
    effects: code, uv and the batch on the cluster; one SLURM task; ``submitted`` in
    ``batch.json``.
    """

    batch = Path(batch)
    meta = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    root, alloc = meta["root"], KINDS[meta["kind"]]
    _ssh(f"mkdir -p {shlex.quote(root)}/bin {shlex.quote(root)}/logs "
         f"{shlex.quote(root)}/batches")
    _rsync(str(_uv_binary()), f"{host()}:{root}/bin/uv")
    _push_code(meta["commit"], root)
    _rsync(f"{batch}/", f"{host()}:{root}/batches/{batch.name}/", "--exclude=back/")
    from menagerie.pipeline.admission import _tool

    command = [os.environ.get("MENAGERIE_AXON_SUBMIT") or _tool("axon-submit"), "--project",
               "menagerie",
               "--script", f"{root}/batches/{batch.name}/job.sh", "--cpus", str(alloc.cpus),
               "--memory-gib", str(alloc.memory_gib), "--minutes", str(alloc.minutes),
               *(["--gpu"] if alloc.gpu else []),
               *(["--reviewed", alloc.reviewed] if alloc.reviewed else []),
               *([] if dry_run else ["--submit"])]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if done.returncode:
        raise RuntimeError(f"axon-submit refused: {(done.stderr or done.stdout).strip()[:400]}")
    answer = done.stdout.strip()
    if not dry_run:
        meta["submitted"] = {"at": now(), "job": answer}
        (batch / "batch.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return answer


# --- inside the job ------------------------------------------------------------------------


def _left_s() -> float:
    deadline = os.environ.get("MENAGERIE_DEADLINE")
    return float(deadline) - time.time() if deadline else float("inf")


def _caps(mode: str) -> Any:
    from menagerie import host_budget
    from menagerie.pipeline.records import Caps

    ceiling = FORWARD_TIMEOUT_S if mode == "forward" else TRACE_TIMEOUT_S  # gpu: both
    left = _left_s() - END_MARGIN_S
    return Caps(rss_mb=int(host_budget.declared_memory_gb() * 1024),
                timeout_s=int(max(60, min(ceiling, left))), threads=host_budget.declared_cpus())


def _attempt(record: ModelRecord, mode: str, outcome: Any, *, data: DataLayout,
             store: StoreLayout, attempts: Path, **kw: Any) -> Any:
    """One attempt through the runner; its rows join the outcome and its saves the store."""

    from menagerie.pipeline import flow, runner
    from menagerie.pipeline import store as store_mod

    request = runner.request_for(record, mode, caps=_caps(mode),  # type: ignore[arg-type]
                                 adapter_src=flow.adapter_src(record, data), **kw)
    result = runner.attempt(request, attempt_dir=attempts / request.request)
    outcome.results.append(result)
    outcome.rows.extend(result.rows)
    for item in result.files:
        path = attempts / request.request / item.path
        if item.kind in MIME and path.exists():
            line = store_mod.put(path, store=store, data=data, mime=MIME[item.kind],
                                 license_class="derived")
            outcome.stored[f"{item.kind}:{item.path}"] = line.sha256
    return result


def run_row(record: ModelRecord, *, data: DataLayout, store: StoreLayout, attempts: Path,
            gpu: bool = False) -> Any:
    """A row's full attempt on the cluster: forward, then the full trace (and a train trace on
    a structural divergence); in a gpu batch, one ``gpu`` attempt (forward, then trace)."""

    from menagerie.pipeline import flow

    # The worker gives an off-worker placement a structure-only capture; here is its full attempt.
    here = replace(record, recipe=replace(record.recipe, placement=None))  # type: ignore[arg-type]
    outcome = flow.Outcome(record.id)
    go = lambda mode, **kw: _attempt(here, mode, outcome, data=data, store=store,  # noqa: E731
                                     attempts=attempts, **kw)
    if gpu:
        go("gpu")
        return outcome
    # No structure-only capture here: the worker made it already, at once (an off-worker row).
    forward = go("forward")
    trace = None if (forward.cls or "").startswith("memory_cap") else go("trace")
    if forward.modes and forward.modes.divergence == "structural" and trace is not None \
            and trace.outcome == "pass":
        go("trace", model_mode="train")
    return outcome


def run(batch: Path) -> list[dict[str, Any]]:
    """Run a batch inside its SLURM job; append its rows to the batch's own ledger.

    Returns: one table row per id (``not_reached`` for rows the allocation had no time for).
    Side effects: attempts under ``work/attempts``, ledger and store-index lines and blobs in
    the batch's data root and store, ``table.jsonl``.
    """

    from menagerie.pipeline import flow, ledger

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("`axon run` runs inside its SLURM job only")
    batch = Path(batch)
    meta = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    data, store = DataLayout(batch / "data"), StoreLayout(batch / "store")
    attempts = batch / "work" / "attempts"
    table = []
    for sid in meta["ids"]:
        record = read_json(ModelRecord, data.model(sid))
        if _left_s() < ROW_MIN_S:
            row: dict[str, Any] = {"stable_id": sid, "name": record.name, "trace": "not_reached"}
        else:
            outcome = run_row(record, data=data, store=store, attempts=attempts,
                              gpu=KINDS[meta["kind"]].gpu)
            # A trace no save level accepts is drawn in process; here that row is only "no
            # renderer" (no Graphviz), so it is dropped and the worker renders from the store.
            rows = [r for r in outcome.rows if not isinstance(r, RenderRow)]
            if rows:
                ledger.append(rows, data=data)
            row = flow.table_row(record, outcome)
        table.append(row)
        with (batch / "table.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)
    return table


# --- collecting ----------------------------------------------------------------------------


def _append_new(src: Path, target: Path) -> int:
    """Append ``src``'s lines that ``target`` lacks; return how many."""

    have = set(target.read_text(encoding="utf-8").splitlines()) if target.exists() else set()
    new = [line for line in src.read_text(encoding="utf-8").splitlines()
           if line and line not in have]
    if new:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write("".join(line + "\n" for line in new))
        written.appended(target, [line + "\n" for line in new])  # the tick commits exactly these
    return len(new)


# Run on the worker over ssh: the sha256 of each named blob, as `tree_sha256` computes it.
_VERIFY = r"""
import hashlib, json, sys
from pathlib import Path
def tree(p):
    if p.is_file():
        return hashlib.sha256(p.read_bytes()).hexdigest()
    d = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        d.update(f"{f.relative_to(p).as_posix()}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return d.hexdigest()
root = Path(sys.argv[1]).expanduser()
print(json.dumps({r: tree(root / r) for r in sys.argv[2:] if (root / r).exists()}))
"""


def _relay_blobs(rels: Sequence[str], source: str, remote: str) -> list[str]:
    """Stream blobs from the cluster's batch store to the worker store; return those verified."""

    from menagerie.pipeline import store as store_mod

    if not rels:
        return []
    dest_host, dest_root = store_mod._split(remote)  # noqa: SLF001 - the store's own parser
    names = " ".join(shlex.quote(r) for r in rels)
    pack = subprocess.Popen(["ssh", "-o", "BatchMode=yes", host(),
                             f"cd {shlex.quote(source)} && tar -cf - {names}"],
                            stdout=subprocess.PIPE)
    unpack = f"mkdir -p {dest_root} && tar --skip-old-files -x -C {dest_root}"
    argv = ["ssh", "-o", "BatchMode=yes", dest_host, unpack] if dest_host else \
        ["sh", "-c", unpack]
    done = subprocess.run(argv, stdin=pack.stdout, capture_output=True, text=True, timeout=7200)
    pack.wait()
    if done.returncode or pack.returncode:
        raise RuntimeError(f"blob relay failed: {done.stderr.strip()[:300]}")
    check = ["python3", "-c", _VERIFY, dest_root, *rels]
    argv = ["ssh", "-o", "BatchMode=yes", dest_host, shlex.join(check)] if dest_host else check
    seen = json.loads(subprocess.run(argv, check=True, capture_output=True, text=True,
                                     timeout=3600).stdout)
    return [rel for rel in rels if seen.get(rel) == Path(rel).name]


def _pull_back(batch: Path, meta: dict[str, Any], there: str) -> Path:
    """Copy the job's small outputs (never its blobs) into ``back/``; return that folder."""

    back = batch / "back"
    back.mkdir(parents=True, exist_ok=True)
    keep = ["--include=DONE", "--include=table.jsonl", "--include=data/",
            "--include=data/ledger/***", "--include=data/store-index/***", "--include=work/",
            "--include=work/attempts/", "--include=work/attempts/*/",
            "--include=work/attempts/*/result.json", "--include=work/attempts/*/request.json",
            "--include=work/attempts/*/attempt.log", "--exclude=*"]
    _rsync(f"{host()}:{there}/", f"{back}/", *keep)
    _rsync(f"{host()}:{meta['root']}/logs/{batch.name}-*", f"{back}/")
    return back


def _relay_indexed(back: Path, there: str) -> int:
    """Relay every blob the job indexed to the worker store; raise unless all verify there."""

    shas = sorted({json.loads(line)["sha256"]
                   for shard in (back / "data" / "store-index").glob("*.jsonl")
                   for line in shard.read_text(encoding="utf-8").splitlines() if line})
    if not shas:
        return 0
    remote = os.environ.get("MENAGERIE_STORE_REMOTE")
    if not remote:
        raise RecordError("MENAGERIE_STORE_REMOTE is not set; it names the worker store the "
                          "batch's blobs go to")
    rels = [f"sha256/{sha[:2]}/{sha}" for sha in shas]
    missing = sorted(set(rels) - set(_relay_blobs(rels, f"{there}/store", remote)))
    if missing:  # index lines only for blobs the worker store holds
        raise RuntimeError(f"{len(missing)} blobs did not verify at {remote}: {missing[:3]}")
    return len(rels)


def _summary_rows(back: Path) -> list[dict[str, Any]]:
    table = back / "table.jsonl"
    rows = [json.loads(line) for line in table.read_text().splitlines() if line] \
        if table.is_file() else []
    return [{k: r.get(k) for k in ("stable_id", "forward", "trace", "trace_class", "n_ops")}
            for r in rows]


def collect(batch: Path, *, data: DataLayout, ended: bool = False) -> dict[str, Any]:
    """Bring a finished batch back from the cluster.

    Inputs: the batch folder (as written by ``axon_batch``); ``ended`` when the job has left the
    queue, so a batch with no DONE (killed at its time limit) is collected for what finished. Returns: a summary: ``done`` (false
    while the job runs), lines added per shard, blobs relayed, rows whose recipe revision moved
    on main since the batch was written (``stale``) and the table. Side effects: ``back/`` in the
    batch folder (DONE, table, log, ledger and store-index lines, attempt results), blobs in the
    worker store, then lines appended to main's ledger and store-index shards.
    """

    batch = Path(batch)
    meta = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    there = f"{meta['root']}/batches/{batch.name}"
    status = _ssh(f"cat {shlex.quote(there)}/DONE 2>/dev/null || true").strip()
    if not status and ended:  # SLURM ended the job (its time limit) before the script wrote DONE
        status = "ended without DONE"
    if not status:
        return {"batch": batch.name, "done": False}
    back = _pull_back(batch, meta, there)
    blobs = _relay_indexed(back, there)
    added = {f"{kind}/{shard.name}": _append_new(shard, data.root / kind / shard.name)
             for kind in ("ledger", "store-index")
             for shard in sorted((back / "data" / kind).glob("*.jsonl"))}
    current = {sid: read_json(ModelRecord, data.model(sid)).recipe for sid in meta["ids"]}
    stale = [sid for sid, rev in meta.get("revisions", {}).items()
             if current[sid] is None or current[sid].rev != rev]  # type: ignore[union-attr]
    return {"batch": batch.name, "done": status, "added": added, "blobs": blobs, "stale": stale,
            "rows": _summary_rows(back)}
