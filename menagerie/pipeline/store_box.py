"""Daily off-site batches for the live trace store, through the fleet-storage job ``store-box``.

The trace store (``MENAGERIE_STORE`` on the worker) is content-addressed: a blob never changes
once its name is visible. Its first box copy went as hard-linked prefix batches under the job's
scratch root (``<root>/<batch>/payload/sha256/<xx>/<sha>``). Each day this module:

1. on the worker (``worker``): finishes collected batches (deletes the collector's verification
   restore, writes the published marker that keeps the nightly out), lists every live blob with
   its file count and bytes, hard-links the blobs no sealed batch holds into new
   ``daily-YYYYMMDD-N`` batches and seals them with ``fleet-checkpoint``;
2. on the hub (``daily``): waits for the 15-minute collector to archive and fully restore each
   new batch, finishes them on the worker, then installs the live listing where the fleet's
   excluded-path check reads it (the desk storage state, ``content-listings``). The check counts
   the store as covered only while that listing is under 36 hours old and every listed blob is
   whole in a full-restore-verified snapshot;
3. on the hub, once every new batch verified (``trim``): runs the store trim (decision 6:
   superseded saves, zoom tiles, rendered portable saves, portable twins, packing raw trace saves)
   on the worker against the store's verified batch manifests, and commits the placement lines it
   wrote.

Limits (the collector's rules): each batch keeps its viewable files under 2.5 GiB (the collector
refuses 3 GiB), and the worker keeps free space at least ``--floor-gb`` plus every new batch plus
5 GiB, because each batch is restored in full there. Blobs that do not fit wait for the next day.
The collector restores, checks and deletes one batch at a time, so a day's blobs go in batches of
at most 4 GiB each: the worker then needs only one batch of extra space at a time (one 16.5 GB
batch took the worker's free space from 66 to 50.6 GiB on 2026-10-28 and stalled the pipeline).
Steps 1 and 2 never delete or move anything in the store; step 3 deletes only what
``store_trim`` proves a verified snapshot holds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

GIB = 2**30
VIEW = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".md", ".txt", ".json", ".csv", ".log",
        ".html", ".htm", ".ipynb", ".tsv"}  # the collector's viewable suffixes
VIEW_FILE_CAP = 250 * 2**20
VIEW_BATCH_CAP = int(2.5 * GIB)
BATCH_BYTES_CAP = 4 * GIB  # the collector's verification restore needs this much free, one batch at a time
MARGIN = 5 * GIB
# fleet-archive-shape refuses an archive file over 1 GiB in a preservation batch; a packed
# trace save that large stays local (reported) instead of failing the whole day's batch.
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar.zst", ".zip", ".7z")
ARCHIVE_FILE_CAP = 4 * GIB
JOB = "store-box"
MARKER = "FLEET-STORAGE.json"


# --- worker side --------------------------------------------------------------------------

def blob_files(path: Path) -> tuple[list[Path], int]:
    """A blob's regular files and its symlink count (symlinks are never batch payload)."""
    if path.is_symlink():
        return [], 1
    if not path.is_dir():
        return [path], 0
    files, links = [], 0
    for parent, dirs, names in os.walk(path):
        entries = [Path(parent, n) for n in dirs + names]
        links += sum(e.is_symlink() for e in entries)
        files += [Path(parent, n) for n in names if not Path(parent, n).is_symlink()]
    return files, links


def blob_stats(path: Path) -> dict:
    """Regular files, bytes, viewable bytes and symlinks of one blob (a file or a tree)."""
    files, links = blob_files(path)
    sizes = [(f.stat().st_size, f.suffix.lower()) for f in files]
    return {"files": len(files), "bytes": sum(s for s, _ in sizes), "symlinks": links,
            "viewable": sum(s for s, x in sizes if x in VIEW and s <= VIEW_FILE_CAP)}


def live_blobs(store: Path) -> dict[str, dict]:
    """Every visible blob under ``store/sha256/<xx>/``; hidden names are in-flight writes."""
    blobs = {}
    root = store / "sha256"
    for prefix in sorted(os.listdir(root)):
        for name in sorted(os.listdir(root / prefix)):
            if not name.startswith("."):
                blobs[f"sha256/{prefix}/{name}"] = blob_stats(root / prefix / name)
    return blobs


def boxed(root: Path) -> set[str]:
    """Blob keys held by any sealed batch under the job root."""
    held = set()
    for batch in sorted(p for p in root.iterdir() if (p / "READY.json").is_file()):
        payload = batch / "payload"
        for prefix in sorted((payload / "sha256").glob("*")):
            held.update(f"sha256/{prefix.name}/{name}" for name in os.listdir(prefix))
    return held


def oversize(new: dict[str, dict]) -> list[str]:
    """Single-file archive blobs over the fleet's archive cap, which no batch may carry."""
    return sorted(k for k, v in new.items()
                  if v["files"] == 1 and k.lower().endswith(ARCHIVE_SUFFIXES) and v["bytes"] > ARCHIVE_FILE_CAP)


def plan_batches(new: dict[str, dict], room: int) -> tuple[list[list[str]], list[str]]:
    """Group new blobs in key order under the viewable and byte caps; the day's total stays
    within the free-space room (the bound should a collector cleanup fail and restores pile up)."""
    batches, waiting, used, view, size = [[]], [], 0, 0, 0
    for key in sorted(new):
        stats = new[key]
        if stats["viewable"] > VIEW_BATCH_CAP or used + stats["bytes"] > room:
            waiting.append(key)
            continue
        if batches[-1] and (view + stats["viewable"] > VIEW_BATCH_CAP
                            or size + stats["bytes"] > BATCH_BYTES_CAP):
            batches.append([])
            view, size = 0, 0
        batches[-1].append(key)
        used += stats["bytes"]
        view += stats["viewable"]
        size += stats["bytes"]
    return [b for b in batches if b], waiting


def link_batch(store: Path, dest: Path, keys: list[str]) -> None:
    """Hard-link each blob (file or tree) into ``dest/payload``; refuses an existing batch."""
    payload = dest / "payload"
    payload.mkdir(parents=True)  # FileExistsError for a batch name already used
    for key in keys:
        src, out = store / key, payload / key
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            os.link(src, out)
            continue
        for parent, _, names in os.walk(src):
            rel = Path(parent).relative_to(src)
            (out / rel).mkdir(parents=True, exist_ok=True)
            for name in names:
                if not Path(parent, name).is_symlink():
                    os.link(Path(parent, name), out / rel / name)


def verification_root(host: str, root: Path, batch: str) -> Path:
    """Where the collector restored a batch to verify it (fleet-storage's manifest id)."""
    ident = hashlib.sha256(f"{host}|{root}|{batch}".encode()).hexdigest()[:32]
    return Path.home() / ".local/state/fleet-storage/verification" / ident


def finish(host: str, root: Path, batch: str, sid: str, verified_at: str) -> dict:
    """Delete a collected batch's verification restore and write its published marker."""
    restore = verification_root(host, root, batch)
    existed = restore.is_dir()
    if existed:
        shutil.rmtree(restore)
    marker = root / batch / MARKER
    if not marker.exists():
        marker.write_text(json.dumps({
            "marker": "published fleet-storage checkpoint batch; the nightly skips this directory",
            "schema": "published-marker/1", "job": f"{host}/menagerie/{JOB}", "batch": batch,
            "snapshot": sid, "catalog": f"mirror/fleet-catalog/{sid}.json",
            "verification": {"status": "full_restore_sha256", "at": verified_at}}, indent=1) + "\n")
    return {"batch": batch, "restore_deleted": existed}


def parse_finish(item: str) -> tuple[str, str, str]:
    """``BATCH=SNAPSHOT@VERIFIED_AT`` -> its three parts."""
    batch, _, rest = item.partition("=")
    sid, _, verified_at = rest.partition("@")
    if not (batch.startswith("daily-") and len(sid) == 64 and verified_at):
        raise ValueError(f"bad --finish value {item!r}")
    return batch, sid, verified_at


def pending(root: Path) -> list[str]:
    """Sealed batches not yet finished (no published marker): the collector may still be
    archiving them, so nothing they link may be removed."""
    return sorted(p.name for p in root.iterdir()
                  if (p / "READY.json").is_file() and not (p / MARKER).exists())


def seal(store: Path, root: Path, date: str, batches: list[list[str]], new: dict[str, dict]) -> list[dict]:
    sealed = []
    for n, keys in enumerate(batches, 1):
        name = f"daily-{date}-{n}"
        link_batch(store, root / name, keys)
        subprocess.run([str(Path.home() / ".local/bin/fleet-checkpoint"), "--content-addressed", str(root / name)],
                       check=True)
        sealed.append({"batch": name, "blobs": len(keys), "bytes": sum(new[k]["bytes"] for k in keys),
                       "viewable": sum(new[k]["viewable"] for k in keys)})
    return sealed


def worker(args: argparse.Namespace) -> int:
    store = Path(os.environ["MENAGERIE_STORE"])
    root = Path.home() / "scratch/menagerie" / JOB
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    finished = [finish(args.host, root, *parse_finish(item)) for item in args.finish]
    if not args.build:
        (out / "summary.json").write_text(json.dumps({"finished": finished}, indent=1))
        print(json.dumps({"finished": finished}))
        return 0
    listed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    roots = sorted(os.listdir(store))
    blobs = live_blobs(store)
    held = boxed(root)
    new = {k: v for k, v in blobs.items() if k not in held}
    too_big = oversize(new)
    new = {k: v for k, v in new.items() if k not in too_big}
    free = shutil.disk_usage(root).free
    batches, waiting = plan_batches(new, max(0, free - int(args.floor_gb * 1e9) - MARGIN))
    sealed = seal(store, root, args.date, batches, new)
    links = sum(v["symlinks"] for v in blobs.values())
    listing = {"host": args.host, "path": str(store), "listed_at": listed_at,
               "complete": roots == ["sha256"] and links == 0, "roots": roots,
               "producer": "menagerie.pipeline.store_box worker",
               "blobs": [{"blob": k, "files": v["files"], "bytes": v["bytes"]} for k, v in blobs.items()]}
    (out / "listing.json").write_text(json.dumps(listing))
    summary = {"listed_at": listed_at, "blobs": len(blobs), "bytes": sum(v["bytes"] for v in blobs.values()),
               "symlinks": links, "roots": roots, "new_blobs": len(new), "sealed": sealed,
               "waiting": len(waiting), "free_before": free, "finished": finished,
               "pending": pending(root), "oversize_local": too_big}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "finished"}))
    return 0


# --- hub side -----------------------------------------------------------------------------

def log(message: str, state: Path) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with (state / "daily.log").open("a") as handle:
        handle.write(line + "\n")


def job_host(desk: Path) -> str:
    records = sorted(desk.glob(f"state/storage/jobs/*/menagerie/{JOB}.json"))
    if len(records) != 1:
        raise SystemExit(f"expected one fleet-storage {JOB} job record, found {len(records)}")
    return records[0].parent.parent.name


def receipt(repo: Path, batch: str) -> dict | None:
    """The collector's archive-only receipt for a batch, once it has verified and catalogued it."""
    path = repo / "outputs" / JOB / batch / MARKER
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    verification = manifest.get("verification") or {}
    if verification.get("status") != "full_restore_sha256":
        return None
    return {"snapshot": manifest["archive"]["snapshot"], "verified_at": verification["at"]}


def run_job(repo: Path, host: str, job: str, command: str, *, pull: str = "outputs/*.json",
            memory_gib: int = 2, hours: float = 1.0) -> tuple[Path, str]:
    """Run ``command`` on the worker through run-on (1 CPU) at the checkout's HEAD; pull its
    outputs whether it passed or failed. Returns the pulled folder and the job's last status."""
    ref = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    subprocess.run(["run-on", "submit", "--admit-wait", "3600", "--project", "menagerie", "--host", host,
                    "--os", "linux", "--no-fallback", "--ref", ref, "--cpus", "1", "--memory-gib",
                    str(memory_gib), "--job", job, "--pull", pull, "--", "env",
                    f"MENAGERIE_MEMORY_GB={memory_gib}", "MENAGERIE_CPUS=1", "bash", "-c", command],
                   cwd=repo, check=True, capture_output=True, text=True)
    last = ""
    for _ in range(int(hours * 360)):
        status = subprocess.run(["run-on", "status", job], capture_output=True, text=True).stdout
        last = status.strip().splitlines()[-1] if status.strip() else ""
        if any(word in last for word in ("DONE", "FAIL", "INTERRUPTED")):
            break
        time.sleep(10)
    else:
        subprocess.run(["run-on", "cancel", job], capture_output=True, text=True)
        raise RuntimeError(f"worker job {job} did not finish within {hours:g} hours; cancelled")
    subprocess.run(["pull-outputs", host, job, "--pull", pull], check=True,
                   capture_output=True, text=True)
    return repo / "outputs" / job, last


def run_worker(repo: Path, host: str, job: str, worker_args: list[str]) -> Path:
    """Run the worker step through run-on (1 CPU, 2 GiB) and pull its two JSON outputs."""
    command = ("export MENAGERIE_STORE=$HOME/data/menagerie/store; uv run --no-project --python 3.11 "
               "python -m menagerie.pipeline.store_box worker " + " ".join(worker_args))
    back, last = run_job(repo, host, job, command)
    if "DONE rc=0" not in last:
        raise RuntimeError(f"worker job {job}: {last}")
    found = sorted(back.rglob("summary.json"))
    if not found:
        raise RuntimeError(f"worker job {job} returned no summary")
    return found[0].parent


def wait_collected(repo: Path, batches: list[str], hours: float, state: Path) -> dict[str, dict]:
    deadline = time.monotonic() + hours * 3600
    while True:
        got = {b: r for b in batches if (r := receipt(repo, b))}
        if len(got) == len(batches) or time.monotonic() > deadline:
            return got
        log(f"waiting for the collector: {len(got)}/{len(batches)} batches archived", state)
        time.sleep(300)


def install_listing(desk: Path, host: str, listing_file: Path) -> Path:
    listing = json.loads(listing_file.read_text())
    name = hashlib.sha256(listing["path"].encode()).hexdigest()[:16]
    dest = desk / "state/storage/content-listings" / host / f"{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(listing))
    tmp.replace(dest)
    return dest


def unfinished(repo: Path) -> list[str]:
    """Daily batches whose receipt arrived in the last 14 days (the worker skips marked ones)."""
    cutoff = time.time() - 14 * 86400
    return sorted(p.parent.name for p in (repo / "outputs" / JOB).glob(f"daily-*/{MARKER}")
                  if p.stat().st_mtime > cutoff)


def daily(args: argparse.Namespace) -> int:
    repo, desk = Path(args.repo), Path(args.desk)
    state = repo / "outputs" / "store-box-daily"
    state.mkdir(parents=True, exist_ok=True)
    with (state / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = job_host(desk)
        today = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M")
        finish_args = [f"{b}={r['snapshot']}@{r['verified_at']}"
                       for b in unfinished(repo) if (r := receipt(repo, b))]
        build_args = ["--host", host, "--date", today, "--floor-gb", str(args.floor_gb), "--build"]
        build_args += [x for b in finish_args for x in ("--finish", b)]
        out = run_worker(repo, host, f"menagerie-storebox-daily-{today}", build_args)
        summary = json.loads((out / "summary.json").read_text())
        log(f"listed {summary['blobs']} blobs, {summary['new_blobs']} new, sealed "
            f"{[s['batch'] for s in summary['sealed']]}, {summary['waiting']} waiting, "
            f"{len(summary.get('oversize_local', []))} over the archive cap kept local", state)
        batches = [s["batch"] for s in summary["sealed"]]
        got = wait_collected(repo, batches, args.wait_hours, state)
        if len(got) < len(batches):
            log(f"collector did not archive {sorted(set(batches) - set(got))} in time; listing kept back", state)
            trim_gate(batches, got, state)  # logs the refusal: nothing is trimmed today
            return 1
        # An older unfinished batch the collector verified while this run waited is finished with today's.
        pending = summary.get("pending", batches)  # today's batches and any older unfinished
        got = {**got, **{b: r for b in pending if b not in got and (r := receipt(repo, b))}}
        if got:
            done = [f"{b}={r['snapshot']}@{r['verified_at']}" for b, r in got.items()]
            run_worker(repo, host, f"menagerie-storebox-finish-{today}",
                       ["--host", host, "--date", today, "--floor-gb", str(args.floor_gb)]
                       + [x for b in done for x in ("--finish", b)])
            log(f"finished {sorted(got)} on the worker (restores deleted, markers written)", state)
        dest = install_listing(desk, host, out / "listing.json")
        log(f"installed live listing {dest.name} (listed {summary['listed_at']}); "
            f"{summary['waiting']} blobs wait for a later batch", state)
        ssh = args.ssh or f"{host}-hub"
        verified = {**got, **marked(ssh, [b for b in pending if b not in got])}
        trimmed = trim(repo, host, today, ssh, state) \
            if args.trim and trim_gate(pending, verified, state) else 0
        return 0 if summary["waiting"] == 0 and trimmed == 0 else 1


# --- hub side: the store trim after a verified batch ------------------------------------------
# Decision 6 (pack and trim): once today's batches are full-restore verified, the blobs they and
# the earlier batches hold may leave the worker store. The proof catalog is the store's own batch
# receipts (``outputs/store-box/*/FLEET-STORAGE.json``), copied gzipped to the worker; the trim
# itself (``store_trim``, through ``store_trim_job.sh``) proves every blob before it goes.

TRIM_STEPS = ("superseded", "tiles", "portable-rendered", "portable-twin", "pack", "pack-views",
              "ceiling")
TRIM_CATALOG = "scratch/menagerie/store-trim/catalog"  # on the worker, under $HOME
TRIM_JOURNAL = "scratch/menagerie/store-trim/journal"  # placement lines, kept until committed
PLACEMENT = Path("menagerie/data/store-index/placement.jsonl")


BOX_ROOT = f"scratch/menagerie/{JOB}"  # the job root on the worker, under $HOME


def marked(ssh: str, batches: list[str]) -> dict[str, dict]:
    """The batches whose published marker on the worker records a full-restore check (another
    run finished them after this run's worker step listed them pending); none when unreachable."""
    if not batches:
        return {}
    paths = " ".join(shlex.quote(f"{BOX_ROOT}/{b}/{MARKER}") for b in batches)
    done = subprocess.run(["ssh", ssh, f"grep -l full_restore_sha256 -- {paths}"],
                          capture_output=True, text=True, timeout=120, check=False)
    found = {Path(line).parent.name for line in done.stdout.split()}
    return {b: {"marker": True} for b in batches if b in found}


def trim_gate(batches: list[str], got: dict[str, dict], state: Path) -> bool:
    """Whether the trim may run: every sealed, unfinished batch (today's, and any older one
    still waiting for the collector) is full-restore verified at trim time, by its receipt on
    the hub or its published marker on the worker."""
    missing = sorted(set(batches) - set(got))
    if missing:
        log(f"trim refused: batches {missing} are not verified", state)
    return not missing


def sync_catalog(repo: Path, ssh: str, state: Path) -> int:
    """Gzip each verified batch receipt once (hub cache), then mirror them to the worker."""
    cache = state / "catalog"
    cache.mkdir(exist_ok=True)
    names = set()
    for marker in sorted((repo / "outputs" / JOB).glob(f"*/{MARKER}")):
        if receipt(repo, marker.parent.name) is None:
            continue
        out = cache / f"{marker.parent.name}.json.gz"
        names.add(out.name)
        if not out.is_file() or out.stat().st_mtime < marker.stat().st_mtime:
            tmp = out.with_name(f".{out.name}")
            with marker.open("rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            tmp.replace(out)
    for stale in set(p.name for p in cache.glob("*.json.gz")) - names:
        (cache / stale).unlink()
    subprocess.run(["ssh", ssh, f"mkdir -p {TRIM_CATALOG}"], check=True, capture_output=True,
                   text=True, timeout=120)
    subprocess.run(["rsync", "-a", "--delete", f"{cache}/", f"{ssh}:{TRIM_CATALOG}/"],
                   check=True, capture_output=True, text=True, timeout=3600)
    return len(names)


def trim(repo: Path, host: str, stamp: str, ssh: str, state: Path) -> int:
    """Run the store trim on the worker and commit its placement lines; 0 when all went well.

    The worker writes each placement line, before its delete, to a journal outside the job
    checkout; every run first and last commits whatever the journals hold (``reconcile``), so a
    failed pull or a timeout never loses a line."""
    last, job = "", f"menagerie-storetrim-{stamp}"
    try:
        reconcile(repo, ssh, stamp, state)
        count = sync_catalog(repo, ssh, state)
        command = (f"TRIM_HOST={host} TRIM_CATALOG=$HOME/{TRIM_CATALOG} TRIM_JOURNAL=$HOME/{TRIM_JOURNAL} "
                   f"TRIM_TAG={job} bash menagerie/pipeline/store_trim_job.sh apply {' '.join(TRIM_STEPS)}")
        _, last = run_job(repo, host, job, command, pull="outputs/*", memory_gib=8, hours=2)
        log(f"trim {job} ({count} catalog manifests): {last}", state)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        log(f"trim {job} failed before it finished: {exc}", state)
    try:
        added = reconcile(repo, ssh, stamp, state)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"placement journal not reconciled ({exc}); the next run retries", state)
        return 1
    return 0 if "DONE rc=0" in last and added >= 0 else 1


def ended(job: str) -> bool:
    """Whether run-on reports the job finished (DONE or FAILED), so its journal is closed."""
    status = subprocess.run(["run-on", "status", job], capture_output=True, text=True).stdout
    last = status.strip().splitlines()[-1] if status.strip() else ""
    return job in last and any(word in last for word in ("DONE", "FAILED"))


def reconcile(repo: Path, ssh: str, stamp: str, state: Path) -> int:
    """Copy the worker's placement journals here and commit the lines the shard lacks; then
    delete the journals of jobs that had ended before the copy (so the copy is complete; a
    running job may still append). Returns the lines added, -1 when the commit was refused
    (every journal stays)."""
    local = state / "journal"
    local.mkdir(exist_ok=True)
    listed = subprocess.run(["ssh", ssh, f"mkdir -p {TRIM_JOURNAL} && ls -1 {TRIM_JOURNAL}"],
                            check=True, capture_output=True, text=True, timeout=120).stdout
    suffix = ".placement.jsonl"
    closed = [n for n in listed.split() if n.endswith(suffix) and ended(n.removesuffix(suffix))]
    subprocess.run(["rsync", "-a", f"{ssh}:{TRIM_JOURNAL}/", f"{local}/"], check=True,
                   capture_output=True, text=True, timeout=3600)
    journals = sorted(local.glob(f"*{suffix}"))  # job names sort by their stamp
    added = commit_placement(repo, [p.read_text(encoding="utf-8") for p in journals], stamp, state)
    if added < 0 or not closed:
        return added
    names = " ".join(shlex.quote(n) for n in closed)
    subprocess.run(["ssh", ssh, f"cd {TRIM_JOURNAL} && rm -f -- {names}"], check=True,
                   capture_output=True, text=True, timeout=120)
    for name in closed:
        (local / name).unlink(missing_ok=True)
    return added


def commit_placement(repo: Path, texts: list[str], stamp: str, state: Path) -> int:
    """Append the placement lines the committed shard does not already hold (each was written
    before its delete, so true even after a failure) and commit that file alone; -1 when
    refused."""
    shard = repo / PLACEMENT
    dirty = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", str(PLACEMENT)])
    if dirty.returncode != 0:  # clean, so the file read below is the committed shard
        log(f"placement lines kept in the journal: {PLACEMENT} has uncommitted changes", state)
        return -1
    body = shard.read_bytes() if shard.is_file() else b""
    held = set(body.decode("utf-8").splitlines())
    new = list(dict.fromkeys(line for text in texts for line in text.splitlines()
                             if line.strip() and line not in held))
    if not new:
        return 0
    with shard.open("a", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in new))
    message = f"data: store placement: {len(new)} lines from the daily trim {stamp}"
    done = subprocess.run([str(Path.home() / "desk/bin/git-commit-paths"), "-C", str(repo), "-m",
                           message, "--", str(PLACEMENT)], capture_output=True, text=True)
    if done.returncode != 0:  # put the file back as committed; the journal keeps the lines
        with shard.open("r+b") as handle:
            handle.truncate(len(body))
        log(f"placement commit failed ({done.stderr.strip()[-200:]}); the journal keeps the lines",
            state)
        return -1
    return len(new)


# --- hub side: restore offloaded blobs -------------------------------------------------------
# ``store_trim offload`` deletes local copies that a verified snapshot holds and records
# ``location: box:<snapshot>`` on an index line. This brings them back: one broker restore per
# snapshot with every blob as an ``--include``, a tree-hash check on the hub, an rsync to a hidden
# name in the worker store, and a tree-hash check there before the blob gets its own name. A blob
# offloaded packed (``encoding: tar.gz``) comes back as its ``<sha>.tar.gz``, checked by the tree
# its stream holds.

PACKED = ".tar.gz"

BOX_REF = "hetzner_storage_box"
BOX_ENV = {"HZ_HOST": "host", "HZ_USER": "username", "HZ_PORT": "port", "HZ_KEY": "private_key",
           "RESTIC_PASSWORD": "restic_password"}
TREE_SHA = """import hashlib, json, pathlib, sys, tarfile
def digest(pairs):
    d = hashlib.sha256()
    for rel, s in sorted(pairs, key=lambda pair: pathlib.PurePosixPath(pair[0]).parts):
        d.update(f"{rel}\\0{s}\\n".encode())
    return d.hexdigest()
def packed(p):
    with tarfile.open(p, "r|gz") as tar:
        return digest([(m.name, hashlib.sha256(tar.extractfile(m).read()).hexdigest())
                       for m in tar if m.isfile()])
def tree(p):
    if p.is_file():
        return packed(p) if p.name.endswith(".tar.gz") else hashlib.sha256(p.read_bytes()).hexdigest()
    return digest([(f.relative_to(p).as_posix(), hashlib.sha256(f.read_bytes()).hexdigest())
                   for f in p.rglob("*") if f.is_file()])
def place(root, s):
    final = root / "sha256" / s[:2] / s
    staged = final.with_name(".restore-" + s)
    want = s[:-len(".tar.gz")] if s.endswith(".tar.gz") else s
    if staged.exists() and tree(staged) == want and not final.exists():
        os.replace(staged, final)  # whole and checked before its name is visible
    if staged.exists():
        shutil.rmtree(staged) if staged.is_dir() else staged.unlink()
    return tree(final) if final.exists() else None
import os, shutil
root = pathlib.Path(sys.argv[1]).expanduser()
print(json.dumps({s: place(root, s) for s in sys.argv[2:]}))
"""


def box_env(desk: Path) -> dict[str, str]:
    """Credential references (never values) for the broker, from the fleet's reference list."""
    refs = json.loads((desk.parent / ".claude/knowledge/secret-references.json").read_text())
    item = refs[BOX_REF]["item"]
    return {name: item + field for name, field in BOX_ENV.items()}


def snapshots_of(shas: list[str], data: Path) -> dict[str, list[str]]:
    """Snapshot id -> the store names it holds (``<sha>``, or ``<sha>.tar.gz`` for a blob
    offloaded packed), from the latest ``location: box:`` index line of each blob."""
    where: dict[str, tuple[str, str]] = {}
    for shard in sorted((data / "store-index").glob("*.jsonl")):
        for text in shard.read_text(encoding="utf-8").splitlines():
            line = json.loads(text) if text.strip() else {}
            if line.get("sha256") in shas and str(line.get("location", "")).startswith("box:"):
                name = line["sha256"] + (PACKED if line.get("encoding") == "tar.gz" else "")
                where[line["sha256"]] = (line["location"][4:], name)
    missing = sorted(set(shas) - set(where))
    if missing:
        raise SystemExit(f"no box location recorded for {missing}")
    grouped: dict[str, list[str]] = {}
    for _, (sid, name) in sorted(where.items()):
        grouped.setdefault(sid, []).append(name)
    return grouped


def name_hash(path: Path) -> str:
    """The tree sha256 a restored store name holds (a packed blob: the tree in its stream)."""
    from menagerie.pipeline import store, tree_sha256

    return store.packed_stats(path)[0] if path.name.endswith(PACKED) else tree_sha256(path)


PREFIX_AT = 8  # this many blobs under one prefix restore as the whole prefix, in one call


def restore_batch(desk: Path, sid: str, shas: list[str], target: Path) -> Path:
    """Restore ``shas`` from snapshot ``sid`` under ``target``; returns the restored ``payload``.
    The broker takes one ``--include`` a call (about 9 s each), so a prefix holding many of the
    blobs is restored whole instead."""
    source = json.loads((desk / "state/storage/catalog" / f"{sid}.json").read_text())["source"]
    prefixes: dict[str, list[str]] = {}
    for sha in shas:
        prefixes.setdefault(sha[:2], []).append(sha)
    includes = [f"{source}/sha256/{p}" if len(group) >= PREFIX_AT else f"{source}/sha256/{p}/{s}"
                for p, group in sorted(prefixes.items())
                for s in (group[:1] if len(group) >= PREFIX_AT else group)]
    env = {**os.environ, **box_env(desk)}
    for n, include in enumerate(includes):
        done = subprocess.run([str(Path.home() / ".local/bin/op-hub"), "agent", "run", "--",
                               str(desk / "bin/storagebox"), "restic", "restore", sid,
                               "--target", str(target / str(n)), "--include", include], env=env,
                              capture_output=True, text=True, timeout=7200, check=False)
        if done.returncode != 0:
            raise SystemExit(f"box restore of {include} from {sid} exited {done.returncode}: "
                             f"{(done.stdout + done.stderr)[-300:]}")
        _merge(target / str(n) / source.lstrip("/"), target / "payload")
    return target / "payload"


def _merge(src: Path, dest: Path) -> None:
    """Move each restored ``sha256/<xx>/<sha>`` from one call's target into the shared payload."""
    for blob in sorted(src.glob("sha256/*/*")):
        out = dest / "sha256" / blob.parent.name / blob.name
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            os.replace(blob, out)


def restore(args: argparse.Namespace) -> int:
    report = restore_blobs(args.sha, remote=args.remote, data=Path(args.data), desk=Path(args.desk))
    print(json.dumps(report))
    return 0 if not report["failed"] else 1


def restore_blobs(shas: list[str], *, remote: str, data: Path, desk: Path) -> dict:
    """Bring offloaded blobs back into the worker store ``remote`` (``host:path``); returns the
    store names restored and failed (each checked by its tree hash there) by snapshot."""
    shas = sorted(set(shas))
    host, root = remote.split(":", 1)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    work = Path.home() / "restores" / f"menagerie-box-restore-{stamp}"  # the broker's only target
    report: dict = {"restored": [], "snapshots": {}}
    try:
        names = []
        for sid, group in snapshots_of(shas, data).items():
            payload = restore_batch(desk, sid, group, work / sid[:12])
            bad = [n for n in group
                   if name_hash(payload / "sha256" / n[:2] / n) != n.removesuffix(PACKED)]
            if bad:
                raise SystemExit(f"restored copies of {bad} from {sid} do not hash to their names")
            for name in group:  # under a hidden name; the worker renames it once it verifies
                src = payload / "sha256" / name[:2] / name
                subprocess.run(["rsync", "-a", f"{src}{'/' if src.is_dir() else ''}",
                                f"{host}:{root}/sha256/{name[:2]}/.restore-{name}"],
                               check=True, capture_output=True, timeout=7200)
            report["snapshots"][sid] = group
            names += group
        there = subprocess.run(["ssh", host, "python3", "-", root, *names], input=TREE_SHA,
                               check=True, capture_output=True, text=True, timeout=7200)
        found = json.loads(there.stdout)
        report["restored"] = [n for n in names if found.get(n) == n.removesuffix(PACKED)]
        report["failed"] = [n for n in names if found.get(n) != n.removesuffix(PACKED)]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="verb", required=True)
    w = sub.add_parser("worker", help="worker step: finish, list, batch and seal")
    w.add_argument("--host", required=True)
    w.add_argument("--date", required=True)
    w.add_argument("--floor-gb", type=float, required=True)
    w.add_argument("--build", action="store_true")
    w.add_argument("--finish", action="append", default=[], metavar="BATCH=SNAPSHOT@VERIFIED_AT")
    d = sub.add_parser("daily", help="hub step: run the worker step, wait for collection, install the listing")
    d.add_argument("--repo", default=str(Path.home() / "projects/menagerie"))
    d.add_argument("--desk", default=str(Path.home() / "desk"))
    d.add_argument("--floor-gb", type=float, default=49.07)
    d.add_argument("--wait-hours", type=float, default=3.0)
    d.add_argument("--no-trim", dest="trim", action="store_false",
                   help="skip the store trim after the batch verifies")
    d.add_argument("--ssh", help="ssh alias of the worker for the catalog copy (default HOST-hub)")
    r = sub.add_parser("restore", help="hub: bring offloaded blobs back into the worker store")
    r.add_argument("sha", nargs="+")
    r.add_argument("--remote", required=True, help="the worker store as host:path")
    r.add_argument("--data", default=str(Path.home() / "projects/menagerie/menagerie/data"))
    r.add_argument("--desk", default=str(Path.home() / "desk"))
    args = parser.parse_args(argv)
    return {"worker": worker, "daily": daily, "restore": restore}[args.verb](args)


if __name__ == "__main__":
    sys.exit(main())
