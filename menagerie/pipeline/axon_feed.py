"""The axon feeder: one cluster batch in flight for the rows past G's 40 GiB (decisions.md 2026-10-28).

The tick runs ``step`` every pass (``tick._step_axon``; config ``axon``, ``CONFIG``):
- an active batch whose SLURM job has left the queue is collected (``cluster.collect``: blobs to
  the worker store, ledger and store-index lines on main). A row it did not finish (not reached,
  infra, or the job ended without it) goes back once, then gets a typed ``timeout:axon_batch``
  trace row. Its passing traces get one render job on the worker (``axon_render_job.sh`` runs
  ``render`` here over the collected attempts), whose lines are appended on main when it ends.
- with none active (full mode, the disk guard clear, no Menagerie task in the cluster queue) the
  next batch is packed and submitted through ``axon-submit``. A row's need is
  ``stage3.projected_peak`` (1.5 times any ``memory_cap`` included): (40, 60] GiB goes in a
  ``cpu`` batch (64 GiB) of up to ``rows.cpu`` rows; only when that band has none ready, (60, 120]
  goes in a ``reviewed`` batch (``giants-128g``, 128 GiB) of up to ``rows.reviewed``, run one after
  another. A batch is packed in feeding order to ``budget_minutes`` of projected row time
  (``minutes``: the band's figure, or ``slow`` for a row whose full trace once timed out on the
  worker); the cluster starts no row in the allocation's last 15 minutes, and a row it did not
  reach goes back once. A row over ``max_gib`` gets one
  typed ``oversized:beyond_allocation`` trace row, which settles it until its recipe or the pin
  moves, and is listed in the state as waiting. Order: rows back from an unfinished batch, then rows with no passing trace of any
  kind, then structure-only ones, then re-traces at an older pin; named models (a library zoo or
  a cited packet) before anonymous repo variants; smaller need first.

State lives in the tick state under ``axon``. The hub imports neither torch nor psutil here.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, now, read_json
from menagerie.pipeline.records import LedgerRow, ModelRecord

CONFIG: dict[str, Any] = {
    "enabled": False,
    # Crawler 2026-10-28: up to four 60-120 GiB rows, one after another, in one giants-128g batch.
    "rows": {"cpu": 6, "reviewed": 4},
    # Projected minutes per row, from the first three batches (2026-10-28: 3 to 11 minutes a
    # row, and 38 for one whose worker trace had timed out); a batch packs to the budget, which
    # leaves the 120-minute allocation room for its setup and a slow row.
    "minutes": {"cpu": 12, "reviewed": 20, "slow": 60},
    "budget_minutes": 100,
    "envs": ["torch-main"],  # one environment per batch; the cluster builds its lock on glibc 2.17
    "max_gib": 120.0,  # past this a row waits: the reviewed 128 GiB allocation cannot hold it
    "render_cpus": 2,
    "render_memory_gib": 6,
}
UNFINISHED_CLASS = "timeout:axon_batch"
OVER_CLASS = "oversized:beyond_allocation"
TRIES = 2  # a row the cluster did not finish goes back once
SKIP_PLACEMENTS = ("gpu-window", "axon-gpu", "structure-only")
ANONYMOUS_ZOOS = ("discovered-", "unregistered-")
HISTORY_KEEP = 50
STEP_ROOM_S = 180  # a collect or a submit starts only with this much of the pass left
RENDER_TEMPLATE = "menagerie/pipeline/axon_render_job.sh"
RENDER_LINES = "outputs/lines.jsonl"
RENDER_TRIES = 3


def config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    given = dict(cfg.get("axon") or {})
    return {**CONFIG, **given, "rows": {**CONFIG["rows"], **(given.get("rows") or {})},
            "minutes": {**CONFIG["minutes"], **(given.get("minutes") or {})}}


# --- choosing rows ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    id: str
    env: str
    need: float
    key: tuple[Any, ...]
    minutes: float = 0.0  # projected; 0 takes the band's figure


def _trace_rows(data: DataLayout) -> dict[str, list[LedgerRow]]:
    from menagerie.pipeline import ledger

    found: dict[str, list[LedgerRow]] = {}
    for row in ledger.read_all(data=data):
        if isinstance(row, LedgerRow) and row.stage == "trace" and row.outcome != "infra" \
                and row.mode in (None, "eval"):
            found.setdefault(row.id, []).append(row)
    return found


def _settled(rows: Sequence[LedgerRow], rev: str, pin: str) -> bool:
    """A full pass at the revision and pin, or a cluster attempt at the revision that ended in
    anything but ``memory_cap`` (a ``memory_cap`` raises the need into the next band)."""

    for r in rows:
        if r.recipe_rev != rev or r.capture not in (None, "full"):
            continue
        if r.outcome == "pass" and r.tl == pin:
            return True
        if r.host == "cluster" and r.outcome == "fail" and \
                not (r.cls or "").startswith("memory_cap"):
            return True
    return False


def _tier(rows: Sequence[LedgerRow]) -> int:
    """0: no passing trace of any kind; 1: structure-only only; 2: a re-trace (an older pin or
    revision passed)."""

    passed = {r.capture or "full" for r in rows if r.outcome == "pass"}
    return 2 if "full" in passed else 1 if passed else 0


def _named(record: ModelRecord, data: DataLayout) -> bool:
    if not record.zoo.startswith(ANONYMOUS_ZOOS):
        return True
    path = data.packet(record.family)
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("citation"))
    except (OSError, ValueError):
        return False


def _eligible(record: ModelRecord, envs: Iterable[str]) -> bool:
    recipe = record.recipe
    return (recipe is not None and record.give_up is None and recipe.env in envs
            and recipe.placement not in SKIP_PLACEMENTS)


def rows(data: DataLayout, cfg: Mapping[str, Any], *, back: Iterable[str] = ()) -> dict[str, list[Row]]:
    """The rows past G by band, ``cpu``, ``reviewed`` and ``over``, each in feeding order."""

    from menagerie.pipeline import stage3
    from menagerie.pipeline.publish import torchlens_pin

    peaks, families = stage3.Peaks.read(data), stage3._families(data)  # noqa: SLF001
    traces, pin, back = _trace_rows(data), torchlens_pin(), set(back)
    bands: dict[str, list[Row]] = {"cpu": [], "reviewed": [], "over": []}
    for path in sorted((data.root / "models").glob("*/*.json")):
        record = read_json(ModelRecord, path)
        mine = traces.get(record.id, [])
        if not _eligible(record, cfg["envs"]) or _settled(mine, record.recipe.rev, pin):  # type: ignore[union-attr]
            continue
        need = stage3.projected_peak(record, peaks, families)
        band = _band(need, cfg)
        if band is not None:
            tier = -1 if record.id in back else _tier(mine)
            key = (tier, not _named(record, data), round(need, 1), record.id)  # type: ignore[arg-type]
            minutes = cfg["minutes"]["slow"] if _slow(mine) else 0.0
            bands[band].append(Row(record.id, record.recipe.env, round(need, 1), key, minutes))  # type: ignore[union-attr,arg-type]
    return {band: sorted(found, key=lambda r: r.key) for band, found in bands.items()}


def _slow(rows: Sequence[LedgerRow]) -> bool:
    """A full trace of the row timed out on the worker once (its cluster trace may be long)."""

    return any((r.cls or "").startswith("timeout") and r.capture in (None, "full") for r in rows)


def _band(need: float | None, cfg: Mapping[str, Any]) -> str | None:
    """``cpu`` (40, 60] GiB, ``reviewed`` up to ``max_gib``, ``over`` past it; ``None`` for G's
    rows and rows with no projection."""

    from menagerie.pipeline import flow

    if need is None or need <= flow.BIG_GIB:
        return None
    if need <= flow.REVIEWED_GIB:
        return "cpu"
    return "reviewed" if need <= float(cfg["max_gib"]) else "over"


def choose(bands: Mapping[str, Sequence[Row]], cfg: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    """The next batch: the cpu band's first rows in one environment, else the reviewed band's;
    packed in order to ``rows`` rows and ``budget_minutes`` of projected time."""

    for kind in ("cpu", "reviewed"):
        found = list(bands.get(kind) or ())
        if found:
            env = found[0].env
            return kind, _pack([r for r in found if r.env == env], kind, cfg)
    return None


def _pack(found: Sequence[Row], kind: str, cfg: Mapping[str, Any]) -> list[str]:
    """Rows in order while they fit; a row that would pass the budget is left for a later batch
    (the first row always goes, so a slow row still runs, alone if need be)."""

    picked: list[str] = []
    used = 0.0
    for r in found:
        minutes = r.minutes or float(cfg["minutes"][kind])
        if picked and used + minutes > float(cfg["budget_minutes"]):
            continue
        picked.append(r.id)
        used += minutes
        if len(picked) == int(cfg["rows"][kind]):
            break
    return picked


# --- the cluster and the worker (tests replace these) ----------------------------------------


def job_id(answer: str) -> str | None:
    """The SLURM job id in ``axon-submit``'s answer (a reviewed submission prints a comment line
    before it)."""

    found = re.findall(r"\b(\d{5,})\b", answer)
    return found[-1] if found else None


class Ops:
    """The feeder's calls out: the cluster's queue and batch verbs, and the worker's render job."""

    def queue(self) -> list[tuple[str, str, str]]:
        """Menagerie's jobs in the cluster queue: (job id, name, state)."""

        from menagerie.pipeline import cluster

        out = cluster._ssh("squeue -h -u \"$(whoami)\" -o '%A|%j|%T'")  # noqa: SLF001
        jobs = [tuple(line.split("|", 2)) for line in out.splitlines() if line.count("|") == 2]
        return [(j, n, s) for j, n, s in jobs if n.startswith("menagerie-")]  # type: ignore[misc]

    def times(self, job: str) -> dict[str, str]:
        """``sacct``'s submit, start and end times and state for a finished job."""

        from menagerie.pipeline import cluster

        out = cluster._ssh(f"sacct -j {shlex.quote(job)} -X -n -P "  # noqa: SLF001
                           "-o Submit,Start,End,State 2>/dev/null | head -1 || true")
        parts = out.strip().split("|")
        return dict(zip(("submit", "start", "end", "state"), parts)) if len(parts) == 4 else {}

    def batch(self, kind: str, ids: Sequence[str], data: DataLayout) -> Path:
        from menagerie.pipeline import cluster

        return cluster.axon_batch(kind, ids, data=data, assign=True).parent  # type: ignore[arg-type]

    def submit(self, batch: Path) -> str:
        from menagerie.pipeline import cluster

        return cluster.submit(batch)

    def collect(self, batch: Path, data: DataLayout) -> dict[str, Any]:
        from menagerie.pipeline import cluster

        return cluster.collect(batch, data=data, ended=True)

    def tidy(self, batch: Path, meta: Mapping[str, Any]) -> None:
        """The batch's folder on the cluster, code trees of other commits, and the staged vendor
        copies on the hub (the worker store holds them)."""

        from menagerie.pipeline import cluster

        root, keep = shlex.quote(meta["root"]), shlex.quote(meta["commit"])
        cluster._ssh(f"rm -rf {root}/batches/{shlex.quote(batch.name)}; cd {root}/code "  # noqa: SLF001
                     f"2>/dev/null && ls | grep -vx {keep} | xargs -r rm -rf; true")
        _remove_tree(batch / "store")

    def render_submit(self, batch: Path, ids: Sequence[str], cfg: Mapping[str, Any]) -> str:
        """Copy the batch's attempts to the worker and start the render job there."""

        host, rel = os.environ["MENAGERIE_RUNNER_HOST"], _worker_attempts(batch.name)
        subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {shlex.quote(rel)}"],
                       check=True, capture_output=True, timeout=60)
        subprocess.run(["rsync", "-a", f"{batch}/back/work/attempts/", f"{host}:{rel}/"],
                       check=True, capture_output=True, timeout=1800)
        mem = int(cfg["render_memory_gib"])
        argv = ["run-on", "submit", "--project", "menagerie", "--os", "linux", "--ref", "main",
                "--cpus", str(cfg["render_cpus"]), "--memory-gib", str(mem), "--no-fallback",
                "--pull", f"{RENDER_LINES},outputs/render.log", "--admit-wait", "60", "--", "env",
                f"AXON_ATTEMPTS={rel}", f"AXON_IDS={' '.join(ids)}", f"MENAGERIE_MEMORY_GB={mem}",
                "bash", RENDER_TEMPLATE]
        done = subprocess.run(argv, capture_output=True, text=True, timeout=180)
        text = done.stdout + done.stderr
        found = re.search(r"^job: +(\S+)", text, re.MULTILINE) or \
            re.search(r"^run-on: (menagerie-\S+) awaiting admission", text, re.MULTILINE)
        if found is None:
            raise OSError(f"render job not started: {text.strip()[-200:]}")
        return found.group(1)

    def render_state(self, job: str) -> str:
        """``TERMINAL:<verdict>`` once the job ended, else its state."""

        done = subprocess.run(["run-on", "state", job, "--json"], capture_output=True, text=True,
                              timeout=60)
        try:
            state = json.loads(done.stdout)
        except ValueError:
            return "unknown"
        return f"TERMINAL:{state.get('verdict')}" if state.get("state") == "TERMINAL" else \
            str(state.get("state"))

    def render_lines(self, job: str, repo: Path) -> list[dict[str, str]]:
        """Pull the job's lines file to the hub and read it."""

        host = (repo / ".compute-host").read_text(encoding="utf-8").strip()
        subprocess.run(["pull-outputs", host, job], capture_output=True, text=True, timeout=600)
        path = repo / "outputs" / job / RENDER_LINES
        if not path.is_file():
            raise FileNotFoundError(f"render job {job} left no {RENDER_LINES}")
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]

    def render_tidy(self, batch: str, job: str, repo: Path) -> None:
        """Remove the batch's staged attempts and the render job's tree (about 0.5 GiB) from the
        worker disk; run-on's reaper takes whatever this misses."""

        host = os.environ["MENAGERIE_RUNNER_HOST"]
        subprocess.run(["ssh", "-o", "BatchMode=yes", host,
                        f"rm -rf {shlex.quote(_worker_attempts(batch).rsplit('/', 1)[0])}"],
                       capture_output=True, timeout=120)
        compute = (repo / ".compute-host").read_text(encoding="utf-8").strip()
        subprocess.run(["run-on", "reap", "--ttl-hours", "0", "--failed-hours", "0", "--host",
                        compute, "--project", "menagerie", "--job", job, "--apply"],
                       capture_output=True, timeout=300)


def _worker_attempts(batch: str) -> str:
    return f"scratch/menagerie/axon/{batch}/attempts"  # relative to the worker home


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# --- one pass ----------------------------------------------------------------------------------


@dataclass
class Feed:
    """One pass's inputs."""

    state: dict[str, Any]
    data: DataLayout
    cfg: dict[str, Any]
    ops: Any
    repo: Path
    submit: bool  # full mode with the disk guard clear
    deadline: float = float("inf")
    held: str = ""  # why ``submit`` is off, for the status line
    clock: Callable[[], str] = now

    def room(self) -> bool:
        return time.monotonic() < self.deadline - STEP_ROOM_S


def step(feed: Feed) -> dict[str, Any]:
    """Collect, render and submit as the state calls for; return what this pass did."""

    out: dict[str, Any] = {}
    queued = feed.ops.queue()
    active = feed.state.get("active")
    running = next((s for j, _, s in queued if active and j == active.get("job")), None)
    if running:
        out.update(active=active["batch"], job_state=running)
    elif active and feed.room():
        out["collected"] = _collect(feed, active)
    out["renders"] = _renders(feed)
    if not feed.submit and feed.held:
        out["held"] = feed.held
    if feed.state.get("active") or not feed.cfg["enabled"] or not feed.submit:
        return out
    if queued:  # another Menagerie task holds the project's one slot: adopt ours, else wait
        out["waiting"] = _adopt(feed, queued)
        return out
    if feed.room():
        out["submitted"] = _submit(feed)
    return out


def _adopt(feed: Feed, queued: Sequence[tuple[str, str, str]]) -> str:
    """A batch this feeder wrote whose job is queued but not recorded (a pass ended mid-submit)."""

    root = _batches_root()
    for job, name, _ in queued:
        batch = root / name.removeprefix("menagerie-")
        if (batch / "batch.json").is_file() and (batch / "feeder").is_file():
            feed.state["active"] = {"batch": batch.name, "job": job, "adopted": feed.clock()}
            return f"adopted {name} ({job})"
    return f"the cluster queue holds {', '.join(n for _, n, _ in queued)}"


def _batches_root() -> Path:
    from menagerie.paths import work_dir

    return work_dir("axon")


def _submit(feed: Feed) -> dict[str, Any]:
    bands = rows(feed.data, feed.cfg, back=_back(feed.state))
    over = [r.id for r in bands["over"]]
    if over:
        _type_over(bands["over"], feed.data, feed.cfg)
    feed.state["waiting_over"] = sorted({*(feed.state.get("waiting_over") or []), *over})
    feed.state["ready"] = {k: len(v) for k, v in bands.items()}
    chosen = choose(bands, feed.cfg)
    if chosen is None:
        return {"none_ready": feed.state["ready"], **({"typed_over": over} if over else {})}
    kind, ids = chosen
    batch = feed.ops.batch(kind, ids, feed.data)
    (batch / "feeder").write_text(feed.clock() + "\n", encoding="utf-8")
    answer = feed.ops.submit(batch)
    job = job_id(answer)
    if job is None:
        raise RuntimeError(f"axon-submit gave no job id for {batch.name}: {answer[:200]}")
    feed.state["active"] = {"batch": batch.name, "job": job, "kind": kind, "ids": ids,
                            "submitted": feed.clock()}
    return {"batch": batch.name, "job": job, "kind": kind, "rows": len(ids),
            **({"typed_over": over} if over else {})}


def _back(state: Mapping[str, Any]) -> list[str]:
    return [key.split("@", 1)[0] for key, n in (state.get("tries") or {}).items() if 0 < n < TRIES]


def _collect(feed: Feed, active: Mapping[str, Any]) -> dict[str, Any]:
    batch = _batches_root() / active["batch"]
    meta = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    summary = feed.ops.collect(batch, feed.data)
    if not summary.get("done"):
        raise RuntimeError(f"{batch.name}: its job left the queue but nothing came back")
    table = {r["stable_id"]: r for r in summary.get("rows") or []}
    passed = [sid for sid in meta["ids"] if _cells(table.get(sid), "trace")[:1] == ["trace:pass"]]
    typed = _unfinished(feed, meta, table)
    if passed:
        feed.state.setdefault("renders", []).append({"batch": batch.name, "ids": passed})
    times = feed.ops.times(active["job"])
    _history(feed.state, {"batch": batch.name, "kind": meta["kind"], "rows": len(meta["ids"]),
                          "passed": len(passed), "unfinished": typed, "done": summary["done"],
                          **times, "collected": feed.clock()})
    feed.state["active"] = None
    out = {"batch": batch.name, "passed": passed, "unfinished": typed,
           "added": summary.get("added"), "blobs": summary.get("blobs")}
    try:
        feed.ops.tidy(batch, meta)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        out["tidy_error"] = str(exc)[:200]  # leftovers cost disk only; the next tidy takes them
    return out


def _unfinished(feed: Feed, meta: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Rows the batch did not finish go back once; the second time they are typed."""

    tries = feed.state.setdefault("tries", {})
    back, typed = [], []
    for sid in meta["ids"]:
        row = table.get(sid) or {}
        if not _not_finished(row):
            tries.pop(f"{sid}@{meta['revisions'].get(sid)}", None)
            continue
        key = f"{sid}@{meta['revisions'].get(sid)}"
        tries[key] = tries.get(key, 0) + 1
        if tries[key] >= TRIES:
            typed.append(sid)
            tries.pop(key)
        else:
            back.append(sid)
    if typed:
        _type_unfinished(typed, feed.data, meta["batch"])
    return {"back": back, "typed": typed}


def _cells(row: Mapping[str, Any] | None, key: str) -> list[str]:
    """A table cell's parts: an eval trace, then a train trace, joined by ``;``."""

    return str((row or {}).get(key) or "").split(";")


def _not_finished(row: Mapping[str, Any]) -> bool:
    if not row or row.get("trace") == "not_reached":
        return True
    cells = [x for k in ("forward", "forward_class", "trace_class") for x in _cells(row, k)]
    return any(x.startswith("infra") for x in cells)


def _type_unfinished(ids: Sequence[str], data: DataLayout, batch: str) -> None:
    """One typed trace row per row the cluster twice did not finish; ``drive`` settles it."""

    _type(data, UNFINISHED_CLASS,
          {sid: f"two axon batches ended without finishing it (last {batch})" for sid in ids})


def _type_over(over: Sequence[Row], data: DataLayout, cfg: Mapping[str, Any]) -> None:
    """One typed trace row per row past the largest allocation; it waits for a bigger one."""

    _type(data, OVER_CLASS, {r.id: f"projected peak {r.need} GiB is past the largest axon "
                                   f"allocation this feeder uses ({cfg['max_gib']} GiB of "
                                   f"giants-128g); it waits for a bigger one" for r in over})


def _type(data: DataLayout, cls: str, msgs: Mapping[str, str]) -> None:
    """A cluster trace row of class ``cls`` at the pin per id, decided without an attempt."""

    from menagerie.pipeline import ledger, runner
    from menagerie.pipeline.publish import torchlens_pin

    pin, made = torchlens_pin(), []
    for sid, msg in msgs.items():
        record = read_json(ModelRecord, data.model(sid))
        request = runner.request_for(record, "trace", caps=runner.default_caps("trace"))
        result = runner.not_run(request, cls, msg)
        made += [replace(r, host="cluster", tl=pin) for r in result.rows]
    ledger.append(made, data=data)


def _history(state: dict[str, Any], entry: Mapping[str, Any]) -> None:
    state["history"] = [*(state.get("history") or []), dict(entry)][-HISTORY_KEEP:]


def _renders(feed: Feed) -> list[dict[str, Any]]:
    """Start each collected batch's render job, and append the lines of those that ended."""

    report, kept = [], []
    for item in feed.state.get("renders") or []:
        try:
            item, line = _render_item(feed, dict(item))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            item, line = item, {"batch": item["batch"], "error": str(exc)[:200]}
        report.append(line)
        if not item.get("finished"):
            kept.append(item)
    feed.state["renders"] = kept
    return report


def _render_item(feed: Feed, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not item.get("job"):
        if not feed.room():
            return item, {"batch": item["batch"], "render": "waits for a pass with room"}
        item["job"] = feed.ops.render_submit(_batches_root() / item["batch"], item["ids"], feed.cfg)
        return item, {"batch": item["batch"], "render_job": item["job"]}
    state = feed.ops.render_state(item["job"])
    if not state.startswith("TERMINAL"):
        return item, {"batch": item["batch"], "render_job": item["job"], "state": state}
    if state != "TERMINAL:succeeded":  # refused or failed: start it again, at most RENDER_TRIES
        item["fails"], job = item.get("fails", 0) + 1, item.pop("job")
        item["finished"] = item["fails"] >= RENDER_TRIES
        return item, {"batch": item["batch"], "render_job": job, "state": state,
                      **({"given_up": "the drive's render step stays open"} if item["finished"]
                         else {})}
    lines = feed.ops.render_lines(item["job"], feed.repo)
    added = append_lines(lines, feed.data)
    feed.ops.render_tidy(item["batch"], item["job"], feed.repo)
    item["finished"] = True
    return item, {"batch": item["batch"], "render_job": item["job"], "state": state,
                  "added": added}


def append_lines(lines: Sequence[Mapping[str, str]], data: DataLayout) -> int:
    """Append the render job's ledger and store-index lines on main, each once."""

    by_path: dict[str, list[str]] = {}
    for item in lines:
        rel = item["path"]
        if not rel.startswith(("ledger/", "store-index/")) or ".." in rel:
            raise ValueError(f"render line for an unexpected file: {rel!r}")
        by_path.setdefault(rel, []).append(item["line"])
    return sum(_append_once(data.root / rel, new) for rel, new in by_path.items())


def _append_once(target: Path, new: Sequence[str]) -> int:
    from menagerie.pipeline import written

    have = set(target.read_text(encoding="utf-8").splitlines()) if target.exists() else set()
    fresh = [x for x in dict.fromkeys(new) if x and x not in have]
    if fresh:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write("".join(x + "\n" for x in fresh))
        written.appended(target, [x + "\n" for x in fresh])  # the tick commits exactly these
    return len(fresh)


# --- inside the render job ---------------------------------------------------------------------


def render(attempts: Path, ids: Sequence[str], out: Path) -> int:
    """Render the collected traces again on the worker (``flow.rerender``); write the ledger and
    store-index lines it appended, relative to the data root, to ``out``."""

    from menagerie import host_budget
    from menagerie.paths import work_dir
    from menagerie.pipeline import StoreLayout, flow, written
    from menagerie.pipeline.records import Caps

    data = DataLayout.from_env()
    caps = Caps(rss_mb=int(host_budget.declared_memory_gb() * 1024), timeout_s=1800,
                threads=host_budget.declared_cpus())
    with written.recording() as rec:
        flow.rerender(ids, [attempts], caps=caps, data=data, store=StoreLayout.from_env(),
                      attempts=work_dir("attempts"))
    root = data.root.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for path, lines in rec.lines.items():
            rel = Path(path).relative_to(root).as_posix()
            handle.writelines(json.dumps({"path": rel, "line": x.rstrip("\n")}) + "\n"
                              for x in lines)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3 or args[0] != "render":
        print("usage: python -m menagerie.pipeline.axon_feed render ATTEMPTS OUT ID...",
              file=sys.stderr)
        return 2
    return render(Path(args[1]), args[3:], Path(args[2]))


if __name__ == "__main__":
    raise SystemExit(main())
