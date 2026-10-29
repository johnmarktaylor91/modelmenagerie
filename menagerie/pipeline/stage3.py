"""Stage 3 as a worker verb: queue, serve and collect traces for recipes 2b accepted.

Owned by the stage3 lane (rung 2, fixes 7, 8, 9 and 10). ``queue`` runs on the hub and sends a
lane's members to the runner's spool only from the lane's terminal event, never from ``done``
(fix 9); committed records no live lane holds (rung 0 rows, catalog rows) are queued by id.
``serve`` runs in the runner job on the worker, beside ``runner serve``: each member runs
``flow.run_model`` through the runner spool (a forward, a trace sized from the forward's measured
peak with one retry, structure-only captures, renders). ``collect`` brings the ledger and
store-index lines back to the main checkout. ``queue`` and ``collect`` import neither torch nor
psutil.

The stage-3 spool sits inside a runner spool, so one name addresses both: ``stage3/new/`` holds
queued members (one JSON job each, with the record pinned at its recipe revision, the adapter's
text and the vendor sha), ``claimed/`` the running ones, ``done/`` the results (the table row and
every ledger and store-index line the member produced), ``collected/`` what the hub has taken,
and ``jobs/<name>/`` each member's own data root and attempt logs. The hub reaches a spool on
``MENAGERIE_RUNNER_HOST`` over ssh (a path under the remote home, default
``MENAGERIE_RUNNER_SPOOL``); with no host the spool is a local directory.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import (
    DataLayout, StoreLayout, now, read_json, recipe_revision, write_json, written,
)
from menagerie.pipeline.records import ModelRecord, Packet, RecordError, RunResult, from_dict, to_dict
# The GPU window is defined once, by the worker that refuses a gpu request outside it.
from menagerie.pipeline.worker import in_gpu_window as gpu_window_open

JOB_SCHEMA, DONE_SCHEMA = "stage3-job/1", "stage3-done/1"
# A trace-only job needs a consumer that honours trace_only and settled_forward. Consumers from
# before the repin read only JOB_SCHEMA and reject this one rather than run a fresh forward and
# an old-pin trace in its place.
TRACE_ONLY_SCHEMA = "stage3-job/2"
NAME = re.compile(r"m\d+-[0-9a-f]{12}-\d{8}t\d{6}z-[0-9a-f]{4}")
DEFAULT_SPOOL = "scratch/menagerie/spool"
LANE_VERBS = ("done", "giveup")  # a lane's events that leave a member with a recipe to trace
POLL_S = 1.0
REAP_S = 24 * 3600  # collected jobs and idle vendor trees are removed after this long
REPO = Path(__file__).resolve().parents[2]


# --- the spool, local or over ssh ------------------------------------------------------------


@dataclass(frozen=True)
class Spool:
    """A runner spool's stage-3 folders, on this machine (``host`` ``None``) or over ssh."""

    host: str | None
    root: str

    @classmethod
    def named(cls, spool: Path | str | None = None) -> Spool:
        root = str(spool or os.environ.get("MENAGERIE_RUNNER_SPOOL") or DEFAULT_SPOOL)
        return cls(os.environ.get("MENAGERIE_RUNNER_HOST") or None, root)

    def _sh(self, script: str, data: bytes | None = None) -> bytes:
        argv = ["ssh", self.host, script] if self.host else ["bash", "-c", script]
        done = subprocess.run(argv, input=data, capture_output=True, timeout=300,
                              cwd=None if self.host else Path.home())
        if done.returncode:
            raise OSError(f"stage3 spool {self}: {done.stderr.decode(errors='replace')[-300:]}")
        return done.stdout

    def _dir(self, name: str) -> str:
        return shlex.quote(f"{self.root}/stage3/{name}")

    def put(self, jobs: dict[str, bytes]) -> None:
        """Write job files into ``new/`` in one call; each appears whole (a rename)."""

        if not jobs:
            return
        incoming = self._dir(f".incoming-{secrets.token_hex(4)}")
        self._sh(f"mkdir -p {incoming} {self._dir('new')} && tar -xf - -C {incoming} && "
                 f"find {incoming} -name '*.json' -exec mv -t {self._dir('new')} {{}} + && "
                 f"rmdir {incoming}", data=_tar({f"{n}.json": b for n, b in jobs.items()}))

    def finished(self) -> list[str]:
        out = self._sh(f"ls {self._dir('done')} 2>/dev/null || true").decode().split()
        return sorted(n[:-5] for n in out if n.endswith(".json") and NAME.fullmatch(n[:-5]))

    def take(self, names: Sequence[str]) -> dict[str, bytes]:
        """The named done files, read in one call."""

        if not names:
            return {}
        listing = "".join(f"{n}.json\n" for n in names).encode()
        body = self._sh(f"tar -cf - -C {self._dir('done')} -T -", data=listing)
        return {Path(k).stem: v for k, v in _untar(body).items()}

    def mark(self, names: Sequence[str]) -> None:
        """Move the named done files to ``collected/`` in one call."""

        if not names:
            return
        kept, done = self._dir("collected"), self._dir("done")
        listing = "".join(f"{n}.json\n" for n in names).encode()
        self._sh(f"mkdir -p {kept} && cd {done} && xargs -r mv -t ../collected", data=listing)


def _tar(files: dict[str, bytes]) -> bytes:
    import io
    import tarfile

    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, body in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(body), 0o644, int(time.time())
            tar.addfile(info, io.BytesIO(body))
    return out.getvalue()


def _untar(body: bytes) -> dict[str, bytes]:
    import io
    import tarfile

    found = {}
    with tarfile.open(fileobj=io.BytesIO(body), mode="r") as tar:
        for member in tar.getmembers():
            handle = tar.extractfile(member) if member.isfile() else None
            if handle is not None and NAME.fullmatch(Path(member.name).stem):
                found[Path(member.name).name] = handle.read()
    return found


def _local(kind: str) -> Path:
    path = work_dir("stage3") / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- queue (hub) -----------------------------------------------------------------------------


def _lane_state(lane: str) -> str:
    from menagerie.pipeline import admission

    status = admission._status(lane)
    return status["run"]["state"] if status else "unknown"


def _event_members(event: dict[str, Any], data: DataLayout) -> list[str]:
    """The members one ``done`` or ``giveup`` event leaves to trace: those ``done`` accepted or
    rewrote, the give-up's own id, else (an event that names neither) the family's members."""

    if event["verb"] == "giveup" and event.get("id"):
        return [event["id"]]
    if "accepted" in event:
        return [*event["accepted"], *event.get("rewrote", ())]
    packet = data.packet(event["family"])
    return [m.id for m in read_json(Packet, packet).members] if packet.exists() else []


def _lane_members(lane: str, data: DataLayout, ids: Sequence[str]) -> list[str]:
    """The members a terminal lane's ``done`` and ``giveup`` events leave to trace."""

    state = _lane_state(lane)
    if state != "TERMINAL":
        raise RecordError(f"lane {lane} is {state}: stage 3 waits for its terminal event (fix 9)")
    events = work_dir("lanes") / lane / "events.jsonl"
    lines = events.read_text(encoding="utf-8").splitlines() if events.is_file() else []
    members = [m for e in map(json.loads, filter(None, lines))
               if e.get("verb") in LANE_VERBS and e.get("family")
               for m in _event_members(e, data)]
    if ids:
        outside = sorted(set(ids) - set(members))
        if outside:
            raise RecordError(f"{outside} are not members lane {lane} finished")
        return list(dict.fromkeys(ids))
    return list(dict.fromkeys(members))


def lane_holds(ids: Sequence[str], data: DataLayout) -> dict[str, str]:
    """The ids a live lane holds by id or family, each mapped to the holding lane's run."""

    from menagerie.pipeline import drive

    held = {k: lease["run"] for lease in drive.lanes(data=data) for k in lease.get("keys", ())}
    if not held:
        return {}
    taken = {}
    for model_id in ids:
        family = read_json(ModelRecord, data.model(model_id)).family
        taken.update({model_id: held[k] for k in (model_id, family) if k in held})
    return taken


def _committed(ids: Sequence[str], data: DataLayout) -> list[str]:
    """Committed members queued by id: refused while a live lane holds their family or id."""

    if not ids:
        raise RecordError("stage3 queue --committed needs the stable ids to queue")
    taken = lane_holds(ids, data)
    if taken:
        raise RecordError(f"live lanes hold {taken}; queue them from the lane's terminal event")
    return list(dict.fromkeys(ids))


def _adapter_bytes(record: ModelRecord, data: DataLayout) -> bytes | None:
    recipe = record.recipe
    assert recipe is not None
    if recipe.kind == "classic" and recipe.adapter:
        base = REPO / Path(*recipe.adapter.partition(":")[0].split("."))
        return (base / "__init__.py" if base.is_dir() else base.with_suffix(".py")).read_bytes()
    if recipe.kind == "adapter" and recipe.adapter:
        return (data.root / recipe.adapter).read_bytes()
    return None


def stale(record: ModelRecord, data: DataLayout) -> str | None:
    """Why a record's recipe revision no longer matches its inputs, or ``None``."""

    from menagerie.pipeline import envs

    if record.recipe is None:
        return "no recipe"
    try:
        lock = envs.lock_path(record.recipe.env).read_bytes()
        rev = recipe_revision(to_dict(record.recipe), _adapter_bytes(record, data), lock)
    except (OSError, RecordError) as exc:
        return f"revision inputs unreadable: {exc}"
    if rev != record.recipe.rev:
        return f"recipe rev {record.recipe.rev} is stale: its adapter or lock now gives {rev}"
    return None


def _job(record: ModelRecord, data: DataLayout, lane: str | None, *, timeout_s: int | None,
         memory_gib: float | None, plan: Any = None) -> dict[str, Any]:
    from dataclasses import asdict, replace

    from menagerie.pipeline import flow
    from menagerie.pipeline.records import CLUSTER_PLACEMENTS

    assert record.recipe is not None
    measured = plan.measured_gib if plan is not None else None
    if record.recipe.placement in CLUSTER_PLACEMENTS and measured and measured <= flow.CLUSTER_GIB:
        # A measured peak that fits beats the estimate that sent it off the worker.
        record = replace(record, recipe=replace(record.recipe,
                                                placement=flow.placement_for(measured)))
    stamp = now().replace("-", "").replace(":", "").lower()
    sizing_fields = asdict(plan) if plan is not None else None
    if sizing_fields is not None:
        # Live runners may still serve the previous revision. Default payloads keep its
        # Sizing constructor's exact field set until the conductor enables a feature.
        if sizing_fields["floor_gib"] == flow.TRACE_FLOOR_GIB:
            sizing_fields.pop("floor_gib")
        if not sizing_fields["big_runner"]:
            sizing_fields.pop("big_runner")
    return {"schema": JOB_SCHEMA, "name": f"{record.id}-{record.recipe.rev}-{stamp}-"
            f"{secrets.token_hex(2)}", "id": record.id, "family": record.family,
            "rev": record.recipe.rev, "lane": lane, "queued": now(), "record": to_dict(record),
            "adapter_src": flow.adapter_src(record, data), "vendor": record.recipe.vendor,
            "placement": record.recipe.placement, "part": "all", "timeout_s": timeout_s,
            "memory_gib": memory_gib, "sizing": sizing_fields}


def _pending(model_id: str, rev: str) -> str | None:
    """A sent, not yet collected job for the same member at the same revision."""

    collected = {p.name for p in _local("collected").glob("*.json")}
    found = sorted(p.stem for p in _local("sent").glob(f"{model_id}-{rev}-*.json")
                   if p.name not in collected)
    return found[-1] if found else None


def _traced(data: DataLayout) -> set[tuple[str, str]]:
    """(id, recipe revision) pairs with a non-``infra`` eval trace row at the TorchLens pin."""

    from menagerie.pipeline import ledger
    from menagerie.pipeline.publish import torchlens_pin
    from menagerie.pipeline.records import LedgerRow

    if not (data.root / "ledger").is_dir():
        return set()
    tl = torchlens_pin()
    return {(r.id, r.recipe_rev) for r in ledger.read_all(data=data)
            if isinstance(r, LedgerRow) and r.stage == "trace" and r.tl == tl
            and r.outcome != "infra" and r.mode in (None, "eval")}


def _skip(record: ModelRecord, data: DataLayout, traced: set[tuple[str, str]]) -> str | None:
    """Why a member is not queued: no recipe or a stale revision, already traced at the pin
    (``traced``; empty with ``again``), or a job already sent at this revision."""

    why = stale(record, data)
    if why is not None or record.recipe is None:
        return why or "no recipe"
    if (record.id, record.recipe.rev) in traced:
        return f"current: traced at recipe {record.recipe.rev} and the TorchLens pin (--again)"
    pending = _pending(record.id, record.recipe.rev)
    return f"already queued as {pending}" if pending else None


@dataclass
class Peaks:
    """Measured peaks in MB from the ledger, by stable id (eval rows, any revision, not infra):
    forward peaks, passing full trace peaks and ``memory_cap`` full trace peaks, with the time of
    each member's latest passing and ``memory_cap`` trace."""

    forward: dict[str, int]
    passed: dict[str, int]
    capped: dict[str, int]
    passed_at: dict[str, datetime] = field(default_factory=dict)
    capped_at: dict[str, datetime] = field(default_factory=dict)

    @classmethod
    def read(cls, data: DataLayout) -> Peaks:
        from menagerie.pipeline import ledger
        from menagerie.pipeline.records import LedgerRow

        found = cls({}, {}, {})
        rows = ledger.read_all(data=data) if (data.root / "ledger").is_dir() else []
        for row in rows:
            if isinstance(row, LedgerRow) and row.outcome != "infra" and row.mode != "train":
                found._take(row)
        return found

    def _take(self, row: Any) -> None:
        if row.stage == "forward":
            # the tree peak, the cap's basis, as flow.forward_peak_mb (no flow import here)
            _keep(self.forward, row.id, row.peak_rss_mb or row.forward_peak_rss_mb)
            return
        if row.stage != "trace" or row.capture not in (None, "full"):
            return
        if row.outcome == "pass":
            into, times = self.passed, self.passed_at
        elif (row.cls or "").startswith("memory_cap"):
            into, times = self.capped, self.capped_at
        else:
            return
        at = datetime.fromisoformat(row.at.replace("Z", "+00:00"))
        at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
        times[row.id] = max(times.get(row.id, at), at)
        _keep(into, row.id, row.peak_rss_mb)

    def outgrown(self, sid: str) -> bool:
        """A ``memory_cap`` after the member's latest passing trace, above that trace's peak: the
        passing peak (an older revision or pin, or a smaller input) no longer bounds the trace."""

        capped, at = self.capped.get(sid), self.capped_at.get(sid)
        passed_at = self.passed_at.get(sid)
        return bool(capped and at and capped > self.passed.get(sid, 0)
                    and (passed_at is None or at > passed_at))

    def ratio(self, ids: Iterable[str]) -> float | None:
        """The largest trace to forward ratio measured among ``ids`` (a ``memory_cap`` peak is a
        lower bound, so it counts too)."""

        found = [max(self.passed.get(i, 0), self.capped.get(i, 0)) / self.forward[i]
                 for i in ids if self.forward.get(i) and (i in self.passed or i in self.capped)]
        return round(max(found), 2) if found else None


def _keep(into: dict[str, int], sid: str, peak: int | None) -> None:
    if peak:
        into[sid] = max(into.get(sid, 0), peak)


def _families(data: DataLayout) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted((data.root / "models").glob("*/*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        found.setdefault(raw.get("family", ""), []).append(raw["id"])
    return found


def sizing(record: ModelRecord, peaks: Peaks, families: dict[str, list[str]], *,
           per_row_floor: bool = False) -> Any:
    """The member's trace sizing from what the ledger measured (fix 10, measured over estimated):
    its own passing peak, else its family's largest measured ratio, and never under a step past
    its own ``memory_cap``. A ``memory_cap`` newer and larger than the passing peak outranks it,
    so a row that outgrew its old peak is sized past the cap it broke (B's day size or G), not
    sent back to the size that failed."""

    from menagerie.pipeline import flow, runner

    sid, gib = record.id, 1 / 1024
    ratio = peaks.ratio(families.get(record.family) or [sid])
    floor = flow.ROW_FLOOR_GIB if per_row_floor else flow.TRACE_FLOOR_GIB
    if sid in peaks.passed and not peaks.outgrown(sid):
        measured = round(peaks.passed[sid] * gib * runner.PEAK_HEADROOM, 2)
        return flow.Sizing(measured_gib=measured, basis=f"its measured trace peak "
                           f"{peaks.passed[sid] * gib:.1f} GiB x {runner.PEAK_HEADROOM}",
                           floor_gib=floor)
    capped = peaks.capped.get(sid)
    at_least = round(capped * gib * flow.MEMORY_CAP_STEP, 2) if capped else None
    basis = (f"{record.family}'s measured ratio {ratio}" if ratio else
             f"{flow.TRACE_PEAK_RATIO}") + " x the forward's measured peak"
    if at_least:
        basis += f", at least {flow.MEMORY_CAP_STEP} x its memory_cap at {capped * gib:.1f} GiB"
    return flow.Sizing(ratio=ratio or flow.TRACE_PEAK_RATIO, at_least_gib=at_least, basis=basis,
                       floor_gib=floor)


def projected_peak(record: ModelRecord, peaks: Peaks, families: dict[str, list[str]]) -> float | None:
    """The queue and placement use the same measured forward, then recipe fallback."""

    forward = peaks.forward.get(record.id)
    fallback = record.recipe.memory_gib if record.recipe else None
    return sizing(record, peaks, families).project(forward / 1024 if forward else fallback)


def _queue_plan(record: ModelRecord, peaks: Peaks, families: dict[str, list[str]],
                cfg: Mapping[str, Any], target: Spool, memory_gib: float | None) -> tuple[ModelRecord, Any]:
    """Validate the big spool's admission and serialize the small spool's opt-in floor."""

    from dataclasses import replace

    from menagerie.pipeline import flow

    name = target.root.rstrip("/").rsplit("/", 1)[-1]
    plan = sizing(record, peaks, families, per_row_floor=cfg.get("per_row_floor", False) and name == "spool-B")
    if name != "spool-G":
        return record, plan
    projected = memory_gib or projected_peak(record, peaks, families)
    if projected is None or not flow.WORKER_GIB < projected <= flow.BIG_GIB:
        raise ValueError(f"{record.id}: spool-G requires a projection in "
                         f"({flow.WORKER_GIB:g}, {flow.BIG_GIB:g}] GiB")
    assert record.recipe is not None
    return replace(record, recipe=replace(record.recipe, placement="single-worker")), replace(plan, big_runner=True)


def placement(model_id: str, *, data: DataLayout, big_runner: bool | None = None) -> dict[str, Any]:
    """Where the placement rule puts a member's full trace, from the ledger alone (for the
    harvest's and the cluster's records): projected GiB, placement and its basis."""

    from menagerie.pipeline import flow, tick

    enabled = tick.config().get("big_runner", False) if big_runner is None else big_runner
    record, peaks = read_json(ModelRecord, data.model(model_id)), Peaks.read(data)
    plan = sizing(record, peaks, _families(data))
    forward = peaks.forward.get(model_id)
    projected = plan.project(forward / 1024 if forward else
                             record.recipe.memory_gib if record.recipe else None)
    where = flow.placement_for(projected, big_runner=enabled) if projected else None
    return {"id": model_id, "projected_gib": projected and round(projected, 2),
            "placement": where, "basis": plan.basis,
            "spool": "scratch/menagerie/spool-G" if enabled and projected and
            flow.WORKER_GIB < projected <= flow.BIG_GIB else None}


def _drain(vendor: str | None) -> None:
    """Move a vendor blob staged on this machine to the worker store (as the bridge does)."""

    if vendor is None or not os.environ.get("MENAGERIE_STORE_REMOTE"):
        return
    from menagerie.pipeline import store

    layout = StoreLayout.from_env()
    if layout.blob(vendor).exists():
        store.drain([vendor], store=layout, data=DataLayout.from_env())


def queue(lane: str | None, *, data: DataLayout, ids: Sequence[str] = (),
          spool: Path | str | None = None, timeout_s: int | None = None,
          memory_gib: float | None = None, again: bool = False,
          trace_only: bool = False) -> list[str]:
    """Queue stage 3 for the members a finished 2b lane accepted, or for committed ids.

    Inputs: the 2b lane's run id (refused unless fleet-run calls the lane terminal), or ``None``
    with the ``ids`` of committed records no live lane holds; ``ids`` narrows a lane's members;
    the runner spool (``Spool.named``); a request's own trace timeout and memory cap. Each job
    pins the record at its recipe revision with its adapter's text and vendor sha. A member with
    no recipe, a stale revision, a trace row at its revision and the TorchLens pin (unless
    ``again``) or a job already sent at the same revision is skipped and named on stderr
    (a family's earlier members, rewritten or not, are queued only when their revision moved).
    Returns: the queued job names. Side effects: job files in the spool, a copy under
    ``work_dir("stage3")/sent``, one ``stage3`` event per member and any vendor blob drained.
    Ordinary queues reuse settled forwards at the recipe revision; ``again`` requests a
    fresh forward too, unless ``trace_only`` explicitly preserves the existing forward.
    """

    from menagerie.pipeline import drive, tick

    members = _lane_members(lane, data, ids) if lane else _committed(ids, data)
    target, jobs = Spool.named(spool), {}
    cfg = tick.config()
    big = target.root.rstrip("/").endswith("/spool-G")
    if big and not cfg.get("big_runner", False):
        raise ValueError("spool-G requires big_runner in config.json")
    traced = set() if again else _traced(data)
    peaks, families = Peaks.read(data), _families(data)
    forwards = _settled_forwards(data, again=again, trace_only=trace_only)
    for model_id in members:
        record = read_json(ModelRecord, data.model(model_id))
        why = _skip(record, data, traced)
        if why is not None:
            print(f"{model_id}: not queued: {why}", file=sys.stderr)
            continue
        record, plan = _queue_plan(record, peaks, families, cfg, target, memory_gib)
        job = _job(record, data, lane, timeout_s=timeout_s, memory_gib=memory_gib, plan=plan)
        _attach_forward(job, record, forwards, trace_only)
        job["seq"] = len(jobs)  # the order queued: the server starts jobs in this order
        _drain(job["vendor"])
        jobs[job["name"]] = job
    bodies = {n: (json.dumps(j, ensure_ascii=False, sort_keys=True) + "\n").encode()
              for n, j in jobs.items()}
    target.put(bodies)
    for name, job in jobs.items():
        (_local("sent") / f"{name}.json").write_bytes(bodies[name])
        drive.event(lane or os.environ.get("FLEET_RUN_ID") or "manual", job["family"], "stage3",
                    id=job["id"], rev=job["rev"], request=name, spool=target.root)
    return list(jobs)


def _attach_forward(job: dict[str, Any], record: ModelRecord, forwards: dict,
                    trace_only: bool) -> None:
    """Settled forwards belong to recipes and remain reusable after a pin change."""

    if trace_only:
        job["trace_only"] = True
        job["schema"] = TRACE_ONLY_SCHEMA
    forward = forwards.get((record.id, record.recipe.rev))
    if forward is None:
        return
    job["settled_forward"] = to_dict(forward)


def _settled_forwards(data: DataLayout, *, again: bool = False,
                      trace_only: bool = False) -> dict[tuple[str, str], Any]:
    """Reusable forwards at each recipe, unless an ordinary queue explicitly retries them."""

    from menagerie.pipeline import ledger
    from menagerie.pipeline.records import RunResult

    if again and not trace_only:
        return {}
    found = {}
    records: dict[str, ModelRecord] = {}
    for row in ledger.read_all(data=data):
        if not _forward_row(row):
            continue
        path = data.model(row.id)
        if not path.is_file():
            continue  # historical rows for removed models do not participate in this queue
        if row.id not in records:
            records[row.id] = read_json(ModelRecord, path)
        record = records[row.id]
        modes = record.recipe.modes if record.recipe and record.recipe.rev == row.recipe_rev else None
        found[(row.id, row.recipe_rev)] = RunResult(
            request=f"{row.id}-forward-eval-reused-0000", id=row.id, mode="forward",
            outcome=row.outcome, started=row.at, ended=row.at, host=row.host, cls=row.cls,
            modes=modes, rows=(row,))
    return found


def _forward_row(row: Any) -> bool:
    """Only settled eval forwards can supply recipe evidence to a queued job."""

    from menagerie.pipeline.records import LedgerRow

    return (isinstance(row, LedgerRow) and row.stage == "forward"
            and row.outcome in ("pass", "fail") and row.mode in (None, "eval"))


# --- serve (worker) --------------------------------------------------------------------------


def _dirs(spool: Path) -> dict[str, Path]:
    names = ("new", "claimed", "done", "collected", "jobs", "rejected")
    dirs = {name: Path(spool) / "stage3" / name for name in names}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _read_job(path: Path) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8"))
    # Essential complexity: every header invariant fails closed at the spool's intake boundary.
    schema = TRACE_ONLY_SCHEMA if job.get("trace_only") is True else JOB_SCHEMA
    if job.get("schema") != schema or job.get("name") != path.stem or not NAME.fullmatch(
            path.stem):
        raise RecordError(f"{path.name} is not a stage-3 job naming itself")
    record = from_dict(ModelRecord, job["record"])
    if record.id != job["id"] or record.recipe is None or record.recipe.rev != job["rev"]:
        raise RecordError(f"{path.name}: its record is not {job['id']} at recipe {job['rev']}")
    if not isinstance(job.get("queued"), str):
        raise RecordError(f"{path.name} has no queue time")
    _checked_forward(job, record)
    return {k: job[k] for k in ("name", "id", "rev", "queued")} | {
        "seq": job.get("seq", 0), "placement": job.get("placement"), "part": job.get("part", "all"),
        "sizing": job.get("sizing") or {}}


def _checked_forward(job: dict[str, Any], record: ModelRecord) -> RunResult | None:
    """A reused forward must name this exact model and recipe, with no payload files."""

    # Essential complexity: this boundary keeps independent reuse invariants together.
    if not isinstance(job.get("trace_only", False), bool):
        raise RecordError("trace_only must be a boolean")
    if "settled_forward" not in job:
        return None
    result = from_dict(RunResult, job["settled_forward"])
    if result.id != record.id or result.mode != "forward":
        raise RecordError("settled forward does not belong to this job")
    if result.files or len(result.rows) != 1:
        raise RecordError("settled forward must have one ledger row and no payload files")
    row = result.rows[0]
    if row.mode not in (None, "eval"):
        raise RecordError("settled forward must be an eval-mode row")
    if row.stage != "forward" or row.id != record.id or row.recipe_rev != record.recipe.rev:
        raise RecordError("settled forward row is not at this model's recipe")
    if result.outcome != row.outcome or row.outcome not in ("pass", "fail"):
        raise RecordError("settled forward must have a consistent non-infrastructure outcome")
    return result


def _held(job: dict[str, Any]) -> bool:
    """A GPU-only job waits, unclaimed, while the GPU window is shut."""

    if os.environ.get("MENAGERIE_BIG_RUNNER") == "1" or (job.get("sizing") or {}).get("big_runner"):
        from menagerie.pipeline.runner_job import night_open

        if not night_open():
            return True
    return job.get("part") == "gpu" and not gpu_window_open()


def _write_done(dirs: dict[str, Path], job: dict[str, Any], **fields: Any) -> None:
    body = {"schema": DONE_SCHEMA, "name": job["name"], "id": job["id"], "rev": job["rev"],
            "lane": job.get("lane"), "queued": job.get("queued"), "ended": now(), **fields}
    tmp = dirs["done"] / f".{job['name']}.part"
    tmp.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, dirs["done"] / f"{job['name']}.json")


def _queue_stamp(job: dict[str, Any]) -> tuple[str, str]:
    """When a job was made: the stamp ``queue`` wrote into its name (``NAME``), then the name."""

    return job["name"].split("-")[2], job["name"]


def _superseded(job: dict[str, Any], waiting: Iterable[dict[str, Any]]) -> str | None:
    """A later job for the same member at another revision makes this one stale.

    Later means made later (the name's stamp), never the ``queued`` field: that field only
    orders the spool, and moving a job to the head rewrites it to an earlier time.
    """

    later = [w["name"] for w in waiting if w["id"] == job["id"] and w["rev"] != job["rev"]
             and _queue_stamp(w) > _queue_stamp(job)]
    return max(later, key=lambda n: _queue_stamp({"name": n})) if later else None


def _lines(root: Path) -> list[str]:
    return [line for shard in sorted(root.glob("*.jsonl"))
            for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()]


def _keep_logs(results: Sequence[Any], spool: Path, job_dir: Path) -> None:
    """Move each attempt's log and result into the job's folder; delete the attempt and its
    runner result (the saves are in the store by now)."""

    for result in results:
        attempt = Path(spool) / "attempts" / result.request
        kept = job_dir / "attempts" / result.request
        kept.mkdir(parents=True, exist_ok=True)
        for name in ("attempt.log", "result.json", "request.json"):
            if (attempt / name).is_file():
                shutil.copy2(attempt / name, kept / name)
        shutil.rmtree(attempt, ignore_errors=True)
        (Path(spool) / "done" / f"{result.request}.json").unlink(missing_ok=True)


def run_job(job: dict[str, Any], *, spool: Path, store: StoreLayout) -> dict[str, Any]:
    """Run one member's stage 3 through the runner spool; return the done file's fields."""

    from menagerie.pipeline import flow, ledger

    job_dir = _dirs(spool)["jobs"] / job["name"]
    data = DataLayout(job_dir / "data")
    record = from_dict(ModelRecord, job["record"])
    write_json(record, data.model(record.id))
    if job.get("adapter_src") is not None and record.recipe and record.recipe.adapter:
        (data.root / record.recipe.adapter).parent.mkdir(parents=True, exist_ok=True)
        (data.root / record.recipe.adapter).write_text(job["adapter_src"], encoding="utf-8")
    part = job.get("part", "all")
    if part == "all" and record.recipe and record.recipe.placement == "gpu-window" \
            and not gpu_window_open():
        part = "cpu"  # the structure-only capture now, the GPU attempt held for the window
    plan = flow.Sizing(**job["sizing"]) if job.get("sizing") else None
    outcome = flow.run_model(record, caps=None, store=store, data=data, attempts=job_dir,
                             spool=spool, timeout_s=job.get("timeout_s"),
                             memory_gib=job.get("memory_gib"), part=part, sizing=plan,
                             settled_forward=_checked_forward(job, record),
                             trace_only=job.get("trace_only", False))
    if outcome.rows:
        ledger.append(outcome.rows, data=data)
    _keep_logs(outcome.results, spool, job_dir)
    held = _hold_gpu(job, spool) if part == "cpu" or outcome.held else None
    return {"outcome": "done", "placement": outcome.placement, "gpu_held_as": held,
            "table": flow.table_row(record, outcome), "ledger": _lines(data.root / "ledger"),
            "store_index": _lines(data.root / "store-index")}


def _hold_gpu(job: dict[str, Any], spool: Path) -> str:
    """Queue the member's GPU attempt as its own job, held until the GPU window opens."""

    stamp = now().replace("-", "").replace(":", "").lower()
    name = f"{job['id']}-{job['rev']}-{stamp}-{secrets.token_hex(2)}"
    follow = {**job, "name": name, "part": "gpu", "queued": now(), "after": job["name"]}
    new = _dirs(spool)["new"]
    tmp = new / f".{name}.part"
    tmp.write_text(json.dumps(follow, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, new / f"{name}.json")
    return name


def _serve_one(path: Path, dirs: dict[str, Path], spool: Path, store: StoreLayout) -> None:
    job = json.loads(path.read_text(encoding="utf-8"))
    try:
        fields = run_job(job, spool=spool, store=store)
    except Exception as exc:  # the serve loop's own fault: never lose the member, never blame it
        fields = {"outcome": "error", "why": repr(exc)[:400]}
    _write_done(dirs, job, **fields)
    path.unlink(missing_ok=True)


def _claim(dirs: dict[str, Path], seen: dict[str, dict[str, Any]]) -> list[Path]:
    """The waiting jobs that may run, in queue order; refuse stale and unreadable ones.

    ``seen`` caches each job's header (name, id, revision, queue time and order) by file name.
    """

    paths = {p.name: p for p in dirs["new"].glob("*.json")}
    for name in set(seen) - set(paths):
        del seen[name]
    for name, path in paths.items():
        if name not in seen:
            try:
                seen[name] = _read_job(path)
            except (RecordError, ValueError, KeyError, OSError):
                os.replace(path, dirs["rejected"] / name)
    jobs = sorted(seen.values(), key=lambda j: (j["queued"], j["seq"], j["name"]))
    ready = []
    for job in jobs:
        if _held(job):
            continue
        later = _superseded(job, jobs)
        path = paths[f"{job['name']}.json"]
        if later is not None:
            _write_done(dirs, job, outcome="refused", why=f"stale: superseded by {later}")
            path.unlink(missing_ok=True)
            continue
        ready.append(path)
    return ready


def _reap(dirs: dict[str, Path]) -> None:
    """Delete collected jobs' folders and idle vendor trees after ``REAP_S``."""

    from menagerie.pipeline import runner

    for kept in dirs["collected"].glob("*.json"):
        if time.time() - kept.stat().st_mtime > REAP_S:
            shutil.rmtree(dirs["jobs"] / kept.stem, ignore_errors=True)
            kept.unlink(missing_ok=True)
    runner.reap_vendor(idle_s=REAP_S)


def serve(*, spool: Path, workers: int | None = None, once: bool = False) -> None:
    """Serve the stage-3 spool inside the runner job.

    Inputs: the runner spool (its ``stage3/`` folders; ``runner serve`` must serve the same
    spool), the number of members run at once (default: the runner's worker count from the job's
    declaration, so the runner always has work) and ``once`` (drain, then exit). Per member:
    ``flow.run_model`` through the runner. A job a later job for the same member at another
    revision supersedes is refused as stale; a dead server's claims are queued again; nothing is
    claimed while the spool's runner heartbeat is stale. A ``gpu-window`` member outside the GPU
    window gets its CPU capture now and its GPU attempt as a second job (``part: gpu``) that stays
    unclaimed until the window opens; ``once`` returns when only such jobs are left. Returns:
    nothing. Side effects: done files, each job's folder, attempts reaped into it, and collected
    jobs and idle vendor trees removed after ``REAP_S``.
    """

    from menagerie import host_budget
    from menagerie.pipeline import runner

    workers = workers or min(runner.DEFAULT_WORKERS, host_budget.declared_cpus())
    dirs, store = _dirs(spool), StoreLayout.from_env()
    for claimed in dirs["claimed"].glob("*.json"):  # a dead server's: run them again
        os.replace(claimed, dirs["new"] / claimed.name)
    running: dict[str, threading.Thread] = {}
    seen: dict[str, dict[str, Any]] = {}
    while True:
        (Path(spool) / "stage3" / "serving").touch()
        _start(dirs, seen, running, workers=workers, spool=Path(spool), store=store)
        if once and not running and all(_held(j) for j in seen.values()) and len(seen) == len(
                list(dirs["new"].glob("*.json"))):
            return  # drained: what is left waits for the GPU window
        _reap(dirs)
        time.sleep(POLL_S)


def _start(dirs: dict[str, Path], seen: dict[str, dict[str, Any]],
           running: dict[str, threading.Thread], *, workers: int, spool: Path,
           store: StoreLayout) -> None:
    """Forget finished members; start waiting ones while a slot is free and the runner serves."""

    from menagerie.pipeline import disk_guard, runner, runner_job

    for name in [n for n, t in running.items() if not t.is_alive()]:
        del running[name]
    # no runner: wait for it; below the disk guard's pause line (jobs the tick started): idle;
    # drained by the tick before a resize: the claimed jobs finish, nothing new starts
    ready = runner.serving(spool) and disk_guard.claim_ok(store.root) and not (
        Path(spool) / runner_job.DRAIN).exists()
    free = workers - len(running) if ready else 0
    for path in _claim(dirs, seen)[:free] if free > 0 else ():
        claimed = dirs["claimed"] / path.name
        os.replace(path, claimed)
        running[path.stem] = threading.Thread(target=_serve_one, daemon=True,
                                              args=(claimed, dirs, spool, store))
        running[path.stem].start()


# --- collect (hub) ---------------------------------------------------------------------------


def _append_new(path: Path, lines: Sequence[str], key: Any) -> int:
    """Append the lines whose key the file lacks, under one lock; return how many."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            have = {key(line) for line in handle.read().splitlines() if line.strip()}
            new = [line for line in dict.fromkeys(lines) if key(line) not in have]
            handle.seek(0, os.SEEK_END)
            handle.write("".join(line + "\n" for line in new))
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    written.appended(path, [line + "\n" for line in new])  # the tick commits exactly these
    return len(new)


def _shard_of(line: str, field: str) -> str:
    return str(json.loads(line)[field])[:7]


def _by_month(lines: Sequence[str], field: str) -> dict[str, list[str]]:
    months: dict[str, list[str]] = {}
    for line in lines:
        months.setdefault(str(json.loads(line)[field])[:7], []).append(line)
    return months


def _indexed(data: DataLayout) -> set[str]:
    shards = sorted((data.root / "store-index").glob("*.jsonl"))
    return {json.loads(line)["sha256"] for shard in shards
            for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()}


def _append_lines(results: Sequence[dict[str, Any]], data: DataLayout) -> dict[str, int]:
    """Append every result's lines, one locked append per shard: ledger lines unless the same
    line is there, store-index lines unless their sha is already indexed."""

    from menagerie.pipeline.records import parse_ledger_line

    ledger = [line for done in results for line in done.get("ledger") or ()]
    for line in ledger:
        parse_ledger_line(json.loads(line))  # refuse a malformed row before anything is written
    added = {f"ledger/{month}": _append_new(data.ledger(month), batch, lambda s: s)
             for month, batch in sorted(_by_month(ledger, "at").items())}
    known = _indexed(data)
    fresh = [line for done in results for line in done.get("store_index") or ()
             if json.loads(line)["sha256"] not in known]
    for month, batch in sorted(_by_month(fresh, "fetched").items()):
        added[f"store-index/{month}"] = _append_new(
            data.store_index(month), batch, lambda s: json.loads(s)["sha256"])
    return added


def _summary(done: dict[str, Any], data: DataLayout) -> dict[str, Any]:
    path = data.model(done["id"])
    current = read_json(ModelRecord, path) if path.exists() else None
    moved = current is None or current.recipe is None or current.recipe.rev != done["rev"]
    table = done.get("table") or {}
    return {"name": done["name"], "id": done["id"], "outcome": done["outcome"],
            "why": done.get("why"), "stale": moved, "rows": len(done.get("ledger") or ()),
            "placement": done.get("placement"), "gpu_held_as": done.get("gpu_held_as"),
            **{k: table.get(k) for k in ("forward", "forward_class", "trace", "trace_class",
                                         "render", "n_ops")}}


def collect(*, data: DataLayout, spool: Path | str | None = None) -> list[str]:
    """Bring finished stage-3 results to the main checkout.

    Inputs: the data root to append to and the runner spool (``Spool.named``). Every done file
    is read in one call; their ledger lines not already in their shard and their store-index
    lines whose sha is not yet indexed are appended (one locked append per shard); then the
    files are kept under ``work_dir("stage3")/collected`` and marked collected on the spool, so
    a collect cut short appends nothing twice when run again. A member whose committed recipe
    has moved on since it was queued is reported ``stale`` (its rows, at the old revision, never
    decide). Returns: the stable ids with rows. Prints one JSON line per result and one total.
    """

    source = Spool.named(spool)
    bodies = source.take(source.finished())
    results = []
    for name, body in sorted(bodies.items()):
        done = json.loads(body)
        if done.get("schema") != DONE_SCHEMA or done.get("name") != name:
            raise RecordError(f"stage-3 result {name} is not a done file naming itself")
        results.append(done)
    added = _append_lines(results, data)
    for done in results:
        (_local("collected") / f"{done['name']}.json").write_bytes(bodies[done["name"]])
        print(json.dumps(_summary(done, data), default=str), flush=True)
    source.mark([done["name"] for done in results])
    print(json.dumps({"collected": len(results), "added": added}), flush=True)
    return list(dict.fromkeys(d["id"] for d in results if d.get("ledger")))
