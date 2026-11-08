"""The tick's ``full``-mode steps (lane r3-tick): requeue, stage-3 queueing, lane top-up, trickle.

``full`` mode starts after C1 drains and JMT's go (BUILD-PLAN r3 section 1); until then the tick
runs in ``collect`` mode and none of this runs. Each step reads the worklist, the rows rung 3
works on in priority order: ``worklist.txt`` in the tick folder (or ``config.json``'s
``worklist``), one ``<id> [cohort]`` line per row (a stable id, then an optional cohort tag such as
``r3`` or ``r4-github`` from ``worklist build``; ``worklist`` reads the id), ``#`` comments allowed,
written by the lanes that plan the work (``residual plan``, ``census draw``, the re-pin list,
``worklist build``). No worklist: nothing is queued or
launched. Each step takes ``(data, cfg, dry_run=...)`` and returns what it did or would do.

The worklist's next steps come from ``drive.next_steps``, or, with ``config.json``'s
``counts_file`` true (default off; ``MENAGERIE_TICK_COUNTS=1`` turns it on for one process, the
trial's dry run), from the worker's counts file (``counts.steps_for``) when it is fresh, with the
in-process load as the fallback.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, read_json, read_json_cached
from menagerie.pipeline.records import ModelRecord

RUN_STEPS = ("forward", "trace")


def worklist(cfg: Mapping[str, Any]) -> list[str]:
    path = Path(cfg.get("worklist") or work_dir("tick") / "worklist.txt").expanduser()
    if not path.is_file():
        return []
    lines = (x.split("#", 1)[0].strip() for x in path.read_text(encoding="utf-8").splitlines())
    return list(dict.fromkeys(x.split()[0] for x in lines if x))


def requeue(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Start queued lanes that admission now admits (``admission.requeue``)."""

    from menagerie.pipeline import admission

    del data, cfg
    if dry_run:
        return {"queued": len(admission.queued())}
    return {"requeued": admission.requeue()}


def _steps(ids: Sequence[str], data: DataLayout, cfg: Mapping[str, Any]) -> dict[str, str]:
    """Each worklist id's next step: the counts file when switched on and fresh, else in-process."""

    from menagerie.pipeline import counts, drive

    if cfg.get("counts_file") or os.environ.get("MENAGERIE_TICK_COUNTS") == "1":
        found = counts.steps_for(ids, data=data)
        if found is not None:
            return found
    return {s["id"]: s["step"] for s in drive.next_steps(ids, data=data)}


def _spent_2b(steps: Mapping[str, str], family: Mapping[str, Any], data: DataLayout) -> list[str]:
    """Families whose last 2b lane ended without touching the rows that sent it.

    A family is spent when its ``BRIEF.result.md`` is newer than its ``BRIEF.md`` (the lane
    ended), no ``deferred.json`` asks for a new session, and neither its packet nor any of its
    rows still at ``2b`` changed since that brief was written. Relaunching it would only repeat
    that lane (re-running the members it can already build), so ``top_up`` holds it until one of
    those files changes. A lane that typed an ``infra`` give-up rewrote the row, so it is retried.
    """

    waiting: dict[str, list[str]] = {}
    for model_id, step in steps.items():
        if step == "2b" and family.get(model_id):
            waiting.setdefault(family[model_id], []).append(model_id)
    spent = []
    for own, members in waiting.items():
        lane = work_dir("2b") / own
        brief, result = lane / "BRIEF.md", lane / "BRIEF.result.md"
        if not result.is_file() or not brief.is_file() or (lane / "deferred.json").exists():
            continue
        started = brief.stat().st_mtime
        changed = [data.packet(own), *(data.model(m) for m in members)]
        if result.stat().st_mtime >= started and all(
                not p.exists() or p.stat().st_mtime < started for p in changed) \
                and not _unbriefed_dispute(members, data):
            spent.append(own)
    return sorted(spent)


def _unbriefed_dispute(members: Sequence[str], data: DataLayout) -> bool:
    """Whether a member waits for a disputed repair round no brief has carried yet
    (``drive.dispute``, ``drive.briefed``): its next lane differs from the last one."""

    from menagerie.pipeline import drive
    from menagerie.pipeline.records import RecordError

    files = drive._load(data, ids=members)
    for model_id in members:
        try:
            record = files.record(model_id)
        except RecordError:  # unreadable: no dispute can be judged, the spent check stands
            continue
        if record is None or record.recipe is None or record.give_up is not None:
            continue
        found = drive.dispute(model_id, record, files.rows.get(model_id, []),
                              files.packet(record.family), data)
        if found is not None and found[1] and \
                drive.briefed(record.family, model_id, record.recipe.rev) is None:
            return True
    return False


def _hold_spent(steps: Mapping[str, str], family: Mapping[str, Any], data: DataLayout
                ) -> tuple[dict[str, str], list[str]]:
    """The steps with a spent family's ``2b`` rows read ``2b:spent``, and those families."""

    spent = _spent_2b(steps, family, data)
    held = {i: "2b:spent" for i, s in steps.items() if s == "2b" and family.get(i) in spent}
    return {**steps, **held}, spent


# The role the tick launches 2b lanes on; a repair round stays on it (``implement.launch``).
REPAIR_ROLE = "model-implementation"


def _hold_repairs(steps: Mapping[str, str], family: Mapping[str, Any], data: DataLayout,
                  cfg: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Unless 2b may take every hub slot (``2b_on_hub`` true), the steps with a repair family's
    ``2b`` rows read ``2b:repair-seat``, and those families. A repair round stays on the role it
    is launched with (``implement.launch``; the tick launches ``model-implementation``), and the
    worker runtime serves Claude alone (``mini_placement._runtime``). While that role resolves to
    another harness, placing the family only spends a worker slot on the same refusal every pass,
    ahead of the families behind it. They start on the hub instead (``_repair_hub``). A capped
    ``{"lanes": N}`` holds them too: its 2b placements fill the workers first, so the 7 repair
    families of r3 took both Macs' slots and 3 of the trace host's every pass (3 Nov 2026)."""

    from menagerie.pipeline import admission, mini_placement

    if mini_placement.hub_2b_lanes(cfg) is None:
        return dict(steps), []
    repairs = _repair_families(sorted({family[i] for i, s in steps.items()
                                       if s == "2b" and family.get(i)}), data)
    if not repairs or admission._resolve(
            REPAIR_ROLE, harness="claude").get("harness") == "claude":
        return dict(steps), []
    held = {i: "2b:repair-seat" for i, s in steps.items() if s == "2b" and family.get(i) in repairs}
    return {**steps, **held}, repairs


def _repair_hub(repairs: Sequence[str], held: set[str], room: int) -> tuple[list[str], str | None]:
    """The repair families that start on the hub this pass, and why none can: not held by a live
    or queued lane, at most ``room`` (the hub slots 2a leaves and the 2b cap's room), and only
    while the Codex gate admits, so a pass without headroom keeps their 2b cap slots for the
    worker lanes."""

    from menagerie.pipeline import admission

    waiting = [f for f in repairs if f not in held]
    chosen = waiting[:max(0, room)]
    if not chosen:
        return [], ("no hub slot or 2b cap room for a repair lane" if waiting else None)
    try:
        if admission._resolve(REPAIR_ROLE).get("harness") == "codex":
            admission._codex_gate(REPAIR_ROLE, "2b")
    except admission.Refused as refused:
        return [], str(refused.result.get("why"))
    return chosen, None


def _launch_repairs(families: Sequence[str], data: DataLayout) -> list[str]:
    """Repair rounds on the hub, on ``REPAIR_ROLE`` alone and never queued: a refusal (that seat
    has no headroom) is held, and the next pass asks again. A different lab from the recipe's
    author is the point of the repair, so it never falls back to the Claude seat (Crawler,
    2026-10-30)."""

    from menagerie.pipeline import admission, implement

    runs = []
    for family in families:
        if not _launch_open():
            runs.append(LAUNCH_SPENT)
            continue
        try:
            runs.append(implement.launch(family, role=REPAIR_ROLE, data=data, queue=False))
        except admission.Refused as refused:
            runs.append(f"held: {refused.result.get('why')}")
    return runs


def _repair_families(families: Sequence[str], data: DataLayout) -> list[str]:
    """The families whose packet has a member a review called a fabrication (a repair round)."""

    from menagerie.pipeline import implement, ledger
    from menagerie.pipeline.records import Packet

    rows = ledger.read_containing('"review"', data=data)
    return [f for f in families if data.packet(f).is_file()
            and implement.fabrications(read_json_cached(Packet, data.packet(f)), data, rows=rows)]


def _not_pending(ids: Sequence[str], data: DataLayout) -> list[str]:
    """The ids ``stage3.queue`` would take: a recipe, no job sent at its revision, and no eval
    trace at its revision and the TorchLens pin. Filtering the traced ones here keeps them from
    filling the queue's room every pass only to be skipped there."""

    from menagerie.pipeline import stage3

    traced, open_ = stage3._traced(data), stage3.open_jobs()
    out = []
    for model_id in ids:
        path = data.model(model_id)
        record = read_json(ModelRecord, path) if path.exists() else None
        if record is None or record.recipe is None or (model_id, record.recipe.rev) in traced:
            continue
        if not stage3._pending(model_id, record.recipe.rev, open_):  # same package
            out.append(model_id)
    return out


def _waiting(spool: str) -> int:
    from menagerie.pipeline import stage3

    source = stage3.Spool.named(spool)
    listing = source._sh(f"ls {source._dir('new')} 2>/dev/null || true")
    return sum(1 for n in listing.decode().split() if n.endswith(".json"))


def stage3_queue(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Queue worklist rows whose next step is a forward pass or trace, up to the spool's depth."""

    from menagerie.pipeline import stage3

    ids = worklist(cfg)
    cohorts = _cohorts(cfg)
    stopped = set(cfg.get("stopped_cohorts", ()))
    ids = [i for i in ids if cohorts[i] not in stopped]
    if not ids and not cfg.get("big_runner", False):
        return {"skipped": "no worklist"}
    want = _without_holds([i for i, step in _steps(ids, data, cfg).items() if step in RUN_STEPS], cfg)
    on = cfg.get("big_runner", False)
    big = _big_reservations(data, cfg, want) if on or cfg.get("reserve_big", True) else None
    if big is not None:
        big = {**big, "reserved": _without_holds(big["reserved"], cfg)}
        want = [sid for sid in want if sid not in big["reserved"]]
    spool = cfg["queue_spool"]
    waiting = _waiting(spool)
    lent = _big_on_b(data, cfg, big["reserved"], waiting) if big is not None and on else {}
    want = [*want, *lent.get("lend", ())]  # after the normal rows
    room = max(0, int(cfg["queue_depth"]) - waiting)
    candidates = _not_pending(want, data)
    # A row a live lane still holds (a mini lane waiting for its relay ingest) would make
    # stage3.queue refuse the whole batch; it waits for a later pass instead.
    held = stage3.lane_holds(candidates, data) if candidates else {}
    chosen = [i for i in candidates if i not in held][:room]
    result = {"eligible": len(want), "waiting": waiting, **({"held_by_lane": held} if held else {}),
              **({"big_on_b": lent} if lent else {})}
    if dry_run or not chosen:
        result = {**result, "would_queue": chosen}
    else:
        result = {**result, "queued": stage3.queue(None, data=data, ids=chosen, spool=spool)}
    if big is None:
        return result
    if not on:  # reserved for G, which the tick starts only with big_runner on
        return {**result, "big": {**big, "held": big.get("held") or "big_runner off: reserved rows wait for G"}}
    return {**result, "big": _big_dispatch(data, cfg, big, dry_run=dry_run)}


def _without_holds(ids: Sequence[str], cfg: Mapping[str, Any]) -> list[str]:
    """``ids`` less the rows of the axon batch in flight (``axon_holds``, set by the tick for
    ``axon_feed`` ``big_cohorts``): no spool takes them while the cluster has them."""

    holds = set(cfg.get("axon_holds") or ())
    return [sid for sid in ids if sid not in holds]


def _big_on_b(data: DataLayout, cfg: Mapping[str, Any], reserved: Sequence[str],
              waiting: int) -> dict[str, Any]:
    """Push only (``big_on_b``, rung 4, 3 Nov 2026): by day, outside G's window and its lead-in,
    B takes big rows into its own 36 GiB declaration, at most ``rows`` at a time (default 1), each
    projected at most ``max_gib`` (default 28), and only while spool-B has at most
    ``max_waiting`` (default 2) unclaimed jobs, so normal trace work always goes first. B's
    runner fits each attempt's cap inside its declaration (``runner.serve`` backfill), so no new
    admission, lane or owner-first claim is involved. ``lend``: the rows to add to B's queue."""

    from menagerie.pipeline import runner_job, stage3

    given = cfg.get("big_on_b")
    if not given:
        return {}
    if not isinstance(given, Mapping):
        raise ValueError("big_on_b must be an object: rows, max_gib, max_waiting")
    rows, max_gib = int(given.get("rows", 1)), float(given.get("max_gib", 28))
    if runner_job.big_hold_open():
        return {"held": "G's night window: big rows are G's"}
    if waiting > int(given.get("max_waiting", 2)):
        return {"held": f"spool-B has {waiting} jobs waiting: normal trace work first"}
    in_b = _big_in_b(reserved)
    room = rows - len(in_b)
    if room <= 0:
        return {"in_b": in_b}
    peaks, families = stage3.Peaks.read(data), stage3._families(data)
    lend = []
    for sid in reversed(_not_pending(reserved, data)):  # smallest projection first
        projected = stage3.projected_peak(read_json(ModelRecord, data.model(sid)), peaks, families)
        if projected and projected <= max_gib:
            lend.append(sid)
        if len(lend) >= room:
            break
    return {"in_b": in_b, "lend": lend}


def _big_in_b(reserved: Sequence[str]) -> list[str]:
    """Reserved big rows with an open job on B (sent without G's ``big_runner`` sizing)."""

    from menagerie.pipeline import stage3

    wanted, out = set(reserved), []
    for name in stage3.open_jobs():
        sid = name.split("-", 1)[0]
        if sid in wanted:
            job = json.loads((stage3._local("sent") / f"{name}.json").read_text(encoding="utf-8"))
            if not (job.get("sizing") or {}).get("big_runner"):
                out.append(sid)
    return out


def _big_reservations(data: DataLayout, cfg: Mapping[str, Any], want: Sequence[str]) -> dict[str, Any]:
    try:
        return {"reserved": _big_candidates(data, want, cfg=cfg)}
    except (OSError, ValueError, KeyError) as exc:
        return {"reserved": [], "held": str(exc)}


def _big_dispatch(data: DataLayout, cfg: Mapping[str, Any], big: dict[str, Any], *,
                  dry_run: bool) -> dict[str, Any]:
    if "held" in big:
        return big
    try:
        return _big_queue(data, cfg, big["reserved"], dry_run=dry_run, reserved=True)
    except (OSError, ValueError) as exc:
        return {**big, "held": str(exc)}


def _big_sent() -> set[tuple[str, str]]:
    """One automatic G attempt per recipe revision; manual retries remain available."""

    from menagerie.pipeline import stage3

    sent = (json.loads(path.read_text(encoding="utf-8")) for path in stage3._local("sent").glob("*.json"))
    return {(job["id"], job["rev"]) for job in sent if (job.get("sizing") or {}).get("big_runner")}


def _structure_ids(data: DataLayout) -> set[str]:
    """Latest eval trace captures, including members outside the current worklist."""

    from menagerie.pipeline import ledger
    from menagerie.pipeline.records import LedgerRow

    latest: dict[str, LedgerRow] = {}
    for row in ledger.read_all(data=data):
        if isinstance(row, LedgerRow) and row.stage == "trace" and row.mode != "train" and row.outcome != "infra":
            if row.id not in latest or row.at >= latest[row.id].at:
                latest[row.id] = row
    return {sid for sid, row in latest.items() if row.capture == "structure_only"}


def _big_candidates(data: DataLayout, want: Sequence[str], *, cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Structure-only rows first, then waiting traces, each in projected-peak order."""

    from menagerie.pipeline import flow, stage3

    structure = _structure_ids(data)
    peaks, families = stage3.Peaks.read(data), stage3._families(data)
    attempted = _big_sent()
    cohorts, stopped = _cohorts(cfg or {}), set((cfg or {}).get("stopped_cohorts", ()))
    ranked = []
    for sid in dict.fromkeys([*sorted(structure), *want]):
        if cohorts.get(sid) in stopped:
            continue
        if not data.model(sid).is_file():
            continue
        record = read_json(ModelRecord, data.model(sid))
        if record.recipe is None or record.give_up or record.recipe.placement in ("gpu-window", "structure-only"):
            continue
        if (sid, record.recipe.rev) in attempted:
            continue
        projected = stage3.projected_peak(record, peaks, families)
        if projected and flow.WORKER_GIB < projected <= flow.BIG_GIB:
            ranked.append((sid not in structure, -projected, sid))
    return [sid for _, _, sid in sorted(ranked)]


def _big_queue(data: DataLayout, cfg: Mapping[str, Any], want: Sequence[str], *,
               dry_run: bool, reserved: bool = False) -> dict[str, Any]:
    """Reserve big rows by day; queue their existing stage-3 requests only at night."""

    from menagerie.pipeline import runner_job, stage3

    reserved = list(want) if reserved else _big_candidates(data, want)
    spool = "scratch/menagerie/spool-G"
    waiting = _waiting(spool)  # by day too: last night's unclaimed jobs still count as G's work
    if not runner_job.night_open():
        return {"reserved": reserved, "waiting": waiting, "skipped": "outside night window"}
    chosen = _not_pending(reserved, data)[:max(0, int(cfg["queue_depth"]) - waiting)]
    result = {"reserved": reserved, "waiting": waiting, "would_queue": chosen}
    if not dry_run and chosen:
        result["queued"] = stage3.queue(None, data=data, ids=chosen, spool=spool, again=True)
    return result


def _busy() -> tuple[Counter[str], set[str]]:
    """Live and queued lanes by kind, and the family keys they hold, with the rows of each 2a lane
    (``research.lane_rows``): a row with no record or packet yet has no family to hold it by.
    ``2b-hub`` counts the 2b lanes on the hub (no worker ``placement``) again, for the hub 2b cap;
    ``2a-worker`` the 2a lanes on a worker, for ``2a_worker_reserve``."""

    from menagerie.pipeline import admission, research

    lanes = [*admission.lanes(data=DataLayout.from_env()), *admission.queued()]
    held = {k for x in lanes for k in x.get("keys", ())}
    held.update(i for x in lanes if x["kind"] == "2a" and x.get("brief")
                for i in research.lane_rows(x["brief"]))
    count = Counter(x["kind"] for x in lanes)
    count.update("2b-hub" for x in lanes if x["kind"] == "2b" and not x.get("placement"))
    count.update("2a-worker" for x in lanes if x["kind"] == "2a" and x.get("placement"))
    return count, held


LAUNCH_SPENT = "held: this pass's launch time is spent; the next pass starts it"


_LAUNCH_BY: ContextVar[float | None] = ContextVar("launch_by", default=None)


@contextmanager
def _launch_window(by: float | None) -> Iterator[None]:
    """Lanes start only before ``by`` (a ``time.monotonic`` reading; none: no limit) inside."""
    token = _LAUNCH_BY.set(by)
    try:
        yield
    finally:
        _LAUNCH_BY.reset(token)


def _launch_open() -> bool:
    """Whether a lane may still start: before the tick's ``cfg["launch_by"]`` (``top_up``)."""
    by = _LAUNCH_BY.get()
    return by is None or time.monotonic() < by


def _launch_2b(families: Sequence[str] | Sequence[Sequence[str]], data: DataLayout, *,
               placements: Sequence[str | None] = (), cfg: Mapping[str, Any] | None = None,
               hub_room: int | None = None) -> list[str]:
    """Start each lane at its placement, then read the worker lanes' run-on admissions.

    Worker lanes are all started first (``mini_placement.start``) and their admissions read after
    (``mini_placement.finish``), so the waits for run-on admission overlap instead of adding up;
    a proven refusal still falls back to a hub slot while ``hub_room`` lasts. No lane starts after
    the pass's launch time (``_launch_open``).
    """
    cfg = cfg or {}
    runs: list[Any] = []
    for n, family in enumerate(families):
        if not _launch_open():
            runs.append(LAUNCH_SPENT)
            continue
        run, used = _placed_2b(family, data, placements[n] if placements else None, cfg, hub_room)
        runs.append(run)
        hub_room = _spend(hub_room, used)
    read = _read_admissions([run for run in runs if not isinstance(run, str)])
    for n, run in enumerate(runs):
        if not isinstance(run, str):
            runs[n], used = _admitted_2b(families[n], data, run, cfg, hub_room,
                                         read.get(run.lease["label"], _NOT_READ))
            hub_room = _spend(hub_room, used)
    return runs


_NOT_READ = object()  # a started lane whose admission ``_read_admissions`` did not read


def _read_admissions(started: Sequence[Any]) -> dict[str, Any]:
    """The started lanes' admissions in one ``run-on state`` call a round
    (``mini_placement.admissions``); an unreadable round leaves each lane to read its own."""
    from menagerie.pipeline import mini_placement

    if len(started) < 2:
        return {}
    try:
        return dict(mini_placement.admissions([run.lease["label"] for run in started]))
    except (OSError, subprocess.TimeoutExpired):
        return {}


def _spend(hub_room: int | None, used: bool) -> int | None:
    return hub_room - 1 if used and hub_room is not None else hub_room


def _placed_2b(family: str | Sequence[str], data: DataLayout, host: str | None,
               cfg: Mapping[str, Any], hub_room: int | None) -> tuple[Any, bool]:
    """One placement and its fallback, returning the run (a started worker lane whose admission
    is still to be read, ``_admitted_2b``) and whether it consumed a hub slot."""
    from menagerie.pipeline import admission, implement, mini_placement

    if host is None and hub_room == 0:
        return "held: hub fallback slots used", False
    try:
        if host is None:
            launch = implement.launch if isinstance(family, str) else implement.launch_batch
            return launch(family, role="model-implementation", data=data), True
        keys = [family] if isinstance(family, str) else list(family)
        return mini_placement.start(keys, role="model-implementation", data=data, host=host,
                                    config=cfg.get("mini_workers", {}).get(host) or {},
                                    cfg=cfg), False
    except admission.Refused as refused:
        return _refused_2b(family, data, host, cfg, hub_room, refused)
    except (OSError, subprocess.TimeoutExpired):
        if host is None:
            raise
        return "held: remote placement uncertain; inspect retained lease", False


def _admitted_2b(family: str | Sequence[str], data: DataLayout, started: Any,
                 cfg: Mapping[str, Any], hub_room: int | None,
                 admitted: Any = _NOT_READ) -> tuple[str, bool]:
    """A started worker lane's admission (``mini_placement.finish``, given ``admitted`` when
    read together), with ``_placed_2b``'s fallback when the worker proved a refusal."""
    from menagerie.pipeline import admission, mini_placement

    try:
        if admitted is _NOT_READ:
            return mini_placement.finish(started), False
        return mini_placement.finish(started, admitted), False
    except admission.Refused as refused:
        return _refused_2b(family, data, started.host, cfg, hub_room, refused)
    except (OSError, subprocess.TimeoutExpired):
        return "held: remote placement uncertain; inspect retained lease", False


def _refused_2b(family: str | Sequence[str], data: DataLayout, host: str | None,
                cfg: Mapping[str, Any], hub_room: int | None, refused: Any) -> tuple[Any, bool]:
    if host and hub_room is not None and hub_room > 0:
        return _placed_2b(family, data, None, cfg, hub_room)
    return f"held: {refused.result.get('why')}", False


def _launch_2a(chunks: Sequence[Sequence[str]], data: DataLayout,
               hosts: list[str] | None = None) -> list[str]:
    """One 2a lane per chunk; ``hosts`` (free worker slots, ``mini_2a`` on) are tried first."""

    from menagerie.pipeline import admission, research

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")
    runs = []
    for n, chunk in enumerate(chunks):
        if not _launch_open():
            runs.append(LAUNCH_SPENT)
            continue
        label = f"2a-tick-{stamp}-{n}"
        path = work_dir("tick") / "batches" / f"{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"parent": None, "batches": [
            {"label": label, "families": [{"ids": [i]} for i in chunk]}]}), encoding="utf-8")
        try:
            runs += research.launch(path, data=data, hosts=hosts)
        except admission.Refused as refused:
            runs.append(f"held: {refused.result.get('why')}")
    return runs


def hub_slots() -> int:
    """Lanes the hub can take now (``agent-capacity``): none at WARN or CRIT."""

    from menagerie.pipeline.admission import _tool

    done = subprocess.run([_tool("agent-capacity"), "--json"], capture_output=True, text=True, check=False, timeout=60)
    if done.returncode not in (0, 1, 2):
        raise OSError(f"agent-capacity rc {done.returncode}: {done.stderr.strip()[-200:]}")
    reading = json.loads(done.stdout)
    return 0 if reading.get("status") in ("WARN", "CRIT") else int(reading.get("free_slots", 0))


def _caps(cfg: Mapping[str, Any]) -> dict[str, int]:
    """Lane caps: ``admission.lane_cap``, lowered (never raised) by ``config.json``'s ``lane_limits``."""

    from menagerie.pipeline import admission

    given = cfg.get("lane_limits") or {}
    return {k: min(admission.lane_cap(k), int(given.get(k, admission.lane_cap(k))))
            for k in ("2a", "2b")}


def _plan_2a(need_2a: Sequence[str], ready_2b: int, busy_2a: int, target: int, cap_2a: int,
             per: int) -> list[list[str]]:
    """2a lanes that keep ``target`` families ready for 2b (see ``_buffer_target``).

    Families a running 2a lane will hand over count as ready (``per`` each); at most
    ``cap_2a - busy_2a`` lanes start.
    """

    short = target - ready_2b - busy_2a * per
    lanes = min(max(0, cap_2a - busy_2a), -(-max(0, short) // per))
    return [list(need_2a[k:k + per]) for k in range(0, len(need_2a), per)][:lanes]


def _buffer_target(cfg: Mapping[str, Any], caps: Mapping[str, int]) -> int:
    """Families 2a keeps ready for 2b, whatever this pass's free 2b slots: ``2b_buffer_families``.

    The default is two full sets of 2b lanes (2 x the 2b cap x ``families_per_2b_lane``), so 2a
    keeps working while 2b is saturated and 2b slots do not idle while a 2a lane catches up.
    """

    given = cfg.get("2b_buffer_families")
    if given is None:
        return 2 * caps["2b"] * int(cfg.get("families_per_2b_lane", 1))
    if isinstance(given, bool) or not isinstance(given, int) or given < 0:
        raise ValueError("2b_buffer_families must be a whole number of families, 0 or more")
    return given


def _cohorts(cfg: Mapping[str, Any]) -> dict[str, str]:
    path = Path(cfg.get("worklist") or work_dir("tick") / "worklist.txt").expanduser()
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else ():
        tokens = line.split("#", 1)[0].split()
        if tokens:
            if len(tokens) < 2 and (cfg.get("tripwires") or cfg.get("cohort_share")):
                raise ValueError("enabled overlap requires every active worklist row to have a cohort tag")
            out.setdefault(tokens[0], tokens[1] if len(tokens) > 1 else "r3")
    return out


def _cohort_waiting(ids: Sequence[str], steps: Mapping[str, str], family: Mapping[str, Any],
                    held: set[str], cfg: Mapping[str, Any]) -> dict[str, dict[str, list[str]]]:
    cohorts, stopped = _cohorts(cfg), set(cfg.get("stopped_cohorts", ()))
    groups = {c: {"2a": [], "2b": []} for c in dict.fromkeys(cohorts.values()) if c not in stopped}
    seen = set()
    for model_id in ids:
        cohort, own = cohorts[model_id], family.get(model_id)
        if cohort not in groups or own in held or model_id in held:
            continue
        step = steps.get(model_id)
        if step == "2a":
            groups[cohort]["2a"].append(model_id)
        elif step == "2b" and own and own not in seen:
            groups[cohort]["2b"].append(own)
            seen.add(own)
    return groups


def _share_room(cohort: str, cfg: Mapping[str, Any], caps: Mapping[str, int], occupied: int) -> int:
    share = (cfg.get("cohort_share") or {}).get(cohort, 0)
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share <= 1:
        raise ValueError("cohort shares must be numbers between 0 and 1")
    return max(0, math.floor(caps["2b"] * share) - occupied)


def _cap_room(cohort: str, cfg: Mapping[str, Any], room: int, busy: int) -> int:
    """``room`` within the cohort's absolute 2b lane cap (``cohort_cap``), which holds whether or
    not an earlier cohort waits; ``room`` when it has none."""

    cap = (cfg.get("cohort_cap") or {}).get(cohort)
    if cap is None:
        return room
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError("cohort caps must be whole numbers of 2b lanes, 0 or more")
    return max(0, min(room, cap - busy))


def _lane_busy(cfg: Mapping[str, Any], family: Mapping[str, Any], kind: str) -> Counter[str]:
    """Count lane slots by cohort, including unknown leases conservatively in every target."""
    from menagerie.pipeline import admission

    cohorts = _cohorts(cfg)
    owners = {family[i]: cohorts[i] for i in cohorts if family.get(i)}
    counts: Counter[str] = Counter()
    for lane in [*admission.lanes(data=DataLayout.from_env()), *admission.queued()]:
        if lane["kind"] == kind:
            names = {owners[k] for k in lane.get("keys", ()) if k in owners}
            counts.update(names or {"unknown"})
    return counts

def _overlap_research(group: Mapping[str, list[str]], ready: int, share: int,
                      cfg: Mapping[str, Any], caps: Mapping[str, int],
                      busy: int, capacity_busy: int, slots: int) -> list[list[str]]:
    """A later cohort's 2a lanes: keep its 2b buffer full, inside the 2a cap and spare hub slots.

    ``share`` is the cohort's 2b lane share; a cohort with none never reaches 2b, so it gets no
    research. Free 2b slots this pass do not matter: research runs while 2b is saturated.
    """

    if share <= 0:
        return []
    return _plan_2a(group["2a"], ready, busy, _buffer_target(cfg, caps), caps["2a"],
                    int(cfg["families_per_2a_lane"]))[:max(0, min(slots, caps["2a"] - capacity_busy))]


def _research_room(slots: int, mini_slots: int, batches: int, worker_2a: bool,
                   reserved: int = 0) -> int:
    """Seats for 2a: the hub slots 2b lanes past the workers leave, plus, with ``mini_2a`` on,
    the worker slots the 2b lanes leave (``hub_slots`` does not bound those) and the worker
    slots held back from 2b for 2a (``reserved``, see ``_worker_reserve``)."""

    hub = slots - max(0, batches - mini_slots)
    return hub + ((max(0, mini_slots - batches) + reserved) if worker_2a else 0)


def _worker_reserve(cfg: Mapping[str, Any], mini_slots: Sequence[str], count: Mapping[str, int],
                    caps: Mapping[str, int], waiting_2a: int) -> tuple[list[str], list[str]]:
    """Split this pass's free worker slots into (2b's, 2a's): ``2a_worker_reserve`` keeps that many
    worker slots for 2a lanes, counting 2a lanes already on workers, while rows wait for 2a.

    2b otherwise takes every free worker slot first, so with 2b's queue deep, 2a got none
    (rung 4 push, 3 Nov 2026). The reserve never exceeds the 2a cap's room or the lanes the
    waiting rows fill (``families_per_2a_lane`` each). It comes off the first configured workers
    (the Macs): 2a needs no runner store, and the trace host's owner-first refusals then hit 2b only.
    """

    given = cfg.get("2a_worker_reserve") or 0
    if isinstance(given, bool) or not isinstance(given, int) or given < 0:
        raise ValueError("2a_worker_reserve must be a whole number of worker slots, 0 or more")
    per = int(cfg["families_per_2a_lane"])
    keep = min(given - count.get("2a-worker", 0), caps["2a"] - count.get("2a", 0),
               -(-waiting_2a // per), len(mini_slots))
    if keep <= 0:
        return list(mini_slots), []
    return list(mini_slots[keep:]), list(mini_slots[:keep])


def _cohort_batches(ids: Sequence[str], steps: Mapping[str, str], family: Mapping[str, Any],
                    held: set[str], cfg: Mapping[str, Any], caps: Mapping[str, int],
                    available: int, slots: int, count: Mapping[str, int],
                    data: DataLayout, mini_slots: int = 0, worker_2a: bool = False,
                    reserved: int = 0) -> tuple[list[list[str]], list[str], list[list[str]]]:
    """Reserve earlier research and recipes first, then use spare slots for later cohorts."""
    from menagerie.pipeline import implement

    groups = _cohort_waiting(ids, steps, family, held, cfg)
    cohorts = _cohorts(cfg)
    waiting = {cohorts[i] for i in ids if steps.get(i) in ("2a", "2b")}
    pending = {c: g for c, g in groups.items() if c in waiting}
    need_2b = [f for g in groups.values() for f in g["2b"]]
    if not pending:
        return [], need_2b, []
    first, *later = pending
    per, batch_size = int(cfg["families_per_2a_lane"]), int(cfg.get("families_per_2b_lane", 1))
    primary = pending[first]
    research_busy = _lane_busy(cfg, family, "2a") if count["2a"] else Counter()
    slots_busy = _lane_busy(cfg, family, "2b") if count["2b"] else Counter()
    first_busy = research_busy[first] + research_busy["unknown"]
    batches = implement.batch_groups(primary["2b"], batch_size, data=data)[
        :_cap_room(first, cfg, available, slots_busy[first] + slots_busy["unknown"])]
    ready = len(primary["2b"]) - sum(map(len, batches))
    seats = min(_research_room(slots, mini_slots, len(batches), worker_2a, reserved),
                caps["2a"] - count["2a"])
    chunks = _plan_2a(primary["2a"], ready, first_busy, _buffer_target(cfg, caps),
                      caps["2a"], per)[:max(0, seats)]
    # Earlier rows that still want a seat: a row a live lane holds is not in ``primary``.
    unseated = set(primary["2a"]).difference(*chunks)
    for cohort in later:
        busy = slots_busy[cohort] + slots_busy["unknown"]
        room = _cap_room(cohort, cfg, _share_room(cohort, cfg, caps, busy), busy)
        limit = min(available - len(batches), slots + mini_slots - len(batches) - len(chunks), room)
        extra = implement.batch_groups(pending[cohort]["2b"], batch_size, data=data)[:max(0, limit)]
        batches.extend(extra)
        # Later research cannot occupy a seat while earlier research still waits for one.
        if unseated:
            continue
        share = _share_room(cohort, cfg, caps, 0)
        chunks.extend(_overlap_research(
            pending[cohort], len(pending[cohort]["2b"]) - sum(map(len, extra)), share,
            cfg, caps, research_busy[cohort] + research_busy["unknown"],
            count["2a"] + len(chunks),
            _research_room(slots, mini_slots, len(batches), worker_2a, reserved) - len(chunks)))
    return batches, need_2b, chunks


def _cohort_plan(ids: Sequence[str], steps: Mapping[str, str], family: Mapping[str, Any],
                 cfg: Mapping[str, Any], batches: Sequence[Sequence[str]],
                 chunks: Sequence[Sequence[str]]) -> dict[str, Any]:
    cohorts = _cohorts(cfg)
    out = {}
    for cohort in dict.fromkeys(cohorts.values()):
        members = [i for i in ids if cohorts[i] == cohort]
        own_families = {family.get(i) for i in members if family.get(i)}
        out[cohort] = {"waiting_2a": sum(steps.get(i) == "2a" for i in members),
                       "waiting_2b": sum(steps.get(i) == "2b" for i in members),
                       "stopped": cohort in cfg["stopped_cohorts"],
                       "2b": sum(bool(own_families.intersection(batch)) for batch in batches),
                       "2a": sum(bool(set(members).intersection(chunk)) for chunk in chunks)}
    return out


def top_up(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Start 2b and 2a lanes for worklist rows, worklist order first, inside caps and hub room.

    A 2b lane takes a configured batch of same-case families, up to the 2b cap (14 by default).
    2a lanes (up to ``families_per_2a_lane`` rows with no packet each) start as needed to keep
    ``2b_buffer_families`` families ready for 2b after this pass (default two full sets of 2b
    lanes), counting what running 2a lanes will hand over, whether or not a 2b slot is free this
    pass; the 2a cap and the hub's free slots still bound them. Families a live or queued lane holds
    are skipped, and so are families whose last lane ended without touching the rows that sent
    it (``_spent_2b``, listed as ``2b_spent``); launches never exceed the hub's free slots
    (``agent-capacity``), and admission is read before each one. Review lanes start in ``reviews``,
    describe lanes in ``describes``.
    With ``describes.workers`` on, the worker slots the plan leaves free are recorded as the
    describe stage's room on the workers (``describe_workers.record_room``). With ``mini_2a`` on,
    2a also gets the worker slots 2b leaves, outside the hub's free slots, and its lanes try those
    workers first (``research.launch``'s ``hosts``); describe gets what 2a leaves.
    2b launches wait for a fresh bridge heartbeat and, in the tick, runner A's live declaration
    (``cfg["runner_declaration"]``); the 2a plan is unchanged by either hold. 2b lanes run model
    code, so they take only worker slots (``mini_workers``) and every hub slot is left for 2a,
    unless ``cfg["2b_on_hub"]`` is true (the old placement: hub slots and hub fallback too) or
    ``{"lanes": N}`` (the same, with at most N 2b lanes on the hub at once, each bounded at launch to
    its share of the hub 2b budget, ``admission.HUB_2B_PROPERTIES``; 2a keeps the other hub slots).
    ``2a_worker_reserve`` (N) keeps N free worker slots for 2a while rows wait for it, counting 2a
    lanes already on workers (``_worker_reserve``); 2b takes the rest first, as before.
    With every slot of both kinds taken it returns at once, before any planning (the tick's
    refill pass caps 2a at zero through ``lane_limits``, so there that means every 2b slot).
    """

    from menagerie.pipeline import drive, mini_2a, mini_placement

    ids = worklist(cfg)
    if not ids:
        return {"skipped": "no worklist"}
    count, held = _busy()
    caps = _caps(cfg)
    # cognitive 17: this early return (before the record reads) is what keeps a refill cheap
    if all(count[kind] >= cap for kind, cap in caps.items()):
        return {"skipped": "every lane slot is taken", "busy": dict(count), "caps": caps}
    steps = _steps(ids, data, cfg)
    family = drive.families_of(ids, data=data)
    steps, spent = _hold_spent(steps, family, data)
    steps, repairs = _hold_repairs(steps, family, data, cfg)
    from menagerie.pipeline import tick

    cfg = {**cfg, "stopped_cohorts": cfg.get("stopped_cohorts", [
        c for c, v in tick.load_state().get("cohort_states", {}).items() if v.get("stopped")])}
    slots = hub_slots()
    worker_2a = mini_2a.enabled(cfg) and bool(cfg.get("mini_workers"))
    groups = _cohort_waiting(ids, steps, family, held, cfg)
    need_2a = [i for group in groups.values() for i in group["2a"]]
    mini_slots, reserved_2a = _worker_reserve(cfg, mini_placement.slots(cfg), count, caps,
                                              len(need_2a)) if worker_2a else (
        mini_placement.slots(cfg), [])
    hub_2b = _hub_2b_slots(cfg, slots, count["2b-hub"])
    batch_size = int(cfg.get("families_per_2b_lane", 1))
    available = max(0, min(caps["2b"] - count["2b"], hub_2b + len(mini_slots)))
    batches, need_2b, chunks = _cohort_batches(
        ids, steps, family, held, cfg, caps, available, slots, count, data, len(mini_slots),
        worker_2a, len(reserved_2a))
    held_2b = _held_2b(cfg)
    # Repair rounds are the earliest rows waiting: their hub lanes come out of the 2b cap first,
    # and under a capped hub ({"lanes": N}) out of its N, since they are hub 2b lanes too.
    capped = bool(mini_placement.hub_2b_lanes(cfg))
    repair_hub, repair_why = ([], None) if held_2b else _repair_hub(
        repairs, held, min(slots - len(chunks), caps["2b"] - count["2b"],
                           hub_2b if capped else caps["2b"]))
    hub_2b -= len(repair_hub) if capped else 0
    batches = [] if held_2b else batches[:max(0, min(caps["2b"] - count["2b"] - len(repair_hub),
                                                     len(mini_slots) + hub_2b))]
    placements = [*mini_slots, *([None] * hub_2b)][:len(batches)]
    hosts_2a = [*reserved_2a, *_left_over(mini_slots, placements)] if worker_2a else []
    launch_2b = [f for batch in batches for f in batch]
    plan = {"2b_where": _2b_where(cfg, placements), "busy": dict(count), "caps": caps,
            "hub_slots": slots, "2b": launch_2b, "2a": chunks,
            "waiting_2b": len(need_2b), "waiting_2a": len(need_2a), "2b_spent": spent}
    if worker_2a:
        plan["2a_worker_slots"] = len(hosts_2a)
        if cfg.get("2a_worker_reserve"):
            plan["2a_worker_reserved"] = len(reserved_2a)
    if cfg.get("mini_workers"):
        plan["2b_placements"] = placements
        plan["2b_minis"] = mini_placement.status(cfg)
    if held_2b:
        plan["2b_held"] = held_2b
    if repairs:
        plan["2b_repair_seat"] = repairs
        plan["2b_repair_hub"] = repair_hub if repair_why is None else f"held: {repair_why}"
    if not dry_run and cfg.get("mini_workers") and (cfg.get("describes") or {}).get("workers"):
        from menagerie.pipeline import describe_workers, pause

        # 2b first: describe seats get only the worker slots this plan leaves free. A 2b hold keeps
        # them free for 2b's return, except under a worker pause, which lasts hours.
        plan["describe_room"] = describe_workers.record_room(
            mini_slots, [*placements, *hosts_2a[:len(chunks)]],
            held=bool(held_2b) and not pause.active(cfg))
    plan["cohorts"] = _cohort_plan(ids, steps, family, cfg, batches, chunks)
    if batch_size > 1:
        plan["2b_batches"] = batches
    if dry_run:
        return plan
    chosen = launch_2b if batch_size == 1 else batches
    placed = {"placements": placements, "cfg": cfg,
              "hub_room": max(0, min(hub_2b, slots - len(chunks)))} if cfg.get("mini_workers") else {}
    with _launch_window(cfg.get("launch_by")):
        with _declared(None if held_2b else cfg.get("runner_declaration")):
            runs = _launch_2b(chosen, data, **placed) + _launch_repairs(repair_hub, data)
        runs_2a = (_launch_2a(chunks, data, hosts=hosts_2a) if worker_2a
                   else _launch_2a(chunks, data))
        return {**plan, "runs": runs + runs_2a}


def reviews(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """The review stage (``review.stage``) over the worklist rows at ``review``, when
    ``config.json``'s ``reviews.enabled`` is true: calibration at the prompt in force, recording
    what ended lanes left, and new review lanes while fewer than ``reviews.lanes`` run (default
    ``admission.lane_cap("review")``, never over its maximum; calibration lanes count).
    ``reviews.cohort_first`` names a cohort whose rows are planned first (worklist order else)."""

    from menagerie.pipeline import admission, review

    given = cfg.get("reviews") or {}
    if not given.get("enabled"):
        return {"skipped": "reviews are not enabled in config.json"}
    ids = worklist(cfg)
    steps = _steps(ids, data, cfg) if ids else {}
    at = [i for i in ids if steps.get(i) == "review"]
    first = given.get("cohort_first")
    if first is not None:  # a push: that cohort's rows get the review seats first
        if not isinstance(first, str):
            raise ValueError("reviews.cohort_first must name one cohort")
        cohorts = _cohorts(cfg)
        at.sort(key=lambda i: cohorts.get(i) != first)
    count, held = _busy()
    cap = min(int(given.get("lanes", admission.lane_cap("review"))), admission.LANE_CAP_MAX["review"])
    return review.stage(at, data=data, room=cap - count["review"], held=held, dry_run=dry_run)


def disputed(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """While reviews are enabled, take ``review_disputed`` back from each row a fully passed main
    seat has not reviewed at its recipe (``review_disputed.reopen``; decisions.md 2026-11-03).
    Typing rows ``review_disputed`` is never a tick step: ``review disputed --apply`` does it."""

    from menagerie.pipeline import review_disputed

    if not (cfg.get("reviews") or {}).get("enabled"):
        return {"skipped": "reviews are not enabled in config.json"}
    return review_disputed.reopen(data=data, apply=not dry_run)


def describes(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """The describe stage (``describe_stage.stage``) when ``config.json``'s ``describes.enabled`` is
    true: one detached ``describe write`` run at a time, with the seat cap, hub reserve, Claude
    week-use ceiling and earlier-pin switch from ``describes``. It reads the data root from the
    tick's environment, as the run does."""

    from menagerie.pipeline import describe_stage

    del data
    given = cfg.get("describes") or {}
    if not given.get("enabled"):
        return {"skipped": "describes are not enabled in config.json"}
    return describe_stage.stage({k: v for k, v in given.items() if k != "enabled"}, dry_run=dry_run)


def _2b_on_hub(cfg: Mapping[str, Any]) -> bool:
    """Whether 2b lanes may use hub slots (``config.json`` ``2b_on_hub``, default false; capped
    with ``{"lanes": N}``, ``mini_placement.hub_2b_lanes``)."""
    from menagerie.pipeline import mini_placement

    return mini_placement.hub_2b_lanes(cfg) != 0


def _left_over(mini_slots: Sequence[str], placements: Sequence[str | None]) -> list[str]:
    """The free worker slots, one host name each, that this pass's 2b placements leave."""

    left = list(mini_slots)
    for host in placements:
        if host in left:
            left.remove(host)
    return left


def _hub_2b_slots(cfg: Mapping[str, Any], slots: int, on_hub: int = 0) -> int:
    """Hub slots 2b lanes may take: none unless ``2b_on_hub``, so 2a keeps every hub slot; every
    free one when it is true; with ``{"lanes": N}`` what N leaves after the ``on_hub`` 2b lanes
    already there, so 2a keeps the rest."""
    from menagerie.pipeline import mini_placement

    lanes = mini_placement.hub_2b_lanes(cfg)
    return slots if lanes is None else min(slots, max(0, lanes - on_hub))


def _2b_where(cfg: Mapping[str, Any], placements: Sequence[str | None]) -> str:
    """Where this pass's 2b lanes go and why, for ``status.md``."""
    from menagerie.pipeline import admission, mini_placement

    lanes = mini_placement.hub_2b_lanes(cfg)
    if lanes is None:
        where = "workers and hub (2b_on_hub true)"
    elif lanes:
        where = (f"workers and hub (2b_on_hub: at most {lanes} hub lanes, each bounded to "
                 f"{admission.HUB_2B_LANE_CPUS} CPU and {admission.HUB_2B_LANE_GIB} GiB, niced)")
    else:
        where = "workers only (2b_on_hub false: 2b runs model code; hub slots are kept for 2a)"
    placed = Counter("hub" if host is None else host for host in placements)
    return f"{where}; this pass: " + (", ".join(f"{n} on {h}" for h, n in placed.items()) or "none")


def _held_2b(cfg: Mapping[str, Any]) -> str | None:
    """Why 2b lanes cannot start this pass: the bridge is not serving, the tick found no
    declaration for them (``cfg["runner_declaration"]``; absent outside the tick, where the
    caller's environment applies and ``implement._wrapper`` checks it), or a worker pause stops
    the runner their tries need (``pause``)."""

    from menagerie.pipeline import pause, runner

    if pause.active(cfg):
        return pause.note(cfg)
    if not runner.serving(Path(os.environ.get("MENAGERIE_BRIDGE_SPOOL") or work_dir("hubspool"))):
        return "bridge not serving"
    return (cfg.get("runner_declaration") or {}).get("held")


@contextmanager
def _declared(env: Mapping[str, str] | None) -> Iterator[None]:
    """Runner A's declaration in this process's environment while 2b lanes launch (their verb
    wrappers copy it, ``implement._wrapper``), then the old values; nothing set without one."""

    old = {k: os.environ.get(k) for k in env or {}}
    os.environ.update(env or {})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def trickle(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> Any:
    """``paper harvest --limit``: bibliographic records for new paper ids, a few a pass; then,
    when ``library_docs`` is positive, that many library packets get their library's own
    documentation as a describe text source (``library_docs.docs``), starting no family after
    ``library_docs_seconds``; then, when ``paper_texts`` is positive, that many packets get the
    open-access text of a paper never fetched (``paper_texts.texts``), starting none after
    ``paper_texts_seconds``; then, when ``paper_find`` is positive, that many families describe found
    no text for look for their paper by exact title or identifier (``paper_texts.identify``), starting
    none after ``paper_find_seconds``; a paper found there gets its text on a later pass.

    The tick commits what it writes (``tick._step_trickle``); a dry run lists the packets due and
    fetches and writes nothing."""

    from menagerie.pipeline import StoreLayout, find_paper, library_docs, paper_texts

    store = StoreLayout.from_env()
    rows = find_paper.harvest(None, data=data, store=store, limit=int(cfg["trickle_limit"]),
                              write=not dry_run)
    limit = int(cfg.get("library_docs") or 0)
    if limit > 0:
        rows = [*rows, *library_docs.docs(
            None, data=data, store=store, limit=limit, write=not dry_run,
            seconds=float(cfg.get("library_docs_seconds") or 90))]
    limit = int(cfg.get("paper_texts") or 0)
    if limit > 0:
        rows = [*rows, *paper_texts.texts(
            None, data=data, store=store, limit=limit, write=not dry_run,
            seconds=float(cfg.get("paper_texts_seconds") or 90))]
    limit = int(cfg.get("paper_find") or 0)
    if limit > 0:
        rows = [*rows, *paper_texts.identify(
            None, data=data, store=store, limit=limit, write=not dry_run,
            seconds=float(cfg.get("paper_find_seconds") or 60))]
    return rows


# Reference yields by tier; thresholds are strict. GitHub is rung 3's measured rate (52 percent of
# 569 models, Crawler 26 Oct), the others rung 2's. The research-gap check waits for 50 settled
# rows; the yield check for 100 attempted ones (settled models whose research found code, so a
# recipe was tried): early settlement is mostly 2a retirements and fast give-ups.
COHORT_REFERENCE = {"github": 0.52, "paper-only": 0.40, "ports": 0.70, "native-jax": 0.10}
MIN_SETTLED = 50
MIN_ATTEMPTED = 100
YIELD_DROP = 0.15
GAP_LIMIT = 0.10


def _cohort_counts(data: DataLayout, cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from menagerie.pipeline import drive

    files = drive._load(data)
    counts: dict[str, dict[str, Any]] = {}
    for model_id, cohort in _cohorts(cfg).items():
        own = counts.setdefault(cohort, {"rows": 0, "settled": 0, "validated": 0, "retired": 0,
                                         "completed_2a": 0, "research_gap": 0, "awaiting_pin": 0})
        for key, n in _row_counts(model_id, files).items():
            own[key] += n
    return counts


def _row_counts(model_id: str, files: Any) -> dict[str, int]:
    """One worklist row's part of its cohort's counts."""
    from menagerie.pipeline import drive, ledger

    record = files.record(model_id)
    family = record.family if record else files.family_of(model_id)
    packet = files.packet(family) if family else None
    giveup = record.give_up if record else None
    rows = files.rows.get(model_id, [])
    valid = ledger.validated_record(record, rows, tl=files.tl) is not None
    # Validated at an earlier pin and not yet re-traced at this one: a pin move, not a failure.
    carried = not valid and ledger.validated_record(record, rows) is not None
    typed = giveup is not None and drive._head(giveup.cls) != "infra" and not valid
    step, why = (None, "") if valid or typed else drive._step(model_id, files)
    return {"rows": 1, "validated": int(valid), "awaiting_pin": int(carried),
            "completed_2a": int(packet is not None or record is not None),
            "research_gap": int(typed and giveup.cls.startswith("research_gap")),
            "settled": int(valid or typed or step == "done"),
            # A row the roster retired (not an architecture, a duplicate) settles with no model.
            "retired": int(step == "done" and why.startswith("roster:"))}


def _trip_reason(cohort: str, counts: Mapping[str, int]) -> str | None:
    """Yield is validated over settled models (retired rows are not models), judged once
    ``MIN_ATTEMPTED`` settled models got past research; research gaps over completed 2a rows.
    A model validated at an earlier TorchLens pin and waiting for its re-trace (``awaiting_pin``)
    counts as settled and validated, so a pin move does not trip the yield; a re-trace that fails
    at the new pin leaves that count and lowers the yield as any failure does."""
    carried = counts.get("awaiting_pin", 0)
    if counts["settled"] + carried < MIN_SETTLED:
        return None
    reference = next((v for k, v in COHORT_REFERENCE.items() if cohort.endswith(k)), None)
    if reference is None:
        return None
    # Integer percentages keep equality at the boundary exact.
    threshold = round((reference - YIELD_DROP) * 100)
    models = counts["settled"] + carried - counts.get("retired", 0)
    validated = counts["validated"] + carried
    attempted = models - counts["research_gap"]
    if attempted >= MIN_ATTEMPTED and validated * 100 < models * threshold:
        return (f"{validated} of {models} settled models validated; "
                f"yield is below {threshold} percent")
    if counts["research_gap"] * 100 > counts["completed_2a"] * round(GAP_LIMIT * 100):
        return (f"{counts['research_gap']} of {counts['completed_2a']} completed research rows "
                "are typed research gaps; more than 10 percent")
    return None


def _standalone_tripwires(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool,
                          resume: str | None) -> dict[str, Any]:
    from menagerie.pipeline import tick

    if dry_run:
        return tripwires(data, cfg, dry_run=True, resume=resume, state=tick.load_state())
    with tick.locked() as acquired:
        if not acquired:
            return {"held": "another tick holds the lock"}
        state = tick.load_state()
        result = tripwires(data, cfg, dry_run=False, resume=resume, state=state)
        state["outbox"] = [*state.get("outbox", []),
                           *[[f"cohort-{c}", t] for c, t in result["facts"].items()]]
        tick.save_state(state)
        return result


def tripwires(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool,
              resume: str | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist cohort stops, never row types; direct calls queue FACTs for the next tick.

    A tick supplies its state under its existing lock. Standalone calls take the same lock.
    A resumed cohort gets another settled row before it can trip again on unchanged evidence.
    """
    if state is None:
        return _standalone_tripwires(data, cfg, dry_run=dry_run, resume=resume)
    counts = _cohort_counts(data, cfg)
    saved = {c: dict(v) for c, v in state.get("cohort_states", {}).items()}
    if resume is not None:
        if resume not in counts:
            raise ValueError(f"unknown cohort: {resume}")
        saved[resume] = {"stopped": False, "resume_after": counts[resume]["settled"]}
    facts, events = {}, []
    for cohort, own in counts.items():
        previous = saved.get(cohort, {})
        reason = _trip_reason(cohort, own)
        if not reason or previous.get("stopped") or own["settled"] <= previous.get("resume_after", -1):
            continue
        facts[cohort] = f"{cohort} stopped: {reason}. Rows stay open; review the evidence before resuming."
        event = {"verb": "cohort:stopped", "cohort": cohort, "reason": reason,
                 "at": datetime.now(timezone.utc).isoformat(), "counts": own}
        events.append(event)
        saved[cohort] = {"stopped": True, "event": event}
    if not dry_run:
        state["cohort_states"] = saved
        state["cohort_counts"] = counts
        state["cohort_events"] = [*state.get("cohort_events", []), *events]
    return {"cohorts": counts, "stopped": [c for c, v in saved.items() if v.get("stopped")],
            "facts": facts, "resumed": resume}
