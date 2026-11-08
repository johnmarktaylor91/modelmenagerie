"""Rung 4's tail for big rows (Lead, 2026-11-04): the deferred give-up ``big_row_deferred``.

A big row (``stage3.projected_peak`` past the worker's ``flow.WORKER_GIB``) is traced only by
runner G at night or by the axon feeder's cluster batches. When the rung's ordinary and render
rows have settled and big rows are still waiting, ``type_waiting`` gives each one of a cohort the
deferred give-up ``big_row_deferred``, which settles it for the close. It types only rows whose
next step is ``trace``, with no eval trace at their revision and the pin and no open stage-3 job
(``tick_full._not_pending``: not queued or running on G or B), not in the axon batch in flight,
and not held by a live or queued lane. It writes only with ``apply``
(``menagerie-pipeline review big-deferred --cohort C --apply``); Lead runs it.

``reopen`` takes the give-up back once G has room: spool-G's free queue room, less the big rows
still waiting for G that are not deferred, is how many rows return (smallest projection first);
``all_rows`` takes every one back. A returned row is a big row again, reserved for G and handed to
axon as before. Runs on the hub: imports neither torch nor psutil.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from menagerie.pipeline import DataLayout, drive, now, read_json, write_json
from menagerie.pipeline.records import GiveUp, ModelRecord

CLASS = "big_row_deferred"
SPOOL_G = "scratch/menagerie/spool-G"


def _cohort_ids(cohort: str, cfg: Mapping[str, Any]) -> list[str]:
    from menagerie.pipeline import tick_full

    return [i for i, c in tick_full._cohorts(cfg).items() if c == cohort]  # same package


def _sizer(data: DataLayout) -> Any:
    """``record -> projected GiB or None``, reading the measured peaks and families once."""

    from menagerie.pipeline import stage3

    peaks, families = stage3.Peaks.read(data), stage3._families(data)  # same package
    return lambda record: stage3.projected_peak(record, peaks, families) if record.recipe else None


def _axon_ids() -> set[str]:
    """The rows of the axon batch in flight, from the tick state."""

    from menagerie.pipeline import tick

    active = (tick.load_state().get("axon") or {}).get("active") or {}
    return set(active.get("ids") or ())


def _lane_held(ids: Sequence[str], data: DataLayout) -> set[str]:
    """Ids a live or queued lane holds, by family (``tick_full._busy``) or by id or family
    (``stage3.lane_holds``)."""

    from menagerie.pipeline import stage3, tick_full

    families = tick_full._busy()[1]  # same package
    held = set(stage3.lane_holds(ids, data))
    return held | {i for i in ids if read_json(ModelRecord, data.model(i)).family in families}


def _big_at_trace(data: DataLayout, cohort: str, cfg: Mapping[str, Any]) -> dict[str, float]:
    """The cohort's rows whose next step is ``trace`` and projection is past the worker's bound."""

    from menagerie.pipeline import flow

    chosen = _cohort_ids(cohort, cfg)
    if not chosen:  # ``next_steps`` reads an empty list as every rostered row
        return {}
    sized, need_of = {}, _sizer(data)
    for step in drive.next_steps(chosen, data=data):
        if step["step"] != "trace":
            continue
        record = read_json(ModelRecord, data.model(step["id"]))
        need = need_of(record) if record.give_up is None else None
        if need is not None and need > flow.WORKER_GIB:
            sized[step["id"]] = round(need, 1)
    return sized


def _why_skipped(sized: Mapping[str, float], data: DataLayout) -> dict[str, str]:
    """Each row ``type_waiting`` must leave alone, with the first reason that holds it."""

    from menagerie.pipeline import tick_full

    free = set(tick_full._not_pending(list(sized), data))  # same package
    axon = _axon_ids()
    out = {i: "traced_or_open_job" for i in sized if i not in free}
    out |= {i: "axon_batch" for i in sized if i in free and i in axon}
    lanes = _lane_held([i for i in sized if i not in out], data)
    return out | {i: "live_lane" for i in lanes}


def waiting(data: DataLayout, cohort: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The cohort's big rows ``type_waiting`` would type, each with its projected GiB, and the
    rows it leaves alone, by reason."""

    sized = _big_at_trace(data, cohort, cfg)
    why = _why_skipped(sized, data) if sized else {}
    skipped: dict[str, list[str]] = {"traced_or_open_job": [], "axon_batch": [], "live_lane": []}
    for model_id, reason in sorted(why.items()):
        skipped[reason].append(model_id)
    return {"rows": {i: gib for i, gib in sized.items() if i not in why},
            "skipped": skipped if sized else {}}


def type_waiting(*, data: DataLayout, cohort: str, apply: bool = False,
                 cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Type ``big_row_deferred`` on each of ``cohort``'s waiting big rows (``waiting``).
    Returns ``{"held": [{"id", "gib", "why"}], "typed": n, "skipped": {...}}``; writes the
    records only with ``apply``."""

    from menagerie.pipeline import flow

    if not cohort:
        raise ValueError("big_row_deferred is typed one cohort at a time: name it with --cohort")
    found = waiting(data, cohort, _cfg(cfg))
    out = []
    for model_id, gib in found["rows"].items():
        why = (f"projected trace peak {gib} GiB is past the worker's {flow.WORKER_GIB:g} GiB and "
               f"neither runner G nor an axon batch reached it before the {cohort} close "
               "(Lead 2026-11-04); reopened when G has room")
        out.append({"id": model_id, "gib": gib, "why": why})
        if apply:
            path = data.model(model_id)
            write_json(replace(read_json(ModelRecord, path), give_up=GiveUp(
                cls=CLASS, reason=why, at=now(), by="script")), path)
    return {"held": out, "typed": len(out) if apply else 0, "skipped": found["skipped"]}


def _cfg(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if cfg is not None:
        return cfg
    from menagerie.pipeline import tick

    return tick.config()


def deferred(data: DataLayout, cohort: str, cfg: Mapping[str, Any]) -> dict[str, float]:
    """The cohort's rows typed ``big_row_deferred``, each with its projected GiB."""

    out, need_of = {}, None
    for model_id in _cohort_ids(cohort, cfg):
        path = data.model(model_id)
        if not path.is_file():
            continue
        record = read_json(ModelRecord, path)
        if record.give_up is not None and record.give_up.cls == CLASS:
            need_of = need_of or _sizer(data)
            out[model_id] = need_of(record) or 0.0
    return out


def g_room(data: DataLayout, cfg: Mapping[str, Any]) -> int:
    """Rows G could take now: spool-G's free queue room less the big rows still waiting for it."""

    from menagerie.pipeline import tick_full

    ids = tick_full.worklist(cfg)
    want = [i for i, step in tick_full._steps(ids, data, cfg).items()  # same package
            if step in tick_full.RUN_STEPS]
    reserved = tick_full._not_pending(tick_full._big_candidates(data, want, cfg=cfg), data)  # same package
    room = int(cfg["queue_depth"]) - tick_full._waiting(SPOOL_G)  # same package
    return max(0, room - len(reserved))


def reopen(*, data: DataLayout, cohort: str, apply: bool = False, all_rows: bool = False,
           cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Take ``big_row_deferred`` back from as many of ``cohort``'s rows as G has room for
    (``g_room``), smallest projection first, or from every one with ``all_rows``. Returns
    ``{"room": n, "deferred": n, "reopened": [...]}``; writes only with ``apply``."""

    if not cohort:
        raise ValueError("big_row_deferred is reopened one cohort at a time: name it with --cohort")
    cfg = _cfg(cfg)
    rows = deferred(data, cohort, cfg)
    if not rows:
        return {"room": None, "deferred": 0, "reopened": []}
    room = len(rows) if all_rows else g_room(data, cfg)
    chosen = sorted(rows, key=lambda i: (rows[i], i))[:room]
    if apply:
        for model_id in chosen:
            path = data.model(model_id)
            write_json(replace(read_json(ModelRecord, path), give_up=None), path)
    return {"room": room, "deferred": len(rows), "reopened": chosen}
