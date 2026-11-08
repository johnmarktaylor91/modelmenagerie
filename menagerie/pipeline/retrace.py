"""The low-priority re-trace feed (tick step ``retrace``): rows to trace again, a few at a time,
into runner B's spool while it has nothing else to do.

The rows are listed one id per line in ``todo.txt`` under the feed's folder (default the work
root's ``retrace/``). A full tick pass feeds the next ``batch`` (default 8) through ``stage3
queue --again`` only when runner B's step ended ``ok`` this pass, that pass's ``stage3_queue``
queued no live rows and was not held, B's spool has nothing waiting, and no worker pause is set
(``hold``). Fed ids leave ``todo.txt`` and are appended to ``queued.txt`` (``<id>\\tqueued`` or
``skipped`` when ``stage3 queue`` took none of its jobs); ids a live lane holds wait for a later
batch. The tick config's ``retrace`` (``{"enabled": true, "batch": 8, "folder": <path>}``)
switches it on; with an empty ``todo.txt`` it does nothing.

``pin.txt`` in the same folder goes first: the TorchLens failure corpus rows the tick step
``torchlens_corpus`` lists when their failing trace ran at another TorchLens than the pin. They
are fed trace-only (the settled forward is kept) and without ``again``, so a row already traced at
the pin is skipped rather than traced twice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout

SPOOL = "scratch/menagerie/spool-B"  # runner B's, where ``stage3_queue`` puts live rows (``queue_spool``)
BATCH = 8
LOOKAHEAD = 3  # batches of todo.txt read to fill one batch past ids a lane holds
PIN_LIST = "pin.txt"  # corpus rows to trace at a new TorchLens pin, fed before todo.txt


def config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The feed's switch, batch size and folder from the tick config's ``retrace``."""

    given = cfg.get("retrace") or {}
    if not isinstance(given, Mapping):
        raise ValueError("retrace must be an object such as {\"enabled\": true}")  # noqa: TRY004 -- config errors are ValueError, as the tick's others
    batch = given.get("batch", BATCH)
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("retrace.batch must be a whole number of rows, 1 or more")
    folder = Path(given["folder"]) if given.get("folder") else work_dir("retrace")
    return {"enabled": given.get("enabled") is True, "batch": batch, "folder": folder}


def hold(cfg: Mapping[str, Any], results: Mapping[str, Any], *, paused: str | None,
         disk_held: bool) -> str | None:
    """Why nothing is fed this pass, else None: the feed is off or paused, the disk guard holds
    queueing, runner B is not ``ok``, or this pass's ``stage3_queue`` queued live rows, was held
    or failed (``results`` is the pass's results so far)."""

    if not config(cfg)["enabled"]:
        return "off"
    if paused:
        return paused
    if disk_held:
        return "the disk guard holds queueing"
    runner_b = next((r for r in results.get("runners") or [] if r.get("runner") == "B"), None)
    if runner_b is None or runner_b.get("action") != "ok":
        return f"runner B is {runner_b.get('action') if runner_b else 'not read'}"
    live = results.get("stage3_queue")
    if not isinstance(live, Mapping) or "held" in live:
        return "stage3_queue did not run cleanly this pass"
    if live.get("queued"):
        return f"this pass queued {len(live['queued'])} live jobs"
    return None


def _ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.is_file() else []


def feed(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool,
         waiting: Callable[[str], int] | None = None,
         holds: Callable[[Sequence[str], DataLayout], Mapping[str, str]] | None = None,
         queue: Callable[..., list[str]] | None = None) -> dict[str, Any]:
    """Feed the next batch of ``todo.txt`` into B's spool when it has nothing waiting."""

    from menagerie.pipeline import stage3, tick_full

    conf = config(cfg)
    done_path = conf["folder"] / "queued.txt"
    lists = ((conf["folder"] / PIN_LIST, {"trace_only": True}),
             (conf["folder"] / "todo.txt", {"again": True}))
    todo_path, how = next(((p, kw) for p, kw in lists if _ids(p)), lists[-1])
    todo = _ids(todo_path)
    if not todo:
        return {"left": 0}
    which = {"list": PIN_LIST} if todo_path.name == PIN_LIST else {}
    spool = cfg.get("queue_spool") or SPOOL
    queued_now = (waiting or tick_full._waiting)(spool)
    if queued_now:
        return {"held": f"spool B has {queued_now} waiting", "left": len(todo), **which}
    head = todo[:conf["batch"] * LOOKAHEAD]
    held = (holds or stage3.lane_holds)(head, data)
    batch = [i for i in head if i not in held][:conf["batch"]]
    if dry_run or not batch:
        return {"would_feed": batch, "left": len(todo), **which}
    names = (queue or stage3.queue)(None, data=data, ids=batch, spool=spool, **how)
    fed = {n.split("-")[0] for n in names}
    with done_path.open("a", encoding="utf-8") as out:
        out.writelines(f"{i}\t{'queued' if i in fed else 'skipped'}\n" for i in batch)
    left = [i for i in todo if i not in batch]
    todo_path.write_text("".join(f"{i}\n" for i in left), encoding="utf-8")
    return {"fed": sorted(fed), "skipped": sorted(set(batch) - fed), "left": len(left), **which}
