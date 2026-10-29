"""The tick's describe stage: one ``describe write`` pass at a time, outside the tick's process.

``tick_full.describes`` calls ``stage`` on each full pass while config.json's
``describes.enabled`` is true. Staging a seat (saved texts from the store, one metered classifier
run per ten families) can take longer than a tick pass may spend, and the tick's oneshot unit
kills what it leaves behind, so the pass starts ``describe write`` as its own transient user unit
(``UNIT``) and returns at once; while that unit is active the next pass only reports it. One run
collects ended seats (grounded records, one return lane, ``describe:ungrounded`` after it), then
reserves and starts new seats of ten families on disjoint leases. The automated checks are
``describe_write``'s own; no human step. The tick's ``lanes`` step commits the records.

Config (``describes`` in the tick's config.json), each passed to the run's environment:

- ``lanes``: seats at once (default and ceiling ``admission.lane_cap("describe")``, 4).
- ``hub_reserve``: hub slots a new seat leaves free for 2a and review lanes (default 6: the 2a
  cap of 4 plus the review cap of 2).
- ``max_week_pct``: the Claude week-use ceiling for describe seats (default 60), under the
  unattended 70, so 2a and the Claude 2b lanes keep the band above it.
- ``earlier_pins``: families validated only at an earlier TorchLens pin follow the current pin's
  (default true).
- ``abstracts``: families without a full text may ground on the cited paper's saved S2 or
  OpenAlex abstract: false (default), true for all, or a list of family ids (a trial, described
  first).
- ``workers``: seats may run on the 2b worker runtimes, 2b first, with no hub-slot gate; the hub
  is the fallback (``describe_workers``; default false: hub only). ``worker_seats``: the most
  seats on workers at once (default and ceiling ``lanes``).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, StoreLayout, admission, describe_workers, describe_write, now

UNIT = "menagerie-describe-stage"
RUNTIME_S = 3600  # a run stops after this; the next tick starts another
REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {"hub_reserve": 6, "max_week_pct": 60, "earlier_pins": True, "abstracts": False,
            "workers": False}
PASS_ENV = ("PATH", "PYTHONPATH")  # with every MENAGERIE_ variable

Run = Callable[..., subprocess.CompletedProcess[str]]


def _state() -> Path:
    return work_dir(describe_write.KIND) / "stage.json"


def _last() -> dict[str, Any] | None:
    path = _state()
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def knobs(given: Mapping[str, Any]) -> dict[str, str]:
    """The run's environment words from the config's ``describes`` (``DEFAULTS`` fill gaps)."""

    cap = admission.LANE_CAP_MAX[describe_write.KIND]
    lanes = max(0, min(int(given.get("lanes", admission.lane_cap(describe_write.KIND))), cap))
    cfg = {**DEFAULTS, **given}
    for name in ("earlier_pins", "workers"):
        if not isinstance(cfg[name], bool):
            raise TypeError(f"describes.{name} must be true or false")
    words = {"MENAGERIE_LANES_DESCRIBE": str(lanes),
             describe_write.HUB_RESERVE: str(int(cfg["hub_reserve"])),
             "MENAGERIE_DESCRIBE_MAX_PCT": str(int(cfg["max_week_pct"])),
             describe_write.EARLIER_PINS: "1" if cfg["earlier_pins"] else "0",
             describe_write.ABSTRACTS: _abstracts(cfg["abstracts"])}
    if cfg["workers"]:
        words[describe_workers.WORKERS] = str(max(0, min(int(cfg.get("worker_seats", lanes)), lanes)))
    return words


def _abstracts(value: Any) -> str:
    if isinstance(value, bool):
        return "all" if value else ""
    if isinstance(value, list) and all(isinstance(f, str) and f and "," not in f for f in value):
        return ",".join(value)
    raise TypeError("describes.abstracts must be true, false or a list of family ids")


def _active(run: Run) -> bool:
    done = run(["systemctl", "--user", "is-active", "--quiet", f"{UNIT}.service"],
               capture_output=True, text=True, timeout=30, check=False)
    return done.returncode == 0


def _idle(words: Mapping[str, str], slots: Callable[[], int]) -> bool:
    """Whether a run could do nothing: every seat is reserved and staged, none has a lane to
    start or collect, the hub has no slot past the reserve and, with workers on, no worker has
    describe room (``describe_workers.hosts``)."""

    folders = describe_write._folders()
    if len(folders) < int(words["MENAGERIE_LANES_DESCRIBE"]):
        return False
    if any((f / "lane.json").is_file() or not (f / "ready.json").is_file() for f in folders):
        return False
    if describe_workers.WORKERS in words and describe_workers.hosts(int(words[describe_workers.WORKERS])):
        return False
    return slots() <= int(words[describe_write.HUB_RESERVE])


def stage(given: Mapping[str, Any], *, dry_run: bool, run: Run = subprocess.run,
          slots: Callable[[], int] | None = None) -> dict[str, Any]:
    """Start one detached describe run unless one is active or it could do nothing (``_idle``);
    report the last run's outcome."""

    from menagerie.pipeline import tick_full

    words = knobs(given)
    last = _last()
    if _active(run):
        return {"running": UNIT, "last": last}
    if _idle(words, slots or tick_full.hub_slots):
        return {"held": "every seat is staged and the hub has no slot past the reserve",
                "last": last}
    if dry_run:
        return {"would_start": words, "last": last}
    env = {**{k: v for k, v in os.environ.items() if k.startswith("MENAGERIE_")},
           **{k: os.environ[k] for k in PASS_ENV if k in os.environ}, **words}
    argv = ["systemd-run", "--user", f"--unit={UNIT}", "--collect", "--quiet",
            f"--property=RuntimeMaxSec={RUNTIME_S}", "--property=CPUWeight=20",
            "--property=IOWeight=20", f"--working-directory={REPO}",
            *(f"--setenv={k}={v}" for k, v in sorted(env.items())),
            "/bin/bash", "-lc", "exec python3 -m menagerie.pipeline.describe_stage"]
    done = run(argv, capture_output=True, text=True, timeout=60, check=False)
    if done.returncode:
        return {"held": f"systemd-run rc {done.returncode}: {done.stderr.strip()[-300:]}",
                "last": last}
    return {"started": UNIT, "knobs": words, "last": last}


def _why(row: Mapping[str, Any]) -> str:
    """Why a seat did not start: its refusal's kind and reason (``describe_write._sweep_folder``)."""

    why = row["why"]
    return f"{why.get('result')}: {why.get('why')}" if isinstance(why, Mapping) else str(why)


def main() -> int:
    """One ``describe write`` pass; its verdict counts and new runs go to ``stage.json`` and one
    line of ``stage.jsonl`` beside it."""

    started = now()
    try:
        rows = describe_write.write(None, data=DataLayout.from_env(), store=StoreLayout.from_env())
        result: dict[str, Any] = {
            "verdicts": dict(Counter(r["verdict"] for r in rows)),
            "launched": sorted({r["run"] for r in rows if r["verdict"] in ("launched", "returned")}),
            "written": sorted(r["family"] for r in rows if r.get("written")),
            "held": dict(Counter(_why(r) for r in rows if r.get("why")))}
    except Exception as exc:  # noqa: BLE001  (recorded for the tick's status, then exit 1)
        result = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    out = {"started": started, "finished": now(), **result}
    path = _state()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    with path.with_suffix(".jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(out, separators=(",", ":")) + "\n")
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
