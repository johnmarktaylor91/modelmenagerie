"""Reviewer calibration: the case set, the calibration lanes, their scores and each seat's standing.

Owned by the review lane (rung 2). Runs on the hub: imports neither torch nor psutil.

A seat is a reviewer role at one prompt hash (``review.prompt_version()``) on one case set (the
hash of the counted cases and their labels). It runs the set ``RUNS`` times in fresh lanes, each
with its own opaque case names and order, and stands ``passed`` when its lowest run agrees with
the labels on at least ``bar(n)`` cases (85 percent, 35 of 41) and at most ``MAX_FLIPS`` cases
change their call across the runs (BUILD-PLAN r2 section 5). Exactly at the bar it stands
``provisional``: it may review, and each ``fabrication`` it returns is confirmed by a second review
before a repair starts. Agreement and flips are on the fabrication line only (``minor`` and
``faithful`` are one call). A changed prompt or a relabelled case is a new seat, so its reviews
count only after three new runs.

Cases live in ``data/review/calibration-cases.jsonl``: schema 1 (the July cases: a paper text, a
card and optionally a classics module) and schema 2 (rung 1 and constructed cases: ``files`` maps
each case-folder path to a store blob, a code snapshot unpacking to a folder). Scores append to
``data/review/calibration.jsonl``, one row per role, prompt, set, run and case.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout, now
from menagerie.pipeline import review as review_mod

RUNS = 3
BAR = 0.85  # rung 1's rate: 23 of 27, and 35 of 41
MAX_FLIPS = 4
RETRIES = 2  # relaunches of a run whose lanes left cases unanswered
CLASSICS = Path(__file__).resolve().parents[1] / "classics"


def bar(cases: int) -> int:
    """The fewest agreeing cases a run may have: 85 percent, rounded up."""

    return math.ceil(BAR * cases - 1e-9)


# --- the case set --------------------------------------------------------------------------


def cases(data: DataLayout) -> list[dict[str, Any]]:
    """The counted calibration cases; an ``excluded`` case names why it is not counted."""

    path = data.root / "review" / "calibration-cases.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [case for case in rows if not case.get("excluded")]


def set_id(counted: Sequence[dict[str, Any]]) -> str:
    """The case set's identity: its cases and their labels, so a relabel is a new set."""

    labels = sorted(f"{c['case']}={c['expected']}" for c in counted)
    return hashlib.sha256("\n".join(labels).encode()).hexdigest()[:8]


def _module_text(module: str) -> str:
    """A classics module with its leading provenance comments dropped (they name the July arm)."""

    lines = (CLASSICS / f"{module}.py").read_text(encoding="utf-8").splitlines(keepends=True)
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return "".join(lines)


def case_folder(case: dict[str, Any], folder: Path, store: StoreLayout) -> None:
    """Write one case's folder as a reviewer sees it."""

    if case.get("schema") == "review-case/2":
        for rel, sha in sorted(case["files"].items()):
            review_mod._put_file(sha, folder / rel, store)
        return
    review_mod._put_file(case["text"], folder / "source" / "paper.txt", store)
    review_mod._put_file(case["card"], folder / "ours" / "card.md", store)
    if case.get("code"):
        (folder / "ours" / f"{case['code']}.py").write_text(_module_text(case["code"]),
                                                             encoding="utf-8")


def _blind(counted: Sequence[dict[str, Any]], key: str) -> dict[str, str]:
    """Opaque folder names in an order shuffled per run, so two builds of one model sit apart."""

    order = sorted(counted, key=lambda c: hashlib.sha256(f"{key}:{c['case']}".encode()).digest())
    return {c["case"]: f"c{i + 1:02d}" for i, c in enumerate(order)}


# --- scores and standing -------------------------------------------------------------------


def rows(data: DataLayout) -> list[dict[str, Any]]:
    path = data.root / "review" / "calibration.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _seat_runs(role: str, prompt: str, sid: str, data: DataLayout) -> dict[int, dict[str, Any]]:
    """The seat's scored runs: run number to ``{case: row}``."""

    runs: dict[int, dict[str, Any]] = {}
    for row in rows(data):
        if (row["role"], row["prompt"], row.get("set")) == (role, prompt, sid):
            runs.setdefault(row["run"], {})[row["case"]] = row
    return runs


def _call(given: str | None) -> str:
    return "none" if given is None else "fabrication" if given == "fabrication" else "other"


def standing(role: str, prompt: str, *, data: DataLayout) -> dict[str, Any]:
    """A seat's standing on the current case set at ``prompt``.

    Returns ``{"role", "prompt", "set", "state", "runs", "agree", "lowest", "bar", "flips"}``;
    ``state`` is ``passed``, ``provisional`` (lowest run exactly at the bar), ``failed`` or
    ``uncalibrated`` (fewer than ``RUNS`` scored runs).
    """

    counted = cases(data)
    sid = set_id(counted)
    runs = _seat_runs(role, prompt, sid, data)
    agree = [sum(r["agree"] for r in run.values()) for _, run in sorted(runs.items())]
    calls = {c["case"]: {_call(run.get(c["case"], {}).get("given")) for run in runs.values()}
             for c in counted}
    flips = sorted(case for case, seen in calls.items() if len(seen) > 1)
    need = bar(len(counted))
    out = {"role": role, "prompt": prompt, "set": sid, "cases": len(counted), "runs": len(runs),
           "agree": agree, "lowest": min(agree, default=None), "bar": need, "flips": flips}
    if len(runs) < RUNS:
        return {**out, "state": "uncalibrated"}
    if min(agree) < need or len(flips) > MAX_FLIPS:
        return {**out, "state": "failed"}
    return {**out, "state": "provisional" if min(agree) == need else "passed"}


def counts(role: str, prompt: str, *, data: DataLayout) -> bool:
    """Whether a seat's reviews count at ``prompt``: it passed, or passed provisionally."""

    return standing(role, prompt, data=data)["state"] in ("passed", "provisional")


def _score(role: str, prompt: str, run: int, folder: Path, counted: Sequence[dict[str, Any]],
           data: DataLayout) -> dict[str, Any]:
    """Score one finished run; append its rows once per role, prompt, set and run."""

    sid = set_id(counted)
    names = json.loads((folder / "names.json").read_text(encoding="utf-8"))
    scored, reasons = [], {}
    for case in counted:
        path = folder / "out" / f"{names[case['case']]}.json"
        try:
            review = review_mod.check_review(json.loads(path.read_text(encoding="utf-8")))
            given, reasons[case["case"]] = review["verdict"], review.get("summary", "")
        except (OSError, ValueError) as err:
            given, reasons[case["case"]] = None, f"no usable review: {err}"
        agree = given is not None and (given == "fabrication") == (case["expected"] == "fabrication")
        scored.append({"role": role, "prompt": prompt, "set": sid, "run": run, "case": case["case"],
                       "expected": case["expected"], "given": given, "agree": agree})
    if run not in _seat_runs(role, prompt, sid, data):
        path = data.root / "review" / "calibration.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in scored)
    wrong = [r for r in scored if not r["agree"]]
    return {"run": run, "agree": len(scored) - len(wrong),
            "disagreements": [{"case": r["case"], "expected": r["expected"], "given": r["given"],
                               "reason": reasons[r["case"]]} for r in wrong]}


# --- calibration lanes ---------------------------------------------------------------------


def _folder(role: str, prompt: str, sid: str, run: int,
            counted: Sequence[dict[str, Any]] = ()) -> Path:
    """A run's folder. After a relabel (a new set with the same cases) the run already made for
    the old labels is scored again under the new ones instead of being run again."""

    root = review_mod.work_root() / "calibration"
    folder = root / f"{role}-{prompt}-{sid}-r{run}"
    if folder.is_dir() or not counted:
        return folder
    names = {c["case"] for c in counted}
    for old in sorted(root.glob(f"{role}-{prompt}-*-r{run}")):
        mapped = old / "names.json"
        if mapped.is_file() and set(json.loads(mapped.read_text(encoding="utf-8"))) == names:
            return old
    return folder


def _prepare(role: str, prompt: str, run: int, counted: Sequence[dict[str, Any]],
             folder: Path) -> tuple[Path, list[str]]:
    """Write one run's case folders under opaque names and its brief; return both."""

    store, names = StoreLayout.from_env(), _blind(counted, f"{prompt}:{run}")
    for case in counted:
        case_folder(case, folder / "cases" / names[case["case"]], store)
    (folder / "names.json").write_text(json.dumps(names, indent=1) + "\n", encoding="utf-8")
    blind = sorted(names.values())
    return review_mod._brief(role, folder, blind), blind


def _clear(path: Path) -> None:
    """Delete a case tree whose snapshot files are read-only (its folders made writable first)."""

    for parent, _, _ in os.walk(path):
        Path(parent).chmod(Path(parent).stat().st_mode | 0o700)
    shutil.rmtree(path)


def _launch(role: str, prompt: str, run: int, counted: Sequence[dict[str, Any]],
            folder: Path) -> str:
    if (folder / "cases").is_dir():  # a start that was refused left its case folders behind
        _clear(folder / "cases")
    brief, blind = _prepare(role, prompt, run, counted, folder)
    label = f"review-calibrate-{role}-{prompt}-r{run}"
    # No keys: calibration cases are not models, and seats review the same cases at once.
    return review_mod._start(role, brief, blind, label, keys=(), prompt=prompt)


def _run_state(folder: Path) -> tuple[str, str]:
    return review_mod.lane_state(folder)


def _failed(folder: Path, run: int, role: str, prompt: str, sid: str, data: DataLayout) -> bool:
    """An unscored run whose lanes ended with cases unanswered, tried fewer than ``RETRIES``
    times: a lane failed (a refused tool, a crash), so it runs again rather than scoring the gaps
    as disagreements. After the last try the run is scored as it stands."""

    if run in _seat_runs(role, prompt, sid, data):
        return False
    tries = len(list(folder.parent.glob(f"{folder.name}.failed-*")))
    answered = len(list((folder / "out").glob("*.json")))
    ids = json.loads((folder / "ids.json").read_text(encoding="utf-8"))
    return answered < len(ids) and tries < RETRIES


def _relaunch(role: str, prompt: str, run: int, counted: Sequence[dict[str, Any]],
              folder: Path) -> str:
    """Keep the failed run's folder beside the new one (``-r<n>.failed-<k>``), then launch."""

    tries = len(list(folder.parent.glob(f"{folder.name}.failed-*")))
    folder.rename(folder.with_name(f"{folder.name}.failed-{tries + 1}"))
    return _launch(role, prompt, run, counted, folder)


def calibrate(*, role: str, data: DataLayout, runs: int = RUNS) -> dict[str, Any]:
    """Run a reviewer seat ``runs`` times on the case set at the current prompt and score it.

    Inputs: a reviewer role and how many runs to hold (rung 2: 3). Returns ``{"role",
    "prompt", "set", "state", "runs": [...], ...standing}``; ``state`` is ``launched`` when
    this call started a run, ``running`` while any run is live, else the seat's standing.
    Side effects: calibration lanes, their case folders in the work root and score rows under
    ``data/review/``.

    Idempotent per role, prompt, set and run: a run starts once, is scored once when its lane
    ends, and a later call only reports. A refused launch raises ``drive.Refused``; the runs
    already started keep going. A read-only seat's run whose later batches were refused is
    ``waiting``, and a later call starts them (``review.resume``) before the run is scored.
    Each seat's lanes start only on the lab it is calibrated for (``review.SEAT_LAB``).
    """

    if role not in review_mod.REVIEWER_ROLES:
        raise review_mod.ReviewError(f"{role!r} is not a reviewer role")
    counted, prompt = cases(data), review_mod.prompt_version()
    if not counted:
        raise review_mod.ReviewError(f"no calibration cases under {data.root / 'review'}")
    sid = set_id(counted)
    report = [_step(role, prompt, sid, run, counted, data) for run in range(1, runs + 1)]
    seat = standing(role, prompt, data=data)
    live = [r for r in report if r["state"] in ("launched", "running", "waiting")]
    launched = any(r["state"] == "launched" for r in report)
    state = "launched" if launched else "running" if live else seat["state"]
    return {**seat, "state": state, "at": now(), "run_reports": report}


def _step(role: str, prompt: str, sid: str, run: int, counted: Sequence[dict[str, Any]],
          data: DataLayout) -> dict[str, Any]:
    """Move one run on: launch it, start what a refusal left, relaunch a failed lane, wait for a
    live one, or score it once its lanes ended. Returns the run's report."""

    folder = _folder(role, prompt, sid, run, counted)
    if not (folder / "lane.json").is_file():
        return {"run": run, "lane": _launch(role, prompt, run, counted, folder),
                "state": "launched"}
    if review_mod.unstarted(folder):  # a read-only seat's batches a refusal left
        started = review_mod.resume(folder)
        return {"run": run, "state": "launched" if started else "waiting",
                "unstarted": review_mod.unstarted(folder)}
    lane, state = _run_state(folder)
    if state == "TERMINAL" and _failed(folder, run, role, prompt, sid, data):
        return {"run": run, "lane": _relaunch(role, prompt, run, counted, folder),
                "state": "launched", "replaces": lane}
    if state != "TERMINAL":
        return {"run": run, "lane": lane, "state": "running"}
    return {"lane": lane, "state": "scored", **_score(role, prompt, run, folder, counted, data)}
