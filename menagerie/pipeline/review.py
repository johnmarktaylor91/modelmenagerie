"""Fidelity review: choosing reviewers, calibrating them and recording their verdicts.

Owned by the review lane (rung 1). Every port, reimplementation, classic and unofficial vendor
recipe that passes stage 3 is reviewed by a role from the other lab than the one that wrote it.
Runs on the hub: imports neither torch nor psutil.

The reviewer is chosen from what actually ran, never from the role a lane asked for: the author's
harness comes from its fleet-run record (and a handoff to the other lab's fallback counts both
labs), each reviewer role's harness from the role table at the time of the call. The written bar
lives in ``prompts/review.md``; its sha256 prefix is the prompt version every calibration row
carries, so a changed bar is a new calibration. A seat reviews, and ``record`` takes its verdict,
only while its calibration at the prompt in force counts (``calibration.py``).

Review work lives under the work root's ``review/`` (``~/scratch/menagerie/review``): one folder
per lane holding its case folders, brief, ``out/<id>.json`` verdicts and the lane's result file. Saved originals
come from the store by sha (``store.materialize``); nothing in a case folder is committed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, StoreLayout, now, read_json, tree_sha256
from menagerie.pipeline import admission, bib_harvest, calibration, drive, implement, ledger
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import (
    LedgerLine, LedgerRow, ModelRecord, Packet, RecordError, to_dict,
)

MAIN_ROLES = ("model-design-review", "model-default")  # tried in this order
THIRD_LAB = "model-review-third-lab-plan"  # Gemini on the subscription, read-only
REVIEWER_ROLES = (*MAIN_ROLES, THIRD_LAB)
VERDICTS = ("faithful", "minor", "fabrication", "cannot_judge")
FINDINGS = ("match", "minor", "missing", "invented", "unclear")
BATCH = 5
# The lab each seat was calibrated on: its lanes start only there (the lab guard), so a role that
# routing moves to the other lab is refused and waits rather than reviewing as another seat.
SEAT_LAB = {"model-design-review": "codex", "model-default": "claude", THIRD_LAB: "antigravity"}
PROVISIONAL = "provisional seat: a second review confirms this fabrication before a repair"
PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "review.md"
CLASSICS = Path(__file__).resolve().parents[1] / "classics"
HANDED_OFF = re.compile(r"lane (\S+) handed off to (\S+)")


class ReviewError(RuntimeError):
    """A review input or lookup that cannot be used as given."""


# --- picking the reviewer ------------------------------------------------------------------


def _json_command(argv: Sequence[str]) -> dict[str, Any]:
    done = subprocess.run(list(argv), capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise ReviewError(f"{' '.join(argv[:3])}: exit {done.returncode}: {done.stderr.strip()[:200]}")
    return json.loads(done.stdout)


def _fleet_run(run_id: str) -> dict[str, Any]:
    """The run's fleet-run row and its launch spec."""

    row = _json_command(["fleet-run", "status", run_id, "--json"])
    row = row.get("run", row)
    spec_path = Path(row.get("spec_path") or "")
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.is_file() else {}
    return {**row, "spec": spec}


def _role_harness(role: str) -> str:
    return str(_json_command(["role", "resolve", role, "--json"])["harness"])


def _option(argv: Sequence[str], *names: str) -> str | None:
    for flag, value in zip(argv, argv[1:]):
        if flag in names:
            return value
    return None


def _handed_off(run_id: str) -> str | None:
    """The fallback component a lane was handed to, from the failover log, if it was."""

    log = Path(os.environ.get("LANE_FAILOVER_LOG")
               or Path.home() / ".local/state/fleet/lane-failover.log")
    if not log.is_file():
        return None
    found = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = HANDED_OFF.search(line)
        if match and match.group(1) == run_id:
            found = match.group(2)
    return found


def author_of(run_id: str) -> dict[str, Any]:
    """What actually ran an authoring lane: ``{"run", "harness", "model", "handed_off_to"}``.

    The harness is the adapter the launch spec ran (``fleet-run-adapter <harness> BRIEF``), else
    the run's adapter or kind, else (a worker or command lane) the lane's launch event; the model
    is the spec's ``--model`` or the event's. A lane handed to another lab's
    fallback mid-run names that fallback in ``handed_off_to``.
    """

    run = _fleet_run(run_id)
    argv = [str(a) for a in run["spec"].get("argv", [])]
    harness = next((argv[i + 1] for i, a in enumerate(argv[:-1])
                    if a.endswith("fleet-run-adapter")), None)
    harness = harness or run.get("adapter") or run.get("kind")
    model = _option(argv, "--model", "-m")
    if harness not in AUTHOR_HARNESSES:  # a worker (run-on) or command lane: its launch event
        launched = _launch_event(run_id)
        harness, model = launched.get("harness"), launched.get("model")
    if harness not in AUTHOR_HARNESSES:
        raise ReviewError(f"run {run_id} is a {harness!r} run, not an agent lane")
    return {"run": run_id, "harness": harness, "model": model, "handed_off_to": _handed_off(run_id)}


AUTHOR_HARNESSES = ("claude", "codex", "antigravity")


def _launch_event(run_id: str) -> dict[str, Any]:
    """The lane's ``launch`` event (``drive.launch``): it names the harness that ran the lane when
    the lane started on a worker through run-on or as a command run."""

    log = work_dir("lanes") / run_id / "events.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines() if log.is_file() else []
    return next((e for e in map(json.loads, filter(str.strip, lines)) if e.get("verb") == "launch"), {})


def pick_reviewer(run_id: str, *, data: DataLayout, exclude: frozenset[str] = frozenset()) -> str:
    """Name the reviewer role for the model an implement lane built.

    Inputs: the 2b lane's run id. Returns: a role from the other lab than the one the
    launcher's record says actually ran. Side effects: none.

    The first main role whose current harness differs from every harness that authored the lane
    and whose calibration counts at the prompt in force; else the third-lab seat if its
    calibration counts; else it raises, and the review waits for a calibrated seat. A lane handed
    off to the other lab's fallback was authored by both main labs, so it goes to the third lab.
    ``exclude`` names seats that already reviewed it (a second review needs another seat).
    """

    author = author_of(run_id)
    harnesses = {role: _role_harness(role) for role in MAIN_ROLES}
    authors = {author["harness"]}
    if author["handed_off_to"]:
        authors |= set(harnesses.values())
    prompt = prompt_version()
    for role in (*[r for r in MAIN_ROLES if harnesses[r] not in authors], *_third_lab()):
        if role not in exclude and calibration.counts(role, prompt, data=data):
            return role
    raise ReviewError(f"no calibrated reviewer at prompt {prompt} for a lane authored on "
                      f"{sorted(authors)}")


# --- one review: its JSON and its ledger row -----------------------------------------------


def check_review(review: Any) -> dict[str, Any]:
    """Validate a reviewer's JSON (the shape in ``prompts/review.md``); return it or raise."""

    if not isinstance(review, dict):
        raise RecordError("a review is a JSON object")
    if review.get("verdict") not in VERDICTS:
        raise RecordError(f"review verdict {review.get('verdict')!r} is not one of {VERDICTS}")
    if review.get("reviewer") not in REVIEWER_ROLES:
        raise RecordError(f"review reviewer {review.get('reviewer')!r} is not a reviewer role")
    components = review.get("components")
    if not isinstance(components, list) or not components:
        raise RecordError("a review lists its components")
    for item in components:
        if not isinstance(item, dict) or not item.get("name") or item.get("finding") not in FINDINGS:
            raise RecordError(f"review component {item!r} needs a name and a finding in {FINDINGS}")
    flagged = [c for c in components if c["finding"] in ("missing", "invented")]
    if review["verdict"] == "fabrication" and not flagged:
        raise RecordError("a fabrication verdict names the missing or invented component")
    return review


def _recipe_rev(model_id: str, data: DataLayout) -> str:
    path = data.model(model_id)
    if not path.is_file():
        raise ReviewError(f"{model_id} has no model record")
    model = read_json(ModelRecord, path)
    if model.recipe is None:
        raise ReviewError(f"{model_id} has no recipe to review")
    return model.recipe.rev


def _review_prompt(path: Path) -> str:
    """The prompt hash a verdict was written under: its lane's launch record when the file sits in
    a review lane folder (``<lane>/out/<id>.json``), else the prompt in force. A lane launched
    under another hash is refused: the prompt changed during its wave."""

    current = prompt_version()
    lane = Path(path).resolve().parent.parent / "lane.json"
    launched = json.loads(lane.read_text(encoding="utf-8")).get("prompt") if lane.is_file() else None
    if launched not in (None, current):
        raise RecordError(f"{path} was written under review prompt {launched}; the prompt is now "
                          f"{current}, so its wave ended; review it again under {current}")
    return current


def record(model_id: str, path: Path, *, data: DataLayout, store: StoreLayout) -> LedgerRow:
    """Record one review: the verdict and the findings file.

    Inputs: a stable id and a findings JSON file with its verdict (``faithful``, ``minor``,
    ``fabrication`` or ``cannot_judge``). Returns: the ``review`` ledger row. Side effects:
    the findings blob in the store and the ledger row.

    Refused unless the reviewer's calibration counts at the prompt the review was written under
    (``calibration.counts``), and when that prompt is no longer the one in force. The row is keyed
    on the model's current recipe revision and names the reviewer role in ``reviewer`` (and, as
    before, in ``env``: ``review:<role>``); ``msg`` carries the prompt hash, and a ``fabrication``
    from a provisional seat says it needs a second review (``needs_second_review``).
    """

    review = check_review(json.loads(Path(path).read_text(encoding="utf-8")))
    if review.get("id") not in (None, model_id):
        raise RecordError(f"{path} reviews {review['id']}, not {model_id}")
    prompt = _review_prompt(path)
    seat = calibration.standing(review["reviewer"], prompt, data=data)
    if seat["state"] not in ("passed", "provisional"):
        raise RecordError(f"{review['reviewer']} has no passing calibration at review prompt "
                          f"{prompt} (set {seat['set']}: {seat['state']}, {seat['runs']} runs); "
                          "run `review calibrate` for it first")
    rev = _recipe_rev(model_id, data)
    msg = f"prompt {prompt}"
    if seat["state"] == "provisional" and review["verdict"] == "fabrication":
        msg = f"{msg}; {PROVISIONAL}"
    blob = store_mod.put(Path(path), store=store, data=data, mime="application/json",
                         license_class="derived")
    row = LedgerRow(id=model_id, stage="review", at=now(), host="hub",
                    env=f"review:{review['reviewer']}", recipe_rev=rev, outcome="pass",
                    verdict=review["verdict"], findings=blob.sha256, reviewer=review["reviewer"],
                    msg=msg)
    ledger.append([row], data=data)
    lane = os.environ.get("FLEET_RUN_ID")
    if lane:
        family = read_json(ModelRecord, data.model(model_id)).family
        drive.event(lane, family, "review", id=model_id, verdict=row.verdict,
                    reviewer=review["reviewer"])
    return row


def needs_second_review(row: LedgerRow) -> bool:
    """A ``fabrication`` from a provisionally calibrated seat: confirm it before a repair."""

    return row.verdict == "fabrication" and PROVISIONAL in (row.msg or "")


# --- review lanes --------------------------------------------------------------------------


def prompt_version() -> str:
    return tree_sha256(PROMPT)[:8]


def work_root() -> Path:
    return work_dir("review")


def _put_file(sha: str, dest: Path, store: StoreLayout) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    return store_mod.materialize(sha, dest, store=store)


def _brief(role: str, folder: Path, ids: Sequence[str], note: str = "") -> Path:
    """Write a review lane's brief: the case list, then the written bar."""

    head = [f"# Review lane, role `{role}`", "",
            f"Case folders: `{folder}/cases/<case>/`. Write `{folder}/out/<case>.json` for each case",
            f'below, with `"reviewer": "{role}"`. When every file is written, write',
            f"`{folder}/BRIEF.result.md`: one line per case, `<case> <verdict>`.", note, "",
            "Cases: " + ", ".join(f"`{i}`" for i in ids), "", ""]
    brief = folder / "BRIEF.md"
    brief.write_text("\n".join(head) + PROMPT.read_text(encoding="utf-8"), encoding="utf-8")
    (folder / "out").mkdir(exist_ok=True)
    return brief


LISTED = 40  # files listed per case in a read-only brief; a code tree beyond that is summarised


def _case_files(folder: Path, case: str) -> list[str]:
    """A case's files by path relative to the lane folder, a long code tree cut short."""

    rels = sorted(p.relative_to(folder).as_posix() for p in (folder / "cases" / case).rglob("*")
                  if p.is_file())
    if len(rels) <= LISTED:
        return rels
    return [*rels[:LISTED], f"... and {len(rels) - LISTED} more under cases/{case}/ (list them)"]


def _batch_brief(folder: Path, k: int, batch: Sequence[str], body: str) -> Path:
    """One read-only batch's brief beside the lane brief (so the seat's working directory is the
    lane folder): its cases, every file they hold, then the lane brief."""

    listing = "\n".join(f"- `{f}`" for case in batch for f in _case_files(folder, case))
    path = folder / f"BRIEF-b{k}.md"
    path.write_text(f"# This lane's cases: {', '.join(batch)} (ignore every other case below)\n\n"
                    f"Their files:\n{listing}\n\n{body}", encoding="utf-8")
    return path


def _start_agy(role: str, brief: Path, ids: Sequence[str], label: str,
               keys: Sequence[str] = (), prompt: str | None = None) -> str:
    """Start the read-only third-lab seat on a lane folder, one lane per ``BATCH`` cases (the
    seat answers in a single turn, so a batch is what one turn can read).

    Each batch starts through ``drive.launch`` like every other lane (the Gemini gate, its lease,
    the launch event), on the lab guard ``antigravity``, answering in ``blocks``: its answer lands
    in ``agy-<label>-<time>/result.md`` and ``collect_agy`` turns it into ``out/<case>.json`` once
    the lanes end. ``lane.json`` lists every batch, a batch not started yet with no run, and is
    written after each start: a refusal after the first batch leaves the rest for ``resume``.
    Returns the first lane's id; a refusal of the first batch raises ``drive.Refused``.
    """

    folder = brief.parent
    (folder / "ids.json").write_text(json.dumps(list(ids)) + "\n", encoding="utf-8")
    body = brief.read_text(encoding="utf-8")
    batches = [{"k": k, "ids": list(ids[start:start + BATCH]), "run": None,
                "label": f"{label}-b{k}", "brief": str(_batch_brief(
                    folder, k, ids[start:start + BATCH], body))}
               for k, start in enumerate(range(0, len(ids), BATCH), 1)]
    record = {"role": role, "ids": list(ids), "keys": list(keys), "prompt": prompt,
              "batches": batches}
    _start_batches(folder, record)
    return next(b["run"] for b in batches if b["run"])


def _start_batches(folder: Path, record: dict[str, Any]) -> None:
    """Start the record's batches that have no run; write ``lane.json`` after each start."""

    started = [b["run"] for b in record["batches"] if b["run"]]
    for batch in record["batches"]:
        if batch["run"]:
            continue
        try:
            batch["run"] = drive.launch(
                "review", record["role"], Path(batch["brief"]),
                keys=[i for i in batch["ids"] if i in record["keys"]], label=batch["label"],
                parent=os.environ.get("FLEET_RUN_ID"), lab=SEAT_LAB[THIRD_LAB],
                answer="blocks", ids=batch["ids"])
        except drive.Refused:
            if not started:
                raise
            break  # the rest wait for ``resume``; the started batches keep running
        started.append(batch["run"])
        _lane_record(folder, started[0], record["role"], record["ids"], harness="antigravity",
                     prompt=record["prompt"], runs=started,
                     extra={"batches": record["batches"], "keys": record["keys"]})


def unstarted(folder: Path) -> list[int]:
    """The batches of a read-only lane folder that a refusal left unstarted."""

    lane = json.loads((Path(folder) / "lane.json").read_text(encoding="utf-8"))
    return [b["k"] for b in lane.get("batches", []) if not b["run"]]


def resume(folder: Path) -> int:
    """Start what a refusal left unstarted in a read-only lane folder; return how many batches
    started now (0 while the seat is still refused)."""

    lane = json.loads((Path(folder) / "lane.json").read_text(encoding="utf-8"))
    record = {"role": lane["role"], "ids": lane["ids"], "keys": lane.get("keys", []),
              "prompt": lane.get("prompt"), "batches": lane["batches"]}
    before = len([b for b in record["batches"] if b["run"]])
    _start_batches(Path(folder), record)
    return len([b for b in record["batches"] if b["run"]]) - before


def collect_agy(folder: Path) -> list[str]:
    """Write ``out/<case>.json`` for each case a read-only seat answered; return those cases.

    Idempotent: a case already written is kept. Writes the folder's result file.
    """

    folder = Path(folder)
    ids = json.loads((folder / "ids.json").read_text(encoding="utf-8"))
    text = "\n".join(r.read_text(encoding="utf-8", errors="replace")
                     for r in sorted(folder.glob("agy*/result.md")))
    written = admission.write_blocks(folder, ids, text)
    lines = []
    for case in ids:
        out = folder / "out" / f"{case}.json"
        verdict = json.loads(out.read_text(encoding="utf-8")).get("verdict") if out.is_file() else None
        lines.append(f"{case} {verdict or 'missing'}")
    (folder / "BRIEF.result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def _lane_record(folder: Path, run_id: str, role: str, ids: Sequence[str], *,
                 harness: str | None = None, prompt: str | None = None,
                 runs: Sequence[str] = (), extra: dict[str, Any] | None = None) -> None:
    (folder / "lane.json").write_text(json.dumps(
        {"run": run_id, "runs": list(runs) or [run_id], "role": role, "harness": harness,
         "ids": list(ids), "at": now(), "prompt": prompt or prompt_version(),
         **(extra or {})}) + "\n", encoding="utf-8")


def lane_state(folder: Path) -> tuple[str, str]:
    """A lane folder's first run id and its state: ``TERMINAL`` once every lane it started has
    ended (a read-only seat's answers are then collected), else the first live state."""

    lane = json.loads((Path(folder) / "lane.json").read_text(encoding="utf-8"))
    states = [_fleet_run(run).get("state", "unknown") for run in lane.get("runs", [lane["run"]])]
    live = [s for s in states if s != "TERMINAL"]
    if not live and lane.get("harness") == "antigravity":
        collect_agy(Path(folder))
    return lane["run"], live[0] if live else "TERMINAL"


def _start(role: str, brief: Path, ids: Sequence[str], label: str,
           keys: Sequence[str] | None = None, prompt: str | None = None) -> str:
    """Start a review lane on ``ids`` (leasing ``keys``, by default the ids) and write its launch
    record with the prompt hash it runs under. A reviewer seat starts only on its own lab
    (``SEAT_LAB``); the third-lab seat runs in read-only batches (``_start_agy``)."""

    keys = tuple(ids if keys is None else keys)
    if role == THIRD_LAB:
        return _start_agy(role, brief, ids, label, keys=keys, prompt=prompt)
    (brief.parent / "ids.json").write_text(json.dumps(list(ids)) + "\n", encoding="utf-8")
    run_id = drive.launch("review", role, brief, keys=keys, label=label,
                          parent=os.environ.get("FLEET_RUN_ID"), lab=SEAT_LAB.get(role))
    _lane_record(brief.parent, run_id, role, ids, prompt=prompt)
    return run_id


def _ours(model: ModelRecord, dest: Path, data: DataLayout) -> None:
    recipe = model.recipe
    if recipe is None:
        raise ReviewError(f"{model.id} has no recipe to review")
    dest.mkdir(parents=True, exist_ok=True)
    if recipe.kind == "adapter":
        shutil.copyfile(data.root / recipe.adapter, dest / Path(recipe.adapter).name)
    elif recipe.kind == "classic":  # "menagerie.classics.<module>:<builder>"
        module = (recipe.adapter or "").partition(":")[0].rsplit(".", 1)[-1]
        shutil.copyfile(CLASSICS / f"{module}.py", dest / f"{module}.py")
    (dest / "recipe.json").write_text(json.dumps(to_dict(recipe), indent=1) + "\n",
                                      encoding="utf-8")


def _member(model_id: str, packet: dict[str, Any], dest: Path) -> None:
    """Write the member's stated variant (``ours/member.md``) when the packet names one, so a
    named simplification is judged against that variant; a member with no delta gets no file."""

    delta = next((m.get("delta", "") for m in packet.get("members", [])
                  if m.get("id") == model_id), "")
    if delta.strip():
        (dest / "member.md").write_text(f"# Stated variant of {model_id}\n\n{delta.strip()}\n",
                                        encoding="utf-8")


def _last_forward(model_id: str, rows: Sequence[LedgerLine]) -> dict[str, Any] | None:
    forwards = [r for r in rows if isinstance(r, LedgerRow) and r.id == model_id
                and r.stage in ("forward", "trace")]
    return to_dict(forwards[-1]) if forwards else None


def _model_case(model_id: str, folder: Path, data: DataLayout, store: StoreLayout,
                rows: Sequence[LedgerLine]) -> None:
    """One model's case folder: the packet's saved sources, our code and its run result."""

    model = read_json(ModelRecord, data.model(model_id))
    packet = json.loads(data.packet(model.family).read_text(encoding="utf-8"))
    held = bib_harvest.held_sources(read_json(Packet, data.packet(model.family)))
    for src in packet.get("sources", []):
        if src["sid"] in held:  # a paper link waiting for its spot check
            continue
        if src.get("role") in ("paper", "code", "page") and (src.get("text") or src.get("blob")):
            sha = src.get("text") or src["blob"]
            name = f"{src['sid']}-{src['role']}" + (".txt" if src.get("text") else "")
            _put_file(sha, folder / "source" / name, store)
    _ours(model, folder / "ours", data)
    _member(model_id, packet, folder / "ours")
    run = _last_forward(model_id, rows)
    if run:
        (folder / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")


def _events() -> list[tuple[str, dict[str, Any]]]:
    """Every drive event, with the lane (run id) that logged it."""

    return [(log.parent.name, json.loads(line)) for log in work_dir("lanes").glob("*/events.jsonl")
            for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def _author_run(family: str) -> str | None:
    """The lane that last ``done`` a family's recipe, from the drive event logs."""

    done = [(str(e.get("at", "")), lane) for lane, e in _events()
            if e.get("verb") == "done" and e.get("family") == family]
    return max(done)[1] if done else None


def launch(ids: Sequence[str], *, data: DataLayout,
           roles: dict[str, str] | None = None) -> list[str]:
    """Start review lanes for these models in batches, each on its picked reviewer role.

    Returns: the lane run ids. Side effects: lanes, leases and launch events.

    Models are grouped by reviewer (from the lane that ``done`` their family; a model with no
    recorded author, such as a June classic, goes to the first seat whose calibration counts) and
    started in batches of ``BATCH``, each in a fresh lane with its own case folders. Raises
    ``ReviewError`` before starting anything when a model has no calibrated reviewer. ``roles``
    names reviewers already picked (the tick's ``plan``), so their models skip the lookup.
    """

    store = StoreLayout.from_env()
    rows = ledger.read_all(data=data)
    by_role: dict[str, list[str]] = {}
    for model_id in ids:
        role = (roles or {}).get(model_id)
        if role is None:
            family = read_json(ModelRecord, data.model(model_id)).family
            author = _author_run(family)
            role = pick_reviewer(author, data=data) if author else _any_seat(data)
        by_role.setdefault(role, []).append(model_id)
    runs = []
    for role, members in sorted(by_role.items()):
        for start in range(0, len(members), BATCH):
            batch = members[start:start + BATCH]
            label = f"review-{batch[0]}-{len(batch)}"
            folder = work_root() / "lanes" / f"{label}-{now().replace(':', '')}"
            try:
                for model_id in batch:
                    _model_case(model_id, folder / "cases" / model_id, data, store, rows)
            except (ReviewError, RecordError, OSError):  # leave no half-built lane folder
                calibration._clear(folder)
                raise
            # Fix 3: the lane runs `review record` through its own verb wrapper, as 2b lanes do.
            wrapper = implement.lane_wrapper(folder, data=data, comment="this review lane")
            note = (f"Then run `{wrapper} review record <case> {folder}/out/<case>.json` for "
                    "each case and fix any refusal.")
            if role == THIRD_LAB:  # read-only: its answers are recorded from the hub
                note = ""
            runs.append(_start(role, _brief(role, folder, batch, note), batch, label))
    return runs


def _any_seat(data: DataLayout) -> str:
    prompt = prompt_version()
    for role in (*MAIN_ROLES, *_third_lab()):
        if calibration.counts(role, prompt, data=data):
            return role
    raise ReviewError(f"no calibrated reviewer at prompt {prompt}")


def _third_lab() -> tuple[str, ...]:
    """The third-lab seat, unless the tick config holds it (``admission.third_lab_held``)."""
    return () if admission.third_lab_held() else (THIRD_LAB,)


# --- calibration (calibration.py) ----------------------------------------------------------


def calibration_cases(data: DataLayout) -> list[dict[str, Any]]:
    """The counted calibration cases (``calibration.cases``)."""

    return calibration.cases(data)


def calibrate(*, role: str, data: DataLayout, runs: int | None = None) -> dict[str, Any]:
    """Run and score a reviewer seat on the calibration set (``calibration.calibrate``)."""

    return calibration.calibrate(role=role, data=data, runs=runs or calibration.RUNS)


def calibrate_all(*, data: DataLayout, runs: int | None = None) -> dict[str, Any]:
    """Calibrate every reviewer seat at the current prompt hash (``review calibrate --all``).

    Inputs: the data root and the runs per seat (rung 3: 3). Returns ``{"prompt", "set",
    "seats": {role: calibrate's report, or {"state": "refused", "result", "why"}}, "non_codex"}``,
    ``non_codex`` naming the seats other than Codex whose calibration counts (the Codex-built
    rows wait for one). Side effects: ``calibrate``'s, for each seat.

    Idempotent like ``calibrate``: call it again until every seat has a standing. A seat refused
    admission (or routed to another lab than its own) is reported and skipped; the others go on.
    """

    seats: dict[str, Any] = {}
    for role in REVIEWER_ROLES:
        try:
            seats[role] = calibrate(role=role, data=data, runs=runs)
        except drive.Refused as refused:
            seats[role] = {**refused.result, "role": role, "state": "refused"}
    prompt = prompt_version()
    non_codex = [r for r in REVIEWER_ROLES if SEAT_LAB[r] != "codex"
                 and calibration.counts(r, prompt, data=data)]
    return {"prompt": prompt, "set": calibration.set_id(calibration.cases(data)), "at": now(),
            "seats": seats, "non_codex": non_codex}


# --- the tick's review stage (``tick_full.reviews``) ----------------------------------------
# Each full tick pass moves the stage on without a person: it keeps the seats' calibration at the
# prompt in force going (``calibrate_all`` is idempotent), records what ended lanes answered but
# never recorded (the read-only seat's answers always; any case a lane left), and starts lanes
# for rows at the ``review`` step while the cap has room.

COUNTED = ("passed", "provisional")
RETRY_AFTER_S = 24 * 3600  # a row a lane took up waits this long before another lane may try it
LOOKUPS = 40  # families whose reviewer is looked up per pass (fleet-run and the role table)
SEAT_CACHE_S = 6 * 3600  # how long a family's looked-up reviewer is reused
SETTLED = "SETTLED.json"


def seats(data: DataLayout) -> dict[str, str]:
    """Each reviewer seat's standing at the prompt in force (a held third-lab seat left out)."""

    prompt = prompt_version()
    return {role: calibration.standing(role, prompt, data=data)["state"]
            for role in (*MAIN_ROLES, *_third_lab())}


def _recorded(rows: Sequence[LedgerLine]) -> set[tuple[str, str]]:
    """(id, findings sha) of every review row: a verdict file is recorded once."""

    return {(r.id, r.findings) for r in rows if isinstance(r, LedgerRow) and r.stage == "review"
            and r.findings}


def settle(*, data: DataLayout, store: StoreLayout) -> dict[str, Any]:
    """Record the verdicts ended review lanes left unrecorded; mark each such folder settled.

    Only lanes launched at the prompt in force are read (``record`` refuses older ones). A verdict
    ``record`` refuses is listed with its reason and not retried; the folder is settled either way.
    """

    root, prompt = work_root() / "lanes", prompt_version()
    out: dict[str, Any] = {"recorded": [], "refused": {}, "live": 0}
    recorded = _recorded(ledger.read_all(data=data))
    for folder in sorted(root.glob("review-*")) if root.is_dir() else ():
        lane_file = folder / "lane.json"
        if (folder / SETTLED).exists() or not lane_file.is_file():
            continue
        if json.loads(lane_file.read_text(encoding="utf-8")).get("prompt") != prompt:
            continue
        if lane_state(folder)[1] != "TERMINAL":
            out["live"] += 1
            continue
        mine: dict[str, Any] = {}
        for path in sorted((folder / "out").glob("*.json")):
            if (path.stem, tree_sha256(path)) in recorded:
                continue
            try:
                record(path.stem, path, data=data, store=store)
                out["recorded"].append(path.stem)
            except (RecordError, ReviewError, ValueError) as exc:
                mine[path.stem] = str(exc)[:200]
        out["refused"].update(mine)
        (folder / SETTLED).write_text(json.dumps({"at": now(), "refused": mine}) + "\n", encoding="utf-8")
    return out


def _authors() -> dict[str, str]:
    """Each family's authoring lane: the lane that last ``done`` its recipe (one read of the logs)."""

    best: dict[str, tuple[str, str]] = {}
    for lane, e in _events():
        if e.get("verb") == "done" and e.get("family"):
            at = str(e.get("at", ""))
            if at >= best.get(e["family"], ("", ""))[0]:
                best[e["family"]] = (at, lane)
    return {family: lane for family, (_, lane) in best.items()}


def _seat_cache() -> Path:
    return work_root() / "reviewer-seats.json"


def _reviewers(families: Sequence[str], data: DataLayout) -> dict[str, str | None]:
    """Each family's reviewer role (``None``: no calibrated seat for it), at most ``LOOKUPS`` new
    lookups a pass; a lookup is reused for ``SEAT_CACHE_S`` at the same prompt."""

    path, prompt, stamp = _seat_cache(), prompt_version(), time.time()
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    fresh = {f: v for f, v in cache.items()
             if v.get("prompt") == prompt and stamp - v.get("t", 0) < SEAT_CACHE_S}
    authors = _authors() if any(f not in fresh for f in families) else {}
    out: dict[str, str | None] = {}
    looked = 0
    for family in families:
        if family not in fresh:
            if looked >= LOOKUPS:
                continue
            looked += 1
            try:
                author = authors.get(family)
                role = pick_reviewer(author, data=data) if author else _any_seat(data)
            except ReviewError:
                role = None
            fresh[family] = {"role": role, "prompt": prompt, "t": stamp}
        out[family] = fresh[family]["role"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fresh) + "\n", encoding="utf-8")
    return out


def _taken_up() -> set[str]:
    """Ids a review lane took up in the last ``RETRY_AFTER_S`` (its case folder's age)."""

    root, stamp, found = work_root() / "lanes", time.time(), set()
    for case in root.glob("review-*/cases/*") if root.is_dir() else ():
        try:
            if stamp - case.stat().st_mtime < RETRY_AFTER_S:
                found.add(case.name)
        except FileNotFoundError:
            continue
    return found


def plan(ids: Sequence[str], *, data: DataLayout,
         held: set[str]) -> tuple[list[tuple[str, list[str]]], dict[str, int]]:
    """``(role, ids)`` batches of rows at the ``review`` step to start, fullest first; and counts
    of the rows left out (``held`` by a live lane, ``taken`` in the last day, ``no_seat``,
    ``unlooked`` past this pass's lookups, ``second``: a provisional fabrication waiting for a
    seat other than the one that made it)."""

    taken = _taken_up()
    want = [i for i in ids if i not in held and i not in taken]
    records = {i: read_json(ModelRecord, data.model(i)) for i in want}
    family = {i: r.family for i, r in records.items()}
    firsts = _awaiting_second(records, data)
    roles = _reviewers(list(dict.fromkeys(f for i, f in family.items() if i not in firsts)), data)
    groups: dict[str, list[str]] = {}
    left = {"held": sum(i in held for i in ids), "taken": sum(i in taken and i not in held for i in ids),
            "no_seat": 0, "unlooked": 0, "second": 0}
    authors: dict[str, str] | None = None
    for model_id in want:
        if model_id in firsts:
            if authors is None and _other_seat_counts(firsts[model_id], data):
                authors = _authors()
            role = _second_seat(firsts[model_id], (authors or {}).get(family[model_id]), data)
            if role is None:
                left["second"] += 1
            else:
                groups.setdefault(role, []).append(model_id)
        elif family[model_id] not in roles:
            left["unlooked"] += 1
        elif roles[family[model_id]] is None:
            left["no_seat"] += 1
        else:
            groups.setdefault(roles[family[model_id]], []).append(model_id)
    batches = [(role, g[k:k + BATCH]) for role, g in groups.items() for k in range(0, len(g), BATCH)]
    return sorted(batches, key=lambda b: len(b[1]), reverse=True), left


def _awaiting_second(records: dict[str, ModelRecord], data: DataLayout) -> dict[str, str]:
    """Rows whose only fabrication verdicts at the current recipe came from one provisional
    seat, mapped to that seat (``drive._confirmed`` wants a second, different seat)."""

    if not records:
        return {}
    seats: dict[str, set[str | None]] = {}
    for row in ledger.read_all(data=data):
        if not (isinstance(row, LedgerRow) and row.stage == "review" and row.outcome != "infra"
                and row.verdict == "fabrication" and row.id in records):
            continue
        recipe = records[row.id].recipe
        if recipe is not None and row.recipe_rev == recipe.rev:
            seats.setdefault(row.id, set()).add(row.reviewer if needs_second_review(row) else None)
    return {i: next(iter(s)) for i, s in seats.items() if len(s) == 1 and None not in s}


def _other_seat_counts(first: str, data: DataLayout) -> bool:
    prompt = prompt_version()
    return any(r != first and calibration.counts(r, prompt, data=data) for r in REVIEWER_ROLES)


def _second_seat(first: str, author: str | None, data: DataLayout) -> str | None:
    """A calibrated seat other than ``first`` that may review a row whose family ``author``
    built (``None``: a June classic, any other seat), or ``None`` while there is none."""

    if not _other_seat_counts(first, data):
        return None
    try:
        if author:
            return pick_reviewer(author, data=data, exclude=frozenset({first}))
        prompt = prompt_version()
        return next(r for r in (*MAIN_ROLES, *_third_lab())
                    if r != first and calibration.counts(r, prompt, data=data))
    except (ReviewError, StopIteration):
        return None


def _calibrate(standing: dict[str, str], data: DataLayout) -> dict[str, str]:
    """Move each uncalibrated seat's calibration on (``calibrate``); a seat whose lanes cannot
    start (a refusal, a broken guard) is reported and the others go on."""

    out = {}
    for role, state in standing.items():
        if state != "uncalibrated":
            continue
        try:
            out[role] = str(calibrate(role=role, data=data).get("state"))
        except (drive.Refused, RecordError, ReviewError, OSError) as exc:
            out[role] = f"not started: {str(exc)[:200]}"
    return out


def stage(ids: Sequence[str], *, data: DataLayout, room: int, held: set[str],
          dry_run: bool) -> dict[str, Any]:
    """One tick pass of the review stage over the rows at ``review``; see the section note.

    ``room`` is how many more review lanes may run (the cap less the live ones, calibration lanes
    included). Calibration runs first: until a seat counts nothing else starts.
    """

    standing = seats(data)
    out: dict[str, Any] = {"prompt": prompt_version(), "seats": standing, "waiting": len(ids)}
    if any(s == "uncalibrated" for s in standing.values()):
        out["calibration"] = "dry run" if dry_run else _calibrate(standing, data)
    if not dry_run:
        out["settle"] = settle(data=data, store=StoreLayout.from_env())
    if not any(s in COUNTED for s in standing.values()):
        out["held"] = f"no reviewer seat counts at prompt {out['prompt']}"
        return out
    batches, out["left"] = plan(ids, data=data, held=held) if room > 0 else ([], {})
    out["launched"], started = [], 0
    for role, batch in batches:
        if started >= room:
            break
        if dry_run:
            out["launched"].append([role, batch])
            started += 1
            continue
        try:
            out["launched"] += launch(batch, data=data, roles=dict.fromkeys(batch, role))
            started += 1
        except drive.Refused as refused:
            out["refused"] = str(refused)[:200]
            break
        except (ReviewError, RecordError, OSError) as error:  # one bad case skips its batch only
            out.setdefault("failed", []).append({"ids": list(batch), "error": str(error)[:200]})
    return out
