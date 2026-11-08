"""The rung's tail for disputed provisional fabrications (decisions.md 2026-11-03).

A provisional seat's fabrication that every other counting seat answered without confirming goes
to one repair round (``drive.dispute``). A row still disputed after it waits at ``review``
(``drive.DISPUTE_HELD``). ``type_held`` gives each such row the deferred give-up
``review_disputed``, which settles it and counts toward the rung's close; it writes only with
``apply`` (``menagerie-pipeline review disputed --apply``), and the Crawler runs it once the
rest of the rung has settled. ``--before-repair`` (Lead, 2026-11-03) also types the disputed rows
still owed their repair round, with a reason that says so; ``--cohort`` keeps one cohort.
``--no-seat --cohort C`` types ``review_no_seat`` on rows no counting seat may review. ``reopen`` takes the give-up back once a main seat passes
calibration at the prompt in force and has not reviewed the row at its recipe: the row returns
to ``review`` and that seat reviews it. The tick runs ``reopen`` every full pass. Runs on the
hub: imports neither torch nor psutil.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, drive, ledger, now, read_json, write_json
from menagerie.pipeline.records import GiveUp, LedgerRow, ModelRecord

CLASS = "review_disputed"
NO_SEAT = "review_no_seat"
# The rows typed ``review_no_seat``, with the author lab and the seats that counted then
# (``{id: {"family", "lab", "seats"}}``): ``reopen`` reads this, not every record.
NO_SEAT_FILE = ("review", "no-seat.json")


def _ids(data: DataLayout) -> list[str]:
    """Ids with a provisional seat's fabrication on the ledger: the only rows a dispute holds."""

    from menagerie.pipeline.review import needs_second_review

    rows = ledger.read_containing('"fabrication"', data=data)
    return sorted({r.id for r in rows if isinstance(r, LedgerRow) and needs_second_review(r)})


def _seen(model_id: str, record: ModelRecord, files: Any) -> set[str]:
    """The seats with a counted review of the row at its recipe (not ``infra``, not ``stale``)."""

    from menagerie.pipeline.review import stale

    packet = files.packet(record.family)
    return {str(r.reviewer) for r in files.rows.get(model_id, [])
            if isinstance(r, LedgerRow) and r.stage == "review" and r.outcome != "infra"
            and r.recipe_rev == record.recipe.rev and not stale(r, packet)}


def _reason(fab: LedgerRow, rev: str, others: Sequence[str], before_repair: bool) -> str:
    said = (f"provisional fabrication by {fab.reviewer} ({fab.at}) at recipe {rev}, not "
            f"confirmed by {', '.join(others)}")
    when = ("typed before any repair round (the rung 4 tail, Lead 2026-11-03)" if before_repair
            else "after its one repair round")
    return f"{said}; {when}; reopened when a main seat passes calibration"


def _disputed_steps(data: DataLayout, ids: list[str] | None, before_repair: bool,
                    cohort: str | None) -> dict[str, str]:
    """The chosen rows (default: every row with a provisional fabrication; ``cohort`` keeps the
    tick worklist's rows of it) whose next step is a held dispute, or with ``before_repair``
    also a dispute still owed its round, each with that step's why."""

    from menagerie.pipeline import tick_full

    wanted = {drive.DISPUTE_HELD, *([drive.DISPUTE_REPAIR] if before_repair else [])}
    chosen = ids or _ids(data)
    if cohort is not None:
        tags = tick_full._cohorts({})  # same package
        chosen = [i for i in chosen if tags.get(i) == cohort]
    if not chosen:  # ``next_steps`` reads an empty list as every rostered row
        return {}
    return {s["id"]: s["why"] for s in drive.next_steps(chosen, data=data) if s["why"] in wanted}


def type_held(*, data: DataLayout, ids: list[str] | None = None, apply: bool = False,
              before_repair: bool = False, cohort: str | None = None) -> dict[str, Any]:
    """Type every row held at ``review`` after its disputed repair round ``review_disputed``;
    with ``before_repair`` also every disputed row still owed its round (``drive.DISPUTE_REPAIR``).
    ``cohort`` keeps the tick worklist's rows of that cohort.

    A row whose family a live or queued lane holds is skipped (``skipped_live_lane``), so the
    type never races that lane's recipe. Returns ``{"held": [{"id", "why"}], "typed": n,
    "skipped_live_lane": [...]}``; writes the records only with ``apply``.
    """

    from menagerie.pipeline import tick_full

    steps = _disputed_steps(data, ids, before_repair, cohort)
    files = drive._load(data, ids=list(steps))  # same package
    live = tick_full._busy()[1]  # same package; a live or queued lane's family is left alone
    out: list[dict[str, str]] = []
    skipped = sorted(i for i in steps if files.record(i).family in live)
    for model_id, step_why in steps.items():
        record = files.record(model_id)
        if model_id in skipped:
            continue
        found = drive.dispute(model_id, record, files.rows.get(model_id, []),
                              files.packet(record.family), data)
        if found is None:
            continue
        others = sorted(_seen(model_id, record, files) - {str(found[0].reviewer)})
        why = _reason(found[0], record.recipe.rev, others, step_why == drive.DISPUTE_REPAIR)
        out.append({"id": model_id, "why": why})
        if apply:
            path = data.model(model_id)
            write_json(replace(read_json(ModelRecord, path), give_up=GiveUp(
                cls=CLASS, reason=why, at=now(), by="script")), path)
    return {"held": out, "typed": len(out) if apply else 0, "skipped_live_lane": skipped}


def _main_seats(data: DataLayout) -> list[str]:
    """The seats that may review a first-time row (not confirm-only) counting now."""

    from menagerie.pipeline.review import CONFIRM_ROLES

    return sorted(drive.counting_seats(data) - set(CONFIRM_ROLES))


def _no_seat_path(data: DataLayout) -> Path:
    return Path(data.root).joinpath(*NO_SEAT_FILE)


def _no_seat_list(data: DataLayout) -> dict[str, Any]:
    try:
        return json.loads(_no_seat_path(data).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _author_lab(family: str, authors: dict[str, str]) -> str:
    from menagerie.pipeline.review import ReviewError, author_of

    try:
        return str(author_of(authors[family])["harness"]) if family in authors else "unknown"
    except (ReviewError, KeyError, OSError, ValueError):
        return "unknown"


def _unseated(data: DataLayout, cohort: str) -> dict[str, str]:
    """The cohort's rows waiting at ``review`` with no review at their recipe whose family the
    reviewer lookup finds no seat for (a family not looked up this pass is left out), each with
    its family."""

    from menagerie.pipeline import tick_full
    from menagerie.pipeline.review import _reviewers

    tags = tick_full._cohorts({})  # same package
    chosen = [i for i, c in tags.items() if c == cohort]
    if not chosen:  # ``next_steps`` reads an empty list as every rostered row
        return {}
    waiting = [s["id"] for s in drive.next_steps(chosen, data=data)
               if s["step"] == "review" and " recipe with no review at " in s["why"]]
    files = drive._load(data, ids=waiting)  # same package
    family = {i: files.record(i).family for i in waiting}
    roles = _reviewers(sorted(set(family.values())), data)
    return {i: f for i, f in family.items() if f in roles and roles[f] is None}


def type_no_seat(*, data: DataLayout, cohort: str, apply: bool = False) -> dict[str, Any]:
    """Type ``review_no_seat`` on each row of ``cohort`` waiting at ``review`` with no review at
    its recipe whose family no counting seat may review (``review._reviewers`` finds none: every
    counting seat is from the author's lab). Returns ``{"held": [{"id", "why"}], "typed": n}``;
    writes the records and ``NO_SEAT_FILE`` only with ``apply``."""

    from menagerie.pipeline.review import _authors, prompt_version

    if not cohort:
        raise ValueError("review_no_seat is typed one cohort at a time: name it with --cohort")
    family = _unseated(data, cohort)
    authors, seats, prompt = _authors(), _main_seats(data), prompt_version()
    listed, out = _no_seat_list(data), []
    for model_id in family:
        lab = _author_lab(family[model_id], authors)
        why = (f"no counting seat may review a recipe authored on {lab}: the seats counting at "
               f"prompt {prompt} ({', '.join(seats) or 'none'}) are from its author's lab or "
               "confirm only; typed at the rung 4 tail (Lead 2026-11-03); reopened when a seat "
               "for it passes calibration")
        out.append({"id": model_id, "why": why})
        listed[model_id] = {"family": family[model_id], "lab": lab, "seats": seats}
        if apply:
            path = data.model(model_id)
            write_json(replace(read_json(ModelRecord, path), give_up=GiveUp(
                cls=NO_SEAT, reason=why, at=now(), by="script")), path)
    if apply and out:
        write_json_plain(_no_seat_path(data), listed)
    return {"held": out, "typed": len(out) if apply else 0}


def write_json_plain(path: Path, value: dict[str, Any]) -> None:
    """Write ``NO_SEAT_FILE`` atomically and report it, so a tick pass commits it (``written``)."""

    from menagerie.pipeline import written

    body = json.dumps(value, indent=1, sort_keys=True) + "\n"
    prior = written.peek(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
    written.rewrote(path, prior, body.encode("utf-8"))


def _take_back(data: DataLayout, ids: Sequence[str]) -> None:
    """Clear the give-up of each of ``ids``: the row returns to ``review``."""

    for model_id in ids:
        path = data.model(model_id)
        write_json(replace(read_json(ModelRecord, path), give_up=None), path)


def _reopen_no_seat(data: DataLayout, apply: bool) -> list[str]:
    """``review_no_seat`` rows whose family a seat may now review (looked up only once the
    counting seats differ from those it was typed under). A listed row whose give-up is gone
    some other way leaves the list."""

    from menagerie.pipeline.review import _reviewers

    listed, seats = _no_seat_list(data), _main_seats(data)
    changed = {i: v for i, v in listed.items() if v.get("seats") != seats}
    if not changed:
        return []
    roles = _reviewers(sorted({v["family"] for v in changed.values()}), data)
    reopened, gone = [], []
    for model_id, entry in changed.items():
        give = read_json(ModelRecord, data.model(model_id)).give_up
        if give is None or give.cls != NO_SEAT:
            gone.append(model_id)
        elif roles.get(entry["family"]) is not None:
            reopened.append(model_id)
    if apply and (reopened or gone):
        _take_back(data, reopened)
        drop = {*reopened, *gone}
        write_json_plain(_no_seat_path(data), {i: v for i, v in listed.items() if i not in drop})
    return reopened


def _passed_main(data: DataLayout) -> list[str]:
    """The main seats (the third lab's included, unless held) fully passed at the prompt in force."""

    from menagerie.pipeline.review import MAIN_ROLES, _third_lab, seats

    standing = seats(data)
    return [r for r in (*MAIN_ROLES, *_third_lab()) if standing.get(r) == "passed"]


def reopen(*, data: DataLayout, apply: bool = False) -> dict[str, Any]:
    """Take ``review_disputed`` back from each row a fully passed main seat has not reviewed at
    its recipe, so that seat reviews it. Returns ``{"passed": [...], "reopened": [...]}``; writes
    only with ``apply``. Reads nothing past the seats while no main seat has passed."""

    no_seat = _reopen_no_seat(data, apply)
    passed = _passed_main(data)
    if not passed:
        return {"passed": [], "reopened": no_seat}
    ids = _ids(data)
    files = drive._load(data, ids=ids)  # same package
    reopened = []
    for model_id in ids:
        record = files.record(model_id)
        if record is None or record.recipe is None or record.give_up is None \
                or record.give_up.cls != CLASS:
            continue
        if not set(passed) - _seen(model_id, record, files):
            continue
        reopened.append(model_id)
    if apply:
        _take_back(data, reopened)
    return {"passed": passed, "reopened": no_seat + reopened}
