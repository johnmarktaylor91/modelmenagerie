"""Disputed provisional fabrications: one repair round, then ``review_disputed`` at the tail.

A provisional seat's fabrication that every other counting seat answered without confirming goes
to 2b once, as a repair family with that seat's findings; still disputed after it, the row waits
at ``review`` until ``type_held`` types it, and ``reopen`` takes the give-up back once a main seat
passes calibration (decisions.md 2026-11-03).
"""

from __future__ import annotations

import dataclasses
import os
import time
from collections import Counter
from pathlib import Path

import pytest

from menagerie.paths import work_dir
from menagerie.pipeline import (
    DataLayout, drive, implement, publish, read_json, read_jsonl, review, review_disputed,
    tick_full, write_json,
)
from menagerie.pipeline.records import LedgerRow, ModelRecord, Packet, dumps, parse_ledger_line

FIXTURES = Path(__file__).parent / "fixtures"
TL = "2.35.2"
SEATS = frozenset({"model-design-review", "model-judgment-3vote"})


@pytest.fixture()
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """resnet18 as a classic (so it is reviewed), traced and rendered at its recipe."""

    monkeypatch.setattr(publish, "torchlens_pin", lambda: TL)
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(drive, "counting_seats", lambda data: SEATS)
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter(), set()))
    layout = DataLayout(tmp_path / "data")
    layout.roster.parent.mkdir(parents=True)
    layout.roster.write_text((FIXTURES / "roster.jsonl").read_text(encoding="utf-8"))
    record = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    write_json(dataclasses.replace(record, source_case="classic"), layout.model("m8839"))
    write_json(read_json(Packet, FIXTURES / "f-resnet.packet.json"), layout.packet("f-resnet"))
    return layout


def _run_rows() -> list[LedgerRow]:
    trace, render = read_jsonl(parse_ledger_line, FIXTURES / "m8839.e2e.ledger.jsonl")
    forward = dataclasses.replace(trace, stage="forward", capture=None, trace=None, n_ops=None,
                                  graph_hash=None, validation_state=None)
    return [forward, trace, render]


def _review(seat: str, verdict: str, at: str, rev: str | None = None) -> LedgerRow:
    msg = "prompt p; case 2"
    if verdict == "fabrication" and seat == "model-design-review":
        msg = f"{msg}; {review.PROVISIONAL}"
    return LedgerRow(id="m8839", stage="review", at=at, host="hub", env=f"review:{seat}",
                     recipe_rev=rev or "457cba223fe9", outcome="pass", verdict=verdict,
                     findings="a" * 64, reviewer=seat, msg=msg)


def _ledger(data: DataLayout, *reviews: LedgerRow) -> None:
    path = data.ledger("2026-10")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dumps(r) + "\n" for r in [*_run_rows(), *reviews]), encoding="utf-8")


def _next(data: DataLayout) -> dict[str, str]:
    (row,) = drive.next_steps(["m8839"], data=data)
    return row


def _packet(data: DataLayout) -> Packet:
    return read_json(Packet, data.packet("f-resnet"))


ASTRA = _review("model-design-review", "fabrication", "2026-10-30T01:00:00Z")
NO = _review("model-judgment-3vote", "minor", "2026-10-30T02:00:00Z")


def test_a_disputed_provisional_fabrication_goes_to_one_repair_round(data: DataLayout) -> None:
    _ledger(data, ASTRA)
    assert _next(data)["step"] == "review"  # the second seat has not answered yet
    _ledger(data, ASTRA, NO)
    assert _next(data) == {"id": "m8839", "step": "2b", "why": drive.DISPUTE_REPAIR}
    found = implement.fabrications(_packet(data), data)  # its findings, its family
    assert [(r.reviewer, r.at) for r in found] == [(ASTRA.reviewer, ASTRA.at)]
    assert tick_full._repair_families(["f-resnet"], data) == ["f-resnet"]


def test_a_seat_that_could_still_be_asked_keeps_the_row_waiting(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    _ledger(data, ASTRA, NO)
    monkeypatch.setattr(drive, "counting_seats", lambda data: SEATS | {"model-default"})
    assert _next(data)["step"] == "review"
    assert implement.fabrications(_packet(data), data) == []
    monkeypatch.setattr(drive, "counting_seats", lambda data: frozenset({"model-design-review"}))
    assert _next(data)["step"] == "review"  # no other seat counts: nobody has disputed it


def test_a_confirmation_still_takes_the_confirmed_path(data: DataLayout) -> None:
    yes = dataclasses.replace(NO, verdict="fabrication")
    _ledger(data, ASTRA, yes)
    assert _next(data) == {"id": "m8839", "step": "2b",
                           "why": "review found fabrication: the repair round"}


def _age(path: Path, day: int) -> None:
    stamp = time.mktime((2026, 10, day, 12, 0, 0, 0, 0, 0))
    os.utime(path, (stamp, stamp))


def test_the_repair_round_is_spent_once_a_lane_briefed_with_it_ends(data: DataLayout) -> None:
    _ledger(data, ASTRA, NO)
    lane = work_dir("2b") / "f-resnet"
    lane.mkdir(parents=True)
    (lane / "BRIEF.md").write_text("a lane for other members, after the dispute")
    (lane / "BRIEF.result.md").write_text("it ended")
    assert _next(data)["step"] == "2b"  # that lane never carried this row's findings
    drive.mark_briefed("f-resnet", [ASTRA])
    drive.mark_briefed("f-resnet", [ASTRA])  # a relaunch keeps the first time
    _age(lane / "BRIEF.result.md", 31)
    assert _next(data)["step"] == "2b"  # briefed, its lane still running
    (lane / "BRIEF.result.md").write_text("changed nothing")
    ended = drive.briefed("f-resnet", "m8839", ASTRA.recipe_rev) + 60  # minutes later in life
    os.utime(lane / "BRIEF.result.md", (ended, ended))
    assert _next(data) == {"id": "m8839", "step": "review", "why": drive.DISPUTE_HELD}
    assert implement.fabrications(_packet(data), data) == []


def test_a_family_owed_an_unbriefed_disputed_round_is_not_spent(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    _ledger(data, ASTRA, NO)
    lane = work_dir("2b") / "f-resnet"
    lane.mkdir(parents=True)
    for path in (data.model("m8839"), data.packet("f-resnet")):
        _age(path, 29)
    for name, day in (("BRIEF.md", 30), ("BRIEF.result.md", 31)):
        (lane / name).write_text(name)
        _age(lane / name, day)
    steps, family = {"m8839": "2b"}, {"m8839": "f-resnet"}
    assert tick_full._spent_2b(steps, family, data) == []
    monkeypatch.setattr(drive, "counting_seats", lambda data: SEATS | {"model-default"})
    assert tick_full._spent_2b(steps, family, data) == ["f-resnet"]  # nothing new for a lane


def test_a_row_disputed_again_after_its_repair_is_held_then_typed_and_reopened(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    before = [_review("model-design-review", "fabrication", "2026-10-28T01:00:00Z", rev="0123456789ab"),
              _review("model-judgment-3vote", "faithful", "2026-10-28T02:00:00Z", rev="0123456789ab")]
    _ledger(data, *before, ASTRA, NO)
    assert _next(data) == {"id": "m8839", "step": "review", "why": drive.DISPUTE_HELD}

    listed = review_disputed.type_held(data=data)
    assert [h["id"] for h in listed["held"]] == ["m8839"] and listed["typed"] == 0
    assert read_json(ModelRecord, data.model("m8839")).give_up is None  # a dry run writes nothing
    assert review_disputed.type_held(data=data, apply=True)["typed"] == 1
    give = read_json(ModelRecord, data.model("m8839")).give_up
    assert give.cls == "review_disputed" and give.by == "script"
    assert _next(data) == {"id": "m8839", "step": "done", "why": "gave up: review_disputed"}

    monkeypatch.setattr(review, "seats", lambda data: {"model-design-review": "provisional",
                                                       "model-default": "failed"})
    assert review_disputed.reopen(data=data, apply=True) == {"passed": [], "reopened": []}
    monkeypatch.setattr(review, "seats", lambda data: {"model-design-review": "passed"})
    assert review_disputed.reopen(data=data, apply=True)["reopened"] == []  # it already answered
    monkeypatch.setattr(review, "seats", lambda data: {"model-default": "passed"})
    monkeypatch.setattr(drive, "counting_seats", lambda data: SEATS | {"model-default"})
    assert review_disputed.reopen(data=data)["reopened"] == ["m8839"]
    assert read_json(ModelRecord, data.model("m8839")).give_up is not None  # dry run
    assert review_disputed.reopen(data=data, apply=True)["reopened"] == ["m8839"]
    assert read_json(ModelRecord, data.model("m8839")).give_up is None
    assert _next(data)["step"] == "review"  # back for the newly passed seat


def test_the_tick_reopens_only_while_reviews_are_enabled(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review, "seats", lambda data: {"model-default": "failed"})
    assert "skipped" in tick_full.disputed(data, {}, dry_run=False)
    assert tick_full.disputed(data, {"reviews": {"enabled": True}}, dry_run=True)["reopened"] == []


def test_before_repair_types_a_disputed_row_still_owed_its_round(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    _ledger(data, ASTRA, NO)
    assert _next(data)["step"] == "2b"
    assert review_disputed.type_held(data=data)["held"] == []  # not held: its round is owed
    tick = work_dir("tick")
    tick.mkdir(parents=True)
    (tick / "worklist.txt").write_text("m8839 r4-github\n")
    assert review_disputed.type_held(data=data, before_repair=True, cohort="r3")["held"] == []
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter(), {"f-resnet"}))
    busy = review_disputed.type_held(data=data, before_repair=True, apply=True)
    assert busy["held"] == [] and busy["skipped_live_lane"] == ["m8839"]  # its lane runs
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter(), set()))
    listed = review_disputed.type_held(data=data, before_repair=True, cohort="r4-github",
                                       apply=True)
    (held,) = listed["held"]
    assert listed["typed"] == 1 and "before any repair round" in held["why"]
    assert "not confirmed by model-judgment-3vote" in held["why"]
    give = read_json(ModelRecord, data.model("m8839")).give_up
    assert give.cls == "review_disputed" and give.reason == held["why"]
    assert _next(data) == {"id": "m8839", "step": "done", "why": "gave up: review_disputed"}
    assert implement.fabrications(_packet(data), data) == []  # no repair round for a typed row
    assert tick_full._repair_families(["f-resnet"], data) == []
    monkeypatch.setattr(review, "seats", lambda data: {"model-default": "passed"})
    assert review_disputed.reopen(data=data, apply=True)["reopened"] == ["m8839"]
    assert read_json(ModelRecord, data.model("m8839")).give_up is None


def test_a_row_no_counting_seat_may_review_is_typed_and_reopened_when_one_may(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    _ledger(data)  # traced and rendered, never reviewed
    tick = work_dir("tick")
    tick.mkdir(parents=True)
    (tick / "worklist.txt").write_text("m8839 r4-github\n")
    looked: list[list[str]] = []

    def reviewers(families: list[str], data: DataLayout) -> dict[str, str | None]:
        looked.append(families)
        seat = "model-default" if "model-default" in drive.counting_seats(data) else None
        return dict.fromkeys(families, seat)

    monkeypatch.setattr(review, "_reviewers", reviewers)
    monkeypatch.setattr(review, "_authors", lambda: {})
    monkeypatch.setattr(review, "seats", lambda data: {"model-default": "failed"})
    with pytest.raises(ValueError, match="--cohort"):
        review_disputed.type_no_seat(data=data, cohort="")
    assert review_disputed.type_no_seat(data=data, cohort="r3")["held"] == []
    out = review_disputed.type_no_seat(data=data, cohort="r4-github", apply=True)
    (held,) = out["held"]
    assert out["typed"] == 1 and "no counting seat may review" in held["why"]
    give = read_json(ModelRecord, data.model("m8839")).give_up
    assert give.cls == "review_no_seat" and give.by == "script"
    assert _next(data) == {"id": "m8839", "step": "done", "why": "gave up: review_no_seat"}
    looked.clear()
    assert review_disputed.reopen(data=data, apply=True)["reopened"] == []
    assert looked == []  # the same seats count: nothing is looked up
    monkeypatch.setattr(drive, "counting_seats", lambda data: SEATS | {"model-default"})
    assert review_disputed.reopen(data=data, apply=True)["reopened"] == ["m8839"]
    assert read_json(ModelRecord, data.model("m8839")).give_up is None
    assert review_disputed._no_seat_list(data) == {}
    assert _next(data)["step"] == "review"
