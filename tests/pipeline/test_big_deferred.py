"""``big_row_deferred``: rung 4's waiting big rows typed at the tail and taken back when G has room
(Lead 2026-11-04)."""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import DataLayout, big_deferred, drive, read_json, stage3, tick_full, write_json
from menagerie.pipeline.records import ModelRecord

FIXTURES = Path(__file__).parent / "fixtures"
COHORTS = {"m1": "r4-github", "m2": "r4-github", "m3": "r4-github", "m4": "r4-github",
           "m5": "r4-github", "m7": "r4-github", "m6": "r3"}
NEED = {"m1": 30.0, "m2": 25.0, "m3": 10.0, "m4": 35.0, "m5": 30.0, "m6": 30.0, "m7": 22.0}
CFG = {"queue_depth": 40}


@pytest.fixture()
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """Seven records; m5 waits for a render, m4 has a job open on G, m2 is in the axon batch in
    flight, m7's family a live lane holds, m3 is an ordinary row and m6 is another cohort's."""

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    layout = DataLayout(tmp_path / "data")
    base = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    for sid in COHORTS:
        write_json(dataclasses.replace(base, id=sid), layout.model(sid))
    monkeypatch.setattr(tick_full, "_cohorts", lambda cfg: dict(COHORTS))
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "render" if i == "m5" else "trace", "why": ""} for i in ids])
    monkeypatch.setattr(big_deferred, "_sizer", lambda data: lambda record: NEED[record.id])
    monkeypatch.setattr(tick_full, "_not_pending", lambda ids, data: [i for i in ids if i != "m4"])
    monkeypatch.setattr(big_deferred, "_axon_ids", lambda: {"m2"})
    monkeypatch.setattr(big_deferred, "_lane_held",
                        lambda ids, data: {i for i in ids if i == "m7"})
    return layout


def _give_up(data: DataLayout, sid: str) -> str | None:
    record = read_json(ModelRecord, data.model(sid))
    return record.give_up.cls if record.give_up else None


def test_only_waiting_big_rows_of_the_cohort_are_typed_and_only_with_apply(data: DataLayout) -> None:
    dry = big_deferred.type_waiting(data=data, cohort="r4-github", cfg=CFG)
    assert [h["id"] for h in dry["held"]] == ["m1"] and dry["typed"] == 0
    assert dry["skipped"] == {"traced_or_open_job": ["m4"], "axon_batch": ["m2"],
                              "live_lane": ["m7"]}
    assert all(_give_up(data, sid) is None for sid in COHORTS)
    done = big_deferred.type_waiting(data=data, cohort="r4-github", apply=True, cfg=CFG)
    assert done["typed"] == 1 and _give_up(data, "m1") == big_deferred.CLASS
    assert "30.0 GiB" in done["held"][0]["why"]
    assert {sid for sid in COHORTS if _give_up(data, sid)} == {"m1"}
    # typed rows no longer wait: a second run types nothing
    assert big_deferred.type_waiting(data=data, cohort="r4-github", cfg=CFG)["held"] == []


def test_a_cohort_is_required(data: DataLayout) -> None:
    with pytest.raises(ValueError, match="--cohort"):
        big_deferred.type_waiting(data=data, cohort="", cfg=CFG)
    with pytest.raises(ValueError, match="--cohort"):
        big_deferred.reopen(data=data, cohort="", cfg=CFG)


def test_reopen_takes_back_as_many_as_g_has_room_for_smallest_first(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(big_deferred, "_lane_held", lambda ids, data: set())
    big_deferred.type_waiting(data=data, cohort="r4-github", apply=True, cfg=CFG)
    assert {sid for sid in COHORTS if _give_up(data, sid)} == {"m1", "m7"}
    monkeypatch.setattr(big_deferred, "g_room", lambda data, cfg: 0)
    assert big_deferred.reopen(data=data, cohort="r4-github", apply=True, cfg=CFG)["reopened"] == []
    monkeypatch.setattr(big_deferred, "g_room", lambda data, cfg: 1)
    dry = big_deferred.reopen(data=data, cohort="r4-github", cfg=CFG)
    assert dry == {"room": 1, "deferred": 2, "reopened": ["m7"]} and _give_up(data, "m7")
    big_deferred.reopen(data=data, cohort="r4-github", apply=True, cfg=CFG)
    assert _give_up(data, "m7") is None and _give_up(data, "m1") == big_deferred.CLASS
    out = big_deferred.reopen(data=data, cohort="r4-github", apply=True, all_rows=True, cfg=CFG)
    assert out["reopened"] == ["m1"] and _give_up(data, "m1") is None
    assert big_deferred.reopen(data=data, cohort="r4-github", cfg=CFG)["deferred"] == 0


def test_g_room_is_free_spool_g_room_less_rows_still_waiting_for_g(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    waiting: dict[str, Any] = {"spool": 30, "reserved": ["a"] * 4}
    monkeypatch.setattr(tick_full, "worklist", lambda cfg: ["m1"])
    monkeypatch.setattr(tick_full, "_steps", lambda ids, data, cfg: {"m1": "trace"})
    monkeypatch.setattr(tick_full, "_big_candidates",
                        lambda data, want, cfg=None: list(waiting["reserved"]))
    monkeypatch.setattr(tick_full, "_not_pending", lambda ids, data: list(ids))
    monkeypatch.setattr(tick_full, "_waiting", lambda spool: waiting["spool"])
    assert big_deferred.g_room(data, CFG) == 6
    waiting["reserved"] = ["a"] * 20
    assert big_deferred.g_room(data, CFG) == 0


def test_a_live_or_queued_lane_holds_a_row_by_family_or_id(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import reload

    module = reload(big_deferred)  # the fixture replaced _lane_held; use the real one
    family = read_json(ModelRecord, data.model("m1")).family
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter(), {family}))
    monkeypatch.setattr(stage3, "lane_holds", lambda ids, data: {})
    assert module._lane_held(["m1", "m2"], data) == {"m1", "m2"}
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter(), set()))
    monkeypatch.setattr(stage3, "lane_holds", lambda ids, data: {"m2": "run-1"})
    assert module._lane_held(["m1", "m2"], data) == {"m2"}
