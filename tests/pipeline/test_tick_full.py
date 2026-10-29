"""The tick's full-mode steps: the ``<id> [cohort]`` worklist format and the overlap lane's stubs."""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest

from menagerie.pipeline import (
    DataLayout,
    drive,
    publish,
    read_json,
    read_jsonl,
    runner,
    tick,
    tick_full,
    worklist_build,
    write_json,
)
from menagerie.pipeline.records import GiveUp, ModelRecord, Packet, dumps, parse_ledger_line

FIXTURES = Path(__file__).parent / "fixtures"
# The ids of the sampled worklist fixture, in file order (held '#m...' lines are comments).
TODAY = ["m4527", "m1707", "m1708", "m3142", "m10244", "m3595"]


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.delenv("MENAGERIE_BRIDGE_SPOOL", raising=False)
    spool = tmp_path / "work" / "hubspool"
    spool.mkdir(parents=True)
    (spool / "serving").touch()
    folder = tmp_path / "work" / "tick"
    folder.mkdir(parents=True)
    return folder


def test_todays_worklist_parses_unchanged(work: Path) -> None:
    (work / "worklist.txt").write_text((FIXTURES / "worklist.txt").read_text(encoding="utf-8"))
    assert tick_full.worklist({}) == TODAY


def test_a_cohort_tag_is_read_as_its_id(work: Path) -> None:
    tagged = []
    for line in (FIXTURES / "worklist.txt").read_text(encoding="utf-8").splitlines():
        tagged.append(line if line.startswith("#") else f"{line} r3")
    tagged += ["m20601 r4-github", "m20612\tr4-github  # a census row", "m4527 r4-github"]
    (work / "worklist.txt").write_text("\n".join(tagged) + "\n")
    assert tick_full.worklist({}) == [*TODAY, "m20601", "m20612"]


def test_a_named_worklist_file_reads_the_same_way(work: Path, tmp_path: Path) -> None:
    path = tmp_path / "elsewhere.txt"
    path.write_text("m2 r3\nm1\n# m9 r3\n")
    assert tick_full.worklist({"worklist": str(path)}) == ["m2", "m1"]


def _cohort(data: DataLayout, work: Path, monkeypatch: pytest.MonkeyPatch,
            *, total: int, passes: int, gaps: int = 0, pending: int = 0) -> None:
    record = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    packet = read_json(Packet, FIXTURES / "f-resnet.packet.json")
    trace = read_jsonl(parse_ledger_line, FIXTURES / "m8839.e2e.ledger.jsonl")[0]
    monkeypatch.setattr(publish, "torchlens_pin", lambda: trace.tl)
    lines, ledger = [], []
    for n in range(total + pending):
        model_id = f"m{n}"
        lines.append(f"{model_id} r4-github")
        own = dataclasses.replace(record, id=model_id)
        if n >= passes:
            gap = n < passes + gaps
            own = dataclasses.replace(own, recipe=None, give_up=GiveUp(
                cls="research_gap:no_source" if gap else "needs_gpu", reason="evidence", at=trace.at))
        if n >= total:
            own = dataclasses.replace(own, recipe=None, give_up=None)
        write_json(own, data.model(model_id))
        if n < passes:
            ledger.append(dataclasses.replace(trace, id=model_id, recipe_rev=own.recipe.rev,
                                              validation_state="passed", graph_hash="a" * 64))
    packet = dataclasses.replace(packet, members=tuple(
        dataclasses.replace(packet.members[0], id=f"m{n}") for n in range(total + pending)))
    write_json(packet, data.packet(packet.family))
    data.ledger("2026-10").parent.mkdir(parents=True, exist_ok=True)
    data.ledger("2026-10").write_text("".join(dumps(row) + "\n" for row in ledger))
    (work / "worklist.txt").write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("total,passes,stops", [(100, 36, True), (100, 37, False), (49, 0, False)])
def test_yield_boundary_on_real_records(tmp_path: Path, work: Path, monkeypatch: pytest.MonkeyPatch,
                                        total: int, passes: int, stops: bool) -> None:
    data = DataLayout(tmp_path / "data")
    _cohort(data, work, monkeypatch, total=total, passes=passes)
    before = {p: p.read_bytes() for p in data.root.rglob("*") if p.is_file()}
    out = tick_full.tripwires(data, {}, dry_run=False)
    assert bool(out["stopped"]) == stops
    assert out["cohorts"]["r4-github"]["settled"] == total
    assert {p: p.read_bytes() for p in before} == before  # a cohort stop never types a row
    assert not tick_full.tripwires(data, {}, dry_run=False)["facts"]
    assert len(tick.load_state().get("outbox", [])) == int(stops)
    assert len(tick.load_state().get("cohort_events", [])) == int(stops)


def test_rows_the_roster_retired_are_not_counted_against_yield(
        tmp_path: Path, work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline.records import RosterRow

    data = DataLayout(tmp_path / "data")
    _cohort(data, work, monkeypatch, total=100, passes=45)
    retired = [f"m{n}" for n in range(100, 120)]  # 2a found these are not architectures
    data.roster.write_text("".join(dumps(RosterRow(name=f"row {i}", zoo="discovered-pytorch",
                                                   stable_id=i, state="not_architecture")) + "\n"
                                   for i in retired), encoding="utf-8")
    with (work / "worklist.txt").open("a") as handle:
        handle.write("".join(f"{i} r4-github\n" for i in retired))
    out = tick_full.tripwires(data, {}, dry_run=True)
    counts = out["cohorts"]["r4-github"]
    assert (counts["settled"], counts["retired"], counts["validated"]) == (120, 20, 45)
    assert not out["stopped"]  # 45 of 100 models, not 45 of 120 rows


def test_yield_waits_for_a_hundred_attempted_models() -> None:
    early = {"settled": 51, "retired": 18, "validated": 15, "completed_2a": 60, "research_gap": 0}
    assert tick_full._trip_reason("r4-github", early) is None  # rung 4's stop at 11:50 AM PT
    assert tick_full._trip_reason("r4-github", {**early, "settled": 118, "validated": 44}) is None
    full = {**early, "settled": 118, "validated": 36}  # GitHub's line: rung 3's 52 percent less 15
    assert tick_full._trip_reason("r4-github", full) == (
        "36 of 100 settled models validated; yield is below 37 percent")
    gaps = {**full, "research_gap": 1, "completed_2a": 200}  # one gap: 99 attempted
    assert tick_full._trip_reason("r4-github", gaps) is None


@pytest.mark.parametrize("gaps,stops", [(10, False), (11, True)])
def test_gap_denominator_is_completed_research(tmp_path: Path, work: Path,
                                              monkeypatch: pytest.MonkeyPatch,
                                              gaps: int, stops: bool) -> None:
    data = DataLayout(tmp_path / "data")
    _cohort(data, work, monkeypatch, total=50, passes=30, gaps=gaps, pending=50)
    out = tick_full.tripwires(data, {}, dry_run=True)
    assert bool(out["stopped"]) == stops
    assert out["cohorts"]["r4-github"]["completed_2a"] == 100
    assert not (work / "state.json").exists()


def test_resume_uses_the_tick_lock_and_does_not_retrip_unchanged_rows(
        tmp_path: Path, work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = DataLayout(tmp_path / "data")
    _cohort(data, work, monkeypatch, total=50, passes=0)
    tick_full.tripwires(data, {}, dry_run=False)
    with tick.locked() as acquired:
        assert acquired
        assert "held" in tick_full.tripwires(data, {}, dry_run=False, resume="r4-github")
    out = tick_full.tripwires(data, {}, dry_run=False, resume="r4-github")
    assert out["stopped"] == []
    assert tick_full.tripwires(data, {}, dry_run=False)["stopped"] == []
    with pytest.raises(ValueError, match="unknown cohort"):
        tick_full.tripwires(data, {}, dry_run=True, resume="r5-missing")


def _scheduler(work: Path, monkeypatch: pytest.MonkeyPatch, *, early: int = 2,
               busy: int = 0, held: set[str] | None = None, batch_size: int = 1,
               on_hub: bool = True) -> dict:
    lines = [*[f"m{n} r3" for n in range(early)],
             *[f"m{n} r4-github" for n in range(10, 20)]]
    (work / "worklist.txt").write_text("\n".join(lines))
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "2b"} for i in ids])
    monkeypatch.setattr(drive, "families_of", lambda ids, data: {i: "f-" + i for i in ids})
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter({"2b": busy}), held or set()))
    monkeypatch.setattr(tick_full, "_lane_busy", lambda cfg, family, kind: Counter(
        "r3" if int(key.removeprefix("f-m")) < 10 else "r4-github" for key in held or set()))
    monkeypatch.setattr(tick_full, "_caps", lambda cfg: {"2a": 4, "2b": 5})
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 10)
    return {**tick.DEFAULTS, "cohort_share": {"r4-github": 0.4},
            "families_per_2b_lane": batch_size, "2b_on_hub": on_hub}


def test_share_caps_later_slots_and_lifts_without_earlier_waits(work: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _scheduler(work, monkeypatch)
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    assert out["2b"] == ["f-m0", "f-m1", "f-m10", "f-m11"]
    cfg = _scheduler(work, monkeypatch, early=0)
    assert len(tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)["2b"]) == 5


def test_a_refill_top_up_starts_2b_only_and_returns_at_once_when_no_slot_is_free(
        work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import implement

    cfg = _scheduler(work, monkeypatch)
    monkeypatch.setattr(tick_full, "_caps", lambda cfg: {"2a": 0, "2b": 5})  # a refill's lane_limits
    monkeypatch.setattr(implement, "source_case", lambda family, *, data: "saved-pytorch")
    (work / "worklist.txt").write_text("m0 r3\nm1 r3\n")
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "2b" if i == "m0" else "2a"} for i in ids])
    _bridge_heartbeat(work, monkeypatch, "fresh", False)
    launches: list[tuple[str, object]] = []
    monkeypatch.setattr(tick_full, "_launch_2b", lambda families, data:
                        _capture_launches(launches, "2b", "recipe", families))
    monkeypatch.setattr(tick_full, "_launch_2a", lambda chunks, data:
                        _capture_launches(launches, "2a", "research", chunks))
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=False)
    assert out["2b"] == ["f-m0"] and out["2a"] == [] and launches == [("2b", "f-m0")]

    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter({"2b": 5}), set()))
    monkeypatch.setattr(drive, "families_of", lambda ids, data: pytest.fail("planned with no slot free"))
    assert tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=False) == {
        "skipped": "every lane slot is taken", "busy": {"2b": 5}, "caps": {"2a": 0, "2b": 5}}


def test_rows_traced_at_their_revision_never_take_the_queue_s_room(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from menagerie.pipeline import stage3

    data = DataLayout(tmp_path / "data")
    for model_id in ("m1", "m2", "m3", "m4"):
        data.model(model_id).parent.mkdir(parents=True, exist_ok=True)
        data.model(model_id).write_text("{}")
    monkeypatch.setattr(tick_full, "read_json", lambda cls, path: SimpleNamespace(
        recipe=None if path.stem == "m4" else SimpleNamespace(rev="r1")))
    monkeypatch.setattr(stage3, "_traced", lambda data: {("m1", "r1"), ("m2", "r0")})
    monkeypatch.setattr(stage3, "_pending", lambda model_id, rev: "sent" if model_id == "m3" else None)
    assert tick_full._not_pending(["m1", "m2", "m3", "m4"], data) == ["m2"]


def _bridge_heartbeat(
        work: Path, monkeypatch: pytest.MonkeyPatch, heartbeat: str,
        override: bool) -> None:
    now = 1_800_000_000.0
    monkeypatch.setattr(runner.time, "time", lambda: now)
    default = work.parent / "hubspool"
    # An override must win even when the default heartbeat says the opposite.
    default_age = runner.HEARTBEAT_STALE_S + 1 if heartbeat == "fresh" and override else 0
    os.utime(default / "serving", (now - default_age, now - default_age))
    spool = work.parent / "custom-spool" if override else default
    spool.mkdir(exist_ok=True)
    if override:
        monkeypatch.setenv("MENAGERIE_BRIDGE_SPOOL", str(spool))
    beat = spool / "serving"
    if heartbeat == "missing":
        beat.unlink(missing_ok=True)
    else:
        age = {"fresh": runner.HEARTBEAT_STALE_S - 1,
               "stale": runner.HEARTBEAT_STALE_S + 1,
               "boundary": runner.HEARTBEAT_STALE_S}[heartbeat]
        beat.touch()
        os.utime(beat, (now - age, now - age))


def _capture_launches(launches: list[tuple[str, object]], kind: str, run: str,
                      items: Sequence[object]) -> list[str]:
    launches.extend((kind, item) for item in items)
    return [run] * len(items)


@pytest.mark.parametrize("heartbeat", ["fresh", "stale", "missing", "boundary"])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_top_up_holds_only_2b_when_bridge_not_serving(
        work: Path, monkeypatch: pytest.MonkeyPatch, heartbeat: str,
        override: bool, dry_run: bool, batch_size: int) -> None:
    from menagerie.pipeline import implement

    cfg = _scheduler(work, monkeypatch, batch_size=batch_size)
    monkeypatch.setattr(implement, "source_case", lambda family, *, data: "saved-pytorch")
    (work / "worklist.txt").write_text("m0 r3\nm1 r3\n")
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "2b" if i == "m0" else "2a"} for i in ids])
    _bridge_heartbeat(work, monkeypatch, heartbeat, override)
    launches = []
    monkeypatch.setattr(tick_full, "_launch_2b", lambda families, data:
                        _capture_launches(launches, "2b", "recipe", families))
    monkeypatch.setattr(tick_full, "_launch_2a", lambda chunks, data:
                        _capture_launches(launches, "2a", "research", chunks))

    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=dry_run)

    fresh = heartbeat == "fresh"
    expected = {"2b_where": "workers and hub (2b_on_hub true); this pass: "
                            + ("1 on hub" if fresh else "none"),
                "busy": {"2b": 0}, "caps": {"2a": 4, "2b": 5}, "hub_slots": 10,
                "2b": ["f-m0"] if fresh else [], "2a": [["m1"]],
                "waiting_2b": 1, "waiting_2a": 1, "2b_spent": [],
                "cohorts": {"r3": {"waiting_2a": 1, "waiting_2b": 1,
                                   "stopped": False, "2b": int(fresh), "2a": 1}}}
    batches = [["f-m0"]] if fresh else []
    if batch_size > 1:
        expected["2b_batches"] = batches
    if not fresh:
        expected["2b_held"] = "bridge not serving"
    if not dry_run:
        expected["runs"] = ["recipe"] * int(fresh) + ["research"]
    recipe = "f-m0" if batch_size == 1 else ["f-m0"]
    planned = [("2b", recipe)] if fresh else []
    assert out == expected
    assert launches == ([] if dry_run else [*planned, ("2a", ["m1"])])


def test_active_later_leases_count_toward_share(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _scheduler(work, monkeypatch, busy=2, held={"f-m10", "f-m11"})
    assert tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)["2b"] == ["f-m0", "f-m1"]


def test_stopped_rows_are_skipped_and_default_shares_wait(work: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _scheduler(work, monkeypatch)
    tick.save_state({"cohort_states": {"r4-github": {"stopped": True}}})
    assert tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)["2b"] == ["f-m0", "f-m1"]
    tick.save_state({})
    cfg["cohort_share"] = {}
    assert tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)["2b"] == ["f-m0", "f-m1"]


def test_later_rows_never_displace_earlier_rows(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _scheduler(work, monkeypatch, early=8)
    assert tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)["2b"] == [
        f"f-m{n}" for n in range(5)]


def test_builder_priority_tags_duplicates_and_held_rows(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(publish, "torchlens_pin", lambda: "test")
    rows = tmp_path / "rows.tsv"
    rows.write_text("id\tfamily\tlist\n"
                    "m2\tf-two\tx_standin\n"
                    "m1\tf-one\tc2_tail\n"
                    "m3\tf-three\trung3-env\n"
                    "m2\tf-two\tx_standin\n")
    draw = tmp_path / "draw.jsonl"
    def census(model_id: str, family: str) -> str:
        return json.dumps({"schema": "census/1", "id": model_id, "family": family,
                           "repo": "org/repo", "checked": "now", "alive": True,
                           "archived": False, "eligible": True, "pytorch": "imports", "size_kb": 1})
    draw.write_text("\n".join([census("m9", "f-nine"), census("m8", "f-nine"),
                                census("m7", "f-two"), census("m2", "f-new"),
                                census("m6", "f-six")]))
    out = tmp_path / "preview.txt"
    result = worklist_build.build(data=DataLayout(tmp_path / "data"), rows=rows, draw=draw, out=out)
    assert out.read_text() == "m1 r3\n#m3 r3\nm2 r3\nm9 r4-github\nm6 r4-github\n"
    assert result["cohorts"] == {"r3": 2, "r4-github": 2} and result["held"] == 1
    with pytest.raises(ValueError, match="requires --out"):
        worklist_build.build(data=DataLayout(tmp_path), rows=rows)


def test_earlier_research_reserves_the_last_hub_slot(work: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _scheduler(work, monkeypatch)
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "2a" if int(i[1:]) < 10 else "2b"} for i in ids])
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 1)
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    assert out["2b"] == [] and out["2a"] == [["m0", "m1"]]


@pytest.mark.parametrize("on_hub", [False, True])
def test_hub_slots_go_to_2a_while_2b_is_off_the_hub(work: Path, monkeypatch: pytest.MonkeyPatch,
                                                    on_hub: bool) -> None:
    cfg = _scheduler(work, monkeypatch, early=0, on_hub=on_hub)
    (work / "worklist.txt").write_text("m0 r3\nm1 r3\nm2 r3\n")
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "2b" if i == "m0" else "2a"} for i in ids])
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 1)
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    if on_hub:  # the old placement: the 2b lane takes the hub's last slot ahead of research
        assert out["2b"] == ["f-m0"] and out["2a"] == []
    else:  # no workers: 2b waits and the hub slot goes to research
        assert out["2b"] == [] and out["2a"] == [["m1", "m2"]] and out["waiting_2b"] == 1
        assert out["2b_where"].startswith("workers only (2b_on_hub false") and out["2b_where"].endswith("none")


def test_2b_on_hub_defaults_false_and_must_be_a_boolean(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert tick.DEFAULTS["2b_on_hub"] is False
    cfg = _scheduler(work, monkeypatch)
    with pytest.raises(ValueError, match="2b_on_hub must be true or false"):
        tick_full.top_up(DataLayout(work / "data"), {**cfg, "2b_on_hub": "false"}, dry_run=True)


def test_cohort_share_counts_batches_and_never_mixes_cohorts(work: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import implement

    cfg = _scheduler(work, monkeypatch, early=4, batch_size=2)
    monkeypatch.setattr(implement, "batch_groups", lambda families, size, data: [
        families[n:n + size] for n in range(0, len(families), size)])
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    assert out["2b_batches"] == [["f-m0", "f-m1"], ["f-m2", "f-m3"],
                                 ["f-m10", "f-m11"], ["f-m12", "f-m13"]]
    assert out["cohorts"]["r4-github"]["2b"] == 2


def test_share_stays_capped_when_earlier_waiting_rows_are_leased(work: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _scheduler(work, monkeypatch, busy=2, held={"f-m0", "f-m1"})
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    assert out["2b"] == ["f-m10", "f-m11"]


def test_overlap_research_uses_only_spare_slots_and_stopped_research_is_skipped(
        work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _scheduler(work, monkeypatch)
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "2b" if int(i[1:]) < 10 else "2a"} for i in ids])
    data = DataLayout(work / "data")
    out = tick_full.top_up(data, cfg, dry_run=True)
    # buffer 2 x 5 x 1 = 10 families, none ready: both 2a lanes the ten rows fill
    assert out["2b"] == ["f-m0", "f-m1"] and out["2a"] == [
        [f"m{n}" for n in range(10, 18)], ["m18", "m19"]]
    tick.save_state({"cohort_states": {"r4-github": {"stopped": True}}})
    assert tick_full.top_up(data, cfg, dry_run=True)["2a"] == []


@pytest.mark.parametrize("cohort,threshold", [("r4-github", 37), ("r5-paper-only", 25),
                                             ("r5-ports", 55), ("r5-native-jax", -5)])
def test_reference_thresholds_use_strict_points(cohort: str, threshold: int) -> None:
    counts = {"settled": 100, "validated": max(0, threshold), "completed_2a": 100, "research_gap": 0}
    assert tick_full._trip_reason(cohort, counts) is None
    if threshold > 0:
        assert tick_full._trip_reason(cohort, {**counts, "validated": threshold - 1})
    assert tick_full._trip_reason(cohort, {**counts, "settled": 49, "research_gap": 20}) is None


def test_untagged_rows_are_rejected_only_when_overlap_is_enabled(work: Path) -> None:
    (work / "worklist.txt").write_text("m1 r3\nm2\n")
    assert tick_full._cohorts({}) == {"m1": "r3", "m2": "r3"}
    for cfg in ({"tripwires": True}, {"cohort_share": {"r4-github": .4}}):
        with pytest.raises(ValueError, match="every active worklist row"):
            tick_full._cohorts(cfg)


def test_first_cohort_is_never_stopped_by_later_tripwires() -> None:
    counts = {"settled": 100, "validated": 0, "completed_2a": 100, "research_gap": 100}
    assert tick_full._trip_reason("r3", counts) is None


def test_later_busy_research_does_not_reduce_primary_demand(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _scheduler(work, monkeypatch, early=16)
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [{"id": i, "step": "2a"} for i in ids])
    monkeypatch.setattr(tick_full, "_caps", lambda cfg: {"2a": 4, "2b": 14})
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter({"2a": 1}), {"f-m20"}))
    monkeypatch.setattr(tick_full, "_lane_busy", lambda cfg, family, kind: Counter({"r4-github": 1}))
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    assert out["cohorts"]["r3"]["2a"] == 2 and out["cohorts"]["r4-github"]["2a"] == 0


def test_later_research_can_use_spare_capacity_while_primary_2a_is_live(
        work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _scheduler(work, monkeypatch)
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "2b" if int(i[1:]) < 10 else "2a"} for i in ids])
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter({"2a": 1}), {"f-m0"}))
    monkeypatch.setattr(tick_full, "_lane_busy", lambda cfg, family, kind: Counter({"r3": 1}))
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    assert out["cohorts"]["r4-github"]["2a"] == 2


def test_stopped_cohort_is_not_newly_queued_for_forward_or_trace(work: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    (work / "worklist.txt").write_text("m1 r3\nm2 r4-github\n")
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "trace"} for i in ids])
    monkeypatch.setattr(tick_full, "_waiting", lambda spool: 0)
    monkeypatch.setattr(tick_full, "_not_pending", lambda ids, data: list(ids))
    cfg = {**tick.DEFAULTS, "stopped_cohorts": ["r4-github"]}
    out = tick_full.stage3_queue(DataLayout(work / "data"), cfg, dry_run=True)
    assert out["would_queue"] == ["m1"] and out["eligible"] == 1


def test_a_row_a_live_lane_holds_waits_and_the_rest_still_queue(work: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import stage3
    (work / "worklist.txt").write_text("m1 r3\nm2 r3\nm3 r3\n")
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "trace"} for i in ids])
    monkeypatch.setattr(tick_full, "_waiting", lambda spool: 0)
    monkeypatch.setattr(tick_full, "_not_pending", lambda ids, data: list(ids))
    monkeypatch.setattr(stage3, "lane_holds", lambda ids, data: {"m2": "run-1"})
    out = tick_full.stage3_queue(DataLayout(work / "data"), {**tick.DEFAULTS}, dry_run=True)
    assert out["would_queue"] == ["m1", "m3"] and out["held_by_lane"] == {"m2": "run-1"}


def test_2b_share_counts_lane_slots_not_research_keys(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import admission
    original = tick_full._lane_busy
    cfg = _scheduler(work, monkeypatch)
    monkeypatch.setattr(tick_full, "_lane_busy", original)
    family = {f"m{n}": f"f-m{n}" for n in range(20)}
    monkeypatch.setattr(admission, "lanes", lambda data: [
        {"kind": "2a", "keys": ["f-m10", "f-m11"]},
        {"kind": "2b", "keys": ["f-m12", "f-m13"]}])
    monkeypatch.setattr(admission, "queued", list)
    occupied = tick_full._lane_busy(cfg, family, "2b")
    assert occupied == {"r4-github": 1}
    assert tick_full._share_room("r4-github", cfg, {"2b": 5}, occupied["r4-github"]) == 1


def test_builder_keeps_scripted_retypes_and_adds_reopened_rows(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """A row a script retyped since the freeze stays; a row from an earlier rung that a script
    typed (m1778's case: not in the frozen rows) is appended; an agent's give-up settles a row."""

    monkeypatch.setattr(publish, "torchlens_pin", lambda: "test")
    data = DataLayout(tmp_path / "data")
    record = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    packet = read_json(Packet, FIXTURES / "f-resnet.packet.json")
    write_json(packet, data.packet(record.family))

    def typed(model_id: str, cls: str, by: str) -> None:
        give = GiveUp(cls=cls, reason="r", at="2026-10-09T10:00:00Z", by=by)
        write_json(dataclasses.replace(record, id=model_id, recipe=None, give_up=give),
                   data.model(model_id))

    typed("m2", "june_code:flagged", "script")      # frozen as needs_adapter, retyped by a script
    typed("m4", "needs_adapter", "agent")           # frozen, an agent gave up: settled
    typed("m1778", "standin:dense", "script")       # an earlier rung's row, outside the frozen set
    typed("m1779", "missing_dependency", "agent")   # outside, but an agent typed it
    rows = tmp_path / "rows.tsv"
    rows.write_text("id\tfamily\tlist\tstate\tclass\n"
                    "m2\tf-two\tneeds_adapter\ttyped\tneeds_adapter\n"
                    "m4\tf-four\tneeds_adapter\ttyped\tneeds_adapter\n")
    out = tmp_path / "preview.txt"
    result = worklist_build.build(data=data, rows=rows, out=out)
    assert out.read_text() == "m2 r3\nm1778 r3\n"
    assert result["reopened"] == 1 and result["cohorts"] == {"r3": 2}


def _saturated(work: Path, monkeypatch: pytest.MonkeyPatch, lines: Sequence[str],
               research: set[str], *, busy_2a: int = 0, held: set[str] | None = None) -> dict:
    """Every 2b slot busy (5 of 5); ``research`` rows read 2a, the rest 2b."""
    cfg = _scheduler(work, monkeypatch, busy=5, held=held)
    (work / "worklist.txt").write_text("\n".join(lines))
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "2a" if i in research else "2b"} for i in ids])
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter({"2b": 5, "2a": busy_2a}), held or set()))
    later = Counter("r4-github" for key in held or ())
    monkeypatch.setattr(tick_full, "_lane_busy", lambda cfg, family, kind: (
        Counter({"r3": busy_2a}) if kind == "2a" else later))
    return cfg


def test_saturated_2b_with_a_short_queue_still_launches_2a(work: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    research = {f"m{n}" for n in range(10, 30)}
    lines = [f"m{n} r3" for n in [*range(6), *range(10, 30)]]
    cfg = _saturated(work, monkeypatch, lines, research)
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    # six families ready, buffer 10: one 2a lane though no 2b slot is free this pass
    assert out["2b"] == [] and out["2a"] == [[f"m{n}" for n in range(10, 18)]]


def test_a_later_cohort_with_no_free_2b_share_still_researches(work: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    research = {f"m{n}" for n in range(14, 30)}
    lines = ["m0 r3", *[f"m{n} r4-github" for n in range(10, 30)]]
    cfg = _saturated(work, monkeypatch, lines, research, held={"f-m10", "f-m11"})
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    assert out["2b"] == [] and out["2a"] == [[f"m{n}" for n in range(14, 22)]]
    cfg["cohort_share"] = {}  # a cohort with no 2b share never reaches 2b: no research for it
    assert tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)["2a"] == []


@pytest.mark.parametrize("ready,busy_2a,buffer", [(12, 0, None), (3, 1, None), (0, 0, 0)])
def test_a_full_buffer_launches_no_2a(work: Path, monkeypatch: pytest.MonkeyPatch,
                                      ready: int, busy_2a: int, buffer: int | None) -> None:
    research = {f"m{n}" for n in range(20, 30)}
    lines = [f"m{n} r3" for n in [*range(ready), *range(20, 30)]]
    cfg = {**_saturated(work, monkeypatch, lines, research, busy_2a=busy_2a),
           "2b_buffer_families": buffer}
    # buffer 10: 12 ready, or 3 ready plus a running lane's 8; or a configured buffer of 0
    assert tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)["2a"] == []


@pytest.mark.parametrize("busy_2a,hub,lanes", [(0, 10, 4), (1, 10, 3), (0, 2, 2)])
def test_the_2a_cap_and_hub_slots_still_bind_a_large_buffer(
        work: Path, monkeypatch: pytest.MonkeyPatch, busy_2a: int, hub: int, lanes: int) -> None:
    research = {f"m{n}" for n in range(100)}
    cfg = {**_saturated(work, monkeypatch, [f"m{n} r3" for n in range(100)], research,
                        busy_2a=busy_2a), "2b_buffer_families": 1000}
    monkeypatch.setattr(tick_full, "hub_slots", lambda: hub)
    out = tick_full.top_up(DataLayout(work / "data"), cfg, dry_run=True)
    assert len(out["2a"]) == lanes and all(len(chunk) == 8 for chunk in out["2a"])


def test_buffer_defaults_to_two_sets_of_2b_lanes_and_rejects_bad_values() -> None:
    caps = {"2a": 4, "2b": 14}
    assert tick.DEFAULTS["2b_buffer_families"] is None
    assert tick_full._buffer_target({}, caps) == 28
    assert tick_full._buffer_target({"families_per_2b_lane": 4}, caps) == 112
    assert tick_full._buffer_target({"2b_buffer_families": 50}, caps) == 50
    for bad in (-1, True, 2.5, "40"):
        with pytest.raises(ValueError, match="2b_buffer_families"):
            tick_full._buffer_target({"2b_buffer_families": bad}, caps)


def _lane_ended(work: Path, family: str, at: float) -> Path:
    """A 2b lane folder whose brief was written at ``at`` and whose result came a minute later."""

    lane = work.parent / "2b" / family
    lane.mkdir(parents=True)
    for name, when in (("BRIEF.md", at), ("BRIEF.result.md", at + 60)):
        (lane / name).write_text(name)
        os.utime(lane / name, (when, when))
    return lane


def test_a_family_whose_last_lane_left_its_2b_rows_untouched_is_not_relaunched(
        work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The 2026-10-21 forward loop: a lane re-ran the member it could build and ended, the row
    that sent it stayed at 2b, and every tick pass sent the family round again."""

    cfg = _scheduler(work, monkeypatch)
    (work / "worklist.txt").write_text("m0 r3\nm1 r3\n")
    data = DataLayout(work / "data")
    record = data.model("m0")
    record.parent.mkdir(parents=True)
    record.write_text("{}")
    os.utime(record, (1_000, 1_000))
    lane = _lane_ended(work, "f-m0", 2_000)
    out = tick_full.top_up(data, cfg, dry_run=True)
    assert out["2b"] == ["f-m1"] and out["2b_spent"] == ["f-m0"]
    assert out["waiting_2b"] == 1
    os.utime(record, (2_030, 2_030))  # the lane wrote the row (an infra give-up): retried
    assert tick_full.top_up(data, cfg, dry_run=True)["2b"] == ["f-m0", "f-m1"]
    os.utime(record, (1_000, 1_000))
    (lane / "deferred.json").write_text("{}")  # the lane asked for a new session
    assert tick_full.top_up(data, cfg, dry_run=True)["2b"] == ["f-m0", "f-m1"]
    (lane / "deferred.json").unlink()
    os.utime(lane / "BRIEF.result.md", (1_500, 1_500))  # a result from an earlier lane
    assert tick_full.top_up(data, cfg, dry_run=True)["2b"] == ["f-m0", "f-m1"]
