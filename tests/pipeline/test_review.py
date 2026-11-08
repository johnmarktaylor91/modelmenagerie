"""Fidelity review: the reviewer picked from what actually ran, review records, calibration scoring
and review batches. fleet-run, the role table and lane launches are stood in for; nothing starts."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, calibration, ledger, read_json, review, write_json
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import Amendment, LedgerRow, ModelRecord, Packet, RecordError

FIXTURES = Path(__file__).parent / "fixtures"
HARNESS = {"model-design-review": "codex", "model-default": "claude",
           "model-review-third-lab-plan": "antigravity"}
ADAPTER = "/tools/fleet-run/abi/v7/fleet-run-adapter"


def _run(harness: str, model: str) -> dict:
    argv = ["/usr/bin/env", "python3", "lane_failover.py", "run", "--", ADAPTER, harness,
            "BRIEF.md", "--model", model]
    return {"kind": harness, "adapter": harness, "state": "TERMINAL", "spec": {"argv": argv}}


@pytest.fixture
def lab(monkeypatch, tmp_path):
    """Stand-ins for fleet-run and the role table; ``lab.runs`` maps run ids to records."""

    class Lab:
        runs = {"claude-run": _run("claude", "opus-x"), "codex-run": _run("codex", "sol-x")}
        harness = dict(HARNESS)

    monkeypatch.setattr(review, "_fleet_run", lambda run_id: Lab.runs[run_id])
    monkeypatch.setattr(review, "_role_harness", lambda role: Lab.harness[role])
    monkeypatch.setenv("LANE_FAILOVER_LOG", str(tmp_path / "failover.log"))
    return Lab


@pytest.fixture
def data(tmp_path) -> DataLayout:
    root = tmp_path / "data"
    for name, dest in [("m8839.model.json", "models/88/m8839.json"),
                       ("f-resnet.packet.json", "research/re/f-resnet.json")]:
        (root / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURES / name, root / dest)
    return DataLayout(root)


def _seat_all(data: DataLayout) -> None:
    for role in review.REVIEWER_ROLES:
        _seat(data, role, (19, 19, 19))


def test_reviewer_is_the_other_lab_from_what_ran(lab, data):
    _seat_all(data)
    assert review.pick_reviewer("claude-run", data=data) == "model-design-review"
    assert review.pick_reviewer("codex-run", data=data) == "model-default"
    assert review.author_of("codex-run")["model"] == "sol-x"


def test_third_lab_when_no_main_role_is_independent(lab, data, tmp_path):
    _seat_all(data)
    lab.harness["model-default"] = "codex"  # routing moved both main roles onto one lab
    assert review.pick_reviewer("codex-run", data=data) == review.THIRD_LAB
    (tmp_path / "failover.log").write_text(
        "2026-10-03T06:56:45Z lane-failover run=claude-run both Claude accounts are at a usage "
        "limit; lane claude-run handed off to gpt-sol as prompt-handoff (started)\n")
    lab.harness["model-default"] = "claude"
    assert review.author_of("claude-run")["handed_off_to"] == "gpt-sol"
    assert review.pick_reviewer("claude-run", data=data) == review.THIRD_LAB



def test_a_held_third_lab_seat_is_never_picked(lab, data, monkeypatch):
    _seat_all(data)
    lab.harness["model-default"] = "codex"  # only the third lab is independent of a Codex author
    monkeypatch.setattr(review.admission, "third_lab_held", lambda: True)
    with pytest.raises(review.ReviewError, match="no calibrated reviewer"):
        review.pick_reviewer("codex-run", data=data)
    assert review.pick_reviewer("claude-run", data=data) == "model-design-review"

def _seat(data: DataLayout, role: str, agree: tuple[int, ...], cases: int = 20) -> None:
    """Calibration rows for ``role`` at the current prompt: one run per entry of ``agree`` (for
    a voting seat, its base role's runs, each entry ``votes`` times)."""

    if role in review.ENSEMBLES:
        base, votes = review.ENSEMBLES[role]
        _seat(data, base, tuple(a for a in agree for _ in range(votes)), cases)
        return
    path = data.root / "review" / "calibration-cases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    counted = [{"case": f"k{i}", "expected": "not_fabrication"} for i in range(cases)]
    path.write_text("".join(json.dumps(c) + "\n" for c in counted))
    sid, prompt = calibration.set_id(counted), review.prompt_version()
    with (data.root / "review" / "calibration.jsonl").open("a") as handle:
        for run, good in enumerate(agree, 1):
            for i, case in enumerate(counted):
                handle.write(json.dumps({"role": role, "prompt": prompt, "set": sid, "run": run,
                                         "case": case["case"], "expected": case["expected"],
                                         "given": "minor" if i < good else "fabrication",
                                         "agree": i < good}) + "\n")


def test_only_a_seat_whose_calibration_counts_reviews(lab, data):
    with pytest.raises(review.ReviewError, match="no calibrated reviewer"):
        review.pick_reviewer("claude-run", data=data)  # nobody calibrated at this prompt
    _seat(data, "model-design-review", (17, 17, 17))  # bar 17 of 20: provisional counts
    _seat(data, "model-default", (16, 18, 18))  # its lowest run is under the bar
    assert review.pick_reviewer("claude-run", data=data) == "model-design-review"
    with pytest.raises(review.ReviewError, match="no calibrated reviewer"):
        review.pick_reviewer("codex-run", data=data)
    _seat(data, review.THIRD_LAB, (19, 20, 19))
    assert review.pick_reviewer("codex-run", data=data) == review.THIRD_LAB


def test_a_non_agent_run_is_refused(lab, data):
    lab.runs["job"] = {"kind": "remote", "adapter": "run-on", "spec": {"argv": ["run-on"]}}
    with pytest.raises(review.ReviewError, match="not an agent lane"):
        review.pick_reviewer("job", data=data)


def test_a_worker_lane_is_authored_by_its_launch_event_harness(lab, data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    _seat_all(data)
    for run, harness in (("w-claude", "claude"), ("c-agy", "antigravity")):
        lab.runs[run] = {"kind": "run-on", "adapter": "run-on", "spec": {"argv": ["run-on"]}}
        log = tmp_path / "work" / "lanes" / run / "events.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text(json.dumps({"verb": "tier", "family": "f"}) + "\n" + json.dumps(
            {"verb": "launch", "harness": harness, "model": "m-x", "kind": "2b"}) + "\n")
    assert review.author_of("w-claude")["harness"] == "claude"
    assert review.pick_reviewer("w-claude", data=data) == "model-design-review"
    assert review.pick_reviewer("c-agy", data=data) == "model-design-review"  # either main lab


def _review(**changes) -> dict:
    base = {"schema": "review/1", "id": "m8839", "reviewer": "model-default",
            "verdict": "fabrication", "signature": ["residual shortcut (sec. 3.2)"],
            "components": [{"name": "shortcut", "source": "identity add", "ours": "absent",
                            "finding": "missing", "note": "no residual add"}],
            "summary": "The residual shortcut is missing."}
    return {k: v for k, v in {**base, **changes}.items() if v is not None}


def test_record_refuses_a_review_without_a_verdict(data, tmp_path, monkeypatch):
    monkeypatch.delenv("FLEET_RUN_ID", raising=False)
    store = StoreLayout(tmp_path / "store")
    path = tmp_path / "r.json"
    for bad in (_review(verdict=None), _review(verdict="ok"), _review(reviewer="someone"),
                _review(components=[{"name": "shortcut", "finding": "match"}])):
        path.write_text(json.dumps(bad))
        with pytest.raises(RecordError):
            review.record("m8839", path, data=data, store=store)
    assert ledger.read_all(data=data) == []


def test_record_appends_a_review_row_with_the_findings_blob(data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("FLEET_RUN_ID", "review-lane-1")
    store = StoreLayout(tmp_path / "store")
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_review()))
    _seat(data, "model-default", (19, 18, 20))
    row = review.record("m8839", path, data=data, store=store)
    assert (row.stage, row.verdict, row.recipe_rev, row.env, row.reviewer) == (
        "review", "fabrication", "457cba223fe9", "review:model-default", "model-default")
    assert row.msg == f"prompt {review.prompt_version()}; case 1" and not review.needs_second_review(row)
    assert store_mod.get(row.findings, store=store).read_bytes() == path.read_bytes()
    assert ledger.read_all(data=data) == [row]
    event = json.loads((tmp_path / "work/lanes/review-lane-1/events.jsonl").read_text())
    assert (event["family"], event["verb"], event["id"], event["verdict"]) == (
        "f-resnet", "review", "m8839", "fabrication")


def test_record_refuses_an_uncalibrated_seat_and_a_changed_prompt(data, tmp_path,
                                                                   monkeypatch):
    monkeypatch.delenv("FLEET_RUN_ID", raising=False)
    store = StoreLayout(tmp_path / "store")
    lane = tmp_path / "lane"
    (lane / "out").mkdir(parents=True)
    path = lane / "out" / "m8839.json"
    path.write_text(json.dumps(_review()))
    with pytest.raises(RecordError, match="no passing calibration"):
        review.record("m8839", path, data=data, store=store)
    _seat(data, "model-default", (19, 19))  # two runs are not a calibration
    with pytest.raises(RecordError, match="uncalibrated, 2 runs"):
        review.record("m8839", path, data=data, store=store)
    _seat(data, "model-default", (17, 19, 19))  # a lowest run at the bar: provisional
    (lane / "lane.json").write_text(json.dumps({"run": "r", "prompt": "0ldhash0"}))
    with pytest.raises(RecordError, match="was written under review prompt 0ldhash0"):
        review.record("m8839", path, data=data, store=store)
    (lane / "lane.json").write_text(json.dumps({"run": "r", "prompt": review.prompt_version()}))
    row = review.record("m8839", path, data=data, store=store)
    assert review.needs_second_review(row)
    assert ledger.read_all(data=data) == [row]


def test_record_stamps_the_source_digest_its_case_was_built_with(data, tmp_path, monkeypatch):
    monkeypatch.delenv("FLEET_RUN_ID", raising=False)
    monkeypatch.setattr(store_mod, "materialize", lambda sha, dest, *, store: dest)
    lane = tmp_path / "lane"
    (lane / "out").mkdir(parents=True)
    digest = review._model_case("m8839", lane / "cases" / "m8839", data,
                                StoreLayout(tmp_path / "store"), [])
    assert digest == review.source_digest(read_json(Packet, data.packet("f-resnet")))
    (lane / "sources.json").write_text(json.dumps({"m8839": digest}))
    path = lane / "out" / "m8839.json"
    path.write_text(json.dumps(_review()))
    _seat(data, "model-default", (19, 18, 20))
    row = review.record("m8839", path, data=data, store=StoreLayout(tmp_path / "store"))
    assert row.msg.endswith(f"; case 1; src {digest}")
    assert not review.stale(row, read_json(Packet, data.packet("f-resnet")))


@pytest.mark.parametrize("delta", ["", "C3 fully connected, plain linear output"])
def test_case_folder_carries_the_member_delta(data, tmp_path, monkeypatch, delta):
    monkeypatch.setattr(store_mod, "materialize", lambda sha, dest, *, store: dest)
    path = data.packet("f-resnet")
    packet = json.loads(path.read_text())
    packet["members"] = [{"id": "m8839", "delta": delta}]
    path.write_text(json.dumps(packet))
    case = tmp_path / "case"
    review._model_case("m8839", case, data, StoreLayout(tmp_path / "store"), [])
    member = case / "ours" / "member.md"
    assert (case / "ours" / "recipe.json").is_file()
    assert member.is_file() == bool(delta)
    if delta:
        assert delta in member.read_text()


def test_launch_batches_by_reviewer(lab, data, tmp_path, monkeypatch):
    _seat_all(data)
    for i in range(6):
        model = json.loads((FIXTURES / "m8839.model.json").read_text())
        path = data.model(f"m90{i}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**model, "id": f"m90{i}"}))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(store_mod, "materialize", lambda sha, dest, *, store: dest)
    lanes = tmp_path / "work" / "lanes" / "codex-run"
    lanes.mkdir(parents=True)
    (lanes / "events.jsonl").write_text(json.dumps(
        {"at": "2026-10-04T18:00:00Z", "verb": "done", "family": "f-resnet"}) + "\n")
    calls = []
    monkeypatch.setattr(review.drive, "launch", lambda kind, role, brief, **kw: calls.append(
        (role, kw["keys"], kw["lab"])) or f"run{len(calls)}")
    runs = review.launch([f"m90{i}" for i in range(6)], data=data)
    assert runs == ["run1", "run2"]
    assert calls == [("model-default", ("m900", "m901", "m902", "m903", "m904"), "claude"),
                     ("model-default", ("m905",), "claude")]


def test_codex_built_rows_wait_for_a_seat_other_than_codex(lab, data, tmp_path, monkeypatch):
    """m14359 and m1191 (rung 3 section 3.3): reviewed only once a non-Codex seat counts."""

    _seat(data, "model-design-review", (19, 19, 19))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    lanes = tmp_path / "work" / "lanes" / "codex-run"
    lanes.mkdir(parents=True)
    (lanes / "events.jsonl").write_text(json.dumps(
        {"at": "2026-10-04T19:41:05Z", "verb": "done", "family": "f-resnet"}) + "\n")
    monkeypatch.setattr(review.drive, "launch", lambda *a, **kw: pytest.fail("launched"))
    with pytest.raises(review.ReviewError, match="no calibrated reviewer"):
        review.launch(["m8839"], data=data)
    _seat(data, review.THIRD_LAB, (19, 19, 19))
    assert review.pick_reviewer("codex-run", data=data) == review.THIRD_LAB


def test_third_lab_seat_runs_read_only_batches_and_its_answer_becomes_out_files(tmp_path,
                                                                                 monkeypatch):
    folder = tmp_path / "lane"
    (folder / "cases" / "x0").mkdir(parents=True)
    (folder / "cases" / "x0" / "paper.txt").write_text("p")
    brief = folder / "BRIEF.md"
    brief.write_text("brief")
    calls = []

    def launch(kind, role, path, **kw):
        if "refuse" in kw["label"]:
            raise review.drive.Refused("no_admission", "agy-quota gate: no room", role=role)
        calls.append({"kind": kind, "role": role, "brief": path, **kw})
        return f"agy-{len(calls)}"

    monkeypatch.setattr(review.drive, "launch", launch)
    assert review._start(review.THIRD_LAB, brief, ["c01", "c02"], "lbl") == "agy-1"
    assert (calls[0]["kind"], calls[0]["lab"], calls[0]["answer"]) == (
        "review", "antigravity", "blocks")
    assert calls[0]["ids"] == ["c01", "c02"] and calls[0]["keys"] == ["c01", "c02"]
    seven = [f"x{i}" for i in range(7)]
    review._start(review.THIRD_LAB, brief, seven, "lbl7", keys=())  # five cases per lane
    lane = json.loads((folder / "lane.json").read_text())
    assert lane["runs"] == ["agy-2", "agy-3"] and lane["harness"] == "antigravity"
    first, second = calls[1]["brief"], calls[2]["brief"]
    assert (first.parent, first.name, calls[1]["keys"]) == (folder, "BRIEF-b1.md", [])
    assert "x4" in first.read_text().splitlines()[0] and "`cases/x0/paper.txt`" in first.read_text()
    assert second.read_text().startswith("# This lane's cases: x5, x6")
    (folder / "agy-a").mkdir()
    one, two = _review(id="x0", reviewer=review.THIRD_LAB), _review(id="x1", verdict="minor")
    (folder / "agy-a" / "result.md").write_text(
        f"x0:\n```json\n{json.dumps(one)}\n```\nand\n```json\n{json.dumps(two)}\n```\n")
    assert review.collect_agy(folder) == ["x0", "x1"]
    assert review.collect_agy(folder) == []
    result = (folder / "BRIEF.result.md").read_text()
    assert result.startswith("x0 fabrication\nx1 minor\nx2 missing")
    with pytest.raises(review.drive.Refused):
        review._start(review.THIRD_LAB, brief, ["c01"], "refuse")


def test_a_refused_later_batch_is_started_by_resume(tmp_path, monkeypatch):
    folder = tmp_path / "lane"
    folder.mkdir()
    brief = folder / "BRIEF.md"
    brief.write_text("brief")
    refused, started = {"lbl-b2"}, []

    def launch(kind, role, path, **kw):
        if kw["label"] in refused:
            raise review.drive.Refused("no_admission", "no room", role=role)
        started.append(kw["label"])
        return f"agy-{len(started)}"

    monkeypatch.setattr(review.drive, "launch", launch)
    review._start(review.THIRD_LAB, brief, [f"x{i}" for i in range(7)], "lbl", keys=())
    assert started == ["lbl-b1"] and review.unstarted(folder) == [2]
    assert review.resume(folder) == 0 and review.unstarted(folder) == [2]
    refused.clear()
    assert review.resume(folder) == 1 and review.unstarted(folder) == []
    assert json.loads((folder / "lane.json").read_text())["runs"] == ["agy-1", "agy-2"]


# --- the tick's review stage -----------------------------------------------------------------


def _lane(name: str, prompt: str, verdicts: dict[str, dict]) -> Path:
    folder = review.work_root() / "lanes" / name
    (folder / "out").mkdir(parents=True)
    (folder / "lane.json").write_text(json.dumps({"run": f"run-{name}", "prompt": prompt}))
    for case, body in verdicts.items():
        (folder / "cases" / case).mkdir(parents=True)
        (folder / "out" / f"{case}.json").write_text(json.dumps(body))
    return folder


def test_the_stage_calibrates_and_holds_until_a_seat_counts(data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(review.admission, "third_lab_held", lambda: False)
    called = []

    def calibrate(*, role, data):
        called.append(role)
        if role == review.THIRD_LAB:
            raise RecordError("fleet-run start failed: Guard admission refused")
        return {"state": "launched"}

    monkeypatch.setattr(review, "calibrate", calibrate)
    monkeypatch.setattr(review, "launch", lambda *a, **kw: pytest.fail("launched"))
    out = review.stage(["m8839"], data=data, room=2, held=set(), dry_run=False)
    assert called == list(review.REVIEWER_ROLES) and out["held"].startswith("no reviewer seat counts")
    assert out["calibration"]["model-default"] == "launched"  # a broken seat never stops the others
    assert out["calibration"][review.THIRD_LAB].startswith("not started: fleet-run start failed")
    for role in review.REVIEWER_ROLES:  # every seat calibrated: no new calibration
        _seat(data, role, (10, 10, 10))
    out = review.stage(["m8839"], data=data, room=2, held=set(), dry_run=False)
    assert len(called) == len(review.REVIEWER_ROLES) and "calibration" not in out and "held" in out


def test_the_stage_starts_one_role_batches_up_to_its_room(lab, data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(review.admission, "third_lab_held", lambda: True)
    _seat(data, "model-design-review", (19, 19, 19))
    ids = [f"m9{i:02d}" for i in range(13)]
    model = json.loads((FIXTURES / "m8839.model.json").read_text())
    for i, model_id in enumerate(ids):
        path = data.model(model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        family = "f-claude" if i < 11 else "f-codex"
        path.write_text(json.dumps({**model, "id": model_id, "family": family}))
    monkeypatch.setattr(review, "_authors", lambda: {"f-claude": "claude-run", "f-codex": "codex-run"})
    _lane("review-m900-1-x", review.prompt_version(), {"m900": _review()})
    monkeypatch.setattr(review, "calibrate", lambda *, role, data: {"state": "running"})  # the others
    started = []
    monkeypatch.setattr(review, "launch", lambda batch, *, data, roles: started.append(
        (sorted(set(roles.values())), list(batch))) or [f"run{len(started)}"])
    monkeypatch.setattr(review, "lane_state", lambda folder: ("r", "RUNNING"))
    out = review.stage(ids, data=data, room=2, held={"m901"}, dry_run=False)
    # m900 was taken up today and m901 is held; the codex-built family has no other calibrated seat
    assert started == [(["model-design-review"], ["m902", "m903", "m904", "m905", "m906"]),
                       (["model-design-review"], ["m907", "m908", "m909", "m910"])]
    assert out["left"] == {"held": 1, "taken": 1, "no_seat": 2, "unlooked": 0, "second": 0}
    assert out["launched"] == ["run1", "run2"] and out["settle"]["live"] == 1
    cache = json.loads((tmp_path / "work" / "review" / "reviewer-seats.json").read_text())
    assert cache["f-claude"]["role"] == "model-design-review" and cache["f-codex"]["role"] is None


def test_a_classic_recipe_reference_copies_its_module_file(data, tmp_path):
    model = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    recipe = replace(model.recipe, kind="classic", call=None,
                     adapter="menagerie.classics.CrossFormer:build_crossformer")
    review._ours(replace(model, recipe=recipe), tmp_path / "ours", data)
    assert (tmp_path / "ours" / "CrossFormer.py").read_bytes() == (
        review.CLASSICS / "CrossFormer.py").read_bytes()


def test_a_case_that_cannot_be_built_leaves_no_lane_folder(lab, data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))

    def broken(model_id, folder, data, store, rows):
        folder.mkdir(parents=True)
        raise FileNotFoundError("classics/x:build.py")

    monkeypatch.setattr(review, "_model_case", broken)
    with pytest.raises(FileNotFoundError):
        review.launch(["m8839"], data=data, roles={"m8839": "model-default"})
    assert not list((tmp_path / "work" / "review" / "lanes").glob("review-*"))


def test_a_refused_launch_leaves_no_folder_and_holds_no_row(lab, data, tmp_path, monkeypatch):
    """2 Nov 2026: a batch refused at launch kept its case folders, and each held its 5 rows a
    day, so 226 of the 231 rows at review waited with no lane on them."""
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(review, "_model_case", lambda m, folder, *a: folder.mkdir(parents=True) or "d")

    def refused(*args, **kwargs):
        raise review.drive.Refused("no_admission", "role model-judgment: no eligible seat")

    monkeypatch.setattr(review, "_start", refused)
    with pytest.raises(review.drive.Refused):
        review.launch(["m8839"], data=data, roles={"m8839": "model-default"})
    lanes = tmp_path / "work" / "review" / "lanes"
    assert not list(lanes.glob("review-*"))
    (lanes / "review-m8839-1-old" / "cases" / "m8839").mkdir(parents=True)  # one left before
    assert review._taken_up(data) == set()


def test_a_refused_seat_does_not_hold_another_seats_batch(data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(review, "seats", lambda data: {"model-design-review": "passed"})
    monkeypatch.setattr(review, "settle", lambda **kw: {})
    monkeypatch.setattr(review, "plan", lambda ids, **kw: (
        [("model-judgment-3vote", ["m1", "m2"]), ("model-judgment-3vote", ["m3"]),
         ("model-design-review", ["m4"])], {}))
    tried = []

    def launch(batch, *, data, roles):
        tried.append(list(batch))
        if "model-judgment-3vote" in roles.values():
            raise review.drive.Refused("no_admission", "no eligible seat")
        return ["run1"]

    monkeypatch.setattr(review, "launch", launch)
    out = review.stage(["m1", "m2", "m3", "m4"], data=data, room=2, held=set(), dry_run=False)
    assert tried == [["m1", "m2"], ["m4"]] and out["launched"] == ["run1"]
    assert "no eligible seat" in out["refused"]


def test_an_adapter_case_carries_the_classics_it_imports(tmp_path):
    adapter = tmp_path / "f-x.py"
    adapter.write_text("from menagerie.classics.CrossFormer import build\n")
    review._copy_classics(adapter, tmp_path)
    assert (tmp_path / "CrossFormer.py").read_bytes() == (
        review.CLASSICS / "CrossFormer.py").read_bytes()


def test_a_batch_whose_case_cannot_be_built_skips_only_that_batch(lab, data, tmp_path,
                                                                  monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(review.admission, "third_lab_held", lambda: True)
    _seat(data, "model-design-review", (19, 19, 19))
    monkeypatch.setattr(review, "calibrate", lambda *, role, data: {"state": "running"})
    monkeypatch.setattr(review, "settle", lambda **_: {"recorded": [], "refused": {}, "live": 0})
    batches = [("model-design-review", ["m1", "m2"]), ("model-design-review", ["m3"]),
               ("model-design-review", ["m4"])]
    monkeypatch.setattr(review, "plan", lambda ids, *, data, held: (batches, {}))

    def launch(batch, *, data, roles):
        if batch == ["m1", "m2"]:
            raise FileNotFoundError("classics/x:build.py")
        return [f"run-{batch[0]}"]

    monkeypatch.setattr(review, "launch", launch)
    out = review.stage(["m1", "m2", "m3", "m4"], data=data, room=1, held=set(), dry_run=False)
    assert out["launched"] == ["run-m3"]
    assert out["failed"] == [{"ids": ["m1", "m2"], "error": "classics/x:build.py"}]


def test_a_provisional_fabrication_waits_for_a_different_seat(lab, data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(review.admission, "third_lab_held", lambda: True)
    monkeypatch.setattr(review, "_authors", lambda: {})  # a June classic: no authoring lane
    model = read_json(ModelRecord, data.model("m8839"))
    first = LedgerRow(id="m8839", stage="review", at="2026-10-29T03:45:00Z", host="hub",
                      env="review:model-design-review", recipe_rev=model.recipe.rev,
                      outcome="pass", msg=f"prompt p; {review.PROVISIONAL}",
                      verdict="fabrication", reviewer="model-design-review")
    ledger.append([first], data=data)
    _seat(data, "model-design-review", (17, 18, 17))
    batches, left = review.plan(["m8839"], data=data, held=set())
    assert batches == [] and left["second"] == 1  # only the seat that made the call counts
    _seat(data, "model-default", (19, 19, 19))
    batches, left = review.plan(["m8839"], data=data, held=set())
    assert batches == [("model-default", ["m8839"])] and left["second"] == 0


def test_a_confirm_only_seat_takes_a_claude_built_provisional_fabrication_once(
        lab, data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(review.admission, "third_lab_held", lambda: True)
    monkeypatch.setattr(review, "_authors", lambda: {"f-resnet": "claude-run"})
    model = read_json(ModelRecord, data.model("m8839"))
    first = LedgerRow(id="m8839", stage="review", at="2026-10-29T03:45:00Z", host="hub",
                      env="review:model-design-review", recipe_rev=model.recipe.rev,
                      outcome="pass", msg=f"prompt p; {review.PROVISIONAL}",
                      verdict="fabrication", reviewer="model-design-review")
    ledger.append([first], data=data)
    _seat(data, "model-design-review", (17, 18, 17))
    _seat(data, "model-judgment", (19, 19, 19))
    batches, left = review.plan(["m8839"], data=data, held=set())
    assert batches == [("model-judgment", ["m8839"])]  # the own-lab seat may only confirm
    no = LedgerRow(id="m8839", stage="review", at="2026-10-29T05:00:00Z", host="hub",
                   env="review:model-judgment", recipe_rev=model.recipe.rev, outcome="pass",
                   msg="prompt p", verdict="minor", reviewer="model-judgment")
    ledger.append([no], data=data)
    batches, left = review.plan(["m8839"], data=data, held=set())
    assert batches == [] and left["second"] == 1  # never asked twice; the row waits
    packet = read_json(Packet, data.packet("f-resnet"))
    retired = Amendment(at="2026-10-29T06:00:00Z", why="paper check", evidence=("s1",),
                        changed=("sources",), by="paper-rejected")
    write_json(replace(packet, amendments=(*packet.amendments, retired)), data.packet("f-resnet"))
    batches, left = review.plan(["m8839"], data=data, held=set())
    assert left["second"] == 0 and ("model-judgment", ["m8839"]) not in batches  # a first review anew


def test_a_taken_row_is_free_once_its_review_is_recorded(data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    _lane("review-m8839-1-x", review.prompt_version(), {"m8839": _review()})
    assert review._taken_up(data) == {"m8839"}  # a lane took it up today
    model = read_json(ModelRecord, data.model("m8839"))
    earlier = LedgerRow(id="m8839", stage="review", at="2020-01-01T00:00:00Z", host="hub",
                        env="review:model-default", recipe_rev=model.recipe.rev, outcome="pass",
                        msg="prompt p", verdict="minor", reviewer="model-default")
    ledger.append([earlier], data=data)
    assert review._taken_up(data) == {"m8839"}  # a verdict from before the case is not its answer
    ledger.append([replace(earlier, at="2999-01-01T00:00:00Z")], data=data)
    assert review._taken_up(data) == set()


def test_a_voting_seat_confirms_only_on_a_majority_of_its_votes(data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("FLEET_RUN_ID", raising=False)
    assert review.majority(["fabrication", "minor", "fabrication"]) == "fabrication"
    assert review.majority(["fabrication", "minor", "faithful", "minor"]) == "minor"
    assert review.majority(["fabrication", None, None]) is None
    role = "model-judgment-3vote"
    folder = review.work_root() / "lanes" / "review-m8839-1-x"
    (folder / "cases" / "m8839").mkdir(parents=True)
    (folder / "cases" / "m8839" / "run.json").write_text("{}")
    started = []
    monkeypatch.setattr(review.drive, "launch", lambda kind, r, brief, **kw: started.append(
        (r, brief, kw["keys"])) or f"run-{len(started)}")
    run = review._start(role, review._brief(role, folder, ["m8839"]), ["m8839"], "review-m8839-1")
    assert run == "run-1" and [(r, tuple(k)) for r, _, k in started] == [
        ("model-judgment", ("m8839",)), ("model-judgment", ()), ("model-judgment", ())]
    assert all(b.parent.name == f"v{k}" and (b.parent / "cases/m8839/run.json").is_file()
               for k, (_, b, _) in enumerate(started, 1))
    lane = json.loads((folder / "lane.json").read_text())
    assert (lane["harness"], lane["runs"], lane["role"]) == ("votes", ["run-1", "run-2", "run-3"], role)
    for k, verdict in enumerate(("fabrication", "minor", "fabrication"), 1):
        body = _review() if verdict == "fabrication" else {**_review(), "verdict": verdict}
        (folder / f"v{k}" / "out" / "m8839.json").write_text(json.dumps(
            {**body, "reviewer": "model-judgment", "summary": f"vote {k}"}))
    _seat(data, role, (19, 19, 19))
    monkeypatch.setattr(review, "lane_state", lambda f: ("run-1", "TERMINAL"))
    out = review.settle(data=data, store=StoreLayout(tmp_path / "store"))
    assert out["recorded"] == ["m8839"] and not out["refused"]
    (row,) = ledger.read_all(data=data)
    assert (row.reviewer, row.verdict) == (role, "fabrication")
    tallied = json.loads((folder / "out" / "m8839.json").read_text())
    assert tallied["votes"] == ["fabrication", "minor", "fabrication"] and tallied["summary"] == "vote 1"


def test_a_voting_seats_calibration_runs_are_its_base_runs_in_threes(data):
    _seat(data, "model-judgment", (19, 20, 18, 20, 20, 20, 17, 17, 20))
    seat = calibration.standing("model-judgment-3vote", review.prompt_version(), data=data)
    assert seat["runs"] == 3 and seat["agree"] == [19, 20, 17]  # majority of each three, per case


def test_settle_records_what_an_ended_lane_left_once(data, tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("FLEET_RUN_ID", raising=False)
    _seat(data, "model-default", (19, 19, 19))
    ended = _lane("review-m8839-1-a", review.prompt_version(), {"m8839": _review()})
    old = _lane("review-m8839-1-b", "0ldhash0", {"m8839": _review()})
    monkeypatch.setattr(review, "lane_state", lambda folder: ("r", "TERMINAL"))
    store = StoreLayout(tmp_path / "store")
    out = review.settle(data=data, store=store)
    assert out["recorded"] == ["m8839"] and out["refused"] == {}
    assert (ended / review.SETTLED).is_file() and not (old / review.SETTLED).exists()
    (ended / review.SETTLED).unlink()  # a second look records nothing again
    assert review.settle(data=data, store=store)["recorded"] == []
    assert len(ledger.read_all(data=data)) == 1


def test_the_tick_step_runs_only_when_enabled(data, tmp_path, monkeypatch):
    from menagerie.pipeline import tick_full

    assert "skipped" in tick_full.reviews(data, {}, dry_run=False)
    (tmp_path / "worklist.txt").write_text("m8839\nm1\n")
    cfg = {"reviews": {"enabled": True, "lanes": 5}, "worklist": str(tmp_path / "worklist.txt")}
    monkeypatch.setattr(tick_full, "_steps", lambda ids, data, cfg: {"m8839": "review", "m1": "done"})
    monkeypatch.setattr(tick_full, "_busy", lambda: (__import__("collections").Counter(review=1), {"k"}))
    seen = {}
    monkeypatch.setattr(review, "stage", lambda ids, **kw: seen.update(ids=ids, **kw) or {})
    tick_full.reviews(data, cfg, dry_run=True)
    assert seen["ids"] == ["m8839"] and seen["room"] == 1 and seen["held"] == {"k"}
    (tmp_path / "worklist.txt").write_text("m8839 r3\nm1 r4\n")  # a push puts r4 first
    monkeypatch.setattr(tick_full, "_steps", lambda ids, data, cfg: dict.fromkeys(ids, "review"))
    tick_full.reviews(data, {**cfg, "reviews": {**cfg["reviews"], "cohort_first": "r4"}}, dry_run=True)
    assert seen["ids"] == ["m1", "m8839"]
    with pytest.raises(ValueError, match="cohort_first"):
        tick_full.reviews(data, {**cfg, "reviews": {"enabled": True, "cohort_first": 4}}, dry_run=True)
