"""Reviewer calibration: three blinded runs per seat, scored once each, and the seat's standing
(passed, provisional, failed) under the 85 percent lowest-run bar and the flip limit. fleet-run
and the lanes are stood in for; nothing starts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, calibration, review
from menagerie.pipeline import store as store_mod


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A data root, a store, a work root, a stand-in fleet-run and a launch recorder."""

    data, store = DataLayout(tmp_path / "data"), StoreLayout(tmp_path / "store")
    monkeypatch.setenv("MENAGERIE_STORE", str(store.root))
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(store_mod, "materialize",
                        lambda sha, dest, *, store: Path(shutil.copyfile(store.blob(sha), dest)))

    class Env:
        runs: dict[str, dict] = {}
        started: list[Path] = []
        calls: list[dict] = []
        refuse: set[str] = set()  # roles or batch labels the fake admission refuses

    def launch(kind, role, brief, **kw):
        if role in Env.refuse or kw.get("label") in Env.refuse:
            raise review.drive.Refused("no_admission", "fake seat full", role=role)
        Env.started.append(brief)
        Env.calls.append({"role": role, **kw})
        run = f"run-{len(Env.started)}"
        Env.runs[run] = {"state": "RUNNING"}
        return run

    monkeypatch.setattr(review.drive, "launch", launch)
    monkeypatch.setattr(review, "_fleet_run", lambda run: Env.runs[run])
    Env.data, Env.store, Env.tmp = data, store, tmp_path
    return Env


def _put(env, name: str, text: str) -> str:
    path = env.tmp / name
    path.write_text(text)
    return store_mod.put(path, store=env.store, data=env.data, mime="text/plain",
                         license_class="derived").sha256


def _cases(env, expected: list[str]) -> list[dict]:
    """Schema 2 cases (files by store sha) plus one July schema 1 case."""

    rows = [{"schema": "review-case/2", "case": f"k{i}", "expected": label,
             "files": {"source/s1-paper.txt": _put(env, f"p{i}", f"paper {i}"),
                       "ours/recipe.json": _put(env, f"r{i}", "{}"),
                       **({"ours/member.md": _put(env, f"m{i}", "delta")} if i == 0 else {})}}
            for i, label in enumerate(expected)]
    rows.append({"schema": "review-case/1", "case": "a1-x", "expected": "not_fabrication",
                 "text": _put(env, "jt", "july paper"), "card": _put(env, "jc", "july card"),
                 "code": None})
    path = env.data.root / "review" / "calibration-cases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def _body(name: str, verdict: str, role: str = "model-default") -> dict:
    return {"schema": "review/1", "id": name, "reviewer": role, "verdict": verdict,
            "summary": "s", "components": [
                {"name": "core", "finding": "missing" if verdict == "fabrication" else "match"}]}


def _answer(env, brief: Path, verdicts: dict[str, str]) -> None:
    names = json.loads((brief.parent / "names.json").read_text())
    for case, verdict in verdicts.items():
        body = _body(names[case], verdict)
        (brief.parent / "out" / f"{names[case]}.json").write_text(json.dumps(body))


def _agy_answer(folder: Path, verdicts: dict[str, str], tag: str = "a") -> None:
    """A read-only seat's fenced answer, as agy-bg.sh leaves it beside a batch brief."""

    names = json.loads((folder / "names.json").read_text())
    (folder / f"agy-{tag}").mkdir()
    (folder / f"agy-{tag}" / "result.md").write_text("".join(
        f"```json\n{json.dumps(_body(names[c], v, review.THIRD_LAB))}\n```\n"
        for c, v in verdicts.items()))


def test_bar_is_85_percent_rounded_up():
    assert (calibration.bar(41), calibration.bar(27), calibration.bar(20)) == (35, 23, 17)


def test_three_blinded_runs_are_launched_once_and_scored_once(env):
    cases = _cases(env, ["fabrication"] + ["not_fabrication"] * 5)
    first = calibration.calibrate(role="model-default", data=env.data)
    assert first["state"] == "launched" and len(env.started) == 3
    folders = [b.parent for b in env.started]
    assert len({f.name for f in folders}) == 3 and all(f.name.endswith(f"-r{i + 1}")
                                                      for i, f in enumerate(folders))
    names = [json.loads((f / "names.json").read_text()) for f in folders]
    k0 = folders[0] / "cases" / names[0]["k0"]
    assert (k0 / "ours" / "member.md").read_text() == "delta"
    assert (k0 / "source" / "s1-paper.txt").read_text() == "paper 0"
    assert (folders[0] / "cases" / names[0]["a1-x"] / "ours" / "card.md").read_text() == "july card"
    prompt = review.prompt_version()
    assert json.loads((folders[0] / "lane.json").read_text())["prompt"] == prompt
    assert "k0" not in env.started[0].read_text()  # the brief names only opaque folders
    assert calibration.calibrate(role="model-default", data=env.data)["state"] == "running"
    assert len(env.started) == 3

    right = {"k0": "fabrication", "a1-x": "minor"} | {f"k{i}": "faithful" for i in range(1, 6)}
    for brief in env.started:
        _answer(env, brief, right)
    for run in env.runs.values():
        run["state"] = "TERMINAL"
    seat = calibration.calibrate(role="model-default", data=env.data)
    assert (seat["state"], seat["agree"], seat["lowest"], seat["bar"]) == (
        "passed", [7, 7, 7], 7, 6)
    calibration.calibrate(role="model-default", data=env.data)
    rows = calibration.rows(env.data)
    assert len(rows) == 21 and {r["set"] for r in rows} == {calibration.set_id(cases)}
    assert calibration.counts("model-default", prompt, data=env.data)
    assert not calibration.counts("model-default", "0ther000", data=env.data)


def _scored(env, role: str, runs: list[dict[str, str]]) -> dict:
    """Launch, answer and score three runs of ``role`` with the given verdicts."""

    calibration.calibrate(role=role, data=env.data)
    briefs = [b for b in env.started if role in b.parent.name][-3:]
    for brief, verdicts in zip(briefs, runs):
        _answer(env, brief, verdicts)
    for run in env.runs.values():
        run["state"] = "TERMINAL"
    return calibration.calibrate(role=role, data=env.data)


def test_flips_and_the_lowest_run_decide_the_standing(env):
    _cases(env, ["fabrication"] * 5 + ["not_fabrication"] * 15)  # 21 cases: bar 18
    good = {f"k{i}": "fabrication" if i < 5 else "minor" for i in range(20)} | {"a1-x": "minor"}

    def wrong(n, start=0):
        return good | {f"k{i}": "minor" if i < 5 else "fabrication"
                       for i in range(start, start + n)}

    seat = _scored(env, "model-default", [good, wrong(3), good])
    assert (seat["state"], seat["lowest"], len(seat["flips"])) == ("provisional", 18, 3)
    seat = _scored(env, "model-design-review", [wrong(2), wrong(2, 2), wrong(2, 4)])
    assert (seat["state"], seat["lowest"], len(seat["flips"])) == ("failed", 19, 6)


def test_a_wrong_call_disagrees_and_a_relabel_is_a_new_set(env):
    cases = _cases(env, ["fabrication", "not_fabrication"])
    calls = {"k0": "fabrication", "k1": "fabrication", "a1-x": "fabrication"}
    seat = _scored(env, "model-default", [calls] * 3)
    assert seat["state"] == "failed" and seat["agree"] == [1, 1, 1]
    old = seat["set"]
    cases[1]["expected"] = "fabrication"
    (env.data.root / "review" / "calibration-cases.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in cases))
    fresh = calibration.standing("model-default", review.prompt_version(), data=env.data)
    assert fresh["set"] != old and fresh["state"] == "uncalibrated" and fresh["runs"] == 0
    launched = len(env.started)
    rescored = calibration.calibrate(role="model-default", data=env.data)  # the old runs, rescored
    assert len(env.started) == launched and rescored["set"] == fresh["set"]
    assert rescored["agree"] == [2, 2, 2] and rescored["runs"] == 3


def test_third_lab_runs_in_read_only_batches_through_the_shared_launcher(env):
    _cases(env, ["fabrication"] + ["not_fabrication"] * 5)  # 7 cases: batches of 5 and 2
    assert calibration.calibrate(role=review.THIRD_LAB, data=env.data)["state"] == "launched"
    assert len(env.started) == 6 and {b.name for b in env.started} == {"BRIEF-b1.md",
                                                                       "BRIEF-b2.md"}
    first = env.calls[0]
    assert (first["lab"], first["answer"], len(first["ids"]), first["keys"]) == (
        "antigravity", "blocks", 5, [])
    right = {"k0": "fabrication", "a1-x": "minor"} | {f"k{i}": "faithful" for i in range(1, 6)}
    for folder in {b.parent for b in env.started}:
        _agy_answer(folder, right)
    for run in env.runs.values():
        run["state"] = "TERMINAL"
    seat = calibration.calibrate(role=review.THIRD_LAB, data=env.data)
    assert (seat["state"], seat["agree"]) == ("passed", [7, 7, 7])


def test_a_refused_later_batch_waits_and_starts_on_a_later_call(env):
    _cases(env, ["fabrication"] * 6)
    prompt = review.prompt_version()
    env.refuse |= {f"review-calibrate-{review.THIRD_LAB}-{prompt}-r{r}-b2" for r in (1, 2, 3)}
    assert calibration.calibrate(role=review.THIRD_LAB, data=env.data)["state"] == "launched"
    assert len(env.started) == 3  # each run's first batch
    for run in env.runs.values():
        run["state"] = "TERMINAL"
    waiting = calibration.calibrate(role=review.THIRD_LAB, data=env.data)
    assert waiting["state"] == "running" and calibration.rows(env.data) == []
    assert {r["state"] for r in waiting["run_reports"]} == {"waiting"}
    env.refuse.clear()
    assert calibration.calibrate(role=review.THIRD_LAB, data=env.data)["state"] == "launched"
    assert len(env.started) == 6
    right = {f"k{i}": "fabrication" for i in range(6)} | {"a1-x": "minor"}
    for folder in {b.parent for b in env.started}:
        _agy_answer(folder, right)
    for run in env.runs.values():
        run["state"] = "TERMINAL"
    seat = calibration.calibrate(role=review.THIRD_LAB, data=env.data)
    assert (seat["state"], seat["agree"]) == ("passed", [7, 7, 7])


def test_calibrate_all_runs_every_seat_on_its_lab_and_reports_a_refusal(env):
    _cases(env, ["fabrication", "not_fabrication"])
    env.refuse.add("model-design-review")
    out = review.calibrate_all(data=env.data, runs=3)
    seats = out["seats"]
    assert seats["model-design-review"]["state"] == "refused"
    assert seats["model-design-review"]["why"] == "fake seat full"
    assert seats["model-default"]["state"] == seats[review.THIRD_LAB]["state"] == "launched"
    assert {c["lab"] for c in env.calls if c["role"] == "model-default"} == {"claude"}
    assert out["prompt"] == review.prompt_version() and out["non_codex"] == []
    right = {"k0": "fabrication", "k1": "minor", "a1-x": "minor"}
    for folder in {b.parent for b in env.started}:
        if review.THIRD_LAB in folder.name:
            _agy_answer(folder, right)
        else:
            _answer(env, folder / "BRIEF.md", right)
    for run in env.runs.values():
        run["state"] = "TERMINAL"
    out = review.calibrate_all(data=env.data, runs=3)
    assert out["non_codex"] == ["model-default", review.THIRD_LAB]
    assert out["seats"]["model-default"]["state"] == "provisional"  # 3 of 3 is the bar


def test_committed_set_v2_has_41_counted_cases_with_labels_and_evidence():
    data = DataLayout(Path(__file__).parents[2] / "menagerie" / "data")
    counted = calibration.cases(data)
    names = [c["case"] for c in counted]
    assert len(counted) == 41 and len(set(names)) == 41
    assert {c["expected"] for c in counted} == {"fabrication", "not_fabrication"}
    new = [c for c in counted if c.get("schema") == "review-case/2"]
    assert len(new) == 14 and all(c["evidence"] and c["files"] for c in new)
    stated = {c["case"] for c in new if "ours/member.md" in c["files"]}
    for case in new:
        if case["origin"] == "constructed":  # each pair: the stated half passes, the bare fails
            assert (case["case"] in stated) == (case["expected"] == "not_fabrication")
    assert calibration.bar(len(counted)) == 35


def test_a_run_whose_lane_wrote_nothing_is_run_again_not_scored(env):
    _cases(env, ["fabrication"])
    calibration.calibrate(role="model-default", data=env.data)
    for run in env.runs.values():
        run["state"] = "TERMINAL"
    again = calibration.calibrate(role="model-default", data=env.data)
    assert again["state"] == "launched" and len(env.started) == 6
    assert calibration.rows(env.data) == []
    kept = sorted(p.name for p in env.started[0].parent.parent.glob("*.failed-1"))
    assert len(kept) == 3 and all(".failed-1" in k for k in kept)


def test_a_run_with_gaps_is_retried_twice_then_scored(env):
    _cases(env, ["fabrication", "not_fabrication"])
    calibration.calibrate(role="model-default", data=env.data)
    for attempt in range(3):
        for brief in env.started[-3:]:
            _answer(env, brief, {"k0": "fabrication"})  # k1 and a1-x unanswered
        for run in env.runs.values():
            run["state"] = "TERMINAL"
        seat = calibration.calibrate(role="model-default", data=env.data)
        assert seat["state"] == ("launched" if attempt < 2 else "failed")
    assert len(env.started) == 9 and seat["agree"] == [1, 1, 1]


def test_a_run_whose_start_was_refused_starts_again_over_its_read_only_cases(env):
    _cases(env, ["fabrication", "not_fabrication"])
    env.refuse = {review.THIRD_LAB}
    with pytest.raises(review.drive.Refused):
        calibration.calibrate(role=review.THIRD_LAB, data=env.data, runs=1)
    folder = next((review.work_root() / "calibration").glob(f"{review.THIRD_LAB}-*-r1"))
    assert not (folder / "lane.json").exists()
    for f in (folder / "cases").rglob("*"):
        if f.is_file():
            f.chmod(0o444)  # as a materialized snapshot leaves them
    env.refuse = set()
    report = calibration.calibrate(role=review.THIRD_LAB, data=env.data, runs=1)
    assert report["state"] == "launched" and (folder / "lane.json").is_file()
