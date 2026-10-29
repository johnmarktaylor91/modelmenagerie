"""``axon_feed``: the cluster feeder's choices and its pass, on stand-ins for the cluster and worker."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.pipeline import DataLayout, axon_feed, written
from menagerie.pipeline.axon_feed import Feed, Row


def trace(rev: str = "r1", outcome: str = "fail", capture: str | None = "full", tl: str = "2.36.0",
          host: str = "worker", cls: str | None = None) -> Any:
    return SimpleNamespace(recipe_rev=rev, outcome=outcome, capture=capture, tl=tl, host=host,
                           cls=cls)


def test_a_full_pass_at_the_pin_or_a_typed_cluster_failure_settles_a_row() -> None:
    assert axon_feed._settled([trace(outcome="pass")], "r1", "2.36.0")
    assert not axon_feed._settled([trace(outcome="pass", tl="2.35.3")], "r1", "2.36.0")
    assert not axon_feed._settled([trace(outcome="pass", rev="r0")], "r1", "2.36.0")
    assert axon_feed._settled([trace(host="cluster", cls="torchlens_gap:replay")], "r1", "2.36.0")
    # a memory_cap on the cluster raises the need into the next band instead
    assert not axon_feed._settled([trace(host="cluster", cls="memory_cap")], "r1", "2.36.0")
    assert not axon_feed._settled([trace(cls="oversized:projected")], "r1", "2.36.0")
    assert axon_feed._settled([trace(host="cluster", cls=axon_feed.OVER_CLASS)], "r1", "2.36.0")


def test_a_row_past_the_largest_allocation_gets_one_typed_cluster_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    from menagerie.pipeline import ledger, read_json, write_json
    from menagerie.pipeline.publish import torchlens_pin
    from menagerie.pipeline.records import ModelRecord

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    data = DataLayout(tmp_path / "data")
    base = read_json(ModelRecord, Path(__file__).with_name("fixtures") / "m8839.model.json")
    write_json(replace(base, id="m9201"), data.model("m9201"))
    axon_feed._type_over([row("m9201", 150.4)], data, axon_feed.config({}))
    made = [r for r in ledger.read_all(data=data) if r.id == "m9201"]
    assert [(r.stage, r.host, r.outcome, r.cls, r.tl) for r in made] == \
        [("trace", "cluster", "fail", axon_feed.OVER_CLASS, torchlens_pin())]
    assert "150.4 GiB" in made[0].msg
    assert axon_feed._settled(made, base.recipe.rev, torchlens_pin())  # type: ignore[union-attr]


def test_rows_with_no_trace_come_first_then_structure_only_then_re_traces() -> None:
    assert axon_feed._tier([trace(cls="oversized:projected")]) == 0
    assert axon_feed._tier([trace(outcome="pass", capture="structure_only")]) == 1
    assert axon_feed._tier([trace(outcome="pass", tl="2.35.3")]) == 2


def row(sid: str, need: float, tier: int = 0, named: bool = True, env: str = "torch-main") -> Row:
    return Row(sid, env, need, (tier, not named, need, sid))


def test_the_cpu_band_goes_first_in_one_environment_then_the_reviewed_band() -> None:
    cfg = axon_feed.config({"axon": {"rows": {"cpu": 2}}})
    bands = {"cpu": [row("m1", 45), row("m2", 55, env="mmlab"), row("m3", 50)],
             "reviewed": [row("m4", 90), row("m5", 70)], "over": [row("m6", 150)]}
    assert axon_feed.choose(bands, cfg) == ("cpu", ["m1", "m3"])
    assert axon_feed.choose({**bands, "cpu": []}, cfg) == ("reviewed", ["m4", "m5"])
    assert axon_feed.choose({"cpu": [], "reviewed": [], "over": bands["over"]}, cfg) is None


def test_a_batch_packs_rows_in_order_to_its_row_count_and_time_budget() -> None:
    cfg = axon_feed.config({})  # reviewed: 4 rows, 20 minutes each, a 100-minute budget
    many = [row(f"m{i}", 70) for i in range(6)]
    assert axon_feed._pack(many, "reviewed", cfg) == ["m0", "m1", "m2", "m3"]
    slow = replace(many[1], minutes=60.0)
    # the slow row fits after one; the next rows fill what is left, in order
    assert axon_feed._pack([many[0], slow, *many[2:]], "reviewed", cfg) == ["m0", "m1", "m2"]
    # a slow row first still goes, and a second slow row waits for a later batch
    assert axon_feed._pack([slow, replace(many[2], minutes=60.0), many[3]], "reviewed",
                           cfg) == ["m1", "m3"]
    assert axon_feed._pack(many, "cpu", cfg) == [r.id for r in many]  # 6 x 12 = 72 minutes
    assert axon_feed._slow([trace(cls="timeout")]) and \
        not axon_feed._slow([trace(cls="timeout", capture="structure_only")])


def test_rows_band_by_need_and_order_named_and_untraced_first(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import publish, stage3

    data = DataLayout(tmp_path / "data")
    zoos = {"m1": "timm", "m2": "discovered-pytorch", "m3": "timm", "m4": "timm",
            "m5": "timm", "m6": "timm"}
    needs = {"m1": 48.0, "m2": 44.0, "m3": 90.0, "m4": 130.0, "m5": 30.0, "m6": 41.0}
    for sid, zoo in zoos.items():
        path = data.model(sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        records = {sid: SimpleNamespace(id=sid, zoo=zoo, family=f"f-{sid}", give_up=None,
                                        recipe=SimpleNamespace(env="torch-main", rev="r1",
                                                               placement=None))}
        monkeypatch.setitem(_RECORDS, path.name, records[sid])
    monkeypatch.setattr(axon_feed, "read_json", lambda cls, path: _RECORDS[Path(path).name])
    monkeypatch.setattr(stage3.Peaks, "read", classmethod(lambda cls, data: None))
    monkeypatch.setattr(stage3, "_families", lambda data: {})
    monkeypatch.setattr(stage3, "projected_peak", lambda r, peaks, fam: needs[r.id])
    monkeypatch.setattr(publish, "torchlens_pin", lambda: "2.36.0")
    monkeypatch.setattr(axon_feed, "_trace_rows", lambda data: {
        "m6": [trace(outcome="pass", capture="structure_only")]})
    bands = axon_feed.rows(data, axon_feed.config({}), back=["m6"])
    # m6 came back from an unfinished batch: first; then named m1 before anonymous m2
    assert [r.id for r in bands["cpu"]] == ["m6", "m1", "m2"]
    assert [r.id for r in bands["reviewed"]] == ["m3"] and [r.id for r in bands["over"]] == ["m4"]


_RECORDS: dict[str, Any] = {}


def test_job_ids_come_from_the_answer_or_its_last_line() -> None:
    assert axon_feed.job_id("6411088") == "6411088"
    assert axon_feed.job_id("# reviewed allocation giants-128g (128 GiB)\n6411100") == "6411100"
    assert axon_feed.job_id("refused") is None


class FakeOps:
    """The cluster queue, batches, collection and the worker's render job, in memory."""

    def __init__(self, root: Path) -> None:
        self.root, self.jobs, self.next_job = root, [], 7000001
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.render_jobs: dict[str, str] = {}
        self.lines: list[dict[str, str]] = []
        self.tidied: list[str] = []

    def queue(self) -> list[tuple[str, str, str]]:
        return list(self.jobs)

    def times(self, job: str) -> dict[str, str]:
        return {"submit": "s", "start": "a", "end": "e", "state": "COMPLETED"}

    def batch(self, kind: str, ids: list[str], data: DataLayout) -> Path:
        batch = self.root / f"{kind}-{len(list(self.root.glob('*')))}"
        batch.mkdir(parents=True)
        meta = {"batch": batch.name, "kind": kind, "ids": ids, "root": "/r", "commit": "c",
                "revisions": {i: "r1" for i in ids}}
        (batch / "batch.json").write_text(json.dumps(meta))
        return batch

    def submit(self, batch: Path) -> str:
        job, self.next_job = str(self.next_job), self.next_job + 1
        self.jobs.append((job, f"menagerie-{batch.name}", "PENDING"))
        return job

    def collect(self, batch: Path, data: DataLayout) -> dict[str, Any]:
        return {"done": "rc=0", "rows": self.tables.get(batch.name, []), "added": {}, "blobs": 0}

    def tidy(self, batch: Path, meta: dict[str, Any]) -> None:
        self.tidied.append(batch.name)

    def render_submit(self, batch: Path, ids: list[str], cfg: dict[str, Any]) -> str:
        self.render_jobs[f"render-{batch.name}"] = "RUNNING"
        return f"render-{batch.name}"

    def render_state(self, job: str) -> str:
        return self.render_jobs[job]

    def render_lines(self, job: str, repo: Path) -> list[dict[str, str]]:
        return self.lines

    def render_tidy(self, batch: str, job: str, repo: Path) -> None:
        self.tidied.append(f"render {batch} {job}")


@pytest.fixture
def feed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Feed:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    ops = FakeOps(tmp_path / "work" / "axon")
    bands = {"cpu": [row("m1", 45), row("m2", 50), row("m3", 55)], "reviewed": [row("m9", 80)],
             "over": [row("m10", 150)]}
    monkeypatch.setattr(axon_feed, "rows", lambda data, cfg, back=(): {
        k: sorted(v, key=lambda r: (r.id not in set(back), r.key)) for k, v in bands.items()})
    typed: list[str] = []
    monkeypatch.setattr(axon_feed, "_type_unfinished", lambda ids, data, batch: typed.extend(ids))
    over: list[str] = []
    monkeypatch.setattr(axon_feed, "_type_over",
                        lambda rows, data, cfg: over.extend(r.id for r in rows))
    cfg = axon_feed.config({"axon": {"enabled": True, "rows": {"cpu": 3}}})
    made = Feed({}, DataLayout(tmp_path / "data"), cfg, ops, tmp_path, submit=True,
                clock=lambda: "2026-10-28T12:00:00Z")
    made.typed = typed  # type: ignore[attr-defined]
    made.over = over  # type: ignore[attr-defined]
    return made


def test_one_batch_in_flight_collected_rendered_and_unfinished_rows_back_once(feed: Feed) -> None:
    ops: FakeOps = feed.ops
    first = axon_feed.step(feed)
    assert first["submitted"]["kind"] == "cpu" and first["submitted"]["rows"] == 3
    assert feed.state["waiting_over"] == ["m10"] and first["submitted"]["typed_over"] == ["m10"]
    assert feed.over == ["m10"]  # type: ignore[attr-defined]
    batch = feed.state["active"]["batch"]
    # while its job is queued, nothing else is submitted
    assert "submitted" not in axon_feed.step(feed) and len(ops.jobs) == 1
    ops.jobs.clear()
    ops.tables[batch] = [{"stable_id": "m1", "forward": "pass", "trace": "trace:pass;trace:pass",
                          "trace_class": ";"},
                         {"stable_id": "m2", "forward": "pass", "trace": "trace:fail",
                          "trace_class": "memory_cap"},
                         {"stable_id": "m3", "trace": "not_reached"}]
    second = axon_feed.step(feed)
    assert second["collected"]["passed"] == ["m1"]
    assert second["collected"]["unfinished"] == {"back": ["m3"], "typed": []}
    assert feed.state["renders"] == [{"batch": batch, "ids": ["m1"], "job": f"render-{batch}"}]
    assert ops.tidied == [batch] and feed.state["history"][0]["passed"] == 1
    # the same pass submits the next batch, the returned row first
    assert feed.state["active"]["ids"][0] == "m3"
    assert ops.render_jobs  # its render job started
    ops.jobs.clear()
    ops.tables[feed.state["active"]["batch"]] = [{"stable_id": "m1", "trace": "trace:pass"}]
    third = axon_feed.step(feed)
    assert third["collected"]["unfinished"]["typed"] == ["m3"] and feed.typed == ["m3"]  # type: ignore[attr-defined]


def test_a_finished_render_job_appends_its_lines_once(feed: Feed) -> None:
    ops: FakeOps = feed.ops
    feed.state["renders"] = [{"batch": "cpu-0", "ids": ["m1"], "job": "render-cpu-0"}]
    ops.render_jobs["render-cpu-0"] = "RUNNING"
    feed.cfg["enabled"] = False
    assert axon_feed.step(feed)["renders"][0]["state"] == "RUNNING"
    ops.render_jobs["render-cpu-0"] = "TERMINAL:succeeded"
    ops.lines = [{"path": "ledger/2026-10.jsonl", "line": '{"id":"m1","stage":"render"}'},
                 {"path": "store-index/2026-10.jsonl", "line": '{"sha256":"aa"}'}]
    (feed.data.root / "ledger").mkdir(parents=True)
    (feed.data.root / "ledger" / "2026-10.jsonl").write_text('{"id":"m1","stage":"render"}\n')
    with written.recording() as rec:
        out = axon_feed.step(feed)
    assert out["renders"][0]["added"] == 1 and feed.state["renders"] == []
    index = feed.data.root / "store-index" / "2026-10.jsonl"
    assert index.read_text() == '{"sha256":"aa"}\n'
    assert list(rec.lines.values()) == [['{"sha256":"aa"}\n']]
    assert "render cpu-0 render-cpu-0" in ops.tidied


def test_render_lines_for_other_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        axon_feed.append_lines([{"path": "models/m1.json", "line": "{}"}], DataLayout(tmp_path))


def test_another_menagerie_task_in_the_queue_holds_the_slot(feed: Feed) -> None:
    feed.ops.jobs.append(("6400000", "menagerie-cpu-manual", "RUNNING"))
    out = axon_feed.step(feed)
    assert "submitted" not in out and "menagerie-cpu-manual" in out["waiting"]


def test_a_queued_feeder_batch_the_state_lost_is_adopted(feed: Feed) -> None:
    batch = feed.ops.batch("cpu", ["m1"], feed.data)
    (batch / "feeder").write_text("x")
    feed.ops.jobs.append(("6400001", f"menagerie-{batch.name}", "PENDING"))
    assert axon_feed.step(feed)["waiting"].startswith("adopted")
    assert feed.state["active"]["job"] == "6400001"


def test_without_submit_only_collection_runs(feed: Feed) -> None:
    feed.submit, feed.held = False, "disk paused"
    out = axon_feed.step(feed)
    assert "submitted" not in out and feed.state.get("active") is None
    assert out["held"] == "disk paused"


def test_a_failed_render_job_is_started_again_then_given_up(feed: Feed) -> None:
    ops: FakeOps = feed.ops
    feed.cfg["enabled"] = False
    feed.state["renders"] = [{"batch": "cpu-0", "ids": ["m1"], "job": "render-cpu-0"}]
    for attempt in range(axon_feed.RENDER_TRIES):
        ops.render_jobs["render-cpu-0"] = "TERMINAL:admission_timeout"
        out = axon_feed.step(feed)["renders"][0]
        if attempt < axon_feed.RENDER_TRIES - 1:
            assert "given_up" not in out and axon_feed.step(feed)["renders"][0]["render_job"]
    assert "given_up" in out and feed.state["renders"] == []
