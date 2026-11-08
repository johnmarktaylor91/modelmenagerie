"""The TorchLens failure corpus: attribution on evidence, clustering, new-cluster FACTs and the
re-trace of corpus rows at a new TorchLens pin."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import DataLayout, retrace, tick, tl_corpus, tl_corpus_step, worker
from menagerie.pipeline import tl_attribution as attrib
from menagerie.pipeline.records import LedgerRow, ModelRecord, from_dict

REV, OLD_REV = "a" * 12, "b" * 12
T0 = datetime(2026, 11, 5, 8, 0, tzinfo=timezone.utc)
CLASSES = "class ConfigurationError(TorchLensError):\nclass TorchLensError(Exception):\n" \
          "class Helper(object):\nclass MetadataInvariantError(TorchLensError):\n"
SOURCE = {  # one TorchLens release's grep-able lines: path:line:text
    "torchlens/receptive_field/_types.py:158":
        '            raise ValueError("input_axis must be non-negative and input_extent must be '
        'positive.")',
    "torchlens/notes.py:3": "# you can only change requires_grad flags of leaf variables",
    "torchlens/debug/classify.py:22": '    r"|got multiple values for argument|takes N arguments")',
}


def fake_git(tags: tuple[str, ...] = ("v2.36.0",)) -> Any:
    calls: list[tuple[str, ...]] = []

    def git(*args: str) -> str | None:
        calls.append(args)
        if args[0] == "rev-parse":
            return "sha\n" if args[-1].split("^")[0] in tags else ""
        if "-h" in args:
            return CLASSES
        wanted = [args[i + 1] for i, a in enumerate(args) if a == "-e"]
        return "".join(f"{args[-3]}:{site.replace(':', ':', 1)}:{text}\n"
                       for site, text in SOURCE.items() if any(w in text for w in wanted))

    git.calls = calls  # type: ignore[attr-defined]
    return git


def index(tmp_path: Path, **kw: Any) -> attrib.SourceIndex:
    return attrib.SourceIndex(tmp_path / "src", tmp_path / "cache", git=fake_git(**kw))


def model(sid: str, *, rev: str = REV, give_up: dict[str, Any] | None = None) -> ModelRecord:
    raw = {"schema": "model/1", "id": sid, "name": f"net-{sid}", "zoo": "timm",
           "family": f"f-{sid}", "source_case": "vendor", "variant": "", "delta": {},
           "recipe": {"rev": rev, "kind": "adapter", "env": "torch-main", "backend": "torch",
                      "input": {"preset": "image", "kind": "random-fallback"},
                      "adapter": f"adapters/f-{sid}.py"}}
    if give_up:
        raw["give_up"] = give_up
    return from_dict(ModelRecord, raw)


def trace(sid: str, outcome: str = "fail", cls: str | None = "capture_error", msg: str = "",
          *, rev: str = REV, at: str = "2026-11-01T00:00:00Z", tl: str = "2.36.0",
          **extra: Any) -> LedgerRow:
    return LedgerRow(id=sid, stage="trace", at=at, host="worker", env="torch-main",
                     recipe_rev=rev, outcome=outcome, mode="eval", capture="full", tl=tl,
                     cls=cls if outcome != "pass" else None, msg=msg or None, **extra)


# --- attribution ---------------------------------------------------------------------------

def test_attribution_needs_evidence_never_wording(tmp_path: Path) -> None:
    idx = index(tmp_path)
    tl = attrib.TORCHLENS

    def who(cls: str, msg: str, pin: str = "2.36.0", frames: Any = None) -> tuple[str, str]:
        found = attrib.attribute(cls, msg, pin, frames, idx)
        return found.who, found.kind

    assert who("torchlens_gap:replay", "perturbation mismatch") == (tl, "validation_status")
    assert who("torchlens_unverified", "unverified (x)") == (tl, "validation_status")
    assert who("capture_error", "validation refused before capture: precondition_refusal "
               "mismatch -- cannot prove") == (tl, "validation_status")
    site = attrib.attribute("capture_error", "ValueError: input_axis must be non-negative and "
                            "input_extent must be positive.", "2.36.0", None, idx)
    assert (site.who, site.kind) == (tl, "raise_site")
    assert "torchlens/receptive_field/_types.py:158" in site.evidence
    # TorchLens's own exception class, inherited through another of its classes
    assert who("capture_error", "ConfigurationError: Cannot run log_backward") == (
        tl, "exception_type")
    # a comment holding torch's message is not a raise site, and wording alone never decides
    assert who("capture_error", "RuntimeError: you can only change requires_grad flags of leaf "
               "variables.") == (attrib.UNATTRIBUTED, "none")
    # torch's message copied into a TorchLens pattern table, with no raise near it
    assert who("capture_error", "TypeError: forward() got multiple values for argument 'x'") == (
        attrib.UNATTRIBUTED, "none")
    assert who("capture_error", "RuntimeError: TorchLens failed badly in torchlens") == (
        attrib.UNATTRIBUTED, "none")
    assert who("capture_error:backend", "gpu mode runs torch recipes only") == (
        attrib.MENAGERIE, "class")
    # a release with no tag in the checkout has no index: nothing is guessed
    assert who("capture_error", "ConfigurationError: x", pin="2.30.0") == (
        attrib.UNATTRIBUTED, "none")


def test_frames_decide_before_source_evidence(tmp_path: Path) -> None:
    idx = index(tmp_path)
    inner_tl = ("model/net.py:10 forward", "torchlens/capture/wrap.py:88 wrapped")
    found = attrib.attribute("capture_error", "RuntimeError: x", "2.36.0", inner_tl, idx)
    assert (found.who, found.kind) == (attrib.TORCHLENS, "frame")
    assert "torchlens/capture/wrap.py:88" in found.evidence
    worker_frame = ("menagerie/pipeline/worker.py:900 _capture",)
    assert attrib.attribute("capture_error", "RuntimeError: x", "2.36.0", worker_frame,
                            idx).who == attrib.MENAGERIE
    # a torch frame under TorchLens's wrapper names both sides but proves neither
    in_torch = ("torchlens/capture/wrap.py:88 wrapped", "torch/nn/functional.py:5 conv2d")
    found = attrib.attribute("capture_error", "RuntimeError: x", "2.36.0", in_torch, idx)
    assert found.who == attrib.UNATTRIBUTED and found.kind == "frame"
    assert "called from torchlens/capture/wrap.py:88" in found.evidence
    # source evidence still attributes a row whose frames are not decisive
    found = attrib.attribute("capture_error", "ConfigurationError: y", "2.36.0", in_torch, idx)
    assert (found.who, found.kind) == (attrib.TORCHLENS, "exception_type")


def test_the_index_is_cached_per_release(tmp_path: Path) -> None:
    first = index(tmp_path)
    msg = "ValueError: input_axis must be non-negative and input_extent must be positive."
    assert first.raise_site("2.36.0", msg)
    assert first.types("2.36.0") == {"ConfigurationError", "TorchLensError",
                                     "MetadataInvariantError"}
    first.save()
    again = index(tmp_path)
    assert again.raise_site("2.36.0", msg) == first.raise_site("2.36.0", msg)
    assert again.types("2.36.0") == first.types("2.36.0")
    assert all(c[0] == "rev-parse" for c in again.git.calls)  # type: ignore[attr-defined]


def test_a_lookup_past_its_deadline_stops_and_keeps_what_it_found(tmp_path: Path) -> None:
    idx = attrib.SourceIndex(tmp_path, tmp_path / "cache", git=fake_git(),
                             deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        idx.raise_site("2.36.0", "ValueError: some long message that nobody has looked up yet")


def test_pieces_are_the_literal_text_between_formatted_values() -> None:
    msg = ("MetadataInvariantError: [graph_connectivity] Layer 'to_1_1' is marked "
           "is_internal_source but consumed unattributed tensor data (args=('arg0',)); a tensor")
    found = attrib.pieces(msg)
    assert "is marked is_internal_source but consumed" in found
    assert "but consumed unattributed tensor data" in found
    assert attrib.pieces("RuntimeError: short") == []


# --- signatures and clusters -----------------------------------------------------------------

@pytest.mark.parametrize(("cls", "msg", "sig"), [
    ("torchlens_gap:replay", ("single-thread retry: perturbation mismatch at sdpa_1_40:1 "
                              "(scaled_dot_product_attention) -- output insensitive"),
     "replay perturbation at scaled_dot_product_attention"),
    ("torchlens_gap:replay", ("single-thread retry: bfs_completeness mismatch -- Validation "
                              "dispatcher op count does not match captured operation count: "
                              "1222 dispatched"),
     "replay bfs_completeness: Validation dispatcher op count does not match captured …"),
    ("torchlens_gap:replay", ("bfs_completeness mismatch -- BFS reached 5/9 ops; 4 unreached "
                              "(e.g. ['slogdet_3_4:1'])"),
     "replay bfs_completeness: unreached slogdet"),
    ("torchlens_unverified", "unverified (region_not_per_op_replayable): single-thread retry",
     "unverified region_not_per_op_replayable"),
    ("capture_error", "AttributeError: 'dict' object has no attribute 'detach'",
     "AttributeError: 'dict' object has no attribute 'detach'"),
    ("capture_error", ("TypeError: build.<locals>.Net.forward() got multiple values for "
                       "argument 'x'"), "TypeError: forward() got multiple values for argument '…'"),
])
def test_signatures(cls: str, msg: str, sig: str) -> None:
    assert tl_corpus.signature(cls, msg) == sig


def test_one_bug_on_different_layers_is_one_cluster() -> None:
    a = tl_corpus.signature("capture_error", "MetadataInvariantError: [graph_connectivity] Layer "
                            "'to_1_1' is marked is_internal_source but consumed")
    b = tl_corpus.signature("capture_error", "MetadataInvariantError: [graph_connectivity] Layer "
                            "'view_12_40' is marked is_internal_source but consumed")
    assert a == b and a.startswith("MetadataInvariantError [graph_connectivity]")
    refusal = ("validation refused before capture: precondition_refusal mismatch -- TorchLens "
               "validation cannot prove model-state restoration because these plain attributes "
               "are unsupported: ('{}',)")
    assert tl_corpus.signature("capture_error", refusal.format("RNG")) == tl_corpus.signature(
        "capture_error", refusal.format("Gate"))


def corpus_world(tmp_path: Path) -> tuple[list[Any], dict[str, ModelRecord]]:
    replay = ("single-thread retry: perturbation mismatch at x_1_2:1 (einsum) -- output "
              "insensitive")
    records = {sid: model(sid) for sid in ("m1", "m2", "m3", "m4", "m5", "m6", "m7")}
    records["m8"] = model("m8", give_up={"class": "torchlens_gap:foreign_kernel",
                                         "reason": "DGL kernels", "at": "2026-10-27T00:00:00Z"})
    records["m9"] = model("m9", give_up={"class": "torchlens_gap:foreign_kernel",
                                         "reason": "DGL kernels", "at": "2026-10-27T00:00:00Z"})
    rows = [trace("m1", cls="torchlens_gap:replay", msg=replay, tl="2.35.3"),
            trace("m2", cls="torchlens_gap:replay", msg=replay),
            trace("m3", msg="ConfigurationError: Cannot run log_backward"),
            trace("m4", outcome="pass"),
            trace("m4", msg="ConfigurationError: old", at="2026-10-01T00:00:00Z"),
            trace("m5", cls="memory_cap", msg="MemoryError"),
            trace("m6", msg="RuntimeError: element 0 of tensors does not require grad"),
            trace("m7", msg="ConfigurationError: Cannot run log_backward", rev=OLD_REV),
            trace("m9", outcome="pass", at="2026-11-02T00:00:00Z")]
    return rows, records


def test_collect_takes_the_latest_current_recipe_trace_and_tl_give_ups(tmp_path: Path) -> None:
    rows, records = corpus_world(tmp_path)
    corpus, other, fixed = tl_corpus.collect(
        rows, records, cohorts={"m1": "r3", "m2": "r4-github"},
        jobs={("m2", REV): "m2-aaaaaaaaaaaa-20261101t000000z-0001"}, index=index(tmp_path),
        inactive={"m6"})
    by = {r["id"]: r for r in corpus}
    assert sorted(by) == ["m1", "m2", "m3", "m8"]  # m4 passed, m5 memory, m6 inactive, m7 old rev
    assert other == {"pass": 1, "memory_cap": 1} and fixed == ["m9"]
    assert by["m1"]["cohort"] == "r3" and by["m3"]["cohort"] == tl_corpus.EARLIER
    assert by["m2"]["job"].startswith("m2-") and by["m2"]["reproduce"].endswith("trace m2")
    assert by["m8"]["signature"] == "giveup torchlens_gap:foreign_kernel"
    assert by["m3"]["evidence_kind"] == "exception_type"
    assert by["m1"]["recipe"] == {"rev": REV, "kind": "adapter", "env": "torch-main",
                                  "adapter": "adapters/f-m1.py"}
    groups = tl_corpus.clusters(corpus)
    assert [g["count"] for g in groups] == [2, 1, 1]
    assert groups[0]["representatives"] == ["m2", "m1"]  # the newest pin first
    assert groups[0]["pins"] == {"2.35.3": 1, "2.36.0": 1}


def test_build_writes_latest_and_dated_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                             ) -> None:
    rows, records = corpus_world(tmp_path)
    data = DataLayout(tmp_path / "data")
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr("menagerie.pipeline.ledger.read_all", lambda data: rows)
    monkeypatch.setattr(tl_corpus, "_records", lambda data: records)
    monkeypatch.setattr(tl_corpus, "_inactive", lambda data: set())
    out, worklist = tmp_path / "out", tmp_path / "worklist.txt"
    worklist.write_text("# cohorts\nm1 r3\nm2\n")
    built = tl_corpus.build(data=data, out=out, worklist=worklist, index=index(tmp_path), now=T0)
    # m6 (not marked inactive here) is torch's autograd message: in the corpus, unattributed
    assert built["counts"] == {"models": 5, "clusters": 4, "torchlens": 4, "unattributed": 1,
                               "menagerie": 0}
    assert built["cohorts"] == {"earlier": 3, "r3": 1, "untagged": 1}
    lines = (out / "corpus.jsonl").read_text().splitlines()
    assert len(lines) == built["counts"]["models"]
    md = (out / "clusters.md").read_text()
    assert md.startswith("# TorchLens failure corpus") and "Reproduce: `python3 -m" in md
    assert (out / "history" / "2026-11-05" / "clusters.md").read_text() == md
    one = tl_corpus.build(data=data, out=tmp_path / "one", cohort="r3", worklist=worklist,
                          index=index(tmp_path), now=T0)
    assert one["counts"]["models"] == 1


# --- the tick step ---------------------------------------------------------------------------

def fake_build(groups: list[dict[str, Any]], rows: list[dict[str, Any]]) -> Any:
    def build(**kw: Any) -> dict[str, Any]:
        return {"clusters": groups, "rows": rows,
                "counts": {"models": len(rows), "clusters": len(groups), "torchlens": len(rows),
                           "unattributed": 0, "menagerie": 0}}
    return build


def group(cid: str, who: str = attrib.TORCHLENS) -> dict[str, Any]:
    return {"id": cid, "signature": f"sig {cid}", "count": 1, "attribution": who,
            "representatives": ["m1"]}


def test_new_clusters_are_told_once_and_the_first_refresh_only_seeds(tmp_path: Path) -> None:
    cfg = {"torchlens_corpus": {"enabled": True, "out": str(tmp_path / "out")},
           "retrace": {"enabled": True, "folder": str(tmp_path / "retrace")}}
    data, state = DataLayout(tmp_path / "data"), {}
    def run(groups: list[dict[str, Any]], now: datetime) -> tuple[dict[str, Any], list[str]]:
        return tl_corpus_step.step(data, cfg, state, now=now, dry_run=False, pin="2.36.0",
                                   build=fake_build(groups, []))
    out, facts = run([group("c-1")], T0)
    assert out["seeded"] and facts == []
    assert run([group("c-1"), group("c-2")], T0) == ({"fresh": "2026-11-05"}, [])
    out, facts = run([group("c-1"), group("c-2"), group("c-3", attrib.MENAGERIE)],
                     T0 + timedelta(days=1))
    assert out["new"] == ["c-2"] and len(facts) == 1 and "c-2 sig c-2" in facts[0]
    assert run([group("c-2")], T0 + timedelta(days=2))[1] == []  # unchanged: nothing sent
    assert run([group("c-1"), group("c-2")], T0 + timedelta(days=3))[1] == []  # back: seen


def test_a_new_pin_lists_stale_corpus_rows_for_a_trace_only_retrace_once(tmp_path: Path) -> None:
    folder = tmp_path / "retrace"
    cfg = {"torchlens_corpus": {"enabled": True, "out": str(tmp_path / "out")},
           "retrace": {"enabled": True, "folder": str(folder)}}
    rows = [{"id": "m1", "tl": "2.36.0", "attribution": attrib.TORCHLENS},
            {"id": "m2", "tl": "2.36.0", "attribution": attrib.UNATTRIBUTED},
            {"id": "m3", "tl": "2.36.0", "attribution": attrib.MENAGERIE},
            {"id": "m4", "attribution": attrib.TORCHLENS}]  # a give-up never traced
    data, state = DataLayout(tmp_path / "data"), {}
    build = fake_build([group("c-1")], rows)
    out, _ = tl_corpus_step.step(data, cfg, state, now=T0, dry_run=False, pin="2.36.0",
                                 build=build)
    assert out["listed_for_retrace"] == 1  # only the give-up never traced at its recipe
    pin_list = folder / retrace.PIN_LIST
    assert pin_list.read_text() == "m4\n"
    pin_list.write_text("")  # the feed took it
    out, _ = tl_corpus_step.step(data, cfg, state, now=T0, dry_run=False, pin="2.37.0",
                                 build=build)  # the pin moved: refreshed the same day
    assert out["listed_for_retrace"] == 3
    assert pin_list.read_text() == "m1\nm2\nm4\n"
    pin_list.write_text("")
    tl_corpus_step.step(data, cfg, state, now=T0 + timedelta(days=1), dry_run=False,
                        pin="2.37.0", build=build)
    assert pin_list.read_text() == ""  # listed once per pin, even after the feed took them
    assert state["torchlens_corpus"]["queued"] == {"2.37.0": ["m1", "m2", "m4"]}


def test_off_and_dry_run_do_nothing(tmp_path: Path) -> None:
    data, state = DataLayout(tmp_path / "data"), {}
    assert tl_corpus_step.step(data, {}, state, now=T0, dry_run=False, pin="1") == (
        {"skipped": "off"}, [])
    cfg = {"torchlens_corpus": {"enabled": True, "out": str(tmp_path)}}
    out, _ = tl_corpus_step.step(data, cfg, state, now=T0, dry_run=True, pin="1")
    assert "would_refresh" in out and state == {}
    with pytest.raises(ValueError, match="torchlens_corpus"):
        tl_corpus_step.config({"torchlens_corpus": True})


def test_the_feed_drains_the_pin_list_first_trace_only(tmp_path: Path) -> None:
    (tmp_path / "todo.txt").write_text("m10\nm11\n")
    (tmp_path / retrace.PIN_LIST).write_text("m1\nm2\n")
    cfg = {"retrace": {"enabled": True, "batch": 4, "folder": str(tmp_path)}}
    calls: list[dict[str, Any]] = []

    def queue(lane: None, **kw: Any) -> list[str]:
        calls.append(kw)
        return [f"{i}-trace-eval-x" for i in kw["ids"] if i != "m2"]  # m2 already traced

    def feed() -> dict[str, Any]:
        return retrace.feed(DataLayout(tmp_path / "data"), cfg, dry_run=False,
                            waiting=lambda s: 0, holds=lambda ids, data: {}, queue=queue)
    assert feed() == {"fed": ["m1"], "skipped": ["m2"], "left": 0, "list": retrace.PIN_LIST}
    assert calls[0]["trace_only"] is True and "again" not in calls[0]
    assert feed() == {"fed": ["m10", "m11"], "skipped": [], "left": 0}
    assert calls[1]["again"] is True and "trace_only" not in calls[1]


# --- the worker's frames and the tick's route -----------------------------------------------

def test_the_worker_records_the_raising_frames() -> None:
    def deep() -> None:
        raise ValueError("boom")

    with pytest.raises(worker.Failure) as caught, worker._stage("capture_error"):
        deep()
    frames = caught.value.frames
    assert frames and frames[-1].startswith("tests/pipeline/test_tl_corpus.py:")
    assert frames[-1].endswith(" deep")
    assert worker._where("/x/lib/python3.11/site-packages/torchlens/a/b.py") == "torchlens/a/b.py"
    assert worker._where("/src/torchlens/torchlens/v.py") == "torchlens/v.py"
    row = trace("m1", frames=frames)
    assert json.loads(json.dumps(row.frames)) == list(frames)


def test_corpus_facts_go_to_torchlens_and_the_rest_to_menagerie(tmp_path: Path) -> None:
    to_tl: list[str] = []
    to_crawler: list[str] = []
    deps = tick.Deps(send=lambda text, timeout: to_crawler.append(text),
                     send_torchlens=lambda text, timeout: to_tl.append(text))
    p = tick.Pass(mode="full", dry_run=False, data=DataLayout(tmp_path), deps=deps, cfg={},
                  state={"outbox": [[tick.TL_KEY, "new cluster"], ["disk", "disk paused"]]},
                  started=T0, deadline=time.monotonic() + 60)
    assert tick._send_outbox(p, time.monotonic() + 60) == [tick.TL_KEY, "disk"]
    assert to_tl == ["new cluster"] and to_crawler == ["disk paused"]
