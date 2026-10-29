"""The rung 0 regression driver and flow on real rows: the frozen Slice A table, the June catalog,
the probe's run result and the first integration rows (convnext_atto and coat_lite_mini at the
pin), plus the committed r0 table."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, read_json
from menagerie.pipeline import regression as reg
from menagerie.pipeline.flow import model_mode_of
from menagerie.pipeline.records import OutputFile, RunResult
from menagerie.pipeline.research import RUNG0_PAPERS

FIXTURES = Path(__file__).parent / "fixtures"
DATA = DataLayout(Path(__file__).parents[2] / "menagerie" / "data")
# The June validator at the pin and the new worker on two Slice A rows (2026-10-04, 3 threads).
CONVNEXT = {"stable_id": "m3882", "status": "validated", "n_ops": "207",
            "graph_shape_hash": "e254fceb0631042c12c98e588e19af7648f340433e4e3186966d3044e86f01b8"}
COAT = {"stable_id": "m3798", "status": "failed:replay", "n_ops": "0", "graph_shape_hash": ""}


def _new(base: dict, **changes: str) -> dict:
    row = {"stable_id": base["stable_id"], "mode": "trace", "status": base["status"],
           "n_ops": base["n_ops"], "graph_hash": base["graph_shape_hash"]}
    return {**row, **changes}


def _verdicts(new: list[dict], manifest: Path) -> dict[str, dict]:
    table = reg.compare(new, [manifest], data=DATA, stand_ins=reg.stand_ins(DATA))
    return {row["stable_id"]: row for row in table}


@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "june.tsv"
    reg.write_tsv([CONVNEXT, COAT], path, ("stable_id", "status", "n_ops", "graph_shape_hash"))
    return path


def test_the_frozen_table_has_the_two_batches() -> None:
    rows = reg.frozen(DATA)
    assert len(rows) == 200
    assert {r["batch"] for r in rows} == set(reg.BATCHES)
    assert sum(r["batch"] == "large" for r in rows) == 18


def test_caps_follow_the_baseline_batches() -> None:
    assert reg.caps_for("small", 6) == dataclasses.replace(
        reg.caps_for("small", 6), rss_mb=20 * 1024, threads=3, timeout_s=reg.TIMEOUT_S)
    assert reg.caps_for("large", 6).rss_mb == 40 * 1024
    assert reg.caps_for("large", 6).threads == 6


def test_june_stand_ins_are_the_eleven_resid_slice_rows() -> None:
    found = reg.stand_ins(DATA)
    assert sorted(found) == ["m3485", "m3487", "m3488", "m3489", "m3490", "m3626", "m3733",
                             "m3734", "m4095", "m4096", "m4098"]
    assert set(found.values()) == {"resid_slice_e"}


def test_classics_built_rows_cover_the_whole_catalog() -> None:
    rows = reg.classics_built(DATA)
    assert len(rows) == 1369
    assert all(r["stable_id"] for r in rows)
    assert sum(r["module"] == "resid_slice_e" for r in rows) == 117


def test_equal_verdicts_and_hashes_agree(manifest: Path) -> None:
    table = _verdicts([_new(CONVNEXT), _new(COAT)], manifest)
    assert (table["m3882"]["agree"], table["m3798"]["agree"]) == ("yes", "yes")
    assert table["m3485"]["agree"] == "incomplete"


def test_a_same_pin_difference_blocks(manifest: Path) -> None:
    table = _verdicts([_new(CONVNEXT, status="failed:replay"),
                       _new(COAT, status="validated", n_ops="460")], manifest)
    assert table["m3882"]["agree"] == table["m3798"]["agree"] == "no"
    other_hash = _verdicts([_new(CONVNEXT, graph_hash="0" * 64)], manifest)["m3882"]
    assert other_hash["agree"] == "no" and "graph hash equal False" in other_hash["note"]


def test_a_rerun_replaces_the_first_result_and_a_cause_explains(manifest: Path) -> None:
    first = [_new(CONVNEXT, status="failed:replay"), _new(COAT, status="validated")]
    rerun = [_new(COAT)]
    table = {r["stable_id"]: r for r in reg.compare(
        first, [manifest], data=DATA, stand_ins={}, reruns=rerun,
        explained={"m3882": "weights draw"})}
    assert (table["m3798"]["agree"], table["m3798"]["first_new"]) == ("yes", "validated")
    assert table["m3882"]["agree"] == "explained" and "weights draw" in table["m3882"]["note"]


def test_a_resource_limit_is_listed_not_blocking(manifest: Path) -> None:
    table = _verdicts([_new(CONVNEXT, status="failed:memory_cap")], manifest)
    assert table["m3882"]["agree"] == "resource"


def test_status_of_maps_the_worker_classes() -> None:
    result = read_json(RunResult, FIXTURES / "run_result.json")
    assert reg.status_of(result) == "validated"
    failed = dataclasses.replace(result, outcome="fail", cls="torchlens_gap:replay")
    assert reg.status_of(failed) == "failed:replay"
    assert reg.status_of(dataclasses.replace(result, outcome="fail", cls="memory_cap")) == (
        "failed:memory_cap")
    assert reg.status_of(dataclasses.replace(result, outcome="infra", cls="infra:no_result")) == (
        "infra")


def test_ratios_pair_rows_by_id() -> None:
    on = [{"stable_id": "m3882", "status": "validated", "save_capture_secs": "3.0", "wall_s": "20"},
          {"stable_id": "m3798", "status": "failed:replay", "save_capture_secs": "6.0",
           "wall_s": "40"}]
    off = [{"stable_id": "m3882", "status": "validated", "save_capture_secs": "1.0", "wall_s": "10"},
           {"stable_id": "m3798", "status": "failed:replay", "save_capture_secs": "3.0",
            "wall_s": "40"}]
    summary = reg.ratios(on, off)
    assert summary["pairs"] == 2 and summary["capture_ratio_median"] == 2.5
    assert summary["wall_ratio_median"] == 1.5 and summary["verdicts_differ"] == []


def test_request_names_carry_the_model_mode() -> None:
    result = read_json(RunResult, FIXTURES / "run_result.json")
    named = dataclasses.replace(result, request="m11497-trace-train-20261004t084151z-5ed5")
    assert model_mode_of(named) == "train"
    shaped = dataclasses.replace(result,
                                 request="m3635-trace-structure-only-eval-20261004t084151z-a1b2")
    assert model_mode_of(shaped) == "eval"


def test_rung0_papers_name_existing_packets() -> None:
    for family in RUNG0_PAPERS:
        assert DATA.packet(family).is_file(), family


def test_the_committed_r0_table_has_no_blocking_row() -> None:
    path = DATA.root / "regression" / "slice_a_200_r0.tsv"
    if not path.is_file():
        pytest.skip("written by the integration run")
    rows = reg.read_tsv(path)
    assert len(rows) == 200
    assert not [r["stable_id"] for r in rows if r["agree"] == "no"]
    assert all(r["note"] for r in rows if r["agree"] == "explained")
    assert {r["tl"] for r in rows if r["tl"]} == {"2.35.2"}


# --- pin check: a later run against the committed r0 table -------------------------------


def _r0_table() -> list[dict[str, str]]:
    return reg.read_tsv(DATA.root / "regression" / "slice_a_200_r0.tsv")


def _r1(r0: dict[str, str], **changes: str) -> dict[str, str]:
    """A later-run row that repeats r0's result for one row; the hash comes from ``r0_rows``."""

    row = {"stable_id": r0["stable_id"], "mode": "trace", "status": r0["new"],
           "n_ops": r0["new_n_ops"], "graph_hash": "a" * 64}
    return {**row, **changes}


def _pin(r1: list[dict], reruns: list[dict] = ()) -> dict[str, dict]:
    r0 = _r0_table()
    old = [{"stable_id": r["stable_id"], "mode": "trace", "graph_hash": "a" * 64} for r in r0]
    shapes = [{"stable_id": r["stable_id"], "mode": "trace-structure-only",
               "status": r["structure_only"]} for r in r0 if r["structure_only"]]
    table = reg.pincheck(r1 + shapes, r0, r0_rows=old, reruns=list(reruns))
    return {row["stable_id"]: row for row in table}


def test_a_repeat_of_r0_agrees_on_every_row() -> None:
    table = _pin([_r1(r) for r in _r0_table()])
    assert {row["agree"] for row in table.values()} == {"yes"}


def test_a_moved_verdict_op_count_or_hash_blocks() -> None:
    rows = {r["stable_id"]: r for r in _r0_table()}
    convnext = rows["m3882"]  # validated in r0
    table = _pin([_r1(convnext, n_ops="208"), _r1(rows["m3798"], status="validated")])
    assert table["m3882"]["agree"] == "no" and "n_ops r0 207, r1 208" in table["m3882"]["note"]
    assert table["m3798"]["agree"] == "no" and "verdict" in table["m3798"]["note"]
    assert _pin([_r1(convnext, graph_hash="b" * 64)])["m3882"]["note"] == "graph hash differs"
    assert _pin([])["m3882"]["agree"] == "incomplete"
    cause = reg.pincheck([_r1(convnext, n_ops="208")], _r0_table(),
                         explained={"m3882": "worker change"})
    assert next(r for r in cause if r["stable_id"] == "m3882")["agree"] == "explained"


def test_a_coatnet_difference_counts_only_after_its_rerun() -> None:
    rows = {r["stable_id"]: r for r in _r0_table()}
    coatnet = next(r for r in rows.values() if r["agree"] == "explained")
    flipped = _r1(coatnet, status="validated")
    assert _pin([flipped])[coatnet["stable_id"]]["agree"] == "recheck"
    settled = _pin([flipped], reruns=[_r1(coatnet)])[coatnet["stable_id"]]
    assert (settled["agree"], settled["first_r1"]) == ("yes", "validated")
    assert _pin([flipped], reruns=[flipped])[coatnet["stable_id"]]["agree"] == "no"


# --- sizing (fork 11) ------------------------------------------------------------------------


def _attempt(root: Path, name: str, *, outcome: str = "pass", rows: list[dict] = (),
             files: list[dict] = (), log: list[dict] = (), summary: dict | None = None) -> None:
    import json

    attempt = root / name
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps(
        {"outcome": outcome, "rows": list(rows), "files": list(files)}))
    if summary is not None:
        (attempt / "summary.json").write_text(json.dumps(summary))
    if log:
        (attempt / "render").mkdir()
        (attempt / "render" / "render.log").write_text(
            "".join(json.dumps(e) + "\n" for e in log))


def _sha(n: int) -> str:
    return f"{n:064x}"


def test_model_sizes_sum_both_modes_and_take_the_newest_passing_trace(tmp_path: Path) -> None:
    trace_row = {"stage": "trace", "n_ops": 245, "capture": "full"}
    saves = [{"kind": "tlspec", "path": "t", "sha256": _sha(1), "bytes": 1000},
             {"kind": "tlspec-portable", "path": "p", "sha256": _sha(2), "bytes": 400}]
    _attempt(tmp_path, "m11497-trace-eval-20261004t080000z-aaaa", outcome="fail")
    _attempt(tmp_path, "m11497-trace-eval-20261004t070000z-bbbb", rows=[trace_row], files=saves,
             summary={"save_capture_secs": 2.5})
    _attempt(tmp_path, "m11497-trace-train-20261004t070500z-cccc", rows=[trace_row],
             files=saves[:1], summary={"save_capture_secs": 1.5})
    svg = {"kind": "svg", "variant": "unrolled-none", "sha256": _sha(3), "bytes": 50}
    _attempt(tmp_path, "m11497-render-eval-20261004t071000z-dddd",
             rows=[{"stage": "render", "secs": 4.0}],
             log=[svg, {"kind": "html", "sha256": _sha(4), "bytes": 70},
                  {"kind": "svg", "variant": "unrolled-max", "why": "render:same-as-unrolled-auto"}])
    _attempt(tmp_path, "m11497-render-train-20261004t071100z-eeee",
             rows=[{"stage": "render", "secs": 3.0}],
             log=[{**svg, "variant": "unrolled-none-train", "sha256": _sha(5)}])
    (row,) = reg.sizes([tmp_path])
    assert (row["tlspec"], row["tlspec-portable"], row["svg"], row["html"]) == (2000, 400, 100, 70)
    assert row["total"] == 2570 and (row["save_s"], row["render_s"]) == (4.0, 7.0)
    assert row["train"] and row["rendered"] and not row["monster"] and not row["recurrent"]
    assert row["variants"] == "unrolled-none;unrolled-none-train"


def test_a_refused_runnable_save_without_a_portable_twin_is_no_missing_artifact(
        tmp_path: Path) -> None:
    """Since 2026-10-28 a trace whose runnable save was refused lists ``tlspec`` alone (it is the
    portable save): its attempt still counts as saved, sizes with ``tlspec-portable`` at 0, and
    its verdict reads from the outcome, never from the files."""

    trace_row = {"stage": "trace", "n_ops": 12, "capture": "full", "runnable": "refused:x"}
    alone = [{"kind": "tlspec", "path": "trace.tlspec", "sha256": _sha(1), "bytes": 900}]
    _attempt(tmp_path, "m20100-trace-eval-20261028t070000z-aaaa", outcome="fail",
             rows=[trace_row], files=alone, summary={"save_capture_secs": 1.0})
    _attempt(tmp_path, "m20100-render-eval-20261028t071000z-bbbb",
             rows=[{"stage": "render", "secs": 2.0}],
             log=[{"kind": "svg", "variant": "unrolled-none", "sha256": _sha(3), "bytes": 50}])
    (row,) = reg.sizes([tmp_path])
    assert (row["tlspec"], row["tlspec-portable"], row["svg"], row["total"]) == (900, 0, 50, 950)
    assert row["rendered"] and set(reg.ARTIFACTS) <= set(row)
    result = RunResult(request="m20100-trace-eval-20261028t070000z-aaaa", id="m20100",
                       mode="trace", outcome="pass", started="2026-10-28T07:00:00Z",
                       ended="2026-10-28T07:00:05Z", host="worker",
                       files=(OutputFile(kind="tlspec", path="trace.tlspec", sha256=_sha(1),
                                         bytes=900),))
    assert reg.status_of(result) == "validated"


def test_the_sizing_sample_keeps_every_forced_row_and_fills_each_decile() -> None:
    def row(i: int, **flags: bool) -> dict:
        return {"stable_id": f"m{i}", "n_ops": i, "capture": "full", "monster": False,
                "recurrent": False, "train": False, **flags}

    pool = [row(i) for i in range(1, 301)]
    pool[5] = row(6, recurrent=True)
    pool[7] = {**row(8), "capture": "structure_only"}
    pool[299] = row(300, monster=True)
    picked = [r["stable_id"] for r in reg.sizing_pick(pool)]
    assert len(picked) == len(set(picked)) == 100
    assert {"m6", "m8", "m300"} <= set(picked)
    deciles = [sum(1 for r in reg.sizing_pick(pool) if b * 30 < r["n_ops"] <= (b + 1) * 30)
               for b in range(10)]
    assert deciles == [10] * 10


def test_the_projection_weights_deciles_by_the_reference_counts_and_adds_monsters() -> None:
    sample = [{"stable_id": f"m{i}", "n_ops": i, "monster": False,
               **dict.fromkeys(reg.ARTIFACTS, 0), "total": float(i)} for i in range(1, 101)]
    sample.append({"stable_id": "m9999", "n_ops": 3000, "monster": True,
                   **dict.fromkeys(reg.ARTIFACTS, 0), "total": 1000.0})
    low = reg.project(sample, [1] * 50, monster_share=0.0, models=10)
    assert low["total"]["per_model"] == 5.5  # all in the first decile: mean of 1..10
    mixed = reg.project(sample, [1, 100], monster_share=0.5, models=10)
    assert mixed["total"]["per_model"] == 0.5 * (5.5 + 95.5) / 2 + 0.5 * 1000
    assert mixed["total"]["total"] == 10 * mixed["total"]["per_model"]


def test_the_committed_r1_pin_check_has_no_blocking_row() -> None:
    path = DATA.root / "regression" / "slice_a_200_r1.tsv"
    if not path.is_file():
        pytest.skip("written by the pin check")
    rows = reg.read_tsv(path)
    assert len(rows) == 200
    assert {r["agree"] for r in rows} <= {"yes", "explained"}
    assert all(r["note"] for r in rows if r["agree"] == "explained")
    assert {r["tl"] for r in rows if r["tl"]} == {"2.35.3"}


def test_a_replay_failure_with_a_saved_trace_still_counts_for_sizing(tmp_path: Path) -> None:
    save = [{"kind": "tlspec", "path": "t", "sha256": _sha(1), "bytes": 9000}]
    _attempt(tmp_path, "m1481-trace-eval-20261004t070000z-aaaa", outcome="fail",
             rows=[{"stage": "trace", "n_ops": 2839}], files=save)
    _attempt(tmp_path, "m482-trace-eval-20261004t070000z-bbbb", outcome="fail",
             rows=[{"stage": "trace"}])
    (row,) = reg.sizes([tmp_path])
    assert (row["stable_id"], row["tlspec"], row["monster"]) == ("m1481", 9000, True)
