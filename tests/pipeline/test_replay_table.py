"""Replay diagnostics preserve the cohort and read packed, metadata-only saves."""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, append_jsonl, replay_table, store
from menagerie.pipeline.records import LedgerRow, RosterRow


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE", str(tmp_path / "cache"))
    data, blobs = DataLayout(tmp_path / "data"), StoreLayout(tmp_path / "store")
    data.root.mkdir()
    append_jsonl([RosterRow(name="toy", zoo="timm", family="f-toy", stable_id="m1")], data.roster)
    return data, blobs, tmp_path / "out"


def save(data: DataLayout, blobs: StoreLayout, folder: Path, ops: list[dict]) -> str:
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"torchlens_version": "2.35.3"}))
    (folder / "metadata.pkl").write_bytes(pickle.dumps({"layer_list": ops}))
    return store.put(folder, store=blobs, data=data, mime="application/x-tlspec",
                     license_class="open").sha256


def trace(data: DataLayout, sha: str | None, msg: str, **extra) -> LedgerRow:
    row = LedgerRow(id="m1", stage="trace", at="2026-10-09T08:00:00Z", host="worker",
                    env="torch-main", recipe_rev="0" * 12, outcome="fail", tl="2.35.3",
                    cls="torchlens_gap:replay", validation_state="failed", trace=sha, msg=msg)
    row = replace(row, **extra)
    append_jsonl([row], data.ledger("2026-10"))
    return row


def read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


OPS = [{"label": "view_1_2:1", "func_name": "view", "shape": (1, 64, 7, 7), "parents": []},
       {"label": "conv2d_33_391:1", "func_name": "conv2d", "shape": (1, 64, 7, 7),
        "parents": ["view_1_2:1"]}]
MSG = ("single-thread retry: forward_replay mismatch at conv2d_33_391:1 (conv2d): "
       "max_abs_diff=4.768e-07 max_rel_diff=7.022e-04 reduction_depth=9")


def test_shallow_conv_and_packed_metadata_only_trace(world, tmp_path: Path) -> None:
    data, blobs, out = world
    sha = save(data, blobs, tmp_path / "trace", OPS)
    assert store.packed_path(sha, store=blobs).is_file()  # trace saves are written packed
    assert not blobs.blob(sha).exists()
    trace(data, sha, MSG)
    result = replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)
    row, = read(result["table"])
    assert (row["op_type"], float(row["max_abs_diff"]), row["reduction_depth"]) == (
        "conv2d", 4.768e-7, "9")
    assert row["family"] == "f-toy" and row["thread_mode"] == "single-thread retry"
    assert json.loads(row["parents"]) == [{"label": "view_1_2:1", "op_type": "view",
                                           "shape": [1, 64, 7, 7]}]
    assert row["metadata_status"] == "ok" and result["by_op_type"] == {"conv2d": 1}


def test_insensitive_row_names_the_perturbed_parent(world, tmp_path: Path) -> None:
    data, blobs, out = world
    sha = save(data, blobs, tmp_path / "trace", OPS)
    trace(data, sha, "single-thread retry: perturbation mismatch at conv2d_33_391:1 (conv2d) "
          "-- output insensitive to perturbing parent(s) ['view_1_2']; the parent does not influence")
    row, = read(replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)["table"])
    assert row["failure_kind"] == "insensitive"
    assert json.loads(row["perturbed_parents"]) == ["view_1_2"]
    assert json.loads(row["parents"])[0]["op_type"] == "view"


def test_latest_trace_per_pin_and_ties_never_duplicate_ids(world) -> None:
    data, blobs, out = world
    old = trace(data, None, MSG)
    trace(data, None, "unverified", cls="torchlens_unverified", validation_state="unverified")
    result = replay_table.table(data=data, store=blobs, pin="2.35.3", out=out / "before")
    assert read(result["table"]) == []
    append_jsonl([replace(old, at="2026-10-09T09:00:00Z")], data.ledger("2026-10"))
    assert replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)["ids"] == ["m1"]


def test_baseline_cohort_stays_frozen_after_new_pin_and_new_failures(world) -> None:
    data, blobs, out = world
    trace(data, None, MSG)
    replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)
    trace(data, None, "", tl="2.36.0", outcome="pass", cls=None, validation_state="passed")
    trace(data, None, MSG, id="m2")
    assert replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)["ids"] == ["m1"]
    result = replay_table.table(data=data, store=blobs, pin="2.36.0", out=out)
    assert result["ids"] == ["m1"] and result["by_outcome"] == {"pass": 1}


def test_malformed_parent_message_is_a_diagnostic_not_a_dropped_row(world) -> None:
    data, blobs, out = world
    trace(data, None, MSG + "; perturbing parent(s) [broken]")
    result = replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)
    row, = read(result["table"])
    assert result["ids"] == ["m1"] and row["outcome"] == "fail"
    assert row["metadata_status"] == "unavailable" and "ValueError" in row["metadata_error"]


def test_new_pin_keeps_pass_and_unverified_in_frozen_baseline_cohort(world) -> None:
    data, blobs, out = world
    trace(data, None, MSG)
    replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)
    trace(data, None, "structure mismatch", tl="2.36.0", cls="torchlens_unverified",
          validation_state="unverified")
    result = replay_table.table(data=data, store=blobs, pin="2.36.0", out=out)
    row, = read(result["table"])
    assert row["outcome"] == "unverified" and result["ids"] == ["m1"]
    trace(data, None, "", tl="2.36.0", outcome="pass", cls=None, validation_state="passed")
    row, = read(replay_table.table(data=data, store=blobs, pin="2.36.0", out=out)["table"])
    assert row["outcome"] == "pass" and row["failing_op"] == ""


def test_missing_retrace_refuses_before_writing_new_pin_table(world) -> None:
    data, blobs, out = world
    trace(data, None, MSG)
    replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)
    with pytest.raises(ValueError, match="not re-traced.*m1"):
        replay_table.table(data=data, store=blobs, pin="2.36.0", out=out)
    assert not (out / "replay-ops-2.36.0.tsv").exists()


def test_missing_blob_is_reported_without_dropping_the_id(world) -> None:
    data, blobs, out = world
    trace(data, "0" * 64, MSG)
    result = replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)
    row, = read(result["table"])
    assert row["metadata_status"] == "unavailable" and row["outcome"] == "fail"
    assert result["metadata_errors"] == 1 and result["ids"] == ["m1"]


def test_refuses_unsafe_metadata_without_execution(world, tmp_path: Path) -> None:
    data, blobs, out = world
    class Gadget:
        def __reduce__(self):
            return eval, ("1 / 0",)
    sha = save(data, blobs, tmp_path / "trace", [Gadget()])
    trace(data, sha, MSG)
    row, = read(replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)["table"])
    assert row["metadata_status"] == "unavailable"
    assert "ZeroDivisionError" not in row["metadata_error"]


@pytest.mark.parametrize("pin", ["../escape", "bad/pin", "", "2.35.3\tbad"])
def test_pin_cannot_escape_output_directory(world, pin: str) -> None:
    data, blobs, out = world
    with pytest.raises(ValueError, match="pin"):
        replay_table.table(data=data, store=blobs, pin=pin, out=out)


def test_excluded_id_is_a_row_with_its_reason_and_others_still_refuse(world) -> None:
    data, blobs, out = world
    trace(data, None, MSG)
    trace(data, None, MSG, id="m2")
    replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)
    trace(data, None, "", tl="2.36.0", outcome="pass", cls=None, validation_state="passed")
    with pytest.raises(ValueError, match="not re-traced.*m2"):
        replay_table.table(data=data, store=blobs, pin="2.36.0", out=out, exclude={"m1": "x"})
    result = replay_table.table(data=data, store=blobs, pin="2.36.0", out=out,
                                exclude={"m2": "cluster only; next axon batch"})
    rows = {row["id"]: row for row in read(result["table"])}
    assert rows["m1"]["outcome"] == "pass" and result["count"] == 2
    assert rows["m2"]["outcome"] == "excluded" and rows["m2"]["pin"] == "2.36.0"
    assert rows["m2"]["msg"] == "cluster only; next axon batch"
    assert result["excluded"] == {"m2": "cluster only; next axon batch"}
    assert result["by_outcome"] == {"pass": 1, "excluded": 1}


def test_exclusion_never_hides_a_real_retrace(world) -> None:
    data, blobs, out = world
    trace(data, None, MSG)
    replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)
    trace(data, None, "", tl="2.36.0", outcome="pass", cls=None, validation_state="passed")
    result = replay_table.table(data=data, store=blobs, pin="2.36.0", out=out,
                                exclude={"m1": "late"})
    row, = read(result["table"])
    assert row["outcome"] == "pass" and result["excluded"] == {}


@pytest.mark.parametrize("exclude", [{"m9": "not in cohort"}, {"m1": " "}])
def test_exclusion_outside_cohort_or_without_reason_refuses(world, exclude) -> None:
    data, blobs, out = world
    trace(data, None, MSG)
    replay_table.table(data=data, store=blobs, pin="2.35.3", out=out)
    with pytest.raises(ValueError, match="exclu"):
        replay_table.table(data=data, store=blobs, pin="2.36.0", out=out, exclude=exclude)
    assert not (out / "replay-ops-2.36.0.tsv").exists()


def test_cli_exclude_parses_id_reason_pairs_and_rejects_bare_ids(capsys) -> None:
    from menagerie.pipeline import cli
    args = cli.build_parser().parse_args(
        ["replay", "table", "--exclude", "m4310=axon-cpu only", "--exclude", "m4311=a=b"])
    assert dict(args.exclude) == {"m4310": "axon-cpu only", "m4311": "a=b"}
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["replay", "table", "--exclude", "m4310"])
    assert "ID=REASON" in capsys.readouterr().err
