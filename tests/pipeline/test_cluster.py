"""``cluster``: AXON batches written, run and collected without a cluster.

Torch-free: ``run`` goes through a stand-in for ``runner.attempt``, and ``collect`` reads a
batch folder laid out as the job leaves it, through stand-ins for ssh and rsync.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, cluster, read_json, write_json
from menagerie.pipeline.records import (
    Caps, LedgerRow, ModelRecord, Modes, RecordError, RunRequest, RunResult,
)

FIXTURES = Path(__file__).with_name("fixtures")
SHA = "ab" * 32
HOME = "/home/someone"


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """Three giants (two cpu rows, one reviewed) and a worker row, a private work root and a
    store holding the cpu adapter row's vendor snapshot."""

    layout = DataLayout(tmp_path / "data")
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(cluster, "_commit", lambda: "c" * 40)
    base = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    rows = {"m9101": ("axon-cpu", None), "m9102": ("axon-cpu", SHA),
            "m9103": ("axon-reviewed", None), "m9104": (None, None)}
    for sid, (placement, vendor) in rows.items():
        recipe = replace(base.recipe, placement=placement)
        if vendor:
            recipe = replace(recipe, kind="adapter", call=None, args=(), kwargs={},
                             adapter="adapters/f-giant.py", vendor=vendor)
        write_json(replace(base, id=sid, name=f"giant-{sid}", family="f-giant", recipe=recipe),
                   layout.model(sid))
    (layout.root / "adapters").mkdir(parents=True)
    (layout.root / "adapters" / "f-giant.py").write_text("def build(cfg):\n    return None\n")
    blob = StoreLayout(tmp_path / "store").blob(SHA)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"snapshot")
    return layout


def test_batch_writes_script_records_and_vendor(data: DataLayout, tmp_path: Path) -> None:
    script = cluster.axon_batch("cpu", ["m9101", "m9102"], data=data, home=HOME)
    batch = script.parent
    assert batch.parent == tmp_path / "work" / "axon" and batch.name.startswith("cpu-")
    meta = json.loads((batch / "batch.json").read_text())
    assert meta["ids"] == ["m9101", "m9102"] and meta["env"] == "torch-main"
    assert meta["root"] == f"{HOME}/projects/menagerie" and meta["commit"] == "c" * 40
    assert meta["revisions"]["m9101"] == read_json(ModelRecord, data.model("m9101")).recipe.rev
    copied = DataLayout(batch / "data")
    assert read_json(ModelRecord, copied.model("m9102")) == read_json(ModelRecord,
                                                                      data.model("m9102"))
    assert (copied.root / "adapters" / "f-giant.py").is_file()
    assert StoreLayout(batch / "store").blob(SHA).read_bytes() == b"snapshot"
    text = script.read_text()
    root = f"{HOME}/projects/menagerie"
    assert f"#SBATCH --output={root}/logs/{batch.name}-%A.log" in text
    assert f"CODE=$ROOT/code/{'c' * 40}" in text and "MENAGERIE_SANDBOX=none" in text
    assert "MENAGERIE_CPUS=8 MENAGERIE_MEMORY_GB=60" in text  # 64 GiB less the headroom
    from menagerie.pipeline import envs

    assert (f"bash envs/axon/build.sh torch-main $MENAGERIE_ENVS/torch-main-"
            f"{envs.lock_id('torch-main')[-8:]} {envs.lock_id('torch-main')} pyvips pyvips-binary"
            in text)
    assert "envs canary torch-main" in text and (cluster.SPECS / "build.sh").is_file()
    assert "axon run \"$BATCH\"" in text and "DONE" in text
    assert "--exclude" not in text and "gpu.sh" not in text


def test_batch_refuses_rows_the_kind_does_not_run(data: DataLayout) -> None:
    with pytest.raises(RecordError, match="placement None"):
        cluster.axon_batch("cpu", ["m9104"], data=data, home=HOME)
    with pytest.raises(RecordError, match="placement 'axon-reviewed'"):
        cluster.axon_batch("cpu", ["m9101", "m9103"], data=data, home=HOME)
    with pytest.raises(RecordError, match="holds at most 4 rows"):
        cluster.axon_batch("reviewed", ["m9103"] * 5, data=data, home=HOME)
    with pytest.raises(RecordError, match="kind"):
        cluster.axon_batch("tpu", ["m9101"], data=data, home=HOME)  # type: ignore[arg-type]
    script = cluster.axon_batch("reviewed", ["m9103"], data=data, home=HOME)
    assert "MENAGERIE_MEMORY_GB=124" in script.read_text()
    with pytest.raises(RecordError, match="placement 'axon-cpu'"):
        cluster.axon_batch("reviewed", ["m9101"], data=data, home=HOME)
    from menagerie.pipeline import ledger

    rev = read_json(ModelRecord, data.model("m9101")).recipe.rev
    capped = LedgerRow(id="m9101", stage="trace", at="2026-10-05T12:00:00Z", host="cluster",
                       env="torch-main@00000000", recipe_rev=rev, outcome="fail", mode="eval",
                       capture="full", cls="memory_cap")
    ledger.append([capped], data=data)
    assert cluster.axon_batch("reviewed", ["m9101"], data=data, home=HOME).is_file()


def test_gpu_script_avoids_the_old_cards_and_loads_the_toolchain() -> None:
    text = cluster.job_script(batch="gpu-x", kind="gpu", env="torch-main", commit="c" * 40,
                              root="/r")
    assert f"#SBATCH --exclude={cluster.GPU_EXCLUDE}" in text
    assert 'source "$CODE/envs/axon/gpu.sh"' in text
    assert (cluster.SPECS / "gpu.sh").is_file()
    version, digest = (cluster.SPECS / "uv.txt").read_text().split()[:2]
    assert version.count(".") == 2 and len(digest) == 64


def test_submit_copies_and_calls_axon_submit(data: DataLayout, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    script = cluster.axon_batch("cpu", ["m9101"], data=data, home=HOME)
    calls: list[Any] = []
    monkeypatch.setattr(cluster, "_ssh", lambda cmd, **kw: calls.append(("ssh", cmd)) or "")
    monkeypatch.setattr(cluster, "_rsync", lambda *a: calls.append(("rsync", a)))
    monkeypatch.setattr(cluster, "_push_code", lambda commit, root: calls.append(("code", commit)))
    monkeypatch.setattr(cluster, "_uv_binary", lambda: tmp_path / "uv")

    class Done:
        returncode, stdout, stderr = 0, "4242\n", ""

    seen: list[list[str]] = []
    monkeypatch.setattr(cluster.subprocess, "run", lambda argv, **kw: seen.append(argv) or Done())
    assert cluster.submit(script.parent) == "4242"
    (argv,) = seen
    assert argv[argv.index("--memory-gib") + 1] == "64" and argv[-1] == "--submit"
    assert argv[argv.index("--script") + 1].endswith(f"/batches/{script.parent.name}/job.sh")
    assert ("code", "c" * 40) in calls
    assert json.loads((script.parent / "batch.json").read_text())["submitted"]["job"] == "4242"
    assert "--reviewed" not in argv
    reviewed = cluster.axon_batch("reviewed", ["m9103"], data=data, home=HOME)
    cluster.submit(reviewed.parent, dry_run=True)
    argv = seen[-1]
    assert argv[argv.index("--reviewed") + 1] == "giants-128g" and "--submit" not in argv
    assert argv[argv.index("--memory-gib") + 1] == "128"


def _result(request: RunRequest, outcome: str, cls: str | None = None,
            modes: Modes | None = None) -> RunResult:
    stage = "forward" if request.mode == "forward" else "trace"
    capture = None if stage == "forward" else (
        "structure_only" if request.mode == "trace-structure-only" else "full")
    row = LedgerRow(id=request.id, stage=stage, at="2026-10-05T12:00:00Z", host="cluster",
                    env="torch-main@00000000", recipe_rev=request.recipe.rev, outcome=outcome,
                    mode=request.model_mode, capture=capture, cls=cls,
                    validation_state="structure_only" if capture == "structure_only" else None)
    return RunResult(request=request.request, id=request.id, mode=request.mode,
                     outcome=outcome, started="2026-10-05T12:00:00Z",
                     ended="2026-10-05T12:01:00Z", host="cluster", cls=cls, rows=(row,),
                     modes=modes)


def test_run_does_the_full_attempt_and_stops_before_the_deadline(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = cluster.axon_batch("cpu", ["m9101", "m9102"], data=data, home=HOME)
    batch = script.parent
    from menagerie.pipeline import runner

    modes: list[tuple[str, str, str | None, int]] = []

    def attempt(request: RunRequest, *, attempt_dir: Path) -> RunResult:
        modes.append((request.id, request.mode, request.recipe.placement, request.caps.rss_mb))
        if request.mode == "trace":
            return _result(request, "fail", "memory_cap")
        return _result(request, "pass", modes=Modes(eval="pass", train="pass",
                                                    divergence="structural"))

    monkeypatch.setattr(runner, "attempt", attempt)
    monkeypatch.setenv("MENAGERIE_MEMORY_GB", "60")
    monkeypatch.setenv("MENAGERIE_CPUS", "8")
    with pytest.raises(RuntimeError, match="SLURM"):
        cluster.run(batch)
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    # Time runs short once the first row's two attempts are done.
    monkeypatch.setattr(cluster, "_left_s", lambda: cluster.ROW_MIN_S + (-1 if len(modes) >= 2
                                                                        else 4000))
    table = cluster.run(batch)
    assert [m[:2] for m in modes] == [("m9101", "forward"), ("m9101", "trace")]
    assert {m[2] for m in modes} == {None} and {m[3] for m in modes} == {60 * 1024}
    assert table[1] == {"stable_id": "m9102", "name": "giant-m9102", "trace": "not_reached"}
    rows = [json.loads(line) for line in next((batch / "data" / "ledger").glob("*.jsonl"))
            .read_text().splitlines()]
    assert [(r["stage"], r["outcome"]) for r in rows] == [("forward", "pass"), ("trace", "fail")]


def test_run_gpu_batch_makes_one_gpu_attempt(data: DataLayout, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    record = read_json(ModelRecord, data.model("m9101"))
    write_json(replace(record, recipe=replace(record.recipe, placement="gpu-window")),
               data.model("m9101"))
    batch = cluster.axon_batch("gpu", ["m9101"], data=data, home=HOME).parent
    from menagerie.pipeline import runner

    seen: list[str] = []
    monkeypatch.setattr(runner, "attempt", lambda request, attempt_dir: seen.append(
        request.mode) or _result(request, "pass"))
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    cluster.run(batch)
    assert seen == ["gpu"]


def _finished(batch: Path, data: DataLayout) -> None:
    """Lay out ``back``'s source as the job leaves it: DONE, table, ledger and index lines."""

    there = batch / "remote"
    (there / "data" / "ledger").mkdir(parents=True)
    (there / "data" / "store-index").mkdir(parents=True)
    rev = read_json(ModelRecord, data.model("m9101")).recipe.rev
    line = json.dumps({"schema": "ledger/1", "id": "m9101", "stage": "trace", "rev": rev})
    (there / "data" / "ledger" / "2026-10.jsonl").write_text(line + "\n")
    index = json.dumps({"sha256": "cd" * 32, "bytes": 3})
    (there / "data" / "store-index" / "2026-10.jsonl").write_text(index + "\n")
    (there / "table.jsonl").write_text(json.dumps({"stable_id": "m9101", "trace": "trace:pass"})
                                       + "\n")
    (there / "DONE").write_text("rc=0\n")


def test_collect_appends_new_lines_after_the_blobs_verify(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = cluster.axon_batch("cpu", ["m9101"], data=data, home=HOME).parent
    monkeypatch.setattr(cluster, "_ssh", lambda cmd, **kw: "")
    assert cluster.collect(batch, data=data) == {"batch": batch.name, "done": False}
    _finished(batch, data)
    monkeypatch.setattr(cluster, "_ssh", lambda cmd, **kw: "rc=0")

    def rsync(src: str, dest: str, *extra: str) -> None:
        if "/logs/" not in src:
            shutil.copytree(batch / "remote", dest, dirs_exist_ok=True)

    monkeypatch.setattr(cluster, "_rsync", rsync)
    monkeypatch.setenv("MENAGERIE_STORE_REMOTE", "worker:store")
    monkeypatch.setattr(cluster, "_relay_blobs", lambda rels, source, remote: [])
    with pytest.raises(RuntimeError, match="did not verify"):
        cluster.collect(batch, data=data)
    assert not (data.root / "ledger").exists()  # nothing indexed before its blob is there
    relayed: list[list[str]] = []
    monkeypatch.setattr(cluster, "_relay_blobs",
                        lambda rels, source, remote: relayed.append(rels) or list(rels))
    first = cluster.collect(batch, data=data)
    assert first["added"] == {"ledger/2026-10.jsonl": 1, "store-index/2026-10.jsonl": 1}
    assert first["blobs"] == 1 and first["stale"] == [] and relayed[0] == [f"sha256/cd/{'cd' * 32}"]
    again = cluster.collect(batch, data=data)
    assert set(again["added"].values()) == {0}
    record = read_json(ModelRecord, data.model("m9101"))
    write_json(replace(record, recipe=replace(record.recipe, rev="0" * 12)), data.model("m9101"))
    assert cluster.collect(batch, data=data)["stale"] == ["m9101"]


def test_extra_code_trees_are_staged_and_drained_and_enter_the_revision(
        data: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A recipe's ``code`` trees (a compiled library's saved source the adapter materializes) go
    into the batch store with its vendor and through the bridge; unset, nothing changes: no key in
    the record, the same revision."""

    from menagerie.pipeline import bridge, recipe_revision
    from menagerie.pipeline.records import to_dict

    extra = "cd" * 32
    blob = StoreLayout(tmp_path / "store").blob(extra)
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"library source")
    record = read_json(ModelRecord, data.model("m9102"))
    assert "code" not in to_dict(record.recipe)
    with_code = replace(record, recipe=replace(record.recipe, code=(extra,)))
    write_json(with_code, data.model("m9102"))
    assert read_json(ModelRecord, data.model("m9102")).recipe.code == (extra,)
    with pytest.raises(RecordError):
        replace(record.recipe, code=("not-a-sha",))

    script = cluster.axon_batch("cpu", ["m9102"], data=data, home=HOME)
    staged = StoreLayout(script.parent / "store")
    assert staged.blob(SHA).read_bytes() == b"snapshot"
    assert staged.blob(extra).read_bytes() == b"library source"

    plain = to_dict(record.recipe)
    assert recipe_revision(plain, None, b"") == recipe_revision({**plain, "code": None}, None, b"")
    assert recipe_revision(plain, None, b"") != recipe_revision(to_dict(with_code.recipe), None, b"")

    drained: list[str | None] = []
    monkeypatch.setattr(bridge, "_drain", drained.append)
    local = tmp_path / "hubspool"
    for name in ("new", "claimed", "done", "rejected"):
        (local / name).mkdir(parents=True)
    request = RunRequest(request="m9102-forward-eval-x", id="m9102", mode="forward",
                         recipe=with_code.recipe,
                         caps=Caps(rss_mb=1024, timeout_s=60, threads=1))
    path = local / "new" / f"{request.request}.json"
    write_json(request, path)
    link = type("Link", (), {"run": lambda self, *a, **k: b""})()
    bridge._forward(link, path, local, "spool")
    assert drained == [SHA, extra] and (local / "claimed" / path.name).exists()


def test_the_feeder_places_rows_in_the_batch_copy_only(data: DataLayout) -> None:
    batch = cluster.axon_batch("cpu", ["m9104"], data=data, home=HOME, assign=True).parent
    copied = read_json(ModelRecord, DataLayout(batch / "data").model("m9104"))
    assert copied.recipe.placement == "axon-cpu"
    assert read_json(ModelRecord, data.model("m9104")).recipe.placement is None
    assert read_json(ModelRecord, data.model("m9104")).recipe.rev == copied.recipe.rev


def test_an_ended_job_without_done_is_collected_and_its_lines_reported(
        data: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import written

    batch = cluster.axon_batch("cpu", ["m9101"], data=data, home=HOME).parent
    _finished(batch, data)
    (batch / "remote" / "DONE").unlink(missing_ok=True)
    monkeypatch.setattr(cluster, "_ssh", lambda cmd, **kw: "")

    def rsync(src: str, dest: str, *extra: str) -> None:
        if "/logs/" not in src:
            shutil.copytree(batch / "remote", dest, dirs_exist_ok=True)

    monkeypatch.setattr(cluster, "_rsync", rsync)
    monkeypatch.setenv("MENAGERIE_STORE_REMOTE", "worker:store")
    monkeypatch.setattr(cluster, "_relay_blobs", lambda rels, source, remote: list(rels))
    with written.recording() as rec:
        out = cluster.collect(batch, data=data, ended=True)
    assert out["done"] == "ended without DONE" and out["added"]["ledger/2026-10.jsonl"] == 1
    assert len(rec.lines) == 2  # the ledger and store-index lines, for the tick to commit
