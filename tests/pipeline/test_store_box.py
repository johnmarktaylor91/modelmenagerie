"""Daily store-box batches: listing, planning, hard-link batches and finishing collected ones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pytest

from menagerie.pipeline import store_box

A, B, C = "a" * 64, "b" * 64, "c" * 64


def make_store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    (store / "sha256/aa").mkdir(parents=True)
    (store / "sha256/aa" / A).write_bytes(b"12345")
    tree = store / "sha256/bb" / B
    (tree / "sub").mkdir(parents=True)
    (tree / "x.json").write_text("{}")
    (tree / "sub/y.bin").write_bytes(b"1234")
    (store / "sha256/bb/.tmp-inflight").mkdir()
    return store


def test_live_blobs_counts_files_and_skips_in_flight(tmp_path):
    blobs = store_box.live_blobs(make_store(tmp_path))
    assert set(blobs) == {f"sha256/aa/{A}", f"sha256/bb/{B}"}
    assert blobs[f"sha256/bb/{B}"] == {"files": 2, "bytes": 6, "symlinks": 0, "viewable": 2}


def test_plan_batches_splits_on_viewable_cap_and_holds_back_what_does_not_fit():
    cap = store_box.VIEW_BATCH_CAP
    new = {"k1": {"bytes": 10, "viewable": cap - 1}, "k2": {"bytes": 10, "viewable": 2},
           "k3": {"bytes": 10, "viewable": cap + 1}, "k4": {"bytes": 100, "viewable": 0}}
    batches, waiting = store_box.plan_batches(new, room=30)
    assert batches == [["k1"], ["k2"]]
    assert waiting == ["k3", "k4"]



def test_plan_batches_splits_on_the_byte_cap_so_one_restore_at_a_time_stays_small(monkeypatch):
    monkeypatch.setattr(store_box, "BATCH_BYTES_CAP", 25)
    new = {"k1": {"bytes": 10, "viewable": 0}, "k2": {"bytes": 10, "viewable": 0},
           "k3": {"bytes": 10, "viewable": 0}, "k4": {"bytes": 40, "viewable": 0},
           "k5": {"bytes": 5, "viewable": 0}}
    batches, waiting = store_box.plan_batches(new, room=100)
    # a blob over the cap goes alone; the day's total still stops at the room
    assert batches == [["k1", "k2"], ["k3"], ["k4"], ["k5"]]
    assert waiting == []
    batches, waiting = store_box.plan_batches(new, room=30)
    assert batches == [["k1", "k2"], ["k3"]]
    assert waiting == ["k4", "k5"]

def test_oversize_names_only_single_archive_files_over_the_fleet_cap():
    cap = store_box.ARCHIVE_FILE_CAP
    new = {"sha256/aa/big.tar.gz": {"files": 1, "bytes": cap + 1},
           "sha256/bb/ok.tar.gz": {"files": 1, "bytes": cap},
           "sha256/cc/rawtree": {"files": 40, "bytes": 5 * cap},
           "sha256/dd/plainblob": {"files": 1, "bytes": 2 * cap}}
    assert store_box.oversize(new) == ["sha256/aa/big.tar.gz"]


def test_link_batch_hard_links_and_refuses_reuse(tmp_path):
    store = make_store(tmp_path)
    dest = tmp_path / "box/daily-1"
    store_box.link_batch(store, dest, [f"sha256/aa/{A}", f"sha256/bb/{B}"])
    assert os.path.samefile(dest / "payload/sha256/aa" / A, store / "sha256/aa" / A)
    assert os.path.samefile(dest / "payload/sha256/bb" / B / "sub/y.bin", store / "sha256/bb" / B / "sub/y.bin")
    with pytest.raises(FileExistsError):
        store_box.link_batch(store, dest, [f"sha256/aa/{A}"])


def test_boxed_counts_only_sealed_batches(tmp_path):
    store = make_store(tmp_path)
    root = tmp_path / "box"
    store_box.link_batch(store, root / "sealed", [f"sha256/aa/{A}"])
    (root / "sealed/READY.json").write_text("{}")
    store_box.link_batch(store, root / "unsealed", [f"sha256/bb/{B}"])
    assert store_box.boxed(root) == {f"sha256/aa/{A}"}


def test_verification_root_matches_the_collector_manifest_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = Path("/srv/w/scratch/menagerie/store-box")
    ident = hashlib.sha256(b"worker|/srv/w/scratch/menagerie/store-box|daily-1").hexdigest()[:32]
    assert store_box.verification_root("worker", root, "daily-1").name == ident


def test_finish_deletes_restore_writes_marker_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "box"
    (root / "daily-1").mkdir(parents=True)
    restore = store_box.verification_root("worker", root, "daily-1")
    (restore / "x").mkdir(parents=True)
    assert store_box.finish("worker", root, "daily-1", C, "2026-10-08T00:00:00Z")["restore_deleted"]
    marker = json.loads((root / "daily-1" / store_box.MARKER).read_text())
    assert marker["snapshot"] == C and not restore.exists()
    assert not store_box.finish("worker", root, "daily-1", C, "2026-10-08T00:00:00Z")["restore_deleted"]


@pytest.mark.parametrize("value", ["full-1=" + C + "@t", "daily-1=abc@t", "daily-1=" + C])
def test_parse_finish_rejects_malformed(value):
    with pytest.raises(ValueError):
        store_box.parse_finish(value)


def test_worker_lists_every_blob_and_seals_only_new_ones(tmp_path, monkeypatch):
    home = tmp_path / "home"
    store = make_store(home / "data")
    root = home / "scratch/menagerie/store-box"
    store_box.link_batch(store, root / "full-1", [f"sha256/aa/{A}"])
    (root / "full-1/READY.json").write_text("{}")
    tool = home / ".local/bin/fleet-checkpoint"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\n[ \"$1\" = --content-addressed ] || exit 9\ntouch \"$2/READY.json\"\n")
    tool.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MENAGERIE_STORE", str(store))
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(host="worker", date="20261008", floor_gb=0.0, build=True, finish=[])
    monkeypatch.setattr(store_box, "MARGIN", 0)
    assert store_box.worker(args) == 0
    listing = json.loads((tmp_path / "outputs/listing.json").read_text())
    assert listing["complete"] and listing["roots"] == ["sha256"]
    assert [b["blob"] for b in listing["blobs"]] == [f"sha256/aa/{A}", f"sha256/bb/{B}"]
    summary = json.loads((tmp_path / "outputs/summary.json").read_text())
    assert [s["batch"] for s in summary["sealed"]] == ["daily-20261008-1"]
    assert (root / "daily-20261008-1/READY.json").is_file()
    assert (root / "daily-20261008-1/payload/sha256/bb" / B / "x.json").is_file()
    assert not (root / "daily-20261008-1/payload/sha256/aa").exists()


def _daily_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(repo=str(tmp_path / "repo"), desk=str(tmp_path / "desk"), floor_gb=1.0,
                              wait_hours=0.0, trim=True, ssh="w-hub")


@pytest.mark.parametrize("verified,pending", [
    (["daily-1", "daily-2"], ["daily-1", "daily-2"]),
    (["daily-1"], ["daily-1", "daily-2"]),
    (["daily-1", "daily-2"], ["daily-0", "daily-1", "daily-2"])])  # an older batch still unfinished
def test_daily_trims_only_after_every_new_batch_verifies(tmp_path, monkeypatch, verified, pending):
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({
        "blobs": 2, "new_blobs": 2, "waiting": 0, "listed_at": "t", "pending": pending,
        "sealed": [{"batch": "daily-1"}, {"batch": "daily-2"}]}))
    (out / "listing.json").write_text("{}")
    trims, workers = [], []
    monkeypatch.setattr(store_box, "job_host", lambda desk: "w")
    monkeypatch.setattr(store_box, "run_worker", lambda *a: workers.append(a[2]) or out)
    monkeypatch.setattr(store_box, "wait_collected", lambda repo, batches, hours, state: {
        b: {"snapshot": C, "verified_at": "t"} for b in verified})
    monkeypatch.setattr(store_box, "install_listing", lambda desk, host, f: tmp_path / "l.json")
    monkeypatch.setattr(store_box, "trim", lambda *a: trims.append(a) or 0)
    monkeypatch.setattr(store_box, "marked", lambda ssh, batches: {})
    rc = store_box.daily(_daily_args(tmp_path))
    log = (tmp_path / "repo/outputs/store-box-daily/daily.log").read_text()
    if set(pending) <= set(verified):
        assert rc == 0 and len(trims) == 1 and trims[0][1:4] == ("w", trims[0][2], "w-hub")
    elif len(verified) == 1:  # today's batch did not verify: no listing, no trim
        assert rc == 1 and trims == [] and len(workers) == 1
        assert "trim refused: batches ['daily-2'] are not verified" in log
    else:  # today's verified, an older one is still with the collector: listing, no trim
        assert rc == 0 and trims == [] and len(workers) == 2
        assert "trim refused: batches ['daily-0'] are not verified" in log


@pytest.mark.parametrize("late", ["receipt", "marker", "neither"])
def test_an_older_batch_verified_while_the_run_waited_does_not_block_the_trim(tmp_path, monkeypatch,
                                                                              late):
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({
        "blobs": 2, "new_blobs": 2, "waiting": 0, "listed_at": "t", "pending": ["daily-0", "daily-1"],
        "sealed": [{"batch": "daily-1"}]}))
    (out / "listing.json").write_text("{}")
    if late == "receipt":  # the collector archived daily-0 after this run's worker step listed it
        receipt = tmp_path / "repo/outputs" / store_box.JOB / "daily-0" / store_box.MARKER
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({"archive": {"snapshot": C},
                                       "verification": {"status": "full_restore_sha256", "at": "t"}}))
    trims, workers, asked = [], [], []
    monkeypatch.setattr(store_box, "job_host", lambda desk: "w")
    monkeypatch.setattr(store_box, "run_worker", lambda *a: workers.append(a[3]) or out)
    monkeypatch.setattr(store_box, "wait_collected", lambda repo, batches, hours, state: {
        "daily-1": {"snapshot": C, "verified_at": "t"}})
    monkeypatch.setattr(store_box, "install_listing", lambda desk, host, f: tmp_path / "l.json")
    monkeypatch.setattr(store_box, "trim", lambda *a: trims.append(a) or 0)
    monkeypatch.setattr(store_box, "marked", lambda ssh, batches: asked.append(batches) or (
        {b: {"marker": True} for b in batches} if late == "marker" else {}))
    rc = store_box.daily(_daily_args(tmp_path))
    log = (tmp_path / "repo/outputs/store-box-daily/daily.log").read_text()
    assert rc == 0 and len(workers) == 2
    if late == "receipt":  # finished on the worker with today's batch, no worker check needed
        assert f"daily-0={C}@t" in workers[1] and asked == [[]] and len(trims) == 1
    elif late == "marker":
        assert asked == [["daily-0"]] and len(trims) == 1
    else:
        assert trims == [] and "trim refused: batches ['daily-0'] are not verified" in log


def test_marked_reads_only_full_restore_markers(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return argparse.Namespace(stdout=f"{store_box.BOX_ROOT}/daily-0/{store_box.MARKER}\n",
                                  returncode=2)  # grep exits 2 when another marker is missing

    monkeypatch.setattr(store_box.subprocess, "run", run)
    assert store_box.marked("w-hub", []) == {} and calls == []
    assert store_box.marked("w-hub", ["daily-0", "daily-9"]) == {"daily-0": {"marker": True}}
    assert calls[0][:2] == ["ssh", "w-hub"] and "full_restore_sha256" in calls[0][2]


def test_pending_lists_sealed_batches_without_a_published_marker(tmp_path):
    for name, files in {"a": ["READY.json", store_box.MARKER], "b": ["READY.json"], "c": []}.items():
        (tmp_path / name).mkdir()
        for f in files:
            (tmp_path / name / f).write_text("{}")
    assert store_box.pending(tmp_path) == ["b"]


def test_a_failed_trim_job_still_commits_its_journal(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(store_box, "reconcile", lambda *a: calls.append("reconcile") or 3)
    monkeypatch.setattr(store_box, "sync_catalog", lambda *a: 4)

    def boom(*a, **k):
        calls.append("run")
        raise RuntimeError("pull-outputs failed")

    monkeypatch.setattr(store_box, "run_job", boom)
    assert store_box.trim(tmp_path, "w", "202610121400", "w-hub", tmp_path) == 1
    assert calls == ["reconcile", "run", "reconcile"]
    assert "failed before it finished: pull-outputs failed" in (tmp_path / "daily.log").read_text()


def test_trim_gate_needs_a_receipt_for_every_sealed_batch(tmp_path):
    assert store_box.trim_gate([], {}, tmp_path)  # nothing new today: earlier batches still prove
    assert not store_box.trim_gate(["daily-1"], {}, tmp_path)


def _git(repo: Path, *args: str) -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


def test_commit_placement_appends_the_trim_lines_and_commits_that_file_alone(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    shard = repo / store_box.PLACEMENT
    shard.parent.mkdir(parents=True)
    shard.write_text('{"old": 1}\n')
    (repo / "other.txt").write_text("x")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    (repo / "other.txt").write_text("someone else's edit")
    tool = tmp_path / "home/desk/bin/git-commit-paths"  # the fleet's named-path commit
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\n# -C REPO -m MSG -- PATH\ngit -C \"$2\" -c user.email=t@t -c user.name=t "
                    "commit -q --only -m \"$4\" -- \"$6\"\n")
    tool.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    journals = ['{"a": 1}\n{"old": 1}\n', '{"p": 1}\n{"a": 1}\n']  # a rerun repeats lines
    assert store_box.commit_placement(repo, journals, "202610121400", tmp_path) == 2
    assert shard.read_text() == '{"old": 1}\n{"a": 1}\n{"p": 1}\n'
    assert "2 lines from the daily trim" in _git(repo, "log", "-1", "--format=%s")
    assert _git(repo, "status", "--porcelain") == " M other.txt\n"  # left alone
    assert store_box.commit_placement(repo, journals, "202610121401", tmp_path) == 0  # all held
    before = shard.read_text()
    tool.write_text("#!/bin/sh\nexit 1\n")  # a commit that fails leaves the file as committed
    assert store_box.commit_placement(repo, ['{"q": 1}\n'], "202610121402", tmp_path) == -1
    assert shard.read_text() == before and _git(repo, "status", "--porcelain") == " M other.txt\n"
    shard.write_text(before + "uncommitted\n")  # someone's pending edit: refuse, even with nothing new
    assert store_box.commit_placement(repo, journals, "202610121403", tmp_path) == -1
    assert shard.read_text().endswith("uncommitted\n")


def test_reconcile_deletes_only_journals_of_jobs_that_ended(tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    (worker / store_box.TRIM_JOURNAL).mkdir(parents=True)
    for job in ("menagerie-storetrim-1", "menagerie-storetrim-2"):
        (worker / store_box.TRIM_JOURNAL / f"{job}.placement.jsonl").write_text(f'{{"{job}": 1}}\n')
    real = store_box.subprocess.run

    def run(argv, **kw):  # ssh and rsync against the fake worker home
        if argv[0] == "ssh":
            return real(["sh", "-c", f"cd {worker} && {argv[2]}"], **kw)
        if argv[0] == "rsync":
            return real([*argv[:-2], argv[-2].replace("w-hub:", f"{worker}/"), argv[-1]], **kw)
        return real(argv, **kw)

    monkeypatch.setattr(store_box.subprocess, "run", run)
    done = {"menagerie-storetrim-1"}  # job 2 still runs when the journals are listed
    monkeypatch.setattr(store_box, "ended", lambda job: job in done)
    texts = []

    def commit(repo, t, *a):  # job 2 appends a last line and ends after the copy
        texts.extend(t)
        with (worker / store_box.TRIM_JOURNAL / "menagerie-storetrim-2.placement.jsonl").open("a") as f:
            f.write('{"late": 1}\n')
        done.add("menagerie-storetrim-2")
        return 2

    monkeypatch.setattr(store_box, "commit_placement", commit)
    state = tmp_path / "state"
    state.mkdir()
    assert store_box.reconcile(tmp_path, "w-hub", "s", state) == 2
    assert len(texts) == 2  # every line copied is committed
    left = sorted(p.name for p in (worker / store_box.TRIM_JOURNAL).iterdir())
    assert left == ["menagerie-storetrim-2.placement.jsonl"]  # open at the copy: it stays
    assert store_box.reconcile(tmp_path, "w-hub", "s", state) == 2  # the next run takes the rest
    assert '{"late": 1}' in texts[-1] and not list((worker / store_box.TRIM_JOURNAL).iterdir())


def test_the_worker_place_script_checks_a_packed_restore_by_its_stream(tmp_path):
    import subprocess

    from menagerie.pipeline import store, tree_sha256

    src = tmp_path / "src"
    (src / "blobs").mkdir(parents=True)
    (src / "manifest.json").write_text("{}")
    (src / "blobs/a").write_bytes(b"1" * 300)
    sha = tree_sha256(src)
    root = tmp_path / "root"
    (root / "sha256" / sha[:2]).mkdir(parents=True)
    store.pack_tree(src, root / "sha256" / sha[:2] / f".restore-{sha}.tar.gz")
    done = subprocess.run(["python3", "-", str(root), f"{sha}.tar.gz"], input=store_box.TREE_SHA,
                          check=True, capture_output=True, text=True)
    assert json.loads(done.stdout) == {f"{sha}.tar.gz": sha}
    assert (root / "sha256" / sha[:2] / f"{sha}.tar.gz").is_file()
