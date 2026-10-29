"""The store size ceiling: offload the oldest box-verified blobs outside the working set."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import pytest
from test_store_trim import HOST, World

from menagerie.pipeline import read_jsonl, store, store_box, store_ceiling, tree_sha256
from menagerie.pipeline.records import StoreIndexLine

LATER = time.time() + 30 * 86400  # every blob put now is past the working-set age


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MENAGERIE_STORE_PACK", "0")
    return World(tmp_path)


def _ceiling(w: World, target: int, *, apply: bool = True, now: float = LATER,
             working: tuple[str, ...] = ()) -> dict:
    return store_ceiling.main(store=w.store, data=w.data, catalog=w.catalog, host=HOST,
                              ceiling=store_ceiling.Ceiling(target, 3 * 86400, working),
                              links=[w.links], apply=apply, now=now)


def _total(w: World) -> int:
    return sum(b.disk for b in store_ceiling.local_blobs(w.store).values())


def _disk(w: World, sha: str) -> int:
    return store_ceiling.local_blobs(w.store)[sha].disk


def _pdf(w: World, body: str, extracted: bool = True) -> str:
    sha = w.file(f"%PDF-1.4 {body}", "application/pdf")
    if extracted:
        text = w.tmp / "src" / f"{sha}.txt"
        text.write_text(body)
        store.put(text, store=w.store, data=w.data, mime="text/plain; charset=utf-8",
                  license_class="third-party", derived_from=sha)
    return sha


def test_under_the_ceiling_nothing_moves(world: World) -> None:
    sha = world.tree("a")
    world.seal([sha])
    summary = _ceiling(world, _total(world))
    assert summary["done"] == 0 and summary["candidates"] == 0
    assert world.store.blob(sha).is_dir()


def test_the_oldest_verified_blob_goes_first_and_only_until_under_the_target(world: World) -> None:
    old, new = world.tree("old"), world.tree("new")
    sid = world.seal([old])
    world.seal([new])  # linked later, so its ctime is newer
    summary = _ceiling(world, _total(world) - _disk(world, old))
    assert [item["sha256"] for item in summary["items"]] == [old]
    assert summary["freed_disk_bytes"] == summary["local_before"] - summary["local_after"] > 0
    assert summary["short_of_target"] == 0
    assert not world.store.blob(old).exists() and world.store.blob(new).is_dir()
    assert not (world.links / "full-1/payload/sha256" / old[:2] / old).exists()
    placed = read_jsonl(StoreIndexLine, world.data.root / "store-index" / "placement.jsonl")
    assert [(line.sha256, line.location) for line in placed] == [(old, f"box:{sid}")]
    assert store_ceiling.load_index(world.data).offloaded == {old}
    with pytest.raises(store.BlobOffloaded, match=f"store fetch {old}"):
        store.get(old, store=world.store, data=world.data)


def test_a_report_touches_nothing(world: World) -> None:
    sha = world.tree("a")
    world.seal([sha])
    summary = _ceiling(world, 0, apply=False)
    assert summary["done"] == 1 and world.store.blob(sha).is_dir()
    assert not (world.data.root / "store-index" / "placement.jsonl").exists()


def test_the_working_set_stays(world: World, tmp_path: Path) -> None:
    queued, young = world.tree("queued"), world.tree("young")
    world.seal([queued, young])
    spool = tmp_path / "spool-B" / "new"
    spool.mkdir(parents=True)
    (spool / "r1.json").write_text(f'{{"trace": "{queued}"}}')
    summary = _ceiling(world, 0, now=time.time() + 3600,  # young is an hour old
                       working=(str(tmp_path / "spool-*" / "new" / "*.json"),))
    assert summary["done"] == 0 and summary["working_set"] == 1
    assert summary["short_of_target"] == summary["local_before"]
    assert world.store.blob(queued).is_dir() and world.store.blob(young).is_dir()


def test_only_trace_saves_and_extracted_pdfs_are_offloadable(world: World) -> None:
    pdf, raw_pdf = _pdf(world, "paper"), _pdf(world, "unread", extracted=False)
    svg, snap = world.file("<svg/>"), world.file("snapshot", "application/gzip; menagerie=snapshot")
    world.seal([pdf, raw_pdf, svg, snap])
    summary = _ceiling(world, 0)
    assert summary["by_kind"] == {"pdf": 1}
    assert not world.store.blob(pdf).exists()
    for sha in (raw_pdf, svg, snap):
        assert world.store.blob(sha).is_file()


def test_an_unverified_blob_is_skipped_and_the_next_one_taken(world: World) -> None:
    first, second = world.tree("first"), world.tree("second")
    world.seal([first], status="sample_restore")
    world.seal([second])
    summary = _ceiling(world, _total(world) - 1)
    assert summary["skipped"] == {"not_in_verified_snapshot": 1}
    assert [item["sha256"] for item in summary["items"]] == [second]
    assert world.store.blob(first).is_dir()


def test_a_packed_save_offloads_and_restores_by_its_packed_name(world: World,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MENAGERIE_STORE_PACK")
    sha = world.tree("packed")
    assert store.packed_path(sha, store=world.store).is_file()
    sid = world.seal([sha])
    assert _ceiling(world, 0)["by_kind"] == {"trace": 1}
    assert not store.packed_path(sha, store=world.store).exists()
    assert store_box.snapshots_of([sha], world.data.root) == {sid: [sha + ".tar.gz"]}


def test_materialize_fetches_an_offloaded_blob_back_on_the_hub(world: World,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    sha = world.tree("fetched")
    keep = world.tmp / "box-copy"
    import shutil
    shutil.copytree(world.store.blob(sha), keep)
    shutil.rmtree(world.store.blob(sha))
    remote = world.tmp / "worker-store"
    monkeypatch.setenv("MENAGERIE_STORE_REMOTE", str(remote))
    asked = []

    def restore_blobs(shas: list[str], **kw: object) -> dict:
        asked.append((shas, kw["remote"]))
        shutil.copytree(keep, remote / "sha256" / sha[:2] / sha)
        return {"restored": shas, "failed": []}

    monkeypatch.setattr(store_box, "restore_blobs", restore_blobs)
    monkeypatch.setattr(store, "can_fetch", lambda: False)
    with pytest.raises(store.BlobOffloaded, match="store fetch"):
        store.materialize(sha, world.tmp / "out0", store=world.store)
    monkeypatch.setattr(store, "can_fetch", lambda: True)
    out = store.materialize(sha, world.tmp / "out", store=world.store)
    assert asked == [([sha], str(remote))] and tree_sha256(out) == sha



def test_the_default_working_set_age_is_one_day(world: World) -> None:
    old, young = world.tree("old"), world.tree("young")
    world.seal([old, young])
    def run(days: float, apply: bool) -> dict:
        return store_ceiling.main(store=world.store, data=world.data, catalog=world.catalog, host=HOST,
                                  ceiling=store_ceiling.Ceiling(0), links=[world.links], apply=apply,
                                  now=time.time() + days * 86400)
    assert run(0.5, apply=False)["done"] == 0  # half a day old: still the working set
    assert run(2, apply=True)["done"] == 2  # two days old: past the one-day working-set age
    assert not world.store.blob(old).exists() and not world.store.blob(young).exists()


def test_the_age_counts_from_the_fetch_unless_the_copy_came_back_from_the_box(world: World) -> None:
    kept, back = world.tree("kept"), world.tree("back")
    index = dataclasses.replace(store_ceiling.load_index(world.data), offloaded={back})
    now = store_ceiling._epoch(index.first[kept].fetched) + 2 * 86400
    relinked = {sha: store_ceiling.Local(1, now - 3600) for sha in (kept, back)}  # linked or fetched an hour ago
    picked = store_ceiling.select(relinked, index, set(), now=now, min_age=86400)
    assert [item.sha256 for item in picked] == [kept]


def test_the_trim_job_keeps_the_axon_render_attempts() -> None:
    job = Path(store_ceiling.__file__).with_name("store_trim_job.sh").read_text()
    assert '--working-set "$HOME/scratch/menagerie/axon/*/attempts/*/result.json"' in job

def test_the_daily_trim_ends_with_the_ceiling() -> None:
    assert store_box.TRIM_STEPS[-1] == "ceiling"


def test_a_failed_fetch_back_keeps_the_offloaded_error(world: World,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    sha = world.tree("lost")
    shutil.rmtree(world.store.blob(sha))
    monkeypatch.setenv("MENAGERIE_STORE_REMOTE", str(world.tmp / "worker-store"))
    monkeypatch.setattr(store, "can_fetch", lambda: True)

    def refused(shas: list[str], **kw: object) -> dict:
        raise SystemExit("no box location recorded")

    monkeypatch.setattr(store_box, "restore_blobs", refused)
    with pytest.raises(store.BlobOffloaded, match="store fetch"):
        store.materialize(sha, world.tmp / "out", store=world.store)
