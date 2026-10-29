"""Pack and offload behind verified box snapshots (rung 3 store gate): proof, selection, action."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    append_jsonl,
    read_jsonl,
    store,
    store_trim,
    tree_sha256,
)
from menagerie.pipeline.records import LedgerRow, RenderArtifact, RenderRow, StoreIndexLine

HOST = "worker-a"
REV = "aaaaaaaaaaaa"


class World:
    """A store, its committed data, a catalog folder and a box job root of sealed batches."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.store = StoreLayout(tmp / "store")
        self.data = DataLayout(tmp / "data")
        self.catalog = tmp / "catalog"
        self.links = tmp / "box"
        self.catalog.mkdir()
        (self.links / "full-1" / "payload").mkdir(parents=True)
        (self.links / "full-1" / "READY.json").write_text("{}")
        self.n = 0

    def tree(self, name: str, level: str = "runnable", structure: str = "s1",
             body: bytes = b"") -> str:
        src = self.tmp / "src" / name
        (src / "blobs").mkdir(parents=True)
        manifest = {"save_level": level, "n_layers": 3,
                    "provenance": {"input_hash": "i1", "model_structure_hash": structure}}
        (src / "manifest.json").write_text(json.dumps(manifest))
        (src / "blobs" / "a-b").write_bytes(name.encode() * 50 + body)
        (src / "blobs" / "a").mkdir()
        (src / "blobs" / "a" / "b").write_bytes(b"payload " * 200)  # a/b sorts before a-b
        return self.put(src, "application/x-tlspec")

    def file(self, text: str, mime: str = "image/svg+xml") -> str:
        path = self.tmp / "src" / f"f{self.n}"
        self.n += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return self.put(path, mime)

    def put(self, path: Path, mime: str) -> str:
        line = store.put(path, store=self.store, data=self.data, mime=mime, license_class="derived")
        return line.sha256

    def trace(self, model: str, at: str, sha: str, outcome: str = "pass", **kw) -> None:
        extra = {"cls": "torchlens_gap:x"} if outcome == "fail" else {}
        row = LedgerRow(id=model, stage="trace", at=at, host="w", env="e", recipe_rev=REV,
                        outcome=outcome, mode="eval", capture="full", tl="2.35.3", trace=sha,
                        **extra, **kw)
        append_jsonl([row], self.data.ledger("2026-10"))

    def render(self, model: str, at: str, trace: str, outcome: str = "pass",
               **kinds: str) -> None:
        arts = [RenderArtifact(kind=k, sha256=s, bytes=1,
                               variant=None if k in ("html", "netron") else "unrolled-none")
                for k, s in kinds.items()]
        row = RenderRow(id=model, at=at, trace=trace, renderer_rev="r", tl="2.35.3",
                        outcome=outcome, renders=tuple(arts))
        append_jsonl([row], self.data.ledger("2026-10"))

    def model(self, model: str) -> None:
        path = self.data.model(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": model, "recipe": {"rev": REV}}))

    def seal(self, shas: list[str], status: str = "full_restore_sha256", gz: bool = False) -> str:
        """Hard-link blobs into a sealed batch and write its catalog manifest (gzipped copy)."""
        files = []
        for sha in shas:
            src = self.store.blob(sha)
            if not src.exists():  # a packed blob: the batch holds its .tar.gz file
                src = store.packed_path(sha, store=self.store)
            out = self.links / "full-1" / "payload" / "sha256" / sha[:2] / src.name
            out.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file():
                os.link(src, out)
                files.append({"path": f"sha256/{sha[:2]}/{src.name}", "bytes": src.stat().st_size,
                              "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
                continue
            for rel, path in store.tree_files(src):
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                os.link(path, out / rel)
                files.append({"path": f"sha256/{sha[:2]}/{sha}/{rel}", "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        sid = hashlib.sha256(json.dumps(files).encode()).hexdigest()
        manifest = {"archive": {"snapshot": sid}, "created_by": {"host": HOST},
                    "verification": {"status": status}, "files": files}
        if gz:
            (self.catalog / f"{sid}.json.gz").write_bytes(gzip.compress(json.dumps(manifest).encode()))
        else:
            (self.catalog / f"{sid}.json").write_text(json.dumps(manifest))
        return sid

    def offload(self, policy: str, apply: bool = True) -> dict:
        return store_trim.main(policy, store=self.store, data=self.data, catalog=self.catalog,
                               host=HOST, links=[self.links], apply=apply)


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MENAGERIE_STORE_PACK", "0")  # saves written raw, as before pack on write
    return World(tmp_path)


@pytest.fixture()
def packing(world: World, monkeypatch: pytest.MonkeyPatch) -> World:
    """The same world with trace saves written packed (the default)."""
    monkeypatch.delenv("MENAGERIE_STORE_PACK")
    return world


def _history(w: World) -> dict[str, str]:
    """m1: an old trace with a portable save and renders, then the current trace and renders."""
    shas = {"old": w.tree("old")}
    shas["old_portable"] = w.tree("old-p", level="portable")
    shas["shared_thumb"] = w.file("thumb", "image/png")
    shas["old_svg"] = w.file("<svg>old</svg>")
    shas["cur"] = w.tree("cur")
    shas["cur_portable"] = w.tree("cur-p", level="portable")
    shas["cur_svg"] = w.file("<svg>cur</svg>")
    shas["tiles"] = w.tree("tiles")
    w.model("m1")
    w.trace("m1", "2026-10-01T00:00:00Z", shas["old"])
    w.render("m1", "2026-10-01T00:01:00Z", shas["old"], svg=shas["old_svg"], thumb=shas["shared_thumb"])
    w.trace("m1", "2026-10-02T00:00:00Z", shas["cur"])
    w.render("m1", "2026-10-02T00:01:00Z", shas["cur"], svg=shas["cur_svg"],
             thumb=shas["shared_thumb"], tiles=shas["tiles"])
    return shas


def test_a_packed_tree_hashes_like_the_raw_tree_and_reads_back_through_get(world: World) -> None:
    sha = world.tree("m")
    raw = world.store.blob(sha)
    out = store.packed_path(sha, store=world.store)
    store.pack_tree(raw, out)
    first = out.read_bytes()
    store.pack_tree(raw, out)
    assert out.read_bytes() == first  # deterministic
    assert store.packed_stats(out)[0] == tree_sha256(raw) == sha
    import shutil
    shutil.rmtree(raw)
    got = store.get(sha, store=world.store)
    assert got.is_dir() and tree_sha256(got) == sha
    assert store.materialize(sha, world.tmp / "copy", store=world.store) == world.tmp / "copy"
    assert tree_sha256(world.tmp / "copy") == sha


def test_offload_refuses_blobs_no_verified_snapshot_holds(world: World) -> None:
    shas = _history(world)
    world.seal([shas["old_svg"]], status="sample_restore")  # not a full restore check
    summary = world.offload("superseded")
    assert summary["done"] == 0
    assert summary["skipped"]["not_in_verified_snapshot"] >= 2
    assert world.store.blob(shas["old"]).is_dir() and world.store.blob(shas["old_svg"]).is_file()


def test_offload_superseded_drops_the_old_trace_its_portable_and_renders_only(world: World) -> None:
    shas = _history(world)
    sid = world.seal(list(shas.values()))
    dry = world.offload("superseded", apply=False)
    assert {i["sha256"] for i in dry["items"]} == {shas["old"], shas["old_portable"], shas["old_svg"]}
    assert world.store.blob(shas["old"]).exists()  # a report touches nothing
    summary = world.offload("superseded")
    assert summary["by_kind"] == {"trace": 1, "portable": 1, "render:svg": 1}
    for key in ("old", "old_portable", "old_svg"):
        assert not world.store.blob(shas[key]).exists()
        assert not (world.links / "full-1/payload/sha256" / shas[key][:2] / shas[key]).exists()
    for key in ("cur", "cur_portable", "shared_thumb", "cur_svg", "tiles"):
        assert world.store.blob(shas[key]).exists()
    placed = read_jsonl(StoreIndexLine, world.data.root / "store-index" / "placement.jsonl")
    assert {line.location for line in placed} == {f"box:{sid}"}
    with pytest.raises(store.BlobOffloaded, match=sid):
        store.get(shas["old"], store=world.store, data=world.data)
    assert world.offload("superseded")["done"] == 0  # a rerun finds nothing left


def test_a_failed_latest_trace_keeps_the_last_passing_save(world: World) -> None:
    world.model("m2")
    good, bad = world.tree("good"), world.tree("bad")
    world.trace("m2", "2026-10-01T00:00:00Z", good)
    world.trace("m2", "2026-10-02T00:00:00Z", bad, outcome="fail")
    world.seal([good, bad])
    assert world.offload("superseded")["done"] == 0


def test_tiles_and_rendered_portables(world: World) -> None:
    shas = _history(world)
    world.seal(list(shas.values()), gz=True)
    assert {i["sha256"] for i in world.offload("tiles")["items"]} == {shas["tiles"]}
    # Both traces drew; the superseded trace's portable save counts too.
    assert {i["sha256"] for i in world.offload("portable-rendered")["items"]} == {
        shas["cur_portable"], shas["old_portable"]}
    assert world.store.blob(shas["cur"]).exists() and world.store.blob(shas["cur_svg"]).exists()


def test_a_portable_whose_manifest_does_not_match_its_trace_stays(world: World) -> None:
    world.model("m3")
    trace = world.tree("t3")
    other = world.tree("p3", level="portable", structure="another-model")
    world.trace("m3", "2026-10-01T00:00:00Z", trace)
    world.render("m3", "2026-10-01T00:01:00Z", trace)
    world.seal([trace, other])
    summary = world.offload("portable-rendered")
    assert summary["done"] == 0 and summary["skipped"] == {"portable_pair_unconfirmed": 1}


def test_a_live_blob_that_reads_back_wrong_stops_the_run(world: World) -> None:
    shas = _history(world)
    world.seal(list(shas.values()))
    victim = world.store.blob(shas["old"]) / "blobs" / "a-b"
    data = bytearray(victim.read_bytes())
    data[0] ^= 1  # same size, different bytes
    os.unlink(victim)  # break the hard link so the sealed copy keeps its bytes
    victim.write_bytes(bytes(data))
    loaded = store_trim.load(world.data)
    first = {line.sha256: line for line in loaded.index}
    summary = store_trim.run(
        store_trim.select("superseded", loaded),
        store_trim.Run(world.store, store_trim.held_blobs(world.catalog, HOST, shas.values()),
                       first, (world.links,), True, world.tmp / "lines.jsonl"), pack=False)
    assert summary["stopped"] and "hashes to" in summary["stopped"]
    assert world.store.blob(shas["old"]).exists()


def test_pack_replaces_trace_saves_and_get_reads_them(world: World) -> None:
    shas = _history(world)
    world.seal(list(shas.values()))
    summary = world.offload("pack")
    assert {i["sha256"] for i in summary["items"]} == {shas["old"], shas["cur"]}
    for key in ("old", "cur"):
        assert not world.store.blob(shas[key]).exists()
        assert store.packed_path(shas[key], store=world.store).is_file()
        assert tree_sha256(store.get(shas[key], store=world.store)) == shas[key]
    placed = read_jsonl(StoreIndexLine, world.data.root / "store-index" / "placement.jsonl")
    assert {(line.encoding, line.location) for line in placed} == {("tar.gz", "local")}
    # A put of the same tree keeps the packed blob and adds no raw copy.
    store.put(world.tmp / "src" / "cur", store=world.store, data=world.data,
              mime="application/x-tlspec", license_class="derived")
    assert not world.store.blob(shas["cur"]).exists()


def test_get_names_an_absent_blob_as_offloaded(world: World) -> None:
    with pytest.raises(FileNotFoundError):
        store.get("0" * 64, store=world.store)


def test_stage_assets_reads_packed_blobs_and_reports_offloaded_ones(world: World) -> None:
    from menagerie.pipeline import publish

    shas = _history(world)
    world.seal(list(shas.values()))
    world.offload("pack")
    world.offload("superseded")
    export = world.tmp / "export"
    export.mkdir()
    (export / "assets.tsv").write_text("key\tsha256\tbytes\n"
                                       f"cur\t{shas['cur']}\t1\nold-svg\t{shas['old_svg']}\t1\n")
    missing = publish.stage_assets(export, store=world.store, dest=world.tmp / "site")
    assert missing == ["old-svg"] and tree_sha256(world.tmp / "site" / "cur") == shas["cur"]


def test_box_restore_groups_offloaded_blobs_by_snapshot(world: World, tmp_path: Path) -> None:
    from menagerie.pipeline import store_box

    shas = _history(world)
    sid = world.seal(list(shas.values()))
    world.offload("superseded")
    grouped = store_box.snapshots_of([shas["old"], shas["old_svg"]], world.data.root)
    assert grouped == {sid: sorted([shas["old"], shas["old_svg"]])}
    with pytest.raises(SystemExit, match="no box location"):
        store_box.snapshots_of([shas["cur"]], world.data.root)
    refs = tmp_path / "home" / ".claude/knowledge/secret-references.json"
    refs.parent.mkdir(parents=True)
    refs.write_text(json.dumps({"hetzner_storage_box": {"item": "ref:V/Box/"}}))
    env = store_box.box_env(tmp_path / "home" / "desk")
    assert env["HZ_KEY"] == "ref:V/Box/private_key" and set(env) == set(store_box.BOX_ENV)


def test_a_reader_that_loses_the_raw_path_reads_the_packed_form(world: World) -> None:
    import shutil

    sha = world.tree("busy")
    raw = world.store.blob(sha)
    store.pack_tree(raw, store.packed_path(sha, store=world.store))
    seen = []

    def read(path: Path) -> str:
        seen.append(path)
        if path == raw:  # a pack takes the raw tree away while this reader works
            shutil.rmtree(raw)
            raise FileNotFoundError(path)
        return tree_sha256(path)

    assert store.read_blob(sha, read, store=world.store) == sha and seen[0] == raw


def test_the_unpack_cache_keeps_leased_trees_past_its_cap(world: World,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE_GB", "0.000001")  # about 1 KB: every tree is over
    shas = [world.tree(name) for name in ("one", "two")]
    for sha in shas:
        store.pack_tree(world.store.blob(sha), store.packed_path(sha, store=world.store))
        shutil.rmtree(world.store.blob(sha))
    first = store.get(shas[0], store=world.store)
    store.get(shas[1], store=world.store)
    assert first.is_dir()  # just handed out: not evicted
    os.utime(first, (0, 0))  # its lease has run out
    store.get(shas[1], store=world.store)  # a cache hit evicts nothing
    (world.tmp / "cache" / shas[1]).rename(world.tmp / "cache" / f".old-{shas[1]}")
    store.get(shas[1], store=world.store)  # a fresh unpack makes room
    assert not first.exists()


def test_box_restore_takes_one_include_a_call_and_whole_prefixes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import store_box

    desk = tmp_path / "home" / "desk"
    (desk / "state/storage/catalog").mkdir(parents=True)
    (desk / "state/storage/catalog" / f"{'e' * 64}.json").write_text(json.dumps({"source": "/src/p"}))
    monkeypatch.setattr(store_box, "box_env", lambda desk: {})
    calls = []

    def run(argv, **kw):
        include = argv[argv.index("--include") + 1]
        calls.append(include)
        target = Path(argv[argv.index("--target") + 1]) / "src/p"
        rel = include.removeprefix("/src/p/")
        names = [f"{rel}/{'1' * 63}{i}" for i in range(9)] if rel.count("/") == 1 else [rel]
        for name in names:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).write_text("x")
        return subprocess.CompletedProcess(argv, 0, "", "")

    import subprocess
    monkeypatch.setattr(store_box.subprocess, "run", run)
    many = [f"{'1' * 63}{i}" for i in range(9)]
    payload = store_box.restore_batch(desk, "e" * 64, [*many, "2" * 64], tmp_path / "t")
    assert calls == ["/src/p/sha256/11", f"/src/p/sha256/22/{'2' * 64}"]
    assert sorted(p.name for p in payload.glob("sha256/*/*")) == sorted([*many, "2" * 64])


def _src_files(src: Path) -> dict[str, bytes]:
    return {p.relative_to(src).as_posix(): p.read_bytes() for p in src.rglob("*") if p.is_file()}


def test_a_trace_save_is_written_packed_and_reads_back_byte_identical(packing: World) -> None:
    w = packing
    sha = w.tree("fresh")
    src = w.tmp / "src" / "fresh"
    assert sha == tree_sha256(src)  # the name is the tree's, as when written raw
    packed = store.packed_path(sha, store=w.store)
    assert packed.is_file() and not w.store.blob(sha).exists()  # no raw tree is kept
    assert not [p for p in packed.parent.iterdir() if p.name.startswith(".")]  # no temp left
    store.pack_tree(src, w.tmp / "trim-form.tar.gz")
    assert packed.read_bytes() == (w.tmp / "trim-form.tar.gz").read_bytes()  # the trim's form
    got = store.get(sha, store=w.store)
    assert _src_files(got) == _src_files(src)  # byte-identical read back
    line = store.index_line(sha, data=w.data)
    assert line is not None and line.bytes == sum(len(b) for b in _src_files(src).values())
    assert (line.encoding, line.location) == ("tar.gz", "local")
    assert store.placement(sha, data=w.data) == line
    assert store.put(src, store=w.store, data=w.data, mime="application/x-tlspec",
                     license_class="derived") == line  # a second put keeps the blob and line
    svg = w.file("<svg/>")  # other blobs stay raw
    assert w.store.blob(svg).is_file() and store.index_line(svg, data=w.data).encoding is None


def test_an_opted_out_store_keeps_raw_trees(world: World) -> None:
    sha = world.tree("raw")
    assert world.store.blob(sha).is_dir()
    assert not store.packed_path(sha, store=world.store).exists()
    assert store.index_line(sha, data=world.data).encoding is None


def test_offload_takes_packed_saves_held_by_a_snapshot_of_the_packed_file(packing: World) -> None:
    w = packing
    shas = _history(w)
    sid = w.seal(list(shas.values()))
    assert w.offload("pack")["skipped"] == {"already_packed": 2}  # written packed: nothing to do
    summary = w.offload("superseded")
    assert summary["by_kind"] == {"trace": 1, "portable": 1, "render:svg": 1}
    for key in ("old", "old_portable", "old_svg"):
        sha = shas[key]
        assert not store.packed_path(sha, store=w.store).exists() and not w.store.blob(sha).exists()
        assert not list((w.links / "full-1/payload/sha256" / sha[:2]).glob(f"{sha}*"))
    placed = {line.sha256: line for line in
              read_jsonl(StoreIndexLine, w.data.root / "store-index" / "placement.jsonl")}
    assert (placed[shas["old"]].encoding, placed[shas["old"]].location) == ("tar.gz", f"box:{sid}")
    assert placed[shas["old_svg"]].encoding is None
    with pytest.raises(store.BlobOffloaded, match=sid):
        store.get(shas["old"], store=w.store, data=w.data)
    # The pair check reads both manifests from the packed streams.
    assert {i["sha256"] for i in w.offload("portable-rendered")["items"]} == {shas["cur_portable"]}
    assert tree_sha256(store.get(shas["cur"], store=w.store)) == shas["cur"]


def test_a_packed_save_boxed_only_as_a_raw_tree_offloads_as_raw(world: World,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _history(world)
    world.seal(list(shas.values()))  # raw trees boxed, then packed here
    world.offload("pack")
    summary = world.offload("superseded")
    assert {i["sha256"]: i["held"] for i in summary["items"] if i["kind"] == "trace"} == {
        shas["old"]: "raw"}
    placed = [line for line in read_jsonl(StoreIndexLine,
                                          world.data.root / "store-index" / "placement.jsonl")
              if line.sha256 == shas["old"]]
    assert placed[-1].encoding == "raw" and placed[-1].location.startswith("box:")


def test_a_packed_blob_that_reads_back_wrong_stops_the_run(packing: World) -> None:
    w = packing
    shas = _history(w)
    w.seal(list(shas.values()))
    packed = store.packed_path(shas["old"], store=w.store)
    os.unlink(packed)  # the sealed link keeps the good bytes
    store.pack_tree(w.tmp / "src" / "cur", packed)  # another tree under this name
    summary = w.offload("superseded")
    assert summary["stopped"] and "unpacks to" in summary["stopped"]
    assert packed.is_file()


def test_box_restore_names_the_packed_form_of_a_blob_offloaded_packed(packing: World) -> None:
    from menagerie.pipeline import store_box

    shas = _history(packing)
    sid = packing.seal(list(shas.values()))
    packing.offload("superseded")
    grouped = store_box.snapshots_of([shas["old"], shas["old_svg"]], packing.data.root)
    assert grouped == {sid: sorted([shas["old"] + ".tar.gz", shas["old_svg"]])}
    restored = packing.tmp / "restored.tar.gz"  # a restored packed copy checks by its stream
    store.pack_tree(packing.tmp / "src" / "old", restored)
    assert store_box.name_hash(restored) == shas["old"]


def test_pack_views_gzips_view_files_behind_a_verified_snapshot(world: World) -> None:
    shas = _history(world)
    world.model("m2")
    trace = world.tree("m2-trace")
    summary = world.file('{"layers": [' + '{"n": 1}, ' * 400 + "{}]}", "application/json")
    netron = world.file('{"graph": [' + '"node", ' * 400 + '""]}', "application/json")
    unsealed = world.file("<svg>" + "<g/>" * 400 + "</svg>")
    world.trace("m2", "2026-10-03T00:00:00Z", trace, summary=summary)
    world.render("m2", "2026-10-03T00:01:00Z", trace, svg=unsealed, netron=netron)
    views = [shas["old_svg"], shas["cur_svg"], summary, netron]
    world.seal([*views, shas["shared_thumb"], shas["cur"]])
    bodies = {sha: world.store.blob(sha).read_bytes() for sha in [*views, unsealed]}

    dry = world.offload("pack-views", apply=False)
    assert dry["done"] == 4 and all(world.store.blob(s).exists() for s in views)
    done = world.offload("pack-views")
    assert {i["sha256"] for i in done["items"]} == set(views)
    assert done["skipped"] == {"not_in_verified_snapshot": 1}
    for sha in views:
        assert not world.store.blob(sha).exists()
        assert store.gzipped_path(sha, store=world.store).is_file()
        assert not (world.links / "full-1" / "payload" / "sha256" / sha[:2] / sha).exists()
        assert store.get(sha, store=world.store).read_bytes() == bodies[sha]
    # The thumbnail (not a view type), the trace saves and the unsealed SVG stay raw.
    assert world.store.blob(shas["shared_thumb"]).is_file()
    assert world.store.blob(shas["cur"]).is_dir() and world.store.blob(unsealed).is_file()
    placed = read_jsonl(StoreIndexLine, world.data.root / "store-index" / "placement.jsonl")
    assert {(line.sha256, line.encoding, line.location) for line in placed} == {
        (sha, "gzip", "local") for sha in views}
    assert world.offload("pack-views")["skipped"] == {"already_packed": 4,
                                                      "not_in_verified_snapshot": 1}
    # A gzipped view file is never offloaded: the superseded policy keeps the old SVG.
    assert world.offload("superseded")["skipped"].get("gzipped_kept") == 1
    assert store.gzipped_path(shas["old_svg"], store=world.store).is_file()


def test_a_gzipped_view_file_that_reads_back_wrong_stops_the_run(world: World,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _history(world)
    world.seal([shas["old_svg"], shas["cur_svg"]])
    monkeypatch.setattr(store, "gzipped_stats", lambda path: ("0" * 64, 0))
    summary = world.offload("pack-views")
    assert summary["stopped"] and summary["done"] == 0
    assert world.store.blob(shas["old_svg"]).is_file() and world.store.blob(shas["cur_svg"]).is_file()


# --- portable-twin: a portable save after a trace save that is itself portable ---------------

def _twins(w: World, model: str = "m20100", structure: str = "s1",
           runnable: str | None = "ready") -> dict[str, str]:
    """A refused runnable save (``trace.tlspec`` portable) and the twin put right after it,
    before 2026-10-28; then an ordinary runnable save (its row says ``runnable``) and its
    portable save."""

    w.model(model)
    shas = {"trace": w.tree(f"{model}-t", level="portable"),
            "twin": w.tree(f"{model}-r", level="portable", structure=structure)}
    w.trace(model, "2026-10-28T00:00:00Z", shas["trace"], runnable="refused:missing_input")
    w.model("m2")
    shas["runnable"] = w.tree("m2-t")
    shas["portable"] = w.tree("m2-p", level="portable")
    w.trace("m2", "2026-10-28T00:01:00Z", shas["runnable"], runnable=runnable)
    return shas


@pytest.mark.parametrize("packed", [False, True])
def test_portable_twin_drops_only_a_twin_both_saves_of_which_are_boxed(
        world: World, monkeypatch: pytest.MonkeyPatch, packed: bool) -> None:
    if packed:
        monkeypatch.delenv("MENAGERIE_STORE_PACK")  # saves written packed, as on the worker
    shas = _twins(world)
    sid = world.seal(list(shas.values()))
    summary = world.offload(store_trim.TWIN)
    assert [(i["sha256"], i["kind"], i["pair"]) for i in summary["items"]] == [
        (shas["twin"], "portable-twin", shas["trace"])]  # the report lists what went
    assert summary["skipped"] == {}  # m2's row says ready: its pair is not even read
    assert not world.store.blob(shas["twin"]).exists()
    assert not store.packed_path(shas["twin"], store=world.store).exists()
    assert not list((world.links / "full-1/payload/sha256" / shas["twin"][:2]).glob(
        f"{shas['twin']}*"))
    for key in ("trace", "runnable", "portable"):  # the trace save stays for render to read
        assert tree_sha256(store.get(shas[key], store=world.store)) == shas[key]
    placed = read_jsonl(StoreIndexLine, world.data.root / "store-index" / "placement.jsonl")
    assert [(p.sha256, p.location) for p in placed] == [(shas["twin"], f"box:{sid}")]
    with pytest.raises(store.BlobOffloaded, match=sid):  # `store fetch` brings it back
        store.get(shas["twin"], store=world.store, data=world.data)
    assert world.offload(store_trim.TWIN)["done"] == 0


def test_portable_twin_keeps_a_twin_whose_pair_no_verified_snapshot_holds(world: World) -> None:
    shas = _twins(world)
    world.seal([shas["twin"]])
    summary = world.offload(store_trim.TWIN)
    assert summary["done"] == 0
    assert summary["skipped"]["pair_not_in_verified_snapshot"] == 1
    world.seal([shas["trace"]], status="sample_restore")  # not a full restore check
    assert world.offload(store_trim.TWIN)["skipped"]["pair_not_in_verified_snapshot"] == 1
    assert world.store.blob(shas["twin"]).is_dir()


def test_portable_twin_keeps_a_portable_save_of_another_capture(world: World) -> None:
    shas = _twins(world, structure="another-model")
    world.seal(list(shas.values()))
    summary = world.offload(store_trim.TWIN)
    assert summary["done"] == 0 and summary["skipped"] == {"portable_pair_unconfirmed": 1}
    assert world.store.blob(shas["twin"]).is_dir()


def test_portable_twin_checks_by_manifest_whatever_the_row_says(world: World) -> None:
    """A row without ``ready`` (an older row, a round-trip refusal of a runnable save) is
    checked by manifest: a runnable pair keeps its portable save."""

    shas = _twins(world, runnable="refused:round_trip_hash")
    world.seal(list(shas.values()))
    summary = world.offload(store_trim.TWIN)
    assert [i["sha256"] for i in summary["items"]] == [shas["twin"]]
    assert summary["skipped"] == {"portable_pair_unconfirmed": 1}
    assert world.store.blob(shas["portable"]).is_dir()


def test_portable_twin_keeps_a_twin_whose_pair_is_gone_locally(world: World) -> None:
    """The pair's manifest is what proves one capture; without it the twin stays."""

    import shutil

    shas = _twins(world)
    world.seal(list(shas.values()))
    shutil.rmtree(world.store.blob(shas["trace"]))
    assert world.offload(store_trim.TWIN)["skipped"] == {"portable_pair_unconfirmed": 1}


def test_portable_rendered_still_leaves_twins_to_their_own_step(world: World) -> None:
    shas = _twins(world)
    world.seal(list(shas.values()))
    world.render("m20100", "2026-10-28T00:02:00Z", shas["trace"])
    world.render("m2", "2026-10-28T00:02:00Z", shas["runnable"])
    summary = world.offload("portable-rendered")
    assert {i["sha256"] for i in summary["items"]} == {shas["portable"]}
    assert summary["skipped"] == {"portable_pair_unconfirmed": 1}


def test_the_daily_trim_runs_portable_twin_after_portable_rendered() -> None:
    from menagerie.pipeline import store_box

    steps = store_box.TRIM_STEPS
    assert steps.index("portable-twin") == steps.index("portable-rendered") + 1
    assert store_trim.TWIN in store_trim.POLICIES and store_trim.TWIN in store_trim.POLICY
    usage = (Path(store_trim.__file__).with_name("store_trim_job.sh")).read_text()
    assert "portable-rendered|portable-twin|pack" in usage
