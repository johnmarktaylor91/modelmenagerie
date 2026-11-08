"""The content-addressed store: put and get, index lines, text extraction, mirror and restore."""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, read_jsonl, store, tree_sha256
from menagerie.pipeline.records import StoreIndexLine

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = Path(__file__).parents[2] / "menagerie" / "data" / "routing_manifest.tsv"


@pytest.fixture()
def layout(tmp_path: Path) -> tuple[StoreLayout, DataLayout]:
    return StoreLayout(tmp_path / "store"), DataLayout(tmp_path / "data")


def _index(data: DataLayout) -> list[StoreIndexLine]:
    return [line for shard in sorted((data.root / "store-index").glob("*.jsonl"))
            for line in read_jsonl(StoreIndexLine, shard)]


def test_put_and_get_round_trip_a_real_file(layout: tuple) -> None:
    st, data = layout
    line = store.put(MANIFEST, store=st, data=data, mime="text/tab-separated-values",
                     license_class="own", url="menagerie/data/routing_manifest.tsv")
    assert line.sha256 == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert line.bytes == MANIFEST.stat().st_size
    assert store.get(line.sha256, store=st).read_bytes() == MANIFEST.read_bytes()
    # The same bytes again keep the one blob and the one index line.
    assert store.put(MANIFEST, store=st, data=data, mime="x", license_class="own") == line
    assert _index(data) == [line]
    # A new address for the same bytes gets its own line; a known address adds none.
    again = store.put(MANIFEST, store=st, data=data, mime="x", license_class="own",
                      url="https://example.org/routing_manifest.tsv")
    assert (again.sha256, again.url) == (line.sha256, "https://example.org/routing_manifest.tsv")
    assert store.put(MANIFEST, store=st, data=data, mime="x", license_class="own",
                     url="https://example.org/routing_manifest.tsv") == again
    assert [x.url for x in _index(data)] == [line.url, again.url]
    # The same bytes saved from another tree keep that tree as their source (one stock LICENSE).
    tree = "a" * 64
    copy = store.put(MANIFEST, store=st, data=data, mime="x", license_class="own",
                     url="https://example.org/routing_manifest.tsv", derived_from=tree)
    assert (copy.url, copy.derived_from) == (again.url, tree)
    assert store.put(MANIFEST, store=st, data=data, mime="x", license_class="own",
                     url="https://example.org/routing_manifest.tsv", derived_from=tree) == copy
    assert [x.derived_from for x in _index(data)] == [None, None, tree]


def test_a_tlspec_directory_is_one_blob(layout: tuple, tmp_path: Path) -> None:
    st, data = layout
    spec = tmp_path / "resnet18.tlspec"
    (spec / "parts").mkdir(parents=True)
    shutil.copy(FIXTURES / "ledger.jsonl", spec / "manifest.jsonl")
    shutil.copy(FIXTURES / "run_result.json", spec / "parts" / "result.json")
    line = store.put(spec, store=st, data=data, mime="application/x-tlspec",
                     license_class="own")
    blob = store.get(line.sha256, store=st)
    assert blob.is_dir() and tree_sha256(blob) == line.sha256
    assert line.bytes == sum(p.stat().st_size for p in spec.rglob("*") if p.is_file())


def test_get_of_a_missing_blob_raises(layout: tuple) -> None:
    with pytest.raises(FileNotFoundError):
        store.get("0" * 64, store=layout[0])


def test_html_text_drops_scripts_and_keeps_blocks(layout: tuple, tmp_path: Path) -> None:
    st, data = layout
    page = tmp_path / "abs.html"
    page.write_text("<html><head><title>t</title><script>var x=1;</script></head><body>"
                    "<h1>Deep Residual Learning</h1><blockquote>Deeper neural networks are"
                    " more <b>difficult</b>&nbsp;to train.</blockquote><style>p{}</style>"
                    "</body></html>", encoding="utf-8")
    source = store.put(page, store=st, data=data, mime="text/html; charset=utf-8",
                       license_class="third-party", url="https://example.org/abs")
    text = store.extract_text(source.sha256, store=st, data=data)
    assert text.derived_from == source.sha256 and text.url == source.url
    body = store.get(text.sha256, store=st).read_text(encoding="utf-8")
    assert body == "Deep Residual Learning\nDeeper neural networks are more difficult to train.\n"


@pytest.mark.skipif(not (shutil.which("dot") and shutil.which("pdftotext")),
                    reason="needs Graphviz and poppler")
def test_pdf_text_is_extracted(layout: tuple, tmp_path: Path) -> None:
    st, data = layout
    pdf = tmp_path / "graph.pdf"
    subprocess.run(["dot", "-Tpdf", "-o", str(pdf)], input=b'digraph { conv1 -> relu1 }',
                   check=True)
    source = store.put(pdf, store=st, data=data, mime="application/pdf",
                       license_class="own")
    text = store.extract_text(source.sha256, store=st, data=data)
    body = store.get(text.sha256, store=st).read_text(encoding="utf-8")
    assert "conv1" in body and "relu1" in body
    assert text.mime.startswith("text/plain")


def test_mirror_and_restore_keep_the_sha(layout: tuple, tmp_path: Path) -> None:
    st, data = layout
    lines = [store.put(p, store=st, data=data, mime="text/plain", license_class="own")
             for p in (MANIFEST, FIXTURES / "ledger.jsonl")]
    remote = tmp_path / "mirror"
    remote.mkdir()
    assert store.mirror(store=st, dest=str(remote)) == sorted(line.sha256 for line in lines)
    assert store.mirror(store=st, dest=str(remote)) == []  # --ignore-existing: nothing new
    fresh = StoreLayout(tmp_path / "fresh")
    restored = store.restore(lines[0].sha256, store=fresh, source=str(remote))
    assert restored == fresh.blob(lines[0].sha256)
    assert hashlib.sha256(restored.read_bytes()).hexdigest() == lines[0].sha256
    # A damaged mirror copy is refused, and nothing is placed.
    damaged = remote / "sha256" / lines[1].sha256[:2] / lines[1].sha256
    damaged.chmod(0o644)
    damaged.write_bytes(b"not the blob")
    with pytest.raises(ValueError, match="hashes to"):
        store.restore(lines[1].sha256, store=fresh, source=str(remote))
    assert not fresh.blob(lines[1].sha256).exists()


# --- Rung 1: snapshots, staging, drain and materialize ----------------------------------------

FILES = {"model.py": b"import torch\n", "LICENSE": b"MIT License\n", "cfg/a.yaml": b"x: 1\n"}


def _snapshot(st: StoreLayout, data: DataLayout) -> str:
    blob = st.root.parent / "snap.tgz"
    blob.write_bytes(store.pack_snapshot(FILES, {"url": "u", "rev": "r"}))
    return store.put(blob, store=st, data=data, mime=store.SNAPSHOT_MIME,
                     license_class="third-party").sha256


def test_snapshots_are_deterministic_and_carry_their_manifest(tmp_path: Path) -> None:
    one = store.pack_snapshot(FILES, {"url": "u", "rev": "r"})
    two = store.pack_snapshot(dict(reversed(FILES.items())), {"rev": "r", "url": "u"})
    assert one == two
    path = tmp_path / "s.tgz"
    path.write_bytes(one)
    assert store.snapshot_manifest(path) == {"url": "u", "rev": "r"}
    assert store.snapshot_files(path) == FILES
    (tmp_path / "plain").write_bytes(b"not a tar")
    assert store.snapshot_manifest(tmp_path / "plain") is None


def test_drain_moves_verified_blobs_and_keeps_a_bad_copy_staged(
        layout: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = layout
    remote = tmp_path / "worker"
    monkeypatch.setenv("MENAGERIE_STORE_REMOTE", str(remote))
    sha = _snapshot(st, data)
    other = tmp_path / "note.txt"
    other.write_text("a note\n")
    note = store.put(other, store=st, data=data, mime="text/plain", license_class="own").sha256
    bad = StoreLayout(remote).blob(note)
    bad.parent.mkdir(parents=True)
    bad.write_text("damaged\n")  # an earlier, broken copy at the worker
    with pytest.raises(ValueError, match="do not verify"):
        store.drain(store=st, data=data)
    assert not st.blob(sha).exists() and StoreLayout(remote).blob(sha).is_file()
    assert st.blob(note).is_file()  # kept staged, never deleted unverified
    bad.unlink()
    assert store.drain([note], store=st, data=data) == [note]
    assert store.drain(store=st, data=data) == []
    with pytest.raises(FileNotFoundError, match="not staged"):
        store.drain([note], store=st, data=data)


def test_materialize_round_trips_a_snapshot_through_the_worker_store(
        layout: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = layout
    monkeypatch.setenv("MENAGERIE_STORE_REMOTE", str(tmp_path / "worker"))
    sha = _snapshot(st, data)
    store.drain([sha], store=st, data=data)
    out = store.materialize(sha, tmp_path / "tree", store=st)
    got = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert got == FILES
    assert not (out / "model.py").stat().st_mode & 0o222  # read-only
    StoreLayout(tmp_path / "worker").blob(sha).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hashes to"):
        store.materialize(sha, tmp_path / "again", store=st)


def test_materialize_copies_a_plain_blob(layout: tuple, tmp_path: Path) -> None:
    st, data = layout
    line = store.put(MANIFEST, store=st, data=data, mime="text/plain", license_class="own")
    out = store.materialize(line.sha256, tmp_path / "copy.tsv", store=st)
    assert out.read_bytes() == MANIFEST.read_bytes()


def test_a_snapshot_member_cannot_escape_its_folder(tmp_path: Path) -> None:
    blob = tmp_path / "evil.tgz"
    blob.write_bytes(store.pack_snapshot({"../escape.py": b"x"}, {}))
    with pytest.raises(ValueError, match="escapes"):
        store._unpack(blob, tmp_path / "dest")
    assert not (tmp_path / "escape.py").exists()


def test_the_staging_store_refuses_past_its_cap(layout: tuple, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = layout
    _snapshot(st, data)
    store.check_room(st, 10**12)  # not staging: no cap
    monkeypatch.setenv("MENAGERIE_STORE_REMOTE", str(tmp_path / "worker"))
    monkeypatch.setenv("MENAGERIE_STAGE_CAP_GB", "0.0000001")  # about 107 bytes
    with pytest.raises(store.StoreFull, match="store drain"):
        store.check_room(st)


def test_pdf_text_runs_on_the_worker_host_when_this_machine_lacks_pdftotext(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kw: object) -> subprocess.CompletedProcess:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b"page text\n")

    monkeypatch.setattr(store.shutil, "which", lambda name: None)
    monkeypatch.setattr(store.subprocess, "run", run)
    pdf, out = tmp_path / "a.pdf", tmp_path / "a.txt"
    pdf.write_bytes(b"%PDF-1.5 body %%EOF")
    monkeypatch.delenv("MENAGERIE_STORE_REMOTE", raising=False)
    with pytest.raises(RuntimeError, match="MENAGERIE_STORE_REMOTE"):
        store._pdf_text(pdf, out)
    monkeypatch.setenv("MENAGERIE_STORE_REMOTE", "worker:data/store")
    store._pdf_text(pdf, out)
    assert calls == [["ssh", "worker", "pdftotext -enc UTF-8 - -"]]
    assert out.read_text() == "page text\n"


VIEW = b'{"nodes": [' + b'{"id": 1, "op": "conv2d"}, ' * 2000 + b'{}]}'


def test_a_view_file_is_written_gzipped_where_the_store_opts_in(
        layout: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = layout
    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MENAGERIE_STORE_PACK_FILES", "1")
    src = tmp_path / "view.json"
    src.write_bytes(VIEW)
    sha = hashlib.sha256(VIEW).hexdigest()
    line = store.put(src, store=st, data=data, mime="application/json", license_class="derived")
    gzipped = store.gzipped_path(sha, store=st)
    assert (line.sha256, line.bytes, line.encoding, line.location) == (
        sha, len(VIEW), "gzip", "local")
    assert gzipped.is_file() and not st.blob(sha).exists()
    assert gzipped.stat().st_size * 10 < len(VIEW)
    assert store.get(sha, store=st).read_bytes() == VIEW
    assert store.read_blob(sha, lambda p: p.read_bytes(), store=st) == VIEW
    # The same bytes again keep the one blob and the one index line.
    assert store.put(src, store=st, data=data, mime="application/json", license_class="d") == line
    assert _index(data) == [line]
    again = tmp_path / "again.gz"
    store.gzip_file(src, again)
    assert again.read_bytes() == gzipped.read_bytes()  # deterministic under any name
    # A hub reader pulls the gzipped form from the worker store and gets the original bytes.
    monkeypatch.setenv("MENAGERIE_STORE_REMOTE", str(st.root))
    hub = StoreLayout(tmp_path / "hub")
    assert store.materialize(sha, tmp_path / "out.json", store=hub).read_bytes() == VIEW


def test_view_files_stay_raw_without_the_opt_in_and_other_types_always(
        layout: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = layout
    monkeypatch.delenv("MENAGERIE_STORE_PACK_FILES", raising=False)
    src = tmp_path / "view.json"
    src.write_bytes(VIEW)
    line = store.put(src, store=st, data=data, mime="application/json", license_class="derived")
    assert st.blob(line.sha256).is_file() and line.encoding is None
    monkeypatch.setenv("MENAGERIE_STORE_PACK_FILES", "1")
    text = tmp_path / "log.txt"
    text.write_bytes(VIEW + b"\n")
    plain = store.put(text, store=st, data=data, mime="text/plain", license_class="derived")
    assert st.blob(plain.sha256).is_file() and plain.encoding is None
    monkeypatch.setenv("MENAGERIE_STORE_PACK", "0")  # the cluster's batch store
    svg = tmp_path / "graph.svg"
    svg.write_bytes(b"<svg>" + b"<g/>" * 500 + b"</svg>")
    drawn = store.put(svg, store=st, data=data, mime="image/svg+xml", license_class="derived")
    assert st.blob(drawn.sha256).is_file()


def test_a_gzipped_blob_that_unzips_wrong_is_refused(
        layout: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = layout
    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MENAGERIE_STORE_PACK_FILES", "1")
    src = tmp_path / "view.json"
    src.write_bytes(VIEW)
    line = store.put(src, store=st, data=data, mime="application/json", license_class="derived")
    other = tmp_path / "other.json"
    other.write_bytes(b'{"tampered": true}')
    store.gzip_file(other, store.gzipped_path(line.sha256, store=st))
    with pytest.raises(ValueError, match="unzips to"):
        store.get(line.sha256, store=st)


def test_trim_unpack_cache_removes_only_entries_past_the_lease(
        layout: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    st, data = layout
    cache = tmp_path / "cache"
    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE", str(cache))
    monkeypatch.setenv("MENAGERIE_STORE_PACK_FILES", "1")
    src = tmp_path / "view.json"
    src.write_bytes(VIEW)
    line = store.put(src, store=st, data=data, mime="application/json", license_class="derived")
    assert store.get(line.sha256, store=st).read_bytes() == VIEW  # gunzipped into the cache
    old_tree = cache / ("e" * 64)
    (old_tree / "sub").mkdir(parents=True)
    (old_tree / "sub" / "f").write_bytes(b"x" * 10)
    dead = cache / ".abcdef123456.tmp"  # an unpack that died
    dead.mkdir()
    now = (cache / line.sha256).stat().st_mtime + 60
    past = now - store.UNPACK_LEASE_SECONDS - 1
    for path in (old_tree, dead):
        os.utime(path, (past, past))
    report = store.trim_unpack_cache(now=now)
    assert report == {"entries": 3, "removed": 2, "freed_bytes": 10}
    assert sorted(p.name for p in cache.iterdir()) == [line.sha256]
    later = now + store.UNPACK_LEASE_SECONDS + 1
    assert store.trim_unpack_cache(now=later)["removed"] == 1
    assert store.get(line.sha256, store=st).read_bytes() == VIEW  # rebuilt on next use
    assert store.trim_unpack_cache(tmp_path / "absent") == {"entries": 0, "removed": 0,
                                                            "freed_bytes": 0}


def test_trace_metrics_reads_a_gzipped_summary(
        layout: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import trace_metrics
    from menagerie.pipeline.records import LedgerRow

    st, data = layout
    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MENAGERIE_STORE_PACK_FILES", "1")
    src = tmp_path / "summary.json"
    src.write_text('{"n_layers": 7}')
    sha = store.put(src, store=st, data=data, mime="application/json", license_class="derived").sha256
    row = LedgerRow(id="m1", stage="trace", at="2026-10-14T00:00:00Z", host="w", env="e",
                    recipe_rev="aaaaaaaaaaaa", outcome="pass", mode="eval", capture="full",
                    tl="2.35.3", trace="b" * 64, summary=sha)
    assert trace_metrics.summary(row, st) == {"n_layers": 7}
    assert trace_metrics.summary(dataclasses.replace(row, summary="c" * 64), st) is None


def test_a_signed_url_is_recorded_without_its_query(tmp_path):
    data, layout = DataLayout(tmp_path / "data"), StoreLayout(tmp_path / "store")
    doc = tmp_path / "paper.pdf"
    doc.write_bytes(b"%PDF-1.4 signed")
    signed = ("https://bucket.r2.example.com/pdfs/a/b.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256"
              "&X-Amz-Credential=KEYID%2F20261028%2Fauto%2Fs3%2Faws4_request&X-Amz-Signature=ab12")
    line = store.put(doc, store=layout, data=data, mime="application/pdf",
                         license_class="third-party", url=signed)
    assert line.url == "https://bucket.r2.example.com/pdfs/a/b.pdf"
    assert "X-Amz" not in data.store_index(line.fetched[:7]).read_text()
    plain = "https://arxiv.org/abs/1810.01367?context=cs"
    assert store.public_url(plain) == plain


def test_public_url_drops_each_signing_family_and_keeps_the_rest():
    base = "https://cdn.example.com/f.bin"
    cases = {
        "?X-Amz-Date=1&X-Amz-Signature=ab&X-Amz-Expires=60": "",  # S3 and R2 presigning
        "?X-Goog-Signature=ab&X-Goog-Date=1&alt=media": "?alt=media",  # Google signed URL
        "?xip=1&response-content-disposition=x&X-Xet-Cas-Uid=u&user_id=u&Expires=9&Policy=p&Signature=s"
        "&Key-Pair-Id=k": "?xip=1&response-content-disposition=x",  # CloudFront, with the HF account ids
        "?sv=1&st=2&se=3&sr=b&sp=r&sig=ab&rscd=x": "?rscd=x",  # Azure SAS
        "?token=ab&page=2": "?page=2",
        "?page=2": "?page=2",
        "?kid=ab12&meta=eyJhbGciOiJ.eyJzdWIiOiJ.c2lnbmF0dXJl&redirect_url=%2F": "?kid=ab12&redirect_url=%2F",
    }
    for query, kept in cases.items():
        assert store.public_url(base + query) == base + kept
    assert store.public_url(base + "?sig=ab#part") == base + "#part"


def test_public_line_rewrites_only_a_signed_url():
    signed = '{"schema":"store-index/1","sha256":"aa","url":"https://e.com/x.pdf?a=1&X-Amz-Signature=ab"}'
    assert store.public_line(signed) == '{"schema":"store-index/1","sha256":"aa","url":"https://e.com/x.pdf?a=1"}'
    plain = '{"sha256": "aa", "url": "https://e.com/x.pdf?a=1"}'
    assert store.public_line(plain) is plain
    assert store.public_line('{"old": 1}') == '{"old": 1}'


def test_a_packed_bundle_with_no_tensors_unpacks_with_its_empty_blobs_folder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE", str(tmp_path / "cache"))
    bundle = tmp_path / "bundle"
    (bundle / "blobs").mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"tensors": []}')
    (bundle / "metadata.pkl").write_bytes(b"meta")
    sha = tree_sha256(bundle)
    packed = tmp_path / "bundle.tar.gz"
    store.pack_tree(bundle, packed)
    path = store._unpacked(sha, packed)
    assert (path / "blobs").is_dir() and tree_sha256(path) == sha
    shutil.rmtree(path / "blobs")  # a tree unpacked before the fix, found in the cache
    assert (store._unpacked(sha, packed) / "blobs").is_dir()


def test_tree_sha256_streams_files_without_reading_them_whole(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A big row's save holds single files of about 2 GiB; hashing one whole killed the trim."""

    tree = tmp_path / "save"
    (tree / "blobs").mkdir(parents=True)
    big = (b"0123456789abcdef" * 200_000)  # over two 1 MiB chunks
    (tree / "blobs" / "0000000005.safetensors").write_bytes(big)
    (tree / "manifest.json").write_bytes(b"{}")
    lines = [f"{rel}\0{hashlib.sha256(body).hexdigest()}\n" for rel, body in
             (("blobs/0000000005.safetensors", big), ("manifest.json", b"{}"))]
    expected = hashlib.sha256("".join(lines).encode()).hexdigest()

    def refuse(self: Path) -> bytes:
        raise AssertionError(f"read whole: {self}")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    assert tree_sha256(tree) == expected
    assert tree_sha256(tree / "manifest.json") == hashlib.sha256(b"{}").hexdigest()
