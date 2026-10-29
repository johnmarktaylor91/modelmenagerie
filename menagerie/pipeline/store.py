"""Content-addressed store: put and get blobs, index lines, extracted text, the off-site mirror.

Owned by the data lane. Blobs live at ``StoreLayout.blob(sha256)``; every put appends one
``StoreIndexLine`` to the committed ``store-index/<yyyy-mm>.jsonl``. A ``.tlspec`` directory is
keyed by ``tree_sha256``. Blobs are immutable: a put of bytes the store already holds keeps
the existing blob and returns its first index line. The mirror is plain ``rsync`` of the
``sha256/`` tree with ``--ignore-existing``; ``dest`` and ``source`` are anything rsync takes
(a local directory or ``host:path``).

Rung 1 adds code snapshots (deterministic ``tar.gz`` trees with a manifest), the hub staging cap
and the move from hub staging to the worker store (``drain``) and back (``materialize``).
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import html.parser
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from menagerie.local_config import local_value
from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    append_jsonl,
    month_of,
    now,
    read_jsonl,
    tree_sha256,
)
from menagerie.pipeline.records import SHA256, StoreIndexLine

T = TypeVar("T")


def _bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _index_lines(data: DataLayout) -> list[StoreIndexLine]:
    folder = data.root / "store-index"
    return [line for shard in sorted(folder.glob("*.jsonl"))
            for line in read_jsonl(StoreIndexLine, shard)]


def index_line(sha256: str, *, data: DataLayout) -> StoreIndexLine | None:
    """Return the first index line recorded for a blob, or ``None``."""

    return next((line for line in _index_lines(data) if line.sha256 == sha256), None)


def _known(sha: str, url: str | None, derived_from: str | None,
           data: DataLayout) -> StoreIndexLine | None:
    """The index line ``put`` returns for a blob the store already holds, or ``None``.

    The same bytes from a new source (two Hub repos shipping one ``config.json``, two code trees
    holding one stock LICENSE) get their own line with that ``url`` and ``derived_from``,
    restating the blob's latest placement, so a lookup by address or by tree finds every source."""

    lines = [line for line in _index_lines(data) if line.sha256 == sha]

    def same(line: StoreIndexLine) -> bool:
        return line.derived_from == derived_from and url in (None, line.url)

    if not lines or (url is None and derived_from is None):
        return lines[0] if lines else None
    if any(map(same, lines)):
        return next(filter(same, lines))
    line = replace(lines[-1], url=url, derived_from=derived_from, fetched=now())
    append_jsonl([line], data.store_index(month_of(line.fetched)))
    return line


def _place(src: Path, dest: Path) -> None:
    """Copy ``src`` beside ``dest`` under a temporary name, then rename it into place."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name[:12]}.", dir=dest.parent)) / "blob"
    try:
        if src.is_dir():
            shutil.copytree(src, tmp)
        else:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def _place_packed(src: Path, sha256: str, out: Path) -> None:
    """Write a directory blob straight to its packed form ``out``: packed under a hidden name,
    read back from the stream (its tree must hash to ``sha256``), then renamed into place."""

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".pack-{sha256[:12]}.", dir=out.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        pack_tree(src, tmp)
        actual = packed_stats(tmp)[0]
        if actual != sha256:
            raise ValueError(f"packed form of {sha256} reads back as {actual}")
        os.replace(tmp, out)
        fsync(out.parent)
    finally:
        tmp.unlink(missing_ok=True)


def packs_on_write(path: Path, mime: str) -> bool:
    """Whether ``put`` stores this blob packed: a trace save directory, unless the store's
    reader needs raw trees (``MENAGERIE_STORE_PACK=0``, the cluster's batch store)."""

    return (mime in PACK_ON_WRITE and Path(path).is_dir()
            and os.environ.get("MENAGERIE_STORE_PACK", "1") != "0")


def gzips_on_write(path: Path, mime: str) -> bool:
    """Whether ``put`` stores this file blob gzipped: a view file (``GZIP_FILES``) in a store that
    opts in (``MENAGERIE_STORE_PACK_FILES=1``, the worker runner's store). The hub's staging store
    never does: ``drain`` moves raw names only."""

    return (mime in GZIP_FILES and Path(path).is_file()
            and os.environ.get("MENAGERIE_STORE_PACK", "1") != "0"
            and os.environ.get("MENAGERIE_STORE_PACK_FILES") == "1")


def _place_gzipped(src: Path, sha256: str, out: Path) -> None:
    """Write a file blob straight to its gzipped form ``out``: written under a hidden name, read
    back from the stream (its bytes must hash to ``sha256``), then renamed into place."""

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".gzip-{sha256[:12]}.", dir=out.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        gzip_file(src, tmp)
        actual = gzipped_stats(tmp)[0]
        if actual != sha256:
            raise ValueError(f"gzipped form of {sha256} reads back as {actual}")
        os.replace(tmp, out)
        fsync(out.parent)
    finally:
        tmp.unlink(missing_ok=True)


def put(path: Path, *, store: StoreLayout, data: DataLayout, mime: str, license_class: str,
        url: str | None = None, derived_from: str | None = None) -> StoreIndexLine:
    """Copy a file or ``.tlspec`` directory into the store, append its index line, return it.

    A trace save (``PACK_ON_WRITE``) is written packed at ``<sha>.tar.gz`` and no raw tree is
    kept; its name and index ``bytes`` are those of the tree, and ``get`` reads it back. A view
    file (``gzips_on_write``) is written gzipped at ``<sha>.gz`` the same way."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    sha = tree_sha256(path)
    dest, packed, gzipped = (store.blob(sha), packed_path(sha, store=store),
                             gzipped_path(sha, store=store))
    if dest.exists() or packed.is_file() or gzipped.is_file():
        known = _known(sha, url, derived_from, data)
        if known is not None:
            return known
    elif packs_on_write(path, mime):
        _place_packed(path, sha, packed)
    elif gzips_on_write(path, mime):
        _place_gzipped(path, sha, gzipped)
    else:
        _place(path, dest)
    form = ({} if dest.exists() else {"encoding": "gzip", "location": "local"} if gzipped.is_file()
            else {"encoding": "tar.gz", "location": "local"})
    line = StoreIndexLine(sha256=sha, bytes=_bytes(path), mime=mime, fetched=now(),
                          license_class=license_class, url=url, derived_from=derived_from,
                          **form)
    append_jsonl([line], data.store_index(month_of(line.fetched)))
    return line


def read_blob(sha256: str, read: Callable[[Path], T], *, store: StoreLayout) -> T:
    """``read(get(sha256))``, once more when the path vanished mid-read (a pack or offload took
    the raw blob away): the second ``get`` finds the packed form, or raises ``BlobOffloaded``."""

    try:
        return read(get(sha256, store=store))
    except FileNotFoundError:
        return read(get(sha256, store=store))


def get(sha256: str, *, store: StoreLayout, data: DataLayout | None = None) -> Path:
    """Return a readable path of a blob: the raw blob, else its packed (or gzipped) form unpacked
    into the capped unpack cache. Raises ``BlobOffloaded`` (a ``FileNotFoundError``) when neither is here;
    with ``data`` the error names the box snapshot that holds it (``store_box restore`` on the
    hub brings it back)."""

    path = store.blob(sha256)
    if path.exists():
        return path
    packed = packed_path(sha256, store=store)
    if packed.is_file():
        return _unpacked(sha256, packed)
    gzipped = gzipped_path(sha256, store=store)
    if gzipped.is_file():
        return _gunzipped(sha256, gzipped)
    line = placement(sha256, data=data) if data is not None else None
    where = line.location if line is not None else None
    snapshot = where[4:] if where and where.startswith("box:") else None
    raise BlobOffloaded(sha256, snapshot, store.root)


# --- Rung 3: packed blobs and offloaded blobs -----------------------------------------------
# A packed blob is a deterministic tar.gz of a directory blob's files at ``<sha>.tar.gz`` beside
# the raw path; its name is unchanged, because its unpacked tree hashes to the same sha. An
# offloaded blob is held only by a full-restore-verified box snapshot (``store_trim``).

PACKED_SUFFIX = ".tar.gz"
PACK_ON_WRITE = frozenset({"application/x-tlspec"})  # trace saves and their portable saves
# A gzipped blob is a deterministic gzip of a file blob's bytes at ``<sha>.gz`` beside the raw
# path (view JSON, SVG and HTML compress about 30 times); its name is unchanged, because its
# bytes hash to the same sha.
GZIP_SUFFIX = ".gz"
GZIP_FILES = frozenset({"application/json", "image/svg+xml", "text/html"})
UNPACK_CACHE_GB = 8.0
UNPACK_LEASE_SECONDS = 6 * 3600  # a tree used this recently is never evicted (soft cap)


class BlobOffloaded(FileNotFoundError):
    """A blob is not in this store, raw or packed; ``snapshot`` is the box snapshot holding it."""

    def __init__(self, sha256: str, snapshot: str | None, root: Path) -> None:
        held = f"; box snapshot {snapshot} holds it" if snapshot else ""
        super().__init__(f"blob {sha256} is not in the store at {root}{held}; fetch it back on the "
                          f"hub: python -m menagerie.pipeline store fetch {sha256} --remote "
                          f"<worker>-hub:<worker store>")
        self.sha256, self.snapshot = sha256, snapshot


def packed_path(sha256: str, *, store: StoreLayout) -> Path:
    raw = store.blob(sha256)
    return raw.with_name(raw.name + PACKED_SUFFIX)


def gzipped_path(sha256: str, *, store: StoreLayout) -> Path:
    raw = store.blob(sha256)
    return raw.with_name(raw.name + GZIP_SUFFIX)


def placement(sha256: str, *, data: DataLayout) -> StoreIndexLine | None:
    """The latest index line for a blob that states an ``encoding`` or ``location``."""

    found = [line for line in _index_lines(data) if line.sha256 == sha256
             and (line.encoding is not None or line.location is not None)]
    return found[-1] if found else None


def _parts_order(rel: str) -> tuple[str, ...]:
    # ``tree_sha256`` sorts ``Path`` objects, which compare by parts, not by the joined string.
    return Path(rel).parts


def tree_files(path: Path) -> list[tuple[str, Path]]:
    """A directory blob's regular files in ``tree_sha256`` order, by relative POSIX path."""

    files = sorted(p for p in path.rglob("*") if p.is_file())
    return [(p.relative_to(path).as_posix(), p) for p in files]


def _digest(pairs: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for rel, sha in sorted(pairs, key=lambda pair: _parts_order(pair[0])):
        digest.update(f"{rel}\0{sha}\n".encode())
    return digest.hexdigest()


def manifest_digest(files: Mapping[str, str]) -> str:
    """The ``tree_sha256`` of a tree given each file's sha256 by relative path."""

    return _digest(list(files.items()))


def pack_tree(src: Path, out: Path) -> None:
    """Write a directory blob as a deterministic ``tar.gz`` (sorted, zero times and owners, no
    file name in the gzip header), so a tree packs to the same bytes under any output name."""

    with out.open("wb") as raw, \
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for rel, path in tree_files(src):
            info = tarfile.TarInfo(rel)
            info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
            with path.open("rb") as handle:
                tar.addfile(info, handle)
    fsync(out)  # after tar and gzip have written their last bytes


def gzip_file(src: Path, out: Path) -> None:
    """Write a file blob as a deterministic gzip (zero time, no file name in the header), so a
    file gzips to the same bytes under any output name."""

    with src.open("rb") as handle, out.open("wb") as raw, \
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz:
        shutil.copyfileobj(handle, gz, 1 << 20)
    fsync(out)  # after gzip has written its last bytes


def gzipped_stats(gzipped: Path) -> tuple[str, int]:
    """The gzipped file's own sha256 and byte count, read from the stream."""

    digest, size = hashlib.sha256(), 0
    with gzip.open(gzipped, "rb") as handle:
        for chunk in iter(functools.partial(handle.read, 1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def gunzip_file(gzipped: Path, dest: Path) -> None:
    """Write a gzipped blob's bytes to the new file ``dest`` (read-only)."""

    with gzip.open(gzipped, "rb") as handle, dest.open("xb") as out:
        shutil.copyfileobj(handle, out, 1 << 20)
    dest.chmod(0o444)


def fsync(path: Path) -> None:
    """Make a file, or a directory's entries, durable before anything depends on them."""

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _members(packed: Path):
    """Yield ``(relative path, readable stream)`` for each file of a packed blob, in order;
    anything but a plain relative regular file is refused."""

    with tarfile.open(packed, mode="r|gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if not member.isfile() or member.name.startswith("/") or ".." in parts:
                raise ValueError(f"packed blob {packed.name}: refused member {member.name!r}")
            yield member.name, tar.extractfile(member)


def packed_stats(packed: Path) -> tuple[str, int, int]:
    """The unpacked tree's sha256, file count and bytes, read from the stream."""

    pairs, size = [], 0
    for rel, handle in _members(packed):
        digest = hashlib.sha256()
        for chunk in iter(functools.partial(handle.read, 1 << 20), b""):  # type: ignore[union-attr]
            digest.update(chunk)
            size += len(chunk)
        pairs.append((rel, digest.hexdigest()))
    return _digest(pairs), len(pairs), size


def packed_member(packed: Path, rel: str) -> bytes | None:
    """One file of a packed blob, read from the stream without unpacking the tree."""

    for name, handle in _members(packed):
        if name == rel:
            return handle.read()  # type: ignore[union-attr]
    return None


def unpack_tree(packed: Path, dest: Path) -> None:
    """Unpack a packed blob into the new directory ``dest`` (files read-only)."""

    dest.mkdir(parents=True)
    for rel, handle in _members(packed):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            shutil.copyfileobj(handle, out, 1 << 20)  # type: ignore[arg-type]
        target.chmod(0o444)


def _cache_root() -> tuple[Path, int]:
    root = os.environ.get("MENAGERIE_UNPACK_CACHE")
    cap = float(os.environ.get("MENAGERIE_UNPACK_CACHE_GB") or UNPACK_CACHE_GB)
    return (Path(root) if root else Path.home() / ".cache/menagerie/unpack"), int(cap * 2**30)


def _evict(root: Path, cap: int, incoming: int) -> None:
    """Drop the least recently used unpacked trees until ``incoming`` more bytes fit the cap. A
    tree handed out within the lease may still be read, so it stays even past the cap."""

    entries = sorted((p.stat().st_mtime, p) for p in root.iterdir() if not p.name.startswith("."))
    held = sum(_bytes(p) for _, p in entries)
    leased = time.time() - UNPACK_LEASE_SECONDS
    for mtime, path in entries:
        if held + incoming <= cap or mtime >= leased:
            break
        held -= _bytes(path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _unpacked(sha256: str, packed: Path) -> Path:
    """The unpacked tree of a packed blob in the cache, unpacked and checked on first use."""

    root, cap = _cache_root()
    path = root / sha256
    if path.is_dir():
        os.utime(path)
        return path
    root.mkdir(parents=True, exist_ok=True)
    _evict(root, cap, packed.stat().st_size * 6)
    tmp = Path(tempfile.mkdtemp(prefix=f".{sha256[:12]}.", dir=root))
    try:
        unpack_tree(packed, tmp / "tree")
        actual = tree_sha256(tmp / "tree")
        if actual != sha256:
            raise ValueError(f"packed blob {sha256} unpacks to {actual}")
        try:
            os.replace(tmp / "tree", path)
        except OSError:
            if not path.is_dir():  # another reader may have put it in place first
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return path


def _gunzipped(sha256: str, gzipped: Path) -> Path:
    """The bytes of a gzipped blob as a file in the cache, written and checked on first use."""

    root, cap = _cache_root()
    path = root / sha256
    if path.is_file():
        os.utime(path)
        return path
    root.mkdir(parents=True, exist_ok=True)
    _evict(root, cap, gzipped.stat().st_size * 30)
    tmp = Path(tempfile.mkdtemp(prefix=f".{sha256[:12]}.", dir=root))
    try:
        gunzip_file(gzipped, tmp / "blob")
        actual = tree_sha256(tmp / "blob")
        if actual != sha256:
            raise ValueError(f"gzipped blob {sha256} unzips to {actual}")
        try:
            os.replace(tmp / "blob", path)
        except OSError:
            if not path.is_file():  # another reader may have put it in place first
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return path


class _Text(html.parser.HTMLParser):
    """Visible text of an HTML page: script and style dropped, block ends as line breaks."""

    SKIP = frozenset({"script", "style", "noscript", "svg", "head"})
    BREAK = frozenset({"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "blockquote",
                       "section", "article", "dd", "dt"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.SKIP:
            self.skipping += 1
        if tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        if tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, text: str) -> None:
        if not self.skipping:
            self.parts.append(text)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line) + "\n"


def visible_text(page: str) -> str:
    """The visible text of an HTML page: scripts, styles and the head left out."""

    parser = _Text()
    parser.feed(page)
    return parser.text()


def _pdf_text(blob: Path, out: Path) -> None:
    """``pdftotext`` here, or on the worker store's host when this machine lacks it (the hub)."""

    if shutil.which("pdftotext") is not None:
        subprocess.run(["pdftotext", "-enc", "UTF-8", str(blob), str(out)], check=True,
                       capture_output=True, timeout=300)
        return
    host, _ = _split(os.environ.get("MENAGERIE_STORE_REMOTE") or "")
    if host is None:
        raise RuntimeError("pdftotext (poppler-utils) is needed to extract PDF text, here or on "
                           "the MENAGERIE_STORE_REMOTE host")
    for attempt in range(3):  # ssh exits 255 on a dropped connection; pdftotext's own codes stop
        done = subprocess.run(["ssh", host, "pdftotext -enc UTF-8 - -"], input=blob.read_bytes(),
                              capture_output=True, timeout=300)
        if done.returncode != 255 or attempt == 2:
            break
    if done.returncode != 0:
        raise RuntimeError(f"pdftotext on {host} exited {done.returncode}: {done.stderr[-300:]!r}")
    out.write_bytes(done.stdout)


def _sniff(blob: Path) -> str:
    head = blob.read_bytes()[:1024]
    if head.startswith(b"%PDF"):
        return "application/pdf"
    return "text/html" if b"<html" in head.lower() or b"<!doctype" in head.lower() else ""


def extract_text(sha256: str, *, store: StoreLayout, data: DataLayout) -> StoreIndexLine:
    """Store the plain text extracted from a PDF or HTML blob; its index line names the source."""

    blob = get(sha256, store=store)
    source = index_line(sha256, data=data)
    mime = source.mime if source else _sniff(blob)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "text.txt"
        if mime.startswith("application/pdf"):
            _pdf_text(blob, out)
        elif mime.startswith(("text/html", "application/xhtml")):
            text = visible_text(blob.read_bytes().decode("utf-8", errors="replace"))
            out.write_text(text, encoding="utf-8")
        else:
            raise ValueError(f"blob {sha256} is {mime or 'of unknown type'}; need PDF or HTML")
        return put(out, store=store, data=data, mime="text/plain; charset=utf-8",
                   license_class=source.license_class if source else "third-party",
                   url=source.url if source else None, derived_from=sha256)


def _rsync(src: str, dest: str) -> list[str]:
    """Run ``rsync --ignore-existing`` and return the relative paths it transferred."""

    done = subprocess.run(
        ["rsync", "-a", "--ignore-existing", "--out-format=%n", src, dest],
        check=True, capture_output=True, text=True, timeout=3600,
    )
    return [line for line in done.stdout.splitlines() if line and not line.endswith("/")]


def mirror(*, store: StoreLayout, dest: str) -> list[str]:
    """Copy blobs missing from the off-site mirror ``dest``; return the sha256s copied."""

    if not (store.root / "sha256").is_dir():
        return []
    copied = _rsync(f"{store.root / 'sha256'}/", f"{dest.rstrip('/')}/sha256/")
    # A .tlspec blob is a directory: report its sha once, not each file inside it.
    return sorted({Path(rel).parts[1] for rel in copied if len(Path(rel).parts) >= 2})


def restore(sha256: str, *, store: StoreLayout, source: str) -> Path:
    """Fetch one blob back from the mirror ``source``, verify its sha256, return its path."""

    dest = store.blob(sha256)
    rel = dest.relative_to(store.root).as_posix()
    store.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=store.root) as tmp:
        # Into the directory, so a file blob and a .tlspec directory land the same way.
        subprocess.run(["rsync", "-a", f"{source.rstrip('/')}/{rel}", f"{tmp}/"], check=True,
                       capture_output=True, text=True, timeout=3600)
        got = Path(tmp) / sha256
        actual = tree_sha256(got)
        if actual != sha256:
            raise ValueError(f"mirror copy of {sha256} hashes to {actual}")
        if not dest.exists():
            _place(got, dest)
    return dest


# --- Rung 1: code snapshots, hub staging, drain and materialize ----------------------------
# A code snapshot (``fetch.fetch_code``, ``fetch.fetch_hf``) is one deterministic gzipped tar:
# sorted members, zero times and owners, its manifest first, so the same tree at the same commit
# always hashes to the same blob. The hub's store is staging: ``MENAGERIE_STORE_REMOTE`` names
# the worker store (``host:path`` or a local directory) that ``drain`` copies blobs to.

SNAPSHOT_MANIFEST = ".menagerie-snapshot.json"
SNAPSHOT_MIME = "application/gzip; menagerie=snapshot"
STAGE_CAP_BYTES = 10 * 2**30


class StoreFull(RuntimeError):
    """The hub staging store is over its cap; drain it before fetching more."""


def pack_snapshot(files: Mapping[str, bytes], manifest: Mapping[str, Any]) -> bytes:
    """Return the deterministic ``tar.gz`` bytes of ``files`` with ``manifest`` as its first member."""

    members = {SNAPSHOT_MANIFEST: (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode()}
    members.update(sorted(files.items()))
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(body), 0o644, 0
            tar.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def snapshot_manifest(path: Path) -> dict[str, Any] | None:
    """Return a stored snapshot's manifest, or ``None`` when the blob is not a snapshot."""

    try:
        with tarfile.open(path, mode="r:gz") as tar:
            member = tar.next()
            if member is None or member.name != SNAPSHOT_MANIFEST:
                return None
            handle = tar.extractfile(member)
            return json.loads(handle.read()) if handle else None
    except (tarfile.TarError, OSError, ValueError):
        return None


def snapshot_files(path: Path) -> dict[str, bytes]:
    """Return every file of a stored snapshot except its manifest, by relative path."""

    with tarfile.open(path, mode="r:gz") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar  # type: ignore[union-attr]
                if m.isfile() and m.name != SNAPSHOT_MANIFEST}


def _unpack(path: Path, dest: Path) -> None:
    """Unpack a snapshot into ``dest``, refusing any member that would land outside it."""

    root = dest.resolve()
    for name, body in snapshot_files(path).items():
        target = (dest / name).resolve()
        if root not in target.parents:
            raise ValueError(f"snapshot member {name!r} escapes {dest}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        target.chmod(0o444)


def _remote() -> str:
    remote = os.environ.get("MENAGERIE_STORE_REMOTE")
    if not remote:
        raise RuntimeError("MENAGERIE_STORE_REMOTE is not set; it names the worker store "
                           "(an rsync target, host:path or a directory)")
    return remote.rstrip("/")


def _split(remote: str) -> tuple[str | None, str]:
    """``host:path`` -> (host, path); a local directory -> (None, path)."""

    host, sep, path = remote.partition(":")
    if sep and "/" not in host:
        return host, path
    return None, remote


def staged_bytes(store: StoreLayout) -> int:
    """Bytes held under the store's ``sha256/`` tree."""

    return _bytes(store.root / "sha256") if (store.root / "sha256").is_dir() else 0


def check_room(store: StoreLayout, incoming: int = 0) -> None:
    """Refuse (``StoreFull``) when this is a staging store and it would pass its cap.

    A store is staging when ``MENAGERIE_STORE_REMOTE`` is set; the cap is 10 GB, or
    ``MENAGERIE_STAGE_CAP_GB``.
    """

    if not os.environ.get("MENAGERIE_STORE_REMOTE"):
        return
    cap = float(os.environ.get("MENAGERIE_STAGE_CAP_GB") or 0) * 2**30 or STAGE_CAP_BYTES
    held = staged_bytes(store)
    if held + incoming > cap:
        raise StoreFull(f"staging store {store.root} holds {held} bytes, cap {int(cap)}; "
                        "run `store drain` first")


def _remote_sha256(remote: str, rels: Sequence[str]) -> dict[str, str]:
    """sha256 of each relative path at the remote store (missing paths are absent)."""

    host, root = _split(remote)
    if host is None:
        return {rel: tree_sha256(Path(root) / rel) for rel in rels if (Path(root) / rel).is_file()}
    script = f"cd {shlex.quote(root)} && sha256sum -- {' '.join(map(shlex.quote, rels))}"
    done = subprocess.run(["ssh", host, script], capture_output=True, text=True, timeout=600)
    pairs = (line.split(None, 1) for line in done.stdout.splitlines() if line.strip())
    return {rel.strip().lstrip("*"): sha for sha, rel in pairs}


def _staged(shas: Sequence[str], store: StoreLayout) -> list[str]:
    """The named blobs (each must be staged), or every blob the store holds."""

    if shas:
        todo = list(dict.fromkeys(shas))
        missing = [sha for sha in todo if not store.blob(sha).is_file()]
        if missing:
            raise FileNotFoundError(f"not staged in {store.root}: {missing}")
        return todo
    root = store.root / "sha256"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.glob("*/*") if p.is_file() and re.fullmatch(SHA256, p.name))


def _push(store: StoreLayout, remote: str, rels: Sequence[str]) -> None:
    host, root = _split(remote)
    if host is None:
        Path(root).mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--ignore-existing", "--files-from=-", f"{store.root}/",
                    f"{remote}/"], input="\n".join(rels) + "\n", check=True, capture_output=True,
                   text=True, timeout=3600)


def drain(shas: Sequence[str] = (), *, store: StoreLayout, data: DataLayout) -> list[str]:
    """Copy staged blobs to the worker store, verify each there, then delete the staged copy.

    Inputs: the blobs to move (default: every blob in this store). The destination is
    ``MENAGERIE_STORE_REMOTE``. Returns: the sha256s moved, each verified at the destination. A
    blob whose remote copy does not verify stays staged and is reported in the raised error.
    Side effects: files in the worker store; staged files deleted. The index is unchanged (it
    already lists the blobs). ``data`` is unused; it keeps the verb's signature uniform.
    """

    del data
    remote = _remote()
    todo = _staged(shas, store)
    if not todo:
        return []
    rels = [store.blob(sha).relative_to(store.root).as_posix() for sha in todo]
    _push(store, remote, rels)
    there = _remote_sha256(remote, rels)
    good = [sha for sha, rel in zip(todo, rels) if there.get(rel) == sha]
    for sha in good:
        store.blob(sha).unlink()
    bad = sorted(set(todo) - set(good))
    if bad:
        raise ValueError(f"remote copies of {bad} do not verify at {remote}; kept staged")
    return todo


def _pull(sha256: str, store: StoreLayout, tmp: Path) -> Path:
    """Copy a blob back from the worker store into ``tmp``: raw, else packed (or gzipped) and
    unpacked there. Raises ``BlobOffloaded`` when the worker store has no form of it."""

    remote = _remote()
    rel = store.blob(sha256).relative_to(store.root).as_posix()
    for suffix in ("", PACKED_SUFFIX, GZIP_SUFFIX):
        done = subprocess.run(["rsync", "-a", f"{remote}/{rel}{suffix}", f"{tmp}/"],
                              capture_output=True, text=True, timeout=3600, check=False)
        if done.returncode == 0 and suffix == PACKED_SUFFIX:
            unpack_tree(tmp / (sha256 + suffix), tmp / sha256)
        if done.returncode == 0 and suffix == GZIP_SUFFIX:
            gunzip_file(tmp / (sha256 + suffix), tmp / sha256)
        if done.returncode == 0:
            return tmp / sha256
        if done.returncode != 23:  # 23: the source path does not exist
            raise RuntimeError(f"rsync of {rel}{suffix} from {remote} exited {done.returncode}: "
                               f"{done.stderr[-300:]!r}")
    raise BlobOffloaded(sha256, None, Path(remote))


def materialize(sha256: str, dest: Path, *, store: StoreLayout) -> Path:
    """Return a local copy of a blob at ``dest``, pulled back from the worker store if needed.

    A code snapshot is unpacked into the directory ``dest`` (files read-only); a directory blob
    is copied to the directory ``dest``; any other blob is copied to the file ``dest``. A packed
    blob, here or on the worker, is read through its packed form. The copy's sha256 is checked
    before it is used. Returns ``dest``. Raises ``BlobOffloaded`` when only the box holds the
    blob (``store_box restore`` brings it back). Side effects: ``dest`` only; a pulled blob is
    not re-staged.
    """

    dest = Path(dest)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            return read_blob(sha256, functools.partial(_copy_out, sha256, dest), store=store)
        except BlobOffloaded:
            pass
        try:
            return _copy_out(sha256, dest, _pull(sha256, store, Path(tmp)))
        except BlobOffloaded:
            if not fetch_back([sha256]):
                raise
        return _copy_out(sha256, dest, _pull(sha256, store, Path(tmp)))


FETCH_ENV = "MENAGERIE_STORE_FETCH"


def can_fetch() -> bool:
    """Whether this machine can restore from the box: the hub (its storage catalog and the
    credential broker), unless ``MENAGERIE_STORE_FETCH=0``."""

    home = Path.home()
    catalog = local_value("STORE_CATALOG", "")  # the storage catalog; set only on the hub
    return (os.environ.get(FETCH_ENV, "1") != "0" and (home / ".local/bin/op-hub").is_file()
            and bool(catalog) and Path(catalog).expanduser().is_dir())


def fetch_back(shas: Sequence[str], data: DataLayout | None = None) -> bool:
    """On the hub, restore blobs only the box holds into the worker store
    (``MENAGERIE_STORE_REMOTE``), each checked by hash there; ``False`` when this machine cannot
    or a blob did not come back."""

    if not can_fetch():
        return False
    from menagerie.pipeline import store_box

    data = data or DataLayout.from_env()
    try:
        report = store_box.restore_blobs(list(shas), remote=_remote(), data=data.root,
                                         desk=Path.home() / "desk")
    except (SystemExit, OSError, subprocess.SubprocessError, ValueError):
        return False  # no box placement, a refused broker or a failed copy: the caller's error stands
    return not report["failed"] and len(report["restored"]) == len(set(shas))


def _copy_out(sha256: str, dest: Path, local: Path) -> Path:
    actual = tree_sha256(local)
    if actual != sha256:
        raise ValueError(f"copy of {sha256} hashes to {actual}")
    if local.is_dir():
        shutil.copytree(local, dest, dirs_exist_ok=True)
    elif snapshot_manifest(local) is not None:
        dest.mkdir(parents=True, exist_ok=True)
        _unpack(local, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
    return dest
