"""A library's own documentation of a model as a describe text source (``docs``).

A validated family whose packet names a library model but saved no paper, page or library text
gets describe's ``docs`` source here: the module docstring and the docstring of each member's
symbol, read with ``ast`` from the library's source at the exact version (never imported), and,
where the library keeps a page per model at that tag (timm, transformers, ultralytics,
torchvision), that page. One archive per (dist, version) is fetched a run: the GitHub tarball at
the version tag for a mapped library, else the PyPI sdist (or a pure-Python wheel). The combined
text is one store blob, added to the packet as ``Source(sid="docs-<dist>", role="code")`` with an
amendment by ``library-docs``; ``describe_text.candidates`` reads it as kind ``docs``.

Failures and texts too short to ground anything are kept in ``work_dir("library-docs")/tried.json``
and retried after ``RETRY_DAYS``. Fetched archives are untrusted data: read member by member,
parsed with ``ast`` and never extracted or run.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tarfile
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, StoreLayout, fetch, now, read_json, write_json
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import Amendment, Packet, Source

BY = "library-docs"
RETRY_DAYS = 14
MIN_USEFUL = 200  # characters of documentation (whitespace collapsed) worth a source
MAX_DISTS = 3  # libraries read per family
MAX_MODULES = 4
MAX_SYMBOLS = 12
MAX_PAGES = 2
DOCSTRING_CHARS = 4_000
PAGE_CHARS = 20_000
MAX_MEMBER_BYTES = 2 * 2**20  # an archive file larger than this is not read
READ_SUFFIXES = (".py", ".md", ".mdx", ".rst")
TEXT_MIME = "text/plain; charset=utf-8"

# dist -> GitHub repository whose tag ``v<version>`` (or ``<version>``) is that release.
REPOS = {
    "timm": "huggingface/pytorch-image-models",
    "transformers": "huggingface/transformers",
    "torchvision": "pytorch/vision",
    "ultralytics": "ultralytics/ultralytics",
    "segmentation_models_pytorch": "qubvel-org/segmentation_models.pytorch",
    "torchaudio": "pytorch/audio",
    "monai": "Project-MONAI/MONAI",
}
# dist -> (folder, suffix) of its per-model documentation pages in the repository.
PAGES = {
    "timm": ("hfdocs/source/models", ".mdx"),
    "transformers": ("docs/source/en/model_doc", ".md"),
    "ultralytics": ("docs/en/models", ".md"),
    "torchvision": ("docs/source/models", ".rst"),
}
COMMENT = re.compile(r"<!--.*?-->", re.S)


class DocsError(RuntimeError):
    """No archive of the library at that version could be read."""


@dataclass(frozen=True)
class Archive:
    """The readable files of one library release: relative path -> text."""

    dist: str
    version: str
    rev: str
    base_url: str  # a file's URL is ``base_url + path``
    files: Mapping[str, str]


Fetcher = Callable[[str, Path], None]  # url, destination; raises on failure


def _get(url: str, dest: Path) -> None:
    fetch._download(url, dest)  # noqa: SLF001 -- the polite, retrying downloader


# --- which packets are due ---------------------------------------------------------------


def _libraries(packet: Packet) -> list[Source]:
    return [s for s in packet.sources if s.role == "library" and s.dist and s.version]


def due(packet: Packet) -> bool:
    """A library model with no saved paper, page or code text."""

    texts = any(s.text for s in packet.sources if s.role in ("paper", "page", "code"))
    return bool(_libraries(packet)) and not texts


def _dists(packet: Packet) -> list[tuple[str, str]]:
    """The packet's (dist, version) pairs, most library sources first, at most ``MAX_DISTS``."""

    counts = Counter((s.dist, s.version) for s in _libraries(packet))
    ranked = sorted(counts, key=lambda pair: (-counts[pair], pair))
    return [(str(d), str(v)) for d, v in ranked][:MAX_DISTS]


def _tried_path() -> Path:
    return work_dir("library-docs") / "tried.json"


def _read_tried() -> dict[str, dict[str, str]]:
    path = _tried_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_tried(tried: Mapping[str, Any]) -> None:
    path = _tried_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(tried, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _resting(entry: Mapping[str, str] | None, at: datetime) -> bool:
    if not entry:
        return False
    when = datetime.strptime(entry["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return at - when < timedelta(days=RETRY_DAYS)


def _due_packets(families: Sequence[str] | None, data: DataLayout,
                 tried: Mapping[str, Mapping[str, str]]) -> list[Packet]:
    """Due packets not resting after a recent failure, grouped by their first library so one
    archive serves a run's families of that release."""

    paths = ([data.packet(f) for f in dict.fromkeys(families)] if families is not None
             else sorted(data.root.glob("research/*/*.json")))
    at = datetime.now(timezone.utc)
    found = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        roles = {s.get("role") for s in raw.get("sources") or ()}
        if "library" not in roles or _resting(tried.get(str(raw.get("family"))), at):
            continue
        packet = read_json(Packet, path)
        if due(packet):
            found.append(packet)
    return sorted(found, key=lambda p: (_dists(p)[0], p.family))


# --- archives ------------------------------------------------------------------------------


Keep = Callable[[str], bool]  # which archive paths a run reads


def wanted(dist: str, modules: Sequence[str]) -> Keep:
    """The files a run reads from one release: the modules its families name (``src/`` layouts
    and packages included) and the release's model pages; nothing else is held in memory."""

    targets = {m.replace(".", "/") for m in modules}
    folder, suffix = PAGES.get(dist, ("", ""))

    def keep(rel: str) -> bool:
        if folder and rel.startswith(folder + "/") and rel.endswith(suffix):
            return True
        if not rel.endswith(".py"):
            return False
        stem = rel.removesuffix("/__init__.py").removesuffix(".py").split("/")
        return any("/".join(stem[i:]) in targets for i in range(len(stem)))

    return keep


def _members(path: Path, keep: Keep) -> dict[str, str]:
    """The kept text files of a tar or zip archive, by path below its top folder (a wheel has
    none). Nothing is extracted to disk."""

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            items = [(i.filename, i.file_size, lambda i=i: zf.read(i)) for i in zf.infolist()
                     if not i.is_dir()]
            return _texts(items, keep, top=not path.name.endswith(".whl"))
    with tarfile.open(path, "r:*") as tf:
        items = [(m.name, m.size, lambda m=m: tf.extractfile(m).read()) for m in tf.getmembers()
                 if m.isfile()]
        return _texts(items, keep, top=True)


def _texts(items: Sequence[tuple[str, int, Callable[[], bytes]]], keep: Keep, *, top: bool
           ) -> dict[str, str]:
    out = {}
    for name, size, read in items:
        rel = name.split("/", 1)[1] if top and "/" in name else name
        if (rel.endswith(READ_SUFFIXES) and size <= MAX_MEMBER_BYTES
                and ".." not in rel.split("/") and keep(rel)):
            out[rel] = read().decode("utf-8", errors="replace")
    return out


def _github(dist: str, version: str, get: Fetcher, tmp: Path, keep: Keep) -> Archive:
    repo = REPOS[dist]
    problems = []
    for tag in (f"v{version}", version):
        dest = tmp / f"{dist}-{tag}.tar.gz"
        try:
            get(f"https://codeload.github.com/{repo}/tar.gz/refs/tags/{tag}", dest)
        except Exception as exc:  # noqa: BLE001 -- the next tag, then PyPI
            problems.append(f"{tag}: {exc}"[:200])
            continue
        return Archive(dist, version, tag, f"https://github.com/{repo}/blob/{tag}/",
                       _members(dest, keep))
    raise DocsError(f"no GitHub tag of {repo}: {'; '.join(problems)}")


def _pypi(dist: str, version: str, get: Fetcher, tmp: Path, keep: Keep) -> Archive:
    meta = tmp / f"{dist}-{version}.json"
    get(f"https://pypi.org/pypi/{dist}/{version}/json", meta)
    urls = json.loads(meta.read_text(encoding="utf-8")).get("urls") or []
    pick = (next((u for u in urls if u.get("packagetype") == "sdist"), None)
            or next((u for u in urls if str(u.get("filename", "")).endswith("-none-any.whl")),
                    None))
    if pick is None:
        raise DocsError(f"PyPI {dist} {version}: no sdist or pure-Python wheel")
    dest = tmp / Path(str(pick["filename"])).name
    get(str(pick["url"]), dest)
    return Archive(dist, version, version, f"https://pypi.org/project/{dist}/{version}/#",
                   _members(dest, keep))


def _archive(dist: str, version: str, get: Fetcher, tmp: Path, keep: Keep) -> Archive:
    if dist in REPOS:
        try:
            return _github(dist, version, get, tmp, keep)
        except DocsError:
            pass
    try:
        return _pypi(dist, version, get, tmp, keep)
    except DocsError:
        raise
    except Exception as exc:  # noqa: BLE001 -- recorded as this release's failure
        raise DocsError(f"PyPI {dist} {version}: {exc}"[:300]) from exc


# --- reading one release's documentation of a family ------------------------------------------


def _module_file(files: Mapping[str, str], module: str) -> str | None:
    """The archive path of ``module`` (``src/`` layouts and packages included), shortest first."""

    target = module.replace(".", "/")
    names = (f"{target}.py", f"{target}/__init__.py")
    hits = [rel for rel in files if rel in names or rel.endswith(tuple("/" + n for n in names))]
    return min(hits, key=len) if hits else None


def _docstring(tree: ast.Module, symbol: str) -> str | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == symbol:
            text = ast.get_docstring(node)
            if text or not isinstance(node, ast.ClassDef):
                return text
            init = next((n for n in node.body if isinstance(n, ast.FunctionDef)
                         and n.name == "__init__"), None)
            return ast.get_docstring(init) if init else None
    return None


def _module_parts(files: Mapping[str, str], module: str, symbols: Sequence[str]
                  ) -> list[tuple[str, str]]:
    """(heading, text) for the module docstring and each symbol's distinct docstring."""

    rel = _module_file(files, module)
    if rel is None:
        return []
    try:
        tree = ast.parse(files[rel])
    except (SyntaxError, ValueError, RecursionError):
        return []
    parts, seen = [], set()
    doc = ast.get_docstring(tree)
    if doc:
        parts.append((f"{rel}: module docstring", doc[:DOCSTRING_CHARS]))
        seen.add(doc)
    for symbol in symbols:
        doc = _docstring(tree, symbol)
        if doc and doc not in seen:
            parts.append((f"{rel}: {symbol}", doc[:DOCSTRING_CHARS]))
            seen.add(doc)
    return parts


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _under(name: str) -> str:
    return name.lower().replace("-", "_")


def _prefix(stem: str, symbol: str) -> bool:
    """``stem`` names the model ``symbol`` is a size of: ``skresnet`` and ``skresnet18``,
    ``mobilenet-v3`` and ``mobilenetv3_large_100``, never ``resnet`` and ``resnetv2_50``."""

    pairs = ((_under(stem), _under(symbol)), (_norm(stem), _under(symbol)),
             (_norm(stem), _norm(symbol)))
    for a, b in pairs:
        if b.startswith(a) and (len(b) == len(a) or b[len(a)].isdigit() or b[len(a)] == "_"):
            return True
    return False


def _page_names(dist: str, sources: Sequence[Source], family: str) -> tuple[set[str], list[str]]:
    """Names a model page could carry: module and model-type names and the family's own name
    (matched whole), and the member symbols (matched as a prefix)."""

    whole = {_norm(family.removeprefix("f-"))}
    for src in sources:
        parts = str(src.module or "").split(".")
        whole.add(_norm(parts[-1]))
        if dist == "transformers" and "models" in parts[:-1]:
            whole.add(_norm(parts[parts.index("models") + 1]))
    return whole - {""}, [str(s.symbol) for s in sources if s.symbol]


def _pages(archive: Archive, sources: Sequence[Source], family: str) -> list[str]:
    """The release's model pages for this family, at most ``MAX_PAGES``."""

    if archive.dist not in PAGES:
        return []
    folder, suffix = PAGES[archive.dist]
    stems = {rel: Path(rel).stem for rel in archive.files
             if rel.startswith(folder + "/") and rel.endswith(suffix) and rel.count("/") ==
             folder.count("/") + 1}
    whole, symbols = _page_names(archive.dist, sources, family)
    exact = sorted(rel for rel, stem in stems.items() if _norm(stem) in whole)
    if exact:
        return exact[:MAX_PAGES]
    hits = {rel for rel, stem in stems.items() for s in symbols if _prefix(stem, s)}
    return sorted(hits, key=lambda rel: (-len(stems[rel]), rel))[:MAX_PAGES]


def _page_text(text: str) -> str:
    return COMMENT.sub("", text).strip()[:PAGE_CHARS]


def _useful(parts: Sequence[tuple[str, str]]) -> int:
    return sum(len(" ".join(text.split())) for _, text in parts)


@dataclass(frozen=True)
class Docs:
    """One release's documentation of one family, ready to store."""

    dist: str
    text: str
    url: str
    rev: str
    files: tuple[str, ...]
    useful: int


def read_docs(archive: Archive, packet: Packet) -> Docs:
    """The combined documentation text of ``packet``'s models in ``archive``'s release."""

    sources = [s for s in _libraries(packet)
               if (s.dist, s.version) == (archive.dist, archive.version)]
    by_module: dict[str, list[str]] = {}
    for src in sources:
        if src.module:
            by_module.setdefault(src.module, []).append(str(src.symbol))
    parts: list[tuple[str, str]] = []
    for module, symbols in list(by_module.items())[:MAX_MODULES]:
        parts += _module_parts(archive.files, module, list(dict.fromkeys(symbols))[:MAX_SYMBOLS])
    pages = _pages(archive, sources, packet.family)
    parts += [(rel, _page_text(archive.files[rel])) for rel in pages]
    files = tuple(dict.fromkeys(h.split(":", 1)[0] for h, _ in parts))
    main = pages[0] if pages else (files[0] if files else "")
    header = (f"Library documentation: {archive.dist} {archive.version} at {archive.rev}\n"
              f"Files: {', '.join(files) or 'none'}\n")
    body = "".join(f"\n== {heading} ==\n{text}\n" for heading, text in parts)
    return Docs(archive.dist, header + body, archive.base_url + main, archive.rev, files,
                _useful(parts))


# --- writing -------------------------------------------------------------------------------


def _store(docs: Docs, *, store: StoreLayout, data: DataLayout) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "docs.txt"
        path.write_text(docs.text, encoding="utf-8")
        line = store_mod.put(path, store=store, data=data, mime=TEXT_MIME,
                             license_class="third-party", url=docs.url)
    return line.sha256


def _amend(packet: Packet, added: Sequence[tuple[Source, Docs]], *, data: DataLayout) -> None:
    sids = tuple(src.sid for src, _ in added)
    what = "; ".join(f"{d.dist} {d.rev}: {', '.join(d.files)}" for _, d in added)
    amendment = Amendment(at=now(), why=f"library docs: {what}"[:300], evidence=sids,
                          changed=("sources",), by=BY)
    kept = tuple(s for s in packet.sources if s.sid not in sids)
    write_json(replace(packet, sources=(*kept, *(s for s, _ in added)),
                       amendments=(*packet.amendments, amendment)), data.packet(packet.family))


@dataclass
class _Run:
    """One run's fetcher, scratch folder and archives (or their failures) by release."""

    get: Fetcher
    tmp: Path
    modules: Mapping[tuple[str, str], Sequence[str]]
    archives: dict[tuple[str, str], Archive | str]

    def archive(self, dist: str, version: str) -> Archive | str:
        key = (dist, version)
        if key not in self.archives:
            keep = wanted(dist, self.modules.get(key, ()))
            try:
                self.archives[key] = _archive(dist, version, self.get, self.tmp, keep)
            except DocsError as exc:
                self.archives[key] = str(exc)[:300]
            finally:
                for left in self.tmp.iterdir():  # the archive's files are read; free the disk
                    left.unlink()
        return self.archives[key]


def _modules(packets: Sequence[Packet]) -> dict[tuple[str, str], list[str]]:
    found: dict[tuple[str, str], list[str]] = {}
    for packet in packets:
        for src in _libraries(packet):
            if src.module:
                found.setdefault((str(src.dist), str(src.version)), []).append(src.module)
    return found


def _one(packet: Packet, run: _Run, *, store: StoreLayout, data: DataLayout) -> dict[str, Any]:
    added: list[tuple[Source, Docs]] = []
    notes = []
    for dist, version in _dists(packet):
        got = run.archive(dist, version)
        if isinstance(got, str):
            notes.append(got)
            continue
        docs = read_docs(got, packet)
        if docs.useful < MIN_USEFUL:
            notes.append(f"{dist} {version}: {docs.useful} useful characters")
            continue
        sha = _store(docs, store=store, data=data)
        added.append((Source(sid=f"docs-{dist}", role="code", url=docs.url, rev=docs.rev,
                             text=sha, official=True, fetched=now()), docs))
    if not added:
        return {"family": packet.family, "status": "none", "why": "; ".join(notes)[:400]}
    _amend(packet, added, data=data)
    return {"family": packet.family, "status": "docs",
            "sids": [s.sid for s, _ in added], "useful": [d.useful for _, d in added]}


def docs(families: Sequence[str] | None, *, data: DataLayout, store: StoreLayout,
         limit: int | None = None, write: bool = True, seconds: float | None = None,
         get: Fetcher = _get) -> list[dict[str, Any]]:
    """Add a library-documentation source to each due packet, at most ``limit`` and no new
    family after ``seconds``. Without ``write`` it lists the due packets (status ``due``) and
    fetches and writes nothing. A family that gained nothing rests ``RETRY_DAYS``."""

    if limit is not None and limit < 0:
        raise ValueError("library docs limit must be nonnegative")
    tried = _read_tried()
    todo = _due_packets(families, data, tried)[:limit]
    if not write:
        return [{"family": p.family, "status": "due", "dists": [d for d, _ in _dists(p)]}
                for p in todo]
    began, out = time.monotonic(), []
    with tempfile.TemporaryDirectory(prefix="library-docs-") as tmp:
        run = _Run(get, Path(tmp), _modules(todo), {})
        for packet in todo:
            if seconds is not None and time.monotonic() - began > seconds:
                break
            try:
                row = _one(packet, run, store=store, data=data)
            except Exception as exc:  # noqa: BLE001 -- one family's failure never stops the run
                row = {"family": packet.family, "status": "error",
                       "why": f"{type(exc).__name__}: {exc}"[:400]}
            out.append(row)
            if row["status"] == "docs":
                tried.pop(packet.family, None)
            else:
                tried[packet.family] = {"at": now(), "why": row["why"]}
            _write_tried(tried)
    return out

