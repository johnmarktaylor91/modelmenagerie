"""Code repositories and Hugging Face Hub repositories for 2a: ``fetch_code`` and ``fetch_hf``.

Split from ``fetch`` at rung 2 (``fetch.fetch_code`` and ``fetch.fetch_hf`` delegate here). The
HTTP, store and event helpers stay in ``fetch`` and are called through the module, so one patch
of ``fetch._http`` reaches both. Imports neither torch nor psutil.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
import tempfile
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout, now
from menagerie.pipeline import fetch as core
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import Source

FILE_CAP = 2**20                 # a repository file over 1 MB is dropped
TREE_CAP = 20 * 2**20            # at most 20 MB of files per repository
HF_FILE_CAP = 20 * 2**20         # tokenizer files can pass 1 MB
TARBALL_REPO_CAP_KB = 300 * 1024  # bigger GitHub repositories are read file by file
WEIGHTS = (".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".h5", ".msgpack", ".onnx", ".gguf",
           ".npz", ".npy", ".pkl", ".pickle", ".tflite", ".mlmodel", ".pb", ".caffemodel",
           ".params", ".weights", ".t7", ".mat")
BINARY = WEIGHTS + (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".ico",
                    ".pdf", ".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar", ".whl",
                    ".so", ".dll", ".dylib", ".exe", ".o", ".a", ".mp3", ".mp4", ".wav", ".flac",
                    ".avi", ".mov", ".mkv", ".ttf", ".otf", ".woff", ".woff2", ".eot", ".jar",
                    ".class", ".pyc", ".parquet", ".arrow", ".tfrecord", ".lmdb", ".db",
                    ".sqlite", ".mid", ".midi", ".svgz", ".psd", ".dat")
LICENSE_NAME = re.compile(r"(?:un)?licen[cs]e(?:[._-].*)?|copying(?:[._-].*)?", re.IGNORECASE)
REQUIREMENT_NAME = re.compile(r"requirements.*\.txt|setup\.py|setup\.cfg|pyproject\.toml|"
                              r"environment.*\.ya?ml|conda.*\.ya?ml|pipfile|install\.sh",
                              re.IGNORECASE)
FRAMEWORKS = (  # (framework, pattern in requirement files or imports), first hit wins
    ("pytorch", r"\btorch\b|\bpytorch\b"), ("tensorflow", r"\btensorflow\b|\btf-nightly\b"),
    ("jax", r"\bjax\b|\bflax\b"), ("paddle", r"\bpaddle(?:paddle)?\b"), ("mindspore", r"\bmindspore\b"),
    ("mlx", r"\bmlx\b"), ("tinygrad", r"\btinygrad\b"), ("mxnet", r"\bmxnet\b"),
    ("keras", r"\bkeras\b"), ("theano", r"\btheano\b"), ("chainer", r"\bchainer\b"),
    ("caffe", r"\bcaffe\b"),
)
HF_KEEP = re.compile(r".*\.(py|json|txt|md|model|tiktoken|ya?ml|cfg)$|.*tokenizer.*",
                     re.IGNORECASE)
HF_WEIGHT_HINTS = (("pytorch", ("pytorch_model", "model.safetensors", ".pt", ".pth", ".ckpt")),
                   ("tensorflow", ("tf_model.h5",)), ("jax", ("flax_model.msgpack",)))


# --- Code repositories -------------------------------------------------------------------


def _is_text(name: str, body: bytes) -> bool:
    return not name.lower().endswith(BINARY) and b"\0" not in body[:8192]


def _priority(path: str) -> tuple[int, int, str]:
    """Keep order under the 20 MB cap: README, LICENSE and requirements, code, configs, docs."""

    name, ext = path.rsplit("/", 1)[-1].lower(), Path(path).suffix.lower()
    if name.startswith("readme") or LICENSE_NAME.fullmatch(name) or REQUIREMENT_NAME.fullmatch(name):
        rank = 0
    elif ext in (".py", ".pyx", ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".lua", ".m",
                 ".jl", ".rs", ".java", ".scala", ".r"):
        rank = 1
    elif ext in (".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".prototxt", ".gin"):
        rank = 2
    elif ext in (".md", ".rst", ".txt", ".ipynb"):
        rank = 3
    else:
        rank = 4
    return rank, path.count("/"), path


def _select(sizes: Mapping[str, int], subpath: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Choose the files to keep from ``{path: bytes}``; return them and the dropped ones."""

    kept, dropped, total = [], [], 0
    for path in sorted(sizes, key=_priority):
        why = None
        if subpath and not (path.startswith(subpath + "/") or LICENSE_NAME.fullmatch(path)):
            continue
        if path.lower().endswith(BINARY):
            why = "binary"
        elif sizes[path] > FILE_CAP:
            why = "over_1mb"
        elif total + sizes[path] > TREE_CAP:
            why = "over_20mb_total"
        if why:
            dropped.append({"path": path, "bytes": sizes[path], "why": why})
        else:
            kept.append(path)
            total += sizes[path]
    return kept, dropped


FILE_FRAMEWORKS = ((".lua", "lua-torch"), (".prototxt", "caffe"), (".m", "matlab"))


def _first_framework(texts: Iterable[str]) -> str | None:
    blob = "\n".join(texts).lower()
    return next((fw for fw, pattern in FRAMEWORKS if re.search(pattern, blob)), None)


def _framework(files: Mapping[str, bytes]) -> str | None:
    """Guess the framework from requirement files, else imports, else telltale file types."""

    reqs = [b.decode("utf-8", "replace") for p, b in files.items()
            if REQUIREMENT_NAME.fullmatch(p.rsplit("/", 1)[-1])]
    code = "\n".join(b.decode("utf-8", "replace") for p, b in files.items() if p.endswith(".py"))
    imports = re.findall(r"^\s*(?:import|from)\s.*$", code, flags=re.MULTILINE)
    return (_first_framework(reqs) or _first_framework(imports)
            or next((fw for ext, fw in FILE_FRAMEWORKS if any(p.endswith(ext) for p in files)),
                    None))


SPDX_MARKERS = (
    ("Apache-2.0", "apache license", "version 2.0"), ("MIT-0", "mit no attribution"),
    ("MIT", "permission is hereby granted, free of charge"),
    ("BSD-3-Clause", "redistribution and use in source and binary forms", "neither the name"),
    ("BSD-2-Clause", "redistribution and use in source and binary forms"),
    ("GPL-3.0", "gnu general public license", "version 3"),
    ("GPL-2.0", "gnu general public license", "version 2"),
    ("LGPL-3.0", "gnu lesser general public license", "version 3"),
    ("AGPL-3.0", "gnu affero general public license"), ("MPL-2.0", "mozilla public license"),
    ("CC-BY-NC-SA-4.0", "noncommercial-sharealike 4.0"), ("CC-BY-NC-4.0", "attribution-noncommercial 4.0"),
    ("CC-BY-SA-4.0", "attribution-sharealike 4.0"),
    ("CC-BY-4.0", "creative commons attribution 4.0"), ("Unlicense", "this is free and unencumbered"),
)


def spdx_guess(text: str) -> str:
    """A first SPDX guess from LICENSE text (the agent may correct it); ``unknown`` otherwise."""

    low = " ".join(text.lower().split())
    return next((spdx for spdx, *marks in SPDX_MARKERS if all(m in low for m in marks)), "unknown")


def _license_file(files: Mapping[str, bytes], subpath: str) -> str | None:
    """The LICENSE nearest the root: the repository root first, then the subpath's own."""

    for folder in ("", f"{subpath}/" if subpath else None):
        if folder is None:
            continue
        hits = sorted(p for p in files if p.startswith(folder) and "/" not in p[len(folder):]
                      and LICENSE_NAME.fullmatch(p[len(folder):]))
        if hits:
            return hits[0]
    return None


def _save_tree(files: Mapping[str, bytes], dropped: list[dict[str, Any]], *, url: str, rev: str,
               raw: str | None, subpath: str, family: str, store: StoreLayout,
               data: DataLayout) -> Source:
    """Store the snapshot and its LICENSE; return the ``code`` source."""

    reasons: dict[str, int] = {}
    for item in dropped:
        reasons[item["why"]] = reasons.get(item["why"], 0) + 1
    manifest = {"url": url, "rev": rev, "subpath": subpath, "files": len(files),
                "bytes": sum(map(len, files.values())), "dropped_by_reason": reasons,
                "dropped": dropped[:500]}
    body = store_mod.pack_snapshot(files, manifest)
    line = core._put_bytes(body, store=store, data=data, mime=store_mod.SNAPSHOT_MIME, url=url)
    license_path, spdx = _license_file(files, subpath), None
    if license_path:
        text = files[license_path]
        core._put_bytes(text, store=store, data=data, mime="text/plain; charset=utf-8",
                   url=f"{raw}/{license_path}" if raw else f"{url}#{license_path}",
                   derived_from=line.sha256)
        spdx = spdx_guess(text.decode("utf-8", "replace"))
    core._event(family, kind="code", url=url, rev=rev, outcome="saved", bytes=len(body),
           files=len(files), dropped=len(dropped), license=license_path)
    return Source(sid=f"code-{line.sha256[:8]}", role="code", url=url, rev=rev, blob=line.sha256,
                  framework=_framework(files), spdx=spdx, fetched=now())


def _github(url: str) -> tuple[str, str, str | None, str] | None:
    """``(owner, repo, ref, subpath)`` of a GitHub URL, or ``None`` for another host."""

    match = re.fullmatch(r"https?://(?:www\.)?github\.com/([^/]+)/([^/#?]+?)(?:\.git)?"
                         r"(?:/(?:tree|blob)/([^/]+)(?:/(.*?))?)?/?", url.strip())
    if not match:
        return None
    owner, repo, ref, sub = match.groups()
    return owner, repo, ref, (sub or "").strip("/")


def _gh_json(path: str) -> Any:
    done = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        if re.search(r"HTTP (404|403|451)", done.stderr):
            raise core.Inaccessible(f"github {path}: {done.stderr.strip()[:200]}")
        raise core.FetchError(f"github {path}: {done.stderr.strip()[:200]}")
    return json.loads(done.stdout)


def _github_tarball(owner: str, repo: str, sha: str) -> tuple[dict[str, int], dict[str, bytes]]:
    """Stream the codeload tarball; return every file's size and the text files' bytes up to 1 MB."""

    sizes, bodies = {}, {}
    url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{sha}"
    request = urllib.request.Request(url, headers={"User-Agent": core.USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, \
            tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or "/" not in member.name:
                continue
            path = member.name.split("/", 1)[1]
            sizes[path] = member.size
            if member.size <= FILE_CAP and not path.lower().endswith(BINARY):
                handle = tar.extractfile(member)
                body = handle.read() if handle else b""
                if b"\0" in body[:8192]:
                    sizes[path] = -1  # binary content behind a text-looking name
                else:
                    bodies[path] = body
    return sizes, bodies


def _keep_text(kept: list[str], read: Callable[[str], bytes | None], sizes: Mapping[str, int],
               dropped: list[dict[str, Any]]) -> dict[str, bytes]:
    """Read each kept path; a body that is binary or unreadable joins ``dropped`` instead."""

    files = {}
    for path in kept:
        body = read(path)
        if body is not None and _is_text(path, body):
            files[path] = body
        else:
            dropped.append({"path": path, "bytes": sizes[path],
                            "why": "binary" if body is not None else "unreadable"})
    return files


def _github_tarball_files(owner: str, repo: str, sha: str, subpath: str
                          ) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    sizes, bodies = _github_tarball(owner, repo, sha)
    binary = [{"path": p, "bytes": 0, "why": "binary"} for p, s in sizes.items() if s < 0]
    kept, dropped = _select({p: s for p, s in sizes.items() if s >= 0}, subpath)
    return {p: bodies[p] for p in kept if p in bodies}, binary + dropped


def _github_tree_files(owner: str, repo: str, sha: str, subpath: str
                       ) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    """A repository too big for its tarball: list the tree, read the kept files one by one."""

    tree = _gh_json(f"repos/{owner}/{repo}/git/trees/{sha}?recursive=1")
    sizes = {t["path"]: t.get("size", 0) for t in tree.get("tree", []) if t.get("type") == "blob"}
    kept, dropped = _select(sizes, subpath)

    def read(path: str) -> bytes | None:
        got = core._http(f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/"
                    f"{urllib.parse.quote(path)}", limit=FILE_CAP)
        return got.body if got.status == 200 else None

    files = _keep_text(kept, read, sizes, dropped)
    if tree.get("truncated"):
        dropped.append({"path": "*", "bytes": 0, "why": "tree_listing_truncated"})
    return files, dropped


def _github_files(owner: str, repo: str, sha: str, subpath: str
                  ) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    small = _gh_json(f"repos/{owner}/{repo}").get("size", 0) <= TARBALL_REPO_CAP_KB
    return (_github_tarball_files if small else _github_tree_files)(owner, repo, sha, subpath)


GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/false"}


def _git(work: Path, *args: str) -> bytes:
    done = subprocess.run(["git", *args], cwd=work, env={**os.environ, **GIT_ENV},
                          capture_output=True, timeout=600)
    if done.returncode != 0:
        raise core.Inaccessible(f"git {args[0]}: {done.stderr.decode()[-300:].strip()}")
    return done.stdout


def _git_fetch(work: Path, url: str, rev: str | None) -> str:
    """A shallow fetch with a 1 MB blob filter; return the commit fetched."""

    _git(work, "init", "-q")
    # A promisor remote, so git before 2.27 (the worker has 2.25) accepts the blob filter.
    for args in (("remote", "add", "origin", url), ("config", "core.repositoryformatversion", "1"),
                 ("config", "extensions.partialClone", "origin"),
                 ("config", "remote.origin.promisor", "true")):
        _git(work, *args)
    _git(work, "fetch", "-q", "--depth", "1", "--filter=blob:limit=1m", "origin", rev or "HEAD")
    return _git(work, "rev-parse", "FETCH_HEAD").decode().strip()


def _git_sizes(work: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Each path's blob sha, and its size (over the cap when the filter left the blob out)."""

    blobs = {}
    for entry in filter(None, _git(work, "ls-tree", "-r", "-z", "FETCH_HEAD").split(b"\0")):
        meta, path = entry.split(b"\t", 1)
        kind, sha = meta.split()[1:3]
        if kind == b"blob":
            blobs[path.decode("utf-8", "replace")] = sha.decode()
    present = {}
    for line in _git(work, "cat-file", "--batch-check", "--batch-all-objects").splitlines():
        parts = line.decode().split()
        if len(parts) == 3 and parts[1] == "blob":
            present[parts[0]] = int(parts[2])
    return blobs, {p: present.get(sha, FILE_CAP + 1) for p, sha in blobs.items()}


def _git_files(url: str, rev: str | None, subpath: str
               ) -> tuple[str, dict[str, bytes], list[dict[str, Any]]]:
    """Fetch another git host's tree; read the kept blobs from the object store."""

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        commit = _git_fetch(work, url, rev)
        blobs, sizes = _git_sizes(work)
        kept, dropped = _select(sizes, subpath)
        files = _keep_text(kept, lambda path: _git(work, "cat-file", "blob", blobs[path]), sizes,
                           dropped)
    return commit, files, dropped


def fetch_code(url: str, *, rev: str | None = None, family: str, store: StoreLayout,
               data: DataLayout) -> Source:
    """Save a code repository's tree at one resolved commit.

    Inputs: a repository URL (GitHub, with an optional ``/tree/<ref>/<subpath>``; Gitee or any
    other git host) and an optional ``rev`` (commit, tag or branch; default the default branch).
    GitHub resolves the commit through ``gh`` and reads the codeload tarball at it; another host
    is fetched shallow with a 1 MB blob filter. Text and code files under 1 MB are kept, at most
    20 MB per repository, and the snapshot's manifest lists what was dropped and why. The
    LICENSE nearest the root is saved as its own blob (``derived_from`` the snapshot).
    Returns: a ``code`` source with ``url``, ``rev`` (the full commit), ``blob`` (the snapshot),
    ``framework`` (guessed from requirement files, else imports) and ``spdx`` (a first guess
    from the LICENSE, absent when none was found); or ``code-inaccessible`` with no blob.
    Side effects: blobs and index lines; one fetch event for ``family``.
    """

    store_mod.check_room(store)
    parsed = _github(url)
    try:
        if parsed:
            owner, repo, ref, subpath = parsed
            commit = _gh_json(f"repos/{owner}/{repo}/commits/{rev or ref or 'HEAD'}")["sha"]
            files, dropped = _github_files(owner, repo, commit, subpath)
            base, raw = f"https://github.com/{owner}/{repo}", (
                f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}")
        else:
            subpath = ""
            match = re.fullmatch(r"(https?://[^/]+/[^/]+/[^/]+?)(?:\.git)?(?:/tree/([^/]+)(?:/(.*))?)?/?",
                                 url.strip())
            base = match.group(1) if match else url
            rev = rev or (match.group(2) if match else None)
            subpath = (match.group(3) or "").strip("/") if match else ""
            commit, files, dropped = _git_files(base, rev, subpath)
            raw = None
    except core.Inaccessible as err:
        core._event(family, kind="code", url=url, outcome="inaccessible", note=str(err)[:300])
        return core._inaccessible("code", url)
    if not files:
        core._event(family, kind="code", url=url, rev=commit, outcome="empty", dropped=len(dropped))
        return core._inaccessible("code", url)
    return _save_tree(files, dropped, url=base if not subpath else f"{base}/tree/{commit}/{subpath}",
                      rev=commit, raw=raw, subpath=subpath, family=family, store=store, data=data)


# --- Hugging Face Hub -------------------------------------------------------------------------


def _hf_api(repo: str, rev: str | None) -> dict[str, Any]:
    url = (f"https://huggingface.co/api/models/{repo}/revision/"
           f"{urllib.parse.quote(rev or 'main', safe='')}?blobs=true")
    got = core._http(url, retry_on=(500, 502, 503, 504))
    if got.status in (401, 403, 404, 410):
        raise core.Inaccessible(f"hub {repo}@{rev or 'main'}: HTTP {got.status}")
    if got.status != 200:
        raise core.FetchError(f"hub {repo}: HTTP {got.status}")
    return json.loads(got.body)


def _auto_map_repos(config: Mapping[str, Any]) -> list[str]:
    """Other repositories an ``auto_map`` reaches (``other/repo--module.Class`` references)."""

    refs: list[str] = []
    for value in (config.get("auto_map") or {}).values():
        refs += value if isinstance(value, list) else [value]
    return sorted({r.split("--", 1)[0] for r in refs if isinstance(r, str) and "--" in r})


def _hf_why(name: str, size: int, total: int) -> str | None:
    """Why a Hub file is left out, or ``None`` to keep it."""

    if name.lower().endswith(WEIGHTS):
        return "weights"
    if not HF_KEEP.fullmatch(name) or name.lower().endswith(BINARY):
        return "not_kept"
    if size > HF_FILE_CAP:
        return "over_file_cap"
    return "over_total_cap" if total + size > 2 * HF_FILE_CAP else None


def _hf_snapshot(repo: str, rev: str | None) -> tuple[dict[str, Any], dict[str, bytes],
                                                      list[dict[str, Any]]]:
    """Resolve ``repo@rev`` and download its card, configs, tokenizer files and Python files."""

    info = _hf_api(repo, rev)
    files: dict[str, bytes] = {}
    dropped: list[dict[str, Any]] = []
    for sibling in sorted(info.get("siblings", []), key=lambda s: _priority(s["rfilename"])):
        name, size = sibling["rfilename"], sibling.get("size") or 0
        why = _hf_why(name, size, sum(map(len, files.values())))
        got = None if why else core._http(f"https://huggingface.co/{repo}/resolve/{info['sha']}/"
                                     f"{urllib.parse.quote(name)}", limit=HF_FILE_CAP)
        if got is not None and got.status != 200:
            why = f"http_{got.status}"
        if why or got is None:
            dropped.append({"path": name, "bytes": size, "why": why})
        else:
            files[name] = got.body
    return info, files, dropped


def _hf_weights(info: Mapping[str, Any]) -> tuple[str | None, str]:
    names = [s["rfilename"].lower() for s in info.get("siblings", [])]
    for framework, hints in HF_WEIGHT_HINTS:
        if any(h in n for n in names for h in hints):
            return framework, "found"
    return None, "not_found"


def _hf_config(files: Mapping[str, bytes]) -> dict[str, Any]:
    """``config.json`` with the tokenizer config's ``auto_map`` merged in."""

    config = json.loads(files["config.json"]) if "config.json" in files else {}
    tokenizer = json.loads(files["tokenizer_config.json"]) if "tokenizer_config.json" in files \
        else {}
    return {**config, "auto_map": {**(config.get("auto_map") or {}),
                                   **(tokenizer.get("auto_map") or {})}}


def _hf_repo(name: str, at: str | None, *, card: bool, family: str, store: StoreLayout,
             data: DataLayout) -> tuple[list[Source], list[str]]:
    """Save one Hub repository; return its sources and the other repositories it reaches."""

    url = f"https://huggingface.co/{name}"
    try:
        info, files, dropped = _hf_snapshot(name, at)
    except core.Inaccessible as err:
        core._event(family, kind="hf", url=url, outcome="inaccessible", note=str(err)[:300])
        return [core._inaccessible("code", url)], []
    raw = f"{url}/resolve/{info['sha']}"
    sources = []
    if card and "README.md" in files:
        line = core._put_bytes(files["README.md"], store=store, data=data,
                          mime="text/markdown; charset=utf-8", url=f"{raw}/README.md")
        sources.append(Source(sid=f"page-{line.sha256[:8]}", role="page", url=url,
                              rev=info["sha"], blob=line.sha256, text=line.sha256, fetched=now()))
    if "config.json" in files:
        core._put_bytes(files["config.json"], store=store, data=data, mime="application/json",
                   url=f"{raw}/config.json")
    config = _hf_config(files)
    snapshot = _save_tree(files, dropped, url=url, rev=info["sha"], raw=raw, subpath="",
                          family=family, store=store, data=data)
    framework, weights = _hf_weights(info)
    framework = snapshot.framework or framework or (
        "pytorch" if config.get("architectures") else None)
    sources.append(replace(snapshot, framework=framework, weights=weights))
    return sources, _auto_map_repos(config)


def fetch_hf(repo: str, *, rev: str | None = None, family: str, store: StoreLayout,
             data: DataLayout) -> list[Source]:
    """Save a Hugging Face Hub repository's card, config and code at one resolved commit.

    Inputs: a Hub repo id (or its URL) and an optional ``rev`` (resolved to a commit). Saved:
    ``config.json``, tokenizer files, the card (``README.md``) and every Python file, as one
    snapshot; the card and ``config.json`` also as their own blobs (the model record's
    ``hf.card`` and ``hf.config``). Every repository an ``auto_map`` reaches through
    ``other/repo--module`` is saved the same way at its own resolved revision. Weights are never
    fetched. Returns: a ``page`` source for the card, a ``code`` source for the snapshot (with
    ``framework`` and the ``weights`` flag from the file list), and one ``code`` source per other
    repository; or ``code-inaccessible`` for a gated or missing repository. Side effects: blobs
    and index lines; one fetch event per repository.
    """

    store_mod.check_room(store)
    repo = re.sub(r"^https?://huggingface\.co/", "", repo.strip()).strip("/")
    sources: list[Source] = []
    todo, seen = [(repo, rev)], {repo}
    while todo:
        name, at = todo.pop(0)
        got, others = _hf_repo(name, at, card=name == repo, family=family, store=store, data=data)
        sources += got
        todo += [(other, None) for other in others if other not in seen]
        seen |= set(others)
    return sources
