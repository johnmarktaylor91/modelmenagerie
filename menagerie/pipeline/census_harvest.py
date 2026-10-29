"""Commit-pinned document harvest, separate from census output.

Prepare checkpoints on the coordinator with the existing GitHub client; harvest
only public raw files on the store worker. Neither phase types roster rows.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    append_jsonl,
    census,
    fetch,
    find_paper,
    now,
    read_jsonl,
)
from menagerie.pipeline.records import CensusRow

FILE_CAP = 2**20
FILE_LIMIT = 24
COMMIT = re.compile(r"[0-9a-f]{40}")


def candidate_paths(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Bounded, root-first documents; symlinks and unsafe paths are refused."""
    files = []
    for item in tree.get("tree", []):
        path = item.get("path", "")
        name = PurePosixPath(path).name.lower()
        wanted = (name.startswith(("readme", "license")) or
                  name in ("citation.cff", "citation.bib"))
        safe = (bool(path) and not path.startswith("/") and
                not any(p in ("", ".", "..") for p in path.split("/")))
        if (wanted and safe and item.get("type") == "blob" and
                item.get("mode") != "120000" and 0 <= item.get("size", FILE_CAP + 1) <= FILE_CAP):
            files.append({k: item[k] for k in ("path", "size")})
    return sorted(files, key=lambda f: (f["path"].count("/"), f["path"]))[:FILE_LIMIT]


def selected(root: Path, limit: int | None) -> list[CensusRow]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    rows = [r for r in read_jsonl(CensusRow, root / "census.jsonl") if r.eligible]
    rows.sort(key=lambda r: (hashlib.sha256(r.repo.encode()).hexdigest(), r.id))
    repos = list(dict.fromkeys(r.repo for r in rows))
    keep = set(repos if limit is None else repos[:limit])
    return [r for r in rows if r.repo in keep]


def checkpoint_path(root: Path, repo: str) -> Path:
    return root / "harvest-checkpoints" / f"{hashlib.sha256(repo.encode()).hexdigest()}.json"


def checkpoint(root: Path, repo: str) -> dict[str, Any]:
    value = json.loads(checkpoint_path(root, repo).read_text())
    if value.get("repo") != repo or not COMMIT.fullmatch(value.get("commit", "")):
        raise ValueError("harvest checkpoint needs its repository and resolved commit")
    if value.get("truncated"):
        raise ValueError("truncated harvest checkpoint requires ordinary research")
    return value


def prepare(*, data: DataLayout, out: Path | None = None,
            limit: int | None = None) -> list[dict[str, Any]]:
    """Coordinator-only GitHub I/O; never mutate the census checkpoint."""
    root = out or data.root / "sweeps" / census.SWEEP
    api = census.GitHub()
    results = []
    repos = dict.fromkeys(r.repo for r in selected(root, limit))
    with (root / ".harvest-prepare.lock").open("a") as lock:
        census._lock(lock.fileno())
        for repo in repos:
            try:
                status = _prepare_repo(repo, root, api)
            except (census.CensusError, OSError, ValueError) as exc:
                results.append({"repo": repo, "status": "retry", "reason": type(exc).__name__})
            else:
                results.append({"repo": repo, "status": status})
    return results


def _prepare_repo(repo: str, root: Path, api: census.GitHub) -> str:
    path = checkpoint_path(root, repo)
    if path.exists():
        checkpoint(root, repo)
        return "cached"
    raw = api.get(f"repos/{repo}")
    if not raw or raw.get("archived"):
        return "unavailable"
    branch = raw.get("default_branch")
    if not isinstance(branch, str) or not branch:
        return "no_default_branch"
    commit = api.get(f"repos/{repo}/commits/{quote(branch, safe='')}")
    sha = (commit or {}).get("sha", "")
    if not COMMIT.fullmatch(sha):
        raise census.CensusError("harvest could not resolve a commit")
    tree = api.get(f"repos/{repo}/git/trees/{sha}?recursive=1")
    if tree is None or tree.get("truncated"):
        return "tree_unavailable"
    census._save(path, {"repo": repo, "commit": sha, "branch": branch,
                       "tree": candidate_paths(tree), "prepared": now()})
    return "prepared"


def _document(repo: str, commit: str, path: str, store: StoreLayout,
              data: DataLayout) -> tuple[dict[str, Any], str | None]:
    url = fetch._raw_github(f"https://github.com/{repo}/blob/"
                            f"{commit}/{quote(path, safe='/')}")
    value = {"path": path, "url": url, "refs": [], "titles": []}
    response = fetch._http(url, limit=FILE_CAP)
    if response.status == 404:
        return {**value, "status": "not_found"}, None
    if response.status != 200:
        raise fetch.FetchError(f"harvest document HTTP {response.status}")
    line = fetch._put_bytes(response.body, store=store, data=data, mime="text/plain", url=url)
    value["sha256"] = line.sha256
    try:
        text = response.body.decode("utf-8")
    except UnicodeError:
        return {**value, "status": "not_utf8"}, None
    texts = [(path, text)]
    return {**value, "refs": [ref for ref, _ in find_paper._refs(texts)],
            "paper_candidate": PurePosixPath(path).name.lower() != "citation.cff" or
            bool(find_paper._cff_paper(text)),
            "titles": find_paper._titles(None, texts)}, text


def _documents(row: CensusRow, root: Path, store: StoreLayout,
               data: DataLayout) -> list[tuple[str, str]]:
    point = checkpoint(root, row.repo)
    documents, texts = [], []
    tree = {"tree": [{**p, "type": "blob"} for p in point["tree"]]}
    for item in candidate_paths(tree):
        document, text = _document(row.repo, point["commit"], item["path"], store, data)
        documents.append(document)
        if text is not None:
            texts.append((item["path"], text))
    census._save(root / "harvest-files" / f"{row.id}.json",
                 {**point, "documents": documents,
                  "warnings": [d for d in documents if "status" in d]})
    return texts


def _harvest_one(row: CensusRow, root: Path, store: StoreLayout, data: DataLayout) -> CensusRow:
    texts = _documents(row, root, store, data)
    files = json.loads((root / "harvest-files" / f"{row.id}.json").read_text())["documents"]
    present = [f for f in files if f.get("sha256")]
    readme = next((f for f in present if PurePosixPath(f["path"]).name.lower().startswith("readme")), None)
    citation = next((f for f in present if PurePosixPath(f["path"]).name.lower().startswith("citation.")), None)
    titles = find_paper._titles(None, texts)
    return replace(row, readme_sha=readme["sha256"] if readme else None,
                   citation_file=citation["path"] if citation else None,
                   paper_refs=tuple(ref for ref, _ in find_paper._refs(texts)),
                   paper_title=titles[0] if titles else None, harvested=now())


def _attempt_harvest(row: CensusRow, root: Path, store: StoreLayout, data: DataLayout) -> dict[str, Any]:
    try:
        harvested = _harvest_one(row, root, store, data)
        append_jsonl([harvested], root / "harvest.jsonl")
    except FileNotFoundError:
        return {"id": row.id, "status": "no_checkpoint"}
    except ValueError as exc:
        return {"id": row.id, "status": "invalid_checkpoint", "reason": str(exc)[:160]}
    except (fetch.FetchError, OSError) as exc:
        return {"id": row.id, "status": "retry", "reason": type(exc).__name__}
    return {"id": row.id, "status": "harvested"}


def harvest(*, data: DataLayout, store: StoreLayout, out: Path | None = None,
            limit: int | None = None, write: bool = True) -> list[dict[str, Any]]:
    """Hash-ordered repository limit, with retryable errors and append-only progress."""
    root = out or data.root / "sweeps" / census.SWEEP
    rows = selected(root, limit)
    if not write:
        return [{"id": r.id, "repo": r.repo, "status": "planned"} for r in rows]
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".harvest.lock").open("a") as lock:
        census._lock(lock.fileno())
        file = root / "harvest.jsonl"
        done = {r.id for r in read_jsonl(CensusRow, file)} if file.exists() else set()
        results = []
        for row in rows:
            if row.id in done:
                results.append({"id": row.id, "status": "cached"})
                continue
            results.append(_attempt_harvest(row, root, store, data))
        return results
