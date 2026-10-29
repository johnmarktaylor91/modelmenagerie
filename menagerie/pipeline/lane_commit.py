"""Commit the lane results the pipeline wrote to the data checkout (the tick's ``lanes`` step).

2a and 2b lanes, the mini relay's ingest and review write packets, adapters, model records, roster
edits, stable ids, citation, ledger and store-index lines in their own processes, so ``written``
never records them and ``tick_commit`` never commits them. Each pass this step commits by name
(``git-commit-paths``) every changed file under the data root that

- no live or queued lane holds: a packet, adapter or model record whose family (or model id) is a
  key of a lane's lease waits for a later pass, so a lane's files go in once the lane has ended;
- reads whole: a JSON file parses, every JSONL line parses and the last one is finished, a Python
  adapter compiles; one that does not is left uncommitted and listed (``unreadable``);
- still exists: a deleted file is listed (``gone``) and left for a person;
- the commit hooks accept: a refused batch is halved until the refused files are found, and each is
  kept with its content hash (state ``lane_refused``), tried again only once it changes.

Files that carry no family (the roster, stable ids, citation, ledger and store-index shards) go in
whenever they read whole: what a live lane appended there is already a finished row. Bytecode is
never committed, and ``drive._changed`` already drops ``.<name>.tmp`` files.
"""

from __future__ import annotations

import ast
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BATCH = 200  # paths per git-commit-paths call
MAX_FILES = 2000  # files one pass commits at most; the rest wait for the next pass
SPARE_S = 60.0  # no new batch starts when the pass has less than this left

Commit = Callable[[Path, Sequence[str], str, float], None]
Digest = Callable[[Path, str], "str | None"]


@dataclass
class Plan:
    """The changed data files sorted by what happens to them this pass."""

    ready: list[str] = field(default_factory=list)
    held: dict[str, str] = field(default_factory=dict)  # path: the lease key that holds it
    unreadable: dict[str, str] = field(default_factory=dict)  # path: why
    gone: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)  # unchanged since the hooks refused it


def _jsonl_why(body: bytes) -> str | None:
    for n, line in enumerate(body.decode("utf-8").splitlines(), 1):
        try:
            if line.strip():
                json.loads(line)
        except ValueError as err:
            return f"line {n}: {err}"[:200]
    return "its last line is unfinished" if body and not body.endswith(b"\n") else None


def check(path: Path) -> tuple[Any, str | None]:
    """``(parsed JSON or None, why it does not read whole or None)`` for one data file."""

    body = path.read_bytes()
    try:
        if path.suffix == ".json":
            return json.loads(body), None
        if path.suffix == ".jsonl":
            return None, _jsonl_why(body)
        if path.suffix == ".py":
            ast.parse(body, filename=path.name)
    except (ValueError, SyntaxError) as err:  # UnicodeDecodeError and JSONDecodeError included
        return None, f"{type(err).__name__}: {err}"[:200]
    return None, None


def keys_of(rel: str, parsed: Any) -> tuple[str, ...]:
    """The lease keys a file under the data root belongs to (``rel`` relative to that root):
    a packet's and an adapter's family from its name, a model record's family and id."""

    parts = Path(rel).parts
    stem = Path(rel).stem
    if parts[0] == "research" and rel.endswith(".json"):
        return (stem,)
    if parts[0] == "adapters" and rel.endswith(".py"):
        return (stem.split("--")[0],)
    if parts[0] == "models" and isinstance(parsed, Mapping):
        return tuple(str(parsed[k]) for k in ("family", "id") if parsed.get(k))
    return ()


def plan(repo: Path, data_rel: str, changed: Iterable[str], held_keys: set[str],
         refused: Mapping[str, str | None], digest: Digest) -> Plan:
    """Sort the changed repository-relative paths under ``data_rel`` (the data root, relative to
    ``repo``) into ready, held, unreadable, gone and still refused."""

    out = Plan()
    for path in sorted(set(changed)):
        if path.startswith(f"{data_rel}/") and "__pycache__" not in Path(path).parts:
            _sort(out, repo, path, data_rel, held_keys, refused, digest)
    return out


def _sort(out: Plan, repo: Path, path: str, data_rel: str, held_keys: set[str],
          refused: Mapping[str, str | None], digest: Digest) -> None:
    target = repo / path
    if not target.is_file():
        out.gone.append(path)
        return
    if path in refused and digest(repo, path) == refused[path]:
        out.refused.append(path)
        return
    parsed, why = check(target)
    holder = None if why else next(
        (k for k in keys_of(path[len(data_rel) + 1:], parsed) if k in held_keys), None)
    if why:
        out.unreadable[path] = why
    elif holder:
        out.held[path] = holder
    else:
        out.ready.append(path)


def _message(paths: Sequence[str], data_rel: str) -> str:
    tops: dict[str, int] = {}
    for path in paths:
        top = Path(path[len(data_rel) + 1:]).parts[0]
        tops[top] = tops.get(top, 0) + 1
    counts = ", ".join(f"{k} {n}" for k, n in sorted(tops.items()))
    return f"data: tick committed lane results ({len(paths)} files: {counts})"


def _commit_split(repo: Path, paths: Sequence[str], data_rel: str, commit: Commit,
                  deadline: float, refused: list[str]) -> list[str]:
    """Commit ``paths``; a batch the hooks refuse is halved until each refused file stands alone
    (added to ``refused``). Returns: the paths committed. An ``index.lock`` failure is raised."""

    try:
        commit(repo, paths, _message(paths, data_rel), deadline)
        return list(paths)
    except OSError as err:
        if "index.lock" in str(err):
            raise
        if len(paths) == 1:
            refused.append(paths[0])
            return []
    half = len(paths) // 2
    return [*_commit_split(repo, paths[:half], data_rel, commit, deadline, refused),
            *_commit_split(repo, paths[half:], data_rel, commit, deadline, refused)]


def commit_ready(repo: Path, ready: Sequence[str], data_rel: str, commit: Commit, deadline: float,
                 *, batch: int = BATCH, limit: int = MAX_FILES,
                 clock: Callable[[], float] = time.monotonic) -> tuple[list[str], list[str]]:
    """Commit up to ``limit`` ready paths in batches while the pass has time.
    Returns: ``(committed, refused by the hooks)``."""

    committed: list[str] = []
    refused: list[str] = []
    todo = list(ready)[:limit]
    for start in range(0, len(todo), batch):
        if clock() + SPARE_S > deadline:
            break
        chunk = todo[start:start + batch]
        committed += _commit_split(repo, chunk, data_rel, commit, deadline, refused)
    return committed, refused


def step(repo: Path, data_rel: str, changed: Iterable[str], held_keys: set[str],
         state: dict[str, Any], *, commit: Commit, digest: Digest, deadline: float,
         dry_run: bool = False, limit: int = MAX_FILES) -> dict[str, Any]:
    """One pass of the ``lanes`` step: plan, commit, and keep the hook-refused files' hashes in
    ``state["lane_refused"]``. Returns: counts, plus the paths a person may need to look at."""

    known = state.get("lane_refused") or {}
    known = known if isinstance(known, dict) else {}
    found = plan(repo, data_rel, changed, held_keys, known, digest)
    out: dict[str, Any] = {"ready": len(found.ready), "held_by_lane": len(found.held)}
    if found.held:
        out["holding_keys"] = sorted(set(found.held.values()))[:20]
    if found.unreadable:
        out["unreadable"] = dict(sorted(found.unreadable.items())[:20])
    if found.gone:
        out["gone"] = found.gone[:20]
    if dry_run:
        return {**out, "would_commit": found.ready[:20]}
    committed, refused = commit_ready(repo, found.ready, data_rel, commit, deadline, limit=limit)
    still = {p: known[p] for p in found.refused}
    state["lane_refused"] = {**still, **{p: digest(repo, p) for p in refused}}
    out["committed"] = len(committed)
    out["left"] = len(found.ready) - len(committed) - len(refused)
    if state["lane_refused"]:
        out["refused"] = sorted(state["lane_refused"])[:20]
    return out
