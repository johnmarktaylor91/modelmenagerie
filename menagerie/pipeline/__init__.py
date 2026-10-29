"""The rebuilt Menagerie pipeline: four stages joined by committed files.

This package holds the shared contracts every stage module uses: the records
(``records.py``), reading and writing them, the hashes that key them, and where their files
live. ``DataLayout`` reads its root from ``MENAGERIE_DATA`` (default: the package's ``data``
directory); ``StoreLayout`` reads ``MENAGERIE_STORE``. No host name or home path is written
into code.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from menagerie.pipeline.records import SHA256, STABLE_ID, R, Record, RecordError, dumps, from_dict
from menagerie.pipeline import written

__all__ = [
    "NON_TRACE_ROOTS", "DataLayout", "StoreLayout", "append_jsonl", "month_of", "now", "read_json",
    "read_jsonl", "recipe_revision", "shard", "trace_lock_digest", "tree_sha256", "write_json",
]


def read_json(cls: type[R], path: Path) -> R:
    """Read one record from a JSON file."""

    return from_dict(cls, json.loads(Path(path).read_text(encoding="utf-8")))


def write_json(record: Record, path: Path) -> None:
    """Write one record atomically (temporary file, then rename)."""

    path = Path(path)
    body = dumps(record) + "\n"
    prior = written.peek(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    written.rewrote(path, prior, body.encode("utf-8"))


def read_jsonl(parse: type[R] | Callable[[Mapping[str, Any]], Any], path: Path) -> list[Any]:
    """Read a JSONL file with a record class or a parser such as ``parse_ledger_line``."""

    build = functools.partial(from_dict, parse) if isinstance(parse, type) else parse
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [build(json.loads(line)) for line in lines if line.strip()]


def append_jsonl(records: Iterable[Record], path: Path) -> None:
    """Append records to a JSONL file under an exclusive lock, each line in one write call.

    Several hub lanes append to the same index and citation files; the ``flock`` keeps their
    lines whole and in one piece each.
    """

    lines = [(dumps(r) + "\n").encode("utf-8") for r in records]
    if not lines:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("ab", buffering=0) as handle:  # unbuffered: one write(2) per line
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            for line in lines:
                if handle.write(line) != len(line):
                    raise OSError(f"short write appending to {path}")
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    written.appended(path, [line.decode("utf-8") for line in lines])


def now() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``, the format of every record."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def month_of(at: str) -> str:
    """Return the ``YYYY-MM`` shard name for a record timestamp."""

    if not re.match(r"\d{4}-\d{2}", at):
        raise RecordError(f"timestamp {at!r} has no YYYY-MM prefix")
    return at[:7]


# Recipe fields that decide what runs; observations (`modes`, `memory_gib`, `placement`)
# and provenance (`sources`) are excluded so recording them never retires a pass.
REVISION_FIELDS = (
    "kind", "env", "backend", "call", "args", "kwargs", "adapter", "config", "input", "extras",
    "vendor",
)
# Lock packages no trace depends on: the renderer's (``render.EXTRAS``). A package reached only
# from these roots is left out of every recipe revision, so changing it re-renders
# (``renderer_rev``) without retiring a trace.
NON_TRACE_ROOTS = ("pyvips", "pyvips-binary")


State = tuple[str, str | None, str]  # (package name, version or any, extra or the base set)


def _next_states(package: Mapping[str, Any], extra: str) -> list[State]:
    """The states one ``(package, extra)`` state reaches, with edges into non-trace roots cut."""

    if extra:
        edges = list(package.get("optional-dependencies", {}).get(extra, []))
    else:
        groups = package.get("dev-dependencies", {}).values()  # a root's groups install too
        edges = [*package.get("dependencies", []), *(e for group in groups for e in group)]
    return [(e["name"], e.get("version"), x) for e in edges if e["name"] not in NON_TRACE_ROOTS
            for x in ("", *e.get("extra", ()))]


def trace_lock_digest(lock_bytes: bytes) -> str:
    """Return the sha256 over the sorted ``name==version`` pairs a trace can depend on.

    ``lock_bytes`` is a uv lock or a Pixi lock (``_pixi_trace_pairs``). A uv lock's packages are
    walked from the root projects (virtual or editable sources), following extras, with every
    edge into ``NON_TRACE_ROOTS`` cut: a package reached only from those roots is left out.
    Hashes, URLs and the lock's ordering do not count.
    """

    if _PIXI_LOCK.match(lock_bytes):
        return _pairs_digest(_pixi_trace_pairs(lock_bytes))
    import tomllib  # stdlib from 3.11, the version of every environment that hashes a lock

    packages = tomllib.loads(lock_bytes.decode("utf-8")).get("package", [])
    by_name: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        by_name.setdefault(package["name"], []).append(package)
    todo: list[State] = [(p["name"], p["version"], "") for p in packages
                         if {"virtual", "editable"} & set(p.get("source", {}))]
    seen: set[State] = set()
    while todo:
        name, version, extra = todo.pop()
        for package in by_name.get(name, []):
            state = (name, package["version"], extra)
            if version in (None, package["version"]) and state not in seen:
                seen.add(state)
                todo += _next_states(package, extra)
    return _pairs_digest({(name, version) for name, version, _ in seen})


def _pairs_digest(seen: Iterable[tuple[str, str]]) -> str:
    # Preserve the original recipe keys across a TorchLens-only repin. The ledger's
    # separate `tl` field controls trace currency; forwards do not depend on that pin.
    # 2.35.3 is an identity anchor, never an installation requirement.
    pairs = sorted({f"{name}=={'2.35.3' if name == 'torchlens' else version}"
                    for name, version in seen})
    return hashlib.sha256("\n".join(pairs).encode()).hexdigest()


_PIXI_LOCK = re.compile(rb"\s*version:\s*\d")  # a Pixi lock is YAML; a uv lock is TOML
PIXI_LOCK_VERSIONS = frozenset({7})  # the layout `_pixi_graph` reads: entries keyed by artifact URL


def _pixi_trace_pairs(lock_bytes: bytes) -> set[tuple[str, str]]:
    """The ``(name, version)`` pairs of a Pixi lock's default environment a trace can depend on.

    A Pixi lock names no root project, so every package of the environment, on every platform,
    is a root except what ``NON_TRACE_ROOTS`` reach: such a package stays in only when a package
    outside that reach depends on it, every edge into ``NON_TRACE_ROOTS`` cut, as in the uv walk.
    Edges are the Conda ``depends`` and PyPI ``requires_dist`` names, matched by name and by a
    Conda package's PyPI ``purls``, plus each PyPI package's edge to ``python``; markers are
    ignored, since an extra edge only keeps a package in.
    """

    from packaging.utils import canonicalize_name

    graph = _pixi_graph(lock_bytes)
    nodes = graph[0]
    cut = {canonicalize_name(root) for root in NON_TRACE_ROOTS}
    renderer_only = _reach(graph, [i for i, node in enumerate(nodes) if node[0] in cut], set())
    kept = _reach(graph, [i for i in range(len(nodes)) if i not in renderer_only], cut)
    return {nodes[i][:2] for i in kept}


PixiNode = tuple[str, str, set[str]]  # (canonical name, version, names it depends on)
PixiGraph = tuple[list[PixiNode], dict[str, list[int]]]  # nodes, and each name's nodes


def _reach(graph: PixiGraph, start: list[int], skip: set[str]) -> set[int]:
    """The nodes reached from ``start``, never following an edge into a name in ``skip``."""

    nodes, by_name = graph
    todo = list(start)
    found = set(todo)
    while todo:
        for edge in nodes[todo.pop()][2] - skip:
            fresh = [i for i in by_name.get(edge, []) if i not in found]
            found.update(fresh)
            todo += fresh
    return found


def _pixi_graph(lock_bytes: bytes) -> PixiGraph:
    """The default environment's packages and an index from each name or PyPI alias to them."""

    import yaml
    from packaging.utils import canonicalize_name

    from menagerie.pipeline.envs import pixi_identity

    data = yaml.safe_load(lock_bytes.decode("utf-8"))
    env = data.get("environments", {}).get("default") if isinstance(data, dict) else None
    if not isinstance(env, dict) or not isinstance(data.get("packages"), list):
        raise RecordError("a Pixi lock needs a default environment and a package table")
    if data.get("version") not in PIXI_LOCK_VERSIONS:  # another layout would match no package
        raise RecordError(f"Pixi lock version {data.get('version')!r} is not one the trace "
                          f"digest reads ({sorted(PIXI_LOCK_VERSIONS)})")
    wanted = {url for refs in env.get("packages", {}).values() for ref in refs
              for url in (ref.get("conda"), ref.get("pypi")) if url}
    nodes: list[PixiNode] = []
    by_name: dict[str, list[int]] = {}
    for entry in data["packages"]:
        if (entry.get("conda") or entry.get("pypi")) not in wanted:
            continue
        try:
            name, version = pixi_identity(entry)
        except ValueError as exc:
            raise RecordError(f"Pixi lock entry unreadable: {exc}") from exc
        aliases = {canonicalize_name(name), *(canonicalize_name(p.split("/", 1)[1].split("?")[0])
                                              for p in entry.get("purls", [])
                                              if p.startswith("pkg:pypi/"))}
        for alias in aliases:
            by_name.setdefault(alias, []).append(len(nodes))
        nodes.append((canonicalize_name(name), version, _pixi_edges(name, entry)))
    if not nodes:
        raise RecordError("the Pixi lock's default environment matches no locked package")
    return nodes, by_name


def _pixi_edges(name: str, entry: Mapping[str, Any]) -> set[str]:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    edges = {canonicalize_name(dep.split()[0]) for dep in entry.get("depends", [])}
    if "pypi" in entry:  # a wheel runs on the environment's interpreter, an unlisted edge
        edges.add("python")
    for spec in entry.get("requires_dist", []):
        try:
            edges.add(canonicalize_name(Requirement(spec).name))
        except InvalidRequirement as exc:
            raise RecordError(f"{name}: unreadable requirement {spec!r}") from exc
    return edges


def recipe_revision(recipe: Mapping[str, Any], adapter: bytes | None, lock: bytes) -> str:
    """Return the 12-hex revision of the run-deciding fields, adapter file and trace packages.

    ``lock`` is the environment's uv or Pixi lock; only ``trace_lock_digest(lock)`` enters the
    revision.
    """

    payload = {k: recipe.get(k) for k in REVISION_FIELDS}
    if recipe.get("code"):  # only when set, so every recipe without extra trees keeps its revision
        payload["code"] = list(recipe["code"])
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0" + hashlib.sha256(adapter or b"").digest())
    digest.update(b"\0" + trace_lock_digest(lock).encode())
    return digest.hexdigest()[:12]


def tree_sha256(path: Path) -> str:
    """Return the sha256 of a file, or of a directory (a ``.tlspec``) over its sorted files."""

    path = Path(path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        digest.update(f"{rel}\0{hashlib.sha256(item.read_bytes()).hexdigest()}\n".encode())
    return digest.hexdigest()



def shard(key: str) -> str:
    """Return the two-character shard: ``m3486`` -> ``34``, ``f-aimv2`` -> ``ai``."""

    body = key[2:] if key.startswith("f-") else key[1:] if re.fullmatch(STABLE_ID, key) else key
    return body[:2].lower() or "_"


@dataclass(frozen=True)
class DataLayout:
    """Paths of the committed data files under ``MENAGERIE_DATA``."""

    root: Path

    @classmethod
    def from_env(cls) -> DataLayout:
        default = Path(__file__).resolve().parents[1] / "data"
        return cls(Path(os.environ.get("MENAGERIE_DATA") or default))

    @property
    def roster(self) -> Path:
        return self.root / "crawl_roster.jsonl"

    @property
    def routing_manifest(self) -> Path:
        return self.root / "routing_manifest.tsv"

    @property
    def sweep_index(self) -> Path:
        return self.root / "sweeps" / "index.jsonl"

    @property
    def milestones(self) -> Path:
        return self.root / "milestones.txt"

    def sweep(self, sweep: str) -> Path:
        return self.root / "sweeps" / f"{sweep}.jsonl"

    def packet(self, family: str) -> Path:
        return self.root / "research" / shard(family) / f"{family}.json"

    def model(self, stable_id: str) -> Path:
        return self.root / "models" / shard(stable_id) / f"{stable_id}.json"

    def adapter(self, family: str) -> Path:
        return self.root / "adapters" / f"{family}.py"

    def describe(self, family: str) -> Path:
        return self.root / "describe" / shard(family) / f"{family}.json"

    def ledger(self, month: str) -> Path:
        return self.root / "ledger" / f"{month}.jsonl"

    def cite(self, month: str) -> Path:
        return self.root / "cite" / f"{month}.jsonl"

    def store_index(self, month: str) -> Path:
        return self.root / "store-index" / f"{month}.jsonl"

    def release(self, number: int) -> Path:
        return self.root / "releases" / f"v{number}.json"


@dataclass(frozen=True)
class StoreLayout:
    """Paths of content-addressed blobs under ``MENAGERIE_STORE``."""

    root: Path

    @classmethod
    def from_env(cls) -> StoreLayout:
        root = os.environ.get("MENAGERIE_STORE")
        if not root:
            raise RecordError("MENAGERIE_STORE is not set; it names the store root")
        return cls(Path(root))

    def blob(self, sha256: str) -> Path:
        if not re.fullmatch(SHA256, sha256):
            raise RecordError(f"{sha256!r} is not a sha256")
        return self.root / "sha256" / sha256[:2] / sha256
