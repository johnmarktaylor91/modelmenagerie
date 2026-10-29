"""Pack and offload the worker store's blobs, only behind full-restore-verified box snapshots.

Rung 3 store gate (JMT 2026-10-08 4:40 PM PT, ``decisions.md`` "rung 2 check-in decisions"
item 6). Two verbs run on the worker that holds the store:

- ``offload POLICY``: delete the local copies of blobs a policy selects. ``superseded``: older
  trace saves of a model, their portable saves and their render outputs; ``tiles``: zoom tiles;
  ``portable-rendered``: portable saves whose trace has a passing render; ``portable-twin``: a
  portable save put after a trace save that is itself a portable save of the same capture (the
  worker wrote both when the runnable save was refused, before 2026-10-28), when a verified
  snapshot holds both; render reads the trace save instead.
- ``pack``: replace each current trace save (a ``.tlspec`` directory) with its deterministic
  ``tar.gz`` at ``<sha>.tar.gz``; ``store.get`` reads it through the unpack cache.
- ``pack-views``: replace each view file (a trace's summary JSON, a render's SVG, HTML, Netron and
  Model Explorer JSON) with its deterministic gzip at ``<sha>.gz``; ``store.get`` reads it the
  same way. A gzipped blob is never offloaded (it is about a thirtieth of the raw size).

Every action first proves the blob is safe to drop locally: an index line names it, a catalog
manifest of this host with ``full_restore_sha256`` holds it with per-file hashes whose tree hash
is its name, and the live copy has the same file count and bytes and hashes to its name. A blob
any kept ledger row or committed data file names is never selected, and a superseded trace is
only one older than the model's latest, latest passing and latest non-infra saves of its mode and
capture, and not its deciding save at the record's current recipe and the TorchLens pin.
A pack is read back before the raw tree goes. Deleting a blob also deletes its hard links in the
sealed box batches under ``links`` (they pin its bytes on disk). Each action appends one index
line restating the blob with ``location`` ``box:<snapshot>`` or ``encoding`` ``tar.gz``.

Trace saves are written packed (``store.put``), so an offload also takes a packed blob: its tree
read from the stream must hash to its name, and a snapshot must hold the same packed file (by
sha256) or the raw tree; the placement line's ``encoding`` names the form a restore brings back.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout, append_jsonl, read_jsonl, tree_sha256
from menagerie.pipeline import ledger as ledger_mod
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import SHA256, LedgerRow, RenderRow, StoreIndexLine, dumps

POLICIES = ("superseded", "tiles", "portable-rendered", "portable-twin")
TWIN = "portable-twin"  # a candidate kind as well: its pair is checked as a twin
TLSPEC = "application/x-tlspec"
PAIR_SECONDS = 120  # a portable save is put right after its runnable save
PAIR_KEYS = ("n_layers", "input_hash", "model_structure_hash")
HEX = re.compile(SHA256)
PLACEMENT_SHARD = "placement.jsonl"
PACKED = store_mod.PACKED_SUFFIX
GZIPPED = store_mod.GZIP_SUFFIX
VIEW_KINDS = frozenset({"svg", "html", "netron", "model_explorer"})  # render outputs gzipped
PACKS = {"pack": "tar.gz", "pack-views": "gzip"}  # pack policy -> the form it writes


class ReadBackError(RuntimeError):
    """A blob read back differently from its name or its snapshot manifest: stop everything."""


@dataclass(frozen=True)
class Held:
    snapshot: str
    files: int
    bytes: int
    digest: str | None = None  # a packed blob's file sha256 (its tree is checked live)


@dataclass(frozen=True)
class Proof:
    """The live copy that may go and the snapshot copy that holds it; ``encoding`` is the form
    the snapshot holds (``None`` raw as put, ``raw`` or ``tar.gz``), what a restore brings back."""

    path: Path
    held: Held
    encoding: str | None = None


@dataclass(frozen=True)
class Candidate:
    sha256: str
    kind: str  # trace, portable, portable-twin, render:<kind>, tiles
    model: str
    pair: str | None = None  # a portable save's trace save, checked by manifest


@dataclass
class Tally:
    done: list[dict] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bytes: int = 0


# --- proof: the catalog ---------------------------------------------------------------------

def _manifests(catalog: Path):
    """Each catalog manifest in ``catalog`` (a copy may travel gzipped)."""

    for path in sorted(p for p in Path(catalog).iterdir() if p.name.endswith((".json", ".json.gz"))):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            yield json.load(handle)


def _verified_snapshot(manifest: dict, host: str) -> str | None:
    """The snapshot id of a full-restore-verified result manifest of ``host``, else ``None``."""

    sid = (manifest.get("archive") or {}).get("snapshot") or ""
    ok = ((manifest.get("created_by") or {}).get("host") == host and HEX.fullmatch(sid)
          and (manifest.get("verification") or {}).get("status") == "full_restore_sha256"
          and isinstance(manifest.get("files"), list))
    return sid if ok else None


def held_blobs(catalog: Path, host: str, wanted: Iterable[str]) -> dict[str, Held]:
    """Blobs of ``wanted`` that a full-restore-verified result manifest of ``host`` holds, by
    store name: ``<sha>`` when every file of ``sha256/<xx>/<sha>`` is listed with its sha256 and
    their tree hash is the name; ``<sha>.tar.gz`` (a packed blob) with its file's sha256."""

    names = {name for sha in wanted for name in (sha, sha + PACKED)}
    found: dict[str, Held] = {}
    for manifest in _manifests(catalog):
        sid = _verified_snapshot(manifest, host)
        blobs = _manifest_blobs(manifest["files"], names) if sid else {}
        for name, files in blobs.items():
            held = None if name in found else _held(sid, name, files)  # type: ignore[arg-type]
            if held is not None:
                found[name] = held
    return found


def _held(sid: str, name: str, files: dict[str, tuple[str, int]]) -> Held | None:
    """What one manifest proves about one store name, or ``None`` when it does not hold it."""

    if name.endswith(PACKED):
        return Held(sid, 1, files[""][1], files[""][0]) if set(files) == {""} else None
    if _tree_sha(files) != name:
        return None
    return Held(sid, len(files), sum(size for _, size in files.values()))


def _tree_sha(files: dict[str, tuple[str, int]]) -> str:
    if set(files) == {""}:  # a file blob: the manifest entry is the blob itself
        return files[""][0]
    return store_mod.manifest_digest({rel: digest for rel, (digest, _) in files.items()})


def _manifest_blobs(entries: list, wanted: set[str]) -> dict[str, dict[str, tuple[str, int]]]:
    blobs: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)
    for entry in entries:
        parts = str(entry.get("path", "")).split("/") if isinstance(entry, dict) else []
        if len(parts) < 3 or parts[0] != "sha256" or parts[2] not in wanted:
            continue
        rel = "/".join(parts[3:]) if len(parts) > 3 else ""
        blobs[parts[2]][rel] = (str(entry.get("sha256")), int(entry.get("bytes", -1)))
    return blobs


# --- selection: the committed data ----------------------------------------------------------

@dataclass(frozen=True)
class Data:
    index: list[StoreIndexLine]  # first line per sha, in index order
    rows: list[Any]
    current_rev: dict[str, str]
    pinned: set[str]  # shas any committed file outside the ledger and index names
    tl: str | None = None  # the TorchLens pin; ``None`` keeps a deciding save for every version


def load(data: DataLayout, extra: Sequence[Path] = ()) -> Data:
    first: dict[str, StoreIndexLine] = {}
    for shard in sorted((data.root / "store-index").glob("*.jsonl")):
        for line in read_jsonl(StoreIndexLine, shard):
            first.setdefault(line.sha256, line)
    rows = [r for r in ledger_mod.read_all(data=data) if isinstance(r, (LedgerRow, RenderRow))]
    revs = {}
    for path in (data.root / "models").glob("*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        rev = (record.get("recipe") or {}).get("rev")
        if isinstance(rev, str):
            revs[record.get("id", path.stem)] = rev
    from menagerie.pipeline.publish import torchlens_pin

    return Data(list(first.values()), rows, revs, _pinned(data.root, extra), torchlens_pin())


def _pinned(root: Path, extra: Sequence[Path]) -> set[str]:
    skip = {root / "ledger", root / "store-index"}
    found: set[str] = set()
    for base in (root, *extra):
        for path in Path(base).rglob("*"):
            if path.is_file() and not any(s in path.parents for s in skip) \
                    and path.suffix in (".json", ".jsonl", ".tsv", ".csv", ".txt", ".md"):
                found.update(HEX.findall(path.read_text(encoding="utf-8", errors="replace")))
    return found


def _latest(rows: Sequence[Any]) -> Any:
    return max(reversed(rows), key=lambda r: r.at)  # ledger order breaks ties, as ``ledger``


def _trace_groups(traces: Sequence[LedgerRow], revs: dict[str, str],
                  tl: str | None) -> tuple[dict, dict]:
    by_kind: dict[tuple, list[LedgerRow]] = defaultdict(list)
    deciding: dict[tuple, list[LedgerRow]] = defaultdict(list)
    for row in traces:
        key = (row.id, row.mode or "eval", row.capture or "full")
        by_kind[key].append(row)
        if row.recipe_rev == revs.get(row.id) and row.outcome != "infra" and tl in (None, row.tl):
            deciding[(*key, row.tl)].append(row)
    return by_kind, deciding


def _kept_traces(traces: Sequence[LedgerRow], revs: dict[str, str],
                 tl: str | None = None) -> set[str]:
    """Trace saves no policy may drop: per model, mode and capture the latest, latest passing and
    latest non-infra saves, and the deciding save at the current recipe and TorchLens pin."""

    by_kind, deciding = _trace_groups(traces, revs, tl)
    kept = {_latest(rows).trace for rows in deciding.values()}
    for rows in by_kind.values():
        picks = (rows, [r for r in rows if r.outcome == "pass"],
                 [r for r in rows if r.outcome != "infra"])
        kept.update(_latest(p).trace for p in picks if p)
    return kept


def _refs(row: Any, drop: set[str]) -> set[str]:
    return set(HEX.findall(dumps(row))) - drop


def _portables(data: Data, claimed: set[str]) -> dict[str, str]:
    """Runnable save sha -> the portable save put right after it (the next index line)."""

    pairs = {}
    for this, after in zip(data.index, data.index[1:]):
        if (this.mime == TLSPEC and this.sha256 in claimed and after.mime == TLSPEC
                and after.sha256 not in claimed
                and 0 <= _seconds(this.fetched, after.fetched) <= PAIR_SECONDS):
            pairs[this.sha256] = after.sha256
    return pairs


def _seconds(start: str, end: str) -> float:
    from datetime import datetime

    def parse(text: str) -> datetime:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))

    return (parse(end) - parse(start)).total_seconds()


@dataclass(frozen=True)
class View:
    """What every policy reads: trace and render rows, the kept and superseded saves."""

    owner: dict[str, str]  # trace save sha -> model
    renders: list[RenderRow]
    kept: set[str]
    old: dict[str, str]  # superseded trace save sha -> model
    tiles: set[str]
    portable: dict[str, str]  # runnable save sha -> its portable save
    summaries: dict[str, str] = field(default_factory=dict)  # trace summary sha -> model
    ready: set[str] = field(default_factory=set)  # trace saves whose row says runnable ``ready``

    @classmethod
    def of(cls, data: Data) -> View:
        traces = [r for r in data.rows
                  if isinstance(r, LedgerRow) and r.stage == "trace" and r.trace]
        renders = [r for r in data.rows if isinstance(r, RenderRow)]
        kept = _kept_traces(traces, data.current_rev, data.tl)
        owner = {r.trace: r.id for r in traces}
        tiles = {a.sha256 for r in renders for a in r.renders if a.kind == "tiles"}
        summaries = {r.summary: r.id for r in data.rows
                     if isinstance(r, LedgerRow) and r.stage == "trace" and r.summary}
        return cls(owner, renders, kept, {s: m for s, m in owner.items() if s not in kept}, tiles,
                   _portables(data, set(owner)), summaries, _ready(traces))


def _ready(traces: Sequence[LedgerRow]) -> set[str]:
    """Trace saves whose row says the runnable save round-tripped (``ready``)."""

    return {str(r.trace) for r in traces if r.runnable == "ready"}


def _keep(data: Data, view: View) -> set[str]:
    """Every sha a kept row or committed file names. Superseded traces' saves and render
    outputs, and zoom tiles (the gate drops them), are left out; their summaries and logs stay."""

    keep = set(data.pinned) | view.kept
    for row in data.rows:
        dropped = {row.trace} if isinstance(row, LedgerRow) and row.trace in view.old else set()
        if isinstance(row, RenderRow) and row.trace in view.old:
            dropped = {a.sha256 for a in row.renders} | {row.trace}
        keep |= _refs(row, dropped | view.tiles)
    return keep


def _superseded(view: View) -> list[Candidate]:
    found = [Candidate(sha, "trace", model) for sha, model in view.old.items()]
    found += [Candidate(view.portable[sha], "portable", model, sha)
              for sha, model in view.old.items() if sha in view.portable]
    found += [Candidate(a.sha256, f"render:{a.kind}", r.id)
              for r in view.renders if r.trace in view.old for a in r.renders]
    return found


def _tiles(view: View) -> list[Candidate]:
    return [Candidate(a.sha256, "tiles", r.id) for r in view.renders for a in r.renders
            if a.sha256 in view.tiles]


def _rendered_portables(view: View) -> list[Candidate]:
    latest: dict[str, RenderRow] = {}
    for row in view.renders:
        if row.trace not in latest or row.at >= latest[row.trace].at:
            latest[row.trace] = row
    drawn = {sha for sha, row in latest.items() if row.outcome == "pass"}
    return [Candidate(view.portable[sha], "portable", model, sha)
            for sha, model in view.owner.items() if sha in view.portable and sha in drawn]


def _twin_portables(view: View) -> list[Candidate]:
    """Portable saves put after a trace save whose row does not say ``ready`` (a ready save is
    runnable, so its pair is no twin and its packed manifest is not read); ``_prove`` keeps all
    but true twins, by manifest."""

    return [Candidate(view.portable[sha], TWIN, model, sha)
            for sha, model in view.owner.items() if sha in view.portable and sha not in view.ready]


def _trace_saves(view: View) -> list[Candidate]:
    return [Candidate(sha, "trace", model) for sha, model in view.owner.items()]


def _view_files(view: View) -> list[Candidate]:
    found = [Candidate(sha, "summary", model) for sha, model in view.summaries.items()]
    found += [Candidate(a.sha256, f"render:{a.kind}", r.id) for r in view.renders
              for a in r.renders if a.kind in VIEW_KINDS]
    return found


POLICY = {"superseded": _superseded, "tiles": _tiles, "portable-rendered": _rendered_portables,
          TWIN: _twin_portables, "pack": _trace_saves, "pack-views": _view_files}


def select(policy: str, data: Data) -> list[Candidate]:
    """The blobs ``policy`` would drop locally (``pack``: would pack), before any proof against
    the store or catalog. A pack drops no content, so only offload policies honour the keep set."""

    if policy not in POLICY:
        raise ValueError(f"unknown policy {policy!r}; one of {sorted(POLICY)}")
    view = View.of(data)
    keep = set() if policy in PACKS else _keep(data, view)
    mimes = {line.sha256: line.mime for line in data.index}
    wanted = store_mod.GZIP_FILES if policy == "pack-views" else None
    seen: dict[str, Candidate] = {}
    for item in POLICY[policy](view):
        mime = mimes.get(item.sha256)
        if mime is not None and (wanted is None or mime in wanted) and item.sha256 not in keep:
            seen.setdefault(item.sha256, item)
    # A portable save goes before the runnable save it is checked against.
    return sorted(seen.values(), key=lambda c: (c.kind != "portable", c.sha256))


# --- action: the worker store ---------------------------------------------------------------

def _live(path: Path) -> tuple[int, int]:
    if path.is_symlink():
        return -1, -1
    if path.is_file():
        return 1, path.stat().st_size
    files = [p for p in path.rglob("*") if p.is_file() or p.is_symlink()]
    if any(p.is_symlink() for p in files):
        return -1, -1
    return len(files), sum(p.stat().st_size for p in files)


Verdict = tuple[str | None, Proof | None]


def _prove(item: Candidate, store: StoreLayout, held: dict[str, Held],
           pack: bool = False) -> Verdict:
    """Why ``item`` must stay, else ``(None, proof)`` when its local copy may go: the raw blob,
    or for an offload its packed form. Raises ``ReadBackError`` when the live copy is not what its
    name and snapshot say."""

    raw = store.blob(item.sha256)
    packed = store_mod.packed_path(item.sha256, store=store)
    if not raw.exists():
        if store_mod.gzipped_path(item.sha256, store=store).is_file():
            return "already_packed" if pack else "gzipped_kept", None
        if not packed.is_file():
            return "absent", None
        return ("already_packed", None) if pack else _prove_packed(item, store, packed, held)
    proof = held.get(item.sha256)
    if proof is None:
        return "not_in_verified_snapshot", None
    if _live(raw) != (proof.files, proof.bytes):
        return "differs_from_snapshot", None
    unpaired = _pair_problem(item, store, held)
    if unpaired is not None:
        return unpaired, None
    actual = tree_sha256(raw)
    if actual != item.sha256:
        raise ReadBackError(f"live blob {item.sha256} hashes to {actual}")
    return None, Proof(raw, proof)


def _prove_packed(item: Candidate, store: StoreLayout, packed: Path,
                  held: dict[str, Held]) -> Verdict:
    """A packed blob may go when a snapshot holds the same packed file, or its raw tree."""

    boxed, tree = held.get(item.sha256 + PACKED), held.get(item.sha256)
    if boxed is None and tree is None:
        return "not_in_verified_snapshot", None
    unpaired = _pair_problem(item, store, held)
    if unpaired is not None:
        return unpaired, None
    stats = store_mod.packed_stats(packed)
    if stats[0] != item.sha256:
        raise ReadBackError(f"live packed blob {item.sha256} unpacks to {stats[0]}")
    if boxed is not None and (packed.stat().st_size, _file_sha(packed)) == (boxed.bytes,
                                                                          boxed.digest):
        return None, Proof(packed, boxed, "tar.gz")
    if tree is not None and stats[1:] == (tree.files, tree.bytes):
        return None, Proof(packed, tree, "raw")
    return "differs_from_snapshot", None


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(sha: str, store: StoreLayout) -> dict:
    """A trace save's ``manifest.json``, from the raw tree or read from the packed stream."""

    raw = store.blob(sha)
    if raw.is_dir():
        return json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    body = store_mod.packed_member(store_mod.packed_path(sha, store=store), "manifest.json")
    if body is None:
        raise ValueError(f"trace save {sha} has no manifest.json")
    return json.loads(body)


def _pair_problem(item: Candidate, store: StoreLayout, held: dict[str, Held]) -> str | None:
    """Why a portable save's pair keeps it, else ``None``. A rendered or superseded portable save
    needs a non-portable pair of the same capture; a twin needs a portable pair of the same
    capture that a verified snapshot holds too (raw or packed), so either can come back."""

    if item.pair is None:
        return None
    twin = item.kind == TWIN
    if twin and item.pair not in held and item.pair + PACKED not in held:
        return "pair_not_in_verified_snapshot"
    levels = _pair_levels(item.pair, store, item.sha256)
    if levels is None or levels[1] != "portable" or (levels[0] == "portable") != twin:
        return "portable_pair_unconfirmed"
    return None


def _pair_levels(pair: str, store: StoreLayout, portable: str) -> tuple[Any, Any] | None:
    """The save levels of a trace save and the portable save put after it (each raw or packed),
    when their manifests describe one capture (``PAIR_KEYS``); else ``None``."""

    try:
        a, b = _manifest(pair, store), _manifest(portable, store)
    except (OSError, ValueError, tarfile.TarError):
        return None
    same = all(_key(a, k) is not None and _key(a, k) == _key(b, k) for k in PAIR_KEYS)
    return (a.get("save_level"), b.get("save_level")) if same else None


def _key(manifest: dict, key: str) -> Any:
    return manifest.get(key, (manifest.get("provenance") or {}).get(key))


def _remove(path: Path) -> None:
    """Rename out of sight first, so readers never see half a tree, then delete."""

    if not path.exists() and not path.is_symlink():
        return
    gone = path.with_name(f".gone-{path.name}")
    os.replace(path, gone)
    if gone.is_dir() and not gone.is_symlink():
        shutil.rmtree(gone)
    else:
        gone.unlink()


def _unlink_batches(sha: str, links: Sequence[Path], names: Sequence[str] = ()) -> int:
    count = 0
    for root in links:
        for batch in sorted(Path(root).glob("*/payload")):
            for name in names or (sha,):
                linked = batch / "sha256" / sha[:2] / name
                if linked.exists():
                    _remove(linked)
                    count += 1
    return count


def _restated(line: StoreIndexLine, **changes: Any) -> StoreIndexLine:
    return replace(line, **changes)


def _free(store: StoreLayout) -> int:
    stat = os.statvfs(store.root)
    return stat.f_bavail * stat.f_frsize


def _gzip_one(raw: Path, sha: str, store: StoreLayout, proof: Held) -> int:
    """Write and read back the gzipped form; returns its size. The raw file stays."""

    out = store_mod.gzipped_path(sha, store=store)
    tmp = out.with_name(f".gzip-{out.name}")
    store_mod.gzip_file(raw, tmp)
    if store_mod.gzipped_stats(tmp) != (sha, proof.bytes):
        tmp.unlink()
        raise ReadBackError(f"gzipped form of {sha} does not read back as its bytes")
    os.replace(tmp, out)
    store_mod.fsync(out.parent)  # the gzipped name is durable before the raw file goes
    return out.stat().st_size


def _pack_one(raw: Path, sha: str, store: StoreLayout, proof: Held) -> int:
    """Write and read back the packed form; returns its size. The raw tree stays."""

    out = store_mod.packed_path(sha, store=store)
    tmp = out.with_name(f".pack-{out.name}")
    store_mod.pack_tree(raw, tmp)
    if store_mod.packed_stats(tmp) != (sha, proof.files, proof.bytes):
        tmp.unlink()
        raise ReadBackError(f"packed form of {sha} does not read back as its tree")
    os.replace(tmp, out)
    store_mod.fsync(out.parent)  # the packed name is durable before the raw tree goes
    return out.stat().st_size


@dataclass(frozen=True)
class Run:
    store: StoreLayout
    held: dict[str, Held]
    first: dict[str, StoreIndexLine]
    links: tuple[Path, ...]
    apply: bool
    lines: Path
    form: str = "tar.gz"  # what a pack writes: ``tar.gz`` (a tree) or ``gzip`` (a file)


def _one(item: Candidate, ctx: Run, pack: bool, tally: Tally) -> None:
    why, proof = _prove(item, ctx.store, ctx.held, pack)
    tree = ctx.form == "tar.gz"
    if why is None and pack and proof.path.is_dir() != tree:  # type: ignore[union-attr]
        why = "not_a_tree" if tree else "not_a_file"
    if why is not None or proof is None:
        tally.skipped[why or "unproven"] += 1
        return
    entry = {"sha256": item.sha256, "kind": item.kind, "model": item.model,
             "snapshot": proof.held.snapshot, "bytes": proof.held.bytes,
             "held": proof.encoding or "raw", **({"pair": item.pair} if item.pair else {})}
    if ctx.apply:
        entry.update(_act(item, ctx, proof, pack))
    tally.done.append(entry)
    tally.bytes += proof.held.bytes


def run(items: Sequence[Candidate], ctx: Run, *, pack: bool, limit: int | None = None) -> dict:
    """Prove and act on each candidate in turn (only report when ``ctx.apply`` is false)."""

    tally, before, stopped = Tally(), _free(ctx.store), None
    for item in items[:limit]:
        try:
            _one(item, ctx, pack, tally)
        except ReadBackError as exc:  # nothing more is touched once a blob reads back wrong
            stopped = str(exc)
            break
    return {"apply": ctx.apply, "candidates": len(items[:limit]), "done": len(tally.done),
            "content_bytes": tally.bytes, "skipped": dict(tally.skipped),
            "free_before": before, "free_after": _free(ctx.store), "stopped": stopped,
            "items": tally.done}


def _placed(first: StoreIndexLine, proof: Proof, pack: bool,
            form: str = "tar.gz") -> dict[str, Any]:
    """The placement a pack or offload records: packed here, or held by the box in a form."""

    if pack:
        return {"encoding": form, "location": "local"}
    encoding = proof.encoding or ("raw" if first.encoding else None)
    return {"location": f"box:{proof.held.snapshot}", **({"encoding": encoding} if encoding else {})}


def _act(item: Candidate, ctx: Run, proof: Proof, pack: bool) -> dict:
    sha = item.sha256
    write = _pack_one if ctx.form == "tar.gz" else _gzip_one
    packed = write(proof.path, sha, ctx.store, proof.held) if pack else None
    line = _restated(ctx.first[sha], **_placed(ctx.first[sha], proof, pack, ctx.form))
    append_jsonl([line], ctx.lines)  # recorded before the delete: a crash leaves a true line
    store_mod.fsync(ctx.lines)
    _remove(proof.path)
    names = (sha,) if pack else (sha, sha + PACKED)  # an offloaded blob keeps no batch link
    return {"links_removed": _unlink_batches(sha, ctx.links, names), "packed_bytes": packed}


def main(policy: str, *, store: StoreLayout, data: DataLayout, catalog: Path, host: str,
         links: Sequence[Path] = (), apply: bool = False, limit: int | None = None,
         lines: Path | None = None, extra: Sequence[Path] = ()) -> dict:
    """``store offload POLICY`` and ``store pack``: select, prove, act, and summarize."""

    loaded = load(data, extra)
    items = select(policy, loaded)
    held = held_blobs(catalog, host, {s for c in items for s in (c.sha256, c.pair) if s})
    first = {line.sha256: line for line in loaded.index}
    ctx = Run(store, held, first, tuple(Path(p) for p in links), apply,
              Path(lines) if lines else data.root / "store-index" / PLACEMENT_SHARD,
              PACKS.get(policy, "tar.gz"))
    summary = run(items, ctx, pack=policy in PACKS, limit=limit)
    kinds: dict[str, int] = defaultdict(int)
    for item in summary["items"]:
        kinds[item["kind"]] += 1
    return {"policy": policy, **summary, "by_kind": dict(kinds)}


def from_args(args: Any, *, store: StoreLayout, data: DataLayout) -> dict:
    """The CLI verbs; the full item list goes to ``--report``, the printed line is the summary."""

    summary = main(getattr(args, "policy", None) or "pack", store=store, data=data,
                   catalog=args.catalog, host=args.host, links=args.links, apply=args.apply,
                   limit=args.limit, lines=args.lines, extra=args.extra)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if summary["stopped"]:
        raise SystemExit(f"stopped: {summary['stopped']}")
    return {k: v for k, v in summary.items() if k != "items"}
