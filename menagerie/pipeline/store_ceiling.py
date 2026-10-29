"""Keep the worker store under a size ceiling: offload its oldest box-verified blobs.

The last step of the daily store trim (``store ceiling``; decision 2026-10-22). The other trim
steps drop blob classes the pipeline no longer needs or pack them; none caps the store, because
every current trace save and every paper PDF stays named by a kept row. This step measures the
store on disk (allocated bytes) and, while it is over ``target`` bytes, offloads blobs oldest
first until it is not:

- Offloadable: trace saves (``application/x-tlspec``, packed or raw, runnable or portable) and
  paper PDFs whose text is extracted (an index line is ``derived_from`` them). Code snapshots,
  view files, extracted text, tiles and thumbnails stay: every attempt or the site reads them,
  and together they are small.
- The working set always stays: a blob placed in the last ``min_age`` seconds (its first index
  line's ``fetched``; for a blob a placement line once sent to the box, the later of that and the
  local copy's ctime, so a fetched-back blob stays as long as a new one), and every sha a queued
  or running request names (the ``--working-set`` globs:
  the runner and stage 3 spools' ``new`` and ``claimed`` files, and the axon render job's staged
  attempts).

Proof and action are ``store_trim``'s: a full-restore-verified catalog manifest of this host holds
the blob by content hash and the live copy hashes to its name; the placement line
``location: box:<snapshot>`` is written before the delete; the sealed batches' hard links go too.
A blob it cannot prove stays and the next candidate is tried. Fetch back: ``store.materialize``
on the hub restores an offloaded blob into the worker store on demand; elsewhere ``BlobOffloaded``
names the command (``store fetch SHA`` on the hub).
"""

from __future__ import annotations

import glob
import json
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout, read_jsonl
from menagerie.pipeline import store as store_mod
from menagerie.pipeline import store_trim
from menagerie.pipeline.records import StoreIndexLine

GIB = 2**30
# Why 20 GiB: the default worker's disk guard pauses below 62 GiB free and resumes at 72; the store was
# 30 GiB at 72 free on 22 Oct. A 20 GiB ceiling after the daily trim leaves about 18 GiB above the
# pause line for the day's growth (13 GiB on a heavy backlog day), and still holds about a week
# of the newest trace saves locally for render, replay and rerender.
TARGET_GIB = 20.0
MIN_AGE_DAYS = 1.0  # a render waits minutes to hours behind its trace; queued and running renders are the
# working set whatever their age (decision 2026-10-28: the store grows 5 to 16 GB a day, so three days kept it
# above its ceiling)
TRACE = store_trim.TLSPEC
PDF = "application/pdf"


@dataclass(frozen=True)
class Local:
    """A blob's local forms: allocated bytes and the newest ctime among them."""

    disk: int
    ctime: float


def _allocated(path: Path) -> int:
    if not path.is_dir() or path.is_symlink():
        return path.lstat().st_blocks * 512
    total = 0
    for parent, _, names in os.walk(path):
        total += sum(os.lstat(os.path.join(parent, n)).st_blocks * 512 for n in names)
    return total


def _sha_of(name: str) -> str | None:
    for suffix in (store_mod.PACKED_SUFFIX, store_mod.GZIP_SUFFIX, ""):
        if name.endswith(suffix):
            sha = name[:len(name) - len(suffix)] if suffix else name
            if store_trim.HEX.fullmatch(sha):
                return sha
    return None


def local_blobs(store: StoreLayout) -> dict[str, Local]:
    """Every visible blob under ``sha256/<xx>/`` (raw, packed or gzipped), by sha; hidden names
    are writes, restores or deletes in flight."""

    found: dict[str, Local] = {}
    root = store.root / "sha256"
    for prefix in sorted(root.iterdir()) if root.is_dir() else ():
        for path in prefix.iterdir():
            sha = None if path.name.startswith(".") else _sha_of(path.name)
            if sha is None:
                continue
            try:
                disk, ctime = _allocated(path), path.lstat().st_ctime
            except FileNotFoundError:  # removed while listing
                continue
            seen = found.get(sha, Local(0, 0.0))
            found[sha] = Local(seen.disk + disk, max(seen.ctime, ctime))
    return found


def working_set(patterns: Sequence[str]) -> set[str]:
    """Every sha any file matching ``patterns`` names (queued and running requests)."""

    shas: set[str] = set()
    for pattern in patterns:
        for name in glob.glob(os.path.expanduser(pattern)):
            try:
                shas.update(store_trim.HEX.findall(Path(name).read_text(errors="replace")))
            except (FileNotFoundError, IsADirectoryError):  # claimed or finished meanwhile
                continue
    return shas


@dataclass(frozen=True)
class Index:
    first: dict[str, StoreIndexLine]
    extracted: set[str]  # blobs some index line is derived from (a PDF whose text is stored)
    offloaded: set[str] = field(default_factory=set)  # blobs a placement line once sent to the box


def load_index(data: DataLayout) -> Index:
    first: dict[str, StoreIndexLine] = {}
    extracted: set[str] = set()
    offloaded: set[str] = set()
    for shard in sorted((data.root / "store-index").glob("*.jsonl")):
        for line in read_jsonl(StoreIndexLine, shard):
            first.setdefault(line.sha256, line)
            if line.derived_from:
                extracted.add(line.derived_from)
            if (line.location or "").startswith("box:"):
                offloaded.add(line.sha256)
    return Index(first, extracted, offloaded)


def _epoch(at: str) -> float:
    return datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp()


def _kind(sha: str, index: Index) -> str | None:
    mime = index.first[sha].mime.split(";")[0].strip()
    if mime == TRACE:
        return "trace"
    return "pdf" if mime == PDF and sha in index.extracted else None


def select(local: dict[str, Local], index: Index, keep: set[str], *, now: float,
           min_age: float) -> list[store_trim.Candidate]:
    """Offloadable blobs outside the working set, oldest placement first."""

    aged = []
    for sha, held in local.items():
        kind = _kind(sha, index) if sha in index.first and sha not in keep else None
        if kind is None:
            continue
        placed = _epoch(index.first[sha].fetched)
        # A local copy of an offloaded blob was fetched back, so its ctime is the fetch. Any other
        # blob's ctime moves when the daily batch hard-links it or the trim packs it, which would
        # restart its age every day (28 Oct: 4.6 GB of day-old saves held past the ceiling).
        if sha in index.offloaded:
            placed = max(placed, held.ctime)
        if now - placed >= min_age:
            aged.append((placed, held.ctime, sha, kind))
    return [store_trim.Candidate(sha, kind, "") for *_, sha, kind in sorted(aged)]


@dataclass(frozen=True)
class Ceiling:
    target: int
    min_age: float = MIN_AGE_DAYS * 86400
    working: tuple[str, ...] = ()


def run(items: Sequence[store_trim.Candidate], ctx: store_trim.Run, local: dict[str, Local],
        over: int, limit: int | None = None) -> dict:
    """Prove and act on candidates in order until ``over`` allocated bytes are freed."""

    tally, freed, stopped = store_trim.Tally(), 0, None
    for item in items[:limit]:
        if freed >= over:
            break
        done = len(tally.done)
        try:
            store_trim._one(item, ctx, False, tally)
        except store_trim.ReadBackError as exc:  # nothing more is touched once one reads wrong
            stopped = str(exc)
            break
        if len(tally.done) > done:
            tally.done[-1]["disk_bytes"] = local[item.sha256].disk
            freed += local[item.sha256].disk
    return {"done": len(tally.done), "freed_disk_bytes": freed, "content_bytes": tally.bytes,
            "skipped": dict(tally.skipped), "stopped": stopped, "items": tally.done}


def main(*, store: StoreLayout, data: DataLayout, catalog: Path, host: str, ceiling: Ceiling,
         links: Sequence[Path] = (), apply: bool = False, limit: int | None = None,
         lines: Path | None = None, now: float | None = None) -> dict:
    """``store ceiling``: measure, select, prove, act, and summarize."""

    free_before = store_trim._free(store)
    local = local_blobs(store)
    total = sum(blob.disk for blob in local.values())
    summary: dict[str, Any] = {"policy": "ceiling", "apply": apply, "target_bytes": ceiling.target,
                               "local_before": total, "blobs_before": len(local)}
    over = total - ceiling.target
    items: list[store_trim.Candidate] = []
    if over > 0:
        index = load_index(data)
        keep = working_set(ceiling.working)
        items = select(local, index, keep, now=time.time() if now is None else now,
                       min_age=ceiling.min_age)
        held = store_trim.held_blobs(catalog, host, (c.sha256 for c in items))
        ctx = store_trim.Run(store, held, index.first, tuple(Path(p) for p in links), apply,
                             Path(lines) if lines else data.root / "store-index" / store_trim.PLACEMENT_SHARD)
        summary.update(working_set=len(keep))
        summary.update(run(items, ctx, local, over, limit))
    else:
        summary.update(done=0, freed_disk_bytes=0, content_bytes=0, skipped={}, stopped=None, items=[])
    summary["candidates"] = len(items)
    summary["local_after"] = total - summary["freed_disk_bytes"]
    summary["short_of_target"] = max(0, summary["local_after"] - ceiling.target)
    summary["free_before"], summary["free_after"] = free_before, store_trim._free(store)
    kinds: dict[str, int] = defaultdict(int)
    for item in summary["items"]:
        kinds[item["kind"]] += 1
    summary["by_kind"] = dict(kinds)
    return summary


def from_args(args: Any, *, store: StoreLayout, data: DataLayout) -> dict:
    """The CLI verb; the full item list goes to ``--report``, the printed line is the summary."""

    target = TARGET_GIB if args.target_gib is None else args.target_gib
    days = MIN_AGE_DAYS if args.min_age_days is None else args.min_age_days
    ceiling = Ceiling(int(target * GIB), days * 86400, tuple(args.working_set))
    summary = main(store=store, data=data, catalog=args.catalog, host=args.host, ceiling=ceiling,
                   links=args.links, apply=args.apply, limit=args.limit, lines=args.lines)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if summary["stopped"]:
        raise SystemExit(f"stopped: {summary['stopped']}")
    return {k: v for k, v in summary.items() if k != "items"}
