"""The driver: events, next steps, measures, the check-in and commits.

Owned by the drive lane (rung 1; rung 2's lanes lane split admission and launch into
``admission``, re-exported here). Agent lanes run on the hub, started through ``fleet-run``;
every model run goes to the worker through the runner. Imports neither torch nor psutil.

Working files live under ``paths.work_root()`` (``~/scratch/menagerie``), never in the data
checkout: ``leases/<label>.json`` holds a live lane's keys until it ends or its hard timeout
passes, ``queue/<label>.json`` a step no lab admitted, and ``lanes/<run id>/events.jsonl`` is a
lane's event log, which ``measures`` reads. Nothing else is stored: a model's next step is
computed from the committed files each time.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, now, read_json
from menagerie.pipeline.admission import (  # noqa: F401  (re-exported: drive's launch verbs)
    LABEL,
    LANE_CAPS,
    LANE_MINUTES,
    WEB_TOOLS,
    Refused,
    _fleet_run,
    _parse,
    _sh,
    _tool,
    _write_atomic,
    lane_cap,
    lanes,
    launch,
    launch_first,
    queued,
    requeue,
)
from menagerie.pipeline.records import (
    OFF_WORKER,
    LedgerRow,
    ModelRecord,
    Packet,
    RecordError,
    RenderRow,
    RosterRow,
    from_dict,
    parse_ledger_line,
)

# A shell command that reaches the network: the 2b transcript check.
NETWORK = re.compile(r"(?:^|[\s;&|(`'\"])(curl|wget|gh|http)(?=\s|$)|\bpip3? download\b"
                     r"|\bgit clone\b")
# Source cases a reviewer from the other lab must check; an unofficial vendor recipe is added.
AUTHORED = ("port", "reimplement", "classic")
STRUCTURE_ONLY_FAILS = ("memory_cap", "oversized", "needs_gpu")
# Failed renders of one recipe revision's traces after which the row is typed ``render``: the
# driver never requeues a render, so without an end a row could sit at render forever.
RENDER_TRIES = 2
NO_TOKENS = {"tokens_in": "unknown", "tokens_cached": "unknown", "tokens_out": "unknown"}
# A Google lane's commands are not in any transcript we read, and it has network.
AGY_NETWORK = "unknown: a Google lane has network and its commands are not logged"


# --- events ----------------------------------------------------------------------------------


def event(lane: str, family: str, verb: str, **fields: Any) -> None:
    """Append one event to ``lanes/<lane>/events.jsonl`` in the work root.

    Inputs: the lane's run id (``FLEET_RUN_ID`` inside a lane), the family (empty for a lane-wide
    event such as ``launch``), the verb and its JSON-serializable fields. ``measures`` counts
    verbs starting ``fetch`` as fetches, ``try`` as tries, ``giveup`` (or ``give-up``, with a
    ``class`` field) as give-ups, and a ``tier`` field names the family's tier. Side effects:
    one locked append.
    """

    if not LABEL.fullmatch(lane) or not verb:
        raise RecordError(f"event needs a lane run id and a verb, not {lane!r}, {verb!r}")
    line = json.dumps({"at": now(), "lane": lane, "family": family, "verb": verb, **fields},
                      ensure_ascii=False, sort_keys=True) + "\n"
    path = work_dir("lanes") / lane / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line.encode("utf-8"))
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# --- next steps ------------------------------------------------------------------------------


# Memo for ``_load`` and the record and packet reads: a fallback pass in a long process (or the
# second ``next_steps`` of one tick) re-reads only the files whose stamp changed. A file written
# within RACY_S of its read is never served from the memo (its coarse mtime could repeat).
RACY_S = 2.0


def _stamp(path: Path) -> tuple[int, int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_size, st.st_mtime_ns, st.st_ino


# An appended file's first and last EDGE bytes read so far: if either differs on the next read,
# the file was rewritten rather than appended to, and it is parsed again from the start.
EDGE = 256


@dataclass(frozen=True)
class _Tail:
    """What ``_Memo.appended`` has parsed of one file: its whole lines up to ``offset``."""

    ino: int
    offset: int
    head: bytes
    edge: bytes
    rows: tuple[Any, ...]

    def extended_by(self, handle: Any) -> bool:
        """Whether the open file is this one with lines appended (same inode, start and edge)."""

        if os.fstat(handle.fileno()).st_ino != self.ino:
            return False
        handle.seek(0)
        head = handle.read(len(self.head))
        handle.seek(self.offset - len(self.edge))
        return head == self.head and handle.read(len(self.edge)) == self.edge


@dataclass
class _Memo:
    """Parsed files by path, each with the stamp it was read at and whether the stamp is trusted;
    files their writers only append to (the ledger shards) are kept by ``appended``."""

    files: dict[Path, tuple[tuple[int, int, int] | None, bool, Any]] = field(default_factory=dict)
    tails: dict[Path, _Tail] = field(default_factory=dict)

    def appended(self, path: Path, build: Any) -> list[Any]:
        """The rows of a JSONL file that is only appended to, parsing only lines added since the
        last read; a rewritten file (``_Tail.extended_by``) is parsed again in full. A last line
        without its newline is returned when it parses and read again next time, so a line still
        being written is never kept half. Returns the caller's own list."""

        try:
            handle = path.open("rb")
        except FileNotFoundError:
            self.tails.pop(path, None)
            return []
        with handle:
            hit = self.tails.get(path)
            if hit is not None and not hit.extended_by(handle):
                hit = None
            start = hit.offset if hit else 0
            handle.seek(start)
            chunk = handle.read()
            ino = os.fstat(handle.fileno()).st_ino
        whole = chunk.rfind(b"\n") + 1
        rows = (hit.rows if hit else ()) + tuple(
            build(json.loads(line)) for line in chunk[:whole].splitlines() if line.strip())
        head = hit.head if hit else chunk[:min(whole, EDGE)]
        edge = ((hit.edge if hit else b"") + chunk[:whole])[-EDGE:]
        self.tails[path] = _Tail(ino, start + whole, head, edge, rows)
        try:
            last = json.loads(chunk[whole:]) if chunk[whole:].strip() else None
        except ValueError:
            last = None  # a line still being written
        return [*rows, *([] if last is None else [build(last)])]

    def get(self, path: Path, parse: Any) -> Any:
        stamp = _stamp(path)
        hit = self.files.get(path)
        if hit is not None and hit[1] and hit[0] == stamp:
            return hit[2]
        value = None if stamp is None else parse(path)
        trusted = stamp is None or datetime.now(timezone.utc).timestamp() - stamp[1] / 1e9 > RACY_S
        self.files[path] = (stamp, trusted, value)
        return value

    def forget(self, keep: Any) -> None:
        for path in [p for p in self.files if not keep(p)]:
            del self.files[path]
        for path in [p for p in self.tails if not keep(p)]:
            del self.tails[path]


_MEMOS: dict[Path, _Memo] = {}


def _memo(data: DataLayout) -> _Memo:
    return _MEMOS.setdefault(Path(data.root).resolve(), _Memo())


@dataclass
class _Files:
    """The committed files ``next_steps`` reads, loaded once."""

    data: DataLayout
    tl: str
    roster: dict[str, RosterRow]
    rows: dict[str, list[Any]]
    packets: dict[str, Packet | None] = field(default_factory=dict)
    members: dict[str, str] | None = None
    memo: _Memo = field(default_factory=_Memo)

    def record(self, model_id: str) -> ModelRecord | None:
        return self.memo.get(self.data.model(model_id), lambda p: read_json(ModelRecord, p))

    def packet(self, family: str) -> Packet | None:
        if family not in self.packets:
            self.packets[family] = self.memo.get(self.data.packet(family), lambda p: read_json(Packet, p))
        return self.packets[family]

    def family_of(self, model_id: str) -> str | None:
        if self.members is None:
            self.members = {}
            for path in sorted((self.data.root / "research").glob("*/*.json")):
                packet = self.memo.get(path, lambda p: read_json(Packet, p))
                self.packets[packet.family] = packet
                self.members.update((m.id, packet.family) for m in packet.members)
        return self.members.get(model_id)


def _quoted(ids: Collection[str]) -> re.Pattern[bytes]:
    """Any of ``ids`` as a quoted JSON string: a superset filter, checked exactly after parsing."""

    return re.compile(b'"(?:' + b"|".join(re.escape(i.encode()) for i in sorted(ids)) + b')"')


def _jsonl_where(path: Path, build: Any, ids: Collection[str]) -> list[Any]:
    """The lines of a JSONL file that name one of ``ids``, parsed (a superset: check each)."""

    if not ids:
        return []
    pattern = _quoted(ids)
    with path.open("rb") as handle:
        return [build(json.loads(line)) for line in handle if pattern.search(line)]


def _ledger_line(raw: Mapping[str, Any]) -> Any:
    return parse_ledger_line(raw)


def shard_rows(memo: _Memo, shard: Path) -> list[Any]:
    """One ledger shard's rows through the memo: the ledger is append-only, so a grown shard
    costs only its new lines (``_Memo.appended``)."""

    return memo.appended(shard, _ledger_line)


def _roster_row(raw: Mapping[str, Any]) -> RosterRow:
    return from_dict(RosterRow, {"schema": RosterRow.SCHEMA, **raw})  # as ``roster.read``


def _shards(data: DataLayout) -> list[Path]:
    folder = data.root / "ledger"
    return sorted(folder.glob("*.jsonl")) if folder.is_dir() else []


def _read_all(data: DataLayout, memo: _Memo) -> tuple[list[Any], list[RosterRow]]:
    """Every ledger line and roster row, through the memo."""

    from menagerie.pipeline import roster

    shards = _shards(data)
    memo.forget(lambda p: p.parent.name != "ledger" or p in shards)
    lines = [r for shard in shards for r in shard_rows(memo, shard)]
    return lines, memo.get(data.roster, lambda _: roster.read(data=data)) or []


def _read_some(data: DataLayout, ids: Collection[str]) -> tuple[list[Any], list[RosterRow]]:
    """The ledger lines and roster rows of ``ids`` only, by a filtered parse."""

    lines = [r for shard in _shards(data) for r in _jsonl_where(shard, parse_ledger_line, ids)]
    found = _jsonl_where(data.roster, _roster_row, ids) if data.roster.exists() else []
    return [r for r in lines if r.id in ids], [r for r in found if r.stable_id in ids]


def _load(data: DataLayout, ids: Collection[str] | None = None) -> _Files:
    """The roster and ledger rows ``next_steps`` reads, memoised per file (``_Memo``).

    ``ids`` restricts the rows to those models (exact; for a few ids, such as the counts reader's
    correction of rows touched since the counts file's commit): a filtered parse, no memo.
    """

    from menagerie.pipeline.publish import torchlens_pin

    memo = _memo(data)
    lines, found = _read_all(data, memo) if ids is None else _read_some(data, ids)
    rows: dict[str, list[Any]] = defaultdict(list)
    for row in lines:
        rows[row.id].append(row)
    return _Files(data, torchlens_pin(), {r.stable_id: r for r in found if r.stable_id}, rows, memo=memo)


def _latest(rows: Sequence[Any], keep: Any) -> Any:
    kept = [r for r in rows if keep(r)]
    return max(reversed(kept), key=lambda r: r.at) if kept else None


def _head(cls: str | None) -> str:
    return (cls or "").split(":", 1)[0]


def _retried(give_up: Any, packet: Packet, rows: Sequence[Any]) -> bool:
    """Whether an amendment already answered an earlier research-gap give-up of this model."""

    earlier = [r for r in rows if isinstance(r, LedgerRow) and r.outcome == "giveup"
               and _head(r.cls) == "research_gap" and r.at < give_up.at]
    return any(r.at < a.at < give_up.at and r.cls in a.why
               for a in packet.amendments for r in earlier)


def _gap_step(give_up: Any, packet: Packet, rows: Sequence[Any]) -> tuple[str, str]:
    """A research-gap give-up: answered (the 2b retry), unanswered (2a), or final.

    One 2b retry after an amendment (fix 6): a research gap given up after an amendment answered
    this model's earlier one is final. An answering amendment names the give-up's class in its
    ``why``; others do not count.
    """

    if _retried(give_up, packet, rows):
        return "done", f"gave up: {give_up.cls} after the amendment's one 2b retry"
    if any(a.at > give_up.at and give_up.cls in a.why for a in packet.amendments):
        return "2b", f"packet amended for the {give_up.cls} give-up: the 2b retry"
    return "2a", f"give-up {give_up.cls}: the packet needs an amendment"


def unresearched(packet: Packet) -> bool:
    """Whether a packet is a harvester stub no 2a lane has written: not one saved source.

    The library-tier harvester (``harvest._packet``) writes a packet for every June catalog row;
    one whose June recipe named no library carries no source (``harvester@menagerie``, every
    slot ``not_found``, search note "2a searches the rest"). A 2b lane on it has nothing to read
    and mostly gives up ``research_gap``, so the row goes to 2a first. A 2a write stamps its own
    ``by`` (``packet._author``), so after 2a the packet counts as researched even with no source.
    """

    return not packet.sources and packet.by.startswith("harvester@")


def _research_step(record: ModelRecord | None, packet: Packet | None,
                   rows: Sequence[Any] = ()) -> tuple[str, str] | None:
    """2a, 2b or a terminal call before any recipe exists; ``None`` once a recipe exists."""

    if packet is None:
        return "2a", "no packet"
    if not packet.identity.architecture:
        return "done", f"packet {packet.family}: not an architecture"
    give_up = record.give_up if record else None
    if give_up is None:
        if record and record.recipe:
            return None
        if unresearched(packet):
            return "2a", f"packet {packet.family} has no sources: 2a research first"
        return "2b", "packet with no recipe or give-up"
    if _head(give_up.cls) == "research_gap":
        return _gap_step(give_up, packet, rows)
    if _head(give_up.cls) == "infra":
        return "2b", f"give-up {give_up.cls} does not count"
    owed = _owed_step(record, packet, rows)
    if owed is not None:
        return owed
    return "done", f"gave up: {give_up.cls}"


# A scripted route's verdict never settles a row (decisions.md 2026-10-08 9:56 PM PT, "speed
# never costs coverage"): the row is owed one agent attempt, whose own give-up settles it.
# Heads that are not about implementing the model, so no agent attempt is owed for them.
NOT_IMPLEMENTATION = frozenset({"not_architecture", "duplicate_of", "describe", "no_paper",
                                "render"})
# Heads that mean the source is unknown: the owed attempt is 2a research, not a 2b recipe.
SOURCE_UNKNOWN = frozenset({"not_found", "ambiguous", "insufficient_description",
                            "inaccessible"})
# Heads that judge an existing recipe a substitute: a recipe is there, but no faithful one.
SUBSTITUTE = frozenset({"standin", "june_code"})
# How the stand-in sweep (`standin.sweep`, a script) begins its give-up reasons.
STANDIN_SWEEP = "stand-in: "


def agent_tried(record: ModelRecord, rows: Sequence[Any] = ()) -> bool:
    """Whether a 2a or 2b agent typed this record's give-up, so it may settle the row.

    The give-up's ``by`` says who typed it (``implement.giveup``, the lanes' verb, writes
    ``agent``; the scripted routes write ``script``). A give-up from before ``by`` existed counts
    as an agent's only when the ledger holds the give-up row ``implement.giveup`` always appended
    with it (same class, same time). Of the scripts, only the stand-in sweep (its reasons start
    ``stand-in: ``) and ``type_render`` appended such a row; the harvest routes never did.
    """

    give = record.give_up
    if give is None:
        return False
    if give.by is not None:
        return give.by == "agent"
    if give.reason.startswith(STANDIN_SWEEP) or give.cls == "render":
        return False
    return any(isinstance(r, LedgerRow) and r.outcome == "giveup" and r.cls == give.cls
               and r.at == give.at for r in rows)


def _owed_step(record: ModelRecord, packet: Packet,
               rows: Sequence[Any]) -> tuple[str, str] | None:
    """2a or 2b for a give-up no agent typed, or ``None`` when the give-up may settle the row.

    Owed: a give-up on a row with no recipe, or one judging its recipe a substitute
    (``SUBSTITUTE``). Not owed: an agent's give-up, a head about something other than the
    implementation (``NOT_IMPLEMENTATION``), a give-up about a recipe that is there (a run
    failure, a kept draft) and ``needs_env:<name>`` while that environment is not committed
    (the row reopens when it is). The owed attempt is 2a while the packet is ``unresearched``.
    """

    from menagerie.pipeline.records import known_envs

    give = record.give_up
    head = _head(give.cls)
    if agent_tried(record, rows) or head in NOT_IMPLEMENTATION:
        return None
    if record.recipe is not None and head not in SUBSTITUTE:
        return None
    if head == "needs_env" and give.cls.partition(":")[2] not in known_envs():
        return None
    why = f"scripted route typed {give.cls}: owed one agent attempt"
    if unresearched(packet):
        return "2a", f"{why}; packet {packet.family} has no sources: 2a research first"
    answered = any(a.at > give.at for a in packet.amendments)  # 2a amended the packet since
    return ("2a" if head in SOURCE_UNKNOWN and not answered else "2b"), why


def _current(rows: Sequence[Any], stage: str, rev: str, **match: Any) -> LedgerRow | None:
    """The latest non-``infra``, eval-mode row of a stage at the recipe revision, with
    ``match``'s fields (``tl``, ``capture``) equal."""

    def keep(row: Any) -> bool:
        if not isinstance(row, LedgerRow) or row.outcome == "infra" or row.mode == "train":
            return False
        return (row.stage, row.recipe_rev) == (stage, rev) and all(
            getattr(row, k) == v for k, v in match.items())

    return _latest(rows, keep)


def _run_step(model_id: str, record: ModelRecord, files: _Files) -> tuple[str, str]:
    """Forward, trace, render, review or done for a model with a recipe."""

    from menagerie.pipeline import ledger

    rev, rows = record.recipe.rev, files.rows.get(model_id, [])
    forward = _current(rows, "forward", rev)
    full = ledger.deciding_trace(model_id, rows, recipe_rev=rev, tl=files.tl)
    structure = _current(rows, "trace", rev, tl=files.tl, capture="structure_only")
    # A passing full capture decides; else a passing structure-only one stands in.
    trace = next((t for t in (full, structure) if t and t.outcome == "pass"), full or structure)
    # Off-worker placements skip the worker's forward; a GPU row's attempt runs its own forward;
    # a trace at the revision ran its own forward (rows traced before forward rows were written),
    # and ``stage3`` would skip the row as traced, so asking for a forward would hold it forever.
    if forward is None and trace is None and record.recipe.placement not in (*OFF_WORKER,
                                                                             "gpu-window"):
        return "forward", f"no forward row at recipe {rev}"
    if forward and forward.outcome != "pass" and _head(forward.cls) not in STRUCTURE_ONLY_FAILS:
        return "done", f"forward failed: {forward.cls}"
    if trace is None:
        return "trace", f"no trace at recipe {rev} and TorchLens {files.tl}"
    if trace.outcome != "pass":
        return "done", f"trace failed: {trace.cls}"
    saved = trace.trace or trace.summary  # a structure-only capture may save only its summary
    rendered = _latest(rows, lambda r: isinstance(r, RenderRow) and r.trace == saved
                       and r.outcome in ("pass", "partial"))
    if rendered is None:
        why = render_failed(rows, rev, tl=files.tl)
        if why is not None:
            return "done", f"render failed: {why}"
        return "render", f"no render of trace {saved}"
    return _review_step(model_id, record, files, rows)


def render_failed(rows: Sequence[Any], rev: str | None, *, tl: str) -> str | None:
    """Why a row traced at ``rev`` is typed ``render``, or ``None`` while a drawing may come: no
    passing trace saved anything to draw from, or ``RENDER_TRIES`` renders of its passing traces
    at this revision failed (a passing render of any of them settles it instead)."""

    traced = [r for r in rows if isinstance(r, LedgerRow) and r.stage == "trace"
              and r.recipe_rev == rev and r.tl == tl and r.outcome == "pass"]
    if not traced:
        return None
    keys = {r.trace or r.summary for r in traced}
    if keys == {None}:
        return f"no passing trace at recipe {rev} saved a trace or summary to draw from"
    drawings = [r for r in rows if isinstance(r, RenderRow) and r.trace in keys]
    if any(r.outcome in ("pass", "partial") for r in drawings):
        return None
    failed = sorted((r for r in drawings if r.outcome == "fail"), key=lambda r: r.at)
    if len(failed) < RENDER_TRIES:
        return None
    skips = sorted({f"{s.why} ({s.variant or s.kind})" for s in failed[-1].skipped})
    return (f"{len(failed)} renders failed at recipe {rev} (last {failed[-1].at}: "
            f"{', '.join(skips) or 'no skip recorded'})")


def type_render(model_id: str, why: str, *, data: DataLayout) -> ModelRecord:
    """Type a traced row ``render`` with the evidence: the record's give-up and a give-up ledger
    row at its recipe's revision (a render the runner ended ``infra`` leaves no render row, so
    ``render_failed`` cannot see it)."""

    from dataclasses import replace

    from menagerie.pipeline import ledger, runner, write_json
    from menagerie.pipeline.records import GiveUp

    path = data.model(model_id)
    record = read_json(ModelRecord, path)
    if record.recipe is None:
        raise RecordError(f"{model_id} has no recipe to type")
    give = GiveUp(cls="render", reason=why, at=now(), by="script")
    write_json(replace(record, give_up=give), path)
    ledger.append([LedgerRow(id=model_id, stage="forward", at=give.at, host=runner.host_label(),
                             env=record.recipe.env, recipe_rev=record.recipe.rev,
                             outcome="giveup", cls="render", msg=why[:300])], data=data)
    return replace(record, give_up=give)


def _review_step(model_id: str, record: ModelRecord, files: _Files,
                 rows: Sequence[Any]) -> tuple[str, str]:
    packet = files.packet(record.family)
    if not _authored(record, packet):
        return "done", f"{record.source_case}: traced and rendered"
    review = _current(rows, "review", record.recipe.rev)
    unjudged = review is None or review.verdict == "cannot_judge"
    if unjudged and packet is not None and _sourceless(packet):
        return "2a", f"packet {packet.family} has no saved source to review against: 2a first"
    if review is not None and review.verdict == "cannot_judge" and _sources_since(packet, review.at):
        review = None  # the packet gained sources after a review that had none to judge by
    if review is None:
        return "review", f"{record.source_case} recipe with no review at {record.recipe.rev}"
    if review.verdict == "fabrication" and not _confirmed(rows, record.recipe.rev):
        return "review", "a provisional seat's fabrication: a second seat's review before the repair"
    if review.verdict == "fabrication":
        return "2b", "review found fabrication: the repair round"
    return "done", f"reviewed: {review.verdict}"


REVIEW_SOURCES = ("paper", "code", "page")


def _sourceless(packet: Packet) -> bool:
    """A harvester stub no 2a lane has written whose packet saves no paper, code or page: its
    review case would hold no source, so a reviewer could only answer ``cannot_judge``. After a
    2a write (its own ``by``) a review goes ahead, and ``cannot_judge`` then settles the row."""

    return packet.by.startswith("harvester@") and not any(
        s.role in REVIEW_SOURCES and (s.text or s.blob) for s in packet.sources)


def _sources_since(packet: Packet | None, at: str) -> bool:
    """Whether the packet's sources were amended after ``at``."""

    return packet is not None and any(
        a.at > at and any(c.startswith("sources") for c in a.changed) for a in packet.amendments)


def _confirmed(rows: Sequence[Any], rev: str) -> bool:
    """A fabrication counts once a fully calibrated seat returned it, or two different seats did:
    a provisional seat's second look at its own call is not independent (decisions.md
    2026-10-28)."""

    from menagerie.pipeline.review import needs_second_review  # review imports drive

    found = [r for r in rows if isinstance(r, LedgerRow) and r.stage == "review"
             and r.recipe_rev == rev and r.outcome != "infra" and r.verdict == "fabrication"]
    return len({r.reviewer for r in found}) > 1 or any(not needs_second_review(r) for r in found)


def _authored(record: ModelRecord, packet: Packet | None) -> bool:
    """Ports, reimplementations, classics and vendor code not known to be official."""

    if record.source_case in AUTHORED:
        return True
    if record.source_case != "vendor":
        return False
    code = [s for s in (packet.sources if packet else ()) if s.role == "code"]
    used = [s for s in code if s.sid in record.recipe.sources] or code
    return not used or any(s.official is not True for s in used)


def _step(model_id: str, files: _Files) -> tuple[str, str]:
    row = files.roster.get(model_id)
    if row is not None and row.state != "active":
        return "done", f"roster: {row.state}"
    record = files.record(model_id)
    family = record.family if record else files.family_of(model_id)
    packet = files.packet(family) if family else None
    rows = files.rows.get(model_id, [])
    early = _research_step(record, packet, rows)
    if early is not None and not kept_draft_traced(record, rows, tl=files.tl):
        if early[0] == "2b" and _dropped(model_id, packet):
            return "2a", f"not a member of packet {family}: 2a gives it a family first"
        marker = work_dir("2b") / str(family) / "deferred.json"
        if early[0] == "2b" and marker.is_file():
            return "2b", "2b:deferred; draft kept for a new session"
        return early
    return _run_step(model_id, record, files)


def _dropped(model_id: str, packet: Packet | None) -> bool:
    """Whether the row's own family's packet no longer lists it. That family's 2b lane works only
    on the packet's members, so it would be sent round forever: 2a gives the row a family first."""

    return packet is not None and model_id not in {m.id for m in packet.members}


def kept_draft_traced(record: ModelRecord | None, rows: Sequence[Any], *, tl: str) -> bool:
    """Whether a give-up that keeps its draft (``implement.KEEPS_DRAFT``: ``needs_gpu``,
    ``oversized``) has a draft that passed a full trace at its revision where its placement runs
    it (a GPU, the cluster). Such a row goes on to render and review like any traced row and is
    not counted typed."""

    from menagerie.pipeline import ledger
    from menagerie.pipeline.implement import KEEPS_DRAFT

    if record is None or record.give_up is None or record.recipe is None:
        return False
    head = _head(record.give_up.cls)
    if head not in KEEPS_DRAFT or KEEPS_DRAFT[head] != record.recipe.placement:
        return False  # a stand-in's passing trace is not the model's (its placement is None too)
    full = ledger.deciding_trace(record.id, rows, recipe_rev=record.recipe.rev, tl=tl)
    return full is not None and full.outcome == "pass"


def next_steps(ids: Sequence[str] = (), *, data: DataLayout) -> list[dict[str, Any]]:
    """Derive each model's next step from its committed files and the ledger.

    Returns: one ``{"id", "step", "why"}`` per model (default: every rostered model with a
    stable id), ``step`` one of ``2a`` (no packet; a harvester packet with no source,
    ``unresearched``; or a ``research_gap`` give-up the packet has not answered: an answer is an amendment dated after the give-up whose ``why`` names its class;
    a research gap given up after such an answer is final, the one retry spent; or a row owed
    2b whose family's packet does not list it),
    ``2b`` (a packet with neither recipe nor give-up), ``forward`` (no forward row at the
    recipe's revision), ``trace`` (no trace at the revision and the TorchLens pin), ``render``
    (no render of that trace), ``review`` (an authored recipe with no review at its revision) or
    ``done``. The renderer revision is not checked (the hub has no renderer): ``publish`` re-arms
    stale renders. Side effects: none.
    """

    files = _load(data)
    chosen = list(ids) or sorted(files.roster, key=lambda i: int(i[1:]))
    return [dict(zip(("id", "step", "why"), (i, *_step(i, files)), strict=True)) for i in chosen]


def families_of(ids: Sequence[str], *, data: DataLayout) -> dict[str, str | None]:
    """Each id's family: its record's, else the packet that lists it, else ``None``."""

    files = _Files(data, "", {}, {}, memo=_memo(data))
    out = {}
    for model_id in ids:
        record = files.record(model_id)
        out[model_id] = record.family if record else files.family_of(model_id)
    return out


# --- measures --------------------------------------------------------------------------------


@dataclass
class _Lane:
    """One lane's run record, launch event, events and transcript readings."""

    run: dict[str, Any]
    launch: dict[str, Any]
    events: list[dict[str, Any]]
    usage: list[tuple[datetime, int, int, int]] | None  # (at, in, cached, out); None: no transcript
    commands: list[str]

    @property
    def kind(self) -> str:
        return self.launch.get("kind") or self.run.get("kind") or "unknown"

    @property
    def harness(self) -> str:
        return self.launch.get("harness") or self.run.get("kind") or "unknown"

    @property
    def network(self) -> list[str]:
        if self.harness == "antigravity" and self.kind == "2b":
            return [AGY_NETWORK]
        return sorted({c for c in self.commands if NETWORK.search(c)})

    def holds(self, family: str) -> bool:
        return any(e["family"] == family for e in self.events)


def _runs_under(run_id: str) -> list[dict[str, Any]]:
    """The agent lanes among the run and its descendants, from fleet-run's own records.

    Worker jobs a lane started (``remote`` runs) are walked through but are not lanes.
    """

    done = _sh([_fleet_run(), "list", "--json", "--all"])
    every = json.loads(done.stdout) if done.returncode == 0 else []
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in every:
        by_parent[run.get("parent_run_id") or ""].append(run)
    tree = [r for r in every if r["run_id"] == run_id]
    for run in tree:  # grows while it is walked
        tree += by_parent.get(run["run_id"], [])
    return [run for run in tree if run.get("kind") in ("claude", "codex") or _google(run)]


def _google(run: Mapping[str, Any]) -> bool:
    """A Google lane: a command run whose argv is ``agy_lane.py run``."""

    argv = [str(a) for a in _spec(run).get("argv", [])]
    return run.get("kind") == "command" and any(a.endswith("agy_lane.py") for a in argv)


def _read_lines(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        with contextlib.suppress(ValueError):
            found = json.loads(line)
            if isinstance(found, dict):
                yield found


def _claude_transcripts(log: Path, spec: Mapping[str, Any]) -> list[Path]:
    sessions = dict.fromkeys(d["session_id"] for d in _read_lines(log) if d.get("session_id"))
    roots = {Path.home() / ".claude" / "projects"}
    if spec.get("environment", {}).get("CLAUDE_CONFIG_DIR"):
        roots.add(Path(spec["environment"]["CLAUDE_CONFIG_DIR"]) / "projects")
    found = {p.resolve() for root in roots for sid in sessions
             for pattern in (f"*/{sid}.jsonl", f"*/{sid}/subagents/*.jsonl")
             for p in root.glob(pattern)}
    return sorted(found)


def _bash_commands(message: Mapping[str, Any]) -> list[str]:
    blocks = [b for b in message.get("content") or () if isinstance(b, dict)]
    calls = [b.get("input") for b in blocks
             if b.get("type") == "tool_use" and b.get("name") == "Bash"]
    return [call.get("command", "") for call in calls if isinstance(call, dict)]


def _claude_usage(paths: Sequence[Path]) -> tuple[list[tuple[datetime, int, int, int]], list[str]]:
    """Usage per API message (streamed blocks repeat it under one id) and Bash commands."""

    usage, commands, seen = [], [], set()
    entries = (e for path in paths for e in _read_lines(path) if e.get("type") == "assistant")
    for entry in entries:
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        commands += _bash_commands(message)
        got = message.get("usage")
        if got and message.get("id") not in seen:
            seen.add(message.get("id"))
            fresh = got.get("input_tokens", 0) + got.get("cache_creation_input_tokens", 0)
            usage.append((_parse(entry["timestamp"]), fresh,
                          got.get("cache_read_input_tokens", 0), got.get("output_tokens", 0)))
    return usage, commands


def _codex_transcripts(log: Path, spec: Mapping[str, Any]) -> list[Path]:
    threads = dict.fromkeys(d["thread_id"] for d in _read_lines(log)
                            if d.get("type") == "thread.started" and d.get("thread_id"))
    home = Path(spec.get("environment", {}).get("CODEX_HOME") or Path.home() / ".codex")
    return sorted(p for tid in threads for p in home.glob(f"sessions/*/*/*/rollout-*-{tid}.jsonl"))


def _codex_usage(paths: Sequence[Path]) -> tuple[list[tuple[datetime, int, int, int]], list[str]]:
    usage, commands, seen = [], [], set()
    for path in paths:
        for entry in _read_lines(path):
            payload = entry.get("payload") or {}
            if entry.get("type") == "token_usage_record" and payload.get("response_id") not in seen:
                seen.add(payload.get("response_id"))
                got = payload.get("usage") or {}
                cached = got.get("cached_input_tokens", 0)
                usage.append((_parse(entry["timestamp"]), got.get("input_tokens", 0) - cached,
                              cached, got.get("output_tokens", 0)))
            elif payload.get("type") in ("custom_tool_call", "function_call", "local_shell_call"):
                text = payload.get("input") or payload.get("arguments") or payload.get("action")
                commands.append(text if isinstance(text, str) else json.dumps(text))
    return usage, commands


def _agy_transcripts(log: Path, spec: Mapping[str, Any]) -> list[Path]:
    """A Google lane's vendor envelope: ``LOG.json`` beside the log ``agy_lane.py run`` names."""

    argv = [str(a) for a in spec.get("argv", [])]
    at = next((i for i, a in enumerate(argv) if a.endswith("agy_lane.py")), None)
    envelope = Path(argv[at + 2] + ".json") if at is not None and len(argv) > at + 2 else None
    return [envelope] if envelope and envelope.is_file() else []


def _find_usage(node: Any) -> Mapping[str, Any] | None:
    if isinstance(node, dict):
        if "input_tokens" in node and "output_tokens" in node:
            return node
        node = list(node.values())
    if isinstance(node, list):
        return next((u for u in map(_find_usage, node) if u is not None), None)
    return None


def _agy_usage(paths: Sequence[Path]) -> tuple[list[tuple[datetime, int, int, int]], list[str]]:
    """The envelope's token counts (input, cache reads, output plus thinking); no commands."""

    usage = []
    for path in paths:
        got = _find_usage(json.loads(path.read_text(encoding="utf-8")))
        if got is not None:
            when = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            usage.append((when, int(got.get("input_tokens", 0)),
                          int(got.get("cache_read_tokens", 0)),
                          int(got.get("output_tokens", 0)) + int(got.get("thinking_tokens", 0))))
    return usage, []


def _spec(run: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(run.get("spec_path") or "/nonexistent")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _lane(run: dict[str, Any]) -> _Lane:
    events_path = work_dir("lanes") / run["run_id"] / "events.jsonl"
    events = list(_read_lines(events_path)) if events_path.exists() else []
    started = next((e for e in events if e.get("verb") == "launch"), {})
    spec, log = _spec(run), Path(run.get("log_path") or "/nonexistent")
    harness = started.get("harness") or spec.get("adapter") or run.get("kind")
    locate, read = {"codex": (_codex_transcripts, _codex_usage),
                    "antigravity": (_agy_transcripts, _agy_usage)}.get(
        harness, (_claude_transcripts, _claude_usage))
    paths = locate(log, spec) if log.exists() else []
    usage, commands = read(paths) if paths else (None, [])
    return _Lane(run, started, [e for e in events if e.get("family")], usage, commands)


def _model_of(lane: _Lane) -> str:
    if lane.launch.get("model"):
        return lane.launch["model"]
    argv = _spec(lane.run).get("argv", [])
    return next((argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--model"), "unknown")


def _tokens(usage: Sequence[tuple[datetime, int, int, int]] | None) -> dict[str, Any]:
    if usage is None:
        return dict(NO_TOKENS)
    return {name: sum(u[i] for u in usage)
            for i, name in enumerate(("tokens_in", "tokens_cached", "tokens_out"), start=1)}


def _lane_row(lane: _Lane) -> dict[str, Any]:
    run = lane.run
    started, ended = run.get("started_at"), run.get("ended_at")
    if ended is None:  # still running
        ended = datetime.now(timezone.utc).timestamp()
    wall = "unknown" if started is None else round((ended - started) / 60, 1)
    return {
        "run": run["run_id"], "label": run.get("label"), "kind": lane.kind,
        "role": lane.launch.get("role", "unknown"),
        "harness": lane.launch.get("harness") or run.get("kind"), "model": _model_of(lane),
        "account": lane.launch.get("account") or "unknown", "state": run.get("state"),
        "outcome": run.get("outcome"), "wall_min": wall, **_tokens(lane.usage),
        "network": lane.network, "families": sorted({e["family"] for e in lane.events}),
        "gauges": [f"{k} {v}" for k, v in (lane.launch.get("gauges") or {}).items()],
    }


def _span_min(events: Sequence[dict[str, Any]]) -> float:
    stamps = [_parse(e["at"]) for e in events]
    return round((max(stamps) - min(stamps)).total_seconds() / 60, 1)


def _family_usage(lane: _Lane, family: str) -> list[tuple[datetime, int, int, int]] | None:
    """The lane's usage that is this family's: all of it for a one-family lane, else the entries
    inside this family's event window and no other family's."""

    if lane.usage is None:
        return None
    windows: dict[str, tuple[datetime, datetime]] = {}
    for e in (e for e in lane.events if e["verb"] != "batch:assigned"):
        at, (lo, hi) = _parse(e["at"]), windows.get(e["family"], (_parse(e["at"]),) * 2)
        windows[e["family"]] = (min(lo, at), max(hi, at))
    if len(windows) == 1:
        return lane.usage

    def inside(at: datetime) -> list[str]:
        return [f for f, (lo, hi) in windows.items() if lo <= at <= hi]

    return [u for u in lane.usage if inside(u[0]) == [family]]


def _verb(event: Mapping[str, Any]) -> str:
    """``fetch code``, ``give-up`` and ``give_up`` read as ``fetchcode`` and ``giveup``."""

    return re.sub(r"[^a-z]", "", str(event["verb"]).lower())


def _give_up(records: Sequence[ModelRecord], events: Sequence[Mapping[str, Any]]) -> str:
    """The class a member's record gave up with, else the last give-up event's."""

    recorded = [r.give_up.cls for r in records if r.give_up]
    logged = [e.get("class") or e.get("cls") for e in events if _verb(e).startswith("giveup")]
    return next(iter(recorded), None) or next(iter(reversed(logged)), None) or ""


def _tier(records: Sequence[ModelRecord], events: Sequence[Mapping[str, Any]]) -> str:
    """A ``tier`` field on the family's events, else its members' source case."""

    named = [e["tier"] for e in events if e.get("tier")]
    return next(iter(named), None) or next((r.source_case for r in records), "unknown")


def _family_span(lane: _Lane, family: str) -> float | None:
    """A family's measured event window; unstarted batch families have unknown minutes."""

    events = [e for e in lane.events if e["family"] == family]
    if len(lane.launch.get("keys", ())) > 1:
        events = [e for e in events if e["verb"] not in ("tier", "launch", "batch:assigned")]
    return _span_min(events) if events else None


def _minutes(mine: Sequence[_Lane], family: str) -> dict[str, float | None]:
    """Per lane kind, the summed first-to-last spans of the family's events in such lanes."""

    out: dict[str, float | None] = {}
    for kind in ("2a", "2b", "review"):
        spans = [_family_span(lane, family) for lane in mine if lane.kind == kind]
        out[f"min_{kind}"] = round(sum(spans), 1) if spans and None not in spans else None
    return out


def _family_tokens(mine: Sequence[_Lane], family: str) -> dict[str, Any]:
    usage = [_family_usage(lane, family) for lane in mine]
    if None in usage:
        return dict(NO_TOKENS)
    return _tokens([x for u in usage for x in u])


def _family_row(family: str, lanes_in: Sequence[_Lane], files: _Files) -> dict[str, Any]:
    mine = [lane for lane in lanes_in if lane.holds(family)]
    events = [e for lane in mine for e in lane.events if e["family"] == family]
    verbs = Counter(_verb(e) for e in events)
    packet = files.packet(family)
    records = [r for m in (packet.members if packet else ()) if (r := files.record(m.id))]
    return {
        "family": family, "tier": _tier(records, events), **_minutes(mine, family),
        "fetches": sum(n for v, n in verbs.items() if v.startswith("fetch")),
        "tries": verbs.get("try", 0), "giveup": _give_up(records, events),
        "deferred": sum(e["verb"] == "2b:deferred" for e in events),
        **_family_tokens(mine, family),
        "network": any(lane.kind == "2b" and lane.network for lane in mine),
    }


FAMILY_COLUMNS = ("family", "tier", "min_2a", "min_2b", "min_review", "fetches", "tries",
                  "giveup", "deferred", "tokens_in", "tokens_cached", "tokens_out", "network")
LANE_COLUMNS = ("run", "label", "kind", "role", "harness", "model", "account", "state",
                "outcome", "wall_min", "tokens_in", "tokens_cached", "tokens_out", "network",
                "gauges")


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value) or "-"
    return str(value).replace("\t", " ").replace("\n", " ").replace("|", "/")


def _tsv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    return "".join("\t".join(_cell(r.get(c)) for c in columns) + "\n"
                   for r in [dict(zip(columns, columns, strict=True)), *rows])


def _stat(values: Sequence[Any]) -> str:
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numbers:
        return "-"
    unknown = len(values) - len(numbers)
    text = f"{statistics.median(numbers):g} / {sum(numbers):g}"
    return f"{text} ({unknown} unknown)" if unknown else text


def _markdown(run_id: str, lane_rows: Sequence[Mapping[str, Any]],
              family_rows: Sequence[Mapping[str, Any]]) -> str:
    out = [f"# Measures for run {run_id}", "", "## Per tier (median / total)", "",
           "| Tier | Families | 2a min | 2b min | Review min | Fetches | Tries | Tokens in "
           "| Tokens cached | Tokens out | Give-ups | Research gaps |", "|---" * 12 + "|"]
    by_tier: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in family_rows:
        by_tier[row["tier"]].append(row)
    for tier, rows in sorted(by_tier.items()):
        stats = [_stat([r[c] for r in rows if r[c] is not None])
                 for c in ("min_2a", "min_2b", "min_review", "fetches", "tries", "tokens_in",
                           "tokens_cached", "tokens_out")]
        gave = [r["giveup"] for r in rows if r["giveup"]]
        gaps = sum(_head(g) == "research_gap" for g in gave)
        out.append(f"| {tier} | {len(rows)} | " + " | ".join(stats) + f" | {len(gave)} | {gaps} |")
    out += ["", "## Lanes", "", "| " + " | ".join(LANE_COLUMNS) + " |",
            "|---" * len(LANE_COLUMNS) + "|"]
    out += ["| " + " | ".join(_cell(r.get(c)) for c in LANE_COLUMNS) + " |" for r in lane_rows]
    return "\n".join(out) + "\n"


def measures(run_id: str, *, data: DataLayout) -> dict[str, Any]:
    """Turn a run and its child lanes into minutes, tokens, fetches and tries per family.

    ``run_id`` is a lane's run id or the parent run its lanes were launched under. Lanes come from
    fleet-run's records; families from the lanes' event logs; tokens from each lane's transcript
    (Claude JSONL usage, Codex session logs), ``"unknown"`` when a transcript is missing, never
    estimated. A family's tokens in a several-family lane are the usage inside its event window.
    Returns: ``{"run", "lanes", "families", "files"}``; ``network`` lists any network command in
    a lane's transcript (flagged per family for 2b lanes). Side effects: ``measures.tsv``,
    ``lanes.tsv`` and ``measures.md`` under ``lanes/<run id>/`` in the work root, when the run
    has any lane.
    """

    runs = _runs_under(run_id)
    if not runs:
        return {"run": run_id, "lanes": [], "families": [], "files": [],
                "why": f"fleet-run has no run {run_id}"}
    found = [_lane(run) for run in runs]
    files = _load(data)
    lane_rows = [_lane_row(lane) for lane in found]
    family_rows = [_family_row(f, found, files)
                   for f in sorted({e["family"] for lane in found for e in lane.events})]
    out = work_dir("lanes") / run_id
    out.mkdir(parents=True, exist_ok=True)
    written = {"measures.tsv": _tsv(family_rows, FAMILY_COLUMNS),
               "lanes.tsv": _tsv(lane_rows, LANE_COLUMNS),
               "measures.md": _markdown(run_id, lane_rows, family_rows)}
    for name, text in written.items():
        _write_atomic(out / name, text)
    return {"run": run_id, "lanes": lane_rows, "families": family_rows,
            "files": [str(out / name) for name in written]}


# --- collect ---------------------------------------------------------------------------------


def _changed(porcelain: str) -> list[str]:
    """Paths from ``git status --porcelain=v1 -z``; a rename's source entry is skipped."""

    entries, paths, skip = porcelain.split("\0"), [], False
    for entry in entries:
        if skip or len(entry) < 4:
            skip = False
            continue
        skip = entry[0] in "RC"
        if not re.fullmatch(r"\..+\.tmp", Path(entry[3:]).name):
            paths.append(entry[3:])
    return paths


def checkin(run_id: str, *, data: DataLayout, out: Path) -> Path:
    """Write a rung's check-in (``rungs/r2-checkin.md``) from the run's lanes and the ledger.

    Inputs: the rung's run id, the data root and the output file. The check-in holds the funnel
    by case over the frozen rung rows, review verdicts, give-up and failure classes, ten sample
    packets, review findings, renders, dated catalog counts by group and the lanes' minutes and
    tokens, with no fleet paths (``checkin.write``). Returns: the file written. Side effects: that
    file and ``measures``'s files.
    """

    from menagerie.pipeline import checkin as written

    return written.write(run_id, data=data, out=out)


def collect(*, data: DataLayout, commit: bool = False) -> list[str]:
    """List the changed files under the data root and, with ``commit``, commit them by name.

    Dry run by default (the ``collect`` verb): the paths are listed, nothing is committed. With
    ``commit=True`` they go in one ``data:`` commit through ``git-commit-paths``, which leaves
    every other staged or changed file alone. Temporary ``.<name>.tmp`` files are never listed.
    Returns: the repository-relative paths. Side effects: the commit, when asked.
    """

    root = Path(data.root).resolve()
    top = _sh(["git", "-C", str(root), "rev-parse", "--show-toplevel"])
    if top.returncode:
        raise RecordError(f"{root} is not in a git checkout: {top.stderr.strip()}")
    repo = Path(top.stdout.strip())
    status = _sh(["git", "-C", str(repo), "status", "--porcelain=v1", "-z",
                  "--untracked-files=all", "--", str(root.relative_to(repo))])
    if status.returncode:
        raise RecordError(f"git status failed: {status.stderr.strip()}")
    paths = _changed(status.stdout)
    if commit and paths:
        tops = Counter(Path(p).relative_to(root.relative_to(repo)).parts[0] for p in paths)
        counts = ", ".join(f"{k} {n}" for k, n in sorted(tops.items()))
        message = f"data: collect {len(paths)} files ({counts})"
        done = _sh([_tool("git-commit-paths"), "-C", str(repo), "-m", message, "--", *paths])
        if done.returncode:
            raise RecordError(f"git-commit-paths failed: {done.stderr.strip()[-400:]}")
    return paths
