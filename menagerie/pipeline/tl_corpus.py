"""The TorchLens failure corpus (``review torchlens-corpus``; decisions.md 2026-11-04).

Finding TorchLens bugs is part of Menagerie's purpose: every rung's failed traces that TorchLens
may own go to TorchLens as a clustered corpus, and re-trace when a new TorchLens is pinned.

A corpus row is a model whose latest full eval trace at its current recipe failed with a class
TorchLens may own (``capture_error*``, ``torchlens_gap:*``, ``torchlens_unverified``), or whose
record carries a typed ``torchlens_gap:*`` give-up. Each row is attributed (``tl_attribution``)
and clustered by a signature from its class and normalized message. Read-only over data files;
the outputs are ``corpus.jsonl``, ``clusters.md`` and ``summary.json`` in ``out`` plus a dated
copy under ``out/history/<date>/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, read_json
from menagerie.pipeline import tl_attribution as attrib
from menagerie.pipeline.records import GiveUp, LedgerRow, ModelRecord

CORPUS_HEADS = ("capture_error", "torchlens_gap", "torchlens_unverified")
REPRODUCE = "python3 -m menagerie.pipeline trace {id}"
EARLIER = "earlier"  # rows not on the tick worklist: rungs 0 to 2 and anything off-list
_NUM = re.compile(r"\b0x[0-9a-f]+\b|\d+")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_QUALIFIED = re.compile(r"[\w.<>]+\.forward\(\)")
_CHECK = re.compile(r"\b(\w+) mismatch\b")
_OP = re.compile(r"mismatch at \S+ \(([^)]+)\)")
_UNREACHED = re.compile(r"unreached \(e\.g\. \['([a-z_]+?)_\d")
_REASON = re.compile(r"unverified \(([\w:]+)\)")
_TAG = re.compile(r"^\[(\w+)\]")
KEEP_QUOTES = ("AttributeError", "KeyError", "ModuleNotFoundError", "ImportError")


# --- signatures ------------------------------------------------------------------------------

def _norm(text: str, *, quotes: bool = True, width: int = 90) -> str:
    text = _QUALIFIED.sub("forward()", text)
    text = _QUOTED.sub("'…'", text) if quotes else text
    text = re.sub(r"\s+", " ", _NUM.sub("N", text)).strip()
    return text if len(text) <= width else text[:width].rsplit(" ", 1)[0] + " …"


def _replay(msg: str) -> str:
    check = _CHECK.search(msg)
    if check is None:
        return "replay " + _norm(msg.split("; no save")[0], width=60)
    name, detail = check[1], msg.split(" -- ", 1)[-1]
    if name == "bfs_completeness":
        unreached = _UNREACHED.search(msg)
        return ("replay bfs_completeness: unreached " + unreached[1] if unreached else
                "replay bfs_completeness: " + _norm(detail.split(":")[0], width=60))
    op = _OP.search(msg)
    return f"replay {name} at {op[1]}" if op else f"replay {name}: {_norm(detail, width=60)}"


def _capture(msg: str) -> str:
    if msg.startswith(attrib.REFUSAL):
        detail = msg.split(" -- ", 1)[-1]
        cut = re.split(r": \(|: \[|\(e\.g\.", detail, maxsplit=1)[0]
        return "refusal: " + _norm(cut, width=80)
    kind = attrib.exception_type(msg) or "error"
    text = msg.split(": ", 1)[1] if ": " in msg else ""
    tag = _TAG.match(text)
    if tag:
        return f"{kind} [{tag[1]}] " + _norm(text[tag.end():], width=70)
    return f"{kind}: " + _norm(text, quotes=kind not in KEEP_QUOTES)


def signature(cls: str, msg: str | None, *, giveup: bool = False) -> str:
    """The cluster key of one corpus row: its class and its normalized message."""

    msg = re.sub(r"^(single-thread retry|passed on the single-thread retry after): ", "",
                 msg or "")
    if giveup:
        return f"giveup {cls}"
    if cls == "torchlens_gap:replay":
        return _replay(msg)
    if cls.startswith("torchlens_unverified"):
        reason = _REASON.search(msg)
        return "unverified " + (reason[1] if reason else _norm(msg, width=60))
    if cls.startswith("torchlens_gap"):
        return f"{cls}: " + _norm(msg, width=70)
    return _capture(msg)


def cluster_id(sig: str) -> str:
    return "c-" + hashlib.sha256(sig.encode()).hexdigest()[:8]


# --- inputs ----------------------------------------------------------------------------------

def worklist_cohorts(path: Path) -> dict[str, str]:
    """``<id> [cohort]`` lines of the tick worklist; an untagged row is ``untagged``."""

    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("#", 1)[0].split()
        if parts:
            found.setdefault(parts[0], parts[1] if len(parts) > 1 else "untagged")
    return found


def collected_jobs(folder: Path) -> dict[tuple[str, str], str]:
    """The latest collected stage 3 job name per (id, recipe revision)."""

    jobs: dict[tuple[str, str], str] = {}
    names = sorted(p.name[:-5] for p in folder.glob("*.json")) if folder.is_dir() else []
    for name in names:
        parts = name.split("-")
        if len(parts) >= 4:
            jobs[(parts[0], parts[1])] = name
    return jobs


def _current_traces(rows: Iterable[Any], records: Mapping[str, ModelRecord]
                    ) -> dict[str, list[LedgerRow]]:
    by: dict[str, list[LedgerRow]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, LedgerRow) or row.stage != "trace" or row.outcome == "infra":
            continue
        if row.capture not in (None, "full") or row.mode not in (None, "eval"):
            continue
        record = records.get(row.id)
        if record is not None and record.recipe is not None and row.recipe_rev == record.recipe.rev:
            by[row.id].append(row)
    return by


def _latest(rows: Sequence[LedgerRow]) -> LedgerRow | None:
    return max(reversed(rows), key=lambda r: r.at) if rows else None


def _recipe(record: ModelRecord) -> dict[str, Any] | None:
    r = record.recipe
    if r is None:
        return None
    return {k: v for k, v in (("rev", r.rev), ("kind", r.kind), ("env", r.env),
                              ("adapter", r.adapter), ("call", r.call)) if v}


# --- rows ------------------------------------------------------------------------------------

def _row(record: ModelRecord, cls: str, msg: str, tl: str | None, *, cohort: str,
         trace: LedgerRow | None, index: attrib.SourceIndex | None, job: str | None,
         giveup: str | None) -> dict[str, Any]:
    state = trace.validation_state if trace is not None and trace.cls == cls else None
    frames = trace.frames if trace is not None and trace.cls == cls else None
    who = attrib.attribute(cls, msg, tl, frames, index, state=state)
    sig = signature(cls, msg, giveup=giveup is not None and trace is None)
    out = {"id": record.id, "name": record.name, "family": record.family, "cohort": cohort,
           "recipe": _recipe(record), "tl": tl, "class": cls, "attribution": who.who,
           "evidence_kind": who.kind, "evidence": who.evidence, "excerpt": msg[:500],
           "cluster": cluster_id(sig), "signature": sig,
           "reproduce": REPRODUCE.format(id=record.id)}
    if trace is not None:
        out.update(at=trace.at, trace=trace.trace, attempt_log=trace.evidence, job=job)
        out.update({k: v for k, v in (("validation_state", state), ("frames", frames)) if v})
    if giveup:
        out["giveup"] = giveup
    return {k: v for k, v in out.items() if v is not None}


def collect(rows: Sequence[Any], records: Mapping[str, ModelRecord], *,
            cohorts: Mapping[str, str], jobs: Mapping[tuple[str, str], str],
            index: attrib.SourceIndex | None, inactive: Iterable[str] = ()
            ) -> tuple[list[dict[str, Any]], Counter[str], list[str]]:
    """Corpus rows, the other latest outcomes by class head, and fixed TorchLens give-ups."""

    traces, inactive = _current_traces(rows, records), set(inactive)
    corpus: list[dict[str, Any]] = []
    other: Counter[str] = Counter()
    fixed: list[str] = []
    for sid in sorted(records, key=_order):
        if sid in inactive:
            continue
        where, row = _place(records[sid], _latest(traces.get(sid, [])), jobs=jobs,
                            cohort=cohorts.get(sid, EARLIER), index=index)
        if row is not None:
            corpus.append(row)
        elif where == "fixed":
            fixed.append(sid)
        elif where:
            other[where] += 1
    return corpus, other, fixed


def _place(record: ModelRecord, latest: LedgerRow | None, *, cohort: str,
           jobs: Mapping[tuple[str, str], str], index: attrib.SourceIndex | None
           ) -> tuple[str | None, dict[str, Any] | None]:
    """One model's place: ("corpus", row), ("fixed", None), (latest class head, None) or none."""

    give = record.give_up
    head = (latest.cls or "").split(":", 1)[0] if latest is not None else ""
    common = {"cohort": cohort, "index": index, "giveup": give.cls if give else None}
    if latest is not None and latest.outcome == "fail" and head in CORPUS_HEADS:
        return "corpus", _row(record, latest.cls or "", latest.msg or "", latest.tl, trace=latest,
                              job=jobs.get((record.id, latest.recipe_rev)), **common)
    if give is not None and give.cls.startswith("torchlens_gap"):
        return _tl_giveup(record, give, latest, common)
    if latest is None:
        return None, None
    return ("pass" if latest.outcome == "pass" else head or latest.outcome), None


def _tl_giveup(record: ModelRecord, give: GiveUp, latest: LedgerRow | None,
               common: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """A typed ``torchlens_gap`` give-up: fixed when a later trace passed, else a corpus row."""

    if latest is not None and latest.outcome == "pass" and latest.at > give.at:
        return "fixed", None
    return "corpus", _row(record, give.cls, give.reason, latest.tl if latest else None,
                          trace=None, job=None, **common)


def _order(sid: str) -> tuple[int, str]:
    digits = sid[1:]
    return (int(digits), sid) if digits.isdigit() else (1 << 62, sid)


# --- clusters and files ----------------------------------------------------------------------

def clusters(corpus: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Corpus rows grouped by signature, largest first; representatives at the newest pin."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in corpus:
        groups[row["cluster"]].append(row)
    out = []
    for cid, members in groups.items():
        reps = sorted(members, key=lambda r: (-_pin_key(r.get("tl")), _order(r["id"])))
        who = Counter(r["attribution"] for r in members).most_common(1)[0][0]
        out.append({"id": cid, "signature": members[0]["signature"], "count": len(members),
                    "attribution": who, "evidence": reps[0]["evidence"],
                    "pins": dict(Counter(r.get("tl") or "none" for r in members)),
                    "cohorts": dict(Counter(r["cohort"] for r in members)),
                    "representatives": [r["id"] for r in reps[:3]],
                    "names": [r["name"] for r in reps[:3]],
                    "excerpt": reps[0]["excerpt"][:300], "reproduce": reps[0]["reproduce"],
                    "ids": sorted((r["id"] for r in members), key=_order)})
    return sorted(out, key=lambda c: (-c["count"], c["signature"]))


def _pin_key(pin: str | None) -> int:
    parts = [int(p) for p in re.findall(r"\d+", pin or "")[:3]]
    return sum(p * 1000 ** (2 - i) for i, p in enumerate(parts))


SECTIONS = ((attrib.TORCHLENS, "Attributed to TorchLens"),
            (attrib.UNATTRIBUTED, "Not yet attributed (capture-only failures)"),
            (attrib.MENAGERIE, "Raised by Menagerie (for completeness)"))


def markdown(summary: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> str:
    """``clusters.md``: the corpus for a TorchLens reader, largest cluster first per section."""

    lines = _header(summary, groups)
    for who, title in SECTIONS:
        part = [g for g in groups if g["attribution"] == who]
        if part:
            lines += [f"## {title}: {sum(g['count'] for g in part)} models, {len(part)} clusters",
                      ""]
            lines += [line for g in part for line in _cluster_lines(g)]
    return "\n".join(lines)


def _cluster_lines(g: Mapping[str, Any]) -> list[str]:
    reps = ", ".join(f"{i} ({n})" for i, n in zip(g["representatives"], g["names"]))
    pins = ", ".join(f"{k} {v}" for k, v in g["pins"].items())
    return [f"### {g['id']}: {g['signature']} ({g['count']})", "",
            f"- Models: {g['count']} (pins {pins}); representatives {reps}",
            f"- Evidence: {g['evidence']}",
            f"- Excerpt: `{g['excerpt'].replace('`', chr(39))}`",
            f"- Reproduce: `{g['reproduce']}`", ""]


def _header(summary: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> list[str]:
    n = summary["counts"]
    other = ", ".join(f"{k} {v}" for k, v in summary["not_in_corpus"].items() if k != "pass")
    return [f"# TorchLens failure corpus from Model Menagerie ({summary['generated']})", "",
             (f"{n['models']} models in {len(groups)} clusters: {n[attrib.TORCHLENS]} attributed "
              f"to TorchLens, {n[attrib.UNATTRIBUTED]} not yet attributed, "
              f"{n[attrib.MENAGERIE]} raised by Menagerie. Failing pins: "
              + ", ".join(f"{k} ({v})" for k, v in summary["pins"].items()) + "."), "",
             ("Attribution needs evidence: the raising frame, TorchLens's own validation status, "
              "a raise site in the TorchLens source at that pin, or an exception class TorchLens "
              "defines. Each model re-traces when Menagerie pins a new TorchLens release; a model "
              "that then passes leaves the corpus."), "",
             ("Reproduce: run the command from a Menagerie checkout on a worker with the runner's "
              "environment (MENAGERIE_DATA, MENAGERIE_STORE and MENAGERIE_ENVS as "
              "`menagerie/pipeline/runner_job.sh` sets them); it traces the model at its "
              "committed recipe with the pinned TorchLens. Every row: `corpus.jsonl`."), "",
             f"Not in the corpus (latest trace not a TorchLens-side class): {other or 'none'}.", ""]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write(out: Path, corpus: Sequence[Mapping[str, Any]], summary: Mapping[str, Any],
          groups: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The latest files in ``out`` and a dated copy in ``out/history/<date>``."""

    files = {"corpus.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in corpus),
             "clusters.md": markdown(summary, groups),
             "summary.json": json.dumps({**summary, "clusters": groups}, indent=1) + "\n"}
    dated = out / "history" / summary["generated"][:10]
    for name, text in files.items():
        _write(out / name, text)
        dated.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / name, dated / name)
    return {"out": str(out), "dated": str(dated)}


# --- the command -----------------------------------------------------------------------------

def _records(data: DataLayout) -> dict[str, ModelRecord]:
    found = (read_json(ModelRecord, p) for p in sorted((data.root / "models").glob("*/*.json")))
    return {r.id: r for r in found}


def _inactive(data: DataLayout) -> set[str]:
    from menagerie.pipeline import roster

    return {r.stable_id for r in roster.read(data=data) if r.stable_id and r.state != "active"}


def build(*, data: DataLayout, cohort: str | None = None, out: Path | None = None,
          worklist: Path | None = None, index: attrib.SourceIndex | None = None,
          now: datetime | None = None, write_files: bool = True) -> dict[str, Any]:
    """Build the corpus over every cohort (or one), write it to ``out``; returns the summary."""

    from menagerie.pipeline import ledger

    index = index or attrib.SourceIndex(attrib.source_root(), work_dir("torchlens-corpus-cache"))
    cohorts = worklist_cohorts(worklist or work_dir("tick") / "worklist.txt")
    corpus, other, fixed = collect(ledger.read_all(data=data), _records(data), cohorts=cohorts,
                                   jobs=collected_jobs(work_dir("stage3") / "collected"),
                                   index=index, inactive=_inactive(data))
    index.save()
    if cohort is not None:
        corpus = [r for r in corpus if r["cohort"] == cohort]
    groups = clusters(corpus)
    when = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%MZ")
    who = Counter(r["attribution"] for r in corpus)
    summary = {"generated": when, "cohort": cohort or "all",
               "counts": {"models": len(corpus), "clusters": len(groups),
                          **{k: who.get(k, 0) for k, _ in SECTIONS}},
               "by_evidence": dict(Counter(r["evidence_kind"] for r in corpus)),
               "by_class": dict(Counter(r["class"].split(":")[0] for r in corpus)),
               "pins": dict(Counter(r.get("tl") or "none" for r in corpus).most_common()),
               "cohorts": dict(Counter(r["cohort"] for r in corpus).most_common()),
               "not_in_corpus": dict(other.most_common()), "fixed_giveups": fixed}
    if write_files:
        summary.update(write(out or work_dir("torchlens-corpus"), corpus, summary, groups))
    return {**summary, "clusters": groups, "rows": corpus}
