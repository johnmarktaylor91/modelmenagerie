"""A rung's check-in (``rungs/r2-checkin.md``), written from the committed files and the run.

Split from ``drive`` at rung 2; ``drive.checkin`` is the verb. Everything comes from the frozen
rung rows (``data/catalog/r2-rows.tsv``), the model records, packets and ledger, the catalog plan
(``data/catalog/plan.tsv``, when the harvest lane has written it) and ``drive.measures`` for the
run's lanes. The file is meant for the repository: home and work-root paths are replaced, and a
leftover absolute home path refuses the write. Imports neither torch nor psutil.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_root
from menagerie.pipeline import DataLayout, StoreLayout
from menagerie.pipeline.records import LedgerRow, ModelRecord, RecordError, RenderRow

ROWS = Path("catalog") / "r2-rows.tsv"
PLAN = Path("catalog") / "plan.tsv"
STAGES = ("rows", "packet", "recipe", "typed", "forward", "trace", "structure", "render",
          "reviewed", "settled")
SAMPLES = 10


def _tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _latest(rows: Sequence[Any], stage: str, rev: str | None) -> LedgerRow | None:
    kept = [r for r in rows if isinstance(r, LedgerRow) and r.stage == stage
            and r.recipe_rev == rev and r.outcome != "infra" and r.mode != "train"]
    return max(reversed(kept), key=lambda r: r.at) if kept else None


class _Model:
    """One row's state, read once: its record, its ledger rows and its next step."""

    def __init__(self, row: Mapping[str, str], files: Any, step: str) -> None:
        self.row, self.step = row, step
        self.record: ModelRecord | None = files.record(row["id"])
        family = self.record.family if self.record else files.family_of(row["id"])
        self.packet = files.packet(family) if family else None
        self.rows = files.rows.get(row["id"], [])
        self.rev = self.record.recipe.rev if self.record and self.record.recipe else None
        self.tl = files.tl
        from menagerie.pipeline.drive import kept_draft_traced  # drive imports this module

        # A kept draft that passed where its placement runs it is a traced row, not a typed one.
        self.give_up = None if kept_draft_traced(self.record, self.rows, tl=self.tl) else (
            self.record.give_up if self.record else None)

    def trace(self, capture: str) -> LedgerRow | None:
        kept = [r for r in self.rows if isinstance(r, LedgerRow) and r.stage == "trace"
                and r.recipe_rev == self.rev and r.tl == self.tl and r.outcome != "infra"
                and (r.capture or "full") == capture]
        return max(reversed(kept), key=lambda r: r.at) if kept else None

    def stages(self) -> set[str]:
        got = {"rows"} | self._research() | self._traced()
        if self.render():
            got.add("render")
        if _latest(self.rows, "review", self.rev):
            got.add("reviewed")
        if self.step == "done":
            got.add("settled")
        return got

    def _research(self) -> set[str]:
        flags = {"packet": self.packet is not None, "recipe": bool(self.rev),
                 "typed": self.give_up is not None}
        return {name for name, held in flags.items() if held}

    def _traced(self) -> set[str]:
        forward = _latest(self.rows, "forward", self.rev)
        got = {"forward"} if forward and forward.outcome == "pass" else set()
        for capture, name in (("full", "trace"), ("structure_only", "structure")):
            found = self.trace(capture)
            if found and found.outcome == "pass":
                got.add(name)
        return got

    def render(self) -> bool:
        saved = {t.trace or t.summary for t in (self.trace("full"), self.trace("structure_only"))
                 if t and t.outcome == "pass"}
        return any(isinstance(r, RenderRow) and r.trace in saved
                   and r.outcome in ("pass", "partial") for r in self.rows)

    def failure(self) -> str | None:
        """The typed class this row ended with: its give-up, else a failing stage's class."""

        if self.give_up is not None:
            return self.give_up.cls
        found = _latest(self.rows, "forward", self.rev)
        if found and found.outcome != "pass":
            return found.cls
        found = self.trace("full")
        if found and found.outcome != "pass":
            return found.cls
        from menagerie.pipeline.drive import render_failed  # drive imports this module

        return "render" if not self.render() and render_failed(
            self.rows, self.rev, tl=self.tl) else None


def _funnel(models: Sequence[_Model]) -> list[str]:
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for model in models:
        case = model.row.get("case", "?") + ("" if model.row.get("counted") == "1" else " (carried)")
        by_case[case].update(model.stages())
    head = "| Case | " + " | ".join(s.capitalize() for s in STAGES) + " |"
    out = [head, "|---" * (len(STAGES) + 1) + "|"]
    total: Counter[str] = Counter()
    for case, counts in sorted(by_case.items()):
        total += counts
        out.append(f"| {case} | " + " | ".join(str(counts[s]) for s in STAGES) + " |")
    out.append("| all | " + " | ".join(str(total[s]) for s in STAGES) + " |")
    return out


def _verdicts(models: Sequence[_Model]) -> tuple[list[str], list[LedgerRow]]:
    reviews = [r for m in models if (r := _latest(m.rows, "review", m.rev))]
    counts = Counter((_reviewer(r), r.verdict) for r in reviews)
    lines = ["| Reviewer | Verdict | Models |", "|---|---|---|"]
    lines += [f"| {who} | {verdict} | {n} |" for (who, verdict), n in sorted(counts.items())]
    return (lines if counts else ["No review at a current recipe revision yet."]), reviews


def _classes(models: Sequence[_Model]) -> list[str]:
    counts = Counter(c for m in models if (c := m.failure()))
    if not counts:
        return ["No typed class yet."]
    return ["| Class | Rows |", "|---|---|"] + [f"| `{c}` | {n} |" for c, n in counts.most_common()]


def _samples(models: Sequence[_Model]) -> list[str]:
    """Ten packets in a fixed hash order of their family ids."""

    packets = {m.packet.family: m.packet for m in models if m.packet is not None}
    chosen = sorted(packets, key=lambda f: hashlib.sha256(f.encode()).hexdigest())[:SAMPLES]
    if not chosen:
        return ["No packets yet."]
    out = ["| Family | Name | Paper | Code | License | Description | Members |",
           "|---|---|---|---|---|---|---|"]
    for family in chosen:
        packet = packets[family]
        have = packet.availability
        out.append(f"| `{family}` | {packet.identity.canonical} | {have.paper} | "
                   f"{have.code} | {have.license} | {have.description} | "
                   f"{len(packet.members)} |")
    return out


def _reviewer(row: LedgerRow) -> str:
    return row.reviewer or (row.env or "").removeprefix("review:")


def _summary(review: LedgerRow, store: StoreLayout | None) -> str:
    """The findings' summary, pulled back from the worker store when it is not staged here."""

    from menagerie.pipeline import store as stored

    if store is None or not review.findings:
        return ""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = stored.materialize(review.findings, Path(tmp) / "findings.json", store=store)
            found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecordError, subprocess.SubprocessError):
        return ""
    return str(found.get("summary", "")).replace("|", "/").replace("\n", " ")[:300]


def _findings(reviews: Sequence[LedgerRow], store: StoreLayout | None) -> list[str]:
    flagged = [r for r in reviews if r.verdict in ("fabrication", "minor", "cannot_judge")]
    if not flagged:
        return ["No fabrication, minor or cannot-judge verdict."]
    out = ["| Model | Verdict | Reviewer | Summary |", "|---|---|---|---|"]
    out += [f"| {r.id} | {r.verdict} | {_reviewer(r)} | {_summary(r, store)} |"
            for r in sorted(flagged, key=lambda r: r.id)]
    return out


def _renders(models: Sequence[_Model]) -> list[str]:
    done = [m.row["id"] for m in models if m.render()]
    return [f"{len(done)} of {len(models)} rows have a render of their current trace"
            + (f": {', '.join(done[:20])}" + (" and more" if len(done) > 20 else "") if done
               else "") + "."]


def _catalog_state(row: Mapping[str, str], files: Any) -> str:
    """A catalog row's state: validated, traced (not validated), typed or pending."""

    from menagerie.pipeline import ledger

    record = files.record(row["id"]) if row.get("id") else None
    rev = record.recipe.rev if record and record.recipe else None
    trace = ledger.deciding_trace(row["id"], files.rows.get(row["id"], []), recipe_rev=rev,
                                  tl=files.tl) if rev else None
    if ledger.validated(trace):
        return "validated"
    if (record and record.give_up) or (trace and trace.outcome != "pass"):
        return "typed"
    return "traced" if trace else "pending"


def _catalog(data: DataLayout, files: Any, day: str) -> list[str]:
    plan = _tsv(data.root / PLAN)
    if not plan:
        return [f"No catalog plan (`data/{PLAN.as_posix()}`) yet; counts dated {day}."]
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in plan:
        counts[(row.get("group", "?"), row.get("wave", "?"))].update(
            ["rows", _catalog_state(row, files)])
    out = [f"Counts dated {day}, TorchLens {files.tl}.", "",
           "| Group | Wave | Rows | Validated | Traced, not validated | Typed | Pending |",
           "|---|---|---|---|---|---|---|"]
    out += [f"| {g} | {w} | {c['rows']} | {c['validated']} | {c['traced']} | {c['typed']} | "
            f"{c['pending']} |" for (g, w), c in sorted(counts.items())]
    return out


def _lanes(run_id: str, data: DataLayout) -> list[str]:
    from menagerie.pipeline import drive

    found = drive.measures(run_id, data=data)
    if not found["lanes"]:
        return [f"No lanes under run `{run_id}` in fleet-run's records."]
    md = Path(found["files"][-1]).read_text(encoding="utf-8")
    body = md.split("## Per tier (median / total)", 1)[-1].split("## Lanes", 1)[0].strip()
    def tally(key: str) -> str:
        return ", ".join(f"{k} {n}" for k, n in sorted(Counter(r[key] for r in found["lanes"]).items()))

    return [f"{len(found['lanes'])} lanes; by kind: {tally('kind')}; by harness: "
            f"{tally('harness')}.", "", body]


def _scrub(text: str) -> str:
    for root, name in ((str(work_root()), "<work root>"), (str(Path.home()), "~")):
        text = text.replace(root, name)
    if "/home/" in text or "/Users/" in text:
        raise RecordError("the check-in still holds an absolute home path; refusing to write it")
    return text


def write(run_id: str, *, data: DataLayout, out: Path) -> Path:
    """Write the check-in for ``run_id``; see ``drive.checkin``."""

    from menagerie.pipeline import drive

    rows = _tsv(data.root / ROWS)
    if not rows:
        raise RecordError(f"no rung rows at data/{ROWS.as_posix()}")
    files = drive._load(data)
    steps = {s["id"]: s["step"] for s in drive.next_steps([r["id"] for r in rows], data=data)}
    models = [_Model(row, files, steps[row["id"]]) for row in rows]
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        store: StoreLayout | None = StoreLayout.from_env()
    except (KeyError, RecordError, OSError):
        store = None
    verdicts, reviews = _verdicts(models)
    counted = sum(r.get("counted") == "1" for r in rows)
    sections = [
        (f"# Rung 2 check-in ({day})", [
            f"Run `{run_id}`; {counted} counted rows and {len(rows) - counted} carried; "
            f"TorchLens {files.tl}. Stage counts are rows that reached each stage at their "
            "current recipe revision; settled rows have no next step (passed or typed)."]),
        ("## Funnel by case", _funnel(models)),
        ("## Review verdicts", verdicts),
        ("## Give-up and failure classes", _classes(models)),
        ("## Ten sample packets", _samples(models)),
        ("## Review findings", _findings(reviews, store)),
        ("## Renders", _renders(models)),
        ("## Catalog pass by group", _catalog(data, files, day)),
        ("## Lanes, minutes and tokens", _lanes(run_id, data)),
    ]
    text = "\n\n".join(f"{head}\n\n" + "\n".join(body) for head, body in sections) + "\n"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_scrub(text), encoding="utf-8")
    return out
