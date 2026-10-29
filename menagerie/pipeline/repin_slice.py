"""Bounded Slice A repin checks, reusing the frozen regression and sandbox runner."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from menagerie import host_budget
from menagerie.pipeline import DataLayout, read_json, regression, runner
from menagerie.pipeline.records import Caps, ModelRecord, RecordError

BASELINE = "regression/slice_a_200_r1.tsv"
RESERVE_S = 90
COLUMNS = (
    "stable_id", "name", "batch", "r1", "r3", "r1_n_ops", "r3_n_ops", "r1_hash", "r3_hash",
    "r1_structure_only", "r3_structure_only", "agree", "note", "class", "tl", "env",
    "rss_cap_mb", "threads", "peak_rss_mb", "wall_s",
)


def compare(rows: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, str]], *,
            explanations: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Compare verdicts, op counts, legacy hashes and required structure-only verdicts.

    A difference remains unexplained until a caller supplies a reproduced explanation.
    Missing rows and infrastructure failures cannot be explained away.
    """

    full = {r["stable_id"]: r for r in rows if r["mode"] == "trace"}
    shape = {r["stable_id"]: r for r in rows if r["mode"] == "trace-structure-only"}
    return [_compare_one(old, full.get(old["stable_id"], {}),
                         shape.get(old["stable_id"], {}), explanations or {})
            for old in baseline]


def _compare_one(old: Mapping[str, str], new: Mapping[str, Any], meta: Mapping[str, Any],
                 explanations: Mapping[str, str]) -> dict[str, Any]:
    """One baseline join with an explicit, non-infrastructure difference classification."""

    # Essential complexity: compare full and required meta evidence in one joined row.
    sid = old["stable_id"]
    row = {"stable_id": sid, "name": old["name"], "batch": old["batch"],
           "r1": old["r1"], "r3": new.get("status", "missing"),
           "r1_n_ops": old["r1_n_ops"], "r3_n_ops": new.get("n_ops"),
           "r1_hash": old["r1_hash"], "r3_hash": new.get("graph_hash"),
           "r1_structure_only": old["r1_structure_only"],
           "r3_structure_only": meta.get("status", ""),
           **{k: new.get(k) for k in COLUMNS[13:]}}
    differences = [key for key in ("", "_n_ops", "_hash", "_structure_only")
                   if str(row[f"r1{key}"] or "") != str(row[f"r3{key}"] or "")]
    row["agree"] = "same" if not differences else "unexplained"
    row["note"] = "; ".join(f"{key or 'verdict'} differs" for key in differences)
    note = explanations.get(sid)
    required_meta = bool(old["r1_structure_only"])
    explainable = row["r3"] not in ("missing", "infra") and (
        not required_meta or row["r3_structure_only"] not in ("", "missing", "infra"))
    if differences and note and explainable:
        row["agree"], row["note"] = "explained", note
    return row


def _caps(batch: str, *, smoke: bool, deadline: float | None) -> Caps:
    cpus = host_budget.declared_cpus()
    caps = regression.caps_for(batch, cpus)
    if smoke:
        workers = min(regression.BATCHES[batch][0], cpus)
        caps = replace(caps, rss_mb=int(host_budget.declared_memory_gb() * 1024 / workers))
    if deadline is not None:
        remaining = int(deadline - time.time() - RESERVE_S)
        if remaining < 1:
            raise TimeoutError("repin check has reached its stopping deadline")
        caps = replace(caps, timeout_s=min(caps.timeout_s, remaining))
    return caps


def _wave(ids: Sequence[str], batch: str, mode: str, *, data: DataLayout,
          spool: Path, smoke: bool, deadline: float | None, threads: int | None = None) -> None:
    caps = _caps(batch, smoke=smoke, deadline=deadline)
    if threads is not None:
        caps = replace(caps, threads=threads)
    for sid in ids:
        record = read_json(ModelRecord, data.model(sid))
        request = runner.request_for(record, mode, caps=caps, intervention_ready=True)
        runner.submit(request, spool=spool)
    runner.serve(spool=spool, workers=regression.BATCHES[batch][0], once=True)


def _evidence(spool: Path, out: Path) -> None:
    """Keep compact per-attempt diagnostics; trace payloads are regenerable check artifacts."""

    target = out / "evidence"
    target.mkdir(exist_ok=True)
    for attempt in sorted((spool / "attempts").glob("*")):
        for name in ("request.json", "result.json", "summary.json"):
            source = attempt / name
            if source.is_file():
                (target / f"{attempt.name}-{name}").write_bytes(source.read_bytes())
        log = attempt / "attempt.log"
        if log.is_file():
            (target / f"{attempt.name}.log").write_text(
                "\n".join(log.read_text(errors="replace").splitlines()[-40:]) + "\n")


def run(*, data: DataLayout, out: Path, ids: Sequence[str] = (), smoke: bool = False,
        deadline: float | None = None,
        explanations: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run the original 182 small/18 large batches, with a deadline and smoke mode.

    Smoke mode uses the admitted job's smaller caps and never qualifies the pin flip.
    Each wave takes at most two small rows or one large row, so its timeout bounds shutdown.
    """

    import tempfile

    baseline = _selection(data, ids, smoke)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="repin-") as temporary:
        spool = Path(temporary)
        try:
            _batches(baseline, data=data, spool=spool, smoke=smoke, deadline=deadline)
            rows = [r for r, _ in regression.collect(spool)]
            _structures(baseline, rows, data=data, spool=spool, smoke=smoke, deadline=deadline)
        except TimeoutError:
            if deadline is None or time.time() < deadline - RESERVE_S - 1:
                raise  # only the stopping deadline is an expected partial run
        rows = [r for r, _ in regression.collect(spool)]
        _evidence(spool, out)
    return _report(rows, baseline, out, smoke, explanations)


def _selection(data: DataLayout, ids: Sequence[str], smoke: bool) -> list[dict[str, str]]:
    """Refuse unknown rows and production runs outside the original resource contract."""

    if not smoke and (host_budget.declared_cpus() != 6
                      or host_budget.declared_memory_gb() != 40):
        raise RecordError("Slice A requires exactly 6 CPUs and 40 GiB; use --smoke for a dry run")
    baseline = regression.read_tsv(data.root / BASELINE)
    chosen = set(ids) or {r["stable_id"] for r in baseline}
    unknown = chosen - {r["stable_id"] for r in baseline}
    if unknown:
        raise RecordError(f"not Slice A ids: {sorted(unknown)}")
    return [r for r in baseline if r["stable_id"] in chosen]


def _report(rows: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, str]],
            out: Path, smoke: bool, explanations: Mapping[str, str] | None) -> dict[str, Any]:
    """Persist comparison and proof flags even when a deadline left the run incomplete."""

    table = compare(rows, baseline, explanations=explanations)
    regression.write_tsv(table, out / "slice_a_200_r3.tsv", COLUMNS)
    counts = Counter(r["agree"] for r in table)
    complete = len(table) == 200 and all(r["r3"] not in ("missing", "infra") for r in table)
    report = {"same": counts["same"], "explained": counts["explained"],
              "unexplained": counts["unexplained"], "rows": len(table),
              "complete": complete, "smoke": smoke,
              "explanations": dict(explanations or {}),
              "passed": complete and not smoke and counts["unexplained"] == 0}
    (out / "summary.md").write_text(
        f"Same {report['same']}; explained {report['explained']}; unexplained {report['unexplained']} "
        "against slice_a_200_r1.tsv.\n\n"
        f"Rows: {len(table)}; complete: {complete}; smoke: {smoke}; passed: {report['passed']}.\n")
    (out / "slice.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _batches(baseline: Sequence[Mapping[str, str]], *, data: DataLayout, spool: Path,
             smoke: bool, deadline: float | None) -> None:
    for batch in regression.BATCHES:
        ids = [r["stable_id"] for r in baseline if r["batch"] == batch]
        width = regression.BATCHES[batch][0]
        for offset in range(0, len(ids), width):
            if deadline is not None and time.time() >= deadline - RESERVE_S:
                return
            _wave(ids[offset:offset + width], batch, "trace", data=data, spool=spool,
                  smoke=smoke, deadline=deadline)


def _structures(baseline: Sequence[Mapping[str, str]], rows: Sequence[Mapping[str, Any]], *,
                data: DataLayout, spool: Path, smoke: bool, deadline: float | None) -> None:
    needed = {r["stable_id"] for r in rows if r["status"] == "failed:memory_cap"}
    needed |= {r["stable_id"] for r in baseline if r["r1_structure_only"]}
    for sid in sorted(needed):
        if deadline is not None and time.time() >= deadline - RESERVE_S:
            return
        _wave([sid], "large", "trace-structure-only", data=data, spool=spool,
              smoke=smoke, deadline=deadline)
