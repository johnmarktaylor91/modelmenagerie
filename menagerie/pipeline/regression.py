"""Rung 0 regression: the Slice A 200 through the new runner, compared with the June validator.

Owned by integration. ``spool`` queues one batch of the frozen table
(``data/regression/slice_a_200.tsv``) as trace requests with the June baseline's caps: the
batch's per-worker RSS cap, threads split from the job's declaration, and a hang-catcher
timeout. ``runner serve --once`` drains it. ``collect`` reads a drained spool into one row per
model (and the ledger rows, and the store puts of the saved traces); ``compare`` joins those rows
with the June validator's manifests at the pin and the frozen Slice A table into the r0 table.
Torch-free except through the runner on the worker.

Usage: ``python -m menagerie.pipeline.regression spool|collect|compare ...`` (``--help``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, read_json
from menagerie.pipeline.records import Caps, LedgerRow, ModelRecord, RunResult

# Batch: (workers, per-worker RSS cap in GiB), as the June baseline ran them under 6 CPUs, 40 GiB.
BATCHES: dict[str, tuple[int, int]] = {"small": (2, 20), "large": (1, 40)}
# The June validator's timeout is a hang-catcher with at least an hour for unmeasured rows and a
# ceiling floor of three hours (validate_menagerie.py); the new trace mode makes three captures.
TIMEOUT_S = 10800
TABLE = "regression/slice_a_200.tsv"
# June statuses for the new path's outcomes (the worker types replay failures torchlens_gap). June's
# validator returned False for an unverified retrace too, so `torchlens_unverified` compares there.
STATUS_OF_CLASS = {
    "torchlens_gap": "failed:replay", "torchlens_unverified": "failed:replay",
    "memory_cap": "failed:memory_cap",
    "timeout": "failed:timeout", "capture_error": "failed:exception",
    "construct_error": "failed:exception", "forward_error": "failed:exception",
    "missing_dependency": "failed:exception", "infra": "infra",
}
RESOURCE = ("failed:memory_cap", "failed:timeout", "failed:killed")
COLUMNS = (
    "stable_id", "name", "batch", "mode", "intervention_ready", "outcome", "status", "class",
    "n_ops", "graph_hash", "runnable", "peak_rss_mb", "wall_s", "forward_secs", "capture_secs",
    "save_capture_secs", "env", "tl", "rss_cap_mb", "threads", "timeout_s", "msg",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated file with a header into dictionaries."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(rows: Iterable[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> None:
    """Write dictionaries as a tab-separated file; tabs and newlines in values become spaces."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            cells = ("" if row.get(c) is None else " ".join(str(row[c]).split()) for c in columns)
            handle.write("\t".join(cells) + "\n")


def frozen(data: DataLayout) -> list[dict[str, str]]:
    """Return the frozen Slice A 200 rows (ids, June verdicts, batches)."""

    return read_tsv(data.root / TABLE)


def caps_for(batch: str, cpus: int) -> Caps:
    """Return the baseline's caps for one batch under a job declaring ``cpus``."""

    workers, cap_gib = BATCHES[batch]
    return Caps(rss_mb=cap_gib * 1024, timeout_s=TIMEOUT_S, threads=max(1, cpus // workers))


def spool(spool_dir: Path, batch: str, *, data: DataLayout, cpus: int,
          intervention_ready: bool = True, ids: Sequence[str] = (),
          mode: str = "trace") -> list[str]:
    """Queue ``mode`` requests for a batch (or for ``ids``) on the spool; return the ids queued."""

    from menagerie.pipeline import runner

    wanted = set(ids)
    # Trace requests keep each row in its own batch's caps; a structure-only follow-up takes the
    # listed rows from either batch under the given batch's caps.
    by_batch = mode == "trace" or not wanted
    rows = [r for r in frozen(data) if (not by_batch or r["batch"] == batch)
            and (not wanted or r["stable_id"] in wanted)]
    queued = []
    for row in rows:
        record = read_json(ModelRecord, data.model(row["stable_id"]))
        request = runner.request_for(record, mode, caps=caps_for(batch, cpus),  # type: ignore[arg-type]
                                     intervention_ready=intervention_ready)
        runner.submit(request, spool=spool_dir)
        queued.append(record.id)
    return queued


def sample(data: DataLayout, *, size: int = 50, max_peak_mb: int = 12 * 1024) -> list[str]:
    """A stratified sample of the small batch for the ``intervention_ready`` ablation.

    Strata are the Slice A verdict (validated or not) crossed with op-count tertiles; each
    stratum gives its share of ``size``, picked by the sha256 of the id, so the sample is fixed.
    Rows above ``max_peak_mb`` in Slice A are left out so one attempt fits a 16 GiB slot, and so
    are June's stand-ins, whose Slice A peak belongs to the stand-in, not the real model.
    """

    import hashlib

    fake = stand_ins(data)
    rows = [r for r in frozen(data) if r["batch"] == "small" and r["stable_id"] not in fake
            and int(r["peak_rss_mb"] or 0) <= max_peak_mb]
    ops = sorted(int(r["n_ops"] or 0) for r in rows)
    cuts = (ops[len(ops) // 3], ops[2 * len(ops) // 3])
    strata: dict[tuple[bool, int], list[dict[str, str]]] = {}
    for row in rows:
        n_ops = int(row["n_ops"] or 0)
        tertile = 0 if n_ops < cuts[0] else 1 if n_ops < cuts[1] else 2
        strata.setdefault((row["status"] == "validated", tertile), []).append(row)
    picked = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda r: hashlib.sha256(
            r["stable_id"].encode()).hexdigest())
        picked += members[:max(1, round(size * len(members) / len(rows)))]
    return [r["stable_id"] for r in picked][:size]


def spool_pairs(spool_dir: Path, ids: Sequence[str], *, data: DataLayout, caps: Caps) -> int:
    """Queue each id twice, ``intervention_ready`` on then off, so each pair runs back to back."""

    import os
    import time

    from menagerie.pipeline import runner

    count = 0
    for sid in ids:
        record = read_json(ModelRecord, data.model(sid))
        for ready in (True, False):
            path = runner.submit(runner.request_for(record, "trace", caps=caps,
                                                    intervention_ready=ready), spool=spool_dir)
            queued = Path(spool_dir) / "new" / path.name
            stamp = time.time() + count * 1e-3  # FIFO by mtime: keep the pair order exact
            os.utime(queued, (stamp, stamp))
            count += 1
    return count


def _trace_row(result: RunResult) -> LedgerRow | None:
    return next((r for r in result.rows if isinstance(r, LedgerRow) and r.stage == "trace"), None)


def status_of(result: RunResult) -> str:
    """Return the June-vocabulary status of a trace result."""

    if result.outcome == "pass":
        return "validated"
    head = (result.cls or "infra").split(":", 1)[0]
    return STATUS_OF_CLASS.get(head, f"failed:{head}")


def _seconds(started: str, ended: str) -> int:
    from datetime import datetime

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return int((datetime.strptime(ended, fmt) - datetime.strptime(started, fmt)).total_seconds())


def row_of(result: RunResult, request: Mapping[str, Any], attempt: Path) -> dict[str, Any]:
    """Flatten one trace result, its request caps and its attempt summary into a table row."""

    trace = _trace_row(result)
    summary_path = attempt / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    caps = request.get("caps", {})
    get = (lambda name: getattr(trace, name)) if trace else (lambda name: None)
    return {
        "stable_id": result.id, "mode": result.mode,
        "intervention_ready": request.get("intervention_ready"), "outcome": result.outcome,
        "status": status_of(result), "class": result.cls, "n_ops": get("n_ops"),
        "graph_hash": get("graph_hash"), "runnable": get("runnable"),
        "peak_rss_mb": get("peak_rss_mb"), "wall_s": _seconds(result.started, result.ended),
        "forward_secs": get("forward_secs"), "capture_secs": get("capture_secs"),
        "save_capture_secs": summary.get("save_capture_secs"), "env": get("env"), "tl": get("tl"),
        "rss_cap_mb": caps.get("rss_mb"), "threads": caps.get("threads"),
        "timeout_s": caps.get("timeout_s"), "msg": (result.msg or get("msg") or "")[:200],
    }


def collect(spool_dir: Path) -> list[tuple[dict[str, Any], RunResult]]:
    """Read every finished request on a spool: (table row, result), oldest first."""

    spool_dir = Path(spool_dir)
    out = []
    for done in sorted((spool_dir / "done").glob("*.json"), key=lambda p: p.stat().st_mtime):
        result = read_json(RunResult, done)
        attempt = spool_dir / "attempts" / result.request
        request_path = attempt / "request.json"
        request = json.loads(request_path.read_text()) if request_path.is_file() else {}
        out.append((row_of(result, request, attempt), result))
    return out


def _june(manifests: Sequence[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in manifests:
        rows.update({r["stable_id"]: r for r in read_tsv(path)})
    return rows


def compare(new_rows: Sequence[Mapping[str, Any]], manifests: Sequence[Path], *,
            data: DataLayout, stand_ins: Mapping[str, str],
            reruns: Sequence[Mapping[str, Any]] = (),
            explained: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Join the frozen table, the June validator at the pin and the new path, one row per model.

    ``agree`` is ``yes`` when both paths completed with the same verdict (and, for passes, equal
    ``n_ops`` and graph hash), ``resource`` when either hit a resource limit, ``recipe`` when the
    June recipe was a stand-in, ``explained`` for a difference with a cause in ``explained``,
    ``no`` otherwise; ``no`` blocks acceptance. A fix-loop rerun of a row (``reruns``) replaces
    the first run's result, which stays in ``first_new``.
    """

    june = _june(manifests)
    trace = {r["stable_id"]: r for r in new_rows if r["mode"] == "trace"}
    rerun = {r["stable_id"]: r for r in reruns if r["mode"] == "trace"}
    shape = {r["stable_id"]: r for r in new_rows if r["mode"] == "trace-structure-only"}
    table = []
    for old in frozen(data):
        sid = old["stable_id"]
        base, new = june.get(sid, {}), rerun.get(sid, trace.get(sid, {}))
        row = {
            "first_new": trace.get(sid, {}).get("status") if sid in rerun else "",
            "stable_id": sid, "name": old["name"], "batch": old["batch"],
            "slice_a": old["status"], "slice_a_n_ops": old["n_ops"],
            "june_at_pin": base.get("status", "missing"), "june_n_ops": base.get("n_ops"),
            "new": new.get("status", "missing"), "new_n_ops": new.get("n_ops"),
            "new_class": new.get("class"), "runnable": new.get("runnable"),
            "structure_only": shape.get(sid, {}).get("status"),
            "june_peak_mb": base.get("peak_rss_mb"), "new_peak_mb": new.get("peak_rss_mb"),
            "june_s": base.get("elapsed"), "new_s": new.get("wall_s"),
            "capture_s": new.get("capture_secs"), "ready_capture_s": new.get("save_capture_secs"),
            "env": new.get("env"), "tl": new.get("tl"), "rss_cap_mb": new.get("rss_cap_mb"),
            "threads": new.get("threads"),
            "stand_in": stand_ins.get(sid, ""),
        }
        row["agree"], row["note"] = _agreement(row, base, new)
        if row["agree"] == "no" and explained and sid in explained:
            row["agree"], row["note"] = "explained", f"{row['note']}; {explained[sid]}"
        table.append(row)
    return table


def _agreement(row: Mapping[str, Any], base: Mapping[str, str],
               new: Mapping[str, Any]) -> tuple[str, str]:
    june_status, new_status = row["june_at_pin"], row["new"]
    if "missing" in (june_status, new_status) or new_status == "infra":
        return "incomplete", f"june {june_status}, new {new_status}"
    if row["stand_in"]:
        return "recipe", f"June recipe was the stand-in {row['stand_in']}; new path builds the real model"
    if june_status in RESOURCE or new_status in RESOURCE:
        return ("yes" if june_status == new_status else "resource"), (
            f"resource limit: june {june_status}, new {new_status}")
    if june_status != new_status:
        return "no", f"verdict differs: june {june_status}, new {new_status}"
    if new_status == "validated":
        ops_equal = str(base.get("n_ops")) == str(new.get("n_ops"))
        hash_equal = base.get("graph_shape_hash") == new.get("graph_hash")
        if not (ops_equal and hash_equal):
            return "no", f"validated in both; n_ops equal {ops_equal}, graph hash equal {hash_equal}"
    return "yes", ""


R0_COLUMNS = (
    "stable_id", "name", "batch", "slice_a", "slice_a_n_ops", "june_at_pin", "june_n_ops", "new",
    "first_new", "new_n_ops", "new_class", "agree", "note", "runnable", "structure_only", "stand_in",
    "june_peak_mb", "new_peak_mb", "june_s", "new_s", "capture_s", "ready_capture_s", "env", "tl",
    "rss_cap_mb", "threads",
)


PIN_COLUMNS = (
    "stable_id", "name", "batch", "r0", "r1", "first_r1", "r0_n_ops", "r1_n_ops", "r0_hash",
    "r1_hash", "r1_class", "r0_structure_only", "r1_structure_only", "agree", "note", "runnable",
    "r1_peak_mb", "r1_s", "capture_s", "ready_capture_s", "env", "tl", "rss_cap_mb", "threads",
)


def pincheck(r1_rows: Sequence[Mapping[str, Any]], r0_table: Sequence[Mapping[str, str]], *,
             r0_rows: Sequence[Mapping[str, Any]] = (),
             reruns: Sequence[Mapping[str, Any]] = (),
             explained: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """The r0 table's new-path verdicts against a later run of the same rows at a new pin.

    ``r1_rows`` are ``collect`` rows of the later run; ``r0_rows`` the r0 run's own rows (the
    first run, then its fix loop), which carry the graph hashes the r0 table leaves out. ``agree``
    is ``yes`` when the verdict and the structure-only verdict are equal and a pass keeps its
    ``n_ops`` and graph hash; ``recheck`` for a difference on a row r0 explained by its random
    weights, which counts only after a rerun on identical weights and threads (``reruns``, whose
    rows replace the first; the first stays in ``first_r1``); ``explained`` for a difference
    with a cause in ``explained``; ``no`` otherwise, which blocks.
    """

    def by_id(rows: Iterable[Mapping[str, Any]], mode: str) -> dict[str, Mapping[str, Any]]:
        return {r["stable_id"]: r for r in rows if r["mode"] == mode}

    first, rerun = by_id(r1_rows, "trace"), by_id(reruns, "trace")
    shape, old = by_id(r1_rows, "trace-structure-only"), by_id(r0_rows, "trace")
    table = []
    for r0 in r0_table:
        sid = r0["stable_id"]
        r1 = rerun.get(sid, first.get(sid, {}))
        row = {
            "stable_id": sid, "name": r0["name"], "batch": r0["batch"], "r0": r0["new"],
            "r1": r1.get("status", "missing"),
            "first_r1": first.get(sid, {}).get("status") if sid in rerun else "",
            "r0_n_ops": r0["new_n_ops"], "r1_n_ops": r1.get("n_ops"),
            "r0_hash": old.get(sid, {}).get("graph_hash"), "r1_hash": r1.get("graph_hash"),
            "r1_class": r1.get("class"), "r0_structure_only": r0["structure_only"],
            "r1_structure_only": shape.get(sid, {}).get("status", ""),
            "runnable": r1.get("runnable"), "r1_peak_mb": r1.get("peak_rss_mb"),
            "r1_s": r1.get("wall_s"), "capture_s": r1.get("capture_secs"),
            "ready_capture_s": r1.get("save_capture_secs"), "env": r1.get("env"),
            "tl": r1.get("tl"), "rss_cap_mb": r1.get("rss_cap_mb"), "threads": r1.get("threads"),
        }
        row["agree"], row["note"] = _pin_agreement(row, weights=r0["agree"] == "explained",
                                                   rerun=sid in rerun)
        if row["agree"] == "no" and explained and sid in explained:
            row["agree"], row["note"] = "explained", f"{row['note']}; {explained[sid]}"
        table.append(row)
    return table


def _pin_differences(row: Mapping[str, Any]) -> list[str]:
    differs = []
    if row["r0"] != row["r1"]:
        differs.append(f"verdict r0 {row['r0']}, r1 {row['r1']}")
    r0_shape, r1_shape = row["r0_structure_only"] or "-", row["r1_structure_only"] or "-"
    if r0_shape != r1_shape:
        differs.append(f"structure-only r0 {r0_shape}, r1 {r1_shape}")
    if row["r0"] == row["r1"] == "validated":
        if str(row["r0_n_ops"]) != str(row["r1_n_ops"]):
            differs.append(f"n_ops r0 {row['r0_n_ops']}, r1 {row['r1_n_ops']}")
        if row["r0_hash"] != row["r1_hash"]:
            differs.append("graph hash differs")
    return differs


def _pin_agreement(row: Mapping[str, Any], *, weights: bool, rerun: bool) -> tuple[str, str]:
    if row["r1"] in ("missing", "infra"):
        return "incomplete", f"r1 {row['r1']}"
    differs = _pin_differences(row)
    if not differs:
        return "yes", "same after a rerun on identical weights and threads" if rerun else ""
    if weights and not rerun:
        return "recheck", "; ".join(differs) + "; r0 traced this verdict to the random weights"
    return "no", "; ".join(differs)


CLASSIC = re.compile(r"menagerie\.classics\.(\w+)")


def classics_built(data: DataLayout) -> list[dict[str, str]]:
    """Every June catalog row whose recipe imports a ``menagerie.classics`` module.

    Some are faithful ports of models with no library code; the rest are stand-ins. Either way
    the row did not run the named architecture's own code, so none is carried forward under its
    name without a fidelity review.
    """

    from menagerie.identity import load_stable_ids

    ids = load_stable_ids(data.root / "stable_ids.jsonl")
    out = []
    with (data.root / "master_catalog.jsonl").open(encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            recipe = row.get("recipe")
            code = recipe.get("code", "") if isinstance(recipe, dict) else str(recipe or "")
            found = CLASSIC.search(code)
            if found:
                key = (row["name"], row["zoo"], row["variant"])
                out.append({"stable_id": ids.get(key, ""), "name": row["name"], "zoo": row["zoo"],
                            "variant": row["variant"], "module": found.group(1)})
    return out


def stand_ins(data: DataLayout) -> dict[str, str]:
    """Return ``{stable_id: classics module}`` for frozen rows June built from a classics module."""

    ids = {r["stable_id"] for r in frozen(data)}
    return {r["stable_id"]: r["module"] for r in classics_built(data) if r["stable_id"] in ids}


def ratios(on: Sequence[Mapping[str, Any]], off: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Paired ``intervention_ready`` on/off ratios per model: ready capture and attempt wall."""

    by_id = {r["stable_id"]: r for r in off}
    capture, wall, verdicts_differ = [], [], []
    for row in on:
        other = by_id.get(row["stable_id"])
        if not other:
            continue
        if row["status"] != other["status"]:
            verdicts_differ.append(row["stable_id"])
        if row.get("save_capture_secs") and other.get("save_capture_secs"):
            capture.append(float(row["save_capture_secs"]) / float(other["save_capture_secs"]))
        if row.get("wall_s") and other.get("wall_s"):
            wall.append(float(row["wall_s"]) / float(other["wall_s"]))
    def middle(values: list[float]) -> float | None:
        return round(statistics.median(values), 3) if values else None

    return {"pairs": len(capture), "capture_ratio_median": middle(capture),
            "capture_ratio_range": [round(min(capture), 3), round(max(capture), 3)] if capture else None,
            "wall_ratio_median": middle(wall), "verdicts_differ": verdicts_differ}


STORE_KINDS = {"tlspec": "application/x-tlspec", "tlspec-portable": "application/x-tlspec",
               "summary": "application/json"}


def store_put(spool_dir: Path, *, data: DataLayout) -> int:
    """Put every saved trace and summary of a drained spool in the store; return the count."""

    from menagerie.pipeline import StoreLayout, store

    spool_dir, layout, count = Path(spool_dir), StoreLayout.from_env(), 0
    for _, result in collect(spool_dir):
        for item in result.files:
            path = spool_dir / "attempts" / result.request / item.path
            if item.kind in STORE_KINDS and path.exists():
                store.put(path, store=layout, data=data, mime=STORE_KINDS[item.kind],
                          license_class="derived")
                count += 1
    return count


# --- sizing (fork 11): bytes and seconds per artifact class, and a projection -----------------

# The served artifact classes: the two saves, then each render kind.
ARTIFACTS = ("tlspec", "tlspec-portable", "svg", "pdf", "thumb", "tiles", "html", "netron",
             "model_explorer")
MONSTER_OPS = 2000  # the render lane's monster predicate: over 2,000 drawn nodes (or tiles drawn)


def _newest(attempts: Sequence[Path]) -> Path | None:
    return max(attempts, key=lambda p: p.name.split("-")[-2]) if attempts else None


def _outcome(attempt: Path) -> str:
    return json.loads((attempt / "result.json").read_text())["outcome"]


def _saved(attempt: Path) -> bool:
    files = json.loads((attempt / "result.json").read_text()).get("files", [])
    return any(f["kind"] == "tlspec" for f in files)


def _trace_attempts(attempts: Sequence[Path]) -> dict[str, Path]:
    """The newest trace attempt per model mode: a passing full one, else a full one that saved
    its trace (a replay failure still has saves and renders), else a passing structure-only one."""

    found: dict[str, Path] = {}
    for model_mode in ("eval", "train"):
        mine = [a for a in attempts if f"-{model_mode}-" in a.name]
        full = [a for a in mine if "-trace-structure-only-" not in a.name and "-trace-" in a.name]
        shape = [a for a in mine if "-trace-structure-only-" in a.name]
        chosen = (_newest([a for a in full if _outcome(a) == "pass"])
                  or _newest([a for a in full if _saved(a)])
                  or _newest([a for a in shape if _outcome(a) == "pass"]))
        if chosen is not None:
            found[model_mode] = chosen
    return found


def _render_log(attempt: Path) -> list[dict[str, Any]]:
    log = attempt / "render" / "render.log"
    lines = log.read_text().splitlines() if log.is_file() else []
    return [e for e in map(json.loads, lines) if e.get("sha256")]


def _render_secs(attempt: Path) -> float:
    rows = json.loads((attempt / "result.json").read_text()).get("rows", [])
    return sum(r.get("secs") or 0 for r in rows if r.get("stage") == "render")


def model_sizes(sid: str, attempts: Sequence[Path]) -> dict[str, Any] | None:
    """Bytes per artifact class and seconds for one model from its attempt directories.

    Each model mode's newest passing trace (full, else structure-only) gives the saves; its newest
    render (or the trace itself, when it rendered in process) gives the render set. ``None`` when
    no trace passed or saved.
    """

    traces = _trace_attempts(attempts)
    if "eval" not in traces:
        return None
    totals: dict[str, Any] = {**dict.fromkeys(ARTIFACTS, 0), "save_s": 0.0, "render_s": 0.0}
    variants: set[str] = set()
    rendered = False
    for model_mode, trace in traces.items():
        _add_saves(trace, totals)
        render = _newest([a for a in attempts if f"-render-{model_mode}-" in a.name]) or (
            trace if (trace / "render").is_dir() else None)
        if render is not None:
            rendered = True
            variants |= _add_render(render, totals)
    eval_row = next((r for r in json.loads((traces["eval"] / "result.json").read_text())
                     .get("rows", []) if r.get("stage") == "trace"), {})
    n_ops = eval_row.get("n_ops") or 0
    variants.discard("")
    return {
        "stable_id": sid, "n_ops": n_ops, "capture": eval_row.get("capture") or "full",
        "rendered": rendered, "variants": ";".join(sorted(variants)),
        "recurrent": any(v.startswith("rolled") for v in variants), "train": "train" in traces,
        "monster": n_ops > MONSTER_OPS or totals["tiles"] > 0,
        **{k: totals[k] for k in ARTIFACTS}, "total": sum(totals[k] for k in ARTIFACTS),
        "save_s": round(totals["save_s"], 3), "render_s": round(totals["render_s"], 3),
    }


def _add_saves(trace: Path, totals: dict[str, Any]) -> None:
    """Add a trace attempt's two saves and its save-capture seconds to ``totals``."""

    result = json.loads((trace / "result.json").read_text())
    for item in result.get("files", []):
        if item["kind"] in ("tlspec", "tlspec-portable"):
            totals[item["kind"]] += item.get("bytes") or 0
    summary_path = trace / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    totals["save_s"] += summary.get("save_capture_secs") or 0


def _add_render(render: Path, totals: dict[str, Any]) -> set[str]:
    """Add a render attempt's artifacts and seconds to ``totals``; return its variants."""

    entries = _render_log(render)
    for entry in entries:
        totals[entry["kind"]] += entry.get("bytes") or 0
    totals["render_s"] += _render_secs(render)
    return {entry.get("variant") or "" for entry in entries}


def sizes(roots: Sequence[Path]) -> list[dict[str, Any]]:
    """``model_sizes`` for every model with a passing trace under the attempt roots."""

    by_model: dict[str, list[Path]] = {}
    for root in roots:
        for attempt in Path(root).iterdir():
            if (attempt / "result.json").is_file():
                by_model.setdefault(attempt.name.split("-", 1)[0], []).append(attempt)
    found = (model_sizes(sid, attempts) for sid, attempts in sorted(by_model.items()))
    return [row for row in found if row is not None]


def _forced(row: Mapping[str, Any]) -> bool:
    return bool(row["monster"] or row["recurrent"] or row["train"]
                or row["capture"] == "structure_only")


def sizing_pick(rows: Sequence[Mapping[str, Any]], *, per: int = 10,
                bins: int = 10) -> list[Mapping[str, Any]]:
    """The sizing sample: ``per`` models from each op-count decile of ``rows``, always including
    every monster, recurrent, structure-only and train-variant row (which count toward their
    decile; a decile they overfill keeps them all). The rest are picked by the sha256 of the id."""

    import hashlib

    ranked = sorted(rows, key=lambda r: int(r["n_ops"] or 0))
    picked: list[Mapping[str, Any]] = []
    for b in range(bins):
        members = ranked[b * len(ranked) // bins:(b + 1) * len(ranked) // bins]
        forced = [r for r in members if _forced(r)]
        rest = sorted((r for r in members if not _forced(r)),
                      key=lambda r: hashlib.sha256(str(r["stable_id"]).encode()).hexdigest())
        picked += forced + rest[:max(0, per - len(forced))]
    return picked


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"median": 0, "p90": 0, "max": 0, "mean": 0}
    return {"median": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))],
            "max": ordered[-1], "mean": statistics.fmean(ordered)}


def project(sample: Sequence[Mapping[str, Any]], op_counts: Sequence[int], *,
            monster_share: float, models: int) -> dict[str, Any]:
    """Project the sample's bytes to ``models`` models.

    Non-monster models take the mean bytes of the sample's op-count decile that holds each of
    ``op_counts`` (the reference distribution; counts outside the sample's range take the nearest
    decile); monsters take the sample monsters' mean, at ``monster_share`` of the models.
    Returns bytes per model and in total, per artifact class and overall.
    """

    plain = sorted((r for r in sample if not r["monster"]), key=lambda r: int(r["n_ops"] or 0))
    monsters = [r for r in sample if r["monster"]]
    edges, deciles = _deciles(plain)
    weights = [0] * len(deciles)
    for n in op_counts:
        weights[sum(n >= e for e in edges)] += 1
    out: dict[str, Any] = {"edges": edges, "weights": weights, "monsters": len(monsters)}
    for kind in (*ARTIFACTS, "total"):
        means = _decile_means(deciles, kind)
        plain_mean = sum(w * m for w, m in zip(weights, means)) / max(1, sum(weights))
        monster_mean = statistics.fmean(r[kind] for r in monsters) if monsters else plain_mean
        per_model = (1 - monster_share) * plain_mean + monster_share * monster_mean
        out[kind] = {"per_model": per_model, "total": per_model * models}
    return out


def _deciles(ranked: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[list[Mapping[str, Any]]]]:
    """The nine op-count edges of a sorted sample and its rows per decile."""

    edges = [int(ranked[b * len(ranked) // 10]["n_ops"] or 0) for b in range(1, 10)]
    deciles: list[list[Mapping[str, Any]]] = [[] for _ in range(10)]
    for row in ranked:
        deciles[sum(int(row["n_ops"] or 0) >= e for e in edges)].append(row)
    return edges, deciles


def _decile_means(deciles: Sequence[Sequence[Mapping[str, Any]]], kind: str) -> list[float]:
    """Mean bytes of ``kind`` per decile; an empty decile (tied op counts) takes its nearest
    filled neighbour's mean."""

    filled = [b for b, rows in enumerate(deciles) if rows]
    return [statistics.fmean(r[kind] for r in deciles[min(filled, key=lambda f: abs(f - b))])
            for b in range(len(deciles))]


SIZE_COLUMNS = ("stable_id", "n_ops", "capture", "rendered", "variants", "recurrent", "train",
                "monster", *ARTIFACTS, "total", "save_s", "render_s")


def sizing(rows: Sequence[Mapping[str, Any]], op_counts: Sequence[int], *, models: int,
           monster_share: float, per: int = 10) -> dict[str, Any]:
    """Fork 11's numbers from a ``sizes`` table: the sample, per-class quantiles and seconds over
    it, and the projection to ``models`` models over the ``op_counts`` distribution."""

    def typed(row: Mapping[str, Any]) -> dict[str, Any]:
        flags = ("rendered", "recurrent", "train", "monster")
        return {**row, **{k: str(row[k]) == "True" for k in flags},
                **{k: float(row[k] or 0) for k in (*ARTIFACTS, "total", "save_s", "render_s")}}

    sample = sizing_pick([typed(r) for r in rows], per=per)
    return {
        "sample": [r["stable_id"] for r in sample], "n": len(sample),
        "unrendered": [r["stable_id"] for r in sample if not r["rendered"]],
        "quantiles": {k: _quantiles([r[k] for r in sample if r[k]])
                      for k in (*ARTIFACTS, "total", "save_s", "render_s")},
        "projection": project(sample, op_counts, monster_share=monster_share, models=models),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m menagerie.pipeline.regression")
    sub = parser.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("spool", help="queue a batch of the frozen table on a spool")
    p.add_argument("spool", type=Path)
    p.add_argument("batch", choices=sorted(BATCHES))
    p.add_argument("--cpus", type=int, required=True)
    p.add_argument("--ids", nargs="*", default=())
    p.add_argument("--mode", default="trace")
    p.add_argument("--no-intervention-ready", action="store_true")
    p = sub.add_parser("sample", help="queue the stratified on/off ablation sample as pairs")
    p.add_argument("spool", type=Path)
    p.add_argument("--cpus", type=int, required=True)
    p.add_argument("--memory-gib", type=int, required=True)
    p.add_argument("--size", type=int, default=50)
    p = sub.add_parser("collect", help="a drained spool as a table (and ledger rows)")
    p.add_argument("spool", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--rows", type=Path, help="write the ledger rows here as JSON lines")
    p = sub.add_parser("store-put", help="put a drained spool's traces and summaries in the store")
    p.add_argument("spool", type=Path)
    p = sub.add_parser("compare", help="the r0 table from the new rows and June manifests")
    p.add_argument("new", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("manifests", type=Path, nargs="+")
    p.add_argument("--rerun", type=Path, action="append", default=[],
                   help="a fix-loop table whose rows replace the first run's")
    p.add_argument("--explain", type=Path, help="TSV of stable_id and cause for known differences")
    p = sub.add_parser("pincheck", help="a later run of the Slice A 200 against the r0 table")
    p.add_argument("new", type=Path, help="the later run's collect table")
    p.add_argument("out", type=Path)
    p.add_argument("--r0", type=Path, required=True, help="the r0 table (slice_a_200_r0.tsv)")
    p.add_argument("--r0-rows", type=Path, action="append", default=[],
                   help="the r0 run's collect tables, first run then fix loop (graph hashes)")
    p.add_argument("--rerun", type=Path, action="append", default=[],
                   help="a rerun table whose rows replace the later run's")
    p.add_argument("--explain", type=Path, help="TSV of stable_id and cause for known differences")
    p = sub.add_parser("sizes", help="bytes and seconds per artifact class from attempt roots")
    p.add_argument("out", type=Path)
    p.add_argument("roots", type=Path, nargs="+")
    p = sub.add_parser("sizing", help="fork 11: the sizing sample, quantiles and projection")
    p.add_argument("sizes", type=Path)
    p.add_argument("--routing", type=Path, help="op-count distribution (default: the routing "
                   "manifest's validated rows)")
    p.add_argument("--models", type=int, default=28000)
    p.add_argument("--monster-share", type=float, default=242 / 14291,
                   help="share of monsters (default: the live site's 242 of 14,291)")
    p = sub.add_parser("ratios", help="paired intervention_ready on/off ratios")
    p.add_argument("on", type=Path)
    p.add_argument("off", type=Path)
    return parser


def _spool_verb(args: argparse.Namespace, data: DataLayout) -> dict[str, Any]:
    queued = spool(args.spool, args.batch, data=data, cpus=args.cpus, ids=args.ids,
                   intervention_ready=not args.no_intervention_ready, mode=args.mode)
    return {"queued": len(queued)}


def _sample_verb(args: argparse.Namespace, data: DataLayout) -> dict[str, Any]:
    ids = sample(data, size=args.size)
    caps = Caps(rss_mb=args.memory_gib * 1024, timeout_s=TIMEOUT_S, threads=args.cpus)
    return {"ids": ids, "queued": spool_pairs(args.spool, ids, data=data, caps=caps)}


def _collect_verb(args: argparse.Namespace, data: DataLayout) -> dict[str, Any]:
    from menagerie.pipeline.records import dumps

    pairs = collect(args.spool)
    names = {r["stable_id"]: r["name"] for r in frozen(data)}
    write_tsv(({**row, "name": names.get(row["stable_id"], "")} for row, _ in pairs), args.out,
              COLUMNS)
    if args.rows:
        lines = [dumps(r) for _, result in pairs for r in result.rows]
        args.rows.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return {"rows": len(pairs)}


def _compare_verb(args: argparse.Namespace, data: DataLayout) -> dict[str, Any]:
    from collections import Counter

    reruns = [row for path in args.rerun for row in read_tsv(path)]
    explained = {r["stable_id"]: r["cause"] for r in read_tsv(args.explain)} if args.explain else {}
    table = compare(read_tsv(args.new), args.manifests, data=data, stand_ins=stand_ins(data),
                    reruns=reruns, explained=explained)
    write_tsv(table, args.out, R0_COLUMNS)
    return dict(Counter(row["agree"] for row in table))


def _pincheck_verb(args: argparse.Namespace, data: DataLayout) -> dict[str, Any]:
    from collections import Counter

    def rows(paths: Sequence[Path]) -> list[dict[str, str]]:
        return [row for path in paths for row in read_tsv(path)]

    explained = {r["stable_id"]: r["cause"] for r in read_tsv(args.explain)} if args.explain else {}
    table = pincheck(read_tsv(args.new), read_tsv(args.r0), r0_rows=rows(args.r0_rows),
                     reruns=rows(args.rerun), explained=explained)
    write_tsv(table, args.out, PIN_COLUMNS)
    return dict(Counter(row["agree"] for row in table))


def _write_sizes(out: Path, roots: Sequence[Path]) -> int:
    rows = sizes(roots)
    write_tsv(rows, out, SIZE_COLUMNS)
    return len(rows)


def _sizing_verb(args: argparse.Namespace, data: DataLayout) -> dict[str, Any]:
    routing = read_tsv(args.routing or data.root / "routing_manifest.tsv")
    ops = [int(r["n_ops"]) for r in routing
           if r["status"] == "validated" and (r["n_ops"] or "0").isdigit() and int(r["n_ops"]) > 0]
    return sizing(read_tsv(args.sizes), ops, models=args.models, monster_share=args.monster_share)


VERBS = {
    "spool": _spool_verb, "sample": _sample_verb, "collect": _collect_verb,
    "store-put": lambda a, d: {"put": store_put(a.spool, data=d)},
    "compare": _compare_verb, "pincheck": _pincheck_verb, "sizing": _sizing_verb,
    "sizes": lambda a, d: {"models": _write_sizes(a.out, a.roots)},
    "ratios": lambda a, d: ratios(read_tsv(a.on), read_tsv(a.off)),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry; prints a JSON summary."""

    args = _parser().parse_args(argv)
    print(json.dumps(VERBS[args.verb](args, DataLayout.from_env())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
