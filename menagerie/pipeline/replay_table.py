"""Replay failure diagnostics from the ledger and saved, possibly packed op metadata.

The baseline freezes the replay-gap cohort. Later pins report every baseline id,
including passes and unverified results, and refuse an incomplete re-trace queue.
No validation runs here and no validation verdict is changed.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout, ledger, read_json, roster
from menagerie.pipeline import store as blobs
from menagerie.pipeline.records import LedgerRow, ModelRecord

BASELINE_PIN = "2.35.3"
MAX_METADATA_BYTES = 256 * 1024 * 1024
PRESCAN_BYTES = 8 * 1024 * 1024
OP = re.compile(r"mismatch at (\S+) \(([^)]+)\)")
DIFF = re.compile(r"\b(max_abs_diff|max_rel_diff|reduction_depth)=([\d.eE+\-]+)")
PERTURBED = re.compile(r"perturbing parent\(s\) (\[[^\]]*\])")
COLUMNS = ("id", "name", "family", "pin", "at", "outcome", "class", "validation_state",
           "trace", "failure_kind", "failing_op", "op_type", "shape", "parents",
           "perturbed_parents", "max_abs_diff", "max_rel_diff", "reduction_depth",
           "thread_mode", "metadata_status", "metadata_error", "msg")


def _latest(rows: Sequence[Any], pin: str | None = None) -> dict[str, LedgerRow]:
    found: dict[str, LedgerRow] = {}
    for row in rows:
        if not isinstance(row, LedgerRow) or row.stage != "trace" or row.outcome == "infra":
            continue
        if row.capture not in (None, "full") or row.mode not in (None, "eval"):
            continue
        if pin is not None and row.tl != pin:
            continue
        if row.id not in found or row.at >= found[row.id].at:
            found[row.id] = row
    return found


def _cohort(rows: Sequence[Any], pin: str, out: Path) -> list[str]:
    baseline = out / f"replay-ops-{BASELINE_PIN}.tsv"
    if pin == BASELINE_PIN and not baseline.exists():
        return sorted(sid for sid, row in _latest(rows).items()
                      if row.tl == pin and row.cls == "torchlens_gap:replay")
    with baseline.open(encoding="utf-8", newline="") as handle:
        ids = [row["id"] for row in csv.DictReader(handle, delimiter="\t")]
    if len(ids) != len(set(ids)):
        raise ValueError("baseline table contains duplicate ids")
    return sorted(ids)


def _field(op: Any, name: str, default: Any = None) -> Any:
    return op.get(name, default) if isinstance(op, Mapping) else getattr(op, name, default)


def _op_index(state: Mapping[str, Any]) -> dict[str, Any]:
    ops = state.get("layer_list")
    if not isinstance(ops, list):
        raise TypeError("saved metadata has no layer_list")
    indexed: dict[str, Any] = {}
    for op in ops:
        label = _field(op, "label")
        if not isinstance(label, str) or not label:
            raise ValueError("saved op has no label")
        if label in indexed:
            raise ValueError(f"duplicate saved op label {label}")
        indexed[label] = op
    return indexed


def _metadata(path: Path) -> dict[str, Any]:
    # Slim saves omit tensor blobs. Reuse TorchLens's default-deny metadata reader,
    # including its rename map; never use an unrestricted pickle loader.
    from torchlens._io.bundle import _prescan_metadata_pickle_opcodes, _RenameAwareUnpickler

    metadata = path / "metadata.pkl"
    if metadata.is_symlink() or path.is_symlink():
        raise ValueError("symlinked trace metadata refused")
    with os.fdopen(os.open(metadata, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > MAX_METADATA_BYTES:
            raise ValueError("trace metadata exceeds the report reader's byte limit")
        if size > PRESCAN_BYTES:
            _prescan_metadata_pickle_opcodes(handle, metadata)
        state = _RenameAwareUnpickler(handle).load()
    if not isinstance(state, Mapping):
        raise TypeError("trace metadata is not a mapping")
    return _op_index(state)


def _resolve(index: Mapping[str, Any], label: str) -> Any:
    if label in index:
        return index[label]
    qualified = f"{label}:1"
    if qualified in index:
        return index[qualified]
    if label.endswith(":1") and label[:-2] in index:
        return index[label[:-2]]
    raise ValueError(f"op {label} absent from saved metadata")


def _description(op: Any, label: str) -> dict[str, Any]:
    kind = _field(op, "func_name") or _field(op, "type")
    shape = _field(op, "shape")
    if not isinstance(kind, str) or not isinstance(shape, (list, tuple)):
        raise TypeError(f"op {label} lacks its type or shape")
    return {"label": label, "op_type": kind, "shape": list(shape)}


def _join(row: LedgerRow, label: str, *, data: DataLayout,
          store: StoreLayout) -> dict[str, str]:
    if row.trace is None:
        raise ValueError("ledger row has no saved trace")
    index = blobs.read_blob(row.trace, _metadata, store=store)
    op = _resolve(index, label)
    description = _description(op, label)
    parent_labels = _field(op, "parents")
    if not isinstance(parent_labels, (list, tuple)):
        raise TypeError(f"op {label} lacks its parent list")
    parents = [_description(_resolve(index, parent), parent) for parent in parent_labels]
    return {"op_type": description["op_type"], "shape": json.dumps(description["shape"]),
            "parents": json.dumps(parents, separators=(",", ":")), "metadata_status": "ok"}


def _diagnostic(row: LedgerRow, *, data: DataLayout, store: StoreLayout) -> dict[str, str]:
    msg = row.msg or ""
    match = OP.search(msg)
    result = {"msg": msg, "thread_mode": "single-thread retry" if
              "single-thread retry" in msg else "not recorded", "metadata_status": "not applicable"}
    if row.cls != "torchlens_gap:replay":
        return result
    result["failure_kind"] = "insensitive" if "output insensitive" in msg else "mismatch"
    result.update(dict(DIFF.findall(msg)))
    if match is None:
        return {**result, "metadata_status": "unavailable", "metadata_error": "no failing op in ledger message"}
    result.update(failing_op=match[1], op_type=match[2])
    try:
        perturbed = PERTURBED.search(msg)
        if perturbed:
            result["perturbed_parents"] = json.dumps(ast.literal_eval(perturbed[1]))
        result.update(_join(row, match[1], data=data, store=store))
    except Exception as error:  # noqa: BLE001 - record each blob refusal without dropping its id
        # Each read failure is a diagnostic column, not a dropped id or a new verdict.
        detail = re.sub(r"/(?:home|Users)/[^/\s]+/[^\s]*", "<store path>", str(error))
        result.update(metadata_status="unavailable", metadata_error=f"{type(error).__name__}: {detail[:300]}")
    return result


def _identity(data: DataLayout) -> dict[str, dict[str, str]]:
    found = {row.stable_id: {"name": row.name, "family": row.family or ""}
             for row in roster.read(data=data) if row.stable_id}
    for path in sorted((data.root / "models").glob("*/*.json")):
        record = read_json(ModelRecord, path)
        found[record.id] = {"name": record.name, "family": record.family}
    return found


def _record(row: LedgerRow, identity: Mapping[str, str], *, data: DataLayout,
            store: StoreLayout) -> dict[str, str]:
    outcome = "unverified" if row.validation_state == "unverified" else row.outcome
    return {"id": row.id, **identity, "pin": row.tl or "", "at": row.at, "outcome": outcome,
            "class": row.cls or "", "validation_state": row.validation_state or "",
            "trace": row.trace or "", **_diagnostic(row, data=data, store=store)}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _tsv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _excluded(sid: str, reason: str, identity: Mapping[str, str], pin: str) -> dict[str, str]:
    return {"id": sid, **identity, "pin": pin, "outcome": "excluded", "msg": reason}


def _check_exclude(exclude: Mapping[str, str], ids: Sequence[str]) -> None:
    outside = sorted(set(exclude) - set(ids))
    if outside:
        raise ValueError(f"excluded ids are not in the frozen cohort: {', '.join(outside)}")
    blank = sorted(sid for sid, reason in exclude.items() if not reason.strip())
    if blank:
        raise ValueError(f"exclusions need a stated reason: {', '.join(blank)}")


def table(*, data: DataLayout, store: StoreLayout, pin: str | None = None,
          out: Path | None = None, exclude: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Write one row per baseline id at this pin, plus by-op and outcome counts.

    Build the 2.35.3 baseline first. Use the same output directory for a later
    pin so its frozen cohort includes repaired and unverified models as well.
    A pin with missing re-traces refuses before writing either output, except
    ids in ``exclude`` (id to stated reason): one with no re-trace becomes an
    ``excluded`` row carrying its reason; one that has a re-trace reports it.
    """
    from menagerie.pipeline.publish import torchlens_pin

    pin = torchlens_pin() if pin is None else pin
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9.+-]*)", pin):
        raise ValueError("pin must be a version without path separators")
    out = Path(out) if out is not None else Path(__file__).resolve().parents[2] / "outputs" / "replay"
    rows = ledger.read_all(data=data)
    ids = _cohort(rows, pin, out)
    exclude = dict(exclude or {})
    _check_exclude(exclude, ids)
    current = _latest(rows, pin)
    excluded = {sid: exclude[sid] for sid in ids if sid in exclude and sid not in current}
    missing = sorted(set(ids) - current.keys() - excluded.keys())
    if missing:
        raise ValueError(f"ids not re-traced at {pin}: {', '.join(missing)}")
    identities = _identity(data)
    records = [_excluded(sid, excluded[sid], identities.get(sid, {}), pin) if sid in excluded
               else _record(current[sid], identities.get(sid, {}), data=data, store=store)
               for sid in ids]
    counts = Counter(row.get("op_type", "unknown") for row in records if row.get("failing_op"))
    path = out / f"replay-ops-{pin}.tsv"
    _write(path, _tsv(records, COLUMNS))
    summary = out / f"replay-summary-{pin}.json"
    result = {"table": str(path), "summary": str(summary), "pin": pin, "ids": ids,
              "count": len(ids), "excluded": excluded, "by_op_type": dict(sorted(counts.items())),
              "by_outcome": dict(Counter(row["outcome"] for row in records)),
              "metadata_errors": sum(row.get("metadata_status") == "unavailable" for row in records)}
    _write(summary, json.dumps(result, indent=2) + "\n")
    return result
