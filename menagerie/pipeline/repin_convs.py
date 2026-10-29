"""Confirm the thirteen repaired token-view convolutions through the sandbox at three threads."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, repin, repin_slice
from menagerie.pipeline.records import RecordError

VERSION = "2.36.0"
IDS = ("m3798", "m3799", "m3796", "m3800", "m3797", "m3801", "m3803",
       "m3802", "m4094", "m4101", "m4100", "m4099", "m4097")
EXPLANATION = (
    "TorchLens 2.36.0 preserves token-view input strides (upstream 2e368833a); "
    "the original shallow-conv replay failure is fixed, confirmed in the sandbox at three threads. "
    "No tolerance or depth-floor change."
)


def confirm(*, data: DataLayout, out: Path, deadline: float | None = None) -> dict[str, Any]:
    """Keep original per-row memory caps while independently confirming three-thread replay."""

    import torchlens as tl

    if tl.__version__ != VERSION:
        raise RecordError(f"conv confirmation requires {VERSION}, installed {tl.__version__}")
    baseline = repin_slice._selection(data, IDS, False)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="repin-convs-") as temporary:
        spool = Path(temporary)
        try:
            _run(baseline, data, spool, deadline)
        except TimeoutError:
            if deadline is None or time.time() < deadline - repin_slice.RESERVE_S - 1:
                raise
        rows = [row for row, _ in repin_slice.regression.collect(spool)]
        repin_slice._evidence(spool, out)
    return _receipt(rows, out, data)


def _run(baseline: list[dict[str, str]], data: DataLayout, spool: Path,
         deadline: float | None) -> None:
    for batch, (width, _) in repin_slice.regression.BATCHES.items():
        ids = [row["stable_id"] for row in baseline if row["batch"] == batch]
        for offset in range(0, len(ids), width):
            if deadline is not None and time.time() >= deadline - repin_slice.RESERVE_S:
                return
            repin_slice._wave(ids[offset:offset + width], batch, "trace", data=data,
                             spool=spool, smoke=False, deadline=deadline, threads=3)


def _receipt(rows: list[dict[str, Any]], out: Path, data: DataLayout) -> dict[str, Any]:
    from menagerie.pipeline import regression

    passed = ({row["stable_id"] for row in rows} == set(IDS) and len(rows) == len(IDS)
              and all(row["status"] == "validated" and row["tl"] == VERSION
                      and row["threads"] == 3 for row in rows))
    report = {"schema": "repin-conv3/1", "version": VERSION,
              "source_digest": repin.source_digest(), "rows": len(rows),
              "repo_data": repin.repo_data(data),
              "job": Path.cwd().name, "results": rows,
              "threads": 3, "sandbox": True, "passed": passed}
    regression.write_tsv(rows, out / "replay13-3threads.tsv", regression.COLUMNS)
    (out / "repin-conv3.json").write_text(json.dumps(report, indent=2) + "\n")
    notes = {sid: EXPLANATION for sid in IDS} if passed else {}
    (out / "confirmed-conv-explanations.json").write_text(json.dumps(
        {"version": VERSION, "source_digest": report["source_digest"],
         "explanations": notes}, indent=2) + "\n")
    retypes = [{"stable_id": sid, "gap_class": "fixed_by_torchlens_2.36.0",
                "version": VERSION, "job": report["job"], "result": "validated",
                "released_status": "validated", "branch_status": "passes_sandbox_3threads",
                "proof_sketch": EXPLANATION} for sid in IDS] if passed else []
    regression.write_tsv(retypes, out / "gap-retypes.tsv",
                         ("stable_id", "version", "job", "result", "gap_class",
                          "released_status", "branch_status", "proof_sketch"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float)
    arguments = parser.parse_args()
    print(json.dumps(confirm(data=DataLayout.from_env(), out=arguments.out,
                             deadline=arguments.deadline)))
