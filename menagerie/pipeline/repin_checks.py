"""Daytime private-call, hash and stale-reference probes for a released TorchLens pin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, now, read_json

GELU_IDS = ("m3638", "m10919", "m4246")
CHILD_TIMEOUT_S = 1800


def private_calls() -> dict[str, Any]:
    """Exercise the validator and observer, diagnostics and legacy wrapper path."""

    import torch
    from torchlens.backends.torch.wrappers import wrap_torch
    from torchlens.user_funcs import _validate_forward_pass_torch
    from torchlens.validation.diagnostics import get_validation_failure

    states = []

    def observe(trace: Any) -> None:
        # Cleanup resets the trace's replay status after this callback returns.
        states.append(getattr(getattr(trace, "validation_replay_status", None), "state", None))

    old_threads = torch.get_num_threads()
    try:
        torch.manual_seed(0)
        passed = _validate_forward_pass_torch(
            torch.nn.Linear(3, 2).eval(), [torch.ones(1, 3)], {}, random_seed=0,
            validate_metadata=True, num_threads=1, _trace_observer=observe)
        diagnostic = get_validation_failure(None)
    finally:
        torch.set_num_threads(old_threads)
    return {"passed": bool(passed) and states == ["passed"] and callable(wrap_torch),
            "validator": bool(passed), "observer_states": states,
            "diagnostic": None if diagnostic is None else str(diagnostic)[:300],
            "legacy_wrapper_available": callable(wrap_torch)}


def gelu_probe(sid: str, data: DataLayout) -> dict[str, Any]:
    """Two pristine builds in one process, before pre-wrapping torch, at the real recipe."""

    import gc

    import torch
    import torchlens as tl

    from menagerie import trace_summary
    from menagerie.pipeline import worker
    from menagerie.pipeline.records import ModelRecord

    torch.set_num_threads(max(1, int(os.environ.get("MENAGERIE_CPUS", "1"))))
    recipe = read_json(ModelRecord, data.model(sid)).recipe
    if recipe is None:
        raise ValueError(f"{sid} has no recipe")
    captures = []
    for _ in range(2):
        # depth_pro's published recipe exceeded the full-capture cap; its prior
        # regression was structure-only, so keep that contract for this case.
        subject = worker._build(recipe, device="meta" if sid == "m4246" else "cpu")
        trace = worker._hash_capture(subject, structure_only=sid == "m4246")
        captures.append({"hashes": trace_summary.hash_family(trace),
                         "n_ops": trace.num_ops,
                         "gelu": sum(op.layer_type == "gelu" for op in trace.layer_list),
                         "internalsource": sum(op.layer_type == "internalsource"
                                               for op in trace.layer_list)})
        del trace, subject
        gc.collect()
    return {"id": sid, "torchlens": tl.__version__, "captures": captures,
            "passed": captures[0] == captures[1] and captures[0]["gelu"] > 0}


def _gelu_child(sid: str, data: DataLayout) -> dict[str, Any]:
    command = [sys.executable, "-m", "menagerie.pipeline.repin_checks", sid]
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=False,
                              timeout=CHILD_TIMEOUT_S,
                              env={**os.environ, "MENAGERIE_DATA": str(data.root)})
    except subprocess.TimeoutExpired:
        return {"id": sid, "passed": False, "error": "GELU child timed out"}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode or not lines:
        return {"id": sid, "passed": False, "error": done.stderr[-1200:]}
    return json.loads(lines[-1])


def day(*, data: DataLayout, out: Path, ids: Sequence[str] = ()) -> dict[str, Any]:
    """The original hash harness plus GELU fresh-process and private-call checks."""

    import torchlens as tl

    from menagerie.pipeline import hash_harness

    out.mkdir(parents=True, exist_ok=True)
    private = private_calls()
    (out / "private.json").write_text(json.dumps(private, indent=2) + "\n")
    hashes = hash_harness.harness(ids, data=data, out=out / "hash")
    gelu = []
    for sid in GELU_IDS:
        first, fresh = _gelu_child(sid, data), _gelu_child(sid, data)
        gelu.append({"id": sid, "first": first, "fresh": fresh,
                     "passed": first.get("passed", False) and fresh.get("passed", False)
                     and first["captures"] == fresh["captures"]})
    report = {"schema": "repin-day/1", "at": now(), "version": tl.__version__,
              "passed": private["passed"] and hashes["passed"]
              and all(r["passed"] for r in gelu), "private": private, "gelu": gelu,
              "hash_passed": hashes["passed"], "workaround": False}
    (out / "day.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(gelu_probe(sys.argv[1], DataLayout.from_env())))
