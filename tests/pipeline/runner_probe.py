"""A stand-in worker for the runner tests: real attempts in the real sandbox, without TorchLens.

The runner tests point ``runner.WORKER`` here, so the runner starts ``python runner_probe.py
ATTEMPT_DIR`` inside bubblewrap in the built environment. Every name in ``request.extras`` is
imported first. Then the recipe's ``call`` picks the attempt: a library constructor runs a real
eval forward and writes a passing result; ``probe.oversize``, ``probe.hang`` and ``probe.die``
force the runner's memory cap, timeout and missing-result paths; ``probe.isolation`` passes with
``msg`` listing whatever the sandbox failed to hide or protect ("isolated" when nothing).
"""

from __future__ import annotations

import importlib
import os
import socket
import sys
import time
from pathlib import Path

from menagerie.pipeline import now, read_json, write_json
from menagerie.pipeline.records import LedgerRow, RunRequest, RunResult

HOARD_CHUNK = 64 << 20
HOARD_MAX_CHUNKS = 64  # 4 GiB at most, so a broken cap cannot hurt the host


def _writable(path: Path) -> bool:
    try:
        path.write_text("probe")
    except OSError:
        return False
    path.unlink()
    return True


def _isolation(request: RunRequest) -> str:
    problems = []
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        problems.append("network reachable")
    except OSError:
        pass
    for label, path in (("environment", Path(sys.prefix)), ("code", Path(os.environ["PYTHONPATH"]))):
        if _writable(path / ".probe-write"):
            problems.append(f"{label} writable")
    home = Path(request.recipe.kwargs["home"])
    problems += [f"{name} visible" for name in (".ssh", ".config", "data") if (home / name).exists()]
    if os.environ.get("OMP_NUM_THREADS") != str(request.caps.threads):
        problems.append(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    return "; ".join(problems) or "isolated"


def _forward(request: RunRequest) -> RunResult:
    import torch

    torch.set_num_threads(request.caps.threads)
    module, name = request.recipe.call.rsplit(".", 1)
    model = getattr(importlib.import_module(module), name)(
        *request.recipe.args, **request.recipe.kwargs).eval()
    started, clock = now(), time.perf_counter()
    with torch.no_grad():
        out = model(torch.randn(*request.recipe.input.shape))
    secs = round(time.perf_counter() - clock, 3)
    params = sum(p.numel() for p in model.parameters())
    row = LedgerRow(id=request.id, stage="forward", at=now(), host=os.environ["MENAGERIE_HOST"],
                    env=request.recipe.env, recipe_rev=request.recipe.rev, outcome="pass",
                    mode="eval", backend="torch", fw=f"torch {torch.__version__}",
                    params=params, params_source="counted", forward_secs=secs)
    return RunResult(request=request.request, id=request.id, mode=request.mode, outcome="pass",
                     started=started, ended=now(), host=row.host,
                     output_shapes=(tuple(out.shape),), params=params, rows=(row,))


def main(attempt_dir: Path) -> int:
    request = read_json(RunRequest, attempt_dir / "request.json")
    for extra in request.extras:
        importlib.import_module(extra)
    call = request.recipe.call
    if call == "probe.oversize":
        hoard = []
        for _ in range(HOARD_MAX_CHUNKS):
            hoard.append(bytearray(b"\x01") * HOARD_CHUNK)
            time.sleep(0.05)
        return 0
    if call == "probe.hang":
        time.sleep(3600)
    if call == "probe.die":
        os._exit(3)
    if call == "probe.isolation":
        result = RunResult(request=request.request, id=request.id, mode=request.mode,
                           outcome="pass", started=now(), ended=now(),
                           host=os.environ["MENAGERIE_HOST"], msg=_isolation(request))
    else:
        result = _forward(request)
    write_json(result, attempt_dir / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
