"""Which TorchLens a runner job or cluster batch runs, read on the hub (decisions.md 2026-11-06, 3.0.2 pin).

A runner job runs the commit ``run-on`` shipped when it started (``checkout_sha`` in the hub's job
mirror ``~/.local/state/fleet/jobs/<job>/manifest.json``) and builds its environments from that
commit's locks, so after the pin moves on ``main`` a runner started earlier still traces at the old
TorchLens until its next start. The tick holds stage-3 queueing while a running runner is behind
the pin (``tick._step_stage3_queue``), so no job meant for the new pin is traced at the old one,
and a cluster batch's renders run at the commit that traced it when that commit pins another
TorchLens (``render_ref``), so a save is never drawn by a TorchLens that cannot read it.
"""

from __future__ import annotations

import functools
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
JOBS = Path.home() / ".local" / "state" / "fleet" / "jobs"
PIN = re.compile(r'"torchlens==([^"]+)"')
RUNNING = "RUNNING"


@functools.lru_cache(maxsize=64)
def pin_at(commit: str, repo: Path = REPO) -> str | None:
    """The TorchLens pin in ``pyproject.toml`` at ``commit``, or ``None`` when git cannot say."""

    done = subprocess.run(["git", "show", f"{commit}:pyproject.toml"], cwd=repo,
                          capture_output=True, text=True, timeout=30, check=False)
    found = PIN.search(done.stdout) if done.returncode == 0 else None
    return found.group(1) if found else None


def job_pin(job: str, *, jobs: Path = JOBS, repo: Path = REPO) -> str | None:
    """The pin of the commit a run-on job shipped, or ``None`` when its mirror does not say."""

    try:
        manifest = json.loads((jobs / job / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    commit = manifest.get("checkout_sha") if isinstance(manifest, Mapping) else None
    return pin_at(commit, repo) if isinstance(commit, str) and commit else None


def behind(runners: Sequence[Mapping[str, Any]], pin: str, *, names: Sequence[str] = ("B", "G"),
           jobs: Path = JOBS, repo: Path = REPO) -> dict[str, str]:
    """Running runners among ``names`` whose job pins another TorchLens: ``{name: its pin}``.

    ``runners`` is the tick pass's ``runners`` result. A job whose pin cannot be read counts as
    current, so a missing mirror never stops the pipeline."""

    out = {}
    for row in runners:
        name, job = row.get("runner"), row.get("job")
        if name not in names or not job or row.get("job_state") != RUNNING:
            continue
        found = job_pin(str(job), jobs=jobs, repo=repo)
        if found is not None and found != pin:
            out[str(name)] = found
    return out


def render_ref(meta: Mapping[str, Any], pin: str, repo: Path = REPO) -> str:
    """``main``, unless the batch's commit pins another TorchLens than ``pin``: then that commit,
    so its saves are drawn by the TorchLens that wrote them."""

    commit = meta.get("commit")
    if not isinstance(commit, str) or not commit:
        return "main"
    found = pin_at(commit, repo)
    return commit if found is not None and found != pin else "main"
