"""Runner jobs and cluster batches behind the TorchLens pin (``runner_pin``) and the tick's hold."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from menagerie.pipeline import runner_pin, tick


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@e", "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*git, "init", "-q"], check=True)
    shas = []
    for version in ("2.36.0", "3.0.2"):
        (repo / "pyproject.toml").write_text(f'dependencies = [\n    "torchlens=={version}",\n]\n')
        subprocess.run([*git, "add", "pyproject.toml"], check=True)
        subprocess.run([*git, "commit", "-q", "-m", version], check=True)
        shas.append(subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True,
                                   text=True).stdout.strip())
    return repo, shas[0], shas[1]


def _job(jobs: Path, name: str, sha: str) -> None:
    (jobs / name).mkdir(parents=True)
    (jobs / name / "manifest.json").write_text(json.dumps({"checkout_sha": sha}))


def test_behind_names_running_runners_on_another_pin(tmp_path: Path) -> None:
    repo, old, new = _repo(tmp_path)
    jobs = tmp_path / "jobs"
    _job(jobs, "jb", old)
    _job(jobs, "jg", new)
    _job(jobs, "ja", old)
    runners = [{"runner": "B", "job": "jb", "job_state": "RUNNING"},
               {"runner": "G", "job": "jg", "job_state": "RUNNING"},
               {"runner": "A", "job": "ja", "job_state": "RUNNING"},  # tries only: not a stage-3 spool
               {"runner": "B", "job": "gone", "job_state": "RUNNING"}]  # no mirror: counts as current
    assert runner_pin.job_pin("jb", jobs=jobs, repo=repo) == "2.36.0"
    assert runner_pin.behind(runners, "3.0.2", jobs=jobs, repo=repo) == {"B": "2.36.0"}
    stopped = [{"runner": "B", "job": "jb", "job_state": "STOPPED"}]
    assert runner_pin.behind(stopped, "3.0.2", jobs=jobs, repo=repo) == {}


def test_render_ref_keeps_a_batch_at_the_pin_that_traced_it(tmp_path: Path) -> None:
    repo, old, new = _repo(tmp_path)
    assert runner_pin.render_ref({"commit": old}, "3.0.2", repo) == old
    assert runner_pin.render_ref({"commit": new}, "3.0.2", repo) == "main"
    assert runner_pin.render_ref({}, "3.0.2", repo) == "main"


def test_tick_holds_stage3_while_a_runner_is_behind_the_pin() -> None:
    calls = []

    def behind(runners, pin):
        calls.append((runners, pin))
        return {"B": "2.36.0"}

    p = SimpleNamespace(deps=SimpleNamespace(full={"runner_pin_behind": behind}),
                        results={"runners": [{"runner": "B"}]})
    why = tick._runners_behind_pin(p)
    assert why is not None and why.startswith("runner B runs TorchLens 2.36.0, the pin is ")
    assert calls and calls[0][0] == [{"runner": "B"}]
    p.deps.full["runner_pin_behind"] = lambda runners, pin: {}
    assert tick._runners_behind_pin(p) is None
