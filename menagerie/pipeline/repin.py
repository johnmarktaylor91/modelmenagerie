"""Released-pin checks on workers and a separately approved, trace-only flip on the hub."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from packaging.version import Version

from menagerie.pipeline import DataLayout, read_json
from menagerie.pipeline.records import LedgerRow, ModelRecord, RecordError

REPO = Path(__file__).resolve().parents[2]
BRANCH = "rebuild/r3-repin"
# Stable ids for the named carried rows and model families, never substring matches.
CARRIED_IDS = ("m14359", "m1191", "m2305", "m3638", "m10919", "m4246",
               "m14403", "m318", "m929", "m14393", "m14417", "m12019",
               "m4627", "m4628", "m4629", "m4310", "m4311", "m4312", "m4313")


def source_digest(repo: Path = REPO) -> str:
    """Bind evidence to the checked implementation, specs, locks, frozen baseline and models.

    The checked models are the Slice A baseline rows, the GELU cases and the hash harness's
    default cohort: their committed records and the adapter files those records name.
    """

    names = ["pyproject.toml", "docs/torchlens-boundary.md",
             "menagerie/host_budget.py", "menagerie/trace_summary.py",
             "menagerie/structural_digest.py",
             "menagerie/data/regression/slice_a_200_r1.tsv"]
    names += [f"menagerie/pipeline/{name}.py" for name in
              ("__init__", "repin", "repin_checks", "repin_slice", "repin_convs", "worker", "hash_harness",
               "runner", "flow", "stage3", "inputs", "envs", "harvest", "harvest_build",
               "worker_preview", "records", "ledger", "cli", "render", "regression")]
    names += [str(p.relative_to(repo)) for p in (repo / "envs").glob("*/*")
              if p.name in ("uv.lock", "pyproject.toml", "pixi.lock", "pixi.toml")]
    names += checked_model_files(repo)
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode() + b"\0" + (repo / name).read_bytes() + b"\0")
    return digest.hexdigest()



def checked_model_files(repo: Path = REPO) -> list[str]:
    """Committed model records and adapters the day, Slice A and conv3 checks run."""

    from menagerie.pipeline import hash_harness, regression, repin_checks, repin_slice

    data = DataLayout(repo / "menagerie" / "data")
    ids = {row["stable_id"] for row in regression.read_tsv(data.root / repin_slice.BASELINE)}
    ids |= set(repin_checks.GELU_IDS) | set(hash_harness.HARNESS_IDS)
    files = []
    for sid in sorted(ids):
        path = data.model(sid)
        files.append(str(path.relative_to(repo)))
        source = _recipe_source(json.loads(path.read_text()).get("recipe") or {}, data, repo)
        if source is not None:
            files.append(str(source.relative_to(repo)))
    return files


def _recipe_source(recipe: dict[str, Any], data: DataLayout, repo: Path) -> Path | None:
    """An adapter recipe's committed file, or a classic builder's module (``pkg.mod:func``)."""

    adapter = recipe.get("adapter")
    if not adapter:
        return None
    if recipe.get("kind") == "adapter":
        return data.root / adapter  # missing raises: the recipe could not run either
    module = repo / adapter.split(":", 1)[0].replace(".", "/")
    for candidate in (module.with_suffix(".py"), module / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_data(data: DataLayout) -> bool:
    """The digest hashes the checkout's own model records; evidence from another data root is not bound."""

    return data.root.resolve() == (REPO / "menagerie" / "data").resolve()

def check(version: str, *, data: DataLayout, out: Path | None = None,
          part: str = "all", ids: Sequence[str] = (), smoke: bool = False,
          deadline: float | None = None, explanations: Path | None = None) -> dict[str, Any]:
    """Run day probes, Slice A, or both; smoke evidence never authorizes a flip."""

    import torchlens as tl

    from menagerie.pipeline import repin_checks, repin_slice
    from menagerie.pipeline.publish import torchlens_pin

    if Version(version).is_prerelease or tl.__version__ != version or torchlens_pin() != version:
        raise RecordError(f"requested release {version}, installed {tl.__version__}, pin {torchlens_pin()}")
    if part not in ("day", "slice-a", "all"):
        raise RecordError(f"unknown repin check part {part}")
    notes = _explanations(explanations, version)
    target = out or Path("outputs")
    target.mkdir(parents=True, exist_ok=True)
    reports = {}
    if part in ("day", "all"):
        reports["day"] = repin_checks.day(data=data, out=target / "day", ids=ids)
    if part in ("slice-a", "all"):
        reports["slice-a"] = repin_slice.run(data=data, out=target, ids=ids,
                                             smoke=smoke, deadline=deadline, explanations=notes)
    _write_reports(reports, target, version=version, data=data, smoke=smoke, ids=ids)
    return {"version": version, "passed": all(r["passed"] for r in reports.values()),
            "parts": reports}


def _write_reports(reports: dict[str, dict[str, Any]], target: Path, *, version: str,
                   data: DataLayout, smoke: bool, ids: Sequence[str]) -> None:
    """Stamp each part with what it checked; chosen ids or smoke never pass as complete."""

    from menagerie.pipeline import repin_slice

    if "slice-a" in reports:
        reports["slice-a"]["baseline_sha256"] = _sha256(data.root / repin_slice.BASELINE)
    for name, report in reports.items():
        report.update(version=version, source_digest=source_digest(), smoke=smoke,
                      ids=sorted(ids), repo_data=repo_data(data))
        (target / f"repin-{name}.json").write_text(json.dumps(report, indent=2) + "\n")

def _explanations(path: Path | None, version: str) -> dict[str, str]:
    """Only an operator-supplied, source-bound explanation record can classify differences."""

    if path is None:
        return {}
    record = _json_object(path)
    if record.get("version") != version or record.get("source_digest") != source_digest():
        raise RecordError("explanations must match the checked version and source")
    notes = record.get("explanations")
    if not isinstance(notes, dict) or any(not isinstance(v, str) or not v.strip()
                                          for v in notes.values()):
        raise RecordError("explanations must map stable ids to nonempty evidence notes")
    return notes


def _json_object(path: Path) -> dict[str, Any]:
    """Operator and worker records must be JSON objects before their fields are trusted."""

    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RecordError(f"cannot read JSON record {path.name}") from exc
    if not isinstance(document, dict):
        raise RecordError(f"{path.name} must be a JSON object")
    return document


def priority_ids(data: DataLayout, version: str) -> list[str]:
    """Old-pin replay failures, GELU and carried rows, then old-pin validated passes."""

    # Essential complexity: keep the three ordered priority groups together.
    from menagerie.pipeline import ledger

    models = {r.id: r for r in (read_json(ModelRecord, p)
              for p in sorted((data.root / "models").rglob("*.json"))) if r.recipe}
    latest = {row.id: row for row in ledger.read_all(data=data)
              if _current_trace(row, models)}
    replay = [sid for sid, row in latest.items() if row.tl != version
              and row.cls in ("torchlens_gap:replay", "torchlens_unverified")]
    carried = [sid for sid in CARRIED_IDS if sid in models]
    passes = [sid for sid, row in latest.items() if row.tl != version and ledger.validated(row)]
    return list(dict.fromkeys([*sorted(replay), *carried, *sorted(passes)]))


def _current_trace(row: Any, models: dict[str, ModelRecord]) -> bool:
    """Current-recipe, settled full eval traces, at any pin."""

    if not isinstance(row, LedgerRow) or row.stage != "trace" or row.outcome == "infra":
        return False
    if row.capture not in (None, "full") or row.mode not in (None, "eval"):
        return False
    model = models.get(row.id)
    return model is not None and row.recipe_rev == model.recipe.rev


def _proof(version: str, check_dir: Path, repo: Path) -> list[str]:
    """Fail closed on missing, stale, partial, smoke or unexplained check evidence."""

    # Essential complexity: each independent evidence invariant must fail closed.
    problems = []
    for name in ("day", "slice-a", "conv3"):
        path = check_dir / f"repin-{name}.json"
        if not path.is_file():
            problems.append(f"missing {path.name}")
            continue
        try:
            report = _json_object(path)
        except RecordError:
            problems.append(f"invalid {path.name}")
            continue
        if report.get("version") != version or report.get("source_digest") != source_digest(repo):
            problems.append(f"{name} evidence does not match this pin and source")
        if report.get("passed") is not True or report.get("smoke"):
            problems.append(f"{name} did not pass a complete check")
        if not _part_complete(name, report, repo):
            problems.append(f"{name} does not satisfy its complete capture contract")
    return problems


def _part_complete(name: str, report: dict[str, Any], repo: Path = REPO) -> bool:
    """Each check part has its own count and capture contract, on the full default cohort."""

    if report.get("repo_data") is not True:
        return False
    if name in ("day", "slice-a") and (report.get("ids") != [] or report.get("smoke") is not False):
        return False
    if name == "slice-a":
        baseline = repo / "menagerie" / "data" / "regression" / "slice_a_200_r1.tsv"
        return (report.get("rows") == 200 and report.get("complete") is True
                and report.get("unexplained") == 0
                and report.get("baseline_sha256") == _sha256(baseline))
    if name == "conv3":
        return (report.get("rows") == 13 and report.get("threads") == 3
                and report.get("sandbox") is True)
    return True



def _c1_drained(path: Path | None) -> bool:
    """Read the independent catalog collector's C1 section, excluding C2 pending counts."""

    if path is None or not path.is_file():
        return False
    active = False
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            active = bool(re.match(r"^## C1\b", line))
        if active and line.startswith("Pending 0;"):
            return True
    return False


def _checkouts() -> tuple[Path, Path]:
    """Resolve existing main and candidate worktrees without creating either."""

    root = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=REPO,
                          check=True, capture_output=True, text=True).stdout
    paths = {line.removeprefix("branch refs/heads/"): Path(entry.splitlines()[0][9:])
             for entry in root.strip().split("\n\n") for line in entry.splitlines()
             if line.startswith("branch refs/heads/")}
    if "main" not in paths or BRANCH not in paths:
        raise RecordError("repin flip requires existing main and candidate worktrees")
    return paths["main"], paths[BRANCH]


def _merge(main: Path, tip: str) -> None:
    """Fast-forward a clean main to the validated commit, never to the mutable branch name."""

    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=main, check=True, capture_output=True, text=True)
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=main, check=True,
                            capture_output=True, text=True).stdout.strip()
    if status.stdout or branch != "main":
        raise RecordError("repin flip requires a clean main checkout")
    subprocess.run(["git", "merge", "--ff-only", tip], cwd=main, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=main, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != tip:
        raise RecordError(f"main is at {head}, not the validated candidate {tip}")


def _committed_tip(candidate: Path) -> str:
    """The candidate's commit when its tree is clean at the branch tip, else ``""``.

    Evidence hashes the candidate's files and the merge takes this exact commit.
    """

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=candidate, check=True,
                              capture_output=True, text=True).stdout.strip()

    try:
        head = git("rev-parse", "HEAD")
        clean = not git("status", "--porcelain", "--untracked-files=no")
        return head if clean and head == git("rev-parse", f"refs/heads/{BRANCH}") else ""
    except (OSError, subprocess.CalledProcessError):
        return ""


def _approved(record: dict[str, Any], version: str, digest: str) -> bool:
    """Approval fields require literal true, never numerically equal values."""

    return (record.get("version") == version and record.get("source_digest") == digest
            and record.get("c1_drained") is True and record.get("approved") is True)


def flip(version: str, *, data: DataLayout, dry_run: bool = False,
         check_dir: Path | None = None, go: Path | None = None,
         spool: Path | str | None = None, c1_status: Path | None = None) -> dict[str, Any]:
    """Fast-forward a checked branch and queue traces only after drain and operator approval.

    ``go`` is an explicit operator record with ``version``, ``source_digest``,
    ``c1_drained: true`` and ``approved: true``. Creating it is a separate human decision.
    A dry run reads evidence and names blockers without importing torch or touching a spool.
    """

    main, candidate = _checkouts()
    directory = check_dir or data.root / "regression" / f"repin-{version}"
    problems = _proof(version, directory, candidate)
    tip = _committed_tip(candidate)
    if not tip:
        problems.append("candidate checkout must be clean at the branch tip it merges")
    approval = _json_object(go) if go is not None else {}
    if not _approved(approval, version, source_digest(candidate)):
        problems.append("C1 drain and explicit operator approval are required")
    if not _c1_drained(c1_status):
        problems.append("independent C1 status must show Pending 0")
    ids = priority_ids(data, version)
    result = {"version": version, "ready": not problems, "blockers": problems,
              "ids": ids, "dry_run": dry_run, "trace_only": True}
    if dry_run:
        return result
    if problems:
        raise RecordError("repin flip held: " + "; ".join(problems))
    from menagerie.pipeline import stage3
    stage3._committed(ids, data)  # preflight live leases before changing the pin
    if _committed_tip(candidate) != tip:
        raise RecordError("repin flip held: the candidate moved after its evidence was checked")
    _merge(main, tip)
    # The candidate code remains loaded; its committed locks are the merged pin.
    try:
        result["jobs"] = stage3.queue(None, data=data, ids=ids, spool=spool, trace_only=True)
    except Exception as exc:
        # The distributed merge/queue operation cannot be atomic. Keep the checked pin
        # and make the incomplete queue explicit; retry uses stage3's pending-job guard.
        raise RecordError("pin merged, queue incomplete; rerun repin flip after repair") from exc
    return result
