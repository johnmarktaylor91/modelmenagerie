"""Mini-side recipe lane: no connection to the hub; the hub stages, serves verbs and pulls.

Runs inside one run-on job on a mini, by file path with the worker's own Python (3.9 on the
minis, so stdlib only and no package imports). The hub relay (``mini_relay``) stages the brief
and family folders into ``.mini-lane/`` in the job's checkout, answers the verb requests the
lane's wrappers queue in ``.mini-lane/spool/``, and after the job ends pulls ``outputs/lane/``
with ``pull-outputs``. No runtime installation, credentials or SSH configuration happens here.
A describe lane (manifest ``kind`` ``describe``) has no verbs: its agent reads the staged batch
folder beside its brief and writes ``out/<family>.json`` there, which ``finish`` returns.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LANE_TIMEOUT_S = 165 * 60
VERB_TIMEOUT_S = 30 * 60
# The relay may queue a verb behind others and a try may wait ten minutes for a runner.
VERB_WAIT_S = VERB_TIMEOUT_S + 15 * 60
STAGE_TIMEOUT_S = 15 * 60
POLL_S = 1.0
ROOT = ".mini-lane"  # hub-staged inputs and the verb spool; never pulled
OUT = "outputs/lane"  # the job's --pull glob
DRAFT_MAX_BYTES = 5 * 1024 * 1024
SNAP_MAX_FILES = 200
SNAP_MAX_BYTES = 20 * 1024 * 1024
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
BULK = (".pt", ".ckpt", ".safetensors", ".bin", ".npy", ".npz", ".h5", ".parquet", ".tar",
        ".zip", ".gz", ".tgz", ".bz2", ".xz", ".zst", ".7z", ".rar")
NOT_DRAFTS = ("sources", "tries", "menagerie", "BRIEF.md", "BRIEF.result.md", "deferred.json")
# Empty files the agent's Bash sandbox leaves in its working folder for the paths it protects;
# they carry nothing, and the `.env` ones make run-on refuse to archive a held checkout.
SANDBOX_STUBS = (".bash_profile", ".bashrc", ".gitconfig", ".github", ".gitmodules", ".idea", ".mcp.json",
                 ".npmrc", ".profile", ".ripgreprc", ".vscode", ".yarnrc", ".yarnrc.yml", ".zprofile",
                 ".zshrc", "bunfig.toml", "package.json", "package-lock.json", "pnpm-lock.yaml",
                 "yarn.lock", "scripts")
SCRUBBED = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "MENAGERIE_ENVS")


def expand(argv: Sequence[str], values: Mapping[str, str]) -> list[str]:
    """Fill the configured ``{job}`` and ``{account}`` placeholders; nothing else is parsed."""
    out = []
    for word in argv:
        for key, value in values.items():
            word = word.replace("{" + key + "}", value)
        out.append(word)
    return out


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def runtime_env(spec: Mapping[str, Any]) -> dict[str, str]:
    """Recheck the runtime identity inside the admitted worker job before any recipe work."""
    # The configured runtime supplies its own login. Never serialize a login in the job.
    env = {**os.environ, "CLAUDE_ACCOUNT": spec["account"],
           "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1", "FLEET_BACKGROUND_LANE": "1"}
    for key in SCRUBBED:
        env.pop(key, None)
    values = {"job": spec["label"], "account": spec["account"]}
    probe = subprocess.run(expand(spec["identity"], values), capture_output=True, text=True,
                           env=env, timeout=120, check=True)
    actual = json.loads(probe.stdout)
    if not isinstance(actual, dict) or actual.get("harness") != "claude" \
            or actual.get("account") != spec["account"]:
        raise OSError("worker job runtime identity differs from the admitted account")
    return env


def wait_ready(root: Path, timeout: float = STAGE_TIMEOUT_S) -> None:
    """The hub writes READY after staging inputs (and any login delivery), or ABORT."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (root / "ABORT").is_file():
            raise OSError("hub aborted the lane: " + (root / "ABORT").read_text()[:300])
        if (root / "READY").is_file():
            return
        time.sleep(POLL_S)
    raise OSError("hub did not stage the lane's inputs")


def wrapper_text(root: Path, family: str) -> str:
    argv = [sys.executable, str(Path(__file__).resolve()), "verb", str(root), family]
    return "#!/bin/sh\n# This lane's verbs, answered by the hub.\nexec " + " ".join(
        "'" + a.replace("'", "'\\''") + "'" for a in argv) + ' "$@"\n'


def write_wrappers(root: Path, families: Sequence[str]) -> None:
    for family in families:
        path = root / "work" / family / "menagerie"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(wrapper_text(root, family), encoding="utf-8")
        path.chmod(0o755)


def verb(root: Path, family: str, args: Sequence[str], *, wait: float = VERB_WAIT_S) -> int:
    """Queue one verb for the hub and print its answer; 75 when the hub did not answer."""
    name = f"{time.time_ns()}-{secrets.token_hex(4)}"
    request = {"id": name, "family": family, "args": list(args), "cwd": os.getcwd()}
    _write_atomic(root / "spool" / "new" / f"{name}.json", json.dumps(request))
    done = root / "spool" / "done" / f"{name}.json"
    deadline = time.monotonic() + wait
    while not done.is_file():
        if time.monotonic() >= deadline:
            print("infra: the hub did not answer this verb; retry it", file=sys.stderr)
            return 75
        time.sleep(POLL_S)
    answer = json.loads(done.read_text(encoding="utf-8"))
    print(answer.get("stdout", ""), end="")
    print(answer.get("stderr", ""), end="", file=sys.stderr)
    return int(answer.get("rc", 1))


def _draft_files(folder: Path) -> tuple[list[Path], list[str]]:
    """Small non-bulk files the agent wrote, outside saved sources, attempts and wrappers."""
    kept, skipped = [], []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if relative.parts[0] in NOT_DRAFTS or path.is_symlink() or not path.is_file() \
                or _stub(relative, path):
            continue
        if path.suffix.lower() in BULK or path.stat().st_size > DRAFT_MAX_BYTES:
            skipped.append(str(relative))
            continue
        kept.append(relative)
    return kept, skipped


def _stub(relative: Path, path: Path) -> bool:
    name = relative.name
    return len(relative.parts) == 1 and path.stat().st_size == 0 and (
        name in SANDBOX_STUBS or name == ".env" or name.startswith(".env."))


def _copy_drafts(folder: Path, dest: Path) -> list[str]:
    """Copy one family's drafts; return the files left behind as too big or bulk."""
    files, skipped = _draft_files(folder)
    for relative in files:
        (dest / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / relative, dest / relative)
    return skipped


def snapshot(root: Path, name: str, family: str) -> dict[str, Any]:
    """Copy one family's drafts for one verb into ``snap/<name>/``: the hub scans and pulls that
    copy, so the files it checks are the files it gets."""
    if not SAFE.fullmatch(name) or not SAFE.fullmatch(family):
        return {"refused": "unsafe snapshot name"}
    folder, dest = root / "work" / family, root / "snap" / name
    shutil.rmtree(dest, ignore_errors=True)
    files, skipped = _draft_files(folder) if folder.is_dir() else ([], [])
    total = sum((folder / f).stat().st_size for f in files)
    if len(files) > SNAP_MAX_FILES or total > SNAP_MAX_BYTES:
        return {"refused": f"drafts too large for the hub ({len(files)} files, {total} bytes)"}
    if files:
        _copy_drafts(folder, dest / family)
    return {"files": len(files), "bytes": total, "skipped": skipped}


def finish(root: Path, out: Path, spec: Mapping[str, Any], rc: int | None) -> bool:
    """Copy drafts and the result into the pulled folder; return whether the result exists.
    A describe lane's drafts are the ``out/`` folder beside its brief, returned as ``describe/``."""
    if spec.get("kind") == "describe":
        drafts = root / Path(spec["brief"]).parent / "out"
        skipped = {"out": _copy_drafts(drafts, out / "describe")} if drafts.is_dir() else {}
    else:
        skipped = {family: _copy_drafts(root / "work" / family, out / "families" / family)
                   for family in spec["families"] if (root / "work" / family).is_dir()}
    result = root / spec["result"]
    present = result.is_file() and result.stat().st_size > 0
    if present:
        shutil.copy2(result, out / "result.md")
    _write_atomic(out / "lane.json", json.dumps({
        "label": spec["label"], "rc": rc, "result": present, "families": spec["families"],
        "skipped": {k: v for k, v in skipped.items() if v}, "ended": time.time()}))
    return present


def agent_argv(spec: Mapping[str, Any]) -> list[str]:
    runtime = expand(spec["runtime"], {"job": spec["label"], "account": spec["account"]})
    if spec.get("nice"):  # on a phone-build mini, below the phone builds' priority
        runtime = ["nice", "-n", str(int(spec["nice"])), *runtime]
    return [*runtime, "--model", spec["model"], "--effort", spec["effort"],
            "--dangerously-skip-permissions", "--disallowedTools", "WebFetch,WebSearch,mcp__executor",
            "--disable-slash-commands", "--strict-mcp-config", "-p"]


def _prepare(spec: Mapping[str, Any]) -> subprocess.Popen[bytes] | None:
    """The runtime's optional receiver, started before READY (it may wait for a hub delivery)."""
    if not spec.get("prepare"):
        return None
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
    return subprocess.Popen(expand(spec["prepare"], {"job": spec["label"], "account": spec["account"]}),
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=env, start_new_session=True)


def run(spec: Mapping[str, Any], cwd: Path | None = None) -> int:
    base = Path(cwd or Path.cwd()).resolve()
    root, out = base / ROOT, base / OUT
    out.mkdir(parents=True, exist_ok=True)
    rc, helper = None, _prepare(spec)
    try:
        wait_ready(root, float(spec.get("stage_timeout_s", STAGE_TIMEOUT_S)))
        env = runtime_env(spec)
        if spec.get("kind") != "describe":  # a describe lane has no verbs
            write_wrappers(root, spec["families"])
        brief = root / spec["brief"]
        with (out / "agent.log").open("w", encoding="utf-8") as log:
            done = subprocess.run(agent_argv(spec), input=brief.read_text(encoding="utf-8"),
                                  text=True, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=brief.parent, env=env, check=False,
                                  timeout=float(spec.get("lane_timeout_s", LANE_TIMEOUT_S)))
        rc = done.returncode
    finally:
        if helper is not None and helper.poll() is None:
            helper.terminate()
        # A missing result fails the job; drafts and the log are still pulled for evidence.
        present = finish(root, out, spec, rc)
    if not present:
        print("mini lane did not write its required result", file=sys.stderr)
        return rc or 1
    return rc


def main(argv: Sequence[str]) -> int:
    if not argv:
        raise ValueError("expected run, verb or snapshot")
    if argv[0] == "run":
        return run(json.loads(argv[1]))
    if argv[0] == "verb":
        return verb(Path(argv[1]), argv[2], argv[3:])
    if argv[0] == "snapshot":
        print(json.dumps(snapshot(Path(argv[1]), argv[2], argv[3])))
        return 0
    raise ValueError("expected run, verb or snapshot")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
