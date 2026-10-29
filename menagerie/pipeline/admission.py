"""Starting agent lanes: admission, the launch and its lease, and the queue of refused steps.

Split from ``drive`` at rung 2 (``drive`` re-exports ``launch``, ``lanes``, ``Refused`` and
``launch_first``). Runs on the hub; imports neither torch nor psutil.

Admission is read before every launch, fresh, for the subscription the role resolves to: a refusal
raises ``Refused("no_admission")``, never a downgrade, a model swap or a wait. Claude lanes start
on the project's account (``MENAGERIE_CLAUDE_ACCOUNT``, else the checkout's ``.claude-account``,
else the pool; from rung 3 the tick unit sets ``pool``), through the same gate and burst check
whichever account it is, and an account whose quota is unknown is refused. Codex lanes read
the ChatGPT gate; Google lanes (harness ``antigravity``) the Gemini subscription gate
(``agy-quota gate --require-fresh``) and start through ``agy-bg.sh``. Every Google lane, the
review seat's included, starts here (rung 3 merged the review seat's own launcher in).

The refusal rules (BUILD-PLAN r2 section 5): ``launch_first`` tries its roles in order, so 2a and
2b go to whichever lab is admitted and a refusal slows the run without stopping it. A step that
no listed role admits is queued (``queue/<label>.json`` in the work root) with its trigger, and
``requeue`` (the run's watcher calls it every 30 minutes) re-reads admission and starts what it
now admits. A Google implementation lane has network and may write outside its folder, so 2b
reaches it only as the third option after both main labs refuse, and ``measures`` flags it.
A lab-tied step (``lab``) refuses with ``lab_mismatch`` and queues when routing moves its role
to the other lab (BUILD-PLAN r3 section 1, item 4).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, StoreLayout, now
from menagerie.pipeline.records import RecordError

# Lane kind: its hard timeout in minutes, which is also its lease. 2a takes a batch of up to five
# families at 15 minutes each; 2b up to three single-model families at 45 minutes each.
LANE_MINUTES = {"2a": 120, "2b": 180, "review": 90, "prose-audit": 60, "describe": 120}
SOFT_MARGIN_MIN = 15  # fleet-run's soft deadline comes this long before the hard timeout
WEB_TOOLS = "WebFetch,WebSearch,mcp__executor"  # what a Claude lane without web loses
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
HARNESSES = ("claude", "codex", "antigravity")
# The two main labs' lab-agnostic roles, and the Google seats.
MAIN_ROLES = {"model-default": "claude", "model-implementation": "codex"}
THIRD_LAB_IMPLEMENT = "model-implementation-third-lab"
THIRD_LAB_REVIEW = "model-review-third-lab-plan"
# Google lanes: which kinds may run there. A 2b lane there has network (see the module docstring).
ANTIGRAVITY_KINDS = ("review", "2b")
# How a read-only Google lane answers: one ``{"files": ...}`` object, or one fenced JSON block per
# item (a review lane's cases), each written to ``out/<id>.json`` beside the brief.
ANSWERS = ("files", "blocks")
# The week-use ceiling for unattended Claude lanes (the fleet's lane admission ceiling).
UNATTENDED_MAX_PCT = 70
# How many lanes of a kind the tick keeps running at once. 2b runs 12 to 16 wide (JMT 2026-10-08
# 5:06 PM PT, BUILD-PLAN r3 amendment); ``MENAGERIE_LANES_<KIND>`` overrides, never above the max
# (``MENAGERIE_LANES_DESCRIBE`` like ``MENAGERIE_LANES_2B``). Describe is four seats on disjoint
# family leases (speed-up idea 11); the tick's describe stage runs up to its ``describes.lanes``.
LANE_CAPS = {"2a": 4, "2b": 14, "review": 1, "describe": 4}
LANE_CAP_MAX = {"2a": 8, "2b": 16, "review": 2, "describe": 4}


def lane_cap(kind: str) -> int:
    """The concurrent-lane cap for ``kind``: the env override clamped to its maximum, else the default."""

    given = os.environ.get(f"MENAGERIE_LANES_{kind.upper()}")
    cap = int(given) if given else LANE_CAPS.get(kind, 0)
    return max(0, min(cap, LANE_CAP_MAX.get(kind, cap)))
AGY_ANSWER = re.compile(r"\{.*\}", re.DOTALL)
FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
READ_ONLY_NOTE = """

## This seat cannot write files

You run read-only, with no web: read only the local files the brief names, and never open a URL
or use a web tool (the sources you need are saved locally). Write nothing; instead end your answer with one JSON object
`{"files": {"<absolute path>": <the file's content>}}` holding every file this brief asks you to
write (a JSON file's content as a JSON value, any other file's as a string). Paths must be inside
`{folder}`.
"""
BLOCKS_NOTE = """

## This seat cannot write files

Your working directory is the lane folder: read every file by its path relative to it
(`cases/<case>/...`) with your file viewing and listing tools. There is no web access: never fetch
a URL or use a web tool. You cannot write files. Instead, for each item, put the JSON object the
brief asks you to write as `out/<item>.json` (with its `"id"`) in your final answer inside its
own ```json fenced block, one block per item, in order; then one line per item, `<item> <verdict>`.
"""


class Refused(RuntimeError):
    """A launch that did not start: ``result`` is ``no_admission`` or ``lease_held``."""

    def __init__(self, result: str, why: str, **fields: Any) -> None:
        super().__init__(f"{result}: {why}")
        self.result = {"result": result, "why": why, **fields}


def _desk() -> Path:
    return Path(os.environ.get("FLEET_RUN_DESK_ROOT") or Path.home() / "desk")


def _tool(name: str) -> str:
    return shutil.which(name) or str(_desk() / "bin" / name)


def _fleet_run() -> str:
    return os.environ.get("FLEET_RUN_BIN") or _tool("fleet-run")


def _sh(argv: Sequence[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=120, **kw)


def _agy_bg() -> str:
    return os.environ.get("MENAGERIE_AGY_BG") or str(Path.home() / ".claude/scripts/agy-bg.sh")


# --- admission -------------------------------------------------------------------------------


def _plan(role: str, *, background: bool) -> dict[str, Any]:
    argv = [_tool("role"), "resolve", role, *(["--background"] if background else []), "--json"]
    done = _sh(argv)
    if done.returncode == 75:
        raise Refused("no_admission", f"role {role}: no eligible seat", role=role)
    if done.returncode:
        raise RecordError(f"role resolve {role} failed: {done.stderr.strip()[-300:]}")
    plan = json.loads(done.stdout)
    if plan.get("deferred"):
        raise Refused("no_admission", f"role {role}: deferred ({plan.get('routing')})", role=role)
    if plan.get("harness") not in HARNESSES or not plan.get("model"):
        raise RecordError(f"role {role} is not a Claude, Codex or Google lane: "
                          f"{plan.get('harness')}")
    return plan


def third_lab_held() -> bool:
    """Whether the tick config holds the Google seats (``third_lab_hold``, default false)."""
    from menagerie.pipeline import tick

    value = tick.config().get("third_lab_hold", False)
    if not isinstance(value, bool):
        raise ValueError("third_lab_hold must be true or false")
    return value


def _resolve(role: str, account: str = "pool", *, harness: str | None = None) -> dict[str, Any]:
    """The seat a role runs on. Background routing moves a role off Claude when the subscription
    pool is short; a project account outside the pool (``org`` through rung 2) has its own
    headroom, so there a role whose preferred seat is Claude stays on it and that account's own
    gate decides (``_admit``). A held Google seat (``third_lab_held``) is refused like an
    unadmitted one, so ``launch_first`` moves on to the next role or queues the step.

    ``harness`` is the only harness the caller's placement can run (a worker whose installed
    runtime serves Claude alone). Background routing's other-lab seat is unusable there, so the
    role's own unrouted seat is tried before the role is given up; the pool's own gate in
    ``_admit`` still decides whether that seat starts.
    """

    if role in (THIRD_LAB_IMPLEMENT, THIRD_LAB_REVIEW) and third_lab_held():
        raise Refused("no_admission", f"role {role}: held (tick config third_lab_hold)", role=role)
    if account != "pool":
        direct = _plan(role, background=False)
        if direct.get("harness") == "claude":
            return {**direct, "routing": {"component": direct.get("component"),
                                          "reason": f"project account {account}"}}
    routed = _plan(role, background=True)
    if harness is None or routed.get("harness") == harness:
        return routed
    own = _plan(role, background=False)
    if own.get("harness") != harness:
        return routed
    return {**own, "routing": {"component": own.get("component"),
                               "reason": f"placement runs {harness} only"}}


def claude_account(cwd: Path) -> str:
    """The account Claude lanes start on: ``MENAGERIE_CLAUDE_ACCOUNT``, else the checkout's
    ``.claude-account`` (``org`` through rung 2, JMT 2026-10-05 in ``decisions.md``), else
    ``pool``."""

    named = os.environ.get("MENAGERIE_CLAUDE_ACCOUNT", "").strip()
    if named:
        return named
    marker = Path(cwd) / ".claude-account"
    text = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    return text.split()[0] if text else "pool"


def _claude_ceiling(kind: str) -> int:
    """The Claude week-use ceiling: the unattended one, lowered (never raised) for describe seats
    by ``MENAGERIE_DESCRIBE_MAX_PCT``, so 2a and 2b keep the band above it (the tick's describe
    stage sets it)."""

    given = os.environ.get("MENAGERIE_DESCRIBE_MAX_PCT", "").strip() if kind == "describe" else ""
    return min(UNATTENDED_MAX_PCT, int(given)) if given else UNATTENDED_MAX_PCT


def _codex_gate(role: str, kind: str) -> None:
    """The ChatGPT gate, at ``MENAGERIE_CODEX_MAX_PCT`` for review lanes when that is set."""

    ceiling = os.environ.get("MENAGERIE_CODEX_MAX_PCT", "").strip() if kind == "review" else ""
    gate = _sh(["/usr/bin/python3", str(_desk() / "tools" / "chatgpt_usage.py"), "gate",
                "--require-fresh", *(["--max-pct", ceiling] if ceiling else [])])
    if gate.returncode:
        raise Refused("no_admission", gate.stderr.strip() or "no fresh Codex headroom", role=role)


def _gemini_gate(role: str) -> None:
    """The Gemini subscription gate on fresh readings (``agy-quota gate --require-fresh``)."""

    gate = _sh([_tool("agy-quota"), "gate", "--require-fresh"])
    if gate.returncode:
        raise Refused("no_admission", (gate.stderr or gate.stdout).strip()[-300:]
                      or "no fresh Gemini subscription headroom", role=role)


def _admit(harness: str, role: str, cwd: Path, kind: str = "") -> tuple[list[str], str]:
    """Fresh admission for the selected subscription: the lane's env words and its account.

    A refusal raises ``Refused("no_admission")``: never a downgrade, a model swap or a wait.
    Google lanes are admitted by ``agy-quota gate --require-fresh`` after the pause check here
    (``agy-bg.sh`` reads the same gate again and exits 75 on a refusal).
    Codex review lanes take the ChatGPT gate's ceiling from ``MENAGERIE_CODEX_MAX_PCT`` when it
    is set (Prime, 2026-10-05: 88 for review-only lanes until the Oct 9 reset), else its default.
    Claude describe lanes are gated at ``_claude_ceiling``.
    """

    py, tools = "/usr/bin/python3", _desk() / "tools"
    paused = _sh([py, str(tools / "pause.py"), str(Path.home() / ".claude" / "state"),
                  "gemini" if harness == "antigravity" else harness])
    if paused.returncode:
        raise Refused("no_admission", f"{harness} lanes are paused", role=role)
    if harness == "antigravity":
        _gemini_gate(role)
        return [], "gemini-plan"
    if harness == "codex":
        _codex_gate(role, kind)
        return ["-u", "FLEET_RUN_CODEX_AUTH"], "chatgpt"
    asked = claude_account(cwd)
    env = {**os.environ, "FLEET_LOOP_OWNER": "1", "CLAUDE_ACCOUNT": asked}
    picked = _sh([py, str(tools / "lane_admission.py"), "env", "--cwd", str(cwd)], env=env)
    updates = dict(line.split("=", 1) for line in picked.stdout.splitlines() if "=" in line)
    account = updates.get("CLAUDE_ACCOUNT")
    if picked.returncode or not account:
        raise Refused("no_admission", picked.stderr.strip() or f"no admission on {asked}",
                      role=role, account=asked)
    # Unknown quota is a refusal (BUILD-PLAN r3 item 4): the picked account must show known,
    # fresh headroom under the unattended ceiling, whatever the selection step decided.
    known = _sh([py, str(tools / "lane_admission.py"), "gate", account, "--max-pct",
                 str(_claude_ceiling(kind)), "--require-fresh"], env=env)
    if known.returncode:
        raise Refused("no_admission", known.stderr.strip()[-300:]
                      or f"{account}: no known fresh quota", role=role, account=account)
    burst =_sh([_tool("burst-admission"), "continue", "--account", account])
    if burst.returncode:
        raise Refused("no_admission", burst.stderr.strip() or f"burst admission: {account}",
                      role=role, account=account)
    # Without CLAUDE_CONFIG_DIR the `claude` shim goes through claude-as on the picked account.
    unset = ["-u", "CLAUDE_CONFIG_DIR", "-u", "ANTHROPIC_API_KEY", "-u", "ANTHROPIC_AUTH_TOKEN"]
    return [*unset, *(f"{k}={v}" for k, v in updates.items())], account


# A gauge reading serves every launch inside this many seconds: the tick starts many lanes a
# pass, and one reading (seconds each; the Gemini one asks the model) covers them all.
GAUGES_FRESH_S = 300
_GAUGES: dict[tuple[str, ...], tuple[float, list[str]]] = {}


def _gauge(argv: Sequence[str]) -> list[str]:
    """A gauge tool's output lines, read at most once per ``GAUGES_FRESH_S`` (none on error)."""

    hit = _GAUGES.get(tuple(argv))
    if hit is not None and time.monotonic() - hit[0] < GAUGES_FRESH_S:
        return hit[1]
    try:
        done = _sh(argv)
        lines = done.stdout.strip().splitlines() if done.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        lines = []
    _GAUGES[tuple(argv)] = (time.monotonic(), lines)
    return lines


def gauges(account: str | None = None) -> dict[str, str]:
    """The three quota gauges at launch time, as their tools print them (``unknown`` on error),
    each as read within the last ``GAUGES_FRESH_S``."""

    py, tools = "/usr/bin/python3", _desk() / "tools"
    asks = {"claude": [_tool("account-usage"), "show"],
            "codex": [py, str(tools / "chatgpt_usage.py"), "show"],
            "gemini": [_tool("agy-quota"), "show"]}
    out = {}
    for name, argv in asks.items():
        lines = _gauge(argv)
        if name == "claude" and account:
            lines = [line for line in lines if line.startswith(f"{account}:")] or lines
        out[name] = "; ".join(lines)[:300] or "unknown"
    return out


def _harness_args(harness: str, plan: Mapping[str, Any], *, web: bool,
                  writable: Sequence[Path]) -> list[str]:
    """The adapter's vendor options: model, effort and, without web, the restriction."""

    effort = plan.get("effort") or "high"
    if harness == "claude":
        args = ["--model", plan["model"], "--effort", effort, "--dangerously-skip-permissions"]
        return args if web else [*args, "--disallowedTools", WEB_TOOLS]
    effort = effort.replace("max", "xhigh")  # Codex has no max
    args = ["--model", plan["model"], "-c", f"model_reasoning_effort={effort}"]
    if web:
        return [*args, "--sandbox", "danger-full-access"]
    args += ["--sandbox", "workspace-write", "-c", "sandbox_workspace_write.network_access=false"]
    return [*args, *(word for d in writable for word in ("--add-dir", str(d)))]


# --- leases ----------------------------------------------------------------------------------


@contextlib.contextmanager
def _lease_lock() -> Iterator[Path]:
    leases = work_dir("leases")
    leases.mkdir(parents=True, exist_ok=True)
    with (leases / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield leases
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _held(leases: Path, keys: Sequence[str], label: str) -> None:
    """Refuse when a live lease holds the label or a key; release ended leases met on the way."""

    for path in leases.glob("*.json"):
        lease = json.loads(path.read_text(encoding="utf-8"))
        taken = sorted(set(keys) & set(lease["keys"]))
        if (path.stem == label or taken) and _live(path, lease) is None:
            continue
        if path.stem == label:
            raise Refused("lease_held", f"label {label} is live as run {lease['run']}",
                          label=label)
        if taken:
            raise Refused("lease_held", f"{', '.join(taken)} held by {path.stem}",
                          keys=taken, by=path.stem)


def _live(path: Path, lease: Mapping[str, Any]) -> str | None:
    """The fleet-run state of a lease's run, or ``None`` after releasing an ended or expired one.

    An unknown run keeps its lease until the lease expires. A read-only Google lane's answer is
    unpacked into the files its brief asked for when it ends (``unpack``). An ended mini lane
    keeps its lease (state ``INGESTING``) until the hub relay has ingested its pulled outputs
    (``mini_relay``) or the lease expires. Call under the lease lock.
    """

    status = _status(lease["run"])
    state = status["run"]["state"] if status else "unknown"
    expired = _parse(lease["expires"]) <= datetime.now(timezone.utc)
    if state == "TERMINAL" and lease.get("mini") and not expired:
        from menagerie.pipeline import mini_relay

        if not mini_relay.ingested(lease["label"]):
            return "INGESTING"  # the hub relay has not pulled the mini lane's drafts yet
    if state == "TERMINAL" or expired:
        if state == "TERMINAL" and lease.get("answer"):
            with contextlib.suppress(RecordError, OSError, ValueError):
                unpack(lease)
        path.unlink()
        return None
    return state


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _stamp(when: datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(at: str) -> datetime:
    return datetime.fromisoformat(at.replace("Z", "+00:00"))


def _status(run: str) -> dict[str, Any] | None:
    done = _sh([_fleet_run(), "status", "--json", run])
    try:
        return json.loads(done.stdout) if done.returncode == 0 else None
    except ValueError:
        return None


def lanes(*, data: DataLayout) -> list[dict[str, Any]]:
    """List live lanes with their lease and fleet-run state; release ended or expired leases.

    A lease is released (its file deleted) once fleet-run calls the run terminal or its hard
    timeout has passed; an unknown run keeps its lease until it expires. ``data`` is unused: the
    leases live in the work root. Side effects: released leases, unpacked Google answers.
    """

    live = []
    with _lease_lock() as leases:
        for path in sorted(leases.glob("*.json")):
            lease = json.loads(path.read_text(encoding="utf-8"))
            state = _live(path, lease)
            if state is not None:
                age = datetime.now(timezone.utc) - _parse(lease["started"])
                live.append({**lease, "state": state,
                             "age_min": round(age.total_seconds() / 60, 1)})
    return live


# --- launch ----------------------------------------------------------------------------------


def _adapter() -> str:
    found = os.environ.get("FLEET_RUN_ADAPTER")
    return found or str((_desk() / "tools/fleet-run/current/fleet-run-adapter").resolve())


def _writable() -> tuple[Path, ...]:
    """What a lane without network may write besides its folder: the bridge spool, the lane logs,
    the data root (``done``, ``giveup`` and ``review record`` write there) and the store root
    (``review record`` saves the findings there)."""

    store = os.environ.get("MENAGERIE_STORE")
    return (work_dir("hubspool"), work_dir("lanes"), DataLayout.from_env().root,
            *([StoreLayout(Path(store)).root] if store else []))


def _agy_argv(role: str, brief: Path, label: str, minutes: int, *, read_only: bool,
              answer: str = "files") -> tuple[list[str], Path, Path | None]:
    """``agy-bg.sh`` for a Google lane: its argv, the result file it writes and, read-only, the
    answer file to unpack. The prompt is the brief, plus the answer convention when read-only."""

    folder = brief.parent / f"agy-{label}-{now().replace(':', '').lower()}"
    folder.mkdir(parents=True)
    prompt = folder / "prompt.md"
    note = ("" if not read_only else BLOCKS_NOTE if answer == "blocks"
            else READ_ONLY_NOTE.replace("{folder}", str(brief.parent)))
    prompt.write_text(brief.read_text(encoding="utf-8") + note, encoding="utf-8")
    argv = [_agy_bg(), str(folder / "lane.log"), str(prompt), "--role", role,
            "--cd", str(brief.parent), "--label", label,
            "--max-runtime-min", str(min(minutes, 240))]
    answer = folder / "result.md"
    return argv, answer, answer if read_only else None


def _argv(kind: str, harness: str, plan: Mapping[str, Any], brief: Path, *, label: str,
          role: str, cwd: Path, parent: str | None, web: bool, env_words: Sequence[str],
          answer: str = "files") -> tuple[list[str], Path | None]:
    """The fleet-run start for a Claude or Codex lane, or ``agy-bg.sh`` for a Google one."""

    minutes = LANE_MINUTES[kind]
    if harness == "antigravity":
        argv, _, found = _agy_argv(role, brief, label, minutes, read_only=not plan.get("edits"),
                                   answer=answer)
        return argv, found
    result = brief.with_name(f"{brief.stem}.result.md")
    return [
        _fleet_run(), "start", "--kind", harness, "--label", label, "--cwd", str(cwd),
        "--project", "menagerie", "--require-result", "--artifact", str(result),
        "--deadline", f"{minutes - SOFT_MARGIN_MIN}m", "--hard-timeout", f"{minutes}m",
        *(["--parent", parent] if parent else []), "--",
        "/usr/bin/env", *env_words, "/usr/bin/python3", str(_desk() / "tools" / "lane_failover.py"),
        "run", "--expect", str(result), "--", _adapter(), harness, str(brief),
        *_harness_args(harness, plan, web=web, writable=(result.parent, *_writable())),
    ], None


def _check_kind(kind: str, harness: str, role: str, network_ok: bool) -> None:
    if harness != "antigravity":
        return
    if kind not in ANTIGRAVITY_KINDS:
        raise RecordError(f"a {kind} lane does not run on the Google seat ({role})")
    if kind == "2b" and not network_ok:
        raise RecordError(f"{role} has network: a 2b lane reaches it only after both main labs "
                          "refuse (launch_first)")


def _check_lab(lab: str | None, harness: str, role: str) -> None:
    """The lab guard: a lab-tied step never runs on the other lab when routing moves its role."""

    if lab is not None and lab not in HARNESSES:
        raise RecordError(f"lab {lab!r} is not one of {HARNESSES}")
    if lab is not None and harness != lab:
        raise Refused("lab_mismatch", f"role {role} resolves to {harness}; the step needs {lab}",
                      role=role, harness=harness, lab=lab)


def launch(kind: str, role: str, brief: Path, *, keys: Sequence[str] = (),
           parent: str | None = None, label: str | None = None, web: bool = False,
           network_ok: bool = False, lab: str | None = None, answer: str = "files",
           ids: Sequence[str] = ()) -> str:
    """Start one headless agent lane by role and lease its keys.

    Inputs: the lane kind (``2a``, ``2b``, ``review``, ``prose-audit``), a fleet role, the brief
    file, the keys it leases (families or stable ids), the parent run, a label, whether web
    tools are on, ``network_ok`` (a 2b lane may run on the Google seat, which has network) and
    ``lab`` (a harness name for a lab-tied step: describe prose and the ``model-default`` seat
    are ``claude``, reviews of Claude rows ``codex``). ``answer`` is how a read-only Google
    lane answers (``ANSWERS``): ``files``, one ``{"files": ...}`` object, or ``blocks``, one
    fenced JSON object per item of ``ids``, written to ``out/<id>.json`` (the review seat).
    The lane must write ``<brief stem>.result.md`` beside its brief (a read-only Google lane's
    answer is unpacked into it). Returns: the lane's run id. Raises ``Refused``
    (``lab_mismatch`` when the role resolves to another lab than ``lab``, checked before any
    quota is read; ``no_admission`` when the selected subscription's fresh admission says no or
    its quota is unknown; ``lease_held`` when a live lease holds a key or the label). Side
    effects: the lane, its lease ``leases/<label>.json`` and a ``launch`` event recording what
    ran and the quota gauges.
    """

    from menagerie.pipeline import drive

    brief = Path(brief).resolve(strict=True)
    label = label or f"{kind}-{brief.parent.name}"
    if not LABEL.fullmatch(label):
        raise RecordError(f"label {label!r} must match {LABEL.pattern}")
    if kind not in LANE_MINUTES:
        raise RecordError(f"lane kind {kind!r} is not one of {sorted(LANE_MINUTES)}")
    if answer not in ANSWERS or (answer == "blocks") != bool(ids):
        raise RecordError(f"answer {answer!r} is not one of {ANSWERS}, or blocks without ids")
    cwd = Path(__file__).resolve().parents[2]
    plan = _resolve(role, claude_account(cwd))
    harness = plan["harness"]
    _check_lab(lab, harness, role)
    _check_kind(kind, harness, role, network_ok)
    env_words, account = _admit(harness, role, cwd, kind)
    minutes = LANE_MINUTES[kind]
    with _lease_lock() as leases:
        _held(leases, keys, label)
        argv, found = _argv(kind, harness, plan, brief, label=label, role=role, cwd=cwd,
                            parent=parent, web=web, env_words=env_words, answer=answer)
        # agy-bg.sh takes its parent from FLEET_RUN_ID.
        env = {**os.environ, "FLEET_RUN_ID": parent} if parent and harness == "antigravity" \
            else None
        started = _sh(argv, cwd=cwd, env=env)
        run = re.search(r"run_id=(\S+)", started.stdout)
        if harness == "antigravity" and started.returncode == 75:
            raise Refused("no_admission", f"role {role}: no Gemini subscription room", role=role)
        if started.returncode or not run:
            why = (started.stderr or started.stdout)[-400:]
            raise RecordError(f"fleet-run start failed: {why}")
        lease = {
            "label": label, "run": run.group(1), "kind": kind, "role": role, "harness": harness,
            "model": plan["model"], "effort": plan.get("effort"), "account": account,
            "routing": plan.get("routing"),
            "web": web or bool(harness == "antigravity" and plan.get("edits")),
            "keys": list(keys), "parent": parent, "brief": str(brief),
            "result": str(brief.with_name(f"{brief.stem}.result.md")),
            "answer": str(found) if found else None, "answer_kind": answer,
            "ids": list(ids), "started": now(),
            "expires": _stamp(datetime.now(timezone.utc) + timedelta(minutes=minutes)),
        }
        _write_atomic(leases / f"{label}.json", json.dumps(lease, indent=1) + "\n")
    fields = {k: v for k, v in lease.items() if k != "run"}
    drive.event(lease["run"], "", "launch", **fields, gauges=gauges(account))
    return lease["run"]


# --- read-only Google answers ----------------------------------------------------------------


def answer_objects(text: str) -> list[dict[str, Any]]:
    """The JSON objects in a read-only lane's answer: fenced blocks, else bare objects."""

    found = []
    for block in FENCED.findall(text):
        with contextlib.suppress(ValueError):
            found.append(json.loads(block))
    if not found:
        decoder, at = json.JSONDecoder(), text.find("{")
        while at != -1:
            try:
                obj, end = decoder.raw_decode(text, at)
            except ValueError:
                at = text.find("{", at + 1)
                continue
            found.append(obj)
            at = text.find("{", end)
    return [o for o in found if isinstance(o, dict)]


def write_blocks(folder: Path, ids: Sequence[str], text: str) -> list[str]:
    """Write ``out/<id>.json`` for each object in ``text`` whose ``id`` is one of ``ids``; return
    those ids. Idempotent: a file already written is kept."""

    written = []
    for obj in answer_objects(text):
        out = Path(folder) / "out" / f"{obj.get('id')}.json"
        if obj.get("id") in ids and not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(out, json.dumps(obj, indent=1) + "\n")
            written.append(str(obj["id"]))
    return written


def _unpack_blocks(lease: Mapping[str, Any], text: str) -> list[str]:
    """A ``blocks`` answer: the items' ``out/`` files, then the result file, one line per item."""

    folder, ids = Path(lease["brief"]).parent.resolve(), lease["ids"]
    write_blocks(folder, ids, text)
    lines = []
    for item in ids:
        out = folder / "out" / f"{item}.json"
        verdict = json.loads(out.read_text(encoding="utf-8")).get("verdict") \
            if out.is_file() else None
        lines.append(f"{item} {verdict or 'missing'}\n")
    _write_atomic(Path(lease["result"]), "".join(lines))
    outs = [folder / "out" / f"{i}.json" for i in ids]
    return [*(str(p) for p in outs if p.is_file()), str(lease["result"])]


def unpack(lease: Mapping[str, Any]) -> list[str]:
    """Write the files a read-only lane's answer holds.

    A ``files`` answer (``{"files": {path: content}}``): only paths inside the brief's folder are
    written; the result file beside the brief is written last, from the answer when it holds one,
    else as one line per file written. A ``blocks`` answer (``lease["answer_kind"]``): one
    ``out/<id>.json`` per answered item. Returns: the paths written. Raises ``RecordError`` when
    a ``files`` answer holds no such object.
    """

    folder = Path(lease["brief"]).parent.resolve()
    text = Path(lease["answer"]).read_text(encoding="utf-8")
    if lease.get("answer_kind") == "blocks":
        return _unpack_blocks(lease, text)
    match = AGY_ANSWER.search(text)
    files = (json.loads(match.group(0)).get("files") if match else None) or {}
    if not isinstance(files, dict) or not files:
        raise RecordError(f"{lease['answer']}: no {{\"files\": ...}} object in the answer")
    result = Path(lease["result"])
    written = []
    for name, content in files.items():
        path = Path(name).resolve()
        if path == result or not path.is_relative_to(folder):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        body = content if isinstance(content, str) else json.dumps(content, indent=1) + "\n"
        _write_atomic(path, body)
        written.append(str(path))
    given = files.get(str(result))
    _write_atomic(result, given if isinstance(given, str) else "".join(
        f"wrote {Path(p).name}\n" for p in written))
    return [*written, str(result)]


# --- the refusal rules: first admitted role, else a queued step -------------------------------


def other_role(role: str) -> str | None:
    """The other main lab's lab-agnostic role, or ``None`` for any other role."""

    labs = list(MAIN_ROLES)
    return labs[1 - labs.index(role)] if role in labs else None


def _queue_path(label: str) -> Path:
    return work_dir("queue") / f"{label}.json"


def launch_first(kind: str, roles: Sequence[str], brief: Path, *, keys: Sequence[str] = (),
                 parent: str | None = None, label: str | None = None, web: bool = False,
                 network_ok: Sequence[str] = (), queue: bool = True,
                 on_start: Sequence[Mapping[str, Any]] = (), lab: str | None = None) -> str:
    """Start the lane on the first of ``roles`` that is admitted; queue it when none is.

    ``network_ok`` names the roles a 2b lane may take although they have network (the Google
    seat, last). ``lab`` ties the step to one lab: a role routed to another lab is refused
    (``lab_mismatch``) like an unadmitted one, so the step queues for its lab's next window.
    A ``lease_held`` refusal is raised at once: another lane holds the work.
    ``on_start`` lists events (``{"family", "verb", ...fields}``) logged on the lane once it
    starts, such as a family's ``tier``; a queued step keeps them.
    Returns: the run id, or ``queued:<label>`` when every role refused (with ``queue``; without
    it the last ``Refused`` is raised). Side effects: ``launch``'s, plus a ``refused`` event per
    refusal and, when queued, ``queue/<label>.json`` naming the trigger (any of the roles
    admitted) for ``requeue``.
    """

    brief = Path(brief).resolve(strict=True)
    label = label or f"{kind}-{brief.parent.name}"
    step = {"kind": kind, "roles": list(roles), "brief": str(brief), "keys": keys,
            "parent": parent, "label": label, "web": web, "network_ok": list(network_ok),
            "on_start": [dict(item) for item in on_start], "lab": lab}
    run, refusals = _first_admitted(step)
    if run is not None:
        _queue_path(label).unlink(missing_ok=True)
        return run
    if not queue:
        raise Refused("no_admission", "; ".join(r["why"] for r in refusals), roles=list(roles))
    _queue(step, refusals)
    return f"queued:{label}"


def _first_admitted(step: Mapping[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    """Launch on the step's first admitted role; its run id (or ``None``) and the refusals."""

    from menagerie.pipeline import drive

    refusals: list[dict[str, Any]] = []
    for role in step["roles"]:
        try:
            run = drive.launch(step["kind"], role, Path(step["brief"]), keys=step["keys"],
                               parent=step["parent"], label=step["label"], web=step["web"],
                               network_ok=role in step["network_ok"], lab=step.get("lab"))
        except Refused as refused:
            if refused.result["result"] not in ("no_admission", "lab_mismatch"):
                raise
            refusals.append({**refused.result, "role": role})
            continue
        for refusal in refusals:
            drive.event(run, "", "refused", **refusal)
        for item in step["on_start"]:
            fields = {k: v for k, v in item.items() if k not in ("family", "verb")}
            drive.event(run, item["family"], item["verb"], **fields)
        return run, refusals
    return None, refusals


def _queue(step: Mapping[str, Any], refusals: Sequence[Mapping[str, Any]]) -> None:
    """Write the queued step; a requeue keeps the step's place."""

    path = _queue_path(step["label"])
    path.parent.mkdir(parents=True, exist_ok=True)
    queued_at = json.loads(path.read_text(encoding="utf-8"))["queued"] if path.is_file() else now()
    record = {**step, "queued": queued_at, "refusals": list(refusals),
              "trigger": f"any of {', '.join(step['roles'])} admitted"}
    _write_atomic(path, json.dumps(record, indent=1) + "\n")


def queued() -> list[dict[str, Any]]:
    """The queued steps, oldest first."""

    folder = work_dir("queue")
    steps = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")] \
        if folder.is_dir() else []
    return sorted(steps, key=lambda s: s["queued"])


def requeue() -> list[dict[str, Any]]:
    """Re-read admission for every queued step and start the ones now admitted.

    The run's watcher calls this every 30 minutes. Returns: one ``{"label", "run"}`` per step,
    ``run`` a run id, ``queued:<label>`` (still refused) or ``held: <why>`` (a lease holds its
    work; it stays queued). Side effects: the started lanes; their queue files removed.
    """

    out = []
    for step in queued():
        try:
            run = launch_first(step["kind"], step["roles"], Path(step["brief"]),
                               keys=step["keys"], parent=step["parent"], label=step["label"],
                               web=step["web"], network_ok=step["network_ok"],
                               on_start=step.get("on_start", ()), lab=step.get("lab"))
        except Refused as refused:
            run = f"held: {refused.result['why']}"
        except (OSError, RecordError) as exc:
            run = f"error: {exc}"[:300]
        out.append({"label": step["label"], "run": run})
    return out
