"""Hub side of a 2a research lane on a worker runtime (kind ``2a``, label ``2a-mini-``).

A Claude-lab 2a lane may run on a worker through the 2b lanes' machinery (``mini_placement``,
``mini_relay``, ``mini_lane``). The worker never reaches the hub: the relay stages the translated
brief, answers the lane's verbs and pulls the result. Only the 2a prompt's verbs run, with their
documented options and this lane's families: ``family seed``, ``fetch page``, ``fetch paper``,
``fetch code``, ``fetch hf``, ``research second-search``, ``packet write`` (its draft inside the
lane's drafts folder, brought in through the scanned snapshot as a 2b draft is), and ``gh search
repos|code`` with read-only options. Metered ``web-search`` is not relayed: tick 2a lanes have no
allowance for it. Each verb runs on the hub exactly as the hub lane's brief runs it (the lease's
``pipe``, the pipeline command of ``research._pipeline_command``), with the lane's run id, so
fetch events and the packet's author are the lane's. Every store blob a fetch prints is copied to
the worker's ``.mini-lane/store/``, which the brief names in place of the hub store, so the agent
reads and quotes saved text as a hub lane does. The agent's own web search tools stay on.

Off unless the tick config's ``mini_2a`` is true (``enabled``). Imports neither torch nor psutil.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir

KIND = "2a"
LABEL = "2a-mini-"
# verb words: (positional count, or (least, most); {option: takes a value}). Nothing else runs.
SCHEMA: dict[tuple[str, str], tuple[int | tuple[int, int], dict[str, bool]]] = {
    ("family", "seed"): ((1, 50), {}),
    ("fetch", "page"): (1, {"--family": True}),
    ("fetch", "paper"): (1, {}),
    ("fetch", "code"): (1, {"--rev": True, "--family": True}),
    ("fetch", "hf"): (1, {"--rev": True, "--family": True}),
    ("research", "second-search"): (1, {}),
    ("packet", "write"): (2, {}),
}
FETCHES = {("fetch", "page"), ("fetch", "paper"), ("fetch", "code"), ("fetch", "hf")}
NEEDS_FAMILY = {("fetch", "page"), ("fetch", "code"), ("fetch", "hf")}
GH_VERB = "gh"  # the lane's ``gh search`` requests
GH = "gh"  # the hub's GitHub CLI that answers them, signed in for the hub's own 2a lanes
GH_KINDS = ("repos", "code")
GH_OPTIONS = {"--limit": True, "-L": True, "--language": True, "--owner": True, "--repo": True,
              "-R": True, "--filename": True, "--extension": True, "--match": True, "--sort": True,
              "--order": True, "--json": True, "--topic": True, "--stars": True, "--size": True,
              "--updated": True, "--archived": False, "--include-forks": True}
GH_TIMEOUT_S = 120
ROW = re.compile(r"m[0-9]{1,7}")
WORD_MAX = 2000
SHA = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
BLOB_MAX = 64 * 1024 * 1024  # one blob copied to the worker
PUSH_MAX = 256 * 1024 * 1024  # all blobs one verb copies
PUSH_COUNT = 60


def enabled(cfg: Mapping[str, Any]) -> bool:
    """Whether 2a lanes may run on workers: ``mini_2a`` true in the tick config (default off)."""

    return cfg.get("mini_2a") is True


def folder_of(brief: Path) -> Path:
    """A 2a lane's hub folder: ``briefs/<label>/`` under the work root."""

    folder = Path(brief).parent
    if folder.parent != work_dir("briefs"):
        raise ValueError(f"2a brief outside the briefs work root: {brief}")
    return folder


def mapping(root: str, brief: Path, pipe: str, store: str | None) -> dict[str, str]:
    """Hub text to worker text for the brief and answers: the lane folder, the verb command (as
    the lane's wrapper) and the store root (as the worker's copy)."""

    folder = folder_of(brief)
    worker = f"{root}/work/{folder.name}"
    mapped = {str(folder): worker, pipe: f"{worker}/menagerie"}
    if store:
        mapped[str(Path(store))] = f"{root}/store"
    return mapped


@dataclass(frozen=True)
class Plan:
    """An authorized 2a request: the hub argv, whether it brings the drafts in, and its kind."""

    words: tuple[str, ...]
    argv: list[str]
    draft: str | None = None  # packet write's confined draft path

    @property
    def gh(self) -> bool:
        return self.words[0] == GH_VERB

    @property
    def fetch(self) -> bool:
        return self.words in FETCHES


def _plain(word: str) -> str:
    if not word or len(word) > WORD_MAX or word.startswith("-") or any(c in word for c in "\0\n\r"):
        raise ValueError(f"not a plain argument: {word[:80]!r}")
    return word


def _split(args: Sequence[str], allowed: Mapping[str, bool], what: str
           ) -> tuple[list[str], dict[str, str]]:
    positional: list[str] = []
    options: dict[str, str] = {}
    words = list(args)
    while words:
        word = words.pop(0)
        if not word.startswith("-"):
            positional.append(_plain(word))
            continue
        name, value = _option(word, words, allowed, what)
        if name in options:
            raise ValueError(f"{name} given twice")
        options[name] = value
    return positional, options


def _option(word: str, words: list[str], allowed: Mapping[str, bool], what: str) -> tuple[str, str]:
    """One option and its value (taken from ``words`` when not given with ``=``)."""
    name, has_value, value = word.partition("=")
    if name not in allowed:
        raise ValueError(f"{what} does not take {name} here")
    if not allowed[name]:
        if has_value:
            raise ValueError(f"{name} takes no value")
        return name, ""
    if not has_value:
        if not words:
            raise ValueError(f"{name} needs a value")
        value = words.pop(0)
    return name, _plain(value)


def _argv(words: Sequence[str], positional: Sequence[str], options: Mapping[str, str],
          allowed: Mapping[str, bool]) -> list[str]:
    argv = [*words, *positional]
    for name, value in options.items():
        argv += [name, value] if allowed[name] else [name]
    return argv


def authorize(args: Sequence[str], families: Sequence[str], drafts: Path, base: str) -> Plan:
    """The request's hub argv, refused unless it is one of the 2a verbs above with its own options
    and this lane's families; ``args`` has worker paths translated to hub paths already, and
    ``base`` resolves a relative draft path."""

    words = tuple(args[:2])
    if words and words[0] == GH_VERB:
        return _gh(args)
    if words not in SCHEMA:
        names = ", ".join(" ".join(w) for w in SCHEMA)
        raise ValueError(f"verbs here: {names}, gh search repos|code")
    count, allowed = SCHEMA[words]
    positional, options = _split(args[2:], allowed, " ".join(words))
    least, most = count if isinstance(count, tuple) else (count, count)
    if not least <= len(positional) <= most:
        span = least if least == most else f"{least} to {most}"
        raise ValueError(f"{' '.join(words)} takes {span} argument(s) here")
    _targets(words, positional, options, families)
    draft = None
    if words == ("packet", "write"):
        draft = _draft(positional[1], drafts, base)
        positional = [positional[0], draft]
    return Plan(words, _argv(words, positional, options, allowed), draft)


def _targets(words: tuple[str, ...], positional: Sequence[str], options: Mapping[str, str],
             families: Sequence[str]) -> None:
    """The family a verb names is this lane's; ``family seed`` names roster ids."""
    family = options.get("--family") or (positional[0] if words in (
        ("research", "second-search"), ("packet", "write")) else None)
    if words in NEEDS_FAMILY and "--family" not in options:
        raise ValueError(f"{' '.join(words)} needs --family F")
    if family is not None and family not in families:
        raise ValueError(f"{family} is not one of this lane's families: {', '.join(families)}")
    if words == ("family", "seed") and any(not ROW.fullmatch(p) for p in positional):
        raise ValueError("family seed takes roster ids (m123)")


def _draft(path: str, drafts: Path, base: str) -> str:
    """A packet draft's real path, refused unless a ``.json`` file inside the drafts folder."""
    real = os.path.realpath(os.path.join(base, path))
    if not real.startswith(os.path.realpath(drafts) + "/") or not real.endswith(".json"):
        raise ValueError(f"the draft must be a .json file in this lane's drafts folder: {path}")
    return real


def _gh(args: Sequence[str]) -> Plan:
    if len(args) < 3 or args[1] != "search" or args[2] not in GH_KINDS:
        raise ValueError("gh here: gh search repos|code QUERY... [options]")
    positional, options = _split(args[3:], GH_OPTIONS, f"gh search {args[2]}")
    if not positional:
        raise ValueError("gh search needs a query")
    words = (GH_VERB, "search", args[2])
    return Plan(words, _argv(words, positional, options, GH_OPTIONS))


def run(plan: Plan, lease: Mapping[str, Any], cwd: Path, timeout: float) -> dict[str, Any]:
    """The verb on the hub: a 2a verb through the lease's pipeline command, ``gh`` directly, from
    a trusted empty folder, with the lane's run id."""

    env = {k: v for k, v in os.environ.items() if k not in ("FLEET_RUN_ID", "PYTHONSAFEPATH")}
    env.update(FLEET_RUN_ID=lease["run"], MENAGERIE_LANE=lease["run"])
    if plan.gh:
        argv, timeout = [GH, *plan.argv[1:]], min(timeout, GH_TIMEOUT_S)
    else:
        argv = ["bash", "-c", f"{lease['pipe']} {shlex.join(plan.argv)}"]
    cwd.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          env=env, stdin=subprocess.DEVNULL, start_new_session=True) as child:
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)  # the verb and everything it started
            child.communicate()
            return {"rc": 124, "stdout": "", "stderr": "infra: verb timed out on the hub\n"}
    return {"rc": child.returncode, "stdout": stdout, "stderr": stderr}


def blobs(text: str, store: str | None) -> list[str]:
    """Store-relative paths of the blobs a fetch printed that the worker should get: present in
    the hub store, each at most ``BLOB_MAX``, together at most ``PUSH_MAX``."""

    if not store:
        return []
    out, total = [], 0
    for sha in dict.fromkeys(SHA.findall(text)):
        rel = f"sha256/{sha[:2]}/{sha}"
        try:
            size = (Path(store) / rel).stat().st_size
        except OSError:
            continue
        if size > BLOB_MAX or total + size > PUSH_MAX or len(out) >= PUSH_COUNT:
            continue
        out.append(rel)
        total += size
    return out
