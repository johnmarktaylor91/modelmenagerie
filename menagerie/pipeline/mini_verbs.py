"""What the hub runs for a mini lane: the verb schema, its targets and its paths.

The mini and its agent are less trusted than the hub, so a request runs only when it is one
of the 2b prompt's verbs with its documented options, names only the lane's families and their
packet members, and every path it gives (and a recipe's adapter path) resolves, symlinks
followed, inside the lane's hub folders. Paths are made absolute here so the verb runs from a
trusted empty folder (``mini_relay``). Imports neither torch nor psutil.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, read_json

# verb: (positional count, {option: takes a value}); nothing else is accepted.
SCHEMA: dict[str, tuple[int, dict[str, bool]]] = {
    "src": (1, {}),
    "defer": (2, {}),
    "try": (1, {"--recipe": True, "--member": True, "--with": True}),
    "giveup": (3, {"--recipe": True}),
    "done": (1, {"--rewrite-family": False}),
}
REPEATABLE = {"--with"}
CLASS = re.compile(r"[a-z][a-z0-9_]{0,63}")
PACKAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-\[\],=<>!~]{0,199}")


@dataclass(frozen=True)
class Scope:
    """What one lane may name on the hub."""

    families: tuple[str, ...]
    family_of: Mapping[str, str]  # packet member to its family
    folders: tuple[str, ...]  # real paths of the lane's hub draft folders, families first
    adapters: tuple[str, ...]  # folders a recipe's adapter may be read from

    @property
    def members(self) -> frozenset[str]:
        return frozenset(self.family_of)

    def folder(self, family: str) -> str:
        return self.folders[self.families.index(family)]


def members_of(family: str, data: DataLayout) -> set[str]:
    from menagerie.pipeline.records import Packet

    return {m.id for m in read_json(Packet, data.packet(family)).members}


def scope_for(families: Sequence[str], brief: Path, data: DataLayout) -> Scope:
    folders = [os.path.realpath(work_dir("2b") / f) for f in families]
    folders.append(os.path.realpath(Path(brief).parent))
    family_of = {m: f for f in families for m in members_of(f, data)}
    return Scope(tuple(families), family_of, tuple(dict.fromkeys(folders)),
                 (*dict.fromkeys(folders), os.path.realpath(data.root / "adapters")))


def parse(args: Sequence[str]) -> tuple[str, list[str], dict[str, list[str]]]:
    """Split argv by the verb's schema; any other option or count is refused."""
    if not args or args[0] not in SCHEMA:
        raise ValueError(f"verbs here: {', '.join(SCHEMA)}")
    verb, (count, allowed) = args[0], SCHEMA[args[0]]
    positional: list[str] = []
    options: dict[str, list[str]] = {}
    words = list(args[1:])
    while words:
        word = words.pop(0)
        if not word.startswith("--"):
            positional.append(word)
            continue
        name, value = _option(verb, word, words, allowed)
        if name in options and name not in REPEATABLE:
            raise ValueError(f"{name} given twice")
        options.setdefault(name, []).append(value)
    if len(positional) != count:
        raise ValueError(f"{verb} takes {count} argument{'s' if count > 1 else ''} here")
    return verb, positional, options


def _option(verb: str, word: str, words: list[str], allowed: Mapping[str, bool]) -> tuple[str, str]:
    """One option and its value (taken from ``words`` when not given with ``=``)."""
    name, has_value, value = word.partition("=")
    if name not in allowed:
        raise ValueError(f"{verb} does not take {name} here")
    if allowed[name] and not has_value:
        if not words:
            raise ValueError(f"{name} needs a value")
        return name, words.pop(0)
    if not allowed[name] and has_value:
        raise ValueError(f"{name} takes no value")
    return name, value


def _inside(path: str, roots: Sequence[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)


def confined(path: str, base: str, roots: Sequence[str]) -> str:
    """The real path of ``path`` (relative to ``base``), refused outside ``roots``."""
    real = os.path.realpath(os.path.join(base, path))
    if not _inside(real, roots):
        raise ValueError(f"path outside this lane's folders: {path}")
    return real


def check_recipe(path: str, scope: Scope, family: str, *, done: bool = False) -> None:
    """A draft's adapter (named, or the family's default as ``implement.draft`` reads it) must be
    in the family's folder or the committed adapters; for ``done``, the passing try it would
    accept and every variant it would run must belong to ``family``, the one locked."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return  # Missing or malformed: the canonical verb reports it.
    adapter = _adapter(raw, path, scope, family) if isinstance(raw, dict) else None
    if done:
        _done_targets(path, adapter, scope, family)


def _adapter(raw: Mapping[str, Any], path: str, scope: Scope, family: str) -> str | None:
    if raw.get("kind") != "adapter":
        return None
    named = raw.get("adapter")
    if named is not None and not isinstance(named, str):
        raise ValueError("a recipe's adapter is a path")
    roots = (scope.folder(family), scope.adapters[-1])
    return confined(named or f"adapters/{family}.py", os.path.dirname(path), roots)


def _done_targets(path: str, adapter: str | None, scope: Scope, family: str) -> None:
    from menagerie.pipeline import implement
    from menagerie.pipeline.records import RecordError

    if adapter and os.path.isfile(adapter):
        source = Path(adapter).read_text(encoding="utf-8")
        outside = sorted(m for m in implement.variants(source) if scope.family_of.get(m) != family)
        if outside:
            raise ValueError(f"VARIANTS names models outside {family}: {', '.join(outside[:5])}")
    try:
        attempt, _ = implement._passing(Path(path))
    except (RecordError, OSError, ValueError):
        return  # No passing try: the canonical verb says so.
    if attempt.family != family or scope.family_of.get(attempt.member) != family:
        raise ValueError(f"done would accept a try outside {family}")


@dataclass(frozen=True)
class Plan:
    """An authorized request: the hub argv, the family whose drafts it uses, recipes to check."""

    verb: str
    argv: list[str]
    family: str
    recipes: tuple[str, ...]


def authorize(args: Sequence[str], scope: Scope, base: str) -> Plan:
    """The verb, its target family and its hub argv, with every path absolute and confined.

    ``args`` already has worker paths translated to hub paths; ``base`` resolves relative ones.
    """
    verb, positional, options = parse(args)
    _targets(verb, positional, options, scope)
    family, recipes = _family(verb, positional, scope, base)
    if "--recipe" in options:
        # Only the target family's drafts are brought in and locked, so only they may be named.
        options["--recipe"] = [confined(options["--recipe"][0], base, [scope.folder(family)])]
        recipes.append(options["--recipe"][0])
    argv = [verb, *positional]
    for name, values in options.items():
        argv += [name] if not SCHEMA[verb][1][name] else [w for v in values for w in (name, v)]
    return Plan(verb, argv, family, tuple(recipes))


def _family(verb: str, positional: list[str], scope: Scope, base: str) -> tuple[str, list[str]]:
    """The verb's target family and, for ``done``, its confined recipe (``positional`` is
    updated in place to the confined path)."""
    if verb != "done":
        return (positional[0] if verb in ("src", "defer") else scope.family_of[positional[0]]), []
    # done's family is the folder its recipe is in; every family folder is a candidate.
    positional[0] = confined(positional[0], base, scope.folders[:len(scope.families)])
    return next(f for f in scope.families if _inside(positional[0], [scope.folder(f)])), [positional[0]]


def check_recipes(plan: Plan, scope: Scope) -> None:
    """After the drafts are in: adapters are the family's, ``done`` accepts only this lane."""
    for path in plan.recipes:
        check_recipe(path, scope, plan.family, done=plan.verb == "done")


def _targets(verb: str, positional: Sequence[str], options: Mapping[str, Sequence[str]],
             scope: Scope) -> None:
    if verb in ("src", "defer") and positional[0] not in scope.families:
        raise ValueError(f"{positional[0]} is not one of this lane's families")
    if verb in ("try", "giveup") and positional[0] not in scope.members:
        raise ValueError(f"{positional[0]} is not a member of this lane's families")
    if verb == "giveup" and not CLASS.fullmatch(positional[1]):
        raise ValueError("a give-up class is a lowercase word")
    if any(m not in scope.members for m in options.get("--member", ())):
        raise ValueError("--member must be a member of this lane's families")
    if any(not PACKAGE.fullmatch(p) for p in options.get("--with", ())):
        raise ValueError("--with takes a package name")


def base_folder(cwd: Any, family: str, scope: Scope, mapping: Mapping[str, str]) -> str:
    """Relative paths resolve from the agent's folder when it is one of the lane's, else the
    family's draft folder."""
    from menagerie.pipeline.mini_relay import inverse, translate

    folder = os.path.realpath(work_dir("2b") / family)
    if not isinstance(cwd, str) or not cwd.startswith("/"):
        return folder
    hub = os.path.realpath(translate(cwd, inverse(mapping)))
    return hub if _inside(hub, scope.folders) and os.path.isdir(hub) else folder
