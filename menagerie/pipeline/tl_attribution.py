"""Whether a failed trace is TorchLens's: attribution from evidence, never from wording alone.

A row is attributed to TorchLens only on positive evidence, strongest first:

- ``frame``: the raising exception's innermost frame (``LedgerRow.frames``, recorded by the
  worker since 2026-11-05) is inside the ``torchlens`` package;
- ``validation_status``: the class came from TorchLens's own validation result
  (``torchlens_gap:replay`` from ``validation_replay_status``, ``torchlens_unverified``, a
  precondition refusal), or a typed ``torchlens_gap:*`` give-up;
- ``raise_site``: the message's literal text is a string in the TorchLens source at the row's
  pin (``git grep`` of the tag ``v<pin>``) with a ``raise`` statement within four lines (copies
  of torch's messages that TorchLens only matches against do not count);
- ``exception_type``: the exception's type is a class the TorchLens source at that pin defines.

``menagerie`` is the Menagerie worker's own refusal (``capture_error:backend``, a frame in
``menagerie/pipeline``). Everything else is ``unattributed``: the capture raised, the plain
forward passed, and nothing proves which side raised it.
"""

from __future__ import annotations

import builtins
import json
import os
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

TORCHLENS = "torchlens"
MENAGERIE = "menagerie"
UNATTRIBUTED = "unattributed"
REFUSAL = "validation refused before capture"
TL_CLASSES = ("torchlens_gap", "torchlens_unverified")
_TYPE = re.compile(r"^([A-Za-z_][\w.]*)(?::|$)")
_CLASS_DEF = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)")
# Formatted parts of a message (numbers, quoted values, bracketed lists) split it into the
# literal pieces a raise statement can hold verbatim.
_FORMATTED = re.compile(r":\s|\d+|'[^']*'|\"[^\"]*\"|\([^)]*\)|\[[^\]]*\]|[{}<>=`]|\s{2,}")
MIN_PIECE = 24
PIECE = 48
GIT_S = 30


def source_root() -> Path:
    """The TorchLens git checkout whose release tags are read (``MENAGERIE_TORCHLENS_SRC``)."""

    given = os.environ.get("MENAGERIE_TORCHLENS_SRC")
    return Path(given).expanduser() if given else Path.home() / "projects" / "torchlens"


def exception_type(msg: str | None) -> str | None:
    """The exception type name a worker message starts with (``_first_line``), else None."""

    found = _TYPE.match(msg or "")
    return found[1].rsplit(".", 1)[-1] if found else None


def pieces(msg: str) -> list[str]:
    """Literal fragments of a message's text worth searching for in the source."""

    text = msg.split(": ", 1)[1] if exception_type(msg) and ": " in msg else msg
    out = []
    for part in _FORMATTED.split(text):
        part = part.strip(" .,;:-")
        if len(part) < MIN_PIECE or len(part.split()) < 3:
            continue
        out.append(part[:PIECE].rsplit(" ", 1)[0] if len(part) > PIECE else part)
        if len(part) > PIECE:
            out.append(part[-PIECE:].split(" ", 1)[-1])
    return list(dict.fromkeys(p for p in out if len(p) >= MIN_PIECE))


def _in_string(line: str, piece: str) -> bool:
    """A source line holds ``piece`` inside a string literal, not a comment or docstring prose."""

    at = line.find(piece)
    return at >= 0 and not line.lstrip().startswith("#") and any(q in line[:at] for q in "'\"")


_HIT = re.compile(r"^[^:]+:(.+?\.pyi?)([:-])(\d+)[:-](.*)$")  # git grep: tag:path:n:text
_RAISE = re.compile(r"^\s*raise\b|[^#]*\braise\s+[A-Za-z_]")
SITES = 3  # cache format of the raise-site lookups: a change of the rule makes a new file
WINDOW = 4  # lines around a matched string searched for the raise statement that uses it


def _raised_at(block: str, wanted: Sequence[str]) -> str | None:
    """``path:line`` of a matched string in one ``git grep`` block that a ``raise`` within
    ``WINDOW`` lines uses; strings TorchLens only compares with or parses (copies of torch's
    messages in pattern tables) are not raise sites."""

    lines = [m for m in map(_HIT.match, block.splitlines()) if m]
    for i, hit in enumerate(lines):
        if hit[2] != ":" or not any(_in_string(hit[4], p) for p in wanted):
            continue
        near = lines[max(0, i - WINDOW):i + WINDOW + 1]
        if any(_RAISE.match(n[4]) for n in near if n[1] == hit[1]):
            return f"{hit[1]}:{hit[3]}"
    return None


def _exception_classes(bases: dict[str, set[str]]) -> list[str]:
    """The classes (name to base names) that derive from a builtin exception, transitively."""

    known = {n for n in dir(builtins) if isinstance(getattr(builtins, n), type)
             and issubclass(getattr(builtins, n), BaseException)}
    found: set[str] = set()
    while more := {n for n, b in bases.items() if n not in found and b & (known | found)}:
        found |= more
    return sorted(found - known)


@dataclass
class SourceIndex:
    """TorchLens's exception classes and raise sites per release, cached under ``cache``.

    ``git`` runs ``git -C root ...`` and returns stdout (tests pass a fake). A release whose tag
    is missing has no index: its rows stay unattributed rather than guessed.
    """

    root: Path
    cache: Path | None = None
    deadline: float | None = None
    git: object = None
    _types: dict[str, set[str] | None] = field(default_factory=dict)
    _sites: dict[str, dict[str, str | None]] = field(default_factory=dict)
    _tags: dict[str, str | None] = field(default_factory=dict)

    def _run(self, *args: str) -> str | None:
        if self.git is not None:
            return self.git(*args)  # type: ignore[operator]
        done = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                              text=True, check=False, timeout=GIT_S)
        if done.returncode == 1 and not done.stderr:
            return ""  # grep found nothing
        return done.stdout if done.returncode == 0 else None

    def _tag(self, pin: str) -> str | None:
        if pin not in self._tags:
            tag = f"v{pin}"
            found = self._run("rev-parse", "--verify", "-q", f"{tag}^{{commit}}")
            self._tags[pin] = tag if found else None
        return self._tags[pin]

    def types(self, pin: str) -> set[str] | None:
        """Names of the exception classes the TorchLens source at ``pin`` defines."""

        if pin not in self._types:
            self._types[pin] = self._cached(f"types-{pin}.json", lambda: self._read_types(pin))
        found = self._types[pin]
        return set(found) if found is not None else None

    def _read_types(self, pin: str) -> list[str] | None:
        tag = self._tag(pin)
        out = self._run("grep", "-h", "-E", r"^\s*class\s+\w+\s*\(", tag, "--", "torchlens/") \
            if tag else None
        if out is None:
            return None
        bases = {m[1]: {b.strip().rsplit(".", 1)[-1] for b in m[2].split(",")}
                 for m in map(_CLASS_DEF.match, out.splitlines()) if m}
        return _exception_classes(bases)

    def raise_site(self, pin: str, msg: str) -> str | None:
        """``path:line`` of a TorchLens string at ``pin`` holding a literal piece of ``msg``."""

        sites = self._sites.setdefault(pin, self._load(f"sites{SITES}-{pin}.json") or {})
        key = msg[:400]
        if key not in sites:
            if self.deadline is not None and time.monotonic() > self.deadline:
                self._save(f"sites{SITES}-{pin}.json", sites)
                raise TimeoutError("the TorchLens raise-site lookup ran out of time")
            sites[key] = self._grep(pin, pieces(msg))
        return sites[key]

    def _grep(self, pin: str, wanted: Sequence[str]) -> str | None:
        tag = self._tag(pin) if wanted else None
        if tag is None:
            return None
        args = ["grep", "-n", "-F", f"-B{WINDOW}", f"-A{WINDOW}"]
        for piece in wanted:
            args += ["-e", piece]
        out = self._run(*args, tag, "--", "torchlens/") or ""
        return next((site for block in out.split("\n--\n") if (site := _raised_at(block, wanted))),
                    None)

    def save(self) -> None:
        for pin, sites in self._sites.items():
            self._save(f"sites{SITES}-{pin}.json", sites)

    def _cached(self, name: str, make: object) -> list[str] | None:
        found = self._load(name)
        if isinstance(found, dict) and "value" in found:
            return found["value"]
        value = make()  # type: ignore[operator]
        if value is not None:
            self._save(name, {"value": value})
        return value

    def _load(self, name: str) -> dict | None:  # type: ignore[type-arg]
        path = self.cache / name if self.cache else None
        return json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else None

    def _save(self, name: str, value: object) -> None:
        if self.cache is None:
            return
        self.cache.mkdir(parents=True, exist_ok=True)
        tmp = self.cache / f".{name}.tmp"
        tmp.write_text(json.dumps(value, indent=0, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.cache / name)


@dataclass(frozen=True)
class Attribution:
    """Who raised a failure, which kind of evidence says so, and that evidence in words."""

    who: str
    kind: str
    evidence: str


def _package(frame: str) -> str:
    return frame.split("/", 1)[0].split(":", 1)[0]


def from_frames(frames: Sequence[str]) -> Attribution | None:
    """Attribution from the raising frames the worker recorded (innermost last), if decisive."""

    if not frames:
        return None
    inner = frames[-1]
    if _package(inner) == TORCHLENS:
        return Attribution(TORCHLENS, "frame", f"raised at {inner}")
    if inner.startswith("menagerie/pipeline/"):
        return Attribution(MENAGERIE, "frame", f"raised by the Menagerie worker at {inner}")
    caller = next((f for f in reversed(frames[:-1]) if _package(f) != _package(inner)), None)
    via = f", called from {caller}" if caller else ""
    return Attribution(UNATTRIBUTED, "frame", f"raised at {inner}{via}")


def _from_source(msg: str, pin: str, index: SourceIndex) -> Attribution | None:
    """TorchLens's attribution from its source at ``pin``: a raise site, else its exception type."""

    site = index.raise_site(pin, msg)
    if site:
        return Attribution(TORCHLENS, "raise_site",
                           f"message text is a string at {site} in TorchLens {pin}")
    kind, types = exception_type(msg), index.types(pin)
    if kind and types and kind in types:
        return Attribution(TORCHLENS, "exception_type",
                           f"{kind} is an exception class TorchLens {pin} defines")
    return None


def attribute(cls: str, msg: str | None, pin: str | None, frames: Sequence[str] | None,
              index: SourceIndex | None, *, state: str | None = None) -> Attribution:
    """Attribute one failed trace or give-up (``cls`` its failure class, ``pin`` its TorchLens)."""

    head = cls.split(":", 1)[0]
    msg = msg or ""
    if head in TL_CLASSES:
        what = f"validation_state {state}" if state else f"class {cls}"
        return Attribution(TORCHLENS, "validation_status", f"{what} from TorchLens validation")
    if cls == "capture_error:backend":
        return Attribution(MENAGERIE, "class", "the Menagerie worker refused the backend")
    by_frame = from_frames(frames or ())
    if by_frame is not None and by_frame.who != UNATTRIBUTED:
        return by_frame
    if msg.startswith(REFUSAL):
        return Attribution(TORCHLENS, "validation_status",
                           "TorchLens validation refused before capture (precondition_refusal)")
    by_source = _from_source(msg, pin, index) if index is not None and pin else None
    if by_source is not None:
        return by_source
    if by_frame is not None:
        return by_frame
    return Attribution(UNATTRIBUTED, "none", "the plain forward passed and the capture raised; "
                       "no frame or source evidence names the side that raised it")
