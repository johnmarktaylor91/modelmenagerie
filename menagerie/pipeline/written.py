"""Which data files this process appended to or rewrote while a ``recording()`` is open.

The tick commits only what its own steps wrote (``tick_commit``). The shared writers report each
write here: ``append_jsonl`` and stage 3's ``_append_new`` the exact lines they appended,
``write_json`` the file's content hash before its first write and after its last. Outside a
recording each report costs one empty-list check and reads nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

UNSEEN = object()  # ``peek`` outside a recording: nothing to report
BROKEN = "changed-between-writes"  # a file someone else changed between two of this process's writes


@dataclass
class Writes:
    """One recording: absolute path to appended lines (each ending in a newline), and absolute
    path to ``[sha256 before the first write or None when absent, sha256 after the last]``; the
    first item is ``BROKEN`` when the file changed between two recorded writes."""

    lines: dict[str, list[str]] = field(default_factory=dict)
    files: dict[str, list[str | None]] = field(default_factory=dict)


_ACTIVE: list[Writes] = []
_LOCK = threading.Lock()


@contextlib.contextmanager
def recording() -> Iterator[Writes]:
    """Record every reported write until the block ends; recordings may nest."""

    rec = Writes()
    with _LOCK:
        _ACTIVE.append(rec)
    try:
        yield rec
    finally:
        with _LOCK:
            _ACTIVE.remove(rec)


def digest(body: bytes | None) -> str | None:
    return hashlib.sha256(body).hexdigest() if body is not None else None


def _key(path: Path | str) -> str:
    return str(Path(path).resolve())


def appended(path: Path | str, lines: Sequence[str]) -> None:
    """Report lines appended to ``path`` (each with its trailing newline)."""

    if not _ACTIVE or not lines:
        return
    key = _key(path)
    with _LOCK:
        for rec in _ACTIVE:
            rec.lines.setdefault(key, []).extend(lines)


def peek(path: Path | str) -> object:
    """Before rewriting ``path``: its current content hash (None when absent), or ``UNSEEN``."""

    if not _ACTIVE:
        return UNSEEN
    target = Path(path)
    return digest(target.read_bytes()) if target.is_file() else None


def rewrote(path: Path | str, prior: object, body: bytes) -> None:
    """Report that ``path`` now holds ``body``; ``prior`` is what ``peek`` returned before."""

    if prior is UNSEEN or not _ACTIVE:
        return
    key, after = _key(path), digest(body)
    with _LOCK:
        for rec in _ACTIVE:  # a write on top of someone else's change since the last claims nothing
            first = rec.files.get(key, [prior, prior])
            rec.files[key] = [first[0] if prior == first[1] else BROKEN, after]  # type: ignore[list-item]
