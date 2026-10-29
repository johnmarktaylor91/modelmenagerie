"""Commit the lines and files the tick itself wrote, never another writer's edits.

``git-commit-paths`` commits whole files, so a data file another writer had left dirty used to
block the tick's own lines in it. Here each file is committed as exactly what the tick wrote:

- a JSONL file the tick appended to: HEAD's content plus the tick's own lines, in the order it
  wrote them, while another writer's appended lines stay uncommitted in the working tree;
- a JSON file the tick rewrote (a packet): its current content, only when the file held HEAD's
  content before the tick's first write (no other writer's edit inside) and still holds what the
  tick wrote.

A file that cannot be committed this way is held and logged, never overwritten: the next pass
tries again (a held packet goes in once its other writer commits its own edit), and a file
someone changed or committed since is dropped from the tick's care. The commit is built in a
temporary index from HEAD, checked by the configured ``pre-commit`` and ``commit-msg`` hooks, and
the branch moves by compare-and-swap, so a commit that landed meanwhile is never undone. The
shared index entry of each committed path is prepared inside git's ``index.lock`` and installed
together with the branch move (``commit_owned``); a path someone has staged is held instead.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from menagerie.pipeline import written

TRIES = 5  # compare-and-swap attempts when HEAD moves under the commit


@dataclass
class Owned:
    """What the tick wrote, by repository-relative path (``written.Writes`` without the root)."""

    lines: dict[str, list[str]] = field(default_factory=dict)
    files: dict[str, list[str | None]] = field(default_factory=dict)

    @classmethod
    def from_writes(cls, rec: written.Writes, repo: Path,
                    keep: Callable[[str], bool] = lambda rel: True) -> Owned:
        """The recorded writes inside ``repo`` whose relative path ``keep`` accepts."""

        def rel(path: str) -> str | None:
            try:
                name = Path(path).relative_to(repo.resolve()).as_posix()
            except ValueError:
                return None
            return name if keep(name) else None

        lines = {r: list(v) for p, v in rec.lines.items() if (r := rel(p))}
        files = {r: list(v) for p, v in rec.files.items() if (r := rel(p))}
        return cls({k: v for k, v in lines.items() if k not in files}, files)

    @classmethod
    def from_mapping(cls, raw: Any) -> Owned:
        raw = raw if isinstance(raw, Mapping) else {}
        return cls({k: list(v) for k, v in (raw.get("lines") or {}).items()},
                   {k: list(v) for k, v in (raw.get("files") or {}).items()})

    def to_mapping(self) -> dict[str, Any]:
        return {"lines": self.lines, "files": self.files} if self else {}

    def merged(self, other: Owned) -> Owned:
        lines = {k: list(v) for k, v in self.lines.items()}
        for k, v in other.lines.items():
            lines[k] = [*lines.get(k, []), *(x for x in v if x not in lines.get(k, []))]
        files = {**self.files, **{k: _chain(self.files.get(k), v) for k, v in other.files.items()}}
        return Owned({k: v for k, v in lines.items() if k not in files}, files)

    def __bool__(self) -> bool:
        return bool(self.lines or self.files)


def _chain(first: list[str | None] | None, then: list[str | None]) -> list[str | None]:
    """Two writes of one file as one: the later must start from the earlier one's result."""

    if first is None:
        return list(then)
    return [first[0] if then[0] == first[1] else written.BROKEN, then[1]]


@dataclass
class Outcome:
    """``commit`` (None: nothing new), the paths it carries, and per path why it was held (kept
    for the next pass), dropped (no longer the tick's to commit) or already in HEAD."""

    commit: str | None = None
    committed: list[str] = field(default_factory=list)
    held: dict[str, str] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    done: list[str] = field(default_factory=list)
    pending: Owned = field(default_factory=Owned)  # what the next pass should try again

    def to_json(self) -> dict[str, Any]:
        out = {"commit": self.commit, "paths": self.committed, "held": self.held,
               "dropped": self.dropped}
        return {k: v for k, v in out.items() if v}


# --- git -------------------------------------------------------------------------------------


def _git(repo: Path, *args: str, env: Mapping[str, str] | None = None, data: bytes | None = None,
         check: bool = True) -> subprocess.CompletedProcess[bytes]:
    done = subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True,
                          env=dict(env) if env is not None else None, check=False, timeout=120)
    if check and done.returncode:
        raise OSError(f"git {args[0]} rc {done.returncode}: {done.stderr.decode()[-300:]}")
    return done


def _head_blob(repo: Path, rev: str, path: str) -> tuple[str, bytes] | None:
    """``(mode, content)`` of ``path`` at ``rev``, or None when it is not there."""

    listed = _git(repo, "ls-tree", rev, "--", path).stdout.decode().split()
    if not listed:
        return None
    return listed[0], _git(repo, "cat-file", "blob", listed[2]).stdout


# --- what each path's committed content would be ----------------------------------------------


def _extra(base: bytes, work: bytes) -> Counter[str] | None:
    """The working tree's lines beyond HEAD's, or None when HEAD's lines are not all there in
    order (another writer changed or removed a committed line). Lines anywhere count, so another
    writer's line left between committed ones (after an earlier partial commit) is fine."""

    head = base.decode("utf-8", "replace").splitlines(keepends=True)
    lines = work.decode("utf-8", "replace").splitlines(keepends=True)
    it = iter(lines)
    if not all(any(x == y for y in it) for x in head):
        return None
    return Counter(lines) - Counter(head)


def _plan_lines(repo: Path, rev: str, path: str, mine: Sequence[str],
                out: Outcome) -> tuple[str, bytes] | None:
    head = _head_blob(repo, rev, path)
    mode, base = head if head else ("100644", b"")
    have = set(base.decode("utf-8", "replace").splitlines(keepends=True))
    todo = [x for x in mine if x not in have]
    target = repo / path
    if not todo:
        out.done.append(path)
        return None
    if not target.is_file():
        out.dropped[path] = "the file is gone from the working tree"
        return None
    extra = _extra(base, target.read_bytes()) if base.endswith(b"\n") or not base else None
    if extra is None:
        out.held[path] = "its committed lines were changed in the working tree by another writer"
        out.pending.lines[path] = list(mine)
        return None
    if Counter(todo) - extra:
        out.dropped[path] = "some of the tick's lines are no longer in the file"
        return None
    return mode, base + "".join(todo).encode("utf-8")


def _plan_file(repo: Path, rev: str, path: str, before: str | None, after: str | None,
               out: Outcome) -> tuple[str, bytes] | None:
    head = _head_blob(repo, rev, path)
    mode, base = head if head else ("100644", None)
    target = repo / path
    work = target.read_bytes() if target.is_file() else None
    if written.digest(base) == after:
        out.done.append(path)
        return None
    if written.digest(work) != after:
        out.dropped[path] = "another writer changed it after the tick wrote it"
        return None
    if written.digest(base) != before:  # also ``written.BROKEN``: changed between the tick's writes
        out.held[path] = "another writer's uncommitted edit was in it before the tick wrote it"
        out.pending.files[path] = [before, after]
        return None
    return mode, work  # type: ignore[return-value]


def plan(repo: Path, rev: str, owned: Owned) -> tuple[dict[str, tuple[str, bytes]], Outcome]:
    """Each path's ``(mode, content)`` to commit on ``rev``, and what was held or dropped."""

    out = Outcome()
    blobs: dict[str, tuple[str, bytes]] = {}
    for path, mine in sorted(owned.lines.items()):
        if (got := _plan_lines(repo, rev, path, mine, out)) is not None:
            blobs[path] = got
    for path, (before, after) in sorted(owned.files.items()):
        if (got := _plan_file(repo, rev, path, before, after, out)) is not None:
            blobs[path] = got
    return blobs, out


# --- the commit ------------------------------------------------------------------------------


def _hook(repo: Path, name: str, env: Mapping[str, str], *args: str) -> None:
    """Run one configured hook (``core.hooksPath``, else the repository's ``hooks`` folder) the way
    ``git commit`` would: from the top of the checkout, with the commit's index in the environment."""

    configured = _git(repo, "config", "core.hooksPath", check=False).stdout.decode().strip()
    if configured:
        folder = repo / Path(configured).expanduser()  # a relative path is from the checkout's top
    else:
        common = _git(repo, "rev-parse", "--git-common-dir").stdout.decode().strip()
        folder = repo / common / "hooks"
    hook = folder / name
    if not (hook.is_file() and os.access(hook, os.X_OK)):
        return
    done = subprocess.run([str(hook), *args], cwd=repo, env=dict(env), capture_output=True,
                          check=False, timeout=600)
    if done.returncode:
        raise OSError(f"{name} hook refused: {(done.stderr or done.stdout).decode()[-300:]}")


def _hooks(repo: Path, env: Mapping[str, str], message: str, tmp: Path) -> str:
    """Run the ``pre-commit`` and ``commit-msg`` hooks; return the message they leave."""

    _hook(repo, "pre-commit", env)
    note = tmp / "COMMIT_EDITMSG"
    note.write_text(message + "\n", encoding="utf-8")
    _hook(repo, "commit-msg", env, str(note))
    return note.read_text(encoding="utf-8")


def _build(repo: Path, rev: str, blobs: Mapping[str, tuple[str, bytes]], message: str,
           tmp: Path) -> tuple[str, str] | None:
    """A commit on ``rev`` that changes only ``blobs``, and its ``update-index --index-info``
    rows; None when the tree would not change."""

    env = {**os.environ, "GIT_INDEX_FILE": str(tmp / "index")}
    _git(repo, "read-tree", rev, env=env)
    rows = []
    for path, (mode, body) in sorted(blobs.items()):
        sha = _git(repo, "hash-object", "-w", "--stdin", data=body).stdout.decode().strip()
        rows.append(f"{mode} {sha}\t{path}\n")
    _git(repo, "update-index", "--index-info", env=env, data="".join(rows).encode("utf-8"))
    tree = _git(repo, "write-tree", env=env).stdout.decode().strip()
    if tree == _git(repo, "rev-parse", f"{rev}^{{tree}}").stdout.decode().strip():
        return None
    final = _hooks(repo, env, message, tmp)
    new = _git(repo, "commit-tree", tree, "-p", rev, data=final.encode("utf-8")).stdout.decode()
    return new.strip(), "".join(rows)


def _staged(repo: Path, rev: str, paths: Sequence[str]) -> list[str]:
    """Paths whose shared index entry is not ``rev``'s (someone staged them): never committed
    over, so a later commit of that staged entry cannot undo the tick's lines."""

    if not paths:
        return []

    def entries(*args: str) -> dict[str, tuple[str, ...]]:
        out = _git(repo, *args, "-z", "--", *paths).stdout.decode()
        return {row.split("\t", 1)[1]: tuple(row.split("\t", 1)[0].split())
                for row in out.split("\0") if row}

    index = {k: (v[0], v[1], v[2]) for k, v in entries("ls-files", "-s").items()}  # mode sha stage
    tree = {k: (v[0], v[2], "0") for k, v in entries("ls-tree", rev).items()}  # mode type sha
    return sorted(p for p in paths if index.get(p) != tree.get(p))


@contextlib.contextmanager
def _index_lock(repo: Path, deadline: float) -> Iterator[dict[str, Any]]:
    """Hold the shared index's ``index.lock`` the way git does (create it exclusively), waiting out
    another session's lock and never removing it. Yields ``{"fd", "lock", "index", "installed"}``;
    unless the block installed the lock over the index (``installed``), the lock is removed after
    it, and never after an install, when the name may already be another session's new lock."""

    index = repo / _git(repo, "rev-parse", "--git-path", "index").stdout.decode().strip()
    lock = index.with_name(index.name + ".lock")
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            if time.monotonic() + 5 > deadline:
                raise OSError(f"{lock} stayed held by another session") from None
            time.sleep(2)
    held = {"fd": fd, "lock": lock, "index": index, "installed": False}
    try:
        yield held
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
        if not held["installed"]:
            lock.unlink(missing_ok=True)


def _stage_into_lock(repo: Path, rows: str, held: dict[str, Any]) -> None:
    """Write the shared index with ``rows`` applied into the held lock, the way git would: from a
    copy that keeps the index's own mtime, so git's racy-entry check still smudges any entry whose
    file changed within the index's timestamp, then ``update-index`` writes it."""

    with tempfile.TemporaryDirectory(prefix="tick-index-") as tmp:
        copy = Path(tmp) / "index"
        if held["index"].is_file():
            shutil.copy2(held["index"], copy)  # copy2 keeps the mtime git compares against
        env = {**os.environ, "GIT_INDEX_FILE": str(copy)}
        _git(repo, "update-index", "--index-info", env=env, data=rows.encode("utf-8"))
        body = copy.read_bytes()
    view = memoryview(body)
    while view:  # a short write (a full disk) must never become the shared index
        view = view[os.write(held["fd"], view):]
    os.fsync(held["fd"])
    if os.fstat(held["fd"]).st_size != len(body):
        raise OSError("the prepared index was not written whole")


def _publish(repo: Path, old: str, new: str, held: dict[str, Any]) -> bool:
    """Move HEAD from ``old`` to ``new`` and install the prepared index as one step: SIGALRM (the
    tick's bound) waits until both are done, so a pass never stops between them."""

    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            _git(repo, "update-ref", "-m", "commit: tick data", "HEAD", new, old, check=False)
        # HEAD decides, not update-ref's status: a ref hook may fail or stall after the move
        if _git(repo, "rev-parse", "HEAD").stdout.decode().strip() != new:
            return False
        os.replace(held["lock"], held["index"])  # a rename in one folder
        held["installed"] = True
        return True
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def _attempt(repo: Path, owned: Owned, message: str, deadline: float) -> Outcome | None:
    """One try on the current HEAD; None when HEAD moved or a path was staged meanwhile."""

    old = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    blobs, out = plan(repo, old, owned)
    for path in _staged(repo, old, sorted(blobs)):
        out.held[path] = "someone staged it in the shared index; tried again next pass"
        kind = "lines" if path in owned.lines else "files"
        getattr(out.pending, kind)[path] = getattr(owned, kind)[path]
        del blobs[path]
    if not blobs:
        return out
    with tempfile.TemporaryDirectory(prefix="tick-commit-") as tmp:
        built = _build(repo, old, blobs, message, Path(tmp))
    if built is None:
        out.done += sorted(blobs)
        return out
    new, rows = built
    with _index_lock(repo, deadline) as held:
        if _staged(repo, old, sorted(blobs)):
            return None
        _stage_into_lock(repo, rows, held)  # before HEAD moves: a failure here changes nothing
        if not _publish(repo, old, new, held):
            return None
    out.commit, out.committed = new, sorted(blobs)
    return out


def commit_owned(repo: Path, owned: Owned, message: str, deadline: float) -> Outcome:
    """Commit exactly what the tick wrote; see the module docstring.

    The new index is prepared inside the shared ``index.lock`` before HEAD moves, and HEAD moves
    (compare-and-swap) and the index is installed together, so no git command that takes the lock
    (``git add``, ``git commit``, ``git-commit-paths``) runs between them; a path someone has
    staged is held. A git command that read the index before the lock and writes it after can
    still leave a stale entry, as between any two committers. Returns the outcome; ``pending``
    holds what to try on the next pass. Raises ``OSError`` when git or a hook fails; the caller
    keeps ``owned`` and tries it again then.
    """

    for _ in range(TRIES):
        out = _attempt(repo, owned, message, deadline)
        if out is not None:
            return out
        if time.monotonic() + 5 > deadline:
            break
        time.sleep(1)  # HEAD moved (another commit landed) or a path was staged: plan again
    raise OSError("HEAD kept moving while the tick committed its own writes")
