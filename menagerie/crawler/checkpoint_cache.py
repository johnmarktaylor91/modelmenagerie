"""Persistent validation memo behind the final-authority checkpoint cache.

risk_r5 background: per-model final-authority derivation re-ran jsonschema
validation over the COMPLETE artifact history on every close (~88% of the
measured per-call cost on the pilot corpus), so per-model time grew with
terminal count and the campaign close was O(N^2). The memo removes exactly
that pure-function recomputation and nothing else.

Honesty model
-------------
The memo stores POSITIVE outcomes only, for two pure validations:

- ``record``: one persisted ledger record passed ``validate_payload``.
  Key = ``stable_hash([schema_version, payload_sha256])``. The
  ``payload_sha256`` digest covers every field except itself and is
  re-verified LIVE against the record bytes on every ledger read
  (``recordio._verify_hash``), so a hit can only ever skip re-validating
  bytes whose content identity was just re-proven.
- ``transaction-document``: one committed reconstruction document passed
  ``_validate_reconstruction_document`` + ``_validate_projection_dependencies``
  cleanly. Key = ``stable_hash`` over the anchor sha256 (re-verified LIVE
  against the committed bytes on every projection) and the final and
  authorization events' ``payload_sha256``.

Every key is additionally bound to the ENVIRONMENT identity -- a digest over
all schema file bytes, the jsonschema library version, and the
``CHECKPOINT_CACHE_EPOCH`` code constant -- so any schema, library, or
validator-semantics change invalidates the whole store. Bump the epoch in the
same change that alters validation semantics.

The store is a DERIVED artifact outside the records ledgers (the driver
places it under ``<runtime_root>/derived/checkpoint-cache``). Deleting it at
any time reproduces uncached behavior exactly; a corrupt store is discarded
with a typed diagnostic and never raises; failures are never memoized, so the
memo can never change a verdict.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Any

from menagerie.crawler.identity import stable_hash
from menagerie.crawler.recordio import durable_fsync
from menagerie.crawler.schema import SCHEMA_DIRECTORY

# Bump whenever validation semantics change in a way schema bytes do not
# capture (validator code, memo key derivation, record hashing rules).
# v2 (IC-7): the implementation-digest set became the derived import closure
# of the validation entry modules instead of a hand-listed four -- the
# hand list silently under-covered collaborators (identity/mirrors/authority
# hashing and parsing participate in what a memo hit skips).
CHECKPOINT_CACHE_EPOCH = "checkpoint-cache-epoch-v2"

MEMO_FILENAME = "validation-memo-v1.jsonl"

_MEMO_KINDS = frozenset({"record", "transaction-document"})

_ENVIRONMENT_IDENTITY: str | None = None


#: The validation entry modules whose pure recomputation a memo hit skips.
#: The digest participant set is their package-local import CLOSURE, derived
#: from parsed imports -- never a hand list (IC-7: the previous four-file hand
#: list silently omitted identity.py, mirrors.py, and every other collaborator
#: whose edit changes what a hit would have recomputed).
_VALIDATION_ENTRY_MODULES: tuple[str, ...] = (
    "artifact_transactions.py",
    "checkpoint_cache.py",
    "recordio.py",
    "schema.py",
)


def _validation_import_closure(module_root: Path) -> tuple[str, ...]:
    """Return the package-local import closure of the validation entry modules.

    Parameters
    ----------
    module_root:
        The ``menagerie/crawler`` package directory.

    Returns
    -------
    tuple[str, ...]
        Sorted module filenames reachable from the entry modules through
        ``menagerie.crawler`` imports (top-level submodules only).
    """

    package = "menagerie.crawler"
    closed: set[str] = set()
    pending: list[str] = list(_VALIDATION_ENTRY_MODULES)
    while pending:
        name = pending.pop()
        if name in closed:
            continue
        path = module_root / name
        if not path.is_file():
            continue
        closed.add(name)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    # relative import inside the package
                    if module:
                        pending.append(f"{module.split('.')[0]}.py")
                    else:
                        pending.extend(f"{alias.name}.py" for alias in node.names)
                elif module == package:
                    pending.extend(f"{alias.name}.py" for alias in node.names)
                elif module.startswith(f"{package}."):
                    pending.append(module.removeprefix(f"{package}.").split(".")[0] + ".py")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(f"{package}."):
                        pending.append(
                            alias.name.removeprefix(f"{package}.").split(".")[0] + ".py"
                        )
    return tuple(sorted(closed))


def _validator_implementation_digests() -> dict[str, str]:
    """Return digests of every implementation file that can grant a memo hit.

    Returns
    -------
    dict[str, str]
        Module filename to content digest, over the derived import closure of
        the validation entry modules. Hashing the implementations makes any
        validator-collaborator edit invalidate PC1 automatically; the manual
        epoch remains an explicit compatibility/documentation axis, not the
        sole correctness binding.
    """

    module_root = Path(__file__).resolve().parent
    return {
        name: "sha256:" + hashlib.sha256((module_root / name).read_bytes()).hexdigest()
        for name in _validation_import_closure(module_root)
    }


def _replace_memo_store(
    path: Path, keys: set[tuple[str, str]], environment: str
) -> str | None:
    """Atomically replace a derived memo with only live validated keys.

    Parameters
    ----------
    path, keys, environment:
        Store path, positive keys, and current validator identity.

    Returns
    -------
    str | None
        Truncated error text on failure, otherwise ``None``.
    """

    temporary = path.with_name(f".{path.name}.rewrite")
    payload = b"".join(
        json.dumps(
            {"environment": environment, "kind": kind, "key": key},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for kind, key in sorted(keys)
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            durable_fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        return str(exc)[:200]
    return None


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write every payload byte or raise the underlying I/O error.

    Parameters
    ----------
    descriptor:
        Open append descriptor.
    payload:
        Complete memo suffix.

    Raises
    ------
    OSError
        If the operating system makes no forward progress or reports failure.
    """

    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short checkpoint-cache write made no progress")
        remaining = remaining[written:]


def checkpoint_cache_environment_identity() -> str:
    """Return the schema/library/code identity binding every memo entry.

    Returns
    -------
    str
        ``sha256:`` digest over every schema file's bytes, validator
        implementation bytes, the jsonschema library version, and
        ``CHECKPOINT_CACHE_EPOCH``.
    """

    global _ENVIRONMENT_IDENTITY
    if _ENVIRONMENT_IDENTITY is None:
        schema_digests = {
            path.name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(SCHEMA_DIRECTORY.glob("*.json"))
        }
        _ENVIRONMENT_IDENTITY = stable_hash(
            {
                "epoch": CHECKPOINT_CACHE_EPOCH,
                "jsonschema": _package_version("jsonschema"),
                "python": sys.version,
                "referencing": _package_version("referencing"),
                "schemas": schema_digests,
                "validators": _validator_implementation_digests(),
            }
        )
    return _ENVIRONMENT_IDENTITY


class ValidationMemo:
    """Append-only positive-outcome memo for pure checkpoint validations."""

    def __init__(self, path: Path, keys: set[tuple[str, str]], diagnostics: list[dict[str, Any]]):
        """Bind loaded state; use :meth:`open` instead of calling directly.

        Parameters
        ----------
        path, keys, diagnostics:
            Store path, trusted (kind, key) pairs, and load diagnostics.
        """

        self._path = path
        self._keys = keys
        self._pending: list[tuple[str, str]] = []
        self.diagnostics = diagnostics

    @classmethod
    def open(cls, root: Path) -> "ValidationMemo":
        """Load the memo store below ``root``, degrading loudly on corruption.

        A missing store is an empty memo. A torn final line (no newline)
        drops only the tail. Any COMPLETE malformed or implausible line
        discards the whole store: fail toward live derivation, never toward
        trusting a corrupt cache.

        Parameters
        ----------
        root:
            Derived-store directory (created lazily on flush).

        Returns
        -------
        ValidationMemo
            Loaded memo with typed diagnostics for every degrade.
        """

        path = root / MEMO_FILENAME
        keys: set[tuple[str, str]] = set()
        diagnostics: list[dict[str, Any]] = []
        environment = checkpoint_cache_environment_identity()
        stale_rows = 0
        try:
            handle = path.open("rb")
        except FileNotFoundError:
            return cls(path, keys, diagnostics)
        except OSError as exc:
            diagnostics.append(
                {
                    "reason": "checkpoint-cache-unreadable",
                    "path": str(path),
                    "error": str(exc)[:200],
                }
            )
            return cls(path, set(), diagnostics)
        torn_tail = False
        with handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    torn_tail = True
                    diagnostics.append(
                        {"reason": "checkpoint-cache-torn-tail", "path": str(path)}
                    )
                    break
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, ValueError):
                    row = None
                if (
                    not isinstance(row, dict)
                    or row.get("kind") not in _MEMO_KINDS
                    or not isinstance(row.get("key"), str)
                    or not isinstance(row.get("environment"), str)
                ):
                    diagnostics.append(
                        {"reason": "checkpoint-cache-corrupt", "path": str(path)}
                    )
                    rewrite_error = _replace_memo_store(path, set(), environment)
                    if rewrite_error is not None:
                        diagnostics.append(
                            {
                                "reason": "checkpoint-cache-rewrite-failed",
                                "path": str(path),
                                "error": rewrite_error,
                            }
                        )
                    return cls(path, set(), diagnostics)
                if row["environment"] != environment:
                    stale_rows += 1
                    continue
                keys.add((row["kind"], row["key"]))
        if stale_rows:
            diagnostics.append(
                {
                    "reason": "checkpoint-cache-stale-environment",
                    "path": str(path),
                    "stale_rows": stale_rows,
                }
            )
        if torn_tail or stale_rows:
            rewrite_error = _replace_memo_store(path, keys, environment)
            if rewrite_error is not None:
                diagnostics.append(
                    {
                        "reason": "checkpoint-cache-rewrite-failed",
                        "path": str(path),
                        "error": rewrite_error,
                    }
                )
        return cls(path, keys, diagnostics)

    @property
    def pending_count(self) -> int:
        """Return how many recorded outcomes await a durable flush.

        Returns
        -------
        int
            Unflushed row count.
        """

        return len(self._pending)

    @property
    def path(self) -> Path:
        """Return the backing store path.

        Returns
        -------
        pathlib.Path
            Memo JSONL path.
        """

        return self._path

    def seen(self, kind: str, key: str) -> bool:
        """Return whether this exact positive outcome is memoized.

        Parameters
        ----------
        kind, key:
            Memo kind and content-identity key.

        Returns
        -------
        bool
            True only for a previously recorded identical validation.
        """

        return (kind, key) in self._keys

    def record(self, kind: str, key: str) -> None:
        """Memoize one live positive validation outcome.

        Parameters
        ----------
        kind, key:
            Memo kind and content-identity key.

        Raises
        ------
        ValueError
            If ``kind`` is not a known memo kind.
        """

        if kind not in _MEMO_KINDS:
            raise ValueError(f"unknown validation memo kind: {kind!r}")
        if (kind, key) in self._keys:
            return
        self._keys.add((kind, key))
        self._pending.append((kind, key))

    def flush(self) -> None:
        """Append pending rows durably; a write failure degrades to cost only."""

        if not self._pending:
            return
        environment = checkpoint_cache_environment_identity()
        payload = b"".join(
            json.dumps(
                {"environment": environment, "kind": kind, "key": key},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
            for kind, key in self._pending
        )
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                _write_all(descriptor, payload)
                durable_fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            self.diagnostics.append(
                {
                    "reason": "checkpoint-cache-write-failed",
                    "path": str(self._path),
                    "error": str(exc)[:200],
                }
            )
            return
        self._pending.clear()
