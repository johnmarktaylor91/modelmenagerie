"""Input closures for the required gate set (RC-3: certification by closure).

Each required gate declares, in ``required_gate_set.json`` (schema v3), the
repository paths whose committed bytes it reads. The gate's ``closure_sha`` is
a content digest over exactly those tracked files as git records them for a
given tree: sorted ``(blob sha, path)`` pairs, hashed with the gate name. A
verdict keyed by ``closure_sha`` therefore stays consumable for every later
commit that leaves the gate's inputs byte-identical, and stops being
consumable the moment any input moves. Nothing here reads a producer field:
the closure is recomputed from the git object store on every call.

Guarantees preserved from the whole-tree scheme this replaces: evidence is
bound to the exact bytes the gate read (now the exact bytes it DECLARES to
read, with a census test proving no source path sits outside every
declaration); stale, foreign, red, and truncated verdicts still refuse in
``gate_verdicts``; no operator input steers any key.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_RELPATH = Path("menagerie/crawler/tools/required_gate_set.json")
MANIFEST_SCHEMA_VERSION = "menagerie.crawler.required-gate-set.v3"
CLOSURE_DIGEST_DOMAIN = "menagerie.crawler.gate-closure.v1"

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "comment",
        "census_roots",
        "implicit_closure",
        "gates",
        "required_for_invocation",
    }
)
_GATE_FIELDS = frozenset(
    {
        "description",
        "always",
        "input_closure",
        "exclude",
        "command",
        "env",
        "junit",
        "requires_armed_floor",
    }
)
_COMMAND_PLACEHOLDERS = frozenset({"{python}", "{junit}"})
#: Source-file suffixes the closure census demands coverage for.
SOURCE_SUFFIXES = (".py", ".toml")


class GateClosureError(RuntimeError):
    """Raised when the manifest is malformed or a closure cannot be computed."""


@dataclass(frozen=True)
class GateSpec:
    """One required gate's declared inputs and invocation."""

    name: str
    description: str
    always: bool
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    command: tuple[str, ...] | None
    env: Mapping[str, str]
    junit: bool
    requires_armed_floor: bool

    @property
    def is_attestation(self) -> bool:
        """True when the verdict is recorded from an attestation, not a command."""

        return self.command is None


@dataclass(frozen=True)
class GateManifest:
    """The loaded, validated required-gate-set manifest."""

    census_roots: tuple[str, ...]
    implicit_closure: tuple[str, ...]
    gates: Mapping[str, GateSpec]
    required_for_invocation: tuple[str, ...]

    def gate(self, name: str) -> GateSpec:
        try:
            return self.gates[name]
        except KeyError as exc:
            raise GateClosureError(
                f"unknown gate {name!r}; declared gates: {sorted(self.gates)}"
            ) from exc


def _require_str_list(value: Any, *, what: str, allow_duplicates: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise GateClosureError(f"required-gate-set: {what} must be a list of non-empty strings")
    if not allow_duplicates and len(set(value)) != len(value):
        raise GateClosureError(f"required-gate-set: {what} carries duplicates")
    return tuple(value)


def _parse_command(name: str, command: Any) -> tuple[str, ...] | None:
    """Validate a gate command; placeholders stand alone or end a token."""

    if command is None:
        return None
    command_tuple = tuple(
        _require_str_list(command, what=f"gate {name!r} command", allow_duplicates=True)
    )
    for token in command_tuple:
        for placeholder in ("{python}", "{junit}"):
            if (
                placeholder in token
                and token not in _COMMAND_PLACEHOLDERS
                and not token.endswith(placeholder)
            ):
                raise GateClosureError(
                    f"required-gate-set: gate {name!r} command token {token!r} embeds a "
                    "placeholder mid-token; placeholders stand alone or end a token"
                )
    return command_tuple


def _parse_env(name: str, env: Any) -> dict[str, str]:
    """Validate a gate's declared environment as a str-to-str mapping."""

    if not isinstance(env, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise GateClosureError(f"required-gate-set: gate {name!r} env must map str to str")
    return dict(env)


def _parse_gate(name: str, payload: Any) -> GateSpec:
    if not isinstance(payload, Mapping):
        raise GateClosureError(f"required-gate-set: gate {name!r} is not an object")
    keys = set(payload)
    if keys != _GATE_FIELDS:
        raise GateClosureError(
            f"required-gate-set: gate {name!r} fields missing={sorted(_GATE_FIELDS - keys)} "
            f"unknown={sorted(keys - _GATE_FIELDS)}"
        )
    include = _require_str_list(payload["input_closure"], what=f"gate {name!r} input_closure")
    if not include:
        raise GateClosureError(f"required-gate-set: gate {name!r} declares an empty closure")
    exclude = _require_str_list(payload["exclude"], what=f"gate {name!r} exclude")
    command_tuple = _parse_command(name, payload["command"])
    env = _parse_env(name, payload["env"])
    for field in ("always", "junit", "requires_armed_floor"):
        if not isinstance(payload[field], bool):
            raise GateClosureError(f"required-gate-set: gate {name!r} {field} must be a bool")
    if command_tuple is None and payload["junit"]:
        raise GateClosureError(f"required-gate-set: attestation gate {name!r} cannot declare junit")
    if not isinstance(payload["description"], str) or not payload["description"]:
        raise GateClosureError(f"required-gate-set: gate {name!r} needs a description")
    return GateSpec(
        name=name,
        description=payload["description"],
        always=payload["always"],
        include=include,
        exclude=exclude,
        command=command_tuple,
        env=env,
        junit=payload["junit"],
        requires_armed_floor=payload["requires_armed_floor"],
    )


def parse_manifest(payload: Any) -> GateManifest:
    """Validate one manifest document against the closed v3 schema."""

    if not isinstance(payload, Mapping):
        raise GateClosureError("required-gate-set: document is not an object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise GateClosureError(
            f"required-gate-set-legacy: schema {payload.get('schema_version')!r} is not "
            f"{MANIFEST_SCHEMA_VERSION!r}; prior-era manifests refuse typed, no downgrade"
        )
    keys = set(payload)
    if keys != _MANIFEST_FIELDS:
        raise GateClosureError(
            f"required-gate-set: fields missing={sorted(_MANIFEST_FIELDS - keys)} "
            f"unknown={sorted(keys - _MANIFEST_FIELDS)}"
        )
    gates_payload = payload["gates"]
    if not isinstance(gates_payload, Mapping) or not gates_payload:
        raise GateClosureError("required-gate-set: gates must be a non-empty object")
    gates = {name: _parse_gate(name, spec) for name, spec in gates_payload.items()}
    required = _require_str_list(payload["required_for_invocation"], what="required_for_invocation")
    unknown = sorted(set(required) - set(gates))
    if unknown:
        raise GateClosureError(
            f"required-gate-set: required_for_invocation names unknown gates {unknown}"
        )
    return GateManifest(
        census_roots=_require_str_list(payload["census_roots"], what="census_roots"),
        implicit_closure=_require_str_list(payload["implicit_closure"], what="implicit_closure"),
        gates=gates,
        required_for_invocation=required,
    )


def load_manifest(repo_root: Path) -> GateManifest:
    """Load and validate the committed manifest under ``repo_root``."""

    path = repo_root / MANIFEST_RELPATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateClosureError(f"required-gate-set: {path} is missing") from exc
    except json.JSONDecodeError as exc:
        raise GateClosureError(f"required-gate-set: {path} is not JSON: {exc}") from exc
    return parse_manifest(payload)


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise GateClosureError(
            f"git {' '.join(arguments)} failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    return completed.stdout


def tree_sha(repo_root: Path, tree_ref: str = "HEAD") -> str:
    """Return the tree object sha for ``tree_ref``."""

    return _git(repo_root, "rev-parse", f"{tree_ref}^{{tree}}").strip()


def commit_sha(repo_root: Path, ref: str = "HEAD") -> str:
    """Return the commit sha for ``ref``."""

    return _git(repo_root, "rev-parse", f"{ref}^{{commit}}").strip()


def tracked_blobs(repo_root: Path, tree_ref: str = "HEAD") -> tuple[tuple[str, str], ...]:
    """Return every ``(path, blob sha)`` tracked in ``tree_ref``, sorted by path."""

    output = _git(repo_root, "ls-tree", "-r", "-z", "--full-tree", tree_ref)
    entries: list[tuple[str, str]] = []
    for record in output.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if len(parts) != 3:
            raise GateClosureError(f"unparseable ls-tree record: {record!r}")
        _mode, kind, sha = parts
        if kind != "blob":
            continue
        entries.append((path, sha))
    entries.sort()
    return tuple(entries)


def path_covered(path: str, patterns: Iterable[str]) -> bool:
    """True when ``path`` matches any pattern (prefix when it ends with '/')."""

    for pattern in patterns:
        if pattern == "./":
            return True
        if pattern.endswith("/"):
            if path.startswith(pattern):
                return True
        elif path == pattern:
            return True
    return False


def closure_paths(
    manifest: GateManifest, gate: GateSpec, blobs: Sequence[tuple[str, str]]
) -> tuple[tuple[str, str], ...]:
    """Return the tracked ``(path, blob)`` pairs inside ``gate``'s closure."""

    include = (*gate.include, *manifest.implicit_closure)
    return tuple(
        (path, blob)
        for path, blob in blobs
        if path_covered(path, include) and not path_covered(path, gate.exclude)
    )


def closure_sha_from_paths(gate_name: str, pairs: Sequence[tuple[str, str]]) -> str:
    """Digest one gate's closure from its ``(path, blob)`` pairs."""

    digest = hashlib.sha256()
    digest.update(f"{CLOSURE_DIGEST_DOMAIN}\ngate {gate_name}\n".encode())
    for path, blob in sorted(pairs):
        digest.update(f"{blob} {path}\n".encode())
    return digest.hexdigest()


def closure_sha(
    repo_root: Path,
    manifest: GateManifest,
    gate_name: str,
    *,
    tree_ref: str = "HEAD",
    blobs: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Compute ``gate_name``'s closure sha for ``tree_ref``."""

    gate = manifest.gate(gate_name)
    resolved = tracked_blobs(repo_root, tree_ref) if blobs is None else blobs
    pairs = closure_paths(manifest, gate, resolved)
    if not pairs:
        raise GateClosureError(
            f"gate {gate_name!r} closure selects zero tracked files in {tree_ref}; "
            "an empty closure certifies nothing"
        )
    return closure_sha_from_paths(gate_name, pairs)


def all_closure_shas(
    repo_root: Path, manifest: GateManifest, *, tree_ref: str = "HEAD"
) -> dict[str, str]:
    """Compute every declared gate's closure sha for one tree."""

    blobs = tracked_blobs(repo_root, tree_ref)
    return {
        name: closure_sha(repo_root, manifest, name, tree_ref=tree_ref, blobs=blobs)
        for name in manifest.gates
    }


def moved_gates(
    repo_root: Path, manifest: GateManifest, *, base_ref: str, head_ref: str = "HEAD"
) -> dict[str, bool]:
    """Return ``gate -> moved`` between two trees (``always`` gates read as moved)."""

    base = all_closure_shas(repo_root, manifest, tree_ref=base_ref)
    head = all_closure_shas(repo_root, manifest, tree_ref=head_ref)
    return {
        name: manifest.gates[name].always or base[name] != head[name] for name in manifest.gates
    }


def is_source_path(path: str, manifest: GateManifest) -> bool:
    """True when ``path`` is a census-root source file (``.py`` or ``.toml``)."""

    return path_covered(path, manifest.census_roots) and path.endswith(SOURCE_SUFFIXES)


def uncovered_source_paths(
    manifest: GateManifest, blobs: Sequence[tuple[str, str]]
) -> tuple[str, ...]:
    """Census: tracked source paths inside no gate's closure (must be empty)."""

    covered: set[str] = set()
    for gate in manifest.gates.values():
        covered.update(path for path, _blob in closure_paths(manifest, gate, blobs))
    return tuple(
        path for path, _blob in blobs if is_source_path(path, manifest) and path not in covered
    )


def closure_dirty_paths(repo_root: Path, manifest: GateManifest, gate_name: str) -> tuple[str, ...]:
    """Working-tree paths inside ``gate_name``'s closure that differ from HEAD.

    Modified, staged, deleted, renamed, AND untracked files count: an
    untracked test module would run without being part of the closure the
    verdict is keyed on, so it is as disqualifying as an edit.
    """

    gate = manifest.gate(gate_name)
    include = (*gate.include, *manifest.implicit_closure)
    output = _git(repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    dirty: list[str] = []
    records = output.split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status, path = record[:2], record[3:]
        if status[0] == "R" or status[1] == "R":
            # Renames carry the origin path in the following record.
            index += 1
        if path_covered(path, include) and not path_covered(path, gate.exclude):
            dirty.append(path)
    return tuple(sorted(dirty))


def render_command(gate: GateSpec, *, python: str, junit: Path | None) -> tuple[str, ...]:
    """Substitute the command placeholders for one invocation."""

    if gate.command is None:
        raise GateClosureError(f"gate {gate.name!r} is attestation-recorded; it has no command")
    rendered: list[str] = []
    for token in gate.command:
        value = token.replace("{python}", python)
        if "{junit}" in value:
            if junit is None:
                raise GateClosureError(f"gate {gate.name!r} command needs a junit path")
            value = value.replace("{junit}", junit.as_posix())
        rendered.append(value)
    return tuple(rendered)
