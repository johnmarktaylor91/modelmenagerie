"""Append-only gate-verdict ledger v3, keyed by input closure.

Every gate run records ONE verdict row carrying the gate's ``closure_sha``
(the content digest of exactly the tracked files the gate declares it reads,
recomputed from git by :mod:`menagerie.crawler.tools.gate_closure`), the tree
observed before AND after the run (disclosure and drift refusal), the commit,
the exact node sets, exit status, environment identity, host, argv, UTC mint
time, and the content digest of the FULL combined output blob. Rows are
append-only (``.crawl-local/gate-verdicts.jsonl``); nothing rewrites or
truncates them, and consumers read the ledger, never a mutable "latest" file.

Consumption is by closure (RC-3): ``consume_green(gate, closure_sha)`` returns
the newest consumable green row for that exact closure. A later commit that
leaves a gate's inputs byte-identical keeps consuming the same verdict; a
commit that moves any input has no verdict until a fresh run appends one.
Newest row wins (a red recorded after a green refuses until a new green
supersedes it), red refuses, tree drift during the run refuses, missing or
truncated evidence refuses. Verdicts minted by CI on another host are as
consumable as local ones: environment identity is DISCLOSED in the row, never
part of the key, because the invocation preflight must accept CI's verdict
for the same bytes (host in the key would make that impossible by
construction).

Threat model (unchanged, per the adjudicated ruling): ACCIDENT, not malice.
Rows are unsigned; forgery resistance is out of scope. The defenses are
against the accident classes the 200a sprint actually produced: verdicts that
never existed, truncated evidence, stale reuse (an old green for moved
inputs), and red-cache laundering.

Prior-era rows (schema v2, keyed by whole tree) carry no closure and can never
satisfy a closure consumer; :func:`load_ledger` segregates them into a
disclosed legacy count instead of refusing the whole file, so a ledger that
predates RC-3 stays readable while its old rows grant nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from menagerie.crawler.identity import durable_fsync
from menagerie.crawler.tools import gate_closure

GATE_VERDICT_SCHEMA_VERSION = "menagerie.crawler.gate-verdict.v3"
LEGACY_SCHEMA_VERSIONS = frozenset({"menagerie.crawler.gate-verdict.v2"})
VERDICTS_RELPATH = Path(".crawl-local/gate-verdicts.jsonl")
GATE_LOGS_RELDIR = Path(".crawl-local/gate-logs")

#: Closed verdict statuses. There is deliberately no third value: a gate run
#: either terminated fully green or it is red. Flake, cache, and operator
#: attestation are not verdict statuses and can never become one without a
#: reviewed schema change.
VERDICT_STATUSES = frozenset({"green", "red"})

_NODE_SET_KEYS = ("passed", "failed", "skipped", "xfailed")

#: Closed field set for one verdict row. Unknown fields refuse: a producer
#: cannot smuggle steering state through the ledger.
_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "gate",
        "status",
        "exit_code",
        "closure_sha",
        "tree_sha_start",
        "tree_sha_end",
        "commit_sha",
        "env_identity",
        "host",
        "argv",
        "minted_at",
        "pid",
        "execution_floor_armed",
        "node_sets",
        "log_sha256",
    }
)

#: Conservative secret-shaped patterns redacted from stored gate logs. The
#: evidence store is full-output by design; redaction keeps accidental
#: credential echoes out of durable content-addressed blobs.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"(?i)\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)\s*=\s*\S+"),
)
_REDACTION = "[REDACTED]"
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class GateVerdictError(RuntimeError):
    """Raised when a verdict row cannot be recorded, read, or consumed."""


def _sha256_hex(payload: bytes) -> str:
    """Return the bare SHA-256 hex digest of ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    """Return a UTC ISO-8601 ``Z`` timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_secrets(text: str) -> str:
    """Redact secret-shaped substrings before durable evidence storage."""

    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(rf"\1={_REDACTION}", text)
        else:
            text = pattern.sub(_REDACTION, text)
    return text


def store_gate_log(repo_root: Path, output: str) -> str:
    """Store the FULL redacted gate output content-addressed; return its digest."""

    data = redact_secrets(output).encode("utf-8")
    digest = _sha256_hex(data)
    directory = repo_root / GATE_LOGS_RELDIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.log"
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    return digest


def load_gate_log(repo_root: Path, log_sha256: str) -> str:
    """Load one evidence blob, refusing missing or digest-mismatched bytes."""

    path = repo_root / GATE_LOGS_RELDIR / f"{log_sha256}.log"
    if not path.is_file():
        raise GateVerdictError(
            f"gate-log-missing: evidence blob {log_sha256} is absent; "
            "a verdict without intact full evidence refuses"
        )
    data = path.read_bytes()
    observed = _sha256_hex(data)
    if observed != log_sha256:
        raise GateVerdictError(
            f"gate-log-truncated: evidence blob {log_sha256} hashes to {observed}; "
            "truncated or corrupted gate evidence refuses the verdict"
        )
    return data.decode("utf-8")


def host_identity() -> dict[str, str]:
    """Return the minting host's identity triple (disclosure only)."""

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "node": socket.gethostname(),
    }


def environment_identity(python: str | None = None) -> dict[str, str]:
    """Return the disclosed environment identity of one gate run.

    Disclosure, never a key: the closure sha binds the bytes the gate read;
    which interpreter, platform, and environment prefix ran them is recorded
    beside the verdict so a reader can tell a hosted-runner verdict from a
    campaign-host one.
    """

    interpreter = python or sys.executable
    return {
        # Absolute, NOT resolved: a virtualenv interpreter is a symlink to the
        # base interpreter, and resolving it would hide which environment ran.
        "interpreter": str(Path(interpreter).absolute()),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "platform": f"{platform.system()}/{platform.machine()}",
        "real_env_prefix": ("present" if os.environ.get("MENAGERIE_REAL_ENV_PREFIX") else "absent"),
        # The real-environment fixture falls back to the ambient conda prefix
        # when no prefix is selected; a reader adjudicating a red needs to see
        # which family the run was armed against.
        "conda_prefix": "present" if os.environ.get("CONDA_PREFIX") else "absent",
    }


def build_verdict(
    *,
    gate: str,
    status: str,
    exit_code: int,
    closure_sha: str,
    tree_sha_start: str,
    tree_sha_end: str,
    commit_sha: str,
    env_identity: Mapping[str, str],
    argv: Sequence[str],
    execution_floor_armed: bool,
    node_sets: Mapping[str, Iterable[str]],
    log_sha256: str,
) -> dict[str, Any]:
    """Assemble one closed-schema verdict row (validated on append)."""

    return {
        "schema_version": GATE_VERDICT_SCHEMA_VERSION,
        "gate": gate,
        "status": status,
        "exit_code": exit_code,
        "closure_sha": closure_sha,
        "tree_sha_start": tree_sha_start,
        "tree_sha_end": tree_sha_end,
        "commit_sha": commit_sha,
        "env_identity": dict(env_identity),
        "host": host_identity(),
        "argv": [str(argument) for argument in argv],
        "minted_at": utc_now(),
        "pid": os.getpid(),
        "execution_floor_armed": bool(execution_floor_armed),
        "node_sets": {key: sorted(set(node_sets.get(key, ()))) for key in _NODE_SET_KEYS},
        "log_sha256": log_sha256,
    }


def _validate_row_shape(row: Any, *, context: str) -> None:
    """Refuse a non-object row, a field-set drift, or a prior-era schema."""

    if not isinstance(row, Mapping):
        raise GateVerdictError(f"gate-verdict-invalid: {context}: row is not an object")
    keys = set(row)
    if keys != _ROW_FIELDS:
        missing = sorted(_ROW_FIELDS - keys)
        unknown = sorted(keys - _ROW_FIELDS)
        raise GateVerdictError(
            f"gate-verdict-invalid: {context}: missing={missing} unknown={unknown}"
        )
    if row["schema_version"] != GATE_VERDICT_SCHEMA_VERSION:
        raise GateVerdictError(
            f"gate-verdict-legacy: {context}: schema {row['schema_version']!r} is not "
            f"{GATE_VERDICT_SCHEMA_VERSION!r}; prior-era rows refuse typed, no downgrade"
        )


def _validate_digest(row: Mapping[str, Any], field: str, *, context: str) -> None:
    """Refuse a digest field that is not a sha256 hex string."""

    value = row[field]
    if not isinstance(value, str) or not _SHA256_HEX.match(value):
        raise GateVerdictError(
            f"gate-verdict-invalid: {context}: {field} is not a sha256 hex digest"
        )


def _validate_row(row: Mapping[str, Any], *, context: str) -> None:
    """Refuse any row outside the closed v3 verdict schema."""

    _validate_row_shape(row, context=context)
    if row["status"] not in VERDICT_STATUSES:
        raise GateVerdictError(
            f"gate-verdict-invalid: {context}: status {row['status']!r} outside closed vocabulary"
        )
    if not isinstance(row["exit_code"], int) or isinstance(row["exit_code"], bool):
        raise GateVerdictError(f"gate-verdict-invalid: {context}: exit_code is not an int")
    _validate_digest(row, "closure_sha", context=context)
    node_sets = row["node_sets"]
    if not isinstance(node_sets, Mapping) or set(node_sets) != set(_NODE_SET_KEYS):
        raise GateVerdictError(
            f"gate-verdict-invalid: {context}: node_sets must carry exactly {_NODE_SET_KEYS}"
        )
    if row["status"] == "green" and (node_sets["failed"] or row["exit_code"] != 0):
        raise GateVerdictError(
            f"gate-verdict-invalid: {context}: a green verdict cannot carry failed nodes "
            "or a nonzero exit"
        )
    if not isinstance(row["env_identity"], Mapping):
        raise GateVerdictError(f"gate-verdict-invalid: {context}: env_identity is not an object")
    _validate_digest(row, "log_sha256", context=context)


def append_verdict(repo_root: Path, row: Mapping[str, Any]) -> Path:
    """Validate and append one verdict row; the ledger is append-only."""

    _validate_row(row, context="append")
    path = repo_root / VERDICTS_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode("utf-8")
    with path.open("ab") as handle:
        handle.write(encoded + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    return path


def load_ledger(repo_root: Path) -> tuple[list[dict[str, Any]], int]:
    """Load the ledger: ``(v3 rows, count of segregated prior-era rows)``.

    Malformed JSON and rows outside both the v3 schema and the closed legacy
    schema set still refuse typed. Legacy rows are counted, never returned:
    they carry no closure and cannot satisfy any consumer.
    """

    path = repo_root / VERDICTS_RELPATH
    if not path.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    legacy = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateVerdictError(
                f"gate-verdict-invalid: {path}:{line_number} is not JSON: {exc}"
            ) from exc
        if isinstance(value, Mapping) and value.get("schema_version") in LEGACY_SCHEMA_VERSIONS:
            legacy += 1
            continue
        _validate_row(value, context=f"{path}:{line_number}")
        rows.append(dict(value))
    return rows, legacy


def load_verdicts(repo_root: Path) -> list[dict[str, Any]]:
    """Load every consumable-schema row (legacy rows segregated and dropped)."""

    rows, _legacy = load_ledger(repo_root)
    return rows


def consume_green(
    repo_root: Path,
    *,
    gate: str,
    closure_sha: str,
    require_armed_floor: bool = False,
) -> dict[str, Any]:
    """Return the newest CONSUMABLE green verdict for ``gate`` on ``closure_sha``.

    A verdict is consumable only when: status green, exit zero, zero failed
    nodes, tree observed identical before and after the run, evidence blob
    present with matching digest, and (when required) the execution floor was
    armed. Anything else refuses typed: red, missing, stale (other closure),
    foreign, incomplete, or truncated evidence can NEVER satisfy a consumer.
    """

    rows = load_verdicts(repo_root)
    candidates = [row for row in rows if row["gate"] == gate]
    if not candidates:
        raise GateVerdictError(f"gate-verdict-missing: no verdict rows exist for gate {gate!r}")
    matching = [row for row in candidates if row["closure_sha"] == closure_sha]
    if not matching:
        raise GateVerdictError(
            f"gate-verdict-stale: no verdict for gate {gate!r} on closure {closure_sha}; "
            f"newest row is for closure {candidates[-1]['closure_sha']} "
            f"(commit {candidates[-1]['commit_sha']}); the gate's inputs moved, re-run it"
        )
    # The NEWEST row for (gate, closure) is authoritative: a red recorded after
    # a green refuses until a fresh green run supersedes it. A red->green flip
    # is a re-run decision, never a search backwards through history.
    row = matching[-1]
    if row["status"] != "green":
        raise GateVerdictError(
            f"gate-verdict-red: newest verdict for gate {gate!r} on closure {closure_sha} "
            "is red; red or missing evidence never certifies"
        )
    if row["tree_sha_end"] != row["tree_sha_start"]:
        raise GateVerdictError(
            f"gate-verdict-invalid: gate {gate!r} observed tree drift during the "
            f"run ({row['tree_sha_start']} -> {row['tree_sha_end']})"
        )
    if require_armed_floor and not row["execution_floor_armed"]:
        raise GateVerdictError(
            f"gate-verdict-floor-unarmed: gate {gate!r} ran green but the "
            "execution floor was not armed; the manifest requires an armed run"
        )
    load_gate_log(repo_root, row["log_sha256"])
    return row


def consume_required_gates_green(
    repo_root: Path, *, tree_ref: str = "HEAD"
) -> dict[str, dict[str, Any]]:
    """Consume a green verdict for EVERY invocation-required gate on ``tree_ref``.

    This is the closure-green arm of the invocation preflight (RC-1 step 3,
    rekeyed by RC-3): for each gate the manifest marks required, compute the
    gate's closure sha for the running tree and consume by closure. Returns the
    consumed rows keyed by gate; raises :class:`GateVerdictError` naming the
    first gate that refuses.
    """

    manifest = gate_closure.load_manifest(repo_root)
    closures = gate_closure.all_closure_shas(repo_root, manifest, tree_ref=tree_ref)
    consumed: dict[str, dict[str, Any]] = {}
    for name in manifest.required_for_invocation:
        spec = manifest.gate(name)
        consumed[name] = consume_green(
            repo_root,
            gate=name,
            closure_sha=closures[name],
            require_armed_floor=spec.requires_armed_floor,
        )
    return consumed


def verdict_status(repo_root: Path, *, tree_ref: str = "HEAD") -> list[dict[str, Any]]:
    """Per-gate consumability report for one tree (diagnostic, refuses nothing)."""

    manifest = gate_closure.load_manifest(repo_root)
    closures = gate_closure.all_closure_shas(repo_root, manifest, tree_ref=tree_ref)
    report: list[dict[str, Any]] = []
    for name, spec in manifest.gates.items():
        entry: dict[str, Any] = {
            "gate": name,
            "closure_sha": closures[name],
            "required_for_invocation": name in manifest.required_for_invocation,
        }
        try:
            row = consume_green(
                repo_root,
                gate=name,
                closure_sha=closures[name],
                require_armed_floor=spec.requires_armed_floor,
            )
        except GateVerdictError as exc:
            entry["consumable"] = False
            entry["detail"] = str(exc)
        else:
            entry["consumable"] = True
            entry["minted_at"] = row["minted_at"]
            entry["commit_sha"] = row["commit_sha"]
            entry["host"] = row["host"]["node"]
        report.append(entry)
    return report
