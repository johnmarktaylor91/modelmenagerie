"""Durable per-attempt records for the headless author executor.

One **attempt** is one executor pass over one model's author lifecycle: stage 1
discovery, the machine-owned source broker, and stage 2 authoring, all inside a
single per-attempt directory. Nothing is ever shared between attempts, so a
superseded attempt structurally *cannot* publish into a live one — the executor
publishes only from the attempt it opened, and any stray output found inside a
superseded attempt directory is quarantined, never read as a result.

Layout, under the lane's author root (``<work>/<stable_id>/author``)::

    attempts/attempt-<NNN>-<nonce>/
        record.json        current durable state, rewritten atomically
        events.jsonl       append-only event log
        scratch/           cwd for the attempt's ``claude -p`` sessions
        broker/            resolver receipts, raw API evidence, probe outcomes
        quarantine/        late or superseded outputs, moved, never deleted

The record is the resume authority: a killed executor leaves the record behind,
and the next invocation reads it to decide whether to resume the recorded
provider session, replay a completed phase from its recorded output, or open
attempt N+1 with this attempt in ``prior_attempts``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from menagerie.crawler.identity import durable_fsync, fsync_directory, hash_bytes, utc_now
from menagerie.crawler.models import JsonObject

ATTEMPT_RECORD_VERSION = "menagerie.crawler.author-attempt.v1"
ATTEMPTS_DIR_NAME = "attempts"

#: Statuses from which an attempt can never progress again.
TERMINAL_STATUSES = frozenset({"completed", "failed", "superseded"})
#: Every status the record may carry, in rough lifecycle order.
KNOWN_STATUSES = frozenset(
    {
        "created",
        "stage1-running",
        "stage1-complete",
        "sources-published",
        "stage2-running",
        "supplement-running",
        "completed",
        "failed",
        "superseded",
    }
)

_ATTEMPT_DIR_PATTERN = re.compile(r"^attempt-(\d{3,})-([0-9a-f]{8,64})$")
_EXA_QUERY_SECRET_PATTERN = re.compile(r"(?i)(exaApiKey=)([^&\"'\s]+)")
_CREDENTIAL_REDACTION = "[REDACTED-BL11]"


class AttemptRecordError(ValueError):
    """Raised when an attempt record or directory is not exact."""


class CredentialSerializationError(AttemptRecordError):
    """Raised when an attempt emission contains a live provider credential."""


def _credential_variants() -> tuple[str, ...]:
    """Return nonempty provider-secret spellings that must never be serialized.

    Returns
    -------
    tuple[str, ...]
        Raw configured secrets. URL-bearing records are caught independently by
        :data:`_EXA_QUERY_SECRET_PATTERN`, including after the operator rotates.
    """

    return tuple(
        value.strip()
        for name in ("EXA_API_KEY",)
        if (value := os.environ.get(name)) is not None and value.strip()
    )


def assert_credential_free_record(serialized: str) -> None:
    """Fail closed when one emitted attempt record contains a provider secret.

    Parameters
    ----------
    serialized:
        Exact JSON text about to be written.

    Raises
    ------
    CredentialSerializationError
        If an Exa query credential or the configured live secret appears. The
        exception deliberately never repeats the matched bytes.
    """

    query_match = _EXA_QUERY_SECRET_PATTERN.search(serialized)
    if query_match is not None and query_match.group(2) != _CREDENTIAL_REDACTION:
        raise CredentialSerializationError(
            "attempt emission refused: provider credential detected in serialized record"
        )
    if any(secret in serialized for secret in _credential_variants()):
        raise CredentialSerializationError(
            "attempt emission refused: provider credential detected in serialized record"
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist one JSON object with a directory fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert_credential_free_record(data)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path) -> Optional[JsonObject]:
    """Read one JSON object, returning ``None`` for absent or unreadable files."""

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def attempts_root(author_root: Union[str, Path]) -> Path:
    """Return the attempts registry directory under one author root.

    Parameters
    ----------
    author_root:
        The lane's per-model author directory (the request envelope's parent).

    Returns
    -------
    pathlib.Path
        Registry directory holding every attempt of this model.
    """

    return Path(author_root) / ATTEMPTS_DIR_NAME


@dataclass(frozen=True)
class AttemptPaths:
    """Absolute paths owned by exactly one attempt."""

    directory: Path
    record: Path
    events: Path
    scratch: Path
    broker: Path
    quarantine: Path

    @classmethod
    def for_directory(cls, directory: Path) -> "AttemptPaths":
        """Derive every attempt-owned path from the attempt directory."""

        directory = Path(directory)
        return cls(
            directory=directory,
            record=directory / "record.json",
            events=directory / "events.jsonl",
            scratch=directory / "scratch",
            broker=directory / "broker",
            quarantine=directory / "quarantine",
        )


class AttemptHandle:
    """One live attempt: its paths, its durable record, and its event log."""

    def __init__(self, paths: AttemptPaths, record: JsonObject, *, clock: Callable[[], str]):
        self.paths = paths
        self.record = record
        self._clock = clock

    @property
    def nonce(self) -> str:
        """Return the attempt nonce."""

        return str(self.record["attempt_nonce"])

    @property
    def number(self) -> int:
        """Return the attempt number."""

        return int(self.record["attempt_number"])

    @property
    def status(self) -> str:
        """Return the current lifecycle status."""

        return str(self.record["status"])

    def update(self, **fields: Any) -> None:
        """Merge fields into the record and persist it atomically.

        Parameters
        ----------
        **fields:
            Top-level record fields to set. ``status`` is validated against the
            closed vocabulary; a terminal record refuses further transitions.

        Raises
        ------
        AttemptRecordError
            On an unknown status or a transition out of a terminal status.
        """

        new_status = fields.get("status")
        if new_status is not None:
            if new_status not in KNOWN_STATUSES:
                raise AttemptRecordError(f"unknown attempt status {new_status!r}")
            if self.status in TERMINAL_STATUSES and new_status != self.status:
                raise AttemptRecordError(
                    f"attempt {self.nonce} is terminal ({self.status}); "
                    f"refusing transition to {new_status}"
                )
        self.record.update(fields)
        self.record["updated_at"] = self._clock()
        _write_json_atomic(self.paths.record, self.record)

    def event(self, name: str, **fields: Any) -> None:
        """Append one event line to the attempt's durable event log."""

        line = json.dumps(
            {"at": self._clock(), "event": name, **fields},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        assert_credential_free_record(line)
        with self.paths.events.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            durable_fsync(handle.fileno())

    def quarantine(self, path: Path, *, reason: str) -> Optional[Path]:
        """Move one stray output into this attempt's quarantine directory.

        Parameters
        ----------
        path:
            File to quarantine. Absent files are ignored (nothing to launder).
        reason:
            Closed short reason recorded with the move.

        Returns
        -------
        pathlib.Path | None
            Quarantined destination, or ``None`` when the file was absent.
        """

        path = Path(path)
        if not path.is_file():
            return None
        self.paths.quarantine.mkdir(parents=True, exist_ok=True)
        destination = self.paths.quarantine / f"{uuid.uuid4().hex[:8]}-{path.name}"
        os.replace(path, destination)
        entries = list(self.record.get("quarantine", []))
        entries.append(
            {
                "original": str(path),
                "quarantined_to": str(destination),
                "reason": reason,
                "sha256": hash_bytes(destination.read_bytes()),
                "at": self._clock(),
            }
        )
        self.update(quarantine=entries)
        self.event("quarantined", original=str(path), destination=str(destination), reason=reason)
        return destination

    def mark_superseded(self, *, by_nonce: str, reason: str) -> None:
        """Terminalize this attempt as superseded and quarantine stray results.

        Any ``result.json`` / ``source-targets.json`` a still-running orphaned
        provider session may yet write into this attempt lands in a directory
        that is never read as a result; files already present are moved to
        quarantine now. ``discovery.json`` stays: it is crash-recovery input,
        never a publishable result.

        Parameters
        ----------
        by_nonce:
            Nonce of the superseding attempt.
        reason:
            Closed short reason (``retry``, ``timeout``, ``crash-recovery``).
        """

        if self.status in TERMINAL_STATUSES and self.status != "superseded":
            # A completed/failed attempt is history, not supersedable state.
            return
        for name in ("result.json", "source-targets.json"):
            self.quarantine(self.paths.directory / name, reason=f"superseded:{reason}")
            self.quarantine(self.paths.scratch / name, reason=f"superseded:{reason}")
        if self.status != "superseded":
            self.update(
                status="superseded",
                superseded={"at": self._clock(), "by_nonce": by_nonce, "reason": reason},
            )
            self.event("superseded", by_nonce=by_nonce, reason=reason)


def new_attempt(
    author_root: Union[str, Path],
    *,
    stable_id: str,
    campaign_id: Optional[str],
    kind: str,
    clock: Callable[[], str] = utc_now,
    nonce_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    inherit_from: Optional[Mapping[str, Any]] = None,
) -> AttemptHandle:
    """Open attempt N+1, superseding every still-open prior attempt.

    Parameters
    ----------
    author_root:
        The lane's per-model author directory.
    stable_id:
        Model identity the attempt serves.
    campaign_id:
        Campaign the attempt runs under, when known.
    kind:
        Request kind that opened the attempt (``source-request`` or ``author``).
    clock, nonce_factory:
        Injectable time and identity sources.
    inherit_from:
        Optional prior attempt record whose stage-1 session identity and
        discovery output this attempt may reuse (crash recovery).

    Returns
    -------
    AttemptHandle
        Live handle with a persisted ``created`` record.

    Raises
    ------
    AttemptRecordError
        When the stable id is empty.
    """

    if not stable_id.strip():
        raise AttemptRecordError("attempt stable_id must be non-empty")
    root = attempts_root(author_root)
    root.mkdir(parents=True, exist_ok=True)
    existing = list_attempts(author_root, clock=clock)
    number = 1 + max((handle.number for handle in existing), default=0)
    nonce = nonce_factory()
    directory = root / f"attempt-{number:03d}-{nonce}"
    paths = AttemptPaths.for_directory(directory)
    paths.scratch.mkdir(parents=True, exist_ok=True)
    paths.broker.mkdir(parents=True, exist_ok=True)
    record: JsonObject = {
        "record_version": ATTEMPT_RECORD_VERSION,
        "stable_id": stable_id,
        "campaign_id": campaign_id,
        "kind": kind,
        "attempt_number": number,
        "attempt_nonce": nonce,
        "attempt_dir": str(directory),
        "created_at": clock(),
        "updated_at": clock(),
        "status": "created",
        "stage1": None,
        "stage2": None,
        "supplement": None,
        "broker": None,
        "outcome": None,
        "published": None,
        "superseded": None,
        "quarantine": [],
        "inherited": (
            {
                "attempt_nonce": inherit_from.get("attempt_nonce"),
                "attempt_number": inherit_from.get("attempt_number"),
                "stage1": inherit_from.get("stage1"),
                "discovery_path": inherit_from.get("discovery_path"),
            }
            if inherit_from is not None
            else None
        ),
    }
    handle = AttemptHandle(paths, record, clock=clock)
    _write_json_atomic(paths.record, record)
    handle.event("created", kind=kind, stable_id=stable_id)
    for prior in existing:
        if prior.status not in TERMINAL_STATUSES:
            prior.mark_superseded(by_nonce=nonce, reason="retry")
        elif prior.status == "superseded":
            # An orphaned provider session can land its output AFTER the
            # attempt was superseded; every attempt boundary re-sweeps those
            # late writes into quarantine so they never look publishable.
            for name in ("result.json", "source-targets.json"):
                prior.quarantine(prior.paths.directory / name, reason="superseded:late-write")
                prior.quarantine(prior.paths.scratch / name, reason="superseded:late-write")
    return handle


def list_attempts(
    author_root: Union[str, Path],
    *,
    clock: Callable[[], str] = utc_now,
) -> list[AttemptHandle]:
    """Return every readable attempt under one author root, oldest first.

    Directories whose records are missing or unreadable are skipped: an attempt
    without a record cannot be resumed and cannot publish, so it is inert.
    """

    root = attempts_root(author_root)
    if not root.is_dir():
        return []
    handles: list[AttemptHandle] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or _ATTEMPT_DIR_PATTERN.match(entry.name) is None:
            continue
        paths = AttemptPaths.for_directory(entry)
        record = _read_json_object(paths.record)
        if record is None or record.get("record_version") != ATTEMPT_RECORD_VERSION:
            continue
        handles.append(AttemptHandle(paths, record, clock=clock))
    handles.sort(key=lambda handle: handle.number)
    return handles


def latest_attempt(
    author_root: Union[str, Path],
    *,
    stable_id: Optional[str] = None,
    statuses: Optional[frozenset[str]] = None,
    clock: Callable[[], str] = utc_now,
) -> Optional[AttemptHandle]:
    """Return the newest attempt matching the given filters.

    Parameters
    ----------
    author_root:
        The lane's per-model author directory.
    stable_id:
        Require this model identity when given.
    statuses:
        Restrict to these lifecycle statuses when given.
    clock:
        Injectable time source for the returned handle.
    """

    for handle in reversed(list_attempts(author_root, clock=clock)):
        if stable_id is not None and handle.record.get("stable_id") != stable_id:
            continue
        if statuses is not None and handle.status not in statuses:
            continue
        return handle
    return None


def record_checker_findings(
    author_root: Union[str, Path],
    *,
    gate_kind: str,
    generation: int,
    required_repairs: Sequence[str],
    root_cause_fingerprint: Optional[str] = None,
) -> bool:
    """Attach verbatim checker findings to the rejected generation's attempt.

    This is the checker half of the ONE feedback channel: the repair loop
    records what the checker rejected onto the newest attempt record, and the
    next attempt's ``prior_attempts`` summary — on the envelope and rendered
    into the brief — carries it to the repairing author verbatim. Before this
    channel existed the findings went to a file no author ever read.

    Parameters
    ----------
    author_root:
        The lane's per-model author directory.
    gate_kind:
        Rejecting gate (``metadata_batch`` or ``fidelity``).
    generation:
        One-based repair generation the findings open.
    required_repairs:
        Verbatim checker findings.
    root_cause_fingerprint:
        Optional stable fingerprint of the failure class.

    Returns
    -------
    bool
        Whether an attempt record existed to carry the findings.
    """

    for handle in reversed(list_attempts(author_root)):
        if handle.record.get("status") == "created":
            continue
        entries = list(handle.record.get("checker_findings") or [])
        entries.append(
            {
                "gate_kind": gate_kind,
                "generation": generation,
                "required_repairs": list(required_repairs),
                "root_cause_fingerprint": root_cause_fingerprint,
                "recorded_at": utc_now(),
            }
        )
        handle.record["checker_findings"] = entries
        handle.record["updated_at"] = utc_now()
        _write_json_atomic(handle.paths.record, handle.record)
        handle.event("checker-findings-recorded", gate_kind=gate_kind, generation=generation)
        return True
    return False


def prior_attempts_summary(
    author_root: Union[str, Path],
    *,
    limit: int = 5,
) -> list[JsonObject]:
    """Return the feedback channel: what earlier attempts did and why they ended.

    The newest ``limit`` non-live attempts are summarized for the envelope's
    ``prior_attempts`` block and the brief's "WHAT WENT WRONG LAST TIME"
    section. Verbatim failure reasons and checker findings ride along so a
    retry is never blind.

    Parameters
    ----------
    author_root:
        The lane's per-model author directory.
    limit:
        Maximum prior attempts to include, newest last.
    """

    summaries: list[JsonObject] = []
    for handle in list_attempts(author_root):
        record = handle.record
        if record.get("status") == "created":
            continue
        outcome = record.get("outcome") or {}
        published = record.get("published") or {}
        summaries.append(
            {
                "attempt_number": record.get("attempt_number"),
                "attempt_nonce": record.get("attempt_nonce"),
                "status": record.get("status"),
                "kind": record.get("kind"),
                "outcome": outcome,
                "failure_stage": outcome.get("failure_stage") if outcome else None,
                "failure_reason": outcome.get("failure_reason") if outcome else None,
                "failure_detail": outcome.get("detail") if outcome else None,
                "checker_findings": record.get("checker_findings"),
                "prior_result_sha256": published.get("sha256") if published else None,
                "created_at": record.get("created_at"),
            }
        )
    return summaries[-limit:]


def _redact_credential_text(value: str, provider_secret: Optional[str]) -> tuple[str, int]:
    """Redact Exa credential spellings from one string.

    Parameters
    ----------
    value:
        Candidate serialized field.
    provider_secret:
        Optional currently configured key. Historical URL query values redact
        even when the old key is no longer available.

    Returns
    -------
    tuple[str, int]
        Redacted text and number of replacements.
    """

    redacted, count = _EXA_QUERY_SECRET_PATTERN.subn(
        lambda match: f"{match.group(1)}{_CREDENTIAL_REDACTION}", value
    )
    secret = (provider_secret or "").strip()
    if secret:
        occurrences = redacted.count(secret)
        redacted = redacted.replace(secret, _CREDENTIAL_REDACTION)
        count += occurrences
    return redacted, count


def _redact_credential_value(value: Any, provider_secret: Optional[str]) -> tuple[Any, int]:
    """Recursively redact provider credentials from a JSON-compatible value."""

    if isinstance(value, str):
        return _redact_credential_text(value, provider_secret)
    if isinstance(value, list):
        redacted: list[Any] = []
        count = 0
        for item in value:
            clean, replacements = _redact_credential_value(item, provider_secret)
            redacted.append(clean)
            count += replacements
        return redacted, count
    if isinstance(value, dict):
        redacted_object: JsonObject = {}
        count = 0
        for key, item in value.items():
            clean_key, key_replacements = _redact_credential_text(str(key), provider_secret)
            clean, replacements = _redact_credential_value(item, provider_secret)
            redacted_object[clean_key] = clean
            count += key_replacements + replacements
        return redacted_object, count
    return value, 0


def scrub_attempt_credentials(
    root: Union[str, Path], *, provider_secret: Optional[str] = None
) -> JsonObject:
    """Redact leaked Exa credentials from existing on-disk attempt records.

    Parameters
    ----------
    root:
        Campaign work root, model author root, or attempts root to sweep.
    provider_secret:
        Optional exact key to redact in addition to every historical
        ``exaApiKey=...`` query value. Defaults to ``EXA_API_KEY``.

    Returns
    -------
    dict[str, Any]
        Files scanned, files rewritten, and replacement count. No key material
        is returned or printed.
    """

    base = Path(root)
    secret = os.environ.get("EXA_API_KEY") if provider_secret is None else provider_secret
    candidates = sorted(
        path
        for path in {*base.rglob("record.json"), *base.rglob("events.jsonl")}
        if _ATTEMPT_DIR_PATTERN.fullmatch(path.parent.name) is not None
    )
    rewritten = 0
    replacements = 0
    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise AttemptRecordError(f"credential scrub could not read {path}") from exc
        if not lines:
            raise AttemptRecordError(f"credential scrub found an empty record file: {path}")
        output: list[str] = []
        file_replacements = 0
        for line_number, line in enumerate(lines, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AttemptRecordError(
                    f"credential scrub found invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise AttemptRecordError(
                    f"credential scrub found a non-object record at {path}:{line_number}"
                )
            clean, count = _redact_credential_value(payload, secret)
            output.append(
                json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            )
            file_replacements += count
        if file_replacements == 0:
            assert_credential_free_record("\n".join(output))
            continue
        replacement_text = "\n".join(output) + "\n"
        assert_credential_free_record(replacement_text)
        if path.name == "record.json":
            _write_json_atomic(path, json.loads(output[0]))
        else:
            _write_lines_atomic(path, output)
        rewritten += 1
        replacements += file_replacements
    return {
        "root": str(base),
        "files_scanned": len(candidates),
        "files_rewritten": rewritten,
        "credentials_redacted": replacements,
    }


def _write_lines_atomic(path: Path, lines: Sequence[str]) -> None:
    """Atomically rewrite a JSONL file after credential redaction."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    data = "\n".join(lines) + "\n"
    assert_credential_free_record(data)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the BL-11 attempt-record credential scrubber."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2 or arguments[0] != "scrub-credentials":
        print(
            "usage: python -m menagerie.crawler.author_attempts "
            "scrub-credentials <campaign-work-root>",
            file=sys.stderr,
        )
        return 64
    result = scrub_attempt_credentials(arguments[1])
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
