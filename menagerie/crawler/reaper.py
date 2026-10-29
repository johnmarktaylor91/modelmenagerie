"""Referenced-path manifest and tiered, report-only disk reaper (C-DISK-GC).

The crawl's reapable bulk is interleaved with load-bearing state (T4-T9):
``repair/`` globs ARE the live repair budget, ledger diagnostics pointers are
machine-dereferenced, attempt records feed every re-drive, and the author
``source-cas`` is re-read by evidence replay. Naive reaping silently refunds
spent LLM budget or breaks replay, so this module computes two artifacts and
keeps them honest against each other:

* the **referenced-path manifest** -- everything a reap must preserve, as
  typed :class:`ProtectedEntry` rows recomputed from canonical records and
  machine-written envelopes (never from producer flags);
* the **reap plan** -- tiered :class:`ReapCandidate` rows, each admitted only
  after the corpus guard (both containment directions) and a containment check
  against every manifest entry.

The binding GC contract is ``procedures/TEXT_CORPUS.md``: the corpus is never
reapable (``text_corpus.assert_not_corpus_path`` on every candidate), a work
dir's scratch is reapable only once ``text_corpus.work_dir_reap_clearance``
clears it, and garbage collection stays report-only until explicitly applied.
The apply path recomputes the whole plan under the driver lock, so a stale or
forged plan can never force a deletion the fresh computation would refuse.

Environment-operator orphans (T1-19) ride the ``env-orphans`` tier through
:func:`operator_environment.sweep_environment_orphans`; their shared CAS
object trees are kept safe structurally (the sweep only ever names ``.tmp``
partials, staging dirs, and registration-less generation dirs), not by
manifest enumeration.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from menagerie.crawler.constants import TERMINAL_STATUS_CODES
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.state import _select_current
from menagerie.crawler.text_corpus import (
    CorpusReapRefusal,
    assert_not_corpus_path,
    corpus_root,
    work_dir_reap_clearance,
)

JsonObject = dict[str, Any]

REAP_PLAN_VERSION = "menagerie.crawler.reap-plan.v1"

TIER_PROMOTED_ATTEMPT_SCRATCH = "promoted-attempt-scratch"
TIER_DIAGNOSTICS_ORPHANS = "diagnostics-orphans"
TIER_INSTRUMENTATION = "instrumentation"
TIER_DERIVED_CACHES = "derived-caches"
TIER_ENV_ORPHANS = "env-orphans"

#: Every reap tier, in report order. Each is independently selectable; none is
#: ever applied without an explicit ``--apply``.
REAP_TIERS = (
    TIER_PROMOTED_ATTEMPT_SCRATCH,
    TIER_DIAGNOSTICS_ORPHANS,
    TIER_INSTRUMENTATION,
    TIER_DERIVED_CACHES,
    TIER_ENV_ORPHANS,
)

#: Attempt directory naming, mirrored from ``author_attempts`` (its module
#: constant is private and the format is frozen by the attempt registry).
_ATTEMPT_DIR_PATTERN = re.compile(r"^attempt-(\d{3,})-([0-9a-f]{8,64})$")

#: Diagnostics sidecar references as they appear inside canonical ledger rows
#: (schema-pinned relative form, ``attempt-common.schema.json``).
_DIAGNOSTIC_REFERENCE_PATTERN = re.compile(r"\.crawl-local/diagnostics/([A-Za-z0-9._:-]+\.json)")

#: Attempt-terminal statuses (``author_attempts``): anything else is in-flight
#: author work and protects the whole work dir.
_TERMINAL_ATTEMPT_STATUSES = frozenset({"completed", "failed", "superseded"})

#: Files and dirs inside every attempt that are re-read later: the retry
#: feedback channel, session resume, discovery inlining, broker evidence (a
#: corpus retention source), and the terminal-evidence replay fallback.
_ATTEMPT_HISTORY_NAMES = (
    "record.json",
    "events.jsonl",
    "discovery.json",
    "evidence-pack.json",
    "broker",
)

#: Pure per-attempt scratch, reapable in every attempt once the work dir
#: clears: session cwd and quarantined late writes.
_ATTEMPT_SCRATCH_NAMES = ("scratch", "quarantine")


class ReapPlanError(RuntimeError):
    """Raised when plan inputs are structurally unusable."""


@dataclass(frozen=True)
class ProtectedEntry:
    """One path a reap must preserve, with its manifest class."""

    path: Path
    category: str
    reason: str


@dataclass(frozen=True)
class ReapCandidate:
    """One reapable path with its tier, owner, and measured size."""

    path: Path
    tier: str
    stable_id: Optional[str]
    bytes_len: int
    reason: str


@dataclass(frozen=True)
class ReapApplyReport:
    """What one apply pass actually removed and skipped."""

    removed: tuple[Path, ...]
    skipped: tuple[tuple[Path, str], ...]
    bytes_removed: int


@dataclass(frozen=True)
class ReapPlan:
    """One complete report-only reap computation."""

    work_root: Path
    records_root: Path
    runtime_root: Path
    env_state_root: Optional[Path]
    envs_root: Path
    environment_target: str
    tiers: tuple[str, ...]
    candidates: tuple[ReapCandidate, ...] = ()
    protected: tuple[ProtectedEntry, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def tier_totals(self) -> dict[str, tuple[int, int]]:
        """Per-tier ``(candidate count, candidate bytes)``."""

        totals: dict[str, tuple[int, int]] = {}
        for candidate in self.candidates:
            count, size = totals.get(candidate.tier, (0, 0))
            totals[candidate.tier] = (count + 1, size + candidate.bytes_len)
        return totals

    def protected_totals(self) -> dict[str, int]:
        """Per-category count of manifest entries."""

        totals: dict[str, int] = {}
        for entry in self.protected:
            totals[entry.category] = totals.get(entry.category, 0) + 1
        return totals

    def to_json(self) -> JsonObject:
        """Portable report payload (paths as strings, applied=False)."""

        return {
            "plan_version": REAP_PLAN_VERSION,
            "applied": False,
            "roots": {
                "work_root": str(self.work_root),
                "records_root": str(self.records_root),
                "runtime_root": str(self.runtime_root),
                "env_state_root": (
                    str(self.env_state_root) if self.env_state_root is not None else None
                ),
                "envs_root": str(self.envs_root),
                "environment_target": self.environment_target,
            },
            "tiers": {
                tier: {"candidates": count, "bytes": size}
                for tier, (count, size) in sorted(self.tier_totals().items())
            },
            "candidates": [
                {
                    "path": str(candidate.path),
                    "tier": candidate.tier,
                    "stable_id": candidate.stable_id,
                    "bytes": candidate.bytes_len,
                    "reason": candidate.reason,
                }
                for candidate in self.candidates
            ],
            "protected_summary": {
                category: count for category, count in sorted(self.protected_totals().items())
            },
            "notes": list(self.notes),
        }


def _tree_bytes(path: Path) -> int:
    if path.is_symlink():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    if path.is_dir():
        for child in path.rglob("*"):
            try:
                if child.is_file() and not child.is_symlink():
                    total += child.stat().st_size
            except OSError:
                continue
    return total


def current_model_records(records_root: Union[str, Path]) -> dict[str, JsonObject]:
    """Current model revision per stable id, parentage-validated."""

    shard = Path(records_root) / "models" / "current-shard.jsonl"
    if not shard.is_file():
        return {}
    return _select_current(scan_jsonl(shard, validate=False))


def parked_stable_ids(records_root: Union[str, Path]) -> frozenset[str]:
    """Stable ids named by ANY intake-extension valve row.

    Deliberately fail-closed: release predicates are not consulted, so a model
    that ever parked stays protected until its rows age out of the valves.
    Over-protection costs bytes; under-protection costs replay evidence.
    """

    extensions = Path(records_root) / "intake-extensions"
    if not extensions.is_dir():
        return frozenset()
    named: set[str] = set()
    for ledger in sorted(extensions.glob("*.jsonl")):
        for row in scan_jsonl(ledger, validate=False):
            stable_id = row.get("stable_id")
            if isinstance(stable_id, str) and stable_id:
                named.add(stable_id)
    return frozenset(named)


def referenced_diagnostic_names(records_root: Union[str, Path]) -> frozenset[str]:
    """Diagnostics sidecar filenames any canonical ledger row points at.

    Scans raw ledger text: the schema pins the relative-path shape, and a raw
    scan cannot miss a pointer a nested row shape would hide.
    """

    records = Path(records_root)
    corpus = corpus_root(records)
    names: set[str] = set()
    for ledger in sorted(records.rglob("*.jsonl")):
        if corpus == ledger or corpus in ledger.parents:
            continue
        try:
            text = ledger.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names.update(_DIAGNOSTIC_REFERENCE_PATTERN.findall(text))
    return frozenset(names)


def _attempt_dirs(author_root: Path) -> list[Path]:
    attempts_root = author_root / "attempts"
    if not attempts_root.is_dir():
        return []
    found = [
        entry
        for entry in attempts_root.iterdir()
        if entry.is_dir() and _ATTEMPT_DIR_PATTERN.match(entry.name)
    ]
    return sorted(found, key=lambda entry: int(entry.name.split("-")[1]))


def _attempt_status(attempt_dir: Path) -> str:
    record = attempt_dir / "record.json"
    if not record.is_file():
        return "unknown"
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    status = payload.get("status")
    return status if isinstance(status, str) else "unknown"


def _model_work_dirs(work_root: Path) -> list[Path]:
    if not work_root.is_dir():
        return []
    dirs = []
    for entry in sorted(work_root.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            continue
        if (entry / "author").is_dir():
            dirs.append(entry)
    return dirs


def work_dir_eligibility(
    work_dir: Path,
    *,
    current_records: Mapping[str, JsonObject],
    parked: frozenset[str],
    records_root: Path,
) -> tuple[bool, str, str]:
    """Fail-closed reap eligibility for one model work dir.

    Returns ``(eligible, category, reason)``. Eligible means: the model's
    current canonical record is terminal, no park valve names it, no attempt
    record is in flight, and the corpus reap clearance clears the dir. Every
    other state protects the work dir whole.
    """

    stable_id = work_dir.name
    record = current_records.get(stable_id)
    status_code = (
        str(record.get("status", {}).get("code", "")) if isinstance(record, Mapping) else ""
    )
    if record is None or status_code not in TERMINAL_STATUS_CODES:
        return (
            False,
            "live-work-dir",
            f"status {status_code or '<no canonical record>'} is not terminal",
        )
    if stable_id in parked:
        return (False, "parked-work-dir", "an intake-extension valve names this model")
    for attempt in _attempt_dirs(work_dir / "author"):
        status = _attempt_status(attempt)
        if status not in _TERMINAL_ATTEMPT_STATUSES:
            return (
                False,
                "live-work-dir",
                f"attempt {attempt.name} is in flight (status {status!r})",
            )
    clearance = work_dir_reap_clearance(work_dir, records_root)
    if not clearance.cleared:
        return (
            False,
            "unpromoted-work-dir",
            "; ".join(clearance.reasons) or "corpus clearance refused",
        )
    return (True, "", "")


def referenced_path_manifest(
    *,
    work_root: Union[str, Path],
    records_root: Union[str, Path],
    runtime_root: Union[str, Path],
    env_state_root: Optional[Union[str, Path]] = None,
) -> tuple[ProtectedEntry, ...]:
    """Everything a reap must preserve, recomputed from canonical state."""

    work = Path(work_root)
    records = Path(records_root)
    runtime = Path(runtime_root)
    entries: list[ProtectedEntry] = []

    entries.append(
        ProtectedEntry(
            corpus_root(records),
            "corpus",
            "durable text corpus: never reapable under any policy",
        )
    )
    entries.append(
        ProtectedEntry(
            records,
            "canonical-records",
            "append-only ledgers, intake snapshots, partitions; checkpoint-enforced",
        )
    )
    for name, reason in (
        ("campaign-configs", "unattended supervised resume reads these"),
        ("locks", "advisory driver and worker locks"),
        ("mirrors", "durable CAS: inventory must exactly match the artifact ledger"),
        ("instrumentation", "append-only telemetry consumed by throughput reports"),
        ("intake", "operator intake material"),
        ("wakeup", "wakeup receipts"),
    ):
        entries.append(ProtectedEntry(runtime / name, "runtime-control", reason))
    for name in ("driver-state.json", "state.sqlite", "requeue-grants.jsonl"):
        entries.append(ProtectedEntry(runtime / name, "runtime-control", "driver runtime state"))
    for name in referenced_diagnostic_names(records):
        entries.append(
            ProtectedEntry(
                runtime / "diagnostics" / name,
                "diagnostics",
                "ledger row dereferences this sidecar (recipe-refusal kernel)",
            )
        )

    current = current_model_records(records)
    parked = parked_stable_ids(records)
    for work_dir in _model_work_dirs(work):
        eligible, category, reason = work_dir_eligibility(
            work_dir, current_records=current, parked=parked, records_root=records
        )
        if not eligible:
            entries.append(ProtectedEntry(work_dir, category, reason))
            continue
        entries.append(
            ProtectedEntry(
                work_dir / "repair",
                "repair-budget",
                "charge globs ARE the live repair budget; reaping refunds spent LLM budget",
            )
        )
        entries.append(
            ProtectedEntry(
                work_dir / "driver-author-artifact.json",
                "driver-author-cache",
                "deleting this re-buys a full author session",
            )
        )
        # W9 retire-by-rename (A20/IC-Q1): an invalidated author cache is
        # renamed to a typed .stale-<event_id> sibling as durable evidence
        # of the retirement cause; the evidence is protected, never reaped.
        for retired in sorted(work_dir.glob("driver-author-artifact.json.stale-*")):
            entries.append(
                ProtectedEntry(
                    retired,
                    "driver-author-cache",
                    "retired cache evidence (W9 retire-by-rename); names its retirement row",
                )
            )
        author_root = work_dir / "author"
        for child in sorted(author_root.iterdir()):
            if child.name == "attempts":
                continue
            entries.append(
                ProtectedEntry(
                    child,
                    "author-custody",
                    "durable author surface: retention candidates and evidence replay",
                )
            )
        for attempt in _attempt_dirs(author_root):
            for name in _ATTEMPT_HISTORY_NAMES:
                target = attempt / name
                if target.exists():
                    entries.append(
                        ProtectedEntry(
                            target,
                            "attempt-history",
                            "re-read on re-drive: retry feedback, session resume, evidence",
                        )
                    )

    if env_state_root is not None:
        state_root = Path(env_state_root)
        registrations_root = state_root / "prefix-registrations"
        if registrations_root.is_dir():
            for registration_path in sorted(registrations_root.glob("*.json")):
                entries.append(
                    ProtectedEntry(
                        registration_path,
                        "env-live",
                        "prefix registration: the removal authority for its generation",
                    )
                )
                try:
                    payload = json.loads(registration_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                for key in ("package_cache", "wheelhouse"):
                    value = payload.get(key)
                    if isinstance(value, str) and value:
                        entries.append(
                            ProtectedEntry(
                                Path(value),
                                "env-live",
                                f"registered generation-local {key.replace('_', ' ')}",
                            )
                        )

    return tuple(entries)


def build_reap_plan(
    *,
    work_root: Union[str, Path],
    records_root: Union[str, Path],
    runtime_root: Union[str, Path],
    env_state_root: Optional[Union[str, Path]] = None,
    envs_root: Union[str, Path] = DEFAULT_ENVS_ROOT,
    environment_target: str = "osx-arm64",
    tiers: Sequence[str] = REAP_TIERS,
) -> ReapPlan:
    """Compute one tiered, report-only reap plan.

    Every candidate is admitted only after (a) the corpus guard in BOTH
    containment directions, and (b) a containment check against every
    referenced-path manifest entry. Computation never deletes anything.
    """

    work = Path(work_root)
    records = Path(records_root)
    runtime = Path(runtime_root)
    env_root = Path(env_state_root) if env_state_root is not None else None
    committed_envs_root = Path(envs_root)
    selected = tuple(tiers)
    unknown = set(selected) - set(REAP_TIERS)
    if unknown:
        raise ReapPlanError(f"unknown reap tiers: {sorted(unknown)}")

    protected = referenced_path_manifest(
        work_root=work,
        records_root=records,
        runtime_root=runtime,
        env_state_root=env_root,
    )
    shields = [Path(entry.path).resolve() for entry in protected]
    corpus = corpus_root(records).resolve()
    notes: list[str] = []
    candidates: list[ReapCandidate] = []

    def _admit(path: Path, tier: str, stable_id: Optional[str], reason: str) -> None:
        resolved = path.resolve()
        try:
            assert_not_corpus_path(resolved, records)
            assert_not_corpus_path(path, records)
        except CorpusReapRefusal:
            notes.append(f"refused (corpus): {path}")
            return
        if resolved in corpus.parents or resolved == corpus:
            notes.append(f"refused (contains corpus): {path}")
            return
        for shield in shields:
            if resolved == shield or shield in resolved.parents or resolved in shield.parents:
                return
        candidates.append(
            ReapCandidate(
                path=path,
                tier=tier,
                stable_id=stable_id,
                bytes_len=_tree_bytes(path),
                reason=reason,
            )
        )

    eligible_dirs: list[Path] = []
    if TIER_PROMOTED_ATTEMPT_SCRATCH in selected or TIER_INSTRUMENTATION in selected:
        current = current_model_records(records)
        parked = parked_stable_ids(records)
        for work_dir in _model_work_dirs(work):
            eligible, _, _ = work_dir_eligibility(
                work_dir, current_records=current, parked=parked, records_root=records
            )
            if eligible:
                eligible_dirs.append(work_dir)

    if TIER_PROMOTED_ATTEMPT_SCRATCH in selected:
        for work_dir in eligible_dirs:
            attempts = _attempt_dirs(work_dir / "author")
            newest = attempts[-1] if attempts else None
            for attempt in attempts:
                superseded = attempt != newest
                for name in _ATTEMPT_SCRATCH_NAMES:
                    target = attempt / name
                    if target.exists() or target.is_symlink():
                        _admit(
                            target,
                            TIER_PROMOTED_ATTEMPT_SCRATCH,
                            work_dir.name,
                            "per-attempt scratch; promoted and terminal",
                        )
                for probe in sorted(attempt.glob("executor-probe-*")):
                    _admit(
                        probe,
                        TIER_PROMOTED_ATTEMPT_SCRATCH,
                        work_dir.name,
                        "executor probe scratch; promoted and terminal",
                    )
                if superseded:
                    staged = attempt / "model"
                    if staged.exists() or staged.is_symlink():
                        _admit(
                            staged,
                            TIER_PROMOTED_ATTEMPT_SCRATCH,
                            work_dir.name,
                            "superseded attempt's staged model copy",
                        )

    if TIER_DIAGNOSTICS_ORPHANS in selected:
        referenced = referenced_diagnostic_names(records)
        diagnostics = runtime / "diagnostics"
        if diagnostics.is_dir():
            for sidecar in sorted(diagnostics.glob("*.json")):
                if sidecar.name in referenced:
                    continue
                _admit(
                    sidecar,
                    TIER_DIAGNOSTICS_ORPHANS,
                    None,
                    "no canonical ledger row references this sidecar",
                )

    if TIER_INSTRUMENTATION in selected:
        for work_dir in eligible_dirs:
            for log in sorted(work_dir.glob("forward/*/*/supervisor/*.log")):
                _admit(
                    log,
                    TIER_INSTRUMENTATION,
                    work_dir.name,
                    "supervisor stream log; ledger retention pointer is operator triage only",
                )
            for audit in sorted(work_dir.glob("forward/*/*/.menagerie-parent-audit-*")):
                _admit(
                    audit,
                    TIER_INSTRUMENTATION,
                    work_dir.name,
                    "sandbox syscall telemetry for a completed forward",
                )

    if TIER_DERIVED_CACHES in selected:
        checkpoint_cache = runtime / "derived" / "checkpoint-cache"
        if checkpoint_cache.is_dir():
            _admit(
                checkpoint_cache,
                TIER_DERIVED_CACHES,
                None,
                "documented safe-to-delete derived cache",
            )
        rehydrated = work / "rehydrated-artifacts"
        if rehydrated.is_dir():
            _admit(
                rehydrated,
                TIER_DERIVED_CACHES,
                None,
                "rebuildable from the durable mirror store",
            )

    if TIER_ENV_ORPHANS in selected and env_root is not None and env_root.is_dir():
        from menagerie.crawler.operator_environment import (
            environment_cas_reachability,
            sweep_environment_orphans,
        )

        reachability = environment_cas_reachability(committed_envs_root, target=environment_target)
        for orphan in sweep_environment_orphans(env_root, cas_reachability=reachability):
            _admit(
                Path(orphan.path),
                TIER_ENV_ORPHANS,
                None,
                f"{orphan.kind}: {orphan.detail}",
            )

    return ReapPlan(
        work_root=work,
        records_root=records,
        runtime_root=runtime,
        env_state_root=env_root,
        envs_root=committed_envs_root,
        environment_target=environment_target,
        tiers=selected,
        candidates=tuple(candidates),
        protected=protected,
        notes=tuple(notes),
    )


def apply_reap_plan(
    plan: ReapPlan,
    *,
    work_root: Union[str, Path],
    records_root: Union[str, Path],
    runtime_root: Union[str, Path],
    env_state_root: Optional[Union[str, Path]] = None,
    envs_root: Union[str, Path] = DEFAULT_ENVS_ROOT,
    environment_target: str = "osx-arm64",
    owner: Optional[Mapping[str, Any]] = None,
) -> ReapApplyReport:
    """Apply one plan under the driver lock, re-deriving every verdict.

    The whole plan is recomputed first: only candidates present in BOTH the
    provided and the fresh plan are deleted, each behind a final corpus guard
    on its resolved location. Symlinks are unlinked, never followed. Raises
    :class:`DriverLockError` when a live driver holds the runtime lock.
    """

    from menagerie.crawler.driver_contracts import DriverLock

    runtime = Path(runtime_root)
    records = Path(records_root)
    lock = DriverLock(
        runtime / "locks" / "driver.lock",
        dict(owner) if owner is not None else {"tool": "menagerie.crawler.reaper"},
    )
    removed: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    bytes_removed = 0

    with lock:
        fresh = build_reap_plan(
            work_root=work_root,
            records_root=records,
            runtime_root=runtime,
            env_state_root=env_state_root,
            envs_root=envs_root,
            environment_target=environment_target,
            tiers=plan.tiers,
        )
        fresh_by_path = {str(candidate.path): candidate for candidate in fresh.candidates}
        for candidate in plan.candidates:
            key = str(candidate.path)
            fresh_candidate = fresh_by_path.get(key)
            if fresh_candidate is None:
                skipped.append((Path(key), "stale: the fresh plan no longer admits it"))
                continue
            target = Path(key)
            try:
                assert_not_corpus_path(target, records)
                if not target.is_symlink():
                    assert_not_corpus_path(target.resolve(), records)
            except CorpusReapRefusal:
                skipped.append((target, "refused: corpus path"))
                continue
            if candidate.tier == TIER_ENV_ORPHANS:
                from menagerie.crawler.operator_environment import (
                    EnvironmentOrphan,
                    environment_cas_reachability,
                    remove_environment_orphan,
                )

                kind = fresh_candidate.reason.split(":", 1)[0]
                orphan = EnvironmentOrphan(
                    path=target,
                    kind=kind,
                    bytes_len=fresh_candidate.bytes_len,
                    detail=fresh_candidate.reason,
                )
                reachability = environment_cas_reachability(
                    Path(envs_root), target=environment_target
                )
                if env_state_root is None or not remove_environment_orphan(
                    Path(env_state_root), orphan, cas_reachability=reachability
                ):
                    skipped.append((target, "env orphan no longer removable"))
                    continue
                removed.append(target)
                bytes_removed += fresh_candidate.bytes_len
                continue
            if target.is_symlink() or target.is_file():
                target.unlink(missing_ok=True)
            elif target.is_dir():
                shutil.rmtree(target)
            else:
                skipped.append((target, "vanished before removal"))
                continue
            removed.append(target)
            bytes_removed += fresh_candidate.bytes_len

    return ReapApplyReport(
        removed=tuple(removed),
        skipped=tuple(skipped),
        bytes_removed=bytes_removed,
    )
