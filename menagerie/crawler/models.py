"""Typed value objects shared by crawler integrity components."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from menagerie.crawler.constants import FailureStage, StatusKind

JsonValue = Any
JsonObject = dict[str, JsonValue]


#: The manifest key carrying the ONE supplementary broker pack the executor may
#: grant an author session mid-flight. It is kept OUT of ``sources`` on purpose:
#: ``manifest_sha256 == stable_hash(sources)`` is the identity the author echoed
#: back at dispatch, and appending to ``sources`` afterwards would move it and
#: terminalize a good result on an identity the author was never shown.
SUPPLEMENTARY_SOURCES_KEY = "supplementary_sources"


def manifest_source_rows(source_manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return every machine-fetched source row one author session may cite.

    ONE reading of "which sources exist" for every consumer that cares about
    BYTES rather than about identity: evidence grounding, private byte staging,
    and staged-artifact coverage. The frozen ``sources`` rows come first, then
    any supplementary rows the lane ingested after the executor granted its one
    supplementary broker round. Both sets are broker-fetched and digest-pinned;
    the split exists only so the bound manifest identity cannot move.

    Consumers that need the manifest IDENTITY must keep reading ``sources``
    directly -- this function is deliberately not a drop-in for that.

    Parameters
    ----------
    source_manifest:
        Frozen manifest, optionally carrying ``supplementary_sources``.

    Returns
    -------
    list[Mapping[str, Any]]
        Frozen rows followed by supplementary rows, object rows only.
    """

    frozen = source_manifest.get("sources")
    supplementary = source_manifest.get(SUPPLEMENTARY_SOURCES_KEY)
    rows: list[Mapping[str, Any]] = []
    for group in (frozen, supplementary):
        if isinstance(group, list):
            rows.extend(row for row in group if isinstance(row, Mapping))
    return rows


def bounded_json_repr(value: object, *, limit: int = 120) -> str:
    """Render one offending value compactly and boundedly for an error message.

    Shared by every lane that refuses a model-supplied machine-owned value, so a
    refusal reads the same whether it came from the checker gate or the author
    proposal.

    Parameters
    ----------
    value:
        Model-supplied or machine-owned value.
    limit:
        Maximum rendered characters.

    Returns
    -------
    str
        Compact JSON rendering, truncated with an explicit marker.
    """

    try:
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = repr(value)
    if len(rendered) > limit:
        return rendered[:limit] + "...<truncated>"
    return rendered


@dataclass(frozen=True)
class IdentitySet:
    """Current dependency identities for one model.

    Parameters
    ----------
    source, evidence, recipe, environment, fidelity, vet, execution:
        Canonical SHA-256 identities. Optional values represent inapplicable facts.
    """

    source: str
    evidence: str
    recipe: str
    environment: str
    fidelity: Optional[str]
    vet: Optional[str]
    execution: str


@dataclass(frozen=True)
class StalenessReport:
    """Names identity products invalidated by an input change.

    Parameters
    ----------
    stale:
        Frozen names of stale identities or dependent facts.
    """

    stale: frozenset[str]

    def is_stale(self, name: str) -> bool:
        """Return whether an identity or dependent fact is stale.

        Parameters
        ----------
        name:
            Identity/fact name to query.

        Returns
        -------
        bool
            True when the named value must be recomputed.
        """

        return name in self.stale


@dataclass(frozen=True)
class TailRecoveryEvidence:
    """Evidence retained before truncating a torn final JSONL line.

    Parameters
    ----------
    ledger_path:
        Recovered ledger.
    byte_offset:
        First byte of the incomplete tail.
    byte_count:
        Number of removed bytes.
    tail_sha256:
        Hash of the exact removed bytes.
    recovered_at:
        RFC 3339 UTC timestamp.
    evidence_path:
        Append-only recovery evidence ledger.
    """

    ledger_path: Path
    byte_offset: int
    byte_count: int
    tail_sha256: str
    recovered_at: str
    evidence_path: Path


@dataclass(frozen=True)
class AppendResult:
    """Result of an idempotent ledger append.

    Parameters
    ----------
    record:
        Persisted record, including assigned sequence/hash fields.
    appended:
        False for a byte-equivalent logical replay.
    """

    record: JsonObject
    appended: bool


@dataclass(frozen=True, init=False)
class LedgerPaths:
    """Paths to the four canonical fact ledgers.

    Parameters
    ----------
    models, attempts, gates, artifacts:
        JSONL paths for canonical model revisions, attempts, gates, and
        artifact-authority events. The artifact path is derived from the model
        shard when omitted so existing read-only callers remain source compatible.
    """

    models: Path
    attempts: Path
    gates: Path
    artifacts: Path

    def __init__(
        self,
        models: Path,
        attempts: Path,
        gates: Path,
        artifacts: Optional[Path] = None,
    ) -> None:
        """Initialize canonical ledger paths.

        Parameters
        ----------
        models, attempts, gates:
            Existing canonical ledger paths.
        artifacts:
            Artifact-event ledger path. When omitted, use the model shard name
            below the sibling ``artifacts`` directory.
        """

        object.__setattr__(self, "models", models)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "gates", gates)
        records_root = models.parent.parent if models.parent.name == "models" else models.parent
        resolved_artifacts = artifacts or records_root / "artifacts" / models.name
        object.__setattr__(self, "artifacts", resolved_artifacts)

    @property
    def records_root(self) -> Path:
        """Return the records root the model shard implies."""

        parent = self.models.parent
        return parent.parent if parent.name == "models" else parent


@dataclass(frozen=True)
class PartitionReport:
    """Exact terminal partition of an intake snapshot.

    Parameters
    ----------
    intake_ids:
        Stable IDs in the trusted intake snapshot.
    buckets:
        Terminal status-code to stable-ID mapping.
    missing_ids, extra_ids, duplicate_ids:
        Violations of exact coverage or pairwise disjointness.
    machine_deferred:
        Deferral-ledger disposition to stable-ID mapping for models with no
        canonical record that a machine-derived deferral valve row accounts.
        Every intake model is terminal XOR machine-deferral-claimed; a claimed
        model is never also in ``missing_ids``.
    superseded_deferral_ids:
        Models whose deferral valve row is superseded by a canonical terminal
        record. The record wins; the stale row accounts for nothing.
    grant_pending_ids:
        Models with a LIVE grant-pending supersession park: a completed
        superseding revision awaiting an operator requeue grant over a
        review-held terminal. Includes both claiming rows (the review-held
        terminal is dependency-stale) and rows coexisting with their exact
        still-current review-held terminal; a park a newer revision consumed
        appears in ``superseded_deferral_ids`` instead.
    refusal_parked_ids:
        Models with a LIVE terminal-refusal park: a terminal the recording
        path refused to record (intended append and minimal fallback both
        refused), machine-withheld pending a root-cause fix plus an operator
        decision. Accounted revision-aware exactly like ``grant_pending_ids``;
        a park a newer landed revision consumed appears in
        ``superseded_deferral_ids`` instead.
    stale_by_axis:
        In-scope members holding a LEDGER HEAD the dependency-currency
        projection excluded, bucketed by the closed stale axis that excluded
        it (BD-1). Coverage counts these members -- a recorded terminal is
        never a coverage hole -- and this bucket carries the reason instead.
        It is the full disclosure census, and it is a VIOLATION bucket for
        every member NOTHING accounts: raw heads include records the projection
        rejects for cause (a contradicted evidence re-pass, a failed family
        currency, a ``ReductionError``). The step-100 boundary reported all of
        those crudely as "missing", which failed closed; a report-only stale
        bucket would convert that fail-closed tripwire into a green partition
        with a footnote, which the LOCKED Validation-Integrity principle
        forbids. See :attr:`unaccounted_stale_ids`.
    stale_deferred:
        Disposition to stable IDs for members whose head is STALE and whose
        live valve row is the accounting for the withheld RE-DRIVE (an
        environment-rebuild deferral after a checker-digest rotation is the
        worked example). The record stands in history; the row is neither
        superseded nor a coverage claim. Disclosed, counted as a deferral by
        the close, and never counted as audited.
    externally_accounted_stale_ids:
        Stale members a DURABLE row the caller read accounts for, where that
        row is deliberately not a run-scoped claim (the driver's persisted park
        rows: park liveness is a fact about the ledgers, and a park written this
        run may not be in the run's claim stash at all). Suppresses only the
        stale refusal; it never satisfies coverage and never leaves the census.
    """

    intake_ids: frozenset[str]
    buckets: Mapping[str, frozenset[str]]
    missing_ids: frozenset[str]
    extra_ids: frozenset[str]
    duplicate_ids: frozenset[str]
    machine_deferred: Mapping[str, frozenset[str]] = field(default_factory=dict)
    superseded_deferral_ids: frozenset[str] = frozenset()
    grant_pending_ids: frozenset[str] = frozenset()
    refusal_parked_ids: frozenset[str] = frozenset()
    stale_by_axis: Mapping[str, frozenset[str]] = field(default_factory=dict)
    stale_deferred: Mapping[str, frozenset[str]] = field(default_factory=dict)
    externally_accounted_stale_ids: frozenset[str] = frozenset()

    @property
    def stale_ids(self) -> frozenset[str]:
        """Return every in-scope member excluded from the live projection.

        Returns
        -------
        frozenset[str]
            Union of the per-axis stale buckets.
        """

        return frozenset(
            stable_id for stable_ids in self.stale_by_axis.values() for stable_id in stable_ids
        )

    @property
    def stale_deferred_ids(self) -> frozenset[str]:
        """Return every stale-headed member a live valve row accounts.

        Returns
        -------
        frozenset[str]
            Union of the per-disposition stale-deferred buckets.
        """

        return frozenset(
            stable_id for stable_ids in self.stale_deferred.values() for stable_id in stable_ids
        )

    @property
    def unaccounted_stale_ids(self) -> frozenset[str]:
        """Return the stale members NOTHING accounts -- the refusing arm.

        A stale head is a fact the boundary must not bury either way. Where a
        durable row already accounts the member (a live park awaiting an
        operator decision, or a valve row withholding the re-drive), the state
        is disclosed and the close carries its non-``complete`` suffix, exactly
        as it did before coverage moved to the ledger -- no verdict is upgraded.
        Where NOTHING accounts it, the partition is invalid: that is the whole
        step-100 population (24 recorded terminals, 7 of them awards, reported
        as phantom coverage holes) and it must keep failing closed, now with the
        axis named.

        Returns
        -------
        frozenset[str]
            Stale members with no live park and no live valve row.
        """

        return (
            self.stale_ids
            - self.stale_deferred_ids
            - self.grant_pending_ids
            - self.refusal_parked_ids
            - self.externally_accounted_stale_ids
        )

    @property
    def valid(self) -> bool:
        """Return whether the current records exactly partition intake.

        Returns
        -------
        bool
            True for complete, pairwise-disjoint coverage with no unaccounted
            member whose recorded head the live projection cannot vouch for.
        """

        return not (
            self.missing_ids or self.extra_ids or self.duplicate_ids or self.unaccounted_stale_ids
        )


@dataclass(frozen=True)
class CompletenessReport:
    """Crawl completion and funnel diagnostics.

    Parameters
    ----------
    partition:
        Terminal partition report.
    incomplete_by_issue:
        Completeness issue to affected stable IDs.
    workflow_counts:
        Nonterminal workflow/operational rows still pending.
    funnel_counts:
        Queryable current-record aggregate counts.
    complete:
        Whether every canonical completion gate represented here passes.
    """

    partition: PartitionReport
    incomplete_by_issue: Mapping[str, Tuple[str, ...]]
    workflow_counts: Mapping[str, int]
    funnel_counts: Mapping[str, int]
    complete: bool


@dataclass(frozen=True)
class TerminalStatus:
    """Normalized closed terminal status.

    Parameters
    ----------
    kind:
        Public status kind.
    code:
        Full public terminal code.
    stage, reason_code:
        Required failure taxonomy values for failed records.
    """

    kind: StatusKind
    code: str
    stage: Optional[FailureStage] = None
    reason_code: Optional[str] = None


@dataclass
class RebuildSummary:
    """Summary of a disposable SQLite rebuild.

    Parameters
    ----------
    intake_count, model_revision_count, attempt_count, gate_count, artifact_event_count:
        Inserted row counts.
    current_count:
        Materialized current model count.
    """

    intake_count: int = 0
    model_revision_count: int = 0
    attempt_count: int = 0
    gate_count: int = 0
    artifact_event_count: int = 0
    current_count: int = 0


@dataclass(frozen=True)
class FunnelQuery:
    """Optional filters for current-record funnel reports.

    Parameters
    ----------
    framework, rung, status_code:
        Exact current-record filters. None leaves a dimension unrestricted.
    flags:
        All flags that must be present.
    """

    framework: Optional[str] = None
    rung: Optional[str] = None
    status_code: Optional[str] = None
    flags: Sequence[str] = field(default_factory=tuple)
