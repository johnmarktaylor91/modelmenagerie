"""Merge same-campaign host shards as read-only derived projections.

A REALISTIC completed shard is not a tray of uniformly clean terminals: the
admission valves withhold models BY DESIGN (their durable intake-extension
rows are the only truthful accounting for those models), review-held terminals
await an operator decision, and dependency-stale records await their re-drive
under current dependencies. This merge therefore proves, per shard, that
every assigned model is terminal XOR machine-deferral-claimed -- the same
partition contract the driver close enforces -- and carries the held and stale
dispositions into the merged report instead of refusing them. What it never
does is relax the tripwire: a model with neither a canonical record nor a
valid valve row still refuses, exactly as it does at close.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.authority import build_authority_context
from menagerie.crawler.campaign_merge import (
    CampaignMergeError,
    _assert_author_binding,
    _canonical_input_paths,
    _environment_generations,
    _status_report,
    _view_payloads,
)
from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.driver_admission import _persisted_deferral_rows
from menagerie.crawler.grant_parks import GRANT_PENDING_DISPOSITION
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    fsync_directory,
    hash_bytes,
)
from menagerie.crawler.intake import IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.models import JsonObject
from menagerie.crawler.partitioner import DEFAULT_CAMPAIGN_MANIFEST, load_campaign_bindings
from menagerie.crawler.recordio import LedgerError, scan_jsonl
from menagerie.crawler.reducer import default_ledger_paths, project_dependency_current
from menagerie.crawler.refusal_parks import REFUSAL_PARK_DISPOSITION
from menagerie.crawler.rung_staging import (
    registered_prior_intake_members,
    registered_prior_intake_snapshots,
)
from menagerie.crawler.status import (
    checkpoint_consistency_report,
    completeness_report,
    machine_deferral_rejection,
    partition_report,
)
from menagerie.crawler.text_corpus import union_corpus
from menagerie.crawler.tools.handoff_bundle import (
    HandoffBundleError,
    verify_handoff_manifest,
)


HOST_SHARD_COLLISION_ERROR = "stable_id_collision"
ENV_AXIS_DIVERGENT_ERROR = "env_axis_divergent"
HOST_SHARD_MERGE_MANIFEST_NAME = "host-shard-merge-manifest.json"
HOST_SHARD_MERGE_MANIFEST_FORMAT = "menagerie.crawler.host-shard-merge-manifest.v1"
HOST_SHARD_MERGE_REPORT_FORMAT = "menagerie.crawler.host-shard-merge-report.v2"

_PARK_DISPOSITIONS = frozenset({GRANT_PENDING_DISPOSITION, REFUSAL_PARK_DISPOSITION})

# The CLOSED set of completeness issues a LIVE human-review hold may carry
# through a merge. These are exactly the gates an operator adjudication is
# still owed on -- the review hold is the mechanism holding them open -- so
# carrying them (disclosed, segregated, never counted as clean) states the
# truth. Everything outside this set (schema validity, malformed status
# evidence, unexpected recorded issues, stale family variants, partition
# violations) is a defect no review hold explains, and refuses. The carve is
# additionally scoped: an issue is carriable only when EVERY affected record
# actually holds a live review-hold flag.
_REVIEW_HELD_CARRIABLE_ISSUES = frozenset(
    {
        "human_review_pending",
        "mandatory_source_present",
        "source_read_fields_complete",
        "evidence_coverage_complete",
        "accuracy_gate_current",
        "required_fidelity_current",
        "execution_current",
        "family_template_valid",
        "release_eligible",
        "meaningful_mode_outcomes_incomplete",
        "derived_release_ineligible",
    }
)


@dataclass(frozen=True)
class HostShardSource:
    """Physical inputs for one completed host-local shard.

    Parameters
    ----------
    shard_id:
        Caller-stable host/shard label used in diagnostics and fingerprints.
    records_root:
        Host-local canonical records root.
    runtime_root:
        Host-local runtime root used only for report provenance.
    intake_root:
        Immutable intake snapshot that the shard was assigned.
    handoff_manifest:
        Optional source-host handoff manifest; when present the received tree
        re-verifies byte-for-byte before it is allowed into the merge.
    """

    shard_id: str
    records_root: Path
    runtime_root: Path
    intake_root: Path
    handoff_manifest: Optional[Path] = None


@dataclass(frozen=True)
class HostShardMergeResult:
    """Validated merged same-campaign host projection.

    Parameters
    ----------
    current_records:
        Stable-ID ordered terminal current records.
    report:
        Persisted merge report.
    view_digests:
        Digest of each derived output view.
    """

    current_records: tuple[JsonObject, ...]
    report: JsonObject
    view_digests: Mapping[str, str]


@dataclass(frozen=True)
class _ShardProjection:
    """One shard's replayed dispositions under its own authority."""

    current: Mapping[str, JsonObject]
    stale_reasons: Mapping[str, str]
    review_held_ids: tuple[str, ...]
    environment_generations: Mapping[str, str]


def _read_source_models(source: HostShardSource) -> list[JsonObject]:
    """Read one shard's model ledger without selecting current rows.

    Parameters
    ----------
    source:
        Host shard source.

    Returns
    -------
    list[dict[str, Any]]
        Raw model history rows.
    """

    return scan_jsonl(default_ledger_paths(source.records_root).models, validate=False)


def _shard_input_paths(records_root: Path) -> tuple[Path, ...]:
    """Return every merge-steering canonical input below one shard root.

    Beyond the campaign canonical ledgers this includes the intake-extension
    valve ledgers and the rung registry: both now steer the merge verdict
    (deferral ownership and the prior-lineage currency carve-out), so the
    read-only proof must cover them.

    Parameters
    ----------
    records_root:
        Host-local canonical records root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Sorted canonical input paths.
    """

    fixed = set(_canonical_input_paths(records_root))
    fixed.update((records_root / "intake-extensions").glob("*.jsonl"))
    fixed.add(records_root / "intake" / "registered-rungs.jsonl")
    return tuple(sorted(fixed))


def _fingerprint_sources(sources: Sequence[HostShardSource]) -> Mapping[str, str]:
    """Fingerprint all canonical JSONL inputs without opening writers.

    Parameters
    ----------
    sources:
        Host shard sources.

    Returns
    -------
    Mapping[str, str]
        Shard-qualified path to digest or ``missing``.
    """

    fingerprints: dict[str, str] = {}
    for source in sources:
        for path in _shard_input_paths(source.records_root):
            key = f"{source.shard_id}:{path.resolve()}"
            fingerprints[key] = hash_bytes(path.read_bytes()) if path.is_file() else "missing"
    return dict(sorted(fingerprints.items()))


def _snapshot_ids(snapshot: IntakeSnapshot) -> set[str]:
    """Return stable IDs assigned to one intake snapshot.

    Parameters
    ----------
    snapshot:
        Loaded intake snapshot.

    Returns
    -------
    set[str]
        Assigned stable IDs.
    """

    return {item.stable_id for item in snapshot.items}


def _prove_shard_partition(
    expected_ids: set[str],
    snapshots_by_shard: Mapping[str, IntakeSnapshot],
) -> None:
    """Prove pairwise-disjoint shard assignment and exact expected union.

    Parameters
    ----------
    expected_ids:
        Stable IDs expected in the merged campaign-level view.
    snapshots_by_shard:
        Loaded shard snapshots keyed by shard ID.

    Raises
    ------
    CampaignMergeError
        If shard assignments overlap, omit IDs, or include extras.
    """

    owners: dict[str, list[str]] = {}
    for shard_id, snapshot in snapshots_by_shard.items():
        for stable_id in _snapshot_ids(snapshot):
            owners.setdefault(stable_id, []).append(shard_id)
    collisions = {
        stable_id: sorted(shards) for stable_id, shards in sorted(owners.items()) if len(shards) > 1
    }
    observed = set(owners)
    missing = sorted(expected_ids - observed)
    extra = sorted(observed - expected_ids)
    if collisions:
        raise CampaignMergeError(f"{HOST_SHARD_COLLISION_ERROR}: {collisions}")
    if missing or extra:
        raise CampaignMergeError(
            f"host shard partition invalid: missing_stable_ids={missing}, extra_stable_ids={extra}"
        )


def _shard_valve_rows(
    records_root: Path, assigned: set[str]
) -> tuple[dict[str, JsonObject], dict[str, str]]:
    """Read one shard's durable valve rows and select the governing claims.

    The read reuses the driver's own persisted-row reader (governing-row
    selection per valve kind) and then applies the completion partition's
    fail-closed :func:`machine_deferral_rejection` validation to every
    candidate; a model whose rows all fail accounts for nothing.

    Parameters
    ----------
    records_root:
        Shard-local records root.
    assigned:
        The shard's assigned stable IDs; foreign rows never claim here.

    Returns
    -------
    tuple[dict[str, dict[str, Any]], dict[str, str]]
        Governing valid row per stable ID, and the first rejection reason per
        stable ID whose every row failed validation.
    """

    shim = SimpleNamespace(paths=SimpleNamespace(ledgers=default_ledger_paths(records_root)))
    governing: dict[str, JsonObject] = {}
    rejected: dict[str, str] = {}
    for stable_id, rows in sorted(_persisted_deferral_rows(shim).items()):
        if stable_id not in assigned:
            continue
        reasons: list[str] = []
        for row in rows:
            reason = machine_deferral_rejection(stable_id, row)
            if reason is None:
                governing.setdefault(stable_id, row)
            else:
                reasons.append(reason)
        if stable_id not in governing and reasons:
            rejected[stable_id] = reasons[0]
    return governing, rejected


def _prove_processed_ownership(
    source: HostShardSource,
    snapshot: IntakeSnapshot,
    raw_models: Sequence[Mapping[str, Any]],
    deferral_claimed: set[str],
) -> None:
    """Prove one shard owns exactly its assigned stable IDs.

    Ownership is ``assigned == processed | deferral_claimed``: a model the
    shard's admission valves withheld has no model row BY DESIGN, and its
    validated durable valve row is the ownership statement. A model with
    neither refuses; so does a model row outside the assignment.

    Parameters
    ----------
    source:
        Host shard source.
    snapshot:
        Assigned shard snapshot.
    raw_models:
        Raw model history rows.
    deferral_claimed:
        Assigned stable IDs claimed by a validated valve row.

    Raises
    ------
    CampaignMergeError
        If any assigned model is unaccounted or any row is out of assignment.
    """

    assigned = _snapshot_ids(snapshot)
    processed = {str(record.get("stable_id")) for record in raw_models}
    missing = sorted(assigned - processed - deferral_claimed)
    extra = sorted(processed - assigned)
    if missing or extra:
        raise CampaignMergeError(
            f"{source.shard_id} processed ownership invalid: "
            f"missing_stable_ids={missing}, extra_stable_ids={extra} "
            "(missing = neither a model record nor a valid machine-deferral valve row)"
        )


def _campaign_binding(manifest_path: Path, campaign_id: str) -> Any:
    """Return the frozen binding for one campaign.

    Parameters
    ----------
    manifest_path:
        Frozen ``campaigns.json`` path.
    campaign_id:
        Campaign identifier.

    Returns
    -------
    Any
        Existing ``CampaignBinding`` object.
    """

    bindings = load_campaign_bindings(manifest_path)
    by_id = {binding.spec.campaign_id: binding for binding in bindings}
    try:
        return by_id[campaign_id]
    except KeyError as exc:
        raise CampaignMergeError(
            f"campaign {campaign_id!r} is absent from frozen manifest"
        ) from exc


def _assert_cross_shard_identity(
    shard_versions: Mapping[str, tuple[str, str]],
) -> None:
    """Require one author/checker version pair across all shards.

    Parameters
    ----------
    shard_versions:
        Shard ID to ``(author_version, checker_version)``.

    Raises
    ------
    CampaignMergeError
        If any shard was minted by a different model/prompt version.
    """

    versions = set(shard_versions.values())
    if len(versions) != 1:
        raise CampaignMergeError(
            f"host shard author/checker versions differ: {dict(sorted(shard_versions.items()))}"
        )


def _assert_environment_axis(
    generations_by_shard: Mapping[str, Mapping[str, str]],
) -> None:
    """Refuse shards whose shared environment families diverge.

    Every shard's current records were blessed under that shard's OWN latest
    durable environment generations. If two shards disagree about a family
    they both ran, their records would silently merge as jointly current while
    having executed in different environments; there is no cross-host ordering
    authority to pick a winner, so the merge refuses and the lagging shard
    owes a re-validation leg first. A family only one shard ran is no
    conflict.

    Parameters
    ----------
    generations_by_shard:
        Shard ID to that shard's environment family -> generation map.

    Raises
    ------
    CampaignMergeError
        If any family maps to more than one generation across shards.
    """

    by_family: dict[str, dict[str, list[str]]] = {}
    for shard_id, generations in sorted(generations_by_shard.items()):
        for family, generation in generations.items():
            by_family.setdefault(family, {}).setdefault(generation, []).append(shard_id)
    divergent = {
        family: {generation: sorted(shards) for generation, shards in sorted(owners.items())}
        for family, owners in sorted(by_family.items())
        if len(owners) > 1
    }
    if divergent:
        raise CampaignMergeError(
            f"{ENV_AXIS_DIVERGENT_ERROR}: records from different environment "
            f"generations must not merge as current: {divergent}"
        )


def _issue_violations(
    incomplete_by_issue: Mapping[str, Sequence[str]],
    review_held_ids: set[str],
    current_records: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return completeness issues no live review hold accounts for.

    Parameters
    ----------
    incomplete_by_issue:
        Issue name to affected stable IDs.
    review_held_ids:
        Stable IDs whose current record carries a live human-review hold.
    current_records:
        The shard's current records, keyed by stable ID -- the evidence for
        the self-disclosed no-source carve below.

    Returns
    -------
    list[str]
        Human-readable violations; empty when every issue is carriable.
    """

    violations: list[str] = []
    for issue, stable_ids in sorted(incomplete_by_issue.items()):
        if issue in _REVIEW_HELD_CARRIABLE_ISSUES and set(stable_ids) <= review_held_ids:
            continue
        if issue == "mandatory_source_present" and all(
            _self_disclosed_no_source(current_records.get(stable_id))
            for stable_id in stable_ids
        ):
            # Production intentionally writes honest no-source failure
            # terminals (failed:author with source_resolution
            # .mandatory_link_status != "ok", no review hold -- the shape
            # pinned by test_author_failure_without_source_is_honest...);
            # they are internally complete for their terminal status and
            # merge as carried disclosures. Only the SELF-DISCLOSED absence
            # is carriable: a release-track record, or a terminal claiming
            # an ok link while recording the field False, still refuses --
            # the contradiction tripwire keeps its teeth.
            continue
        violations.append(f"{issue}: {sorted(stable_ids)}")
    return violations


def _self_disclosed_no_source(record: Optional[Mapping[str, Any]]) -> bool:
    """Return whether one FAILURE record's own source_resolution discloses the absence.

    Parameters
    ----------
    record:
        Current record for the affected stable ID, when the shard holds one.
    """

    if not isinstance(record, Mapping):
        return False
    if record.get("status", {}).get("kind") != "failed":
        return False
    resolution = record.get("source_resolution")
    if not isinstance(resolution, Mapping):
        return False
    # Closed enumeration (P5 fable-F12): production writes exactly "ok" or
    # "failed" (driver_models). The carve keys on the one honest non-ok value
    # -- an unknown future status is NOT a self-disclosure and refuses the
    # merge until this set is widened in its own reviewed change.
    return resolution.get("mandatory_link_status") == "failed"


def _project_shard(
    source: HostShardSource,
    binding: Any,
    snapshot: IntakeSnapshot,
    author_version: str,
    checker_version: str,
) -> _ShardProjection:
    """Replay one shard through dependency-current under its own authority.

    Stale records are CARRIED (reason per stable ID), not refused: the model
    stays covered by its honest terminal record, which merely awaits a
    re-drive under current dependencies. Review-held terminals stay in the
    current view with their hold disclosed. Structural violations -- out-of-
    intake records, duplicates, invalid statuses, or completeness issues no
    review hold accounts for -- refuse.

    Parameters
    ----------
    source:
        Host shard source.
    binding:
        Campaign binding from ``campaigns.json``.
    snapshot:
        Shard-local assigned intake snapshot.
    author_version, checker_version:
        Frozen identity pair already proven for this shard's history.

    Returns
    -------
    _ShardProjection
        Current records, carried stale reasons, review holds, and the shard's
        environment axis.
    """

    ledgers = default_ledger_paths(source.records_root)
    attempts = scan_jsonl(ledgers.attempts)
    operational = scan_jsonl(canonical_operational_ledger_path(ledgers.models))
    environment_generations = _environment_generations(attempts, operational)
    # The registry anchor mirrors the driver's paths.intake_root shape; the
    # lineage readers only use it to find the records root and key on the
    # snapshot identity, so an out-of-tree shard snapshot directory still
    # resolves this shard's own registered lineage.
    registry_anchor = source.records_root / "intake" / snapshot.snapshot_id
    context = build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=binding.spec.author_model,
        author_version=author_version,
        checker_model=binding.spec.checker_model,
        checker_version=checker_version,
        environment_generations=environment_generations,
        prior_intake_snapshots=registered_prior_intake_snapshots(
            registry_anchor, snapshot.snapshot_id
        ),
        prior_intake_member_ids=registered_prior_intake_members(
            registry_anchor, snapshot.snapshot_id
        ),
    )
    projection = project_dependency_current(ledgers, context=context)
    review_held_ids = tuple(
        sorted(
            stable_id
            for stable_id, record in projection.current_records.items()
            if record.get("status", {}).get("human_review", {}).get("required") is True
        )
    )
    consistency = checkpoint_consistency_report(
        (item.stable_id for item in snapshot.items), projection.current_records
    )
    violations = _issue_violations(
        consistency.incomplete_by_issue, set(review_held_ids), projection.current_records
    )
    if consistency.partition.extra_ids or consistency.partition.duplicate_ids or violations:
        raise CampaignMergeError(
            f"{source.shard_id} checkpoint validation failed: "
            f"extra={sorted(consistency.partition.extra_ids)}, "
            f"duplicates={sorted(consistency.partition.duplicate_ids)}, "
            f"issues={violations}"
        )
    return _ShardProjection(
        current=projection.current_records,
        stale_reasons=dict(sorted(projection.stale_reasons.items())),
        review_held_ids=review_held_ids,
        environment_generations=environment_generations,
    )


def _verify_handoff(source: HostShardSource, campaign_id: str) -> Optional[JsonObject]:
    """Verify one shard's received bytes against its source-host manifest.

    Parameters
    ----------
    source:
        Host shard source.
    campaign_id:
        Campaign being merged.

    Returns
    -------
    dict[str, Any] | None
        Verification report, or ``None`` when no manifest was supplied.

    Raises
    ------
    CampaignMergeError
        If the manifest is malformed, misbound, or any byte disagrees.
    """

    if source.handoff_manifest is None:
        return None
    try:
        report = verify_handoff_manifest(source.records_root, source.handoff_manifest)
    except HandoffBundleError as exc:
        raise CampaignMergeError(f"{source.shard_id} handoff manifest invalid: {exc}") from exc
    if report["shard_id"] != source.shard_id or report["campaign_id"] != campaign_id:
        raise CampaignMergeError(
            f"{source.shard_id} handoff manifest is bound to "
            f"shard {report['shard_id']!r} campaign {report['campaign_id']!r}"
        )
    if not report["verified"]:
        raise CampaignMergeError(
            f"{source.shard_id} handoff verification failed: "
            f"missing={report['missing']}, mismatched={report['mismatched']}, "
            f"extra={report['extra']}"
        )
    return report


def _merge_status(
    *,
    deferred_accounted: bool,
    review_holds: bool,
    refusal_parks: bool,
    stale_records: bool,
) -> str:
    """Derive the merged view's honest closed-vocabulary status.

    Parameters
    ----------
    deferred_accounted:
        Whether non-park valve rows claimed models.
    review_holds:
        Whether review-held terminals or live grant parks merged.
    refusal_parks:
        Whether live terminal-refusal parks merged.
    stale_records:
        Whether dependency-stale records were carried (re-drives owed).

    Returns
    -------
    str
        ``complete`` or ``complete-with-...`` with suffixes in fixed order.
    """

    suffixes = []
    if deferred_accounted:
        suffixes.append("deferrals")
    if review_holds:
        suffixes.append("review-holds")
    if refusal_parks:
        suffixes.append("refusal-parks")
    if stale_records:
        suffixes.append("stale-records")
    if not suffixes:
        return "complete"
    return "complete-with-" + "-and-".join(suffixes)


def _write_views(
    output_root: Path,
    current: Sequence[JsonObject],
    report: JsonObject,
) -> Mapping[str, str]:
    """Write derived views and commit them through one manifest written last.

    Any pre-existing commit manifest is removed FIRST, so a consumer can never
    observe a half-rewritten output as committed: the manifest reappearing is
    the transaction's single commit point, and its digests must re-verify
    against the view bytes.

    Parameters
    ----------
    output_root:
        Destination root.
    current:
        Merged current records.
    report:
        Merge report.

    Returns
    -------
    Mapping[str, str]
        Relative output path to digest for every committed view.
    """

    manifest_path = output_root / HOST_SHARD_MERGE_MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()
        fsync_directory(manifest_path.parent)
    payloads = dict(_view_payloads(current, report))
    payloads["host-shard-merge-report.json"] = canonical_json_bytes(report) + b"\n"
    for relative, payload in sorted(payloads.items()):
        atomic_replace_bytes(output_root / relative, payload)
    digests = {relative: hash_bytes(payload) for relative, payload in sorted(payloads.items())}
    manifest: JsonObject = {
        "format": HOST_SHARD_MERGE_MANIFEST_FORMAT,
        "campaign_id": report["campaign_id"],
        "committed": True,
        "views": digests,
        "report_sha256": digests["host-shard-merge-report.json"],
    }
    atomic_replace_bytes(manifest_path, canonical_json_bytes(manifest) + b"\n")
    return digests


def merge_host_shards(
    manifest_path: Path,
    campaign_id: str,
    sources: Sequence[HostShardSource],
    output_root: Path,
    *,
    expected_intake_root: Optional[Path] = None,
) -> HostShardMergeResult:
    """Merge completed same-campaign host shards without mutating ledgers.

    Parameters
    ----------
    manifest_path:
        Frozen campaign manifest used to bind author/checker identities.
    campaign_id:
        Campaign being assembled from host shards.
    sources:
        One or more host-local shard sources.
    output_root:
        Derived-view destination outside source records roots.
    expected_intake_root:
        Optional campaign-level intake snapshot for pilot subsets. If omitted,
        the frozen manifest's campaign intake is used.

    Returns
    -------
    HostShardMergeResult
        Current records, report, and output digests.
    """

    if not sources:
        raise CampaignMergeError("host shard merge requires at least one source")
    source_by_id = {source.shard_id: source for source in sources}
    if len(source_by_id) != len(sources):
        raise CampaignMergeError("host shard IDs must be unique")
    ordered_sources = tuple(sorted(sources, key=lambda item: item.shard_id))
    resolved_output = output_root.resolve()
    for source in ordered_sources:
        resolved_records = source.records_root.resolve()
        if resolved_output == resolved_records or resolved_records in resolved_output.parents:
            raise CampaignMergeError(
                f"merged views cannot be written inside shard records: {resolved_records}"
            )

    handoff_by_shard = {
        source.shard_id: _verify_handoff(source, campaign_id) for source in ordered_sources
    }
    binding = _campaign_binding(manifest_path, campaign_id)
    expected_root = expected_intake_root or (manifest_path.resolve().parent / binding.intake_path)
    expected_snapshot = load_intake_snapshot(expected_root)
    expected_ids = _snapshot_ids(expected_snapshot)
    snapshots_by_shard = {
        source.shard_id: load_intake_snapshot(source.intake_root) for source in ordered_sources
    }
    _prove_shard_partition(expected_ids, snapshots_by_shard)

    before = _fingerprint_sources(ordered_sources)
    raw_models_by_shard = {
        source.shard_id: _read_source_models(source) for source in ordered_sources
    }
    valve_rows_by_shard: dict[str, dict[str, JsonObject]] = {}
    rejected_valves_by_shard: dict[str, dict[str, str]] = {}
    shard_versions: dict[str, tuple[str, str]] = {}
    for source in ordered_sources:
        assigned = _snapshot_ids(snapshots_by_shard[source.shard_id])
        governing, rejected = _shard_valve_rows(source.records_root, assigned)
        valve_rows_by_shard[source.shard_id] = governing
        rejected_valves_by_shard[source.shard_id] = rejected
        raw_models = raw_models_by_shard[source.shard_id]
        _prove_processed_ownership(
            source, snapshots_by_shard[source.shard_id], raw_models, set(governing)
        )
        shard_versions[source.shard_id] = _assert_author_binding(campaign_id, raw_models, binding)
    _assert_cross_shard_identity(shard_versions)

    projections: dict[str, _ShardProjection] = {}
    for source in ordered_sources:
        author_version, checker_version = shard_versions[source.shard_id]
        projections[source.shard_id] = _project_shard(
            source,
            binding,
            snapshots_by_shard[source.shard_id],
            author_version,
            checker_version,
        )
    _assert_environment_axis(
        {
            shard_id: projection.environment_generations
            for shard_id, projection in projections.items()
        }
    )

    merged_by_id: dict[str, JsonObject] = {}
    for shard_id, projection in sorted(projections.items()):
        for stable_id, record in projection.current.items():
            if stable_id in merged_by_id:
                raise CampaignMergeError(
                    f"{HOST_SHARD_COLLISION_ERROR}: {stable_id} in multiple projections"
                )
            merged_by_id[stable_id] = record
    merged_valves: dict[str, JsonObject] = {}
    for shard_id, governing in sorted(valve_rows_by_shard.items()):
        for stable_id, row in governing.items():
            if stable_id in merged_valves:
                raise CampaignMergeError(
                    f"{HOST_SHARD_COLLISION_ERROR}: {stable_id} valve-claimed twice"
                )
            merged_valves[stable_id] = row

    review_held_union = {
        stable_id for projection in projections.values() for stable_id in projection.review_held_ids
    }
    # Dependency-stale records are RECORDS: the model is covered by an honest
    # terminal whose dependency vector merely awaits its re-drive, exactly the
    # tolerated mid-flight state the driver close accepts. They are carried
    # (excluded from the current view, reasons disclosed per shard), so
    # coverage subtracts them from the missing set. The tripwire target -- a
    # model with NO record and NO valid valve row -- already refused at the
    # per-shard ownership proof and stays refused here.
    stale_carried = {
        stable_id: {"shard": shard_id, "reason": reason}
        for shard_id, projection in sorted(projections.items())
        for stable_id, reason in projection.stale_reasons.items()
    }
    final = completeness_report(expected_ids, merged_by_id, machine_deferrals=merged_valves)
    final_violations = _issue_violations(final.incomplete_by_issue, review_held_union, merged_by_id)
    invalid_buckets = sorted(
        code for code in final.partition.buckets if code.startswith("invalid:")
    )
    missing_unaccounted = sorted(final.partition.missing_ids - set(stale_carried))
    if (
        missing_unaccounted
        or final.partition.extra_ids
        or final.partition.duplicate_ids
        or invalid_buckets
        or final.workflow_counts
        or final_violations
    ):
        raise CampaignMergeError(
            "final host shard validation failed: "
            f"missing={missing_unaccounted}, "
            f"extra={sorted(final.partition.extra_ids)}, "
            f"duplicates={sorted(final.partition.duplicate_ids)}, "
            f"invalid={invalid_buckets}, "
            f"issues={final_violations}"
        )

    current_records = tuple(merged_by_id[stable_id] for stable_id in sorted(merged_by_id))
    deferred_accounted = sorted(
        stable_id
        for disposition, stable_ids in final.partition.machine_deferred.items()
        if disposition not in _PARK_DISPOSITIONS
        for stable_id in stable_ids
    )
    per_source: dict[str, JsonObject] = {}
    for shard_id in sorted(source_by_id):
        snapshot = snapshots_by_shard[shard_id]
        projection = projections[shard_id]
        shard_partition = partition_report(
            (item.stable_id for item in snapshot.items),
            projection.current,
            machine_deferrals=valve_rows_by_shard[shard_id],
        )
        per_source[shard_id] = {
            "assigned_count": len(snapshot.items),
            "processed_count": len(raw_models_by_shard[shard_id]),
            "current_count": len(projection.current),
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "review_held_ids": list(projection.review_held_ids),
            "stale_reasons": dict(projection.stale_reasons),
            "machine_deferred": {
                disposition: sorted(stable_ids)
                for disposition, stable_ids in shard_partition.machine_deferred.items()
            },
            "grant_pending_ids": sorted(shard_partition.grant_pending_ids),
            "refusal_parked_ids": sorted(shard_partition.refusal_parked_ids),
            "superseded_deferral_ids": sorted(shard_partition.superseded_deferral_ids),
            "rejected_valve_rows": dict(rejected_valves_by_shard[shard_id]),
            "environment_generations": dict(projection.environment_generations),
            "handoff": handoff_by_shard[shard_id],
            "status": _status_report(
                [projection.current[stable_id] for stable_id in sorted(projection.current)]
            ),
        }
    merge_status = _merge_status(
        deferred_accounted=bool(deferred_accounted),
        review_holds=bool(review_held_union or final.partition.grant_pending_ids),
        refusal_parks=bool(final.partition.refusal_parked_ids),
        stale_records=bool(stale_carried),
    )
    report: JsonObject = {
        "format": HOST_SHARD_MERGE_REPORT_FORMAT,
        "campaign_id": campaign_id,
        "merge_status": merge_status,
        "manifest_sha256": hash_bytes(manifest_path.read_bytes()),
        "expected_intake": {
            "snapshot_id": expected_snapshot.snapshot_id,
            "snapshot_sha256": expected_snapshot.snapshot_sha256,
            "item_count": len(expected_snapshot.items),
        },
        "proof": {
            "shard_snapshots_validated": True,
            "assigned_stable_ids_pairwise_disjoint": True,
            "assigned_union_equals_expected_intake": True,
            "assigned_ids_terminal_or_deferral_claimed_per_shard": True,
            "valve_rows_validated_fail_closed": True,
            "stale_records_carried_never_current": True,
            "author_checker_versions_match_across_shards": True,
            "environment_axis_consistent_across_shards": True,
            "source_ledgers_read_only": True,
            "final_checkpoint_validation": "passed",
        },
        "partition": {
            "machine_deferred": {
                disposition: sorted(stable_ids)
                for disposition, stable_ids in final.partition.machine_deferred.items()
            },
            "grant_pending_ids": sorted(final.partition.grant_pending_ids),
            "refusal_parked_ids": sorted(final.partition.refusal_parked_ids),
            "superseded_deferral_ids": sorted(final.partition.superseded_deferral_ids),
            "review_held_ids": sorted(review_held_union),
            "stale_carried": {
                stable_id: dict(entry) for stable_id, entry in sorted(stale_carried.items())
            },
            "deferred_accounted": deferred_accounted,
            "carried_issues": {
                issue: list(stable_ids)
                for issue, stable_ids in sorted(final.incomplete_by_issue.items())
            },
        },
        "input_fingerprints": before,
        "sources": per_source,
        "total": _status_report(current_records),
    }
    after_reads = _fingerprint_sources(ordered_sources)
    if after_reads != before:
        raise CampaignMergeError("host shard canonical ledgers changed during read-only merge")
    # Content-addressed union of the shards' durable text corpora into the
    # merged output. Identical digests unify, differing bytes at one address
    # refuse loudly, and index rows dedup on their deterministic identity; the
    # shard roots stay read-only, which the post-write fingerprint re-proves.
    # Corpus payloads land before the view transaction commits: they are
    # self-verifying by content address and idempotent to re-union, so an
    # interrupted merge leaves only the benign payload-first state.
    corpus = union_corpus(
        [source.records_root for source in ordered_sources],
        resolved_output,
    )
    report["text_corpus"] = {
        "payloads_copied": corpus.payloads_copied,
        "payloads_shared": corpus.payloads_shared,
        "rows_appended": corpus.rows_appended,
        "rows_replayed": corpus.rows_replayed,
    }
    view_digests = _write_views(resolved_output, current_records, report)
    after_writes = _fingerprint_sources(ordered_sources)
    if after_writes != before:
        raise CampaignMergeError("host shard canonical ledgers changed while writing views")
    return HostShardMergeResult(current_records, report, view_digests)


def _source_argument(value: str) -> HostShardSource:
    """Parse ``SHARD_ID=CLONE_ROOT=INTAKE_ROOT[=HANDOFF_MANIFEST]`` arguments.

    Parameters
    ----------
    value:
        Raw command-line source argument.

    Returns
    -------
    HostShardSource
        Parsed source roots.
    """

    parts = value.split("=")
    if len(parts) not in {3, 4} or not all(parts):
        raise argparse.ArgumentTypeError(
            "source must be SHARD_ID=CLONE_ROOT=INTAKE_ROOT[=HANDOFF_MANIFEST]"
        )
    shard_id, clone_root, intake_root = parts[:3]
    resolved_clone = Path(clone_root).expanduser().resolve()
    return HostShardSource(
        shard_id=shard_id,
        records_root=resolved_clone / "menagerie" / "crawler" / "records",
        runtime_root=resolved_clone / ".crawl-local",
        intake_root=Path(intake_root).expanduser().resolve(),
        handoff_manifest=(Path(parts[3]).expanduser().resolve() if len(parts) == 4 else None),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the host-shard merge command parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for campaign, source, and output paths.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CAMPAIGN_MANIFEST)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=_source_argument,
        help="repeat as SHARD_ID=CLONE_ROOT=INTAKE_ROOT[=HANDOFF_MANIFEST]",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--expected-intake-root",
        type=Path,
        default=None,
        help="optional parent/pilot subset snapshot; defaults to manifest campaign intake",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the host-shard merge CLI.

    Parameters
    ----------
    argv:
        Optional arguments excluding executable name.

    Returns
    -------
    int
        Zero on success, otherwise one.
    """

    args = build_parser().parse_args(argv)
    try:
        result = merge_host_shards(
            args.manifest,
            args.campaign_id,
            args.source,
            args.output_root,
            expected_intake_root=args.expected_intake_root,
        )
    except (CampaignMergeError, LedgerError, KeyError, OSError, TypeError, ValueError) as exc:
        # LedgerError covers a SIGKILLed source shard (torn tail, corruption):
        # the merge stays read-only, so the refusal must be a typed exit, and
        # the shard owes one writer open (recovery) before it can merge.
        print(f"host shard merge failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "campaign_id": args.campaign_id,
                "merge_status": result.report["merge_status"],
                "terminal_count": len(result.current_records),
                "view_digests": result.view_digests,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
