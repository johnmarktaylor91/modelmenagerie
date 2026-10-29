"""Rung-boundary tally and spend report for one registered ladder rung.

The ladder's per-rung boundary question is "is this rung closable, and what did
it cost?". Before this tool the operator had to assemble the answer by hand:
``status`` reports a funnel over whichever snapshot is active and is NOT
phase-scoped, while ``throughput_report`` reports timing and checker spend for a
whole campaign clone with no rung scoping and no author-session accounting. Both
halves are needed at every rung boundary, so both are computed here, scoped to
exactly ONE registered rung, read-only:

* **identity** -- the rung registration re-verified against its snapshot bytes,
  plus a content digest of every ledger the report read, so a tally can be
  re-derived and compared later.
* **phase split** -- members grouped by the phase the driver would scope them
  to (:func:`~menagerie.crawler.constants.framework_from_markers`). This is the
  reconciliation an unsplit tally hides: a ``pytorch`` campaign whose partition
  carries native-framework rows (the 141 ``paddle-arm64`` rows in ``c1-mech``,
  the ERNIE/PaddleOCR class) closes its pytorch leg with those rows legitimately
  out of scope, and an unscoped ``status --verify-partition`` then reports them
  as ``missing``. Per phase this reads as exactly what it is: a second leg owed.
* **tally** -- per phase, terminal heads by status code, valve-claimed members,
  and the members that are NEITHER (the honest tripwire: those block the close).
* **spend** -- author sessions used, gate rounds used and explicit grants from
  the rung's own model heads; checker sessions bought versus verdicts reused
  from the operational ledger; attempt wall seconds by stage.
* **valve census** -- per valve ledger, rows claiming rung members, SEGREGATED
  into accountable rows and rows the completion partition's own row check
  rejects. A rejected row is never counted as accounting for anything.

Every valve row is filtered through
:func:`menagerie.crawler.status.machine_deferral_rejection` -- the same
fail-closed authority the completion partition applies -- so this report can
never call a model accounted that the driver's partition would refuse.

    python -m menagerie.crawler.tools.rung_report \\
        --records-root menagerie/crawler/records --campaign c1-mech --label step-200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import EnvironmentPhase, framework_from_markers
from menagerie.crawler.intake import IntakeItem, load_intake_snapshot
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import (
    LedgerCorruptionError,
    TornTailError,
    operational_events_path,
    scan_jsonl,
)
from menagerie.crawler.rung_stratification import _valve_paths as _stratifier_valve_paths
from menagerie.crawler.rung_staging import (
    RungRegistration,
    RungStagingError,
    load_rung_registrations,
    rung_registry_path,
)
from menagerie.crawler.constants import HOST_CAPABILITY_DEFERRAL_STATUS_CODES
from menagerie.crawler.host_capacity import mints_host_capability_terminal
from menagerie.crawler.state import _select_current
from menagerie.crawler.status import machine_deferral_rejection

RUNG_REPORT_SCHEMA_VERSION = "menagerie.crawler.rung-report.v2"

#: Terminal status prefixes that mean the model converted. Everything else that
#: holds a head is a terminal failure/deferral or a live non-terminal revision.
_RUNS_STATUS = "runs"


class RungReportError(ValueError):
    """Raised when a rung report cannot be derived from the records state."""


def _file_digest(path: Path) -> str:
    """Return one input's content digest, or ``absent``."""

    if not path.is_file():
        return "absent"
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _scan(path: Path, *, label: str) -> list[JsonObject]:
    """Scan one ledger read-only, turning corruption into a typed refusal."""

    try:
        return scan_jsonl(path, validate=False)
    except (TornTailError, LedgerCorruptionError) as exc:
        raise RungReportError(f"cannot read {label} ledger {path}: {exc}") from exc


def _valve_paths(records_root: Path) -> dict[str, Path]:
    """Return every machine-deferral valve ledger path by short name.

    Delegates to the stratifier's own closed set
    (:func:`menagerie.crawler.rung_stratification._valve_paths`), so a valve
    added there can never silently escape the boundary tally, and a rung tally
    and ``status --verify-partition`` can never disagree about what is claimed.
    """

    return _stratifier_valve_paths(records_root)


def _resolve_registration(
    records_root: Path,
    *,
    campaign_id: str,
    label: Optional[str],
    snapshot_id: Optional[str],
) -> RungRegistration:
    """Resolve exactly one registered rung of one campaign.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Frozen tier campaign owning the rung.
    label, snapshot_id:
        Exactly one selector. A label naming several registrations refuses
        rather than guessing which rung was meant.

    Returns
    -------
    RungRegistration
        The single matching registration.

    Raises
    ------
    RungReportError
        If the registry is unreadable, or the selector matches zero or several
        registrations.
    """

    if (label is None) == (snapshot_id is None):
        raise RungReportError("select the rung with exactly one of --label or --snapshot-id")
    try:
        registrations = load_rung_registrations(records_root)
    except RungStagingError as exc:
        raise RungReportError(f"cannot read the rung registry: {exc}") from exc
    matches = [
        registration
        for registration in registrations
        if registration.campaign_id == campaign_id
        and (
            registration.label == label
            if label is not None
            else registration.snapshot_id == snapshot_id
        )
    ]
    if not matches:
        known = ", ".join(
            f"{row.label}={row.snapshot_id}"
            for row in registrations
            if row.campaign_id == campaign_id
        )
        selector = f"label {label!r}" if label is not None else f"snapshot {snapshot_id!r}"
        raise RungReportError(
            f"no {campaign_id} rung matches {selector}; registered: {known or 'none'}"
        )
    if len(matches) > 1:
        raise RungReportError(
            f"{len(matches)} {campaign_id} rungs share label {label!r} "
            f"({', '.join(row.snapshot_id for row in matches)}); select with --snapshot-id"
        )
    return matches[0]


def _rung_members(records_root: Path, registration: RungRegistration) -> tuple[IntakeItem, ...]:
    """Load the rung's members from bytes that reproduce its registered identity.

    Raises
    ------
    RungReportError
        If the snapshot is missing or does not reproduce its registration. A
        tally over unverifiable membership would be worse than no tally.
    """

    snapshot_root = records_root / "intake" / registration.snapshot_id
    try:
        snapshot = load_intake_snapshot(snapshot_root)
    except (OSError, ValueError) as exc:
        raise RungReportError(
            f"rung {registration.label} ({registration.snapshot_id}) cannot be loaded "
            f"from {snapshot_root}: {exc}"
        ) from exc
    if (
        snapshot.snapshot_id != registration.snapshot_id
        or snapshot.snapshot_sha256 != registration.snapshot_sha256
    ):
        raise RungReportError(
            f"rung {registration.snapshot_id} does not reproduce its registered "
            "identity; refusing to report a tally over unverifiable membership"
        )
    return tuple(snapshot.items)


def _status_code(head: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return one head's status code, if any."""

    if head is None:
        return None
    status = head.get("status")
    if isinstance(status, Mapping):
        code = status.get("code")
        return str(code) if code is not None else None
    return None


def _phase_of(item: IntakeItem) -> str:
    """Return the scheduler phase the driver would scope one member to."""

    native = framework_from_markers(item.zoo, item.name)
    # ``EnvironmentPhase`` is a ``(str, Enum)`` mixin, not a 3.11 ``StrEnum``, so
    # ``str()`` renders the member name. Compare and emit ``.value``.
    return (
        EnvironmentPhase.PYTORCH.value
        if native == EnvironmentPhase.PYTORCH.value
        else EnvironmentPhase.NATIVE_TAIL.value
    )


def _valve_census(
    records_root: Path, member_ids: frozenset[str]
) -> tuple[JsonObject, dict[str, JsonObject]]:
    """Return the per-valve census and the accountable claim per member.

    Returns
    -------
    tuple[dict[str, Any], dict[str, dict[str, Any]]]
        Census rows per valve ledger, and ``stable_id -> claiming ROW`` for every member
        an ACCOUNTABLE row claims. Rows the completion partition's own check rejects are
        counted separately and claim nothing. The claiming row (not merely its valve
        name) is carried so the settlement split can tell a static host-capability
        refusal from a transient valve claim without re-reading the ledgers.
    """

    census: dict[str, JsonObject] = {}
    claims: dict[str, JsonObject] = {}
    for label, path in sorted(_valve_paths(records_root).items()):
        accountable: set[str] = set()
        rejected: list[JsonObject] = []
        for row in _scan(path, label=f"{label} valve"):
            stable_id = row.get("stable_id")
            if not isinstance(stable_id, str) or stable_id not in member_ids:
                continue
            rejection = machine_deferral_rejection(stable_id, row)
            if rejection is None:
                accountable.add(stable_id)
                claims.setdefault(stable_id, row)
            else:
                rejected.append({"stable_id": stable_id, "rejection": rejection})
        census[label] = {
            "ledger": path.as_posix(),
            "ledger_sha256": _file_digest(path),
            "accountable_members": len(accountable),
            # A rejected row is disclosed and segregated: it is never counted as
            # accounting for its model, exactly as the partition treats it.
            "rejected_rows": len(rejected),
            "rejected_detail": sorted(
                rejected, key=lambda entry: (entry["stable_id"], entry["rejection"])
            )[:20],
        }
    return census, claims


def _spend(
    heads: Mapping[str, JsonObject],
    attempts: Sequence[JsonObject],
    operational: Sequence[JsonObject],
    paid: Sequence[JsonObject],
    member_ids: frozenset[str],
) -> JsonObject:
    """Aggregate what the rung's members cost, from the rung's own records."""

    author_sessions = 0
    gate_rounds = 0
    explicit_grants = 0
    budget_missing = 0
    for stable_id, head in heads.items():
        budget = head.get("budget")
        if not isinstance(budget, Mapping):
            budget_missing += 1
            continue
        author_sessions += int(budget.get("author_sessions_used") or 0)
        gate_rounds += int(budget.get("gate_rounds_used") or 0)
        grants = budget.get("explicit_grants")
        explicit_grants += len(grants) if isinstance(grants, (list, tuple)) else 0
    stage_wall: dict[str, list[float]] = defaultdict(list)
    attempt_counts: Counter[str] = Counter()
    for row in attempts:
        if str(row.get("stable_id")) not in member_ids:
            continue
        attempt_counts[str(row.get("stage"))] += 1
        started, finished = row.get("started_at"), row.get("finished_at")
        seconds = _elapsed(started, finished)
        if seconds is not None:
            stage_wall[str(row.get("stage"))].append(seconds)
    checker = {"sessions_bought": 0, "verdicts_reused": 0}
    for row in paid:
        if row.get("event") != "session_reserved" or row.get("lane") != "checker":
            continue
        axis = row.get("axis")
        if not isinstance(axis, Mapping):
            continue
        members = axis.get("work_generations")
        if isinstance(members, list) and any(
            isinstance(member, Mapping) and str(member.get("stable_id")) in member_ids
            for member in members
        ):
            checker["sessions_bought"] += 1
    for row in operational:
        if str(row.get("stable_id")) not in member_ids:
            continue
        details = row.get("details")
        if not isinstance(details, Mapping):
            continue
        if details.get("disposition") == "checker-session-bought":
            checker["sessions_bought"] += 1
        elif details.get("disposition") == "checker-verdict-reused":
            checker["verdicts_reused"] += 1
    return {
        "author_sessions_used": author_sessions,
        "gate_rounds_used": gate_rounds,
        "explicit_grants": explicit_grants,
        "heads_without_budget": budget_missing,
        "checker": dict(checker),
        "attempts_by_stage": dict(sorted(attempt_counts.items())),
        "attempt_wall_seconds_by_stage": {
            stage: {"count": len(values), "total_seconds": math.fsum(values)}
            for stage, values in sorted(stage_wall.items())
        },
    }


def _elapsed(started: object, finished: object) -> Optional[float]:
    """Return non-negative elapsed seconds for one valid timestamp pair."""

    def parse(value: object) -> Optional[datetime]:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    start, finish = parse(started), parse(finished)
    if start is None or finish is None:
        return None
    seconds = (finish - start).total_seconds()
    return seconds if math.isfinite(seconds) and seconds >= 0.0 else None


def split_settlement(
    member_ids: Sequence[str],
    heads: Mapping[str, Mapping[str, Any]],
    claims: Mapping[str, Mapping[str, Any]],
) -> JsonObject:
    """Split a member set into settled, deferred-here, withheld, and unsettled.

    The step-100 tally had TWO buckets -- head or not -- so the five members this host
    could never run read "unsettled" forever while ``settled()`` inside a leg called them
    done. Eight finishing legs chased that difference. The split is now explicit:

    * **settled** -- holds a raw head of any kind, including a host-capability deferral.
      A stale award is still an award and a recorded deferral is still a record;
    * **deferred_here** -- settled BY a host-capability head, plus the legacy row-only
      members whose static refusal is durable but predates the head (the carried five).
      Not runnable on this host, needs no more work here, routes elsewhere;
    * **withheld_not_settled** -- claimed by any OTHER valve disposition (environment
      rebuild, coverage, host disk, checker lane, parks). Transient by nature: 55 such
      members existed at step-100 close and five of them went on to earn awards, so
      settling them out would have been a lie;
    * **unsettled** -- no head and no claim. THIS is what an operator's "unsettled" means
      and the only bucket a finishing leg should ever chase.

    Parameters
    ----------
    member_ids:
        Rung members to classify.
    heads:
        Raw current head per stable ID.
    claims:
        Accepted valve row per stable ID.

    Returns
    -------
    dict[str, Any]
        Counts plus sorted ID lists for every bucket.
    """

    settled: list[str] = []
    deferred_here: list[str] = []
    withheld: list[str] = []
    unsettled: list[str] = []
    for stable_id in member_ids:
        code = _status_code(heads.get(stable_id))
        row = claims.get(stable_id)
        if code is not None:
            settled.append(stable_id)
            if code in HOST_CAPABILITY_DEFERRAL_STATUS_CODES:
                deferred_here.append(stable_id)
            continue
        if row is not None:
            if mints_host_capability_terminal(row):
                # Durable static refusal, head not yet minted: honestly deferred
                # HERE, and never a finishing leg's work.
                deferred_here.append(stable_id)
            else:
                withheld.append(stable_id)
            continue
        unsettled.append(stable_id)
    return {
        "settled": len(settled),
        "deferred_here": len(deferred_here),
        "withheld_not_settled": len(withheld),
        "unsettled": len(unsettled),
        "deferred_here_ids": sorted(deferred_here),
        "withheld_not_settled_ids": sorted(withheld),
        "unsettled_ids": sorted(unsettled),
    }


def _config_homogeneity(
    operational: Sequence[JsonObject], registration: RungRegistration
) -> JsonObject:
    """Census the rung's per-leg ``rung_config_digest`` values (D-63/PM-1).

    Every leg since this rung's registration attests its effective config;
    the rung-invariant projection of that attestation must be ONE value
    across the whole rung. Two values means two different lane widths, two
    checker models, two trees, or two calibration modes adjudicated one
    certified rung -- the exact heterogeneity 200a could not even detect
    (author width 18 configured, 12 ran, nothing recorded).

    Pre-fix history rows (no ``rung_config_digest``) are DISCLOSED under
    ``pre_fix_legs`` and never counted as heterogeneous: the census refuses
    on what it can prove, not on what history could not yet say.

    Returns
    -------
    dict[str, Any]
        ``digests`` (digest -> leg count), ``pre_fix_legs``, ``legs``, and
        ``homogeneous`` (true when at most one distinct digest is present).
    """

    digests: Counter[str] = Counter()
    pre_fix = 0
    legs = 0
    for event in operational:
        if event.get("event_kind") != "leg-config":
            continue
        created_at = str(event.get("created_at", ""))
        if created_at < str(registration.created_at):
            continue
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        legs += 1
        digest = details.get("rung_config_digest")
        if isinstance(digest, str) and digest:
            digests[digest] += 1
        else:
            pre_fix += 1
    return {
        "digests": dict(sorted(digests.items())),
        "pre_fix_legs": pre_fix,
        "legs": legs,
        "homogeneous": len(digests) <= 1,
    }


def build_rung_report(
    records_root: Path,
    *,
    campaign_id: str,
    label: Optional[str] = None,
    snapshot_id: Optional[str] = None,
) -> JsonObject:
    """Build one registered rung's phase-split tally and spend report.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory (read-only).
    campaign_id:
        Frozen tier campaign owning the rung.
    label, snapshot_id:
        Exactly one rung selector.

    Returns
    -------
    dict[str, Any]
        The complete report; see the module docstring for its sections.

    Raises
    ------
    RungReportError
        If the rung cannot be resolved, its membership cannot be verified, or a
        ledger is unreadable.
    """

    registration = _resolve_registration(
        records_root, campaign_id=campaign_id, label=label, snapshot_id=snapshot_id
    )
    members = _rung_members(records_root, registration)
    member_ids = frozenset(item.stable_id for item in members)
    models_path = records_root / "models" / "current-shard.jsonl"
    attempts_path = records_root / "attempts" / "local.jsonl"
    operational_path = operational_events_path(records_root)
    paid_path = records_root / "operational" / "spend.jsonl"
    try:
        all_heads = _select_current(_scan(models_path, label="models"))
    except (LedgerCorruptionError, KeyError) as exc:
        raise RungReportError(f"cannot select current heads from {models_path}: {exc}") from exc
    heads = {stable_id: all_heads[stable_id] for stable_id in member_ids if stable_id in all_heads}
    census, claims = _valve_census(records_root, member_ids)
    attempts = _scan(attempts_path, label="attempts")
    operational = _scan(operational_path, label="operational")
    paid = _scan(paid_path, label="paid sessions") if paid_path.is_file() else []

    phases: dict[str, dict[str, Any]] = {}
    for item in members:
        phase = phases.setdefault(
            _phase_of(item),
            {"members": 0, "by_status": Counter(), "valve_claimed": 0, "unaccounted": []},
        )
        phase["members"] += 1
        code = _status_code(heads.get(item.stable_id))
        if code is not None:
            phase["by_status"][code] += 1
        elif item.stable_id in claims:
            phase["valve_claimed"] += 1
        else:
            phase["unaccounted"].append(item.stable_id)
    phase_report = {
        name: {
            "members": facts["members"],
            "by_status": dict(sorted(facts["by_status"].items())),
            "converted": facts["by_status"].get(_RUNS_STATUS, 0),
            "valve_claimed": facts["valve_claimed"],
            "unaccounted": len(facts["unaccounted"]),
            "unaccounted_ids": sorted(facts["unaccounted"])[:20],
            # A phase leg is closable when every member of it is terminal XOR
            # accountably valve-claimed: the driver's own partition rule.
            "closable": not facts["unaccounted"],
        }
        for name, facts in sorted(phases.items())
    }
    return {
        "schema_version": RUNG_REPORT_SCHEMA_VERSION,
        "rung": {
            **registration.to_row(),
            "verified_snapshot": True,
        },
        "phases": phase_report,
        # D-63 (PM-1): rung certification refuses heterogeneous config
        # digests -- the CLI exits nonzero when this census is not
        # homogeneous, and the section is in the report either way.
        "config_homogeneity": _config_homogeneity(operational, registration),
        # The bucket an operator reads before deciding whether a finishing leg is
        # owed at all: "unsettled" here means never-recorded, never "recorded
        # somewhere this host cannot help with".
        "settlement": split_settlement(sorted(member_ids), heads, claims),
        "legs_owed": sorted(name for name, facts in phase_report.items() if not facts["closable"]),
        "spend": _spend(heads, attempts, operational, paid, member_ids),
        "valves": census,
        "records_state": {
            "records_root": records_root.as_posix(),
            "models_ledger_sha256": _file_digest(models_path),
            "attempts_ledger_sha256": _file_digest(attempts_path),
            "operational_ledger_sha256": _file_digest(operational_path),
            "paid_session_ledger_sha256": _file_digest(paid_path),
            "registered_rungs_sha256": _file_digest(rung_registry_path(records_root)),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the rung-report argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--campaign", required=True, help="frozen tier campaign id")
    parser.add_argument("--label", help="rung label, e.g. step-200")
    parser.add_argument("--snapshot-id", dest="snapshot_id", help="exact rung snapshot id")
    parser.add_argument("--output", type=Path, help="write the report JSON here as well")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Emit one rung's tally and spend report.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero on success, two after a typed refusal.
    """

    args = build_parser().parse_args(argv)
    records_root = (
        args.records_root.resolve()
        if args.records_root is not None
        else args.repo_root.resolve() / "menagerie" / "crawler" / "records"
    )
    try:
        report = build_rung_report(
            records_root,
            campaign_id=args.campaign,
            label=args.label,
            snapshot_id=args.snapshot_id,
        )
    except RungReportError as exc:
        print(f"rung-report refused: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    homogeneity = report["config_homogeneity"]
    if not homogeneity["homogeneous"]:
        # D-63: heterogeneous rung config is a certification refusal, not a
        # footnote -- the legs of this rung did not run one configuration.
        print(
            "rung-report refused: heterogeneous rung_config_digest across the "
            f"rung's legs: {sorted(homogeneity['digests'])}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
