"""Discharge durable work generations explicitly (WS-1 clause 6).

Every consumed requeue grant mints a durable work generation. The step-100
re-adjudication era left ~48 consumed generations whose outcomes were the
downgrades the award guard now refuses -- identities nothing discharged, each
re-binding as "active work" on every resume forever (the vector-5 exhibit).
This tool closes that lifecycle:

* ``void`` appends one typed ``requeue-generation-discharged`` event with the
  closed state ``voided-before-start`` for a LIVE generation an operator has
  decided will never start (the m4933 shape).
* ``migrate`` derives every consumed generation's discharge state from ledger
  evidence (landed revisions carrying the grant, live park rows, successor
  generations, explicit events) and -- with ``--apply`` -- appends explicit
  discharge events for the derivable ones. Generations with no evidence stay
  LIVE and are listed as needing an operator decision; nothing is guessed.

Selection (:mod:`menagerie.crawler.work_selection`) consumes the same
derivation: a discharged generation never replays, and ``--resume`` reads
heads and discharge state, never raw identity backlogs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.checkpoint import canonical_requeue_grants_path
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.driver import DriverLock, DriverLockError
from menagerie.crawler.grant_parks import grant_park_path, grant_park_released
from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.recordio import (
    JsonlLedger,
    operational_events_path,
    scan_jsonl,
)
from menagerie.crawler.refusal_parks import refusal_park_path, refusal_park_released
from menagerie.crawler.state import _select_current
from menagerie.crawler.work_selection import (
    GenerationDischarge,
    derive_generation_discharge,
)


def _models_ledgers(records_root: Path) -> tuple[Path, ...]:
    """Return every present models layout, through the ONE source census.

    The old prefer-first rule was rt3's B' finding: a third private resolver
    disagreeing with the attested set. Every present layout is read (union),
    exactly like the staging door, so no layout can shadow another here.
    """

    from menagerie.crawler.leg_generation import models_source_paths

    return models_source_paths(records_root)


def _requeue_grants_ledger(records_root: Path) -> Path:
    """Return the canonical requeue-grant ledger (layout-independent)."""

    from menagerie.crawler.leg_generation import close_ledger_sources

    return canonical_requeue_grants_path(close_ledger_sources(records_root)["models.shard"])


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL ledger leniently; an absent ledger reads as empty."""

    if not path.is_file():
        return []
    try:
        return [dict(row) for row in scan_jsonl(path, validate=False)]
    except (OSError, ValueError):
        return []


def _live_park_work_ids(
    records_root: Path, heads: Mapping[str, Mapping[str, Any]]
) -> frozenset[str]:
    """Return work IDs held by live operator-owned park rows."""

    released = (
        (grant_park_path(records_root), grant_park_released),
        (refusal_park_path(records_root), refusal_park_released),
    )
    work_ids: set[str] = set()
    for path, release in released:
        for row in _read_rows(path):
            stable_id = str(row.get("stable_id", ""))
            try:
                if release(row, heads.get(stable_id)):
                    continue
            except Exception:  # noqa: BLE001 -- lenient read; a bad row claims nothing
                continue
            work_id = row.get("work_id")
            if isinstance(work_id, str) and work_id:
                work_ids.add(work_id)
    return frozenset(work_ids)


def migration_report(records_root: Path) -> list[dict[str, Any]]:
    """Classify every consumed work generation from ledger evidence.

    Parameters
    ----------
    records_root:
        Canonical records root (``.../records``).

    Returns
    -------
    list[dict[str, Any]]
        One row per consumption event: stable_id, grant_id, work_id,
        generation, derived ``discharge`` (or ``None`` == LIVE), its
        ``evidence``, and ``needs_operator_decision``.
    """

    records_root = Path(records_root)
    model_revisions = [row for path in _models_ledgers(records_root) for row in _read_rows(path)]
    events = _read_rows(operational_events_path(records_root))
    consumptions = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    discharge_events = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GENERATION_DISCHARGED.value
    ]
    try:
        heads: Mapping[str, Mapping[str, Any]] = _select_current(model_revisions)
    except Exception:  # noqa: BLE001 -- park release falls back to row-only liveness
        heads = {}
    park_work_ids = _live_park_work_ids(records_root, heads)
    # A voided grant discharges the generations it minted, and the migration
    # must classify them exactly as selection does (round-2 break 5) -- one
    # derivation, one answer.
    voided_grant_ids = frozenset(
        str(row["void_of"])
        for row in _read_rows(_requeue_grants_ledger(records_root))
        if isinstance(row.get("void_of"), str) and row["void_of"]
    )
    report: list[dict[str, Any]] = []
    for event in consumptions:
        details = event.get("details", {})
        discharge, evidence = derive_generation_discharge(
            details,
            model_revisions=model_revisions,
            live_park_work_ids=park_work_ids,
            discharge_events=discharge_events,
            all_consumptions=consumptions,
            voided_grant_ids=voided_grant_ids,
        )
        # The SAME (stable_id, work_id) match derive_generation_discharge
        # consults first: a generation an explicit event already discharged is
        # settled history the migration must never re-append (H1 F3 -- the
        # synthesized migration event carried different provenance under the
        # same identity, so `void` then `migrate --apply` hard-conflicted and
        # permanently bricked the migration).
        explicit = next(
            (
                str(discharge_event.get("event_id"))
                for discharge_event in discharge_events
                if discharge_event.get("details", {}).get("stable_id")
                == details.get("stable_id")
                and discharge_event.get("details", {}).get("work_id")
                == details.get("new_work_id")
            ),
            None,
        )
        report.append(
            {
                "stable_id": details.get("stable_id"),
                "grant_id": details.get("grant_id"),
                "work_id": details.get("new_work_id"),
                "generation": details.get("new_work_generation"),
                "discharge": discharge.value if discharge is not None else None,
                "evidence": evidence,
                "explicit_discharge_event_id": explicit,
                "needs_operator_decision": discharge is None,
            }
        )
    return report


def _discharge_event(
    *,
    stable_id: str,
    work_id: str,
    grant_id: Optional[str],
    generation: Optional[int],
    discharge: GenerationDischarge,
    reason: str,
    discharged_by: str,
    source: str,
) -> dict[str, Any]:
    """Assemble one typed generation-discharge operational event."""

    event: dict[str, Any] = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": None,
        "event_kind": OperationalEventKind.REQUEUE_GENERATION_DISCHARGED.value,
        "status": OperationalEventStatus.HEALTHY.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": "discharge-generation-tool",
        "machine_id": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "details": {
            "discharge": discharge.value,
            "stable_id": stable_id,
            "work_id": work_id,
            "grant_id": grant_id,
            "generation": generation,
            "reason": reason,
            "discharged_by": discharged_by,
            "source": source,
        },
    }
    from datetime import datetime, timezone

    event["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    # Registered idempotent-first-clock (ID-1): the identity is the
    # DISCHARGED GENERATION's semantic key -- (stable_id, work_id), the exact
    # pair ``derive_generation_discharge`` matches on -- never the clock and
    # never the discharge state. Re-voiding the same generation with the same
    # state is an idempotent replay; a CONFLICTING re-discharge (different
    # state for the same generation) lands on the SAME identity and trips the
    # conflicting-replay refusal instead of forking.
    return mint_operational_event(
        event,
        identity_payload={"stable_id": stable_id, "work_id": work_id},
    )


def _append_event(events_path: Path, event: Mapping[str, Any]) -> None:
    """Append one event, tolerating an identical already-persisted replay."""

    existing = _read_rows(events_path)
    for row in existing:
        if row.get("event_id") == event["event_id"]:
            same_details = row.get("details") == event["details"]
            if same_details:
                return
            raise ValueError(f"conflicting discharge already persisted for {event['event_id']}")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with JsonlLedger(events_path, OPERATIONAL_EVENT_SCHEMA_VERSION, lock_timeout=10.0) as ledger:
        ledger.append(dict(event))


def build_parser() -> argparse.ArgumentParser:
    """Build the generation-discharge tool parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    void = commands.add_parser("void", help="discharge one LIVE generation as voided-before-start")
    void.add_argument("--stable-id", required=True)
    void.add_argument("--work-id", required=True)
    void.add_argument("--reason", required=True)
    void.add_argument("--discharged-by", default="operator")
    migrate = commands.add_parser(
        "migrate", help="classify every consumed generation from ledger evidence"
    )
    migrate.add_argument(
        "--apply",
        action="store_true",
        help="append explicit discharge events for the derivable states",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Void one generation or run the discharge migration.

    Returns
    -------
    int
        Zero on success (for ``migrate`` without ``--apply``: always, the
        report is informational); non-zero on refusal.
    """

    args = build_parser().parse_args(argv)
    records_root = args.records_root.resolve()
    events_path = operational_events_path(records_root)
    owner = {"pid": os.getpid(), "run_id": "discharge-generation-tool", "target": None}
    try:
        if args.command == "migrate":
            report = migration_report(records_root)
            if args.apply:
                already_discharged = [
                    row for row in report if row["explicit_discharge_event_id"] is not None
                ]
                if already_discharged:
                    # Typed skip disclosure: an explicitly discharged generation
                    # is satisfied, never re-appended -- which is what makes the
                    # migration re-runnable after any operator void or a crash
                    # mid-apply (H1 F3).
                    print(
                        f"{len(already_discharged)} generation(s) already carry an "
                        "explicit discharge event; skipped as idempotent: "
                        + ", ".join(
                            f"{row['stable_id']}={row['explicit_discharge_event_id']}"
                            for row in already_discharged
                        ),
                        file=sys.stderr,
                    )
                with DriverLock(
                    args.repo_root.resolve() / ".crawl-local" / "locks" / "driver.lock",
                    owner,
                ):
                    for row in report:
                        if row["discharge"] is None:
                            continue
                        if row["explicit_discharge_event_id"] is not None:
                            continue
                        _append_event(
                            events_path,
                            _discharge_event(
                                stable_id=str(row["stable_id"]),
                                work_id=str(row["work_id"]),
                                grant_id=row["grant_id"],
                                generation=row["generation"],
                                discharge=GenerationDischarge(row["discharge"]),
                                reason=str(row["evidence"]),
                                discharged_by="migration",
                                source="migration",
                            ),
                        )
            print(json.dumps(report, indent=2, sort_keys=True))
            undecided = [row for row in report if row["needs_operator_decision"]]
            if undecided:
                print(
                    f"{len(undecided)} generation(s) need an operator decision "
                    f"(void or re-drive): "
                    f"{sorted(str(row['stable_id']) for row in undecided)}",
                    file=sys.stderr,
                )
            return 0

        report = migration_report(records_root)
        matching = [row for row in report if row["work_id"] == args.work_id]
        if not matching:
            raise ValueError(f"no consumed generation exists with work_id {args.work_id!r}")
        row = matching[0]
        if str(row["stable_id"]) != args.stable_id:
            raise ValueError(
                f"work generation {args.work_id!r} belongs to {row['stable_id']!r}, "
                f"not {args.stable_id!r}"
            )
        if row["discharge"] is not None:
            raise ValueError(
                f"generation is already discharged ({row['discharge']}: "
                f"{row['evidence']}); voiding it would rewrite settled history"
            )
        with DriverLock(args.repo_root.resolve() / ".crawl-local" / "locks" / "driver.lock", owner):
            event = _discharge_event(
                stable_id=args.stable_id,
                work_id=args.work_id,
                grant_id=row["grant_id"],
                generation=row["generation"],
                discharge=GenerationDischarge.VOIDED_BEFORE_START,
                reason=args.reason,
                discharged_by=args.discharged_by,
                source="operator-void",
            )
            _append_event(events_path, event)
        print(json.dumps(event, sort_keys=True))
    except (DriverLockError, OSError, ValueError) as exc:
        print(f"discharge-generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
