"""Append an explicit environment-quarantine clearance receipt, without rewriting history.

An out-of-band repair (a code fix, a backend recovery) can leave the quarantined
setup artifacts byte-identical, so identity comparison alone can never observe
it. The designed release signal is ONE campaign-health event whose
``details.quarantine_event_id`` names the exact quarantine event
(:data:`menagerie.crawler.driver_admission.ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION`).
A clearance releases both the live admission valve and every persisted rebuild
deferral row parked on that quarantine; a LATER quarantine event is a new fact
and governs again until it is itself cleared.

BL-9c: the receipt is appended to the operator-writable clearance SIDECAR
(``records/operational/environment-clearances.jsonl``), never to the canonical
operational ledger, so it can be recorded WHILE A DRIVER LEG RUNS -- the
running driver is the canonical ledger's single writer, and it merges the
sidecar on every release test (including one bounded mid-leg recovery pass, so
a cleared intent-setup park restores the environment lane without a leg
restart). Receipts already persisted in the canonical ledger keep releasing
exactly as before.

The tool refuses ambiguous prefixes, non-quarantine events, and double clears,
and requires the operator to state the repair evidence that justifies the
receipt.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import socket
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.driver_admission import (
    ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION,
    append_environment_clearance_event,
    environment_clearance_events,
)
from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.recordio import scan_jsonl

#: Closed set of quarantine-class dispositions a clearance receipt may name.
#: Anything else is not a quarantine and must not be "cleared".
CLEARABLE_QUARANTINE_DISPOSITIONS = frozenset(
    {
        "environment-cleanup-quarantined",
        "environment-integrity-quarantined",
        "environment-setup-failure-parked",
    }
)


def build_parser() -> argparse.ArgumentParser:
    """Build the clearance tool argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Typed parser for one quarantine clearance receipt.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "quarantine_event_id",
        help="exact event id, or an unambiguous prefix, of the quarantine event",
    )
    parser.add_argument(
        "--evidence",
        required=True,
        help="the out-of-band repair evidence justifying this clearance",
    )
    parser.add_argument("--cleared-by", default="operator")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate, print the receipt, append nothing",
    )
    return parser


def _resolve_quarantine(events: Sequence[Mapping[str, Any]], reference: str) -> Mapping[str, Any]:
    """Resolve one quarantine event from an id or unambiguous prefix.

    Parameters
    ----------
    events:
        Canonical operational events in ledger order.
    reference:
        Exact event id or a prefix matching exactly one event.

    Returns
    -------
    Mapping[str, Any]
        The single matching quarantine event.
    """

    matches = [
        event
        for event in events
        if isinstance(event.get("event_id"), str) and event["event_id"].startswith(reference)
    ]
    if not matches:
        raise ValueError(f"no operational event matches {reference!r}")
    if len(matches) > 1:
        listed = ", ".join(str(event.get("event_id")) for event in matches[:4])
        raise ValueError(f"ambiguous event reference {reference!r}: {listed}")
    event = matches[0]
    details = event.get("details")
    disposition = details.get("disposition") if isinstance(details, Mapping) else None
    if (
        event.get("event_kind") != OperationalEventKind.CAMPAIGN_HEALTH.value
        or disposition not in CLEARABLE_QUARANTINE_DISPOSITIONS
    ):
        raise ValueError(
            f"event {event.get('event_id')!r} is not a clearable quarantine "
            f"(kind={event.get('event_kind')!r}, disposition={disposition!r}); "
            f"clearable dispositions: {sorted(CLEARABLE_QUARANTINE_DISPOSITIONS)}"
        )
    return event


def _already_cleared(events: Sequence[Mapping[str, Any]], quarantine_event_id: str) -> bool:
    """Return whether a clearance receipt already names this quarantine."""

    return any(
        event.get("event_kind") == OperationalEventKind.CAMPAIGN_HEALTH.value
        and isinstance(event.get("details"), Mapping)
        and event["details"].get("disposition") == ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION
        and event["details"].get("quarantine_event_id") == quarantine_event_id
        for event in events
    )


def build_clearance_event(
    quarantine: Mapping[str, Any], *, evidence: str, cleared_by: str
) -> dict[str, Any]:
    """Build the campaign-health clearance receipt for one quarantine event.

    Parameters
    ----------
    quarantine:
        The resolved quarantine event being cleared.
    evidence:
        Operator-stated out-of-band repair evidence.
    cleared_by:
        Operator identity for the receipt.

    Returns
    -------
    dict[str, Any]
        Logical operational event with a content-addressed ``event_id``.
    """

    details = quarantine.get("details")
    intent = details.get("intent") if isinstance(details, Mapping) else None
    event: dict[str, Any] = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
        "status": OperationalEventStatus.HEALTHY.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {},
        "current_environment": intent if isinstance(intent, str) else None,
        "run_id": "clear-environment-quarantine-tool",
        "machine_id": socket.gethostname().split(".")[0],
        "details": {
            "disposition": ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION,
            "quarantine_event_id": quarantine["event_id"],
            "intent": intent,
            "cleared_by": cleared_by,
            "evidence": evidence,
        },
    }
    event = mint_operational_event(event)
    return event


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate and append one quarantine clearance receipt.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero after a durable append (or a valid dry run), otherwise non-zero.
    """

    args = build_parser().parse_args(argv)
    try:
        if not args.evidence.strip():
            raise ValueError("--evidence must state the repair; an empty receipt is not evidence")
        repo_root = args.repo_root.resolve()
        models_path = (
            repo_root / "menagerie" / "crawler" / "records" / "models" / "current-shard.jsonl"
        )
        ledger_path = canonical_operational_ledger_path(models_path)
        records_root = models_path.parent.parent
        # BL-9c: no driver lock -- the receipt lands in the operator sidecar
        # (this tool is the sidecar's only writer; the driver only reads it),
        # so a park is clearable while a leg runs. Quarantine resolution and
        # the double-clear refusal merge BOTH ledgers.
        events = [
            *scan_jsonl(ledger_path, validate=False),
            *environment_clearance_events(records_root),
        ]
        quarantine = _resolve_quarantine(events, args.quarantine_event_id)
        if _already_cleared(events, str(quarantine["event_id"])):
            raise ValueError(
                f"quarantine {quarantine['event_id']!r} already has a clearance receipt"
            )
        event = build_clearance_event(
            quarantine, evidence=args.evidence, cleared_by=args.cleared_by
        )
        if not args.dry_run:
            append_environment_clearance_event(records_root, event)
        print(json.dumps(event, sort_keys=True))
    except (OSError, ValueError, KeyError) as exc:
        print(f"clear-environment-quarantine failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
