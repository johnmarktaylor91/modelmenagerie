"""Repair a torn FINAL line on a models ledger, with evidence (H1 F6).

The models ledger readers are deliberately fail-closed: award decisions never
read a ledger whose final line is torn (a crash-mid-append signature), and --
unlike the rung registry and close ledger, whose torn tails recover through
:func:`menagerie.crawler.recordio.recover_torn_tail` on read -- the models
projection refuses instead of truncating behind the caller's back. That
refusal used to be a campaign-wide brick with no in-tool recovery. This tool
is the typed operator repair path for EXACTLY the torn-final-line case:

* the torn tail bytes are quarantined byte-for-byte to a sidecar next to the
  ledger before anything is truncated;
* the truncation itself routes through the SAME :func:`recover_torn_tail`
  convention every other operator-critical ledger uses (digest evidence row,
  fsync-before-truncate, live-writer refusal);
* one idempotent campaign-health event discloses the repair on the
  operational ledger.

Complete malformed MID-FILE rows are NEVER repaired here: the retained prefix
is re-validated with the models reader's own decode, and any corruption in it
refuses with the reader's typed error, evidence intact. No fail-open.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.driver import DriverLock, DriverLockError
from menagerie.crawler.identity import atomic_replace_bytes, hash_bytes, utc_now
from menagerie.crawler.leg_generation import MODELS_SOURCE_NAMES, close_ledger_sources
from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.recordio import (
    JsonlLedger,
    LedgerError,
    operational_events_path,
    recover_torn_tail,
    scan_jsonl,
)

#: Closed campaign-health disposition for one models-ledger tail repair; a
#: member of ``operational_identity.CAMPAIGN_HEALTH_IDEMPOTENT_DISPOSITIONS``,
#: so a retried repair replays into the persisted row instead of
#: double-appending.
REPAIR_DISPOSITION = "models-ledger-tail-repaired"


def _torn_tail_bytes(path: Path) -> Optional[tuple[int, bytes]]:
    """Return ``(offset, bytes)`` of a torn final line, or ``None``."""

    if not path.is_file():
        return None
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return None
    offset = payload.rfind(b"\n") + 1
    return offset, payload[offset:]


def repair_models_ledger(path: Path) -> Optional[dict]:
    """Quarantine and truncate one models ledger's torn final line.

    Parameters
    ----------
    path:
        Models ledger path.

    Returns
    -------
    dict | None
        Repair facts (offset, digest, quarantine and evidence paths), or
        ``None`` when the tail is already complete.

    Raises
    ------
    menagerie.crawler.recordio.LedgerError
        A live writer owns the ledger, the quarantined bytes disagree with
        the recovery evidence, or a complete retained row is corrupt (the
        reader's own typed refusal -- mid-file corruption is never repaired).
    """

    torn = _torn_tail_bytes(path)
    if torn is None:
        return None
    offset, tail = torn
    tail_sha256 = hash_bytes(tail)
    quarantine = path.with_name(
        f"{path.name}.torn-tail.{tail_sha256.removeprefix('sha256:')[:16]}.quarantine"
    )
    # Bytes FIRST, truncation second: the quarantine sidecar holds the exact
    # removed bytes, so no repair path can destroy evidence.
    atomic_replace_bytes(quarantine, tail)
    evidence = recover_torn_tail(
        path,
        # The models reader's own strictness: validate=False decode (digests
        # still re-verified). A complete malformed retained row refuses here
        # with the reader's typed error before any truncation.
        prefix_check=lambda candidate: scan_jsonl(candidate, validate=False),
    )
    if evidence is None:
        # The tail healed between the read and the recovery (only possible if
        # a writer completed the line); the quarantine copy is then stale.
        quarantine.unlink(missing_ok=True)
        return None
    if evidence.tail_sha256 != tail_sha256 or evidence.byte_offset != offset:
        raise LedgerError(
            f"models ledger changed under repair at {path}: quarantined "
            f"{tail_sha256}@{offset} but recovery evidenced "
            f"{evidence.tail_sha256}@{evidence.byte_offset}"
        )
    return {
        "ledger_path": str(path),
        "byte_offset": offset,
        "byte_count": len(tail),
        "tail_sha256": tail_sha256,
        "quarantine_path": str(quarantine),
        "evidence_path": str(evidence.evidence_path),
        "recovered_at": evidence.recovered_at,
    }


def _repair_event(records_root: Path, facts: dict) -> dict:
    """Assemble the idempotent campaign-health disclosure for one repair."""

    event = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
        "status": OperationalEventStatus.DEGRADED.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": "repair-models-ledger",
        "machine_id": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "details": {"disposition": REPAIR_DISPOSITION, **facts},
    }
    return mint_operational_event(
        event,
        identity_payload={
            "disposition": REPAIR_DISPOSITION,
            "ledger_path": facts["ledger_path"],
            "byte_offset": facts["byte_offset"],
            "tail_sha256": facts["tail_sha256"],
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Repair torn models-ledger tails under the campaign records root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report torn tails without quarantining, truncating, or appending",
    )
    args = parser.parse_args(argv)
    records_root = args.records_root.resolve()
    sources = close_ledger_sources(records_root)
    candidates = [sources[name] for name in MODELS_SOURCE_NAMES if sources[name].is_file()]
    owner = {"pid": os.getpid(), "run_id": "repair-models-ledger", "target": None}
    repairs: list[dict] = []
    try:
        if args.dry_run:
            report = [
                {
                    "ledger_path": str(path),
                    "torn": (torn := _torn_tail_bytes(path)) is not None,
                    "byte_offset": None if torn is None else torn[0],
                    "byte_count": None if torn is None else len(torn[1]),
                }
                for path in candidates
            ]
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        with DriverLock(
            args.repo_root.resolve() / ".crawl-local" / "locks" / "driver.lock", owner
        ):
            for path in candidates:
                facts = repair_models_ledger(path)
                if facts is None:
                    continue
                events_path = operational_events_path(records_root)
                events_path.parent.mkdir(parents=True, exist_ok=True)
                with JsonlLedger(events_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
                    facts["event"] = ledger.append(_repair_event(records_root, facts)).record
                repairs.append(facts)
    except (DriverLockError, LedgerError, OSError, ValueError) as exc:
        print(f"repair-models-ledger failed: {exc}", file=sys.stderr)
        return 1
    if not repairs:
        print(json.dumps({"repaired": [], "note": "no models ledger tail is torn"}))
        return 0
    print(json.dumps({"repaired": repairs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
