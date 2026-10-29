"""Roll a fold-refused paid-session ledger over to a new accounting authority.

The 2026-08-20 step-200 leg durably appended a duplicate-custody row; the
accounting fold has refused every money read of that ledger since, so the
records root bricks loudly at admission-time reconciliation.  That refusal is
CORRECT and permanent: no row may be edited, no exemption may be added, and a
silently fresh ledger would reset the campaign's cumulative QUOTA and forget
the residue's conservative charge.

This tool is the reviewed same-campaign continuation.  Given a records root
whose paid ledger's fold REFUSES (and only then -- a healthy ledger refuses
rollover typed), it:

1. verifies and preserves the poisoned ledger immutably: rename-in-place to a
   ``.poisoned-<utc>`` sibling (anchor alongside), byte digests recorded, NO
   row ever edited;
2. mints a NEW accounting authority whose genesis row
   (``accounting_rollover``) embeds the poisoned ledger's chain-verified rows
   verbatim and names its last verifiable chain head and content digest --
   lineage, not laundering;
3. carries forward the CONSERVATIVE totals as opening charges: every carried
   reservation (settled or not) is charged as consummated at its full
   reserved wall (constant-accept semantics), disclosed as
   ``rollover_opening_charges``, so the cumulative QUOTA/RUNAWAY axes and the
   author attempt count lose nothing;
4. writes one ``paid-ledger-rollover`` operational event disclosing all of it.

Eligibility, the charges, and the healthy-ledger refusal are NOT this tool's
authority: the accounting fold recomputes all three from the embedded rows on
every replay (:meth:`menagerie.crawler.paid_session_accounting._StateBuilder.
_apply_rollover`), per the four-part waiver doctrine.  A rollover row this
tool mints with a healthy predecessor refuses at the fold-first append and
lands nowhere.

Dry-run by default; ``--apply`` performs the rename and mint.  The tool takes
the operational ledger's exclusive writer lock for the whole operation, so a
live driver blocks it (and its own single-writer lock blocks a live driver).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.identity import stable_hash, utc_now
from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.paid_session_accounting import (
    PAID_SESSION_CHAIN_SCOPE,
    AccountingState,
)
from menagerie.crawler.paid_session_durability import (
    AnchoredLedger,
    preserved_lineage_verifier,
)
from menagerie.crawler.paid_session_lifecycle import PaidSessionAuthority
from menagerie.crawler.recordio import JsonlLedger, operational_events_path


def paid_ledger_path(records_root: Path) -> Path:
    """Return the canonical paid-session ledger under one records root."""

    return records_root / "operational" / "spend.jsonl"


def _raw_ledger(path: Path) -> AnchoredLedger:
    """Return a chain-and-anchor-verifying reader with NO accounting fold.

    The money projection refuses a poisoned ledger by design; this reader
    authenticates the bytes (chain links, content addresses, anchored extent)
    without adjudicating them, which is exactly the evidence the rollover row
    embeds.  A distinct projection name keeps it out of the money read cache.
    """

    return AnchoredLedger(
        path,
        chain_scope=PAID_SESSION_CHAIN_SCOPE,
        subject="paid-session-rollover-inspection",
        seed=lambda: None,
        fold=lambda value, _rows: value,
        projection="paid.sessions.rollover-inspection",
    )


def _file_digest(path: Path) -> tuple[str, int]:
    """Return the ledger file's content digest and byte count."""

    payload = path.read_bytes()
    return stable_hash({"bytes_hex": payload.hex()}), len(payload)


def _fold_refusal(
    rows: tuple[dict[str, Any], ...], *, ledger_path: Path
) -> Optional[str]:
    """Return the accounting fold's refusal for these rows, or ``None``.

    The fold runs with the ledger directory's own lineage authority, exactly
    as the production authority folds: a predecessor that is itself a
    legitimate (healthy) rolled-over successor folds cleanly here and refuses
    rollover, while a genuinely poisoned one reports its real refusal.
    """

    lineage = preserved_lineage_verifier(
        ledger_path.parent, ledger_name=ledger_path.name, subject="paid-session"
    )
    try:
        AccountingState(lineage_verifier=lineage).extend(rows)
    except ValueError as exc:
        return str(exc)
    return None


def _append_rollover_event(
    operational: JsonlLedger,
    *,
    operator: str,
    reason: str,
    disclosure: dict[str, Any],
) -> None:
    """Disclose the completed rollover on the operational ledger."""

    event: dict[str, Any] = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
        "status": OperationalEventStatus.DEGRADED.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 0},
        "current_environment": None,
        "run_id": "rollover-paid-ledger-tool",
        "machine_id": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "details": {
            "disposition": "paid-ledger-rollover",
            "operator": operator,
            "reason": reason,
            **disclosure,
        },
    }
    operational.append(mint_operational_event(event))


def rollover_paid_ledger(
    records_root: Path, *, operator: str, reason: str, apply: bool
) -> dict[str, Any]:
    """Verify one records root and, with ``apply``, perform the rollover.

    Returns
    -------
    dict[str, Any]
        The machine-readable report this tool also prints.

    Raises
    ------
    SystemExit
        Typed refusals: absent ledger, unauthenticated bytes, HEALTHY ledger
        (fold accepts -- rollover is not applicable), or a live writer.
    """

    ledger_path = paid_ledger_path(records_root)
    if not ledger_path.is_file():
        raise SystemExit(f"ROLLOVER_REFUSED no paid-session ledger at {ledger_path}")
    try:
        snapshot = _raw_ledger(ledger_path).read_locked()
    except ValueError as exc:
        raise SystemExit(
            "ROLLOVER_REFUSED the paid ledger's bytes do not authenticate "
            f"(chain/anchor): {exc}; rollover carries only authenticated evidence"
        ) from exc
    refusal = _fold_refusal(snapshot.rows, ledger_path=ledger_path)
    if refusal is None:
        raise SystemExit(
            "ROLLOVER_REFUSED the paid ledger folds cleanly: a HEALTHY ledger "
            "remains the accounting authority and must not be reauthored"
        )
    digest, size = _file_digest(ledger_path)
    report: dict[str, Any] = {
        "ledger": str(ledger_path),
        "rows": len(snapshot.rows),
        "chain_head": snapshot.chain_state.head_sha256,
        "ledger_sha256": digest,
        "ledger_bytes": size,
        "fold_refusal": refusal,
        "applied": False,
    }
    if not apply:
        return report

    # The operational writer lock is the whole-operation exclusivity guard: a
    # live driver holds it for the leg, and holding it here blocks one.
    operational = JsonlLedger(
        operational_events_path(records_root), OPERATIONAL_EVENT_SCHEMA_VERSION
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sibling = ledger_path.with_name(f"{ledger_path.name}.poisoned-{stamp}")
    anchor = ledger_path.with_name(f"{ledger_path.name}.anchor.json")
    anchor_sibling = ledger_path.with_name(f"{sibling.name}.anchor.json")
    if sibling.exists() or anchor_sibling.exists():
        raise SystemExit(f"ROLLOVER_REFUSED preservation target exists: {sibling}")
    os.rename(ledger_path, sibling)
    anchor_moved = False
    if anchor.is_file():
        os.rename(anchor, anchor_sibling)
        anchor_moved = True
    preserved_digest, preserved_size = _file_digest(sibling)
    if preserved_digest != digest or preserved_size != size:
        raise SystemExit(
            "ROLLOVER_FAILED the preserved sibling is not byte-identical to the "
            f"verified ledger: {sibling}"
        )
    authority = PaidSessionAuthority(
        ledger_path,
        runtime_root=records_root / ".rollover-runtime",
        clock=utc_now,
    )
    try:
        authority.initialize_rollover(
            carried_rows=snapshot.rows,
            poisoned_ledger_name=sibling.name,
            poisoned_ledger_sha256=digest,
            poisoned_ledger_bytes=size,
        )
    except BaseException:
        # The fold refused the mint (nothing durable landed at the canonical
        # path); restore the predecessor exactly as found.
        if anchor_moved:
            os.rename(anchor_sibling, anchor)
        os.rename(sibling, ledger_path)
        raise
    state = authority.state()
    if state.rollover is None:
        raise SystemExit("ROLLOVER_FAILED the new authority folded without a rollover")
    disclosure = {
        "poisoned_sibling": sibling.name,
        "poisoned_ledger_sha256": digest,
        "poisoned_ledger_bytes": size,
        "poisoned_chain_head": snapshot.chain_state.head_sha256,
        "poisoned_row_count": len(snapshot.rows),
        "fold_refusal": refusal,
        "carried_reservations": state.rollover["carried_reservations"],
        "carried_attempt_groups": state.rollover["carried_attempt_groups"],
        "rollover_opening_charges": state.rollover["rollover_opening_charges"],
    }
    _append_rollover_event(
        operational, operator=operator, reason=reason, disclosure=disclosure
    )
    report.update(applied=True, **disclosure)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """Run the rollover tool CLI."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--records-root",
        type=Path,
        required=True,
        help="campaign canonical records directory (holds operational/spend.jsonl)",
    )
    parser.add_argument("--operator", required=True, help="named human operator")
    parser.add_argument("--reason", required=True, help="one-line justification")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the rollover (default verifies eligibility only)",
    )
    args = parser.parse_args(argv)
    if not args.operator.strip() or not args.reason.strip():
        raise SystemExit("ROLLOVER_REFUSED operator and reason are required")
    report = rollover_paid_ledger(
        args.records_root.resolve(),
        operator=args.operator.strip(),
        reason=args.reason.strip(),
        apply=args.apply,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        "PAID_LEDGER_ROLLOVER_COMPLETE"
        if report["applied"]
        else "PAID_LEDGER_ROLLOVER_ELIGIBLE"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
