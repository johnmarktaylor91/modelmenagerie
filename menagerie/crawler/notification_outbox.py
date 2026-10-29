"""Durable receipt-backed outbox for CRITICAL-class operator notifications.

D-38 (ack half): the OB-M3 doctor mechanism generalized to runtime. The doctor
proves the transport real at PREFLIGHT only; a mid-rung transport regression
could otherwise silently discard decisions-owed pages. So every critical-class
send is recorded as a durable INTENT row before the transport runs, and is
marked SENT only when the transport ECHOED the message's idempotency key --
exit 0 alone is exactly what ``/usr/bin/true`` fakes. An intent with no
echo-verified settlement stays PENDING: the monitor alarms on it and the send
path re-attempts it, so a lost page is loud instead of silent.

The outbox is runtime state (it lives beside the notification receipts under
``.crawl-local``), never part of the canonical records root. Rows are
append-only and fsynced; the pending projection re-derives from the full row
sequence, so a crash between intent and settlement reads as pending -- the
conservative direction for a message that may not have arrived.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import canonical_json_bytes, durable_fsync, utc_now

OUTBOX_SCHEMA_VERSION = "menagerie.crawler.notification-outbox.v1"

#: Closed settlement outcomes. ``sent-echoed`` is the ONLY terminal success;
#: everything else leaves the intent pending (re-queue and alarm) or names a
#: typed refusal that itself demands operator attention.
OUTBOX_OUTCOMES = frozenset(
    {
        "sent-echoed",
        "sent-unechoed",
        "transport-failed",
        "refused-transport-absent",
        "refused-transport-discarding",
    }
)

_ROW_KINDS = frozenset({"send-intent", "send-settled"})


@dataclass(frozen=True)
class PendingCriticalSend:
    """One critical-class send not yet settled by a transport echo."""

    nonce: str
    notification_class: str
    message_head: str
    created_at: str
    attempts: int
    last_outcome: Optional[str]


def default_outbox_path(receipt_root: Path) -> Path:
    """Return the outbox ledger path beside one notification-receipt root."""

    return receipt_root.parent / "notification-outbox.jsonl"


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(row)) + b"\n"
    with open(path, "ab") as handle:
        handle.write(payload)
        handle.flush()
        durable_fsync(handle.fileno())


def record_send_intent(
    path: Path,
    *,
    nonce: str,
    notification_class: str,
    message: str,
) -> None:
    """Durably record one critical-class send BEFORE the transport runs."""

    _append(
        path,
        {
            "schema_version": OUTBOX_SCHEMA_VERSION,
            "kind": "send-intent",
            "nonce": nonce,
            "notification_class": notification_class,
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "message_head": message[:200],
            "created_at": utc_now(),
        },
    )


def record_send_settled(
    path: Path,
    *,
    nonce: str,
    outcome: str,
    detail: str = "",
) -> None:
    """Durably record one send attempt's typed settlement.

    Raises
    ------
    ValueError
        If ``outcome`` is outside the closed vocabulary -- an unknown outcome
        would silently widen what counts as settled.
    """

    if outcome not in OUTBOX_OUTCOMES:
        raise ValueError(f"unknown outbox outcome {outcome!r}")
    _append(
        path,
        {
            "schema_version": OUTBOX_SCHEMA_VERSION,
            "kind": "send-settled",
            "nonce": nonce,
            "outcome": outcome,
            "detail": detail[:500],
            "created_at": utc_now(),
        },
    )


def read_outbox_rows(path: Path) -> tuple[dict[str, Any], ...]:
    """Read outbox rows tolerantly (torn final line skipped, never fatal)."""

    try:
        data = path.read_bytes()
    except OSError:
        return ()
    rows: list[dict[str, Any]] = []
    lines = data.split(b"\n")
    last_content = max(
        (index for index, line in enumerate(lines) if line.strip()), default=-1
    )
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            if index == last_content:
                continue
            raise ValueError(f"corrupt notification-outbox row at {path}:{index + 1}")
        if not isinstance(value, dict) or value.get("kind") not in _ROW_KINDS:
            raise ValueError(f"unknown notification-outbox row at {path}:{index + 1}")
        rows.append(value)
    return tuple(rows)


def pending_critical_sends(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[PendingCriticalSend, ...]:
    """Project the outbox onto its unsettled critical sends, oldest first.

    An intent is settled ONLY by a ``sent-echoed`` outcome. Any other outcome
    keeps it pending with the outcome disclosed, so a discarded or unechoed
    page stays visible until it is truly delivered (or an operator retires the
    outbox with the campaign).
    """

    intents: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = {}
    last_outcome: dict[str, str] = {}
    settled: set[str] = set()
    for row in rows:
        nonce = str(row.get("nonce", ""))
        if not nonce:
            continue
        if row.get("kind") == "send-intent":
            intents.setdefault(nonce, dict(row))
        elif row.get("kind") == "send-settled":
            attempts[nonce] = attempts.get(nonce, 0) + 1
            last_outcome[nonce] = str(row.get("outcome", ""))
            if row.get("outcome") == "sent-echoed":
                settled.add(nonce)
    pending: list[PendingCriticalSend] = []
    for nonce, intent in intents.items():
        if nonce in settled:
            continue
        pending.append(
            PendingCriticalSend(
                nonce=nonce,
                notification_class=str(intent.get("notification_class", "")),
                message_head=str(intent.get("message_head", "")),
                created_at=str(intent.get("created_at", "")),
                attempts=attempts.get(nonce, 0),
                last_outcome=last_outcome.get(nonce),
            )
        )
    pending.sort(key=lambda send: send.created_at)
    return tuple(pending)
