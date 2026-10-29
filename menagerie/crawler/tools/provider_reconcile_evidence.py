"""Receipt-backed ``--provider-reconcile-command`` adapter over durable local evidence.

The deployed adapter answered ``accepted: true`` for every idempotency key
(ambiguity-as-spent).  That reading is safe in the money direction and useless in
the truth direction: a reservation the driver minted and then crashed on BEFORE
spawning anything was charged as a consummated provider session, and no operator
could tell a real spend from a bookkeeping ghost.

This adapter answers the same question from the paid ledger's OWN chain-verified
rows.  It is not the provider and never claims to be: it reports whether this host
holds durable evidence that the reserved session REACHED the provider.

Why the evidence is conclusive
------------------------------
The paid dispatch chokepoint (``driver_admission._run_operator_command``) orders
one paid session as:

1. ``reserve()`` mints the reservation and the provider idempotency key;
2. the guard is spawned behind a driver-owned release pipe -- it inherits the read
   end and BLOCKS on it before executing the operator command;
3. ``bind_custody()`` durably appends ``session_custody_bound``
   (``F_FULLFSYNC`` plus the anchor's commit/intent protocol);
4. and ONLY then does the driver write the release byte.

A guard that reads EOF instead of ``b"1"`` -- exactly what a driver death before
step 4 produces -- prints "not durably released" and exits 70 without ever
starting the operator command (``author_executor.run_parent_deadline_guard``).
So a reservation with NO custody row cannot have exposed a provider byte, and
``accepted: false`` is a machine proof rather than an assumption.  Conversely a
custody row means the subtree was spawned and may have been released, so it is
charged: the residual ambiguity stays on the money-POSITIVE side, where it
belongs.

The executor's attempt-bound publication receipt is deliberately NOT consulted.
It is strictly weaker evidence for this purpose: publication implies the operator
command ran, which implies the release byte, which implies the custody row -- so
it can never be a positive this rule misses -- and the durable attempt records
carry no ``spend_id``, so a publication digest cannot be soundly bound to a
specific reservation.  Reading it would mean attesting one session with another
session's artifact.

Evidence rules (the closed proof rule)
--------------------------------------
For the queried key:

* NO ``session_reserved`` row naming it -> REFUSE (nonzero exit).  Answering
  ``false`` about a key this ledger never minted is the one money-negative
  failure mode available, and a misdirected ``--records-root`` is precisely how it
  would happen.  Leg start retries a refusal, so this delays admission instead.
* more than one such row -> REFUSE.  The key mixes the verified chain head, the
  host, and the boot id, so it is unique by construction; a duplicate is tampering.
* otherwise ``accepted`` is true iff at least one marker in the closed set
  :data:`PROVIDER_REACHED_EVENTS` names the reservation's ``spend_id``, or a prior
  settlement already recorded a provider-accepted verdict for it.

``observed_wall_seconds`` is ALWAYS ``null``: this host meters no provider wall,
and the contract's null answer neither zeroes nor inflates the charge because the
writer's own measurement (for a crash path, the durable reservation age) is the
floor.  Claiming a wall here would be inventing provider attestation.

Nothing about the answer is producer-supplied.  Rows are read through
``load_paid_session_rows``, so the chain, the anchor, and the accounting fold all
have to accept the ledger before a single row is inspected, and the verdict is
recomputed from those rows on every invocation.

Spelling (``PAID_SESSION_CONTRACT.md``: flags first, key last)::

    python -m menagerie.crawler.tools.provider_reconcile_evidence \\
        --records-root /abs/path/to/records <idempotency-key>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.paid_session_lifecycle import (
    PROVIDER_STATUS_FIELDS,
    load_paid_session_rows,
)

#: Paid-ledger row events that PROVE the reserved subtree was spawned, and
#: therefore that the reserved provider session may have been consummated.
#:
#: Closed and enumerated, per the four-part waiver doctrine's proof-rule part:
#: widening it is a new decision needing its own reviewed change, because every
#: addition can only move answers from ``false`` to ``true``.  A release row is
#: listed alongside a bind row as an INDEPENDENT marker rather than a derived
#: one: the fold already refuses a release without its binding, so either row
#: alone is sufficient and neither has to be reconstructed from the other.
PROVIDER_REACHED_EVENTS: frozenset[str] = frozenset(
    {
        "session_custody_bound",
        "session_custody_released",
    }
)

#: Marker name recorded when a prior settlement already carries the provider's
#: own accepted verdict.  Not part of :data:`PROVIDER_REACHED_EVENTS`: it is an
#: echo of an answer, not evidence about custody.
SETTLED_MARKER = "session_settled"

#: This adapter's name/version stamp (D-1).  Rides inside every status receipt,
#: so ``receipt_sha256`` digest-covers WHICH reconciler answered -- the
#: provenance 200a's rows cannot show retroactively.  Bump the version with any
#: change to the evidence rules above.
ADAPTER_IDENTITY = "provider-reconcile-evidence/2"

EXIT_OK = 0
EXIT_REFUSED = 3


class EvidenceRefusal(Exception):
    """One typed unanswerable-query refusal.

    Raised instead of returning a decline, so an ambiguity can never be printed
    as an authoritative ``accepted: false``.
    """


@dataclass(frozen=True)
class EvidenceAnswer:
    """One evidence-derived provider answer for a single idempotency key.

    Parameters
    ----------
    idempotency_key:
        The exact queried key, echoed as the adapter contract requires.
    accepted:
        Whether durable local evidence shows the session reached the provider.
    observed_wall_seconds:
        Always ``None``; this host meters no provider wall.
    evidence:
        Sorted marker names that decided a positive answer, for the operator's
        stderr trail.  Empty for a proven decline.  Never printed on stdout: the
        driver-side reconciler enforces exact field equality.
    """

    idempotency_key: str
    accepted: bool
    observed_wall_seconds: Optional[float]
    evidence: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """Return exactly the contracted adapter field set.

        Returns
        -------
        dict[str, Any]
            The exact contracted status receipt ``CommandProviderReconciler`` parses.
        """

        payload = {
            "idempotency_key": self.idempotency_key,
            "accepted": self.accepted,
            "observed_wall_seconds": self.observed_wall_seconds,
            "adapter_identity": ADAPTER_IDENTITY,
        }
        # The driver side compares key sets for EQUALITY, so a helpful extra
        # field would be a typed PaidSessionError at every leg start. A raise
        # rather than an assert: this runs on the money path and must hold under
        # `python -O` too.
        if set(payload) != PROVIDER_STATUS_FIELDS:
            raise EvidenceRefusal(
                "adapter answer drifted from the contracted field set"
            )
        return payload


def _reservations_for_key(
    rows: Iterable[Mapping[str, Any]], provider_idempotency_key: str
) -> tuple[Mapping[str, Any], ...]:
    """Return every reservation row naming one provider idempotency key."""

    return tuple(
        row
        for row in rows
        if row.get("event") == "session_reserved"
        and row.get("provider_idempotency_key") == provider_idempotency_key
    )


def _settlement_accepted(rows: Iterable[Mapping[str, Any]], spend_id: str) -> bool:
    """Return whether a prior settlement recorded a provider-accepted verdict.

    Legacy settlement schemas carry no ``provider`` object at all; those answer
    nothing here and fall through to the custody markers rather than being read
    as a decline.
    """

    for row in rows:
        if row.get("event") != "session_settled" or row.get("spend_id") != spend_id:
            continue
        provider = row.get("provider")
        if isinstance(provider, Mapping) and provider.get("accepted") is True:
            return True
    return False


def reconcile_from_rows(
    rows: Sequence[Mapping[str, Any]], provider_idempotency_key: str
) -> EvidenceAnswer:
    """Answer one idempotency key from chain-verified paid-ledger rows.

    Parameters
    ----------
    rows:
        Verified paid-session lifecycle rows, in ledger order.
    provider_idempotency_key:
        The exact key to answer about.

    Returns
    -------
    EvidenceAnswer
        The evidence-derived answer, with its deciding markers attached.

    Raises
    ------
    EvidenceRefusal
        When the key is absent from the ledger or claimed by more than one
        reservation.  Both are unanswerable, and neither may read as free.
    """

    if not provider_idempotency_key:
        raise EvidenceRefusal("provider reconciliation requires an idempotency key")
    reservations = _reservations_for_key(rows, provider_idempotency_key)
    if not reservations:
        raise EvidenceRefusal(
            "unknown provider idempotency key: this paid ledger holds no reservation "
            "naming it, so the query is unanswerable here (check --records-root); "
            "answering 'not accepted' would zero a charge this host cannot see"
        )
    if len(reservations) > 1:
        raise EvidenceRefusal(
            "ambiguous provider idempotency key: "
            f"{len(reservations)} reservations name it, but the key mixes the verified "
            "chain head, the host, and the boot id and is unique by construction"
        )
    spend_id = str(reservations[0].get("spend_id"))
    markers = {
        str(row.get("event"))
        for row in rows
        if row.get("event") in PROVIDER_REACHED_EVENTS and row.get("spend_id") == spend_id
    }
    if _settlement_accepted(rows, spend_id):
        markers.add(SETTLED_MARKER)
    return EvidenceAnswer(
        idempotency_key=provider_idempotency_key,
        accepted=bool(markers),
        observed_wall_seconds=None,
        evidence=tuple(sorted(markers)),
    )


def paid_ledger_path(records_root: Path) -> Path:
    """Return the canonical paid-session ledger under one records root.

    Mirrors ``DriverPaths.paid_session_ledger``; kept as one named derivation so
    an operator passes the records root the rest of the tooling already takes.
    """

    return records_root / "operational" / "spend.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """Return the adapter's argument parser (flags first, key last)."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--records-root",
        type=Path,
        help="canonical records root holding operational/spend.jsonl",
    )
    source.add_argument(
        "--ledger",
        type=Path,
        help="explicit paid-session ledger path, for rigs and rolled-over ledgers",
    )
    parser.add_argument(
        "idempotency_key",
        help="the ONE paid-session provider idempotency key to answer about",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Answer one key on stdout, or refuse nonzero with an empty stdout.

    Returns
    -------
    int
        ``0`` with the contracted JSON receipt on stdout, or ``3`` with a
        diagnostic on stderr and NOTHING on stdout, so a refusal can never be
        misread as an answer by a caller that ignores exit status.
    """

    args = build_parser().parse_args(argv)
    ledger = args.ledger if args.ledger is not None else paid_ledger_path(args.records_root)
    try:
        rows = load_paid_session_rows(ledger)
        answer = reconcile_from_rows(rows, args.idempotency_key)
    except EvidenceRefusal as exc:
        print(f"provider-reconcile-evidence: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (OSError, ValueError) as exc:
        # A fold-refused, missing, or unreadable ledger is an outage, not a
        # decline: leg start retries it (PAID_SESSION_CONTRACT adapter section).
        print(
            f"provider-reconcile-evidence: paid ledger {ledger} is unreadable "
            f"or the accounting fold refuses it: {exc}",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    print(
        f"provider-reconcile-evidence: {args.idempotency_key} accepted="
        f"{str(answer.accepted).lower()} evidence={','.join(answer.evidence) or '<none>'}",
        file=sys.stderr,
    )
    print(json.dumps(answer.to_json(), sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
