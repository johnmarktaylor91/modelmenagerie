"""Append an explicit bounded effort grant, or void one, without rewriting history."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    append_canonical_requeue_void,
    build_canonical_requeue_grant,
    build_canonical_requeue_void,
)
from menagerie.crawler.constants import (
    EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT,
    GRANT_CAPABILITIES,
    OperationalEventKind,
)
from menagerie.crawler.driver import DriverLock, DriverLockError
from menagerie.crawler.recordio import scan_jsonl


def _consumption_of(ledger: Path, grant_id: str) -> Optional[dict]:
    """Return the durable consumption event bound to one grant, if any.

    The canonical events ledger is this grant ledger's sibling; an absent or
    unreadable ledger reads as "no consumption seen" -- the disclosure below is
    operator-facing honesty, never a validation gate, so unknown stays quiet
    and the append-side integrity checks remain the authority.

    Parameters
    ----------
    ledger:
        Canonical requeue-grant ledger path.
    grant_id:
        Exact grant identity being voided.

    Returns
    -------
    dict | None
        The ``REQUEUE_GRANT_CONSUMED`` event details, or ``None``.
    """

    events_path = ledger.parent / "events.jsonl"
    if not events_path.is_file():
        return None
    try:
        events = scan_jsonl(events_path, validate=False)
    except (OSError, ValueError):
        return None
    for event in events:
        if (
            event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
            and event.get("details", {}).get("grant_id") == grant_id
        ):
            return dict(event.get("details", {}))
    return None


def build_parser() -> argparse.ArgumentParser:
    """Build the requeue tool argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Typed parser for one bounded attempts grant or one grant void.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stable_id", nargs="?")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--grant", type=int, help="additional attempt count")
    parser.add_argument(
        "--stage",
        help=(
            "closed failure stage the grant re-authorizes, or 'deferred-terminal' "
            "for a review-held deferred:* terminal with no failure stage"
        ),
    )
    parser.add_argument("--granted-by", default="operator")
    parser.add_argument(
        "--void",
        dest="void_of",
        metavar="GRANT_ID",
        help="append an operator void event for an existing unconsumed grant",
    )
    parser.add_argument(
        "--intent",
        dest="target_intent",
        help="dependency-evidenced corrected environment intent",
    )
    parser.add_argument(
        "--blocked-award-revision",
        dest="blocked_award_revision",
        metavar="RECORD_REVISION",
        help=(
            "exact runs record revision this grant authorizes superseding "
            "(AW-1); the ONLY authority that releases the award-custody "
            "tripwire -- an ordinary grant authorizes work, never a downgrade"
        ),
    )
    parser.add_argument(
        "--release-park",
        dest="release_park",
        metavar="ROW_SHA256",
        help=(
            "append an operator void for one live grant-pending or "
            "award-supersession park row (the AW-1 park drain)"
        ),
    )
    parser.add_argument(
        "--capability",
        choices=sorted(GRANT_CAPABILITIES),
        help=(
            "typed capability this grant authorizes; host-capability-routing is what "
            "re-opens a deferred:needs-more-memory head (a generic grant never does)"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ledger", type=Path)
    return parser


def _refuse_provably_unbindable_grant(ledger: Path, stable_id: str) -> None:
    """Refuse minting a grant no selection arm could ever consume (D-48).

    A live terminal-refusal park fires at selection clause 5 BEFORE the grant
    arm, and ``_park_released_for_selection`` releases only the grant-pending
    and award dispositions -- so a grant minted against a refusal-parked member
    is dead on arrival: it re-enters nothing, and its existence tells the
    operator a false story about what was authorized. The refusal names the
    two REAL levers (the machine release lane on a tree change; an explicit
    park void). Read-side tolerance: an unreadable park ledger or models
    ledger reads as "unknown", never as a refusal -- the append-side integrity
    checks stay the authority, exactly like the consumption disclosure above.

    Parameters
    ----------
    ledger:
        Canonical requeue-grant ledger path (the park ledger's sibling root).
    stable_id:
        Member the operator is granting.

    Raises
    ------
    ValueError
        When a live terminal-refusal park row provably withholds the member.
    """

    from menagerie.crawler.refusal_parks import (
        RefusalParkError,
        load_refusal_park_rows,
        refusal_park_path,
        refusal_park_released,
    )
    from menagerie.crawler.state import _select_current

    records_root = ledger.parent.parent if ledger.parent.name == "operational" else ledger.parent
    try:
        rows = [
            row
            for row in load_refusal_park_rows([refusal_park_path(records_root)])
            if str(row.get("stable_id")) == stable_id
        ]
    except (OSError, RefusalParkError):
        return
    if not rows:
        return
    head = None
    models_dir = records_root / "models"
    if models_dir.is_dir():
        try:
            records = [
                record
                for shard in sorted(models_dir.glob("*.jsonl"))
                for record in scan_jsonl(shard, validate=False)
            ]
            head = _select_current(records).get(stable_id)
        except (OSError, ValueError):
            head = None
    live = [row for row in rows if not refusal_park_released(row, head)]
    if not live:
        return
    digests = ", ".join(sorted(str(row.get("row_sha256")) for row in live))
    raise ValueError(
        f"grant for {stable_id!r} is provably unbindable: a live terminal-refusal "
        f"park row ({digests}) withholds the member at selection clause 5 BEFORE the "
        "grant arm, and no grant releases that disposition. Real levers: fix the "
        "recording path in the code tree (the machine release lane readmits the "
        "member for one fresh generation on the next tree change), or void the park "
        "row explicitly (tools/refusal_parks.py)."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate and append one new work generation or one grant void.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero after a durable append, otherwise non-zero.
    """

    args = build_parser().parse_args(argv)
    stable_id: Optional[str] = args.stable_id
    grant_attempts: Optional[int] = args.grant
    stage: Optional[str] = args.stage
    try:
        if args.void_of is not None or args.release_park is not None:
            if args.void_of is not None and args.release_park is not None:
                raise ValueError("--void and --release-park are separate decisions")
            if stable_id is not None or grant_attempts is not None or stage is not None:
                raise ValueError("--void/--release-park take only --reason (and --granted-by)")
            if args.target_intent is not None or args.blocked_award_revision is not None:
                raise ValueError("--void/--release-park cannot carry an intent or award revision")
            if args.capability is not None:
                raise ValueError("--void/--release-park cannot carry a typed capability")
        elif stable_id is None or grant_attempts is None or stage is None:
            raise ValueError("grant mode requires stable_id, --grant, and --stage")
        repo_root = args.repo_root.resolve()
        ledger = args.ledger or (
            repo_root / "menagerie" / "crawler" / "records" / "operational" / "requeue-grants.jsonl"
        )
        owner = {
            "pid": os.getpid(),
            "run_id": "requeue-tool",
            "target": None,
        }
        with DriverLock(repo_root / ".crawl-local" / "locks" / "driver.lock", owner):
            if args.release_park is not None:
                # The AW-1 park drain: an append-only operator void of one
                # live grant-pending or award-supersession park row. It must
                # name a LIVE row exactly -- voiding nothing (or re-voiding)
                # reports failure, never a false success (delta vector-4's
                # no-op-reports-success tool lesson).
                from menagerie.crawler.grant_parks import (
                    append_grant_park_void,
                    build_grant_park_void,
                    grant_park_path,
                    load_grant_park_rows,
                )
                from menagerie.crawler.identity import utc_now

                records_root = (
                    ledger.parent.parent if ledger.parent.name == "operational" else ledger.parent
                )
                park_ledger = grant_park_path(records_root)
                live_rows = {
                    str(row["row_sha256"]): row for row in load_grant_park_rows([park_ledger])
                }
                if args.release_park not in live_rows:
                    raise ValueError(
                        f"no LIVE park row carries row_sha256 {args.release_park!r} "
                        "(already voided, already consumed, or never persisted)"
                    )
                payload = build_grant_park_void(
                    void_of=args.release_park,
                    reason=args.reason,
                    voided_by=args.granted_by,
                    created_at=utc_now(),
                )
                append_grant_park_void(park_ledger, payload)
            elif args.void_of is not None:
                payload = build_canonical_requeue_void(
                    ledger,
                    void_of=args.void_of,
                    reason=args.reason,
                    voided_by=args.granted_by,
                )
                # A void of an ALREADY-CONSUMED grant never reaches this append:
                # `build_canonical_requeue_void` refuses it typed and its message
                # names the consumption and the real lever. SYN-B adjudication
                # (see SYN_B_DOCKET.md, C3 ruling 1): lab c3a appended the row and
                # warned that it cancels nothing, lab c3b refuses at the producer.
                # Refusal wins -- a persisted row that reads as a revocation but
                # provably revokes nothing is a disclosure hazard in an
                # append-only ledger, and c3b's refusal text carries c3a's
                # disclosure content. The read side is untouched: a void already
                # persisted against a consumed grant stays inert, replayable
                # history.
                append_canonical_requeue_void(ledger, payload)
            else:
                assert stable_id is not None and grant_attempts is not None and stage is not None
                _refuse_provably_unbindable_grant(ledger, stable_id)
                payload = build_canonical_requeue_grant(
                    ledger,
                    stable_id=stable_id,
                    stage=stage,
                    reason=args.reason,
                    attempts=grant_attempts,
                    granted_by=args.granted_by,
                    target_intent=args.target_intent,
                    blocked_award_revision=args.blocked_award_revision,
                    capability=args.capability,
                )
                append_canonical_requeue_grant(ledger, payload)
                # MC-Q4 keep-and-disclose: attempts:N semantics are unchanged;
                # the operator sees, at issuance, the guard-session ceiling
                # those attempts imply. Stderr keeps stdout a parseable payload.
                print(
                    "effective guard-session ceiling: "
                    f"{payload['effective_session_ceiling']} paid sessions "
                    f"(attempts {payload['attempts']} x "
                    f"{EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT} sessions/attempt)",
                    file=sys.stderr,
                )
        print(json.dumps(payload, sort_keys=True))
    except (DriverLockError, OSError, ValueError) as exc:
        print(f"requeue failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
