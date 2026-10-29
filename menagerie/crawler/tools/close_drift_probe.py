"""Name a campaign's close-vs-books drift class BEFORE ops acts land.

The recurrence tripwire for the failure class "out-of-band operator append to
a close-attested source between a leg close and the next boundary stage" (the
R7 canary three-door deadlock, 2026-08-31): the PD-7 arm (c) holds staled the
step-200 close, and the staleness was discovered at the STAGING door, after
the acts had landed. This probe classifies the delta per census slot
(:mod:`menagerie.crawler.close_drift`) so an operator can ask, before OR after
an act, "does the books' state still re-observe the licensing close, and if
not, what delta class am I looking at?" — and get the class NAMED
(``review-hold-append`` vs ``out-of-band-drift``) instead of a generic census
mismatch at the stage door.

The probe is diagnostic only; no gate consumes its verdict. The remedy for a
staled close is a fresh close minted by the next leg — on a launched
successor rung, the zero-admission ``licensing_close`` purpose leg
(``crawl resume --leg-purpose licensing_close``, shape 4). The report also
carries the door-B catalog row's advisory: rung-class refinements that landed
retroactively over live leg history.

Standing ops rule: any hand-act on models/attempts/gates/artifacts, valves,
or park ledgers gets a probe run before and after.

Exit codes: 0 = the census re-observes byte-identically; 3 = drift (the JSON
report names the class per slot); 2 = a typed refusal (no close, bad state).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.close_drift import (
    DELTA_CLASS_NONE,
    close_source_drift,
    drift_delta_class,
    retroactive_reclass_warnings,
)
from menagerie.crawler.close_object import (
    CloseObjectError,
    close_object_path,
    find_close_object,
    load_close_objects,
)
from menagerie.crawler.rung_staging import RungStagingError, load_rung_registrations


def build_parser() -> argparse.ArgumentParser:
    """Build the close-drift probe argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--campaign", required=True, help="frozen tier campaign id")
    parser.add_argument(
        "--intake",
        help=(
            "explicit intake snapshot id whose newest close anchors the diff; "
            "defaults to the campaign's registered lineage tip"
        ),
    )
    return parser


def _records_root(args: argparse.Namespace) -> Path:
    """Resolve the records root from the parsed arguments."""

    if args.records_root is not None:
        return args.records_root.resolve()
    return args.repo_root.resolve() / "menagerie" / "crawler" / "records"


class _ProbeRefusal(ValueError):
    """A typed probe refusal, printed and mapped to exit code 2."""


def _resolve_close(
    records_root: Path, campaign_id: str, intake: Optional[str]
) -> tuple[str, dict]:
    """Resolve the diff anchor: the named (or tip) snapshot and its close."""

    intake_snapshot_id = intake
    if intake_snapshot_id is None:
        try:
            registrations = [
                row
                for row in load_rung_registrations(records_root)
                if row.campaign_id == campaign_id
            ]
        except RungStagingError as exc:
            raise _ProbeRefusal(str(exc)) from exc
        if not registrations:
            raise _ProbeRefusal(
                f"{campaign_id} has no registered rung, so no close is demanded "
                "and there is nothing to diff"
            )
        intake_snapshot_id = registrations[-1].snapshot_id
    try:
        close = find_close_object(
            load_close_objects([close_object_path(records_root)]),
            intake_snapshot_id=intake_snapshot_id,
            phase=None,
        )
    except CloseObjectError as exc:
        raise _ProbeRefusal(str(exc)) from exc
    if close is None:
        raise _ProbeRefusal(
            f"{intake_snapshot_id} has no close object; the boundary stage will "
            "demand one before any drift matters"
        )
    return intake_snapshot_id, close


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Probe one campaign's licensing close against the books as they are.

    Returns
    -------
    int
        0 when the census re-observes byte-identically, 3 on drift, 2 on a
        typed refusal.
    """

    args = build_parser().parse_args(argv)
    records_root = _records_root(args)
    try:
        intake_snapshot_id, close = _resolve_close(records_root, args.campaign, args.intake)
    except _ProbeRefusal as refusal:
        print(f"close-drift-probe refused: {refusal}", file=sys.stderr)
        return 2
    drifts = close_source_drift(records_root, close)
    delta_class = drift_delta_class(drifts)
    print(
        json.dumps(
            {
                "campaign_id": args.campaign,
                "intake_snapshot_id": intake_snapshot_id,
                "close_sha256": close.get("close_sha256"),
                "delta_class": delta_class,
                "retroactive_reclass_warnings": list(
                    retroactive_reclass_warnings(records_root)
                ),
                "sources": [
                    {
                        "name": drift.name,
                        "drift": drift.drift,
                        "attested_rows": drift.attested.get("rows"),
                        "observed_rows": drift.observed.get("rows"),
                        "appended_rows": (
                            None
                            if drift.appended_rows is None
                            else [
                                {
                                    "row_index": row_index,
                                    "stable_id": stable_id,
                                    "record_revision": address,
                                    "is_review_hold": is_hold,
                                }
                                for row_index, stable_id, address, is_hold in drift.appended_rows
                            ]
                        ),
                    }
                    for drift in drifts
                    if drift.drift != "identical"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if delta_class == DELTA_CLASS_NONE else 3


if __name__ == "__main__":
    raise SystemExit(main())
