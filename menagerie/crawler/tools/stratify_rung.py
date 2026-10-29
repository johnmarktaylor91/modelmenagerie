"""Compose a stratified rung member list with tail-class salting (STRAT1).

Every ladder rung from step-200 onward is stratified: deliberately salted with
tail-class representatives so surprise classes surface at rung width, never at
full-run width. This tool reads the campaign records state (its one write is
the durable membership-acknowledgment ledger, and only when
``--acknowledge-membership`` is passed and applied) and emits:

* a member list in the exact ``--stable-ids-file`` shape
  :mod:`menagerie.crawler.tools.stage_rung` consumes; and
* a JSON report naming every member's stratum and evidence, plus the
  per-stratum targets, pools, shortfalls, records-state digests, and the B-03
  ``composition_attestation`` binding the composed members to the snapshot
  identity they compose to.

It never stages anything itself; staging stays with ``stage_rung``, which
re-derives the attestation from the stage and refuses on any mismatch -- so pass
the report through, never a hand-copied member list or sha:

    python -m menagerie.crawler.tools.stratify_rung \\
        --records-root <records> --campaign c1-mech --size 200 --seed 200 \\
        --out-members step-200-members.txt --out-report step-200-report.json
    python -m menagerie.crawler.tools.stage_rung stage \\
        --records-root <records> --campaign c1-mech --label step-200 \\
        --stable-ids-file step-200-members.txt \\
        --composition-attestation step-200-report.json \\
        --award-authority <stable_id>=<revision> ...

A rung is a CUMULATIVE successor of its lineage tip (C-2), so the emitted member
list is the tip's membership PLUS the ``--size`` members this composition bought.
The carry is disclosed in the report's ``carried_tip`` block (tip identity, its
members, the count, and which members carry versus which a tail stratum selected
on its own evidence) and rides inside the composition digest as the
``carried-tip`` stratum. Carrying a settled award is still staging operator
property, so the report also names every ``--award-authority
stable_id=revision`` the staging door will demand, under
``award_authorities_required`` -- pass exactly those, never a hand tally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from menagerie.crawler.rung_stratification import (
    DEFAULT_ODDBALL_INTENTS,
    DEFAULT_SHARES,
    StratificationError,
    append_membership_acknowledgment,
    stratify_rung_members,
)


def _parse_share(value: str) -> tuple[str, float]:
    """Parse one ``name=fraction`` share override.

    Parameters
    ----------
    value:
        Raw ``--share`` argument.

    Returns
    -------
    tuple[str, float]
        Stratum name and share.

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument is not ``name=fraction``.
    """

    name, separator, raw_share = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError(f"share {value!r} is not name=fraction")
    try:
        return name, float(raw_share)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"share {value!r} is not name=fraction") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the stratify-rung tool argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--campaign", required=True, help="frozen tier campaign id")
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help=(
            "NEWLY composed rung width. The campaign's lineage tip is carried on "
            "top of it (membership is cumulative), so the emitted member list "
            "holds --size composed members plus every tip member no stratum "
            "already selected; the report's carried_tip block discloses which"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="determinism input; record it with the rung label",
    )
    parser.add_argument(
        "--out-members",
        type=Path,
        required=True,
        help=(
            "member list output (one stable id per line, stage_rung's "
            "--stable-ids-file shape): the CUMULATIVE membership, carried "
            "lineage-tip members first"
        ),
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        help="JSON strata report output; defaults to stdout",
    )
    parser.add_argument(
        "--share",
        dest="shares",
        action="append",
        type=_parse_share,
        metavar="STRATUM=FRACTION",
        help=(
            "override one stratum share; repeatable, unmentioned strata keep "
            f"their defaults ({json.dumps(dict(DEFAULT_SHARES))}); the total "
            "must sum to 1.0, so zeroing one stratum here means re-weighting "
            "the others yourself. To acknowledge a VANISHED stratum use "
            "--acknowledge-empty-stratum instead."
        ),
    )
    parser.add_argument(
        "--acknowledge-empty-stratum",
        dest="acknowledged_empty_strata",
        action="append",
        metavar="STRATUM",
        help=(
            "acknowledge one tail stratum as vanished; repeatable. The stratum "
            "must really hold zero candidates (proven against this records "
            "state, else the tool refuses); its share is zeroed, the reclaimed "
            "share goes to ladder-bulk, and the acknowledgement is disclosed in "
            "the report's acknowledged_empty_strata"
        ),
    )
    parser.add_argument(
        "--acknowledge-membership",
        dest="acknowledged_membership",
        action="append",
        metavar="STABLE_ID",
        help=(
            "acknowledge one membership-reconciliation census id (a model "
            "with canonical records outside every campaign universe or "
            "outside every registered rung); repeatable. The acknowledgment "
            "is written DURABLY to "
            "intake-extensions/membership-acknowledgments.jsonl and read by "
            "every later composition, so it is passed once, not re-typed per "
            "rung. It discloses the id in the report; it never selects it."
        ),
    )
    parser.add_argument(
        "--acknowledge-reason",
        default="operator acknowledgment via stratify_rung",
        help="reason recorded on each durable membership acknowledgment row",
    )
    parser.add_argument(
        "--acknowledged-by",
        default="operator",
        help="operator identity recorded on each durable acknowledgment row",
    )
    parser.add_argument(
        "--pd7-attestation",
        dest="pd7_attestation",
        type=Path,
        metavar="PATH",
        help=(
            "PD-7 slop-family-routes adjudication artifact (the operator "
            "composition attestation carrying pd7_slop_family_routes.status). "
            "Its status and its file sha256 are bound INTO the emitted B-03 "
            "composition attestation (SOL-02), which is what the staging door "
            "consumes — a successor-rung stage refuses without a digest-bound "
            "SATISFIED disposition"
        ),
    )
    parser.add_argument(
        "--oddball-intent",
        dest="oddball_intents",
        action="append",
        metavar="INTENT",
        help=(
            "override the oddball/lockless intent set; repeatable "
            f"(default: {', '.join(sorted(DEFAULT_ODDBALL_INTENTS))})"
        ),
    )
    return parser


def _pd7_disposition(path: Optional[Path]) -> Optional[dict[str, str]]:
    """Read and digest-bind one PD-7 adjudication artifact (SOL-02).

    The artifact's ``pd7_slop_family_routes.status`` becomes the attested
    status and the artifact's own file sha256 becomes the bound source
    reference. An unreadable or shapeless artifact refuses typed — a PD-7
    claim the composer cannot actually read licenses nothing.
    """

    if path is None:
        return None
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StratificationError(f"--pd7-attestation {path} is unreadable: {exc}") from exc
    block = payload.get("pd7_slop_family_routes") if isinstance(payload, dict) else None
    status = block.get("status") if isinstance(block, dict) else None
    if not isinstance(status, str) or not status:
        raise StratificationError(
            f"--pd7-attestation {path} carries no pd7_slop_family_routes.status; "
            "the composer binds the adjudication it actually read, never a guess"
        )
    return {
        "status": status,
        "source_attestation_sha256": "sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
    }


def _resolved_shares(
    overrides: Optional[Sequence[tuple[str, float]]],
) -> Optional[Mapping[str, float]]:
    """Merge share overrides over the defaults, or keep the defaults."""

    if not overrides:
        return None
    merged = dict(DEFAULT_SHARES)
    merged.update(dict(overrides))
    return merged


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Compose one stratified rung member list and its strata report.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero on success, non-zero after a typed stratification refusal.
    """

    args = build_parser().parse_args(argv)
    records_root = (
        args.records_root.resolve()
        if args.records_root is not None
        else args.repo_root.resolve() / "menagerie" / "crawler" / "records"
    )
    try:
        pd7_disposition = _pd7_disposition(args.pd7_attestation)
        plan = stratify_rung_members(
            records_root,
            campaign_id=args.campaign,
            size=args.size,
            seed=args.seed,
            shares=_resolved_shares(args.shares),
            oddball_intents=args.oddball_intents,
            acknowledged_empty_strata=args.acknowledged_empty_strata,
            acknowledged_membership=args.acknowledged_membership,
            pd7_disposition=pd7_disposition,
        )
    except StratificationError as exc:
        print(f"stratify-rung refused: {exc}", file=sys.stderr)
        return 2
    # The flag is the WRITE path of the durable acknowledgment ledger
    # (CAT-02): every flag the census actually APPLIED is persisted once per
    # campaign, so every later composition reads it back without re-typed
    # flags. A flag that matched nothing refused above and persists nothing.
    applied = set(plan.membership_reconciliation.get("acknowledged", ()))
    for stable_id in args.acknowledged_membership or ():
        if str(stable_id) in applied:
            append_membership_acknowledgment(
                records_root,
                campaign_id=args.campaign,
                stable_id=str(stable_id),
                reason=args.acknowledge_reason,
                acknowledged_by=args.acknowledged_by,
                created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
    report = json.dumps(plan.to_report(), sort_keys=True, indent=2)
    args.out_members.parent.mkdir(parents=True, exist_ok=True)
    args.out_members.write_text(
        "".join(f"{stable_id}\n" for stable_id in plan.member_ids()), encoding="utf-8"
    )
    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
