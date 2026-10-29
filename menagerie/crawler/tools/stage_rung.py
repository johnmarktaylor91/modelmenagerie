"""Stage, register, and list durable campaign rung intake snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.close_object import CloseObjectError
from menagerie.crawler.partitioner import (
    CampaignPartitionError,
    refresh_campaign_manifest,
)
from menagerie.crawler.rung_staging import (
    RUNG_CLASSES,
    RungCompositionAttestation,
    RungStagingError,
    StagedRung,
    append_rung_class_refinement,
    load_composition_attestation,
    load_rung_registrations,
    register_existing_snapshot,
    stage_rung,
)


def _utc_now() -> str:
    """Return the current UTC timestamp in the ledger's format."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    """Build the stage-rung tool argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Typed parser with ``stage``, ``register``, and ``list`` actions.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo-root", type=Path, default=Path.cwd())
        subparser.add_argument("--records-root", type=Path)

    stage = actions.add_parser(
        "stage",
        help="build and register the next rung intake from the campaign catalog",
    )
    add_common(stage)
    stage.add_argument("--campaign", required=True, help="frozen tier campaign id")
    stage.add_argument(
        "--stable-id",
        dest="stable_ids",
        action="append",
        metavar="STABLE_ID",
        help="explicit rung member; repeatable (the mini-rung shape)",
    )
    stage.add_argument(
        "--stable-ids-file",
        type=Path,
        help="file with one stable id per line, composed with --stable-id",
    )
    stage.add_argument(
        "--count",
        type=int,
        help="stage the first N partition rows instead (the step-200 shape)",
    )
    stage.add_argument(
        "--derived-from",
        dest="derived_from",
        metavar="SNAPSHOT_ID",
        help=(
            "explicit prior snapshot; defaults to the campaign's lineage tip. Once "
            "any rung is registered the tip is the ONLY lineage either door accepts "
            "-- naming the campaign root or an older rung refuses, because the "
            "close gate reads the lineage from the state, not from this flag"
        ),
    )
    stage.add_argument("--label", required=True, help="rung name, e.g. mini-10 or step-200")
    stage.add_argument("--registered-by", default="operator")
    stage.add_argument(
        "--allow-lockless",
        action="append",
        default=[],
        metavar="INTENT",
        help=(
            "EN-2 exemption: environment intent that may compose members while its "
            "lock family is absent; validated against the LIVE registry (an entry "
            "naming a locked intent refuses as stale); repeatable"
        ),
    )
    stage.add_argument(
        "--target",
        default="osx-arm64",
        help="exact conda-lock platform gating the EN-2 lockless-route refusal",
    )
    stage.add_argument(
        "--rung-class",
        dest="rung_class",
        choices=sorted(RUNG_CLASSES),
        help=(
            "closed staging-time class declaration (W16/S-2): what the composed "
            "rung IS. REQUIRED when the lineage close is live; bound into the "
            "registration's content address, and every certificate consumer "
            "replay-derives its demanded certificate_kind from it — a "
            "successor_rung's legs demand rung_launch, canary_seed and "
            "adopted_history admit under canary_admission"
        ),
    )
    _add_award_authority(stage)
    _add_composition_attestation(stage)

    register = actions.add_parser(
        "register",
        help="adopt an existing snapshot directory into records/intake (OPS-2)",
    )
    add_common(register)
    register.add_argument("--campaign", required=True, help="frozen tier campaign id")
    register.add_argument(
        "--snapshot-root",
        type=Path,
        required=True,
        help="existing intake snapshot directory to adopt",
    )
    register.add_argument(
        "--derived-from",
        dest="derived_from",
        metavar="SNAPSHOT_ID",
        help=(
            "explicit prior snapshot; defaults to the campaign's lineage tip. Once "
            "any rung is registered the tip is the ONLY lineage either door accepts "
            "-- naming the campaign root or an older rung refuses, because the "
            "close gate reads the lineage from the state, not from this flag"
        ),
    )
    register.add_argument("--label", required=True)
    register.add_argument("--registered-by", default="operator")
    register.add_argument(
        "--new-work",
        action="store_true",
        help=(
            "register a snapshot composed for NEW work instead of provable "
            "already-run history; runs through the same AW-2 award-authority "
            "guard as stage"
        ),
    )
    register.add_argument(
        "--rung-class",
        dest="rung_class",
        choices=sorted(RUNG_CLASSES),
        help=(
            "closed staging-time class declaration (W16/S-2) for a --new-work "
            "adoption under a live lineage close; a proven-history adoption "
            "derives adopted_history from the door's own proof and refuses a "
            "conflicting claim"
        ),
    )
    _add_award_authority(register)
    _add_composition_attestation(register)

    refine = actions.add_parser(
        "refine-class",
        help=(
            "class ONE pre-vocabulary (unclassed) registration through the "
            "append-once refinement ledger (R2-4/SOL-01): the registry can never "
            "rebind an existing row, so this is the one lawful door for making a "
            "historical_bootstrap-era rung demand its true certificate kind"
        ),
    )
    add_common(refine)
    refine.add_argument(
        "--snapshot-id",
        required=True,
        help="exact snapshot id of the registered rung being classed",
    )
    refine.add_argument(
        "--snapshot-sha256",
        help=(
            "exact snapshot digest; defaults to the registered row's own digest "
            "(the append-once registry holds at most one row per snapshot id)"
        ),
    )
    refine.add_argument(
        "--rung-class",
        dest="rung_class",
        required=True,
        choices=sorted(RUNG_CLASSES),
        help="the declared class; every certificate consumer replays its demanded kind from it",
    )
    refine.add_argument("--refined-by", default="operator")

    listing = actions.add_parser("list", help="print registered rungs and their lineage")
    add_common(listing)

    refresh = actions.add_parser(
        "refresh-manifest",
        help=(
            "re-emit campaigns.json spec-projection fields from the live frozen "
            "campaign specs, preserving partition and intake bindings (OPS-10)"
        ),
    )
    add_common(refresh)
    refresh.add_argument(
        "--rotate-identity-projections",
        action="store_true",
        help=(
            "also re-project author_model/checker_model, recording each "
            "superseded value under superseded_projection. Off by default: "
            "overwriting them in place is what destroyed which model "
            "adjudicated which rung (BL-3)"
        ),
    )
    return parser


def _add_award_authority(subparser: argparse.ArgumentParser) -> None:
    """Add the repeatable AW-2 award-authority flag to one staging door.

    BOTH doors express award authority (2026-08-16 red-team break 8: the
    ``register`` subparser had no flag at all, so the un-gated door could not
    even be asked for one).
    """

    subparser.add_argument(
        "--award-authority",
        dest="award_authorities",
        action="append",
        metavar="STABLE_ID=RECORD_REVISION",
        help=(
            "explicit operator authority to stage one runs-award holder, "
            "naming its exact current raw revision; repeatable (AW-2 "
            "staging tripwire). A cumulative successor CARRIES its lineage "
            "tip, so the tip's converted models need one too -- "
            "stratify_rung's report names every pair under "
            "award_authorities_required"
        ),
    )


def _add_composition_attestation(subparser: argparse.ArgumentParser) -> None:
    """Add the B-03 composition-attestation flag to one staging door.

    BOTH doors accept it, for the same reason both express award authority: a
    gate that lives on one entry point is a gate the other door walks around
    (2026-08-16 red-team break 1).
    """

    subparser.add_argument(
        "--composition-attestation",
        dest="composition_attestation",
        type=Path,
        metavar="PATH",
        help=(
            "the stratify_rung --out-report (or a bare attestation object) "
            "whose composition is enforced against this stage: campaign, "
            "member set, and the snapshot identity those members compose to "
            "must all match, or staging refuses (B-03)"
        ),
    )


def _composition_attestation(
    args: argparse.Namespace,
) -> Optional[RungCompositionAttestation]:
    """Load the presented composition attestation, if any."""

    path = getattr(args, "composition_attestation", None)
    if path is None:
        return None
    return load_composition_attestation(Path(path))


def _records_root(args: argparse.Namespace) -> Path:
    """Resolve the records root from the parsed arguments."""

    if args.records_root is not None:
        return args.records_root.resolve()
    return args.repo_root.resolve() / "menagerie" / "crawler" / "records"


def _award_authorities(args: argparse.Namespace) -> Optional[dict[str, str]]:
    """Parse repeated ``stable_id=revision`` award authorities."""

    raw = getattr(args, "award_authorities", None)
    if not raw:
        return None
    authorities: dict[str, str] = {}
    for entry in raw:
        stable_id, separator, revision = str(entry).partition("=")
        if not separator or not stable_id or not revision:
            raise RungStagingError(
                f"--award-authority must be STABLE_ID=RECORD_REVISION, got {entry!r}"
            )
        authorities[stable_id] = revision
    return authorities


def _stage_members(args: argparse.Namespace) -> Optional[tuple[str, ...]]:
    """Collect explicit stage members from flags and the optional id file."""

    members: list[str] = list(args.stable_ids or ())
    if args.stable_ids_file is not None:
        members.extend(
            line.strip()
            for line in args.stable_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return tuple(members) or None


def _report(staged: StagedRung) -> str:
    """Render one staging result as a JSON report."""

    registration = staged.registration
    return json.dumps(
        {
            "campaign_id": registration.campaign_id,
            "snapshot_id": registration.snapshot_id,
            "snapshot_sha256": registration.snapshot_sha256,
            "derived_from_snapshot_id": registration.derived_from_snapshot_id,
            "member_count": registration.member_count,
            "label": registration.label,
            "snapshot_root": staged.snapshot_root.as_posix(),
            "snapshot_created": staged.snapshot_created,
            "registration_created": staged.registration_created,
        },
        sort_keys=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Stage, register, or list durable campaign rungs.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero on success, non-zero after a typed staging refusal.
    """

    args = build_parser().parse_args(argv)
    records_root = _records_root(args)
    if args.action == "refresh-manifest":
        manifest_path = records_root / "campaigns.json"
        try:
            bindings = refresh_campaign_manifest(
                manifest_path,
                rotate_identity_projections=bool(
                    getattr(args, "rotate_identity_projections", False)
                ),
            )
        except CampaignPartitionError as exc:
            print(f"refresh-manifest refused: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "manifest": manifest_path.as_posix(),
                    "campaigns": {
                        binding.spec.campaign_id: {
                            "author_model": binding.spec.author_model,
                            "checker_model": binding.spec.checker_model,
                            "intake_snapshot_id": binding.intake_snapshot_id,
                            "partition_sha256": binding.partition_sha256,
                            "recorded_projection": dict(binding.recorded_projection),
                            "projection_drift": list(binding.projection_drift),
                        }
                        for binding in bindings
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        if args.action == "refine-class":
            snapshot_sha256 = args.snapshot_sha256
            if snapshot_sha256 is None:
                registration = next(
                    (
                        row
                        for row in load_rung_registrations(records_root)
                        if row.snapshot_id == args.snapshot_id
                    ),
                    None,
                )
                if registration is None:
                    raise RungStagingError(
                        f"no registration exists for snapshot {args.snapshot_id}"
                    )
                snapshot_sha256 = registration.snapshot_sha256
            refinement, created = append_rung_class_refinement(
                records_root,
                snapshot_id=args.snapshot_id,
                snapshot_sha256=snapshot_sha256,
                rung_class=args.rung_class,
                refined_by=args.refined_by,
                created_at=_utc_now(),
            )
            print(
                json.dumps(
                    {"refinement": refinement.to_row(), "refinement_created": created},
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "list":
            print(
                json.dumps(
                    {
                        "registered_rungs": [
                            registration.to_row()
                            for registration in load_rung_registrations(records_root)
                        ]
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "stage":
            staged = stage_rung(
                records_root,
                campaign_id=args.campaign,
                stable_ids=_stage_members(args),
                count=args.count,
                derived_from_snapshot_id=args.derived_from,
                label=args.label,
                registered_by=args.registered_by,
                created_at=_utc_now(),
                award_authorities=_award_authorities(args),
                composition_attestation=_composition_attestation(args),
                target=args.target,
                allow_lockless=tuple(args.allow_lockless),
                rung_class=args.rung_class,
            )
        else:
            authorities = _award_authorities(args)
            if authorities and not args.new_work:
                # The historical door never consults award authorities; taking
                # them silently would let the operator believe the AW-2 gate
                # adjudicated something it never saw.
                raise RungStagingError(
                    "--award-authority applies to the AW-2 new-work guard: pass "
                    "--new-work with it (a provable historical adoption needs "
                    "no authority; its awards are that run's own output)"
                )
            staged = register_existing_snapshot(
                records_root,
                args.snapshot_root.resolve(),
                campaign_id=args.campaign,
                derived_from_snapshot_id=args.derived_from,
                label=args.label,
                registered_by=args.registered_by,
                created_at=_utc_now(),
                historical_only=not args.new_work,
                award_authorities=authorities,
                composition_attestation=_composition_attestation(args),
                rung_class=args.rung_class,
            )
    except (RungStagingError, CloseObjectError) as exc:
        # The close-object refusal is a staging refusal like any other: it must
        # print and exit non-zero, never traceback (BD-1).
        print(f"stage-rung refused: {exc}", file=sys.stderr)
        return 2
    print(_report(staged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
