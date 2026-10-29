"""Typed command-line interface for the resumable menagerie crawler."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
import json
import os
import shlex
import shutil
import sys
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from menagerie.crawler.authority import AuthorityContext, build_authority_context
from menagerie.crawler.campaign_config import (
    CampaignConfig,
    LaunchEnvironment,
    default_campaign_config_path,
    load_campaign_config,
    resolve_command,
    write_campaign_config,
)
from menagerie.crawler.checkpoint import (
    CheckpointError,
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    canonical_operational_ledger_path,
    create_canonical_checkpoint,
)
from menagerie.crawler.checker_lane import (
    checker_lane_deferral_path,
    load_checker_lane_deferral_rows,
)
from menagerie.crawler.author_dispatch import AuthorEffortGrant
from menagerie.crawler.constants import (
    AUTHOR_WALL_SECONDS_ENV,
    DEFAULT_AUTHOR_WAVE_CONCURRENCY,
    DEFAULT_CHECKER_WAVE_CONCURRENCY,
    DEFAULT_SCHEDULER,
    InvocationOrigin,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
    SCHEDULER_MODES,
    TIER_CAMPAIGN_IDS,
    framework_from_markers,
    resolve_author_wall_seconds,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.doctor import DoctorConfig, DoctorError, DoctorProbes, run_doctor
from menagerie.crawler.driver import (
    CommandAuthorLane,
    CommandCheckerLane,
    CommandNotifier,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverError,
    DriverLock,
    DriverLockError,
    DriverPaths,
    DriverResult,
    QueueAuthorLane,
    SupervisedForwardLane,
    build_command_environment_lane,
    default_driver_paths,
    durable_readiness_accounting,
    utc_now,
)
from menagerie.crawler.crawl_preflight import InvocationRefused
from menagerie.crawler.driver_contracts import RetryableOperatorError
from menagerie.crawler.driver_admission import (
    _deferral_row_fact_still_holds,
    restored_environment_generations,
)
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, load_environment_registry
from menagerie.crawler.environment_coverage import (
    COVERAGE_DEFERRAL_DISPOSITION,
    coverage_deferral_path,
    load_coverage_deferral_rows,
)
from menagerie.crawler.environment_rebuild import (
    load_rebuild_deferral_rows,
    rebuild_deferral_path,
)
from menagerie.crawler.disk_floor import (
    disk_floor_deferral_path,
    load_disk_floor_deferral_rows,
)
from menagerie.crawler.grant_parks import (
    GRANT_PENDING_DISPOSITION,
    SELECTION_EXCLUSION_DISPOSITION,
    grant_park_path,
    grant_park_released,
    load_grant_park_rows,
)
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_DISPOSITION,
    load_refusal_park_rows,
    refusal_park_path,
    refusal_park_released,
)
from menagerie.crawler.scheduler_parks import (
    load_scheduler_quarantine_rows,
    scheduler_quarantine_path,
)
from menagerie.crawler.host_capacity import (
    CAPACITY_DEFERRAL_DISPOSITION,
    capacity_deferral_path,
    host_capacity,
    load_capacity_deferral_rows,
)
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot, load_intake_snapshot
from menagerie.crawler.leg_purpose import LEG_PURPOSES
from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes
from menagerie.crawler.invocation_identity import mint_default_run_id
from menagerie.crawler.operator_environment import build_environment_readiness_receipt
from menagerie.crawler.rung_staging import (
    UnboundIntakeSnapshotError,
    find_rung_registration,
    registered_prior_intake_members,
    registered_prior_intake_snapshots,
    registration_binding,
)
from menagerie.crawler.partitioner import (
    DEFAULT_CAMPAIGN_MANIFEST,
    CampaignBinding,
    emit_campaign_partitions,
    find_campaign_binding,
    load_campaign_bindings,
)
from menagerie.crawler.paid_session_lifecycle import CommandProviderReconciler
from menagerie.crawler.promotion import (
    promotion_records_root,
    promotion_snapshot_destination,
)
from menagerie.crawler.recordio import SingleWriterError
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.recheck_census import recheck_census
from menagerie.crawler.reducer import (
    default_ledger_paths,
    materialize_current,
    project_dependency_current,
)
from menagerie.crawler.rotations import load_rotations, rotation_history
from menagerie.crawler.routing import phase_routes, route_model
from menagerie.crawler.state import _select_current
from menagerie.crawler.status import (
    PartitionError,
    assert_partition,
    awards_stale_on_live_basis,
    funnel_counts,
    grant_release_census_from_ledgers,
    machine_deferral_rejection,
    scheduler_drain_census_from_ledgers,
    partition_report,
    stale_census,
    wakeup_status,
)
from menagerie.crawler.wakeup import (
    OperationalContext,
    WakeupManager,
    reduce_wake_episodes,
    render_wakeup_status,
    wake_liveness_alarms,
    wakeup_projection_root,
    write_fire_intent,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_LOCKED = 3
EXIT_PAUSED = 4
EXIT_REVIEW_PAUSED = 5
EXIT_OPERATOR_OUTAGE = 6


class DivergentIntakeRegistriesError(RuntimeError):
    """D-47 (B3-XO-27): two intake registries disagree on one snapshot.

    The campaign manifest and the rung registry both claim the snapshot but
    name different campaigns. A silent preference (the manifest historically
    won unconsulted) would run the wrong campaign's models; leg open refuses
    typed instead.
    """


class OperatorOutageError(RuntimeError):
    """Raised when required operator-side commands or paths are unavailable."""


class DriverFactory(Protocol):
    """CLI injection point for fully configured live or fake driver lanes."""

    def __call__(self, args: argparse.Namespace) -> CrawlerDriver:
        """Build a driver for parsed run/resume/handoff arguments."""

        ...


def build_parser() -> argparse.ArgumentParser:
    """Build the complete typed crawler argument parser."""

    parser = argparse.ArgumentParser(prog="python -m menagerie.crawler", description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help="TorchLens worktree root"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="run the strict campaign preflight")
    doctor.add_argument("--target", choices=("osx-arm64", "linux-x86_64-cuda"), required=True)
    doctor.add_argument(
        "--strict", action="store_true", help="enforce every frozen preflight check"
    )
    doctor.add_argument("--minimum-disk-gib", type=int, default=100)
    _add_intake_argument(doctor, required=False)
    doctor.add_argument(
        "--intake-items",
        type=int,
        help="declared intake size; scales the disk floor without reading a snapshot",
    )

    intake = subparsers.add_parser("intake", help="create an immutable intake snapshot")
    intake.add_argument("--all-existing", action="store_true", help="use menagerie/data catalogs")
    intake.add_argument("--master", type=Path)
    intake.add_argument("--deferred", type=Path)
    intake.add_argument("--stable-ids", type=Path)
    intake.add_argument("--output-root", type=Path)
    intake.add_argument("--snapshot-date", help="operator label retained by surrounding automation")

    campaigns = subparsers.add_parser(
        "partition-campaigns",
        help="emit the four tier-aligned campaign partitions and intake snapshots",
    )
    campaigns.add_argument("--roster", type=Path)
    campaigns.add_argument("--stable-ids", type=Path)
    campaigns.add_argument("--output-root", type=Path)

    plan = subparsers.add_parser("plan", help="print deterministic phase/intent assignments")
    _add_intake_argument(plan)
    _add_target_phase_arguments(plan)

    readiness = subparsers.add_parser(
        "seal-environment-readiness",
        help="seal the hydrated CAS closure routed by one frozen intake",
    )
    _add_intake_argument(readiness)
    readiness.add_argument("--target", choices=("osx-arm64", "linux-x86_64-cuda"), required=True)
    readiness.add_argument("--envs-root", type=Path, default=DEFAULT_ENVS_ROOT)
    readiness.add_argument("--state-root", type=Path, required=True)
    readiness.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="run or resume the single-writer scheduler")
    _add_intake_argument(run, required=False)
    _add_target_phase_arguments(run)
    _add_driver_config_arguments(run)
    _add_dry_run_arguments(run)
    run.add_argument("--resume", action="store_true", help="resume durable unsatisfied identities")
    _add_selection_scope_arguments(run)
    run.add_argument(
        "--acknowledge-award-holds",
        action="store_true",
        help=(
            "launch despite projection-stale runs awards or live award-"
            "supersession parks (AW-1/AW-2); the award-holds-census event is "
            "appended either way -- read it first. D-47: a nonempty census "
            "also REQUIRES the binding --award-holds-digest"
        ),
    )
    _add_census_ack_arguments(run)
    run.add_argument(
        "--acknowledge-repeat-cause",
        metavar="RATIONALE",
        help=(
            "fund ONE retry past a tripped repeat-cause fuse (PM-5/W6): the "
            "rationale is appended as a durable operator row; never persists"
        ),
    )
    run.add_argument(
        "--supervised-restart-ordinal",
        type=int,
        metavar="N",
        help=(
            "attest this invocation as the supervisor's Nth bounded restart "
            "(W6 ladder); supervisor-provided, never operator-typed"
        ),
    )
    run.add_argument(
        "--sequential-envs",
        action="store_true",
        default=True,
        help="required one-at-a-time environment lifecycle",
    )

    # Overhaul RC-1 (RULING D.1): the continuous-crawl invocation. Binds the
    # campaign from the records root's campaigns.json; the active intake is the
    # campaign's full partition snapshot; the persisted stratified order plus
    # --limit replaces rungs; the in-process preflight replaces certificates.
    crawl = subparsers.add_parser(
        "crawl",
        help=(
            "continuous crawl over a campaign: in-process preflight, persisted "
            "stratified order + --limit, leg_open as the invocation record, typed exit"
        ),
    )
    crawl.add_argument("--records-root", type=Path, required=True)
    crawl.add_argument("--campaign", required=True, choices=sorted(TIER_CAMPAIGN_IDS))
    crawl.add_argument(
        "--limit",
        type=int,
        help="admit the first N rows of the persisted order (omitted: the whole order)",
    )
    crawl.add_argument(
        "--author-width",
        type=int,
        dest="author_width",
        help="author sessions in flight (paced start caps it at 4 for a new configuration)",
    )
    crawl.add_argument("--order-seed", type=int, default=None, help=argparse.SUPPRESS)
    crawl.add_argument(
        "--target", choices=("osx-arm64", "linux-x86_64-cuda"), default="osx-arm64"
    )
    crawl.add_argument("--after-review", action="store_true")
    crawl.add_argument("--supervised-restart-ordinal", type=int, metavar="N", help=argparse.SUPPRESS)
    _add_driver_config_arguments(crawl)

    status = subparsers.add_parser("status", help="print the current terminal funnel")
    _add_intake_argument(status)
    status.add_argument("--records-root", type=Path)
    status.add_argument("--full", action="store_true")
    status.add_argument("--verify-partition", action="store_true")
    status.add_argument(
        "--require-valid-partition",
        action="store_true",
        help=(
            "fail closed: exit non-zero when --verify-partition finds an invalid "
            "partition. Mid-leg progress inspection legitimately reports an "
            "incomplete partition, so this is the explicit rung-boundary form -- "
            "use it wherever a gate keys on the exit code"
        ),
    )
    status.add_argument(
        "--require-live-wake",
        action="store_true",
        help=(
            "fail closed: exit non-zero when an ACTIVE wake episode is overdue "
            "(past its reset plus grace with no recorded fire). The pilot's only "
            "real pause reported healthy while it sat dead past its reset, so a "
            "gate that keys on wake health must key on this"
        ),
    )

    capacity = subparsers.add_parser(
        "capacity-deferrals",
        help="list models withheld because this host is too small to instantiate them",
    )
    capacity.add_argument("--records-root", type=Path)
    capacity.add_argument(
        "--full",
        action="store_true",
        help="print each row's complete estimate and threshold",
    )

    coverage = subparsers.add_parser(
        "coverage-deferrals",
        help="list models whose routed environment cannot install their library",
    )
    coverage.add_argument("--records-root", type=Path)
    coverage.add_argument(
        "--full",
        action="store_true",
        help="print each row's complete coverage assessment and evidence",
    )

    wake = subparsers.add_parser("wake", help="inspect or cancel durable wake episodes")
    wake.add_argument("--records-root", type=Path)
    wake_actions = wake.add_subparsers(dest="wake_command", required=True)
    wake_actions.add_parser("status", help="print durable wake episode state")
    wake_cancel = wake_actions.add_parser("cancel", help="cancel one exact wake episode")
    wake_cancel.add_argument("episode_id")

    checkpoint = subparsers.add_parser("checkpoint", help="verify ledgers and disposable views")
    _add_intake_argument(checkpoint)
    checkpoint.add_argument("--records-root", type=Path)
    checkpoint.add_argument("--verify-ledgers", action="store_true")
    checkpoint.add_argument("--verify-views", action="store_true")

    teardown = subparsers.add_parser(
        "teardown", help="remove only disposable crawler runtime state"
    )
    teardown.add_argument("--target", choices=("osx-arm64", "linux-x86_64-cuda"), required=True)
    teardown.add_argument("--active-env", action="store_true")
    teardown.add_argument("--verify-disk", action="store_true")

    requeue = subparsers.add_parser("requeue", help="record an explicit bounded effort grant")
    requeue.add_argument("stable_id")
    requeue.add_argument("--reason", required=True)
    requeue.add_argument("--grant", required=True, type=int)
    requeue.add_argument("--stage", required=True)
    requeue.add_argument("--granted-by", default="operator")
    requeue.add_argument(
        "--intent",
        dest="target_intent",
        help="dependency-evidenced corrected environment intent",
    )

    handoff = subparsers.add_parser(
        "handoff", aliases=["handoff-linux"], help="run Linux deferred sweep"
    )
    _add_intake_argument(handoff)
    _add_driver_config_arguments(handoff)
    handoff.add_argument("--resume", action="store_true")
    handoff.add_argument("--only-status", default="deferred:*")
    handoff.set_defaults(target="linux-x86_64-cuda", phase=None)

    resume = subparsers.add_parser("resume", help="resume a paused campaign")
    _add_intake_argument(resume, required=False)
    _add_target_phase_arguments(resume)
    _add_driver_config_arguments(resume)
    _add_dry_run_arguments(resume)
    _add_selection_scope_arguments(resume)
    resume.add_argument(
        "--acknowledge-award-holds",
        action="store_true",
        help=(
            "launch despite projection-stale runs awards or live award-"
            "supersession parks (AW-1/AW-2); the award-holds-census event is "
            "appended either way -- read it first. D-47: a nonempty census "
            "also REQUIRES the binding --award-holds-digest"
        ),
    )
    _add_census_ack_arguments(resume)
    resume.add_argument(
        "--acknowledge-repeat-cause",
        metavar="RATIONALE",
        help=(
            "fund ONE retry past a tripped repeat-cause fuse (PM-5/W6): the "
            "rationale is appended as a durable operator row; never persists"
        ),
    )
    resume.add_argument(
        "--supervised-restart-ordinal",
        type=int,
        metavar="N",
        help=(
            "attest this invocation as the supervisor's Nth bounded restart "
            "(W6 ladder); supervisor-provided, never operator-typed"
        ),
    )
    resume.add_argument(
        "--after-review",
        action="store_true",
        help="record human sign-off for the one-shot review checkpoint",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    driver_factory: Optional[DriverFactory] = None,
    doctor_probes: Optional[DoctorProbes] = None,
) -> int:
    """Parse and execute one crawler command with stable exit codes."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor_command(args, doctor_probes)
        if args.command == "intake":
            return _intake_command(args)
        if args.command == "partition-campaigns":
            return _partition_campaigns_command(args)
        if args.command == "plan":
            return _plan_command(args)
        if args.command == "seal-environment-readiness":
            return _seal_environment_readiness_command(args)
        if args.command == "status":
            return _status_command(args)
        if args.command == "capacity-deferrals":
            return _capacity_deferrals_command(args)
        if args.command == "coverage-deferrals":
            return _coverage_deferrals_command(args)
        if args.command == "wake":
            return _wake_command(args)
        if args.command == "checkpoint":
            return _checkpoint_command(args)
        if args.command == "teardown":
            return _teardown_command(args)
        if args.command == "requeue":
            return _requeue_command(args)
        if args.command == "crawl":
            return _crawl_command(args, driver_factory)
        if args.command in {"run", "resume", "handoff", "handoff-linux"}:
            if (
                args.command in {"run", "resume"}
                and driver_factory is None
                and not bool(getattr(args, "dry_run", False))
            ):
                # LB-2: the readiness receipt is checked against
                # MENAGERIE_ENVIRONMENT_STATE_ROOT, and a launchd wake inherits
                # a bare environment -- so the frozen launch contract has to be
                # restored BEFORE the check, not later in the driver factory.
                # Memoized, so the factory reuses this exact resolution and the
                # divergence disclosure is not dropped by a second pass.
                # The pre-pass runs FOR the receipt check's benefit, so its own
                # refusals must not SHADOW the receipt refusal's identity when
                # the receipt is the true blocker: deferred, never swallowed.
                deferred_outage: Optional[OperatorOutageError] = None
                try:
                    _load_or_persist_campaign_config(args)
                except OperatorOutageError as outage:
                    deferred_outage = outage
                _require_environment_readiness_receipt(args)
                if deferred_outage is not None:
                    raise deferred_outage
            return _driver_command(args, driver_factory or _default_driver_factory)
    except DoctorError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    except InvocationRefused as exc:
        # RC-1: typed, already recorded as an invocation-refused event when the
        # ledger was reachable; no traceback -- the arm and detail ARE the record.
        print(
            json.dumps(
                {"status": "invocation-refused", "arm": exc.arm, "detail": exc.detail},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_OPERATOR_OUTAGE
    except DriverLockError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_LOCKED
    except OperatorOutageError as exc:
        print(
            json.dumps({"status": "operator-outage", "detail": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return EXIT_OPERATOR_OUTAGE
    except RetryableOperatorError as exc:
        print(
            json.dumps(
                {
                    "status": "retryable-infrastructure",
                    "detail": str(exc),
                    "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_OPERATOR_OUTAGE
    except (CheckpointError, DriverError, OSError, ValueError) as exc:
        # This clause catches a very wide class -- every ``ValueError`` raised anywhere in the
        # engine lands here, including every ``ReductionError``. Printing only ``str(exc)``
        # discards the one thing needed to act on it: where it came from.
        #
        # A real campaign failure read, in its entirety,
        #     crawler error: missing mandatory exact public primary source link
        # with no stack, no file, no line, and no model ID. Diagnosing it took three rounds of
        # investigation and ultimately required monkeypatching ``ReductionError.__init__`` to
        # dump a stack, because the message names a symptom that several distinct code paths
        # can produce. During an unattended multi-week run the operator gets exactly this text
        # and nothing else, so the traceback is not a debugging luxury -- it is the only
        # forensic record of why the run stopped.
        print(f"crawler error: {exc}", file=sys.stderr)
        print(
            f"exception_type: {type(exc).__module__}.{type(exc).__qualname__}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return EXIT_ERROR
    return EXIT_USAGE


def _add_intake_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add the shared required intake snapshot path."""

    parser.add_argument("--intake", type=Path, required=required, help="intake snapshot directory")


def _add_dry_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the opt-in fixture-corpus acceptance harness arguments."""

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the tiny-model acceptance corpus with deterministic agent lanes and a real prefix",
    )
    parser.add_argument(
        "--dry-run-root",
        type=Path,
        help="isolated dry-run campaign root (defaults under .crawl-local)",
    )
    parser.add_argument(
        "--dry-run-environment-prefix",
        type=Path,
        help="real materialized fixture prefix required by the acceptance dry-run",
    )


def _stable_id_csv(value: str) -> tuple[str, ...]:
    """Parse a comma-separated enumerated stable-ID target list."""

    targets = tuple(part.strip() for part in value.split(",") if part.strip())
    if not targets:
        raise argparse.ArgumentTypeError("target list must name at least one stable ID")
    return targets


def _resolved_redrive_selection(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], Optional[str], Optional[Path]]:
    """Resolve the D-54 re-drive triple (targets, reason, manifest path).

    Inline targets and a manifest are mutually exclusive target SOURCES: a
    manifest carries its own reason and target list, digest-verified at load
    (snapshot currency is re-verified by the driver at leg start against the
    ACTIVE intake). The config's cap/reason validation runs downstream either
    way.
    """

    from menagerie.crawler.redrive_manifest import load_redrive_manifest

    inline = tuple(getattr(args, "redrive_targets", ()) or ())
    reason = getattr(args, "redrive_reason", None)
    manifest_arg = getattr(args, "redrive_manifest", None)
    if manifest_arg is None:
        return inline, reason, None
    if inline:
        raise ValueError(
            "--redrive-manifest and inline --redrive-targets are mutually "
            "exclusive: one target authority per leg"
        )
    if reason is not None:
        raise ValueError(
            "--redrive-manifest carries its own reason; drop --redrive-reason"
        )
    manifest_path = Path(manifest_arg).resolve()
    manifest = load_redrive_manifest(manifest_path)
    return manifest.targets, manifest.reason, manifest_path


def _add_selection_scope_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the WS-1 closed selection-mode and disclosure arguments.

    One selection mode per leg: ordinary (default), ``--only-unsettled``,
    ``--redrive-targets``, or ``--gate-catchup-targets``; the driver config
    validates exclusivity. The continuation identity is machine-plumbed by
    the wake callback, never typed by an operator.
    """

    parser.add_argument(
        "--only-unsettled",
        action="store_true",
        help=(
            "schedule only intake members with no raw model-ledger head; "
            "never re-open settled heads -- current OR dependency-stale -- and "
            "never consume grants (absolute operator scope cut)"
        ),
    )
    parser.add_argument(
        "--recheck",
        action="store_true",
        help=(
            "RC-2 named re-check mode: admit exactly the recheck census's "
            "needs_recheck members (heads recorded before a measured soundness "
            "rotation, or an unmeasured move, on a policy axis), derived at leg "
            "start from the ledger; runs awards still need their AW-1 grant"
        ),
    )
    parser.add_argument(
        "--redrive-targets",
        type=_stable_id_csv,
        default=(),
        metavar="ID,ID,...",
        help=(
            "named re-drive mode: the enumerated members ARE the admission "
            "authority for non-runs raw heads; never reopens an award"
        ),
    )
    parser.add_argument(
        "--redrive-reason",
        metavar="TEXT",
        help=(
            "durable operator rationale for a named re-drive (D-54); REQUIRED "
            "with --redrive-targets, recorded on the leg config event"
        ),
    )
    parser.add_argument(
        "--redrive-manifest",
        type=Path,
        metavar="PATH",
        help=(
            "self-digested re-drive manifest bound to the intake snapshot "
            "(D-54); the ONLY legal target source above the inline cap. "
            "Mutually exclusive with inline --redrive-targets and "
            "--redrive-reason (the manifest carries both)"
        ),
    )
    parser.add_argument(
        "--gate-catchup-targets",
        type=_stable_id_csv,
        default=(),
        metavar="ID,ID,...",
        help=(
            "named gate catch-up mode: enumerated pending-metadata runs "
            "records re-enter for their catch-up gate round only"
        ),
    )
    parser.add_argument(
        "--leg-purpose",
        choices=sorted(LEG_PURPOSES),
        default=None,
        help=(
            "declare the leg's closed purpose (shape 4). licensing_close "
            "opens a ZERO-ADMISSION leg whose only lawful acts are the "
            "orphan-citation adoption pass and the close itself: the WS-1 "
            "plan seals empty, the terminal-review backlog sweep is skipped, "
            "every paid admission refuses typed, and the close refuses any "
            "forfeit violation. On a successor_rung snapshot it derives "
            "certificate_kind canary_admission instead of rung_launch — the "
            "declaration buys a strictly weaker leg, never a weaker gate. "
            "Immutable per generation: resuming an open generation under a "
            "different purpose refuses"
        ),
    )
    parser.add_argument(
        "--ack-stale-excluded",
        action="store_true",
        help=(
            "acknowledge that settled-but-dependency-stale heads exist and are "
            "EXCLUDED; an ordinary leg refuses to start over a nonzero stale "
            "census without this"
        ),
    )
    parser.add_argument("--continuation-identity", help=argparse.SUPPRESS)


def _add_target_phase_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared target and ordered phase arguments."""

    parser.add_argument("--target", choices=("osx-arm64", "linux-x86_64-cuda"), default="osx-arm64")
    parser.add_argument("--phase", choices=("pytorch", "native-tail"))


def _add_driver_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared checkpoint, milestone, and notification configuration."""

    parser.add_argument("--review-checkpoint-at", type=int, default=1000)
    parser.add_argument(
        "--progress-milestones",
        type=_positive_csv,
        default=(900, 950, 1000, 2000, 3000, 5000, 10000, 15000, 20000),
        metavar="N,N,...",
    )
    parser.add_argument("--notify-command")
    # D-22/DX-23 run_id axis rotation: no shared default. Omitted, a real
    # per-invocation identity is minted at launch; the retired constant
    # "crawler-run" is refused by DriverConfig (era rule: old rows stay
    # readable, new rows never reuse it).
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--environment-readiness-receipt",
        type=Path,
        help="sealed routed-closure receipt required by production run/resume",
    )
    parser.add_argument("--wake-episode-id", help=argparse.SUPPRESS)
    parser.add_argument("--campaign-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--refreeze-campaign-config",
        action="store_true",
        help=(
            "overwrite an existing frozen campaign config for this intake; "
            "without it a fresh run over a frozen campaign refuses, and an "
            "exercised refreeze is disclosed as an operational event"
        ),
    )
    parser.add_argument(
        "--author-command",
        default=os.environ.get("MENAGERIE_AUTHOR_COMMAND"),
        help="Claude Code wrapper command; defaults to MENAGERIE_AUTHOR_COMMAND",
    )
    parser.add_argument(
        "--author-queue",
        default=os.environ.get("MENAGERIE_AUTHOR_QUEUE"),
        help=(
            "author-queue root serviced by the managing Claude session; "
            "defaults to MENAGERIE_AUTHOR_QUEUE and replaces --author-command"
        ),
    )
    parser.add_argument(
        "--author-concurrency",
        type=int,
        # D-63 (PM-1): the default is a SENTINEL, not a computed value. An
        # eagerly computed default made an explicit flag indistinguishable
        # from "the operator passed nothing", which is exactly how the 200a
        # explicit-wins override went inert (width 18 configured, 12 ran).
        default=None,
        help=(
            "author sessions one wave may keep in flight; explicit-wins over "
            f"MENAGERIE_AUTHOR_CONCURRENCY, the frozen campaign config, and "
            f"the default {DEFAULT_AUTHOR_WAVE_CONCURRENCY}. Only the sessions "
            "overlap: every record is still written one at a time, in work "
            "order. Use 1 for the historical serial lane."
        ),
    )
    parser.add_argument(
        "--checker-concurrency",
        type=int,
        default=None,
        help=(
            "checker sessions one gate round may keep in flight; explicit-wins "
            f"over MENAGERIE_CHECKER_CONCURRENCY, the frozen campaign config, "
            f"and the default {DEFAULT_CHECKER_WAVE_CONCURRENCY}. Only the "
            "sessions overlap: gates are still normalized, routed, and "
            "appended one at a time, in ladder order. Use 1 for the historical "
            "serial checker lane."
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_MODES,
        default=_default_scheduler(),
        help=(
            "leg scheduling shape; defaults to MENAGERIE_SCHEDULER or 'waves'. "
            "'pipelined' is the per-model execute-first scheduler: models run "
            "their full mechanical execution matrix before any metadata or "
            "fidelity session is purchased, machine deaths terminalize gate-"
            "free, and gates batch across environment turns. Records are "
            "scheduler-agnostic; either mode resumes the other's ledgers."
        ),
    )
    parser.add_argument(
        "--author-wall-seconds",
        type=float,
        default=None,
        help=(
            "per-session author wall grant in seconds; defaults to this "
            "campaign's own grant (c3-classics runs the long classics tail). "
            "The value is frozen into the campaign config and published to the "
            "author executor, so one number governs the whole chain."
        ),
    )
    parser.add_argument(
        "--checker-command",
        default=os.environ.get("MENAGERIE_CHECKER_COMMAND"),
        help="Codex wrapper command; defaults to MENAGERIE_CHECKER_COMMAND",
    )
    parser.add_argument(
        "--provider-reconcile-command",
        default=os.environ.get("MENAGERIE_PROVIDER_RECONCILE_COMMAND"),
        help=(
            "provider status-query command; receives one paid-session idempotency key as "
            "its final argument and must print exactly "
            '{"idempotency_key", "accepted", "observed_wall_seconds"} as JSON. The key '
            "MUST be echoed back and is checked against the queried one, so the answer "
            "attests a specific reservation instead of attesting itself. "
            "observed_wall_seconds may be null when the provider does not meter wall "
            "clock: the driver's own measurement is the charge floor, so a null answer "
            "neither zeroes nor inflates the charge. accepted=false with a nonzero wall "
            "is representable and settles at zero sessions over the metered wall."
        ),
    )
    parser.add_argument(
        "--environment-command",
        default=os.environ.get("MENAGERIE_ENVIRONMENT_COMMAND"),
        help="exact-lock lifecycle wrapper; defaults to MENAGERIE_ENVIRONMENT_COMMAND",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        help=(
            "explicit temporary staging root for the environment lane; the same "
            "path must be supplied to launch_preflight"
        ),
    )


def _doctor_command(args: argparse.Namespace, probes: Optional[DoctorProbes]) -> int:
    """Run and print the strict doctor report."""

    intake_items = args.intake_items
    if intake_items is None and args.intake is not None:
        manifest = json.loads((args.intake / "manifest.json").read_text(encoding="utf-8"))
        intake_items = int(manifest["item_count"])
    config = DoctorConfig(
        repo_root=args.repo_root.resolve(),
        runtime_root=(args.repo_root / ".crawl-local").resolve(),
        target=args.target,
        minimum_disk_bytes=args.minimum_disk_gib * 1024**3,
        strict=args.strict,
        intake_item_count=intake_items,
    )
    report = run_doctor(config, probes)
    print(json.dumps({"passed": report.passed, "checks": report.checks}, sort_keys=True))
    return EXIT_OK


def _intake_command(args: argparse.Namespace) -> int:
    """Create an immutable intake snapshot from explicit or conventional inputs."""

    data_root = args.repo_root / "menagerie" / "data"
    if args.master is not None:
        master = args.master
    elif args.all_existing:
        # Prefer the full crawl roster (every found model: built + classics +
        # to-build) when it is present; fall back to the built-only catalog.
        roster = data_root / "crawl_roster.jsonl"
        master = roster if roster.exists() else data_root / "master_catalog.jsonl"
    else:
        master = None
    deferred = args.deferred or (data_root / "deferred.jsonl" if args.all_existing else None)
    if master is None or deferred is None:
        raise ValueError("intake requires --master/--deferred or --all-existing")
    output = args.output_root or args.repo_root / "menagerie" / "crawler" / "records" / "intake"
    stable_ids = args.stable_ids or data_root / "stable_ids.jsonl"
    snapshot = create_intake_snapshot(master, deferred, output, stable_ids_path=stable_ids)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot.snapshot_id,
                "items": len(snapshot.items),
                "created": snapshot.created,
                "path": str(snapshot.root),
            },
            sort_keys=True,
        )
    )
    return EXIT_OK


def _partition_campaigns_command(args: argparse.Namespace) -> int:
    """Emit the four immutable tier-aligned campaign inputs."""

    data_root = args.repo_root / "menagerie" / "data"
    records_root = args.repo_root / "menagerie" / "crawler" / "records"
    bindings = emit_campaign_partitions(
        args.roster or data_root / "crawl_roster.jsonl",
        args.output_root or records_root,
        stable_ids_path=args.stable_ids or data_root / "stable_ids.jsonl",
    )
    print(
        json.dumps(
            {
                "campaigns": {
                    binding.spec.campaign_id: {
                        "author_model": binding.spec.author_model,
                        "checker_model": binding.spec.checker_model,
                        "items": binding.row_count,
                        "intake": binding.intake_path,
                    }
                    for binding in bindings
                }
            },
            sort_keys=True,
        )
    )
    return EXIT_OK


def _plan_command(args: argparse.Namespace) -> int:
    """Print deterministic phase-first environment assignments."""

    snapshot = load_intake_snapshot(args.intake)
    registry = load_environment_registry(target=args.target)
    routes = phase_routes(
        route_model(item.to_model_requirements(framework_from_markers(item.zoo, item.name)))
        for item in snapshot.items
    )
    if args.phase is not None:
        routes = tuple(route for route in routes if route.phase.value == args.phase)
    payload = {
        "target": args.target,
        "phase_order": [phase.value for phase in registry.phase_order],
        "assignments": [
            {"stable_id": route.stable_id, "intent": route.intent, "phase": route.phase.value}
            for route in routes
        ],
    }
    print(json.dumps(payload, sort_keys=True))
    return EXIT_OK


def _routed_intake_closure(snapshot: IntakeSnapshot) -> tuple[str, ...]:
    """Return the deduplicated environment intents reached by an intake."""

    return tuple(
        sorted(
            {
                route_model(
                    item.to_model_requirements(framework_from_markers(item.zoo, item.name))
                ).intent
                for item in snapshot.items
            }
        )
    )


def _seal_environment_readiness_command(args: argparse.Namespace) -> int:
    """Write one canonical CAS-readiness receipt for a frozen intake."""

    snapshot = load_intake_snapshot(args.intake)
    receipt = build_environment_readiness_receipt(
        args.envs_root,
        args.state_root,
        target=args.target,
        intake_sha256=snapshot.snapshot_sha256,
        routed_intents=_routed_intake_closure(snapshot),
    )
    atomic_replace_bytes(args.output.resolve(), canonical_json_bytes(receipt) + b"\n")
    print(json.dumps(receipt, sort_keys=True))
    return EXIT_OK


def _add_census_ack_arguments(parser: argparse.ArgumentParser) -> None:
    """D-47: the digest companions binding census acknowledgments."""

    parser.add_argument(
        "--award-holds-digest",
        metavar="SHA256",
        default=None,
        help=(
            "the award-holds census digest this acknowledgment binds (from the "
            "award-holds-census event / the refusal message); required beside "
            "--acknowledge-award-holds whenever the census is nonempty"
        ),
    )
    parser.add_argument(
        "--stale-excluded-digest",
        metavar="SHA256",
        default=None,
        help=(
            "the stale-excluded census digest this acknowledgment binds; "
            "required beside --ack-stale-excluded whenever the census is "
            "nonempty"
        ),
    )


def _require_environment_readiness_receipt(args: argparse.Namespace) -> None:
    """Refuse production run/resume without an exact current sealed receipt."""

    path = getattr(args, "environment_readiness_receipt", None)
    if not isinstance(path, Path):
        raise ValueError(
            "environment-readiness-required: production run/resume requires "
            "--environment-readiness-receipt"
        )
    try:
        receipt = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("environment-readiness-invalid: receipt is unreadable") from exc
    if not isinstance(receipt, dict):
        raise ValueError("environment-readiness-invalid: receipt must be an object")
    state_root_value = receipt.get("state_root")
    if not isinstance(state_root_value, str) or not Path(state_root_value).is_absolute():
        raise ValueError("environment-readiness-invalid: receipt state root is invalid")
    configured = os.environ.get("MENAGERIE_ENVIRONMENT_STATE_ROOT")
    if configured is None or Path(configured).resolve() != Path(state_root_value).resolve():
        raise ValueError(
            "environment-readiness-invalid: MENAGERIE_ENVIRONMENT_STATE_ROOT does not "
            "name the sealed CAS root"
        )
    snapshot = load_intake_snapshot(args.intake)
    expected = build_environment_readiness_receipt(
        DEFAULT_ENVS_ROOT,
        Path(state_root_value),
        target=args.target,
        intake_sha256=snapshot.snapshot_sha256,
        routed_intents=_routed_intake_closure(snapshot),
    )
    if receipt != expected:
        raise ValueError(
            "environment-readiness-stale: receipt does not match the current routed closure"
        )


def _crawl_command(args: argparse.Namespace, factory: Optional[DriverFactory]) -> int:
    """Run one continuous-crawl invocation (overhaul RC-1, RULING D.1).

    The campaign binds from ``<records-root>/campaigns.json``; the active intake
    is its full partition snapshot. Everything a leg needed the operator to type
    (intake path, review checkpoint, scheduler, selection scope) is derived
    here; the driver's in-process preflight owns every refusal after the lock.
    """

    from menagerie.crawler.rung_staging import (  # noqa: PLC0415 -- verified-root resolver (frozen manifest read)
        RungStagingError,
        resolve_verified_root,
    )

    repo_root = args.repo_root.resolve()
    records_root = args.records_root.resolve()
    canonical = (repo_root / "menagerie" / "crawler" / "records").resolve()
    if factory is None and records_root != canonical:
        # The production factory binds through the repository's own manifest
        # and ledger layout; an injected factory (tests, the fake-provider
        # acceptance rig) owns its paths and may crawl any records root that
        # carries a campaigns.json.
        raise OperatorOutageError(
            f"crawl requires --records-root to be the repository's canonical records "
            f"directory {canonical} (got {records_root}); run from the campaign clone"
        )
    if not (records_root / "campaigns.json").is_file():
        raise OperatorOutageError(f"campaign-binding: no campaigns.json under {records_root}")
    try:
        binding, _snapshot = resolve_verified_root(records_root, args.campaign)
    except RungStagingError as exc:
        raise OperatorOutageError(f"campaign-binding: {exc}") from exc
    # The leg-era flags the shared factory reads, derived for the crawl shape.
    args.intake = records_root / binding.intake_path
    args.phase = None
    args.resume = False
    args.dry_run = False
    args.dry_run_root = None
    args.dry_run_environment_prefix = None
    args.scheduler = "pipelined"
    args.review_checkpoint_at = binding.spec.review_checkpoint_at
    args.only_status = None
    args.only_unsettled = False
    args.redrive_targets = ()
    args.redrive_reason = None
    args.redrive_manifest = None
    args.gate_catchup_targets = ()
    args.leg_purpose = None
    args.acknowledge_award_holds = False
    args.award_holds_digest = None
    args.ack_stale_excluded = False
    args.stale_excluded_digest = None
    args.continuation_identity = None
    args.acknowledge_repeat_cause = None
    args.preflight_verdict = None
    if getattr(args, "author_width", None) is not None:
        if getattr(args, "author_concurrency", None) is not None:
            raise OperatorOutageError("pass --author-width or --author-concurrency, not both")
        args.author_concurrency = int(args.author_width)
    if factory is None:
        # Production: the same LB-2 environment-readiness gate run/resume pass
        # (restore the frozen launch contract first, then check the receipt).
        deferred_outage: Optional[OperatorOutageError] = None
        try:
            _load_or_persist_campaign_config(args)
        except OperatorOutageError as outage:
            deferred_outage = outage
        _require_environment_readiness_receipt(args)
        if deferred_outage is not None:
            raise deferred_outage
    base_factory = factory or _default_driver_factory

    def crawl_factory(namespace: argparse.Namespace) -> CrawlerDriver:
        driver = base_factory(namespace)
        driver.config = replace(
            driver.config,
            invocation_shape="crawl",
            scheduler="pipelined",
            crawl_limit=namespace.limit,
            crawl_order_seed=(
                int(namespace.order_seed)
                if getattr(namespace, "order_seed", None) is not None
                else driver.config.crawl_order_seed
            ),
        )
        return driver

    return _driver_command(args, crawl_factory)


def _driver_command(args: argparse.Namespace, factory: DriverFactory) -> int:
    """Run a fully configured driver and map pauses to a stable exit code."""

    driver = factory(args)
    after_review = bool(getattr(args, "after_review", False))
    raw_episode_id = getattr(args, "wake_episode_id", None)
    episode_id = raw_episode_id if isinstance(raw_episode_id, str) else None
    restart_ordinal = getattr(args, "supervised_restart_ordinal", None)
    origin = (
        # W6: the ladder's restart is its own continuation authority -- the
        # supervisor drops the consumed wake episode on relaunch, so the two
        # authorities can never both continue one generation.
        InvocationOrigin.SUPERVISED_RESTART
        if restart_ordinal is not None
        else InvocationOrigin.WAKE_CALLBACK
        if episode_id is not None
        else InvocationOrigin.MANUAL_RESUME
        if args.command == "resume" or bool(getattr(args, "resume", False))
        else InvocationOrigin.ORDINARY_RUN
    )
    driver.config = replace(
        driver.config,
        invocation_origin=origin,
        wake_episode_id=episode_id if origin is InvocationOrigin.WAKE_CALLBACK else None,
        invocation_ordinal=restart_ordinal,
    )
    try:
        result: DriverResult = driver.run(after_review=after_review)
    except DriverLockError:
        if origin is not InvocationOrigin.WAKE_CALLBACK:
            raise
        if not isinstance(episode_id, str):
            raise AssertionError("validated wake episode identity was lost")
        fired_at = utc_now()
        _record_scheduled_wake_intent(driver, episode_id, fired_at)
        print(
            json.dumps(
                {
                    "status": OperationalEventStatus.WAKE_NOOP_ALREADY_RUNNING.value,
                    "episode_id": episode_id,
                },
                sort_keys=True,
            )
        )
        return EXIT_OK
    output: dict[str, object] = dict(result.__dict__)
    if result.shutdown_interruption is not None:
        output["shutdown_interruption"] = asdict(result.shutdown_interruption)
    if bool(getattr(args, "dry_run", False)):
        snapshot = load_intake_snapshot(driver.paths.intake_root)
        attempts = scan_jsonl(driver.paths.ledgers.attempts)
        context = _snapshot_authority_context(snapshot, driver.config)
        context = replace(
            context,
            environment_generations=_persisted_environment_generations(attempts),
        )
        current = materialize_current(
            driver.paths.ledgers,
            context=context,
        )
        acceptance = _dry_run_acceptance(
            snapshot,
            current,
            attempts,
            # A guarded wake-noop leaves the campaign paused at a resumable
            # point (the episode's not_before), so the acceptance gate grades
            # it pending, never acceptance-failed.
            paused=result.paused_reason is not None or result.status == "wake-noop",
        )
        output.update(
            {
                "dry_run": True,
                "funnel": funnel_counts(current),
                "acceptance": acceptance,
            }
        )
    print(json.dumps(output, sort_keys=True))
    if result.paused_reason is not None:
        if result.paused_reason == "review-checkpoint":
            return EXIT_REVIEW_PAUSED
        return EXIT_PAUSED
    if bool(getattr(args, "dry_run", False)) and acceptance["status"] != "passed":
        # Pending means the campaign is intentionally stopped at a resumable
        # point (a guarded wake-noop; an ordinary pause already returned
        # EXIT_PAUSED above); only a graded acceptance failure exits EXIT_ERROR.
        if acceptance["status"] == "pending":
            return EXIT_PAUSED
        return EXIT_ERROR
    return EXIT_OK


def _record_scheduled_wake_intent(driver: CrawlerDriver, episode_id: str, fired_at: str) -> None:
    """Fsync one idempotent fire intent when the live driver owns its lock.

    Parameters
    ----------
    driver:
        Lock-contended driver exposing canonical and gitignored runtime paths.
    episode_id, fired_at:
        Exact frozen episode identity and actual callback timestamp.
    """

    path = canonical_operational_ledger_path(driver.paths.ledgers.models)
    projection = reduce_wake_episodes(scan_jsonl(path))
    try:
        episode = projection.episodes[episode_id].episode
    except KeyError as exc:
        raise ValueError(f"scheduled callback names unknown wake episode: {episode_id}") from exc
    write_fire_intent(driver.paths.wakeup_root, episode, fired_at=fired_at)


def _default_driver_factory(args: argparse.Namespace) -> CrawlerDriver:
    """Build production command lanes from explicit flags or supervisor environment."""

    if bool(getattr(args, "dry_run", False)):
        from menagerie.crawler.tests.dry_run_support import build_dry_run_driver

        dry_run_root = args.dry_run_root or args.repo_root / ".crawl-local" / "dry-run"
        environment_prefix = getattr(args, "dry_run_environment_prefix", None)
        if not isinstance(environment_prefix, Path):
            if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
                raise ValueError("unmet-release-gate: --dry-run-environment-prefix is required")
            raise ValueError("dry-run requires --dry-run-environment-prefix")
        return build_dry_run_driver(
            args.repo_root.resolve(),
            dry_run_root.resolve(),
            environment_prefix.resolve(),
            review_checkpoint_at=args.review_checkpoint_at,
            progress_milestones=tuple(args.progress_milestones),
            run_id=args.run_id or mint_default_run_id(),
            # The acceptance dry-run is the only end-to-end path with a real
            # worker and a real environment binder, so the selected scheduler
            # must reach it: a silently ignored --scheduler leaves the pipelined
            # leg with no rehearsal vehicle at all.
            scheduler=getattr(args, "scheduler", "waves"),
        )
    if getattr(args, "dry_run_environment_prefix", None) is not None:
        raise ValueError("--dry-run-environment-prefix requires --dry-run")
    campaign_config_path, launch_env_divergences = _load_or_persist_campaign_config(args)
    if args.intake is None:
        raise ValueError("run/resume requires --intake unless --dry-run is selected")
    author_queue_root = _optional_author_queue_root(args)
    author_command = (
        None if author_queue_root is not None else _required_command(args.author_command, "author")
    )
    checker_command = _required_command(args.checker_command, "checker")
    provider_reconcile_command = _required_command(
        args.provider_reconcile_command, "provider-reconcile"
    )
    environment_command = _required_command(args.environment_command, "environment")
    snapshot = load_intake_snapshot(args.intake.resolve())
    paths = _driver_paths_for_snapshot(args.repo_root.resolve(), snapshot)
    campaign_binding = _required_campaign_binding(args.repo_root.resolve(), snapshot)
    _validate_campaign_launch_policy(args, campaign_binding)
    redrive_targets, redrive_reason, redrive_manifest_path = _resolved_redrive_selection(args)
    config = DriverConfig(
        target=args.target,
        phase=args.phase,
        run_id=args.run_id or mint_default_run_id(),
        review_checkpoint_at=args.review_checkpoint_at,
        progress_milestones=tuple(args.progress_milestones),
        notify_command=args.notify_command,
        author_model=campaign_binding.spec.author_model,
        checker_model=campaign_binding.spec.checker_model,
        only_status=getattr(args, "only_status", None),
        only_unsettled=bool(getattr(args, "only_unsettled", False)),
        recheck=bool(getattr(args, "recheck", False)),
        leg_purpose=getattr(args, "leg_purpose", None),
        acknowledge_award_holds=bool(getattr(args, "acknowledge_award_holds", False)),
        award_holds_digest=getattr(args, "award_holds_digest", None),
        stale_excluded_digest=getattr(args, "stale_excluded_digest", None),
        redrive_targets=redrive_targets,
        redrive_reason=redrive_reason,
        redrive_manifest_path=redrive_manifest_path,
        gate_catchup_targets=tuple(getattr(args, "gate_catchup_targets", ()) or ()),
        ack_stale_excluded=bool(getattr(args, "ack_stale_excluded", False)),
        wake_continuation_identity=getattr(args, "continuation_identity", None),
        environment_readiness_receipt_path=getattr(
            args, "environment_readiness_receipt", None
        ),
        campaign_config_path=campaign_config_path,
        campaign_id=campaign_binding.spec.campaign_id,
        author_queue_root=author_queue_root,
        author_concurrency=args.author_concurrency,
        checker_concurrency=args.checker_concurrency,
        # D-63: which channel decided each width rides into the leg-config
        # attestation, so an override that applied says so durably.
        author_concurrency_source=getattr(args, "concurrency_sources", {}).get(
            "author_concurrency", "default"
        ),
        checker_concurrency_source=getattr(args, "concurrency_sources", {}).get(
            "checker_concurrency", "default"
        ),
        scheduler=getattr(args, "scheduler", "waves"),
        invocation_origin=(
            InvocationOrigin.SUPERVISED_RESTART
            if getattr(args, "supervised_restart_ordinal", None) is not None
            else InvocationOrigin.WAKE_CALLBACK
            if getattr(args, "wake_episode_id", None) is not None
            else InvocationOrigin.MANUAL_RESUME
            if args.command == "resume" or bool(getattr(args, "resume", False))
            else InvocationOrigin.ORDINARY_RUN
        ),
        wake_episode_id=(
            str(args.wake_episode_id)
            if getattr(args, "wake_episode_id", None) is not None
            and getattr(args, "supervised_restart_ordinal", None) is None
            else None
        ),
        invocation_ordinal=getattr(args, "supervised_restart_ordinal", None),
        acknowledge_repeat_cause=getattr(args, "acknowledge_repeat_cause", None),
    )
    _disclose_launch_environment_divergence(paths, config, launch_env_divergences)
    # Resolved once, before any lane exists, so a misconfigured budget refuses at
    # startup rather than surfacing as truncated sessions deep into the campaign.
    author_effort_grant = AuthorEffortGrant(
        wall_seconds=_resolve_author_wall(args, config.campaign_id)
    )
    dependencies = DriverDependencies(
        author=(
            QueueAuthorLane(author_queue_root, effort_grant=author_effort_grant)
            if author_queue_root is not None
            else CommandAuthorLane(
                author_command or (),
                effort_grant=author_effort_grant,
                # The headless author executor prints an attempt-bound
                # publication receipt; requiring it makes the lane verify the
                # published digest on every round trip. Legacy wrappers leave
                # this unset.
                require_receipt=(os.environ.get("MENAGERIE_AUTHOR_REQUIRE_RECEIPT") == "1"),
            )
        ),
        checker=CommandCheckerLane(checker_command),
        forward=SupervisedForwardLane(cwd=args.repo_root.resolve()),
        environments=build_command_environment_lane(
            environment_command,
            paths.runtime_root,
            tmp_root=getattr(args, "tmp_root", None),
            # BL-9b: durable env-turn admission -- attempt/success/clearance
            # outcomes persist beside the canonical ledgers and are rederived
            # on resume, so a restart cannot re-permit env-build churn.
            records_root=paths.ledgers.models.parent.parent,
        ),
        notifier=CommandNotifier(
            config.notify_command,
            receipt_root=paths.runtime_root / "notification-receipts",
        ),
        clock=utc_now,
        provider_reconciler=CommandProviderReconciler(provider_reconcile_command),
    )
    return CrawlerDriver(paths, config, dependencies)


#: Where the memoized resolution rides between the pre-pass and the factory.
_CAMPAIGN_CONFIG_RESOLUTION_ATTRIBUTE = "_campaign_config_resolution"

#: Record kind distinguishing a refreeze disclosure from an env override.
_CAMPAIGN_CONFIG_REFREEZE_KIND = "campaign-config-refreeze"


def _load_or_persist_campaign_config(
    args: argparse.Namespace,
) -> tuple[Path, tuple[JsonObject, ...]]:
    """Load persisted resume inputs or freeze one ordinary campaign invocation.

    MEMOIZED per namespace, because it is now called twice: once as the launch
    pre-pass (so the restored environment is in place before the
    environment-readiness receipt reads ``MENAGERIE_ENVIRONMENT_STATE_ROOT``)
    and once from the driver factory. Without the memo the second call would
    observe the environment IT had already restored, see no disagreement, and
    silently drop the divergence disclosure -- an override that stops being
    disclosed the moment the code that discloses it runs twice.

    Parameters
    ----------
    args:
        Parsed run, resume, or handoff arguments.

    Returns
    -------
    tuple[pathlib.Path, tuple[JsonObject, ...]]
        The exact mode-0600 campaign config wake callbacks use, plus any
        launch-environment divergence records awaiting disclosure.
    """

    memo = getattr(args, _CAMPAIGN_CONFIG_RESOLUTION_ATTRIBUTE, None)
    if memo is not None:
        return memo
    resolution = _resolve_campaign_config(args)
    setattr(args, _CAMPAIGN_CONFIG_RESOLUTION_ATTRIBUTE, resolution)
    return resolution


def _resolve_campaign_config(
    args: argparse.Namespace,
) -> tuple[Path, tuple[JsonObject, ...]]:
    """Do the one-time campaign config load-or-freeze and environment restore.

    Parameters
    ----------
    args:
        Parsed run, resume, or handoff arguments.

    Returns
    -------
    tuple[pathlib.Path, tuple[JsonObject, ...]]
        Config path and launch-environment divergence records.
    """

    repo_root = args.repo_root.resolve()
    requested = getattr(args, "campaign_config", None)
    config_path = (
        requested.resolve()
        if isinstance(requested, Path)
        else _frozen_crawl_config_path(repo_root, args)
    )
    if config_path is None and args.command == "resume":
        if isinstance(args.intake, Path):
            candidate = default_campaign_config_path(repo_root, args.intake.resolve())
            config_path = candidate if candidate.is_file() else None
        else:
            candidates = tuple(
                sorted((repo_root / ".crawl-local" / "campaign-configs").glob("*.json"))
            )
            if len(candidates) == 1:
                config_path = candidates[0]
            elif not candidates:
                raise OperatorOutageError("resume requires --intake or --campaign-config")
            else:
                raise OperatorOutageError("resume is ambiguous; pass --intake or --campaign-config")
    if config_path is not None:
        persisted = load_campaign_config(config_path)
        if persisted.repo_root != repo_root:
            raise ValueError("campaign config belongs to a different repository root")
        return config_path, _apply_campaign_config(args, persisted)

    if not isinstance(args.intake, Path):
        raise OperatorOutageError("campaign launch requires --intake")
    intake_root = args.intake.resolve()
    if args.run_id is None:
        # A fresh launch with no explicit identity mints its own: the frozen
        # campaign config below is what wake callbacks and resumes replay, so
        # the minted identity is durable, never re-minted per resume.
        args.run_id = mint_default_run_id()
    # D-63: a fresh launch resolves the widths from its explicit inputs
    # (flag > env > default) BEFORE freezing, so the frozen contract records
    # what the operator actually asked for.
    _resolve_effective_concurrency(args, None)
    try:
        author = resolve_command(_required_command(args.author_command, "author"), cwd=repo_root)
        checker = resolve_command(_required_command(args.checker_command, "checker"), cwd=repo_root)
        environment = resolve_command(
            _required_command(args.environment_command, "environment"), cwd=repo_root
        )
        provider_reconcile = resolve_command(
            _required_command(args.provider_reconcile_command, "provider-reconcile"),
            cwd=repo_root,
        )
        notifier = CommandNotifier(args.notify_command).command
        resolved_notifier = (
            resolve_command(notifier, cwd=repo_root) if notifier is not None else None
        )
        public_mirror = _required_absolute_environment_path("MENAGERIE_PUBLIC_MIRROR")
        private_mirror = _required_absolute_environment_path("MENAGERIE_PRIVATE_MIRROR")
    except ValueError as exc:
        raise OperatorOutageError(str(exc)) from exc
    persisted = CampaignConfig(
        repo_root=repo_root,
        intake_root=intake_root,
        target=args.target,
        run_id=args.run_id,
        author_queue_root=_optional_author_queue_root(args),
        author_command=author,
        checker_command=checker,
        environment_command=environment,
        notify_command=resolved_notifier,
        public_mirror=public_mirror,
        private_mirror=private_mirror,
        review_checkpoint_at=args.review_checkpoint_at,
        progress_milestones=tuple(args.progress_milestones),
        phase=args.phase,
        only_status=getattr(args, "only_status", None),
        author_wall_seconds=getattr(args, "author_wall_seconds", None),
        author_concurrency=args.author_concurrency,
        checker_concurrency=args.checker_concurrency,
        provider_reconcile_command=provider_reconcile,
        # Frozen from the LAUNCH's own resolved inputs, for the same reason the
        # lane commands are: a launchd wake inherits a bare environment, and
        # `launchctl setenv` keeps the campaign's correctness in per-user
        # machine state that no ledger records and a reboot erases.
        launch_environment=_launch_environment_from_args(args),
    )
    config_path = default_campaign_config_path(repo_root, intake_root)
    refreeze_records: tuple[JsonObject, ...] = ()
    if config_path.is_file():
        # The pilot silently re-froze its campaign from the ambient interactive
        # shell four times; an existing frozen campaign is resumed, never
        # silently replaced.
        if not getattr(args, "refreeze_campaign_config", False):
            raise OperatorOutageError(
                f"a frozen campaign config already exists at {config_path}; "
                "resume consumes it, and replacing it requires an explicit "
                "--refreeze-campaign-config"
            )
        superseded_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        write_campaign_config(config_path, persisted)
        refreeze_records = (
            {
                "kind": _CAMPAIGN_CONFIG_REFREEZE_KIND,
                "config_path": str(config_path),
                "superseded_sha256": superseded_sha256,
                "refrozen_sha256": hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest(),
            },
        )
    else:
        write_campaign_config(config_path, persisted)
    # A fresh launch cannot diverge from itself: the contract was just frozen
    # FROM this environment, so every comparison agrees by construction.
    return config_path, refreeze_records + _apply_campaign_config(args, persisted)


def _frozen_crawl_config_path(repo_root: Path, args: argparse.Namespace) -> Optional[Path]:
    """Return the config a repeat ``crawl`` invocation reuses, if one was frozen.

    RC-1: every crawl invocation over a campaign root reuses the config frozen
    at its first launch (a fresh run over a frozen campaign would otherwise
    refuse); the first launch returns ``None`` and freezes it.
    """

    if args.command != "crawl" or not isinstance(args.intake, Path):
        return None
    candidate = default_campaign_config_path(repo_root, args.intake.resolve())
    return candidate if candidate.is_file() else None


def _launch_environment_from_args(args: argparse.Namespace) -> LaunchEnvironment:
    """Freeze this launch's environment contract from its resolved inputs.

    Parameters
    ----------
    args:
        Parsed launch arguments, with the ambient environment still intact.

    Returns
    -------
    LaunchEnvironment
        The contract a later bare-environment resume rebuilds from.
    """

    tmp_root = getattr(args, "tmp_root", None)
    state_root = os.environ.get("MENAGERIE_ENVIRONMENT_STATE_ROOT")
    author_model = os.environ.get("MENAGERIE_AUTHOR_MODEL")
    return LaunchEnvironment(
        environment_state_root=(
            _absolute_launch_path(state_root, "MENAGERIE_ENVIRONMENT_STATE_ROOT")
            if state_root
            else None
        ),
        tmp_root=Path(tmp_root).resolve() if tmp_root is not None else None,
        author_model=author_model or None,
        require_author_receipt=os.environ.get("MENAGERIE_AUTHOR_REQUIRE_RECEIPT") == "1",
    )


def _absolute_launch_path(value: str, name: str) -> Path:
    """Resolve one launch-environment path, refusing a relative spelling.

    A relative path is meaningless to a resumed process whose working directory
    is launchd's, so it refuses AT LAUNCH -- where the operator can fix it --
    rather than at the wake, hours later and unattended.
    """

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise OperatorOutageError(f"{name} must be an absolute path, not {value!r}")
    return candidate.resolve()


#: The CLOSED set of environment names a resume restores from the frozen
#: campaign config. Enumerated rather than computed: the whole point is that a
#: launchd wake inherits a bare environment, so "which names matter" cannot be
#: discovered from the environment itself. Sorted, and asserted in
#: ``test_campaign_launch_environment.py``.
LAUNCH_ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "MENAGERIE_AUTHOR_CONCURRENCY",
    "MENAGERIE_AUTHOR_MODEL",
    "MENAGERIE_AUTHOR_REQUIRE_RECEIPT",
    "MENAGERIE_CHECKER_CONCURRENCY",
    "MENAGERIE_ENVIRONMENT_STATE_ROOT",
    "MENAGERIE_PROVIDER_RECONCILE_COMMAND",
)

#: The override channel for the environment lane's staging root is its flag, so
#: it is named here rather than in the environment vocabulary above.
_TMP_ROOT_OVERRIDE_NAME = "--tmp-root"

#: D-63 (PM-1): the CLOSED provenance vocabulary for the two lane widths.
#: Every leg-config attestation stamps one of exactly these per width, so
#: "which channel decided this leg's width" is a durable ledger fact instead
#: of an operator's shell history.
CONCURRENCY_SOURCES: tuple[str, ...] = (
    "explicit-flag",
    "explicit-env",
    "frozen-config",
    "default",
)

def _explicit_env_concurrency(name: str, raw: Optional[str]) -> Optional[int]:
    """Parse one explicit environment width, ``None`` when absent.

    A malformed value is rejected rather than silently ignored: quietly
    falling back would hide a misconfigured campaign's real throughput bound
    for a month. The read itself happens at the caller with the literal name,
    so the env-read census can classify it.
    """

    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, not {raw!r}") from exc


def _resolve_effective_concurrency(
    args: argparse.Namespace, config: Optional[CampaignConfig]
) -> tuple[JsonObject, ...]:
    """Resolve both lane widths EXPLICIT-WINS and stamp their provenance.

    Precedence per width: explicit flag > explicit environment > frozen
    campaign config > code default. The 200a bug (risk register R6) was the
    opposite order in disguise: ``_apply_campaign_config`` clobbered the
    parsed flag with the frozen value before the explicit-wins restore ran,
    and the environment name's only consumer was the argparse default that
    had already been evaluated -- so ``--author-concurrency 18`` ran 12 for
    legs 23-43 while the divergence disclosure claimed the override was in
    effect.

    The resolution MUST run before :func:`_restore_launch_environment` writes
    frozen values into the environment, so an inherited operator export is
    still distinguishable from the restore's own writes.

    Parameters
    ----------
    args:
        Mutable parser namespace; ``author_concurrency``/``checker_concurrency``
        are ``None`` unless the operator passed the flag.
    config:
        The frozen campaign config on a resume, ``None`` on a fresh launch
        (nothing frozen exists yet -- the resolved value is what freezes).

    Returns
    -------
    tuple[JsonObject, ...]
        One flag-channel divergence record per width where an explicit flag
        disagreed with the frozen value, in the ``--tmp-root`` disclosure
        shape. The applied value IS the disclosed explicit value.
    """

    # Literal reads, so the env-read census (WS3 Wave-1 tripwire) can see and
    # classify exactly which names steer the widths.
    inherited: dict[str, Optional[str]] = {
        "MENAGERIE_AUTHOR_CONCURRENCY": os.environ.get("MENAGERIE_AUTHOR_CONCURRENCY"),
        "MENAGERIE_CHECKER_CONCURRENCY": os.environ.get("MENAGERIE_CHECKER_CONCURRENCY"),
    }
    divergences: list[JsonObject] = []

    def _resolve_one(
        explicit_flag: Optional[int],
        explicit_env: Optional[int],
        frozen: Optional[int],
        code_default: int,
        flag: str,
    ) -> tuple[int, str]:
        if explicit_flag is not None:
            value = int(explicit_flag)
            if frozen is not None and value != int(frozen):
                divergences.append(
                    {"name": flag, "persisted": str(frozen), "explicit": str(value)}
                )
            return value, "explicit-flag"
        if explicit_env is not None:
            # The env-channel divergence is disclosed by the launch-environment
            # restore itself, which compares this same inherited value against
            # the frozen contract; recording it twice would double-count one
            # operator act.
            return explicit_env, "explicit-env"
        if frozen is not None:
            return int(frozen), "frozen-config"
        return code_default, "default"

    # Static mutation targets only: the composition anti-substitution
    # tripwire (rightly) refuses a dynamic setattr on the parsed namespace.
    args.author_concurrency, author_source = _resolve_one(
        getattr(args, "author_concurrency", None),
        _explicit_env_concurrency(
            "MENAGERIE_AUTHOR_CONCURRENCY", inherited["MENAGERIE_AUTHOR_CONCURRENCY"]
        ),
        config.author_concurrency if config is not None else None,
        DEFAULT_AUTHOR_WAVE_CONCURRENCY,
        "--author-concurrency",
    )
    args.checker_concurrency, checker_source = _resolve_one(
        getattr(args, "checker_concurrency", None),
        _explicit_env_concurrency(
            "MENAGERIE_CHECKER_CONCURRENCY", inherited["MENAGERIE_CHECKER_CONCURRENCY"]
        ),
        config.checker_concurrency if config is not None else None,
        DEFAULT_CHECKER_WAVE_CONCURRENCY,
        "--checker-concurrency",
    )
    args.concurrency_sources = {
        "author_concurrency": author_source,
        "checker_concurrency": checker_source,
    }
    return tuple(divergences)


def _restore_launch_environment(
    args: argparse.Namespace, config: CampaignConfig
) -> tuple[JsonObject, ...]:
    """Rebuild the launch environment contract, letting an explicit value win.

    Precedence is EXPLICIT-WINS in both directions of the contract: whatever the
    operator actually passed at resume time stays, and the persisted value fills
    only what the wake environment lost. The alternative -- persisted-wins --
    would make ``--tmp-root`` and a hand-exported model silently inert on every
    resume, which is the same "your flag did nothing" failure the frozen lane
    commands exist to prevent.

    A disagreement is never absorbed silently: each one is returned for
    disclosure in an operational event, so a campaign that ran under an override
    says so in its own ledger instead of only in an operator's shell history.

    Parameters
    ----------
    args, config:
        Mutable parser namespace and validated private configuration.

    Returns
    -------
    tuple[JsonObject, ...]
        One record per name where an explicit resume-time value disagreed with
        the persisted one. Empty when nothing diverged.
    """

    contract = config.launch_environment
    # Derived from the first-class config fields, never from a second persisted
    # copy of the same value.
    persisted: dict[str, Optional[str]] = {
        "MENAGERIE_AUTHOR_CONCURRENCY": str(config.author_concurrency),
        "MENAGERIE_CHECKER_CONCURRENCY": str(config.checker_concurrency),
        "MENAGERIE_PROVIDER_RECONCILE_COMMAND": (
            shlex.join(config.provider_reconcile_command)
            if config.provider_reconcile_command
            else None
        ),
        "MENAGERIE_ENVIRONMENT_STATE_ROOT": (
            str(contract.environment_state_root)
            if contract.environment_state_root is not None
            else None
        ),
        "MENAGERIE_AUTHOR_MODEL": contract.author_model,
        # The consumer tests `== "1"`, so an off contract must leave the name
        # ABSENT: exporting "0" would be a value that reads as off today and as
        # set tomorrow.
        "MENAGERIE_AUTHOR_REQUIRE_RECEIPT": (
            "1" if contract.require_author_receipt else None
        ),
    }
    if set(persisted) != set(LAUNCH_ENVIRONMENT_VARIABLES):
        raise AssertionError("launch environment vocabulary and restore map disagree")

    divergences: list[JsonObject] = []
    for name in LAUNCH_ENVIRONMENT_VARIABLES:
        frozen_value = persisted[name]
        explicit = os.environ.get(name)
        if explicit is not None:
            if frozen_value is not None and explicit != frozen_value:
                divergences.append(
                    {"name": name, "persisted": frozen_value, "explicit": explicit}
                )
            continue
        if frozen_value is not None:
            os.environ[name] = frozen_value

    explicit_tmp_root = getattr(args, "tmp_root", None)
    if explicit_tmp_root is None:
        if contract.tmp_root is not None:
            args.tmp_root = contract.tmp_root
    elif contract.tmp_root is not None and Path(explicit_tmp_root) != contract.tmp_root:
        divergences.append(
            {
                "name": _TMP_ROOT_OVERRIDE_NAME,
                "persisted": str(contract.tmp_root),
                "explicit": str(explicit_tmp_root),
            }
        )
    return tuple(divergences)


def _disclose_launch_environment_divergence(
    paths: DriverPaths, config: DriverConfig, divergences: Sequence[JsonObject]
) -> None:
    """Append the ONE override-disclosure event, if anything actually diverged.

    Appended here rather than inside the restore itself: the restore runs before
    any records root is resolved, and an operational event with nowhere to land
    is not a disclosure. A failure to append is never fatal -- refusing the whole
    leg because a provenance note could not be written would turn disclosure into
    an outage -- but it is reported on stderr so it is not lost silently.

    Parameters
    ----------
    paths, config:
        Resolved driver paths and effective configuration.
    divergences:
        Records returned by :func:`_restore_launch_environment`.
    """

    if not divergences:
        return
    refreezes = [
        record
        for record in divergences
        if record.get("kind") == _CAMPAIGN_CONFIG_REFREEZE_KIND
    ]
    overrides = [
        record
        for record in divergences
        if record.get("kind") != _CAMPAIGN_CONFIG_REFREEZE_KIND
    ]
    events: list[JsonObject] = []
    if overrides:
        events.append(
            _launch_environment_divergence_event(
                overrides, run_id=config.run_id, machine_id=config.machine_id
            )
        )
        names = ", ".join(str(record["name"]) for record in overrides)
        print(
            f"launch-environment override in effect (explicit wins): {names}",
            file=sys.stderr,
        )
    for record in refreezes:
        events.append(
            _campaign_config_refreeze_event(
                record, run_id=config.run_id, machine_id=config.machine_id
            )
        )
        print(
            f"campaign config refrozen over {record['config_path']} "
            "(--refreeze-campaign-config)",
            file=sys.stderr,
        )
    try:
        with _launch_environment_ledger(paths) as operational:
            for event in events:
                operational.append(event)
    except (OSError, ValueError) as exc:
        print(f"launch-environment override disclosure failed: {exc}", file=sys.stderr)


def _launch_environment_ledger(paths: DriverPaths) -> JsonlLedger:
    """Return the canonical operational ledger this disclosure appends to."""

    return JsonlLedger(
        canonical_operational_ledger_path(paths.ledgers.models),
        OPERATIONAL_EVENT_SCHEMA_VERSION,
    )


def _launch_environment_divergence_event(
    divergences: Sequence[JsonObject], *, run_id: str, machine_id: str
) -> JsonObject:
    """Build the ONE campaign-health event disclosing launch-env overrides.

    A ``campaign-health`` event with an explicit disposition rather than a new
    ``OperationalEventKind``: the kind vocabulary is closed and consumed by the
    wake projection and the identity spec, and an override is exactly the kind
    of "the leg ran, under these inputs" provenance this event already carries.

    Parameters
    ----------
    divergences:
        Records returned by :func:`_restore_launch_environment`.
    run_id, machine_id:
        The resuming leg's identity.

    Returns
    -------
    JsonObject
        Minted operational event, ready to append.
    """

    return mint_operational_event(
        {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": utc_now(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 0},
            "current_environment": None,
            "run_id": run_id,
            "machine_id": machine_id,
            "details": {
                "disposition": "launch-environment-override",
                "overrides": [dict(record) for record in divergences],
            },
        }
    )


def _campaign_config_refreeze_event(
    record: JsonObject, *, run_id: str, machine_id: str
) -> JsonObject:
    """Build the campaign-health event disclosing one exercised refreeze.

    Same closed-vocabulary shape as the launch-environment override event: a
    ``campaign-health`` event with an explicit disposition, carrying both the
    superseded and refrozen config digests so the override is auditable.

    Parameters
    ----------
    record:
        Refreeze record minted by :func:`_resolve_campaign_config`.
    run_id, machine_id:
        The launching leg's identity.

    Returns
    -------
    JsonObject
        Minted operational event, ready to append.
    """

    return mint_operational_event(
        {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": utc_now(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 0},
            "current_environment": None,
            "run_id": run_id,
            "machine_id": machine_id,
            "details": {
                "disposition": _CAMPAIGN_CONFIG_REFREEZE_KIND,
                "refreeze": dict(record),
            },
        }
    )


def _apply_campaign_config(
    args: argparse.Namespace, config: CampaignConfig
) -> tuple[JsonObject, ...]:
    """Apply frozen campaign inputs to one parsed invocation.

    Parameters
    ----------
    args, config:
        Mutable parser namespace and validated private configuration.

    Returns
    -------
    tuple[JsonObject, ...]
        Launch-environment divergence records for operational disclosure.
    """

    args.intake = config.intake_root
    args.target = config.target
    args.run_id = config.run_id
    args.author_queue = (
        str(config.author_queue_root) if config.author_queue_root is not None else None
    )
    args.author_command = shlex.join(config.author_command)
    args.checker_command = shlex.join(config.checker_command)
    args.environment_command = shlex.join(config.environment_command)
    args.provider_reconcile_command = shlex.join(config.provider_reconcile_command)
    args.notify_command = (
        shlex.join(config.notify_command) if config.notify_command is not None else None
    )
    args.review_checkpoint_at = config.review_checkpoint_at
    args.progress_milestones = config.progress_milestones
    args.phase = config.phase
    args.only_status = config.only_status
    args.author_wall_seconds = config.author_wall_seconds
    # D-63 (PM-1): the lane widths are operator tunables and resolve
    # EXPLICIT-WINS -- the historical unconditional clobber here is exactly
    # what made the 200a width override inert while its disclosure lied.
    # Resolution must precede _restore_launch_environment so an inherited
    # operator export is distinguishable from the restore's own writes.
    width_divergences = _resolve_effective_concurrency(args, config)
    # The two mirrors keep their historical persisted-wins spelling: they are
    # the campaign's identity, not a tunable, and every consumer resolves them
    # through this exact pair. The launch environment contract below is the
    # operator-tunable half and is explicit-wins.
    os.environ["MENAGERIE_PUBLIC_MIRROR"] = str(config.public_mirror)
    os.environ["MENAGERIE_PRIVATE_MIRROR"] = str(config.private_mirror)
    return width_divergences + _restore_launch_environment(args, config)


def _validate_campaign_launch_policy(
    args: argparse.Namespace,
    binding: Optional[CampaignBinding],
) -> None:
    """Enforce the four-campaign checkpoint and advance-notification contract.

    Parameters
    ----------
    args:
        Fully restored launch arguments.
    binding:
        Exact manifest binding, when the intake belongs to the production partition.

    Raises
    ------
    OperatorOutageError
        If a production campaign would launch with the wrong human gate or silently
        approach C1's checkpoint.
    """

    if binding is None:
        return
    expected_review = binding.spec.review_checkpoint_at
    if args.review_checkpoint_at != expected_review:
        raise OperatorOutageError(
            f"{binding.spec.campaign_id} requires --review-checkpoint-at {expected_review}"
        )
    if binding.spec.campaign_id != "c1-mech":
        return
    required = {900, 950, 1000}
    missing = sorted(required - set(args.progress_milestones))
    if missing:
        raise OperatorOutageError(
            f"c1-mech requires durable progress notifications at 900,950,1000; missing {missing}"
        )


def _required_absolute_environment_path(name: str) -> Path:
    """Return one mandatory absolute operator path from the environment.

    Parameters
    ----------
    name:
        Environment variable name.

    Returns
    -------
    pathlib.Path
        Resolved absolute path.
    """

    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"campaign launch requires {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path.resolve()


def _dry_run_acceptance(
    snapshot: IntakeSnapshot,
    current: Mapping[str, Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    *,
    paused: bool,
) -> dict[str, object]:
    """Evaluate semantic success for the tiny-model acceptance gate.

    Parameters
    ----------
    snapshot, current, attempts:
        Expected tiny intake, canonical current projection, and authenticated attempt ledger.
    paused:
        Whether the driver intentionally stopped at a resumable checkpoint.

    Returns
    -------
    dict[str, object]
        Structured pending, passed, or acceptance-failed result.
    """

    expected_ids = {item.stable_id for item in snapshot.items}
    runs_records = {
        stable_id: record
        for stable_id, record in current.items()
        if stable_id in expected_ids and record.get("status", {}).get("code") == "runs"
    }
    authenticated_attempt_ids: dict[str, set[str]] = {
        stable_id: set() for stable_id in expected_ids
    }
    for attempt in attempts:
        stable_id = attempt.get("stable_id")
        if not isinstance(stable_id, str) or stable_id not in expected_ids:
            continue
        raw_digest = attempt.get("raw_award_receipt_sha256")
        parent = attempt.get("parent_attestation")
        if (
            attempt.get("result") == "succeeded"
            and isinstance(attempt.get("raw_award_receipt"), Mapping)
            and isinstance(raw_digest, str)
            and isinstance(parent, Mapping)
            and parent.get("named_raw_award_receipt_sha256") == raw_digest
            and isinstance(attempt.get("attempt_id"), str)
        ):
            authenticated_attempt_ids[stable_id].add(str(attempt["attempt_id"]))
    authenticated_models = {
        stable_id
        for stable_id, record in runs_records.items()
        if authenticated_attempt_ids[stable_id]
        and set(record.get("execution", {}).get("accepted_attempt_ids", ()))
        == authenticated_attempt_ids[stable_id]
    }
    counts = {
        "expected_runnable_models": len(expected_ids),
        "runs_revisions": len(runs_records),
        "authenticated_attempt_models": len(authenticated_models),
    }
    if paused:
        return {"status": "pending", "reason_code": "driver-paused", **counts}
    if set(runs_records) == expected_ids and authenticated_models == expected_ids:
        return {"status": "passed", **counts}
    unexpected_statuses = Counter(
        str(record.get("status", {}).get("code", "missing-status"))
        for stable_id, record in current.items()
        if stable_id in expected_ids and stable_id not in runs_records
    )
    missing_records = expected_ids - set(current)
    if missing_records:
        unexpected_statuses["missing"] += len(missing_records)
    all_source_failure = (
        len(current) == len(expected_ids)
        and set(current) == expected_ids
        and set(unexpected_statuses) == {"failed:source"}
    )
    return {
        "status": "acceptance-failed",
        "reason_code": "all-source-failure" if all_source_failure else "expected-runs-missing",
        **counts,
        "unexpected_statuses": dict(sorted(unexpected_statuses.items())),
    }


def _persisted_environment_generations(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Recover exact environment generations already authenticated by attempt rows.

    Parameters
    ----------
    attempts:
        Canonical attempt-ledger rows for the dry-run campaign.

    Returns
    -------
    dict[str, str]
        Environment-family to generation bindings needed for current projection.
    """

    generations: dict[str, str] = {}
    for attempt in attempts:
        environment = attempt.get("environment")
        if not isinstance(environment, Mapping):
            continue
        family = environment.get("family")
        identities = attempt.get("identities")
        generation = identities.get("environment") if isinstance(identities, Mapping) else None
        if not isinstance(family, str) or not isinstance(generation, str):
            continue
        existing = generations.get(family)
        if existing is not None and existing != generation:
            raise ValueError(f"dry-run attempts contain conflicting generations for {family}")
        generations[family] = generation
    return generations


def _default_scheduler() -> str:
    """Return the leg scheduling shape from the environment or the default.

    Returns
    -------
    str
        ``MENAGERIE_SCHEDULER`` when it names a known scheduler, else
        ``waves``. A malformed value is rejected rather than silently ignored,
        exactly like the concurrency bounds.
    """

    raw = os.environ.get("MENAGERIE_SCHEDULER")
    if raw is None or not raw.strip():
        return DEFAULT_SCHEDULER
    value = raw.strip()
    if value not in SCHEDULER_MODES:
        raise ValueError(
            f"MENAGERIE_SCHEDULER must name one of {list(SCHEDULER_MODES)}, not {raw!r}"
        )
    return value


def _resolve_author_wall(args: argparse.Namespace, campaign_id: Optional[str]) -> float:
    """Resolve this campaign's authoritative wall grant.

    One number governs the whole chain: it is the grant published to the author
    session in its envelope, the grant the executor sizes its own per-session
    kill from, and the base the lane derives its outer stall bound from.

    Resolution is a PURE read. The grant reaches the executor through the
    wrapper subprocess's own ``env=`` (see
    ``CommandAuthorLane._child_environment``), never by writing the driver's
    ``os.environ``: a process-wide write is order-dependent and outlives its
    caller, so one campaign's grant would remain visible to whatever ran next in
    the same process and this very guard would then refuse a correct
    configuration.

    An operator-set environment value that disagrees is still a startup failure.
    Letting it win silently is how a campaign runs a month on a budget nobody
    configured; the incoherence this guard replaced -- a 30-minute lane bound
    under a 60-minute c3 grant -- surfaced only as timeouts at minute 30, blind
    retries that died the same way, and a corrupted p95.

    Parameters
    ----------
    args:
        Parsed driver arguments, carrying any explicit operator override.
    campaign_id:
        Tier campaign identity, when the intake belongs to one.

    Returns
    -------
    float
        Authoritative per-session wall grant in seconds.

    Raises
    ------
    OperatorOutageError
        If the override is malformed, or a conflicting grant is already set in
        the ambient environment.
    """

    try:
        grant = resolve_author_wall_seconds(campaign_id, getattr(args, "author_wall_seconds", None))
    except ValueError as exc:
        raise OperatorOutageError(str(exc)) from exc
    existing = os.environ.get(AUTHOR_WALL_SECONDS_ENV)
    if existing is not None and existing.strip():
        try:
            preset = float(existing)
        except ValueError as exc:
            raise OperatorOutageError(
                f"{AUTHOR_WALL_SECONDS_ENV} must be a number of seconds, not {existing!r}"
            ) from exc
        if preset != grant:
            raise OperatorOutageError(
                f"{AUTHOR_WALL_SECONDS_ENV}={preset:g} conflicts with the resolved "
                f"{grant:g}s grant for campaign {campaign_id or '<none>'}. The campaign "
                f"config is authoritative; unset the variable or pass "
                f"--author-wall-seconds {preset:g} so one budget governs the run."
            )
    return grant


def _optional_author_queue_root(args: argparse.Namespace) -> Optional[Path]:
    """Resolve the author-queue root when the in-session pool serves the lane.

    The production author pool is a set of Claude subagents inside one live
    managing session, not a re-entrant CLI subprocess. Setting
    ``--author-queue`` (or ``MENAGERIE_AUTHOR_QUEUE``) selects the file-queue RPC
    bridge and makes ``--author-command`` unnecessary.

    Parameters
    ----------
    args:
        Parsed driver arguments.

    Returns
    -------
    Path | None
        Absolute author-queue root, or ``None`` for the wrapper-command lane.
    """

    raw = getattr(args, "author_queue", None)
    if raw is None or not str(raw).strip():
        return None
    return Path(str(raw)).expanduser().resolve()


def _required_command(value: Optional[str], lane: str) -> tuple[str, ...]:
    """Parse one required non-shell lane command with a clear configuration error."""

    if value is None or not value.strip():
        environment_name = f"MENAGERIE_{lane.upper().replace('-', '_')}_COMMAND"
        raise ValueError(f"{lane} lane requires --{lane}-command or {environment_name}")
    command = tuple(shlex.split(value))
    if not command:
        raise ValueError(f"{lane} lane command cannot be empty")
    return command


def _snapshot_authority_context(
    snapshot: IntakeSnapshot,
    config: DriverConfig,
) -> AuthorityContext:
    """Build current projection authority for a loaded intake snapshot.

    The registered prior-lineage facts are part of the authority, not an extra:
    K2's currency carve-out is what lets a revision recorded under a PRIOR
    registered snapshot of the same campaign stay dependency-current, and the
    T4-T1 membership half is what lets a SUBSET rung read its superset history.
    ``CrawlerDriver`` passes both (``driver.py``); this builder did not, so every
    CLI projection silently disagreed with the driver that produced the ledgers --
    on a real closed rung, 1 of 127 revisions resolved as current instead of 10 of
    10, and ``status --verify-partition`` called a COMPLETE partition invalid. The
    readers live driver-side by design (``authority`` is a sealed worker-closure
    member); the CLI is a driver-side caller exactly like the driver.
    """

    return build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=config.author_model,
        author_version=config.author_version,
        checker_model=config.checker_model,
        checker_version=config.checker_version,
        prior_intake_snapshots=registered_prior_intake_snapshots(
            snapshot.root, snapshot.snapshot_id
        ),
        prior_intake_member_ids=registered_prior_intake_members(
            snapshot.root, snapshot.snapshot_id
        ),
    )


def _optional_campaign_binding(
    repo_root: Path,
    snapshot: IntakeSnapshot,
) -> Optional[CampaignBinding]:
    """Return a campaign binding when the manifest or rung registry owns the snapshot."""

    manifest_path = repo_root / DEFAULT_CAMPAIGN_MANIFEST
    if not manifest_path.is_file():
        return None
    binding = find_campaign_binding(manifest_path, snapshot)
    if binding is not None:
        # D-47 (B3-XO-27, the owed LEDGER answer): two registries disagreeing
        # on the intake set is a TYPED refusal, never a silent pick. The
        # manifest historically won without ever consulting the rung
        # registry; a registration naming a DIFFERENT campaign for the same
        # snapshot now refuses at leg open.
        registration = find_rung_registration(
            manifest_path.parent, snapshot.snapshot_id, snapshot.snapshot_sha256
        )
        if registration is not None:
            registered_campaign = str(registration.campaign_id)
            if registered_campaign != str(binding.spec.campaign_id):
                raise DivergentIntakeRegistriesError(
                    f"intake snapshot {snapshot.snapshot_id} is claimed by TWO "
                    f"registries that disagree: the campaign manifest binds it "
                    f"to {binding.spec.campaign_id!r} while the rung registry "
                    f"registers it under {registered_campaign!r}. Reconcile the "
                    "registries; a silent pick would run the wrong campaign's "
                    "models (B3-XO-27)"
                )
        return binding
    destination = promotion_snapshot_destination(snapshot)
    if destination is None:
        registration = find_rung_registration(
            manifest_path.parent, snapshot.snapshot_id, snapshot.snapshot_sha256
        )
        if registration is None:
            return None
        return registration_binding(manifest_path.parent, registration)
    matches = [
        candidate
        for candidate in load_campaign_bindings(manifest_path)
        if candidate.spec.campaign_id == destination
    ]
    if len(matches) != 1:
        raise ValueError(
            f"promotion destination {destination!r} has {len(matches)} campaign bindings"
        )
    return matches[0]


def _required_campaign_binding(repo_root: Path, snapshot: IntakeSnapshot) -> CampaignBinding:
    """Resolve the campaign binding an intake snapshot MUST have to be used.

    An unbound snapshot used to fall back to the default ``DriverConfig`` --
    silently the wrong author model, checker model, and review checkpoint for
    any campaign whose binding differs (T4-T2). Every consumer now refuses.

    Parameters
    ----------
    repo_root:
        Campaign repository root.
    snapshot:
        Loaded hash-verified intake snapshot.

    Returns
    -------
    CampaignBinding
        Manifest, promotion, or registered-rung binding.

    Raises
    ------
    UnboundIntakeSnapshotError
        If neither the manifest nor the rung registry owns the snapshot.
    """

    manifest_path = repo_root / DEFAULT_CAMPAIGN_MANIFEST
    if not manifest_path.is_file():
        raise UnboundIntakeSnapshotError(
            f"snapshot {snapshot.snapshot_id} cannot bind: no campaign manifest at "
            f"{manifest_path}. Campaign work runs only against a partitioned "
            "repository; defaults would select the wrong author model."
        )
    binding = _optional_campaign_binding(repo_root, snapshot)
    if binding is None:
        raise UnboundIntakeSnapshotError(
            f"intake snapshot {snapshot.snapshot_id} is not bound to any campaign: "
            "no manifest row and no rung registration own it, and defaulting the "
            "driver config would run the wrong author/checker models. Stage or "
            "adopt it first: python -m menagerie.crawler.tools.stage_rung "
            "{stage,register} --campaign <c1-mech|c2-disco|c3-classics|c4-native> ..."
        )
    return binding


def _driver_paths_for_snapshot(repo_root: Path, snapshot: IntakeSnapshot) -> DriverPaths:
    """Return isolated canonical paths for ordinary or promoted campaign work.

    Parameters
    ----------
    repo_root:
        Campaign repository root.
    snapshot:
        Loaded hash-verified intake snapshot.

    Returns
    -------
    DriverPaths
        Ordinary paths, or the isolated C3 promotion ledger/runtime roots.
    """

    paths = default_driver_paths(repo_root, snapshot.root)
    if promotion_snapshot_destination(snapshot) is None:
        return paths
    records_root = paths.ledgers.models.parent.parent
    return replace(
        paths,
        runtime_root=paths.runtime_root / "promotion-campaigns" / "c3-classics",
        ledgers=default_ledger_paths(promotion_records_root(records_root)),
    )


def _snapshot_driver_config(repo_root: Path, snapshot: IntakeSnapshot) -> DriverConfig:
    """Return exact campaign provenance labels for one intake snapshot.

    Raises
    ------
    UnboundIntakeSnapshotError
        If no campaign owns the snapshot; provenance labels are never defaulted.
    """

    binding = _required_campaign_binding(repo_root, snapshot)
    return DriverConfig(
        review_checkpoint_at=binding.spec.review_checkpoint_at,
        author_model=binding.spec.author_model,
        checker_model=binding.spec.checker_model,
    )


def _status_machine_deferrals(
    records_root: Path,
    *,
    repo_root: Optional[Path] = None,
    snapshot: Optional[IntakeSnapshot] = None,
    ledger_heads: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Mapping[str, Any]]:
    """Load durable machine-deferral rows for the CLI partition view.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    repo_root:
        Repository whose committed environment registry defines current setup
        artifact identities. When omitted with ``snapshot``, rebuild rows are
        loaded without currency filtering for narrow ledger projections.
    snapshot:
        Active intake whose routes determine whether a rebuild quarantine fact
        still applies. Must be supplied together with ``repo_root``.
    ledger_heads:
        Raw ledger heads used to select WHICH row represents a model that has
        several. The partition arm accounts one row per stable ID, and the flat
        ``{stable_id: row}`` collapse used to keep whichever the fixed valve
        concatenation ordered LAST -- so a model holding an ordinary valve row
        and a live park (or one park kind under two work generations) was
        represented by an accident of order (T1-opus-04 secondary). Selection
        now prefers, deterministically, a row that is still a LIVE claim
        against the model's ledger head over one the head already released.

    Returns
    -------
    dict[str, Mapping[str, Any]]
        Selected persisted valve row keyed by stable model ID.
    """

    rebuild_rows = load_rebuild_deferral_rows([rebuild_deferral_path(records_root)])
    if repo_root is not None and snapshot is not None and rebuild_rows:
        registry = load_environment_registry(repo_root / "menagerie" / "crawler" / "envs")
        driver = SimpleNamespace(
            registry=registry,
            paths=SimpleNamespace(ledgers=default_ledger_paths(records_root)),
        )
        intake_by_stable_id = {item.stable_id: item for item in snapshot.items}
        current_rebuild_rows: list[Mapping[str, Any]] = []
        for row in rebuild_rows:
            stable_id = str(row["stable_id"])
            intake = intake_by_stable_id.get(stable_id)
            if intake is None:
                continue
            item = WorkItem(
                intake,
                route_model(
                    intake.to_model_requirements(framework_from_markers(intake.zoo, intake.name))
                ),
            )
            if _deferral_row_fact_still_holds(driver, item, dict(row)):
                current_rebuild_rows.append(row)
        rebuild_rows = current_rebuild_rows
    valve_rows = (
        *load_capacity_deferral_rows([capacity_deferral_path(records_root)]),
        *load_coverage_deferral_rows([coverage_deferral_path(records_root)]),
        *rebuild_rows,
        *load_disk_floor_deferral_rows([disk_floor_deferral_path(records_root)]),
        *load_checker_lane_deferral_rows([checker_lane_deferral_path(records_root)]),
        # D-17: the scheduler-quarantine ledger is a member of the closed
        # MACHINE_DEFERRAL_DISPOSITIONS set the close accounts, and this view
        # omitted it -- a member whose only accounting was a quarantine row
        # read as a coverage hole here while the real close accounted it.
        *load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)]),
        *load_grant_park_rows([grant_park_path(records_root)]),
        *load_refusal_park_rows([refusal_park_path(records_root)]),
    )
    heads = dict(ledger_heads or {})
    selected: dict[str, Mapping[str, Any]] = {}
    for row in valve_rows:
        stable_id = str(row["stable_id"])
        incumbent = selected.get(stable_id)
        if incumbent is None or (
            _valve_row_is_live(incumbent, heads.get(stable_id)) is False
            and _valve_row_is_live(row, heads.get(stable_id)) is True
        ):
            selected[stable_id] = row
    return selected


def _status_partition_accounting(
    records_root: Path,
    *,
    scope_ids: Iterable[str],
    ledger_heads: Mapping[str, Mapping[str, Any]],
    repo_root: Optional[Path] = None,
    snapshot: Optional[IntakeSnapshot] = None,
) -> tuple[dict[str, Mapping[str, Any]], frozenset[str]]:
    """Derive the partition inputs the CLI shares verbatim with the close.

    D-17 (LC-3): the boundary gate and the driver's close must consume ONE
    readiness derivation. The claims view is this module's durable selection
    (:func:`_status_machine_deferrals`); the stale accounting is the close's
    own :func:`~menagerie.crawler.driver.durable_readiness_accounting`,
    called with an empty run-scoped claim set because an offline reader has
    no run -- exactly the pre-admission state the close asserts in.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    scope_ids:
        In-scope intake membership.
    ledger_heads:
        RAW ledger heads (the park-release and coverage authority, BD-1).
    repo_root, snapshot:
        Optional environment-registry context for rebuild-row currency
        filtering of the CLAIMS view (see :func:`_status_machine_deferrals`).

    Returns
    -------
    tuple[dict[str, Mapping[str, Any]], frozenset[str]]
        Claims mapping and the exact ``stale_accounted_ids`` the close passes.
    """

    scoped = frozenset(str(stable_id) for stable_id in scope_ids)
    heads = dict(ledger_heads)
    machine_deferrals = _status_machine_deferrals(
        records_root,
        repo_root=repo_root,
        snapshot=snapshot,
        ledger_heads=heads,
    )
    accounting = durable_readiness_accounting(
        SimpleNamespace(paths=SimpleNamespace(ledgers=default_ledger_paths(records_root))),
        scoped_ids=scoped,
        ledger_heads={
            stable_id: head for stable_id, head in heads.items() if stable_id in scoped
        },
    )
    return machine_deferrals, accounting.stale_accounted_ids


def _valve_row_is_live(row: Mapping[str, Any], head: Optional[Mapping[str, Any]]) -> bool:
    """Return whether one valve row still claims its model against the head.

    Parameters
    ----------
    row:
        Candidate persisted valve row.
    head:
        The model's RAW ledger head, or ``None`` when it holds no revision.

    Returns
    -------
    bool
        True when the row is a live claim: any structurally valid non-park
        valve row for a model with no head, or a park row the head has not
        released.
    """

    if machine_deferral_rejection(str(row.get("stable_id")), row) is not None:
        return False
    disposition = str(row.get("disposition"))
    if disposition == GRANT_PENDING_DISPOSITION:
        return not grant_park_released(row, head)
    if disposition == SELECTION_EXCLUSION_DISPOSITION:
        # R2-1: the exclusion park's liveness is the same head-bound fact.
        return not grant_park_released(row, head)
    if disposition == REFUSAL_PARK_DISPOSITION:
        return not refusal_park_released(row, head)
    return head is None


def _wake_liveness_refusal(
    events: Sequence[Mapping[str, object]],
    *,
    now: str,
    required: bool,
) -> Optional[str]:
    """Return the fail-closed wake refusal for ``status``, or ``None``.

    Extracted from :func:`_status_command` so the DECISION is reachable without a
    whole campaign snapshot: an unreachable gate is an untested gate, and this one
    exists precisely because the pilot's only real pause reported ``healthy``
    while it sat dead past its reset (AREA_OPS_opus OPS-08).

    It consults the same one derivation the deadman and the supervisor use, so a
    gate and an alarm can never disagree about whether the campaign's only
    automated resume path is alive.

    Parameters
    ----------
    events:
        Canonical operational events in ledger order.
    now:
        Exact UTC instant to measure liveness against.
    required:
        Whether ``--require-live-wake`` asked for the fail-closed behaviour.

    Returns
    -------
    str | None
        Semicolon-joined alarm summaries, or ``None`` when the gate is off or
        every active episode is live.
    """

    if not required:
        return None
    alarms = wake_liveness_alarms(events, now=now)
    if not alarms:
        return None
    return "; ".join(alarm.summary() for alarm in alarms)


def _status_command(args: argparse.Namespace) -> int:
    """Print current funnel and optional exact partition diagnostics."""

    snapshot = load_intake_snapshot(args.intake)
    records_root = args.records_root or args.repo_root / "menagerie" / "crawler" / "records"
    paths = _driver_paths_for_snapshot(args.repo_root, snapshot).ledgers
    if records_root != args.repo_root / "menagerie" / "crawler" / "records":
        paths = DriverPaths(args.repo_root / ".crawl-local", args.intake, paths).ledgers
        paths = type(paths)(
            records_root / "models" / "current-shard.jsonl",
            records_root / "attempts" / "local.jsonl",
            records_root / "gates" / "current-shard.jsonl",
        )
    # F12, both halves: the CLI projection must carry the SAME authority the
    # driver's close used. `_snapshot_authority_context` restores the registered
    # prior-intake lineage; `restored_environment_generations` restores the
    # per-family CURRENT environment generation from ledger evidence (newest
    # attempt per family plus quarantine bindings -- the shared offline-consumer
    # authority the scheduler shadow audit also routes through). Without the
    # second half every EXECUTED revision (award) reads as non-current and a
    # rung the driver closed COMPLETE reports missing members. The dry-run
    # summary's `_persisted_environment_generations` is NOT this fix: it asserts
    # one generation per family and correctly refuses a real campaign ledger.
    operational_path = canonical_operational_ledger_path(paths.models)
    context = _snapshot_authority_context(
        snapshot,
        _snapshot_driver_config(args.repo_root, snapshot),
    )
    context = replace(
        context,
        environment_generations=restored_environment_generations(
            context.environment_generations,
            paths.attempts,
            scan_jsonl(operational_path) if operational_path.exists() else (),
        ),
    )
    # BD-1: the projection is consumed WITH its stale bucket, and the RAW
    # ledger heads are read beside it. Coverage, park release, and the deferral
    # accounting are facts about recorded revisions; dependency currency is a
    # separate fact about whether the live authority can vouch for them. Fusing
    # the two made a stale head read as NONEXISTENT: 24 recorded terminals (7
    # awards) reported "missing", 8 more awards reported "machine-deferred",
    # 16 of 17 released parks reported LIVE, and `status:runs: 0` in a funnel
    # over a ledger holding 33 awards.
    projection = project_dependency_current(paths, context=context)
    current = projection.current_records
    heads = _select_current(scan_jsonl(paths.models))
    scoped_stale = stale_census(
        projection.stale_reasons, scope=(item.stable_id for item in snapshot.items)
    )
    # RC-2: the recheck census and the configuration history are derived per
    # read from the RAW heads, the measured-rotations ledger, and the live
    # context. A rotation class never enters the currency projection above;
    # this census is the one place it lands, and the release view excludes
    # exactly its ``needs_recheck`` members.
    rotations = load_rotations(records_root)
    recheck = recheck_census(
        heads, rotations, context, scope=(item.stable_id for item in snapshot.items)
    )
    stale_ids = sorted(
        stable_id for stable_ids in scoped_stale.values() for stable_id in stable_ids
    )
    # The clock is passed EXPLICITLY: without it the wake block cannot express
    # overdue-ness and renders `healthy: unknown` rather than an affirmatively
    # reassuring `true` (AREA_OPS_opus OPS-08, BL-6 (c)/(f)).
    operational_events = scan_jsonl(operational_path)
    status_now = utc_now()
    wakeups = wakeup_status(operational_events, now=status_now)
    stale_block: dict[str, object] = {
        "count": len(stale_ids),
        "by_axis": {axis: sorted(ids) for axis, ids in scoped_stale.items()},
        "reasons": {stable_id: projection.stale_reasons[stable_id] for stable_id in stale_ids},
    }
    output: dict[str, object] = {
        "funnel": funnel_counts(current),
        "terminal": len(current),
        "ledger_heads": len(heads),
        "head_status_counts": dict(
            sorted(
                Counter(str(head.get("status", {}).get("code")) for head in heads.values()).items()
            )
        ),
        "stale": stale_block,
        "awards_stale_on_live_basis": len(awards_stale_on_live_basis(heads, stale_ids)),
        "recheck_census": recheck,
        "configuration_history": rotation_history(rotations),
        "wakeups": wakeups,
        # D-19 (waiver doctrine part 4): the grant-release census is a
        # REGISTERED waiver's standing disclosure and must be computed by a
        # production reader, not only by the suite -- an infra-waiver-released
        # grant an operator never sees is not disclosed.
        "grant_release_census": {
            name: list(members)
            for name, members in grant_release_census_from_ledgers(paths).items()
        },
        # W12 (waiver doctrine part 4): the scheduler-quarantine release
        # census; held-awaiting-operator-drain is the operator-owed
        # quarantine debt.
        "scheduler_drain_census": {
            name: list(members)
            for name, members in scheduler_drain_census_from_ledgers(paths).items()
        },
    }
    # Live grant-pending parks are pending OPERATOR decisions: they surface
    # here unconditionally so nobody has to know to ask for them. Release keys
    # on the model's RAW LEDGER HEAD, never on the currency projection: a head
    # the projection dropped is still a landed revision, and keying release on
    # the projection is exactly why the step-100 close reported phantom parks
    # for models that had terminalized hours earlier (T1-opus-04, T2-opus-07).
    park_rows = load_grant_park_rows([grant_park_path(records_root)])
    pending_parks = [
        {
            "stable_id": row["stable_id"],
            "work_id": row["work_id"],
            "stage": row["pending"]["blocked_stage"],
            "blocked_parent_revision": row["pending"]["blocked_parent_revision"],
            "pending_status_code": row["pending"]["pending_status_code"],
            "row_sha256": row["row_sha256"],
        }
        for row in park_rows
        if not grant_park_released(row, heads.get(str(row["stable_id"])))
    ]
    if pending_parks:
        output["review_holds"] = pending_parks
    # Live terminal-refusal parks are pending operator decisions too: models
    # whose terminal the recording path refused to write. Surfaced
    # unconditionally so nobody has to know to ask for them.
    refusal_rows = load_refusal_park_rows([refusal_park_path(records_root)])
    pending_refusals = [
        {
            "stable_id": row["stable_id"],
            "work_id": row["work_id"],
            "refusal_class": row["refusal"]["refusal_class"],
            "record_error": row["refusal"]["record_error"],
            "intended_status_code": row["refusal"]["intended_status_code"],
            "row_sha256": row["row_sha256"],
        }
        for row in refusal_rows
        if not refusal_park_released(row, heads.get(str(row["stable_id"])))
    ]
    if pending_refusals:
        output["refusal_parks"] = pending_refusals
    partition_refusal: Optional[str] = None
    if getattr(args, "require_valid_partition", False) and not args.verify_partition:
        print(
            "--require-valid-partition requires --verify-partition: the boundary "
            "form asserts the partition it verifies, and must never degrade to "
            "asserting nothing",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.verify_partition:
        # One derivation feeds BOTH the report and the boundary assertion, so
        # the two views can never disagree about which valve rows still hold --
        # and (D-17) the stale accounting is the close's OWN derivation, so
        # this gate and the driver's close can never disagree either. Before
        # D-17 this arm loaded seven of the eight deferral sources (no
        # scheduler-quarantine) and passed no stale_accounted_ids at all, so a
        # rung the close completed could refuse here.
        machine_deferrals, stale_accounted_ids = _status_partition_accounting(
            records_root,
            scope_ids=(item.stable_id for item in snapshot.items),
            ledger_heads=heads,
            repo_root=args.repo_root,
            snapshot=snapshot,
        )
        report = partition_report(
            (item.stable_id for item in snapshot.items),
            current,
            machine_deferrals=machine_deferrals,
            ledger_heads=heads,
            stale_reasons=projection.stale_reasons,
            stale_accounted_ids=stale_accounted_ids,
        )
        output["partition_valid"] = report.valid
        output["missing"] = sorted(report.missing_ids)
        # Which stale members actually invalidate the partition, so the
        # report-only form answers the same question the refusing form does:
        # every stale head is in the census, and these are the ones nothing
        # accounts. The remainder are disclosed beside them, segregated.
        stale_block["unaccounted"] = sorted(report.unaccounted_stale_ids)
        stale_block["deferred_for_redrive"] = {
            disposition: sorted(stable_ids)
            for disposition, stable_ids in report.stale_deferred.items()
        }
        if getattr(args, "require_valid_partition", False):
            # The boundary form: the SAME assert_partition call the driver's
            # close makes, with the SAME durable stale accounting (D-17), so
            # the refusal text and the coverage accounting stay one
            # implementation. Reporting only is what a rung-boundary gate keyed
            # on the exit code cannot distinguish from success -- and a stale
            # bucket that only REPORTED would be the same defect one layer up,
            # so it refuses here exactly like a coverage hole.
            try:
                assert_partition(
                    (item.stable_id for item in snapshot.items),
                    current,
                    machine_deferrals=machine_deferrals,
                    ledger_heads=heads,
                    stale_reasons=projection.stale_reasons,
                    stale_accounted_ids=stale_accounted_ids,
                )
            except PartitionError as exc:
                partition_refusal = str(exc)
    wake_refusal = _wake_liveness_refusal(
        operational_events,
        now=status_now,
        required=bool(getattr(args, "require_live_wake", False)),
    )
    if args.full:
        output["current"] = current
    print(json.dumps(output, sort_keys=True))
    for refusal in (partition_refusal, wake_refusal):
        if refusal is not None:
            print(refusal, file=sys.stderr)
    if partition_refusal is not None or wake_refusal is not None:
        return EXIT_ERROR
    return EXIT_OK


def _capacity_deferrals_command(args: argparse.Namespace) -> int:
    """Print every model withheld for size, newest campaign roots included.

    This is the "show me everything deferred for size" view. Each row names the
    parameter bound that was derived and the host ceiling it exceeded, so the
    call can be judged without re-running anything, and the ``recheck_hint``
    names the machine that would clear it.
    """

    records_root = args.records_root or args.repo_root / "menagerie" / "crawler" / "records"
    rows = load_capacity_deferral_rows([capacity_deferral_path(records_root)])
    host = host_capacity()
    output: dict[str, object] = {
        "disposition": CAPACITY_DEFERRAL_DISPOSITION,
        "deferred": len(rows),
        "host_physical_memory_bytes": host.physical_memory_bytes,
        "admissible_parameter_ceiling": host.admissible_parameter_ceiling,
        "models": [
            (
                dict(row)
                if args.full
                else {
                    "stable_id": row["stable_id"],
                    "name": row["name"],
                    "work_id": row["work_id"],
                    "parameter_count_lower_bound": row["capacity"]["estimate"][
                        "parameter_count_lower_bound"
                    ],
                    "estimated_parameter_bytes": row["capacity"]["estimate"][
                        "estimated_parameter_bytes"
                    ],
                    "admissible_parameter_ceiling": row["capacity"]["threshold"][
                        "admissible_parameter_ceiling"
                    ],
                    "recheck_hint": row["recheck_hint"],
                }
            )
            for row in rows
        ],
    }
    print(json.dumps(output, sort_keys=True))
    return EXIT_OK


def _coverage_deferrals_command(args: argparse.Namespace) -> int:
    """Print every model withheld because its routed environment lacks its library.

    This is the "show me everything deferred for routing" view. Each row names
    the distribution that was pinned, the intent that could not install it, and
    the intent that would -- so the call can be judged and acted on without
    re-running anything.
    """

    records_root = args.records_root or args.repo_root / "menagerie" / "crawler" / "records"
    rows = load_coverage_deferral_rows([coverage_deferral_path(records_root)])
    output: dict[str, object] = {
        "disposition": COVERAGE_DEFERRAL_DISPOSITION,
        "deferred": len(rows),
        "models": [
            (
                dict(row)
                if args.full
                else {
                    "stable_id": row["stable_id"],
                    "name": row["name"],
                    "work_id": row["work_id"],
                    "distribution": row["coverage"]["library"]["distribution"],
                    "version": row["coverage"]["library"]["version"],
                    "routed_intent": row["coverage"]["routed_intent"],
                    "covering_intents": [
                        entry["intent"] for entry in row["coverage"]["covering_intents"]
                    ],
                    "recheck_hint": row["recheck_hint"],
                }
            )
            for row in rows
        ],
    }
    print(json.dumps(output, sort_keys=True))
    return EXIT_OK


def _wake_command(args: argparse.Namespace) -> int:
    """Inspect or durably cancel one exact wake episode under the driver lock."""

    records_root = args.records_root or args.repo_root / "menagerie" / "crawler" / "records"
    operational_path = records_root / "operational" / "events.jsonl"
    if args.wake_command == "status":
        events = scan_jsonl(operational_path)
        print(
            json.dumps(
                render_wakeup_status(
                    reduce_wake_episodes(events), now=utc_now(), events=events
                ),
                sort_keys=True,
            )
        )
        return EXIT_OK

    runtime = args.repo_root / ".crawl-local"
    owner = {
        "pid": os.getpid(),
        "run_id": "wake-cancel",
        "target": "wake-control",
        "created_at": utc_now(),
        "episode_id": args.episode_id,
    }
    with DriverLock(runtime / "locks" / "driver.lock", owner):
        with JsonlLedger(operational_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
            projection = reduce_wake_episodes(operational.records)
            try:
                state = projection.episodes[args.episode_id]
            except KeyError as exc:
                raise ValueError(f"unknown wake episode: {args.episode_id}") from exc
            opening = next(
                (
                    event
                    for event in operational.records
                    if isinstance(event.get("details"), dict)
                    and isinstance(event["details"].get("episode"), dict)
                    and event["details"]["episode"].get("episode_id") == args.episode_id
                ),
                None,
            )
            if opening is None:
                raise ValueError("wake episode lacks its owning campaign event")
            context = OperationalContext(
                run_id=str(opening["run_id"]),
                machine_id=str(opening["machine_id"]),
                queued_work_counts=dict(opening["queued_work_counts"]),
                current_environment=(
                    str(opening["current_environment"])
                    if opening["current_environment"] is not None
                    else None
                ),
            )
            manager = WakeupManager(
                # THE one derivation: cancel used to build `wakeups` (plural)
                # while the driver installed into `wakeup`, so it left the real
                # projection installed and bootstrapped duplicates into a
                # directory nothing reads (AREA_OPS_sol OPS-B08).
                wakeup_projection_root(runtime),
                operational,
                state.episode.callback_argv,
                backend=state.backend,
            )
            manager.resolve_episode(
                args.episode_id,
                resolution="operator-cancelled",
                context=context,
                created_at=utc_now(),
            )
            manager.reconcile(context=context, created_at=utc_now())
            output = render_wakeup_status(
                manager.projection, now=utc_now(), events=operational.records
            )
    print(json.dumps(output, sort_keys=True))
    return EXIT_OK


def _checkpoint_command(args: argparse.Namespace) -> int:
    """Run and report one complete fail-closed canonical checkpoint transaction."""

    del args.verify_ledgers, args.verify_views
    owner = {
        "pid": os.getpid(),
        "run_id": "checkpoint",
        "target": "canonical-checkpoint",
        "created_at": utc_now(),
        "command": list(sys.argv),
    }
    try:
        with DriverLock(args.repo_root / ".crawl-local" / "locks" / "driver.lock", owner):
            snapshot = load_intake_snapshot(args.intake)
            defaults = _snapshot_driver_config(args.repo_root, snapshot)
            context = build_authority_context(
                active_intake_snapshot_id=snapshot.snapshot_id,
                active_intake_snapshot_sha256=snapshot.snapshot_sha256,
                intake_rows=(item.to_dict() for item in snapshot.items),
                author_model=defaults.author_model,
                author_version=defaults.author_version,
                checker_model=defaults.checker_model,
                checker_version=defaults.checker_version,
            )
            result = create_canonical_checkpoint(
                args.repo_root,
                args.intake,
                records_root=args.records_root,
                authority_context=context,
            )
    except SingleWriterError as exc:
        raise DriverLockError(str(exc)) from exc
    print(
        json.dumps(
            {
                "verified": True,
                "branch": result.branch,
                "staged_paths": [path.as_posix() for path in result.paths],
                "license_report": result.license_report.to_dict(),
            },
            sort_keys=True,
        )
    )
    return EXIT_OK


def _teardown_command(args: argparse.Namespace) -> int:
    """Lock and remove only active disposable environment/cache/scratch state."""

    runtime = args.repo_root / ".crawl-local"
    before = shutil.disk_usage(runtime if runtime.exists() else args.repo_root).free
    owner = {
        "pid": os.getpid(),
        "run_id": "teardown",
        "target": args.target,
        "created_at": utc_now(),
    }
    with DriverLock(runtime / "locks" / "driver.lock", owner):
        names = ["caches", "scratch"]
        if args.active_env:
            names.append("envs")
        for name in names:
            shutil.rmtree(runtime / name, ignore_errors=True)
    after = shutil.disk_usage(runtime if runtime.exists() else args.repo_root).free
    if args.verify_disk and after < before:
        raise ValueError("teardown did not recover disk capacity")
    print(json.dumps({"teardown": "complete", "bytes_recovered": max(0, after - before)}))
    return EXIT_OK


def _requeue_command(args: argparse.Namespace) -> int:
    """Lock and append one validated grant directly to canonical history."""

    runtime = args.repo_root / ".crawl-local"
    path = (
        args.repo_root
        / "menagerie"
        / "crawler"
        / "records"
        / "operational"
        / "requeue-grants.jsonl"
    )
    owner = {
        "pid": os.getpid(),
        "run_id": "requeue",
        "target": None,
        "created_at": utc_now(),
    }
    with DriverLock(runtime / "locks" / "driver.lock", owner):
        payload = build_canonical_requeue_grant(
            path,
            stable_id=args.stable_id,
            stage=args.stage,
            reason=args.reason,
            attempts=args.grant,
            granted_by=args.granted_by,
            target_intent=args.target_intent,
        )
        append_canonical_requeue_grant(path, payload)
    print(json.dumps(payload, sort_keys=True))
    return EXIT_OK


def _positive_csv(value: str) -> tuple[int, ...]:
    """Parse a comma-separated unique positive integer tuple; empty disables milestones."""

    if not value.strip():
        return ()
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integers") from exc
    if any(item < 1 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("milestones must be positive and unique")
    return parsed


# EN-1: cli planning previously carried its own ``_framework`` derivation (full-word
# markers over "name zoo"), divergent from the driver's ``framework_from_markers`` on
# 105/8,533 catalog and 222/28,482 roster rows -- so the plan census, the
# --allow-lockless justification, and the readiness seal answered a different routing
# question than the leg they were sealing (m10502). There is exactly ONE derivation
# now; anything CLI-side imports it from constants.
