"""PD-11's closed leg-exit taxonomy: ONE vocabulary, four consumers (W6/FT-a).

Every leg exit is typed with exactly one :class:`LegExitClass` on the durable
``leg_ended`` row (and the OB-M12 ``leg-outcome`` operational event), so the
four consumers read one truth:

- the W6 supervised restart ladder (restart only on restartable classes,
  never past a standing :class:`StopClass`);
- the OB-M12 leg-outcome census (``count(leg-config) == count(leg-outcome)
  + running``, orphans closed ``orphaned_uncontained``);
- the PM-5 flail counter (``cause_fingerprint`` keys the K=3 identical-cause
  zero-delta fuse that refuses the 4th leg without an operator
  acknowledgment);
- the W11/OC-10 machinery circuit (``fault_machinery`` is its input).

The vocabulary is CLOSED: an exit that fits no typed class is
``fault_uncontained`` -- the catch-all whose census must read ZERO in a
healthy rung -- and only the boot reconciler may synthesize
``orphaned_uncontained``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import stable_hash


class LegExitClass(str, Enum):
    """Closed classification of HOW one leg ended (v1)."""

    #: The leg reached a lawful close (including a licensing close).
    CLOSE_MINTED = "close_minted"
    #: Typed autonomous pause (usage/review); a wake episode or --after-review
    #: owns resume. Preserves "one leg" per B3-XQ-12.
    PAUSED_TYPED = "paused_typed"
    #: Typed operator-initiated stop/halt (hold sentinel, fuse refusal,
    #: drain tree pin, keyboard interrupt).
    OPERATOR_STOP = "operator_stop"
    #: Machinery-initiated typed shutdown (DriverShutdown fact: wall/disk
    #: authority, signals) -- a restart candidate.
    SHUTDOWN_TYPED = "shutdown_typed"
    #: Typed machinery/infra fault reached the exit (storm, retryable
    #: infrastructure, teardown, integration fault); model-blameless per
    #: W5/W11.
    FAULT_MACHINERY = "fault_machinery"
    #: Host-level condition (disk floor, roots, absent secrets, OS errors).
    FAULT_HOST = "fault_host"
    #: ANY untyped exception reaching the exit -- the catch-all. A healthy
    #: rung's census shows zero of these.
    FAULT_UNCONTAINED = "fault_uncontained"
    #: No leg_ended was written; synthesized ONLY by the boot reconciler for
    #: orphans, never by a live leg.
    ORPHANED_UNCONTAINED = "orphaned_uncontained"
    #: Overhaul RC-1: a ``crawl`` invocation ran its scheduled work to
    #: settlement under its limit and exited typed with no close obligation.
    #: Lawful end, never restartable, never fuse food -- the operator raises
    #: the limit to continue.
    WORK_EXHAUSTED = "work_exhausted"


#: Exit classes the W6 supervised restart ladder may restart (bounded, and
#: only when no StopClass holds). ``paused_typed`` routes to the wake path,
#: never the ladder; ``close_minted``/``operator_stop`` end the leg lawfully.
RESTARTABLE_EXIT_CLASSES: frozenset[str] = frozenset(
    {
        LegExitClass.SHUTDOWN_TYPED.value,
        LegExitClass.FAULT_MACHINERY.value,
        LegExitClass.FAULT_HOST.value,
        LegExitClass.FAULT_UNCONTAINED.value,
        LegExitClass.ORPHANED_UNCONTAINED.value,
    }
)

#: Exit classes that count as leg DEATHS for the PM-5 repeat-cause fuse.
FUSE_COUNTED_EXIT_CLASSES: frozenset[str] = frozenset(
    {
        LegExitClass.SHUTDOWN_TYPED.value,
        LegExitClass.FAULT_MACHINERY.value,
        LegExitClass.FAULT_HOST.value,
        LegExitClass.FAULT_UNCONTAINED.value,
        LegExitClass.ORPHANED_UNCONTAINED.value,
    }
)


class StopClass(str, Enum):
    """Closed RESTART-ADMISSION refusals (PD-11 v1).

    Checked by the W6 ladder before ANY supervised restart; distinct from
    exit classes -- a StopClass names why the NEXT leg may not start.
    """

    CALIBRATION_RECEIPT_STALE_OR_ABSENT = "calibration_receipt_stale_or_absent"
    CERTIFICATE_STALE_VS_TREE = "certificate_stale_vs_tree"
    PARTITION_CONTRADICTION = "partition_contradiction"
    UNRECORDABLE_STREAK = "unrecordable_streak"
    DISK_FLOOR_BREACH = "disk_floor_breach"
    ROOTS_UNMOUNTED_OR_READONLY = "roots_unmounted_or_readonly"
    SECRETS_ABSENT_AT_WAKE = "secrets_absent_at_wake"
    REPEATED_CAUSE_LEG_DEATH = "repeated_cause_leg_death"
    PAID_FUSE_HEADROOM_EXHAUSTED = "paid_fuse_headroom_exhausted"
    HOLD_SENTINEL_PRESENT = "hold_sentinel_present"
    # SD-2: outstanding licensing debt / stale rotated axes at leg start,
    # without a durable operator acknowledgment naming the exact census.
    LICENSING_DEBT_UNACKED = "licensing_debt_unacked"


#: Which StopClass predicates THIS tree evaluates mechanically today. The
#: vocabulary is closed and total; the evaluator's coverage is DISCLOSED, not
#: silently partial -- an unevaluated class is visible here and in the lane's
#: honest remainder, never assumed green. (calibration: W2's receipt
#: machinery merges in the same batch; partition: DX-24's shadow-close probe
#: owns in-flight detection; unrecordable streak / paid headroom: their
#: authorities live in-driver and stop the leg themselves before restart
#: adjudication matters.)
STOP_CLASS_EVALUATOR_COVERAGE: dict[str, bool] = {
    StopClass.CALIBRATION_RECEIPT_STALE_OR_ABSENT.value: False,
    StopClass.CERTIFICATE_STALE_VS_TREE.value: True,
    # D-27: the carried-authority preflight evaluates this class's CARRIED
    # subclass mechanically at leg start (voided/released origin rows); the
    # broader in-flight detection stays DX-24's shadow-close probe.
    StopClass.PARTITION_CONTRADICTION.value: True,
    StopClass.UNRECORDABLE_STREAK.value: False,
    StopClass.DISK_FLOOR_BREACH.value: True,
    StopClass.ROOTS_UNMOUNTED_OR_READONLY.value: True,
    StopClass.SECRETS_ABSENT_AT_WAKE.value: True,
    StopClass.REPEATED_CAUSE_LEG_DEATH.value: True,
    StopClass.PAID_FUSE_HEADROOM_EXHAUSTED.value: False,
    StopClass.HOLD_SENTINEL_PRESENT.value: True,
    StopClass.LICENSING_DEBT_UNACKED.value: True,
}

#: PM-5: identical-cause zero-delta leg deaths in a row before the fuse
#: refuses the next leg without an operator acknowledgment. Matches the
#: project's three-strikes anti-flail rule.
REPEAT_CAUSE_FUSE_LIMIT: int = 3


def cause_fingerprint(
    exit_class: str,
    *,
    exception_types: Sequence[str] = (),
    evidence: str = "",
) -> str:
    """Derive PM-5's stable cause identity for one leg exit.

    Normalized over the exit class, the typed exception chain (TYPE names
    only -- message text is volatile and would defeat the fuse), and a short
    stable evidence token (a StopClass, a pause reason, a completion status).

    Returns
    -------
    str
        ``sha256:`` content address of the normalized cause.
    """

    return stable_hash(
        {
            "cause_version": "menagerie-leg-cause-v1",
            "exit_class": str(exit_class),
            "exception_types": [str(name) for name in exception_types],
            "evidence": str(evidence),
        }
    )


def classify_result_status(status: str) -> str:
    """Classify a DriverResult status string into the closed exit vocabulary."""

    value = str(status)
    if value.startswith("paused:"):
        return LegExitClass.PAUSED_TYPED.value
    if value == "interrupted:shutdown":
        return LegExitClass.SHUTDOWN_TYPED.value
    if value == "work_exhausted":
        return LegExitClass.WORK_EXHAUSTED.value
    if value.startswith(
        (
            "complete",
            "phase-complete",
            "terminal-partition-complete",
            "phase-terminal-partition-complete",
        )
    ):
        return LegExitClass.CLOSE_MINTED.value
    # A status outside the closed completion/pause/shutdown families is an
    # uncontained shape by definition -- the census must see it.
    return LegExitClass.FAULT_UNCONTAINED.value


def exception_type_chain(exc: BaseException) -> tuple[str, ...]:
    """Return the qualified type names along one exception chain."""

    names: list[str] = []
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(f"{type(current).__module__}.{type(current).__qualname__}")
        current = current.__cause__ or current.__context__
    return tuple(names)


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """Classify one exception reaching the leg exit.

    Returns
    -------
    tuple[str, str]
        ``(exit_class, evidence)`` where evidence is the stable token the
        cause fingerprint binds (a stop class for typed leg-start refusals,
        empty otherwise).
    """

    from menagerie.crawler.driver_contracts import (
        AuthorCampaignTeardown,
        DriverShutdown,
        LegStartRefused,
        RetryableOperatorError,
    )

    if isinstance(exc, LegStartRefused):
        mapping = {
            StopClass.SECRETS_ABSENT_AT_WAKE.value: LegExitClass.FAULT_HOST.value,
            StopClass.DISK_FLOOR_BREACH.value: LegExitClass.FAULT_HOST.value,
            StopClass.ROOTS_UNMOUNTED_OR_READONLY.value: LegExitClass.FAULT_HOST.value,
        }
        exit_class = mapping.get(exc.stop_class, LegExitClass.OPERATOR_STOP.value)
        return exit_class, exc.stop_class
    if isinstance(exc, DriverShutdown):
        return LegExitClass.SHUTDOWN_TYPED.value, ""
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return LegExitClass.OPERATOR_STOP.value, ""
    if isinstance(exc, AuthorCampaignTeardown):
        return LegExitClass.FAULT_MACHINERY.value, ""
    if isinstance(exc, RetryableOperatorError):
        return LegExitClass.FAULT_MACHINERY.value, ""
    if isinstance(exc, OSError):
        return LegExitClass.FAULT_HOST.value, ""
    return LegExitClass.FAULT_UNCONTAINED.value, ""


def repeat_cause_fuse_tripped(
    leg_events: Sequence[Mapping[str, Any]],
    *,
    limit: int = REPEAT_CAUSE_FUSE_LIMIT,
) -> Optional[str]:
    """Return the tripped cause fingerprint, if the PM-5 fuse holds.

    Walks the durable ``leg_ended`` rows in ledger order: a trailing run of
    ``limit`` identical-cause, ZERO-terminal-delta leg deaths trips the fuse.
    Any lawful end (close, pause, operator stop), any progress
    (``terminal_delta`` > 0), or a different cause resets the streak --
    restarts, new legs, and tree changes alone never do.

    Parameters
    ----------
    leg_events:
        Validated leg-event rows in ledger order.
    limit:
        Consecutive identical-cause zero-delta deaths that trip.

    Returns
    -------
    str | None
        The repeated ``cause_fingerprint`` when tripped, else ``None``.
    """

    streak_fingerprint: Optional[str] = None
    streak = 0
    for row in leg_events:
        if row.get("event") != "leg_ended":
            continue
        exit_class = str(row.get("exit_class", ""))
        delta = row.get("terminal_delta")
        fingerprint = str(row.get("cause_fingerprint", ""))
        zero_delta = isinstance(delta, int) and delta == 0
        if exit_class not in FUSE_COUNTED_EXIT_CLASSES or not zero_delta:
            streak_fingerprint, streak = None, 0
            continue
        if fingerprint and fingerprint == streak_fingerprint:
            streak += 1
        else:
            streak_fingerprint, streak = fingerprint or None, 1 if fingerprint else 0
    if streak >= limit and streak_fingerprint:
        return streak_fingerprint
    return None


def leg_outcome_census(
    operational_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """OB-M12: ``count(leg-config) == count(leg-outcome) + running``.

    One-ledger census over the operational events: every leg that attested a
    config owes exactly one typed outcome (the boot reconciler closes orphans
    ``orphaned_uncontained``), so ``running`` is derivable and the
    ``fault_uncontained`` count is the health tripwire (zero in a healthy
    rung).

    Returns
    -------
    dict[str, Any]
        ``leg_config``, ``leg_outcome``, ``running``, and ``by_exit_class``.
    """

    configs = 0
    outcomes = 0
    by_exit_class: dict[str, int] = {}
    for event in operational_events:
        kind = event.get("event_kind")
        if kind == "leg-config":
            configs += 1
        elif kind == "leg-outcome":
            outcomes += 1
            details = event.get("details")
            if isinstance(details, Mapping):
                exit_class = str(details.get("exit_class", ""))
                by_exit_class[exit_class] = by_exit_class.get(exit_class, 0) + 1
    return {
        "leg_config": configs,
        "leg_outcome": outcomes,
        "running": configs - outcomes,
        "by_exit_class": dict(sorted(by_exit_class.items())),
    }


def evaluate_stop_classes(
    records_root: Any,
    repo_root: Any,
    *,
    now: str,
) -> tuple[dict[str, Any], ...]:
    """Evaluate every mechanically-evaluable StopClass (restart admission).

    Fail-closed per class where its authority is readable; a class whose
    authority is not yet wired reads as NOT evaluated (see
    :data:`STOP_CLASS_EVALUATOR_COVERAGE`), never as green.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One finding per HOLDING class: ``{"stop_class", "detail"}``.
    """

    import os
    import shutil
    from pathlib import Path

    findings: list[dict[str, Any]] = []
    records = Path(records_root)

    from menagerie.crawler.pause_authority import evaluate_sprint_pause

    hold = evaluate_sprint_pause(records, now=now)
    if hold.held:
        findings.append(
            {
                "stop_class": StopClass.HOLD_SENTINEL_PRESENT.value,
                "detail": f"{hold.decision}: {hold.detail or 'operator hold'}",
            }
        )

    from menagerie.crawler.env_census import missing_wake_secrets

    missing = missing_wake_secrets()
    if missing:
        findings.append(
            {
                "stop_class": StopClass.SECRETS_ABSENT_AT_WAKE.value,
                "detail": f"absent: {', '.join(missing)}",
            }
        )

    probe = records
    while not probe.exists() and probe != probe.parent:
        # A not-yet-materialized records root still lives on SOME filesystem;
        # measure the one it would land on rather than refusing on absence.
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        free = -1
    floor = 5 * 1024**3
    if free < floor:
        findings.append(
            {
                "stop_class": StopClass.DISK_FLOOR_BREACH.value,
                "detail": f"free={free} floor={floor}",
            }
        )

    if records.exists() and (not records.is_dir() or not os.access(records, os.W_OK)):
        # An ABSENT records root is a fresh campaign (the first leg
        # materializes it); a PRESENT-but-unwritable one is the R5 class.
        findings.append(
            {
                "stop_class": StopClass.ROOTS_UNMOUNTED_OR_READONLY.value,
                "detail": f"records root not writable: {records}",
            }
        )

    from menagerie.crawler.leg_generation import load_leg_events

    try:
        events = load_leg_events(records)
    except Exception as exc:  # noqa: BLE001 -- an unreadable ledger holds
        findings.append(
            {
                "stop_class": StopClass.REPEATED_CAUSE_LEG_DEATH.value,
                "detail": f"leg-events unreadable (holding closed): {exc}",
            }
        )
    else:
        fingerprint = repeat_cause_fuse_tripped(events)
        if fingerprint is not None:
            findings.append(
                {
                    "stop_class": StopClass.REPEATED_CAUSE_LEG_DEATH.value,
                    "detail": f"repeat-cause fuse tripped: {fingerprint}",
                }
            )

    return tuple(findings)
