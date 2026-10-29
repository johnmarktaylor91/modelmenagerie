"""Typed identity policy for durable operational events."""

from __future__ import annotations

from dataclasses import dataclass
try:  # python >= 3.11
    from enum import StrEnum
except ImportError:  # python 3.10: the oldest routed-env interpreter (legacy-torch
    # lock resolves py3.10). The worker closure must import there; the constants
    # shim is value-identical (str content == member value), differing only in
    # str()/format() rendering, which nothing identity-bearing uses (DX-3/RE-3).
    from menagerie.crawler.constants import StrEnum  # type: ignore[assignment]
from typing import Mapping

from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import JsonObject


class OperationalEventIdentityClass(StrEnum):
    """Closed replay semantics for operational-event identities."""

    IDEMPOTENT_FIRST_CLOCK = "idempotent-first-clock"
    PER_OCCURRENCE = "per-occurrence"


@dataclass(frozen=True)
class OperationalEventIdentitySpec:
    """Pinned identity semantics for one operational event kind.

    Parameters
    ----------
    identity_class:
        Whether the kind retains the first observation clock for a replayed
        fact or creates one identity per occurrence.
    """

    identity_class: OperationalEventIdentityClass


_A = OperationalEventIdentityClass.IDEMPOTENT_FIRST_CLOCK
_B = OperationalEventIdentityClass.PER_OCCURRENCE

OPERATIONAL_EVENT_IDENTITY_REGISTRY: dict[OperationalEventKind, OperationalEventIdentitySpec] = {
    OperationalEventKind.USAGE_PAUSE: OperationalEventIdentitySpec(_A),
    OperationalEventKind.USAGE_PAUSE_OBSERVATION: OperationalEventIdentitySpec(_B),
    OperationalEventKind.USAGE_RESUME: OperationalEventIdentitySpec(_B),
    OperationalEventKind.WAKEUP: OperationalEventIdentitySpec(_A),
    OperationalEventKind.CHECKPOINT: OperationalEventIdentitySpec(_A),
    OperationalEventKind.CAMPAIGN_HEALTH: OperationalEventIdentitySpec(_B),
    OperationalEventKind.WAKE_NOOP_ALREADY_RUNNING: OperationalEventIdentitySpec(_B),
    OperationalEventKind.CHECKPOINT_REVIEW: OperationalEventIdentitySpec(_A),
    OperationalEventKind.REVIEW_SIGNOFF: OperationalEventIdentitySpec(_A),
    OperationalEventKind.PROGRESS_NOTIFICATION: OperationalEventIdentitySpec(_A),
    OperationalEventKind.NOTIFICATION_DELIVERY: OperationalEventIdentitySpec(_A),
    OperationalEventKind.REQUEUE_GRANT_CONSUMED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.WORKER_LEASE_OPENED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.WORKER_LEASE_STARTED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.WORKER_LEASE_CLOSED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.WORKER_LEASE_REAPED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.ORPHAN_WORKER_RECOVERED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.WAKEUP_INSTALLED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.WAKEUP_REPAIRED: OperationalEventIdentitySpec(_B),
    OperationalEventKind.WAKEUP_FIRED: OperationalEventIdentitySpec(_B),
    OperationalEventKind.WAKEUP_DEACTIVATED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.WAKEUP_HEALTH_DEGRADED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.WAKEUP_FAILED: OperationalEventIdentitySpec(_B),
    OperationalEventKind.CAMPAIGN_COMPLETED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.OPERATOR_CANCELLED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.WORKER_SHUTDOWN_INTERRUPTED: OperationalEventIdentitySpec(_A),
    OperationalEventKind.TERMINAL_UNRECORDABLE: OperationalEventIdentitySpec(_B),
    OperationalEventKind.MODEL_LANE_FAILED: OperationalEventIdentitySpec(_B),
    OperationalEventKind.SCHEDULER_CUTOVER: OperationalEventIdentitySpec(_B),
    OperationalEventKind.SCHEDULER_ROLLBACK: OperationalEventIdentitySpec(_B),
    # WS-1's selection facts (2026-08-16): the census and staleness rows are
    # per-leg observations (occurrence axis = created_at); a generation
    # DISCHARGE is one semantic fact about one generation -- idempotent, so a
    # crash-replayed discharge dedups instead of conflicting.
    OperationalEventKind.WORK_SELECTION_CENSUS: OperationalEventIdentitySpec(_B),
    OperationalEventKind.PROJECTION_STALENESS: OperationalEventIdentitySpec(_B),
    OperationalEventKind.REQUEUE_GENERATION_DISCHARGED: OperationalEventIdentitySpec(_A),
    # B-09 (2026-08-16): the leg-start config record is a per-LEG observation,
    # not a semantic fact about the campaign -- two legs launched with the same
    # flags are two separate attestations, so the occurrence axis is created_at
    # and an identical relaunch must never dedup into the first leg's row.
    OperationalEventKind.LEG_CONFIG: OperationalEventIdentitySpec(_B),
    # W6/OB-M12: one outcome per leg occurrence; a supervised relaunch that
    # ends the same way is a distinct leg and must never dedup into the
    # prior leg's outcome.
    OperationalEventKind.LEG_OUTCOME: OperationalEventIdentitySpec(_B),
    # D-20: the leg-start close preview is a per-leg observation exactly like
    # the selection census; two legs previewing the same debt are two
    # attestations and must never dedup.
    OperationalEventKind.CLOSE_PREVIEW: OperationalEventIdentitySpec(_B),
    # Overhaul RC-1: every crawl-invocation fact is a per-occurrence
    # observation -- two refusals, two canary censuses, or two limit raises
    # at different clocks are distinct attestations and never dedup.
    OperationalEventKind.INVOCATION_REFUSED: OperationalEventIdentitySpec(_B),
    OperationalEventKind.CONFIGURATION_CANARY: OperationalEventIdentitySpec(_B),
    OperationalEventKind.CRAWL_LIMIT: OperationalEventIdentitySpec(_B),
}

if set(OPERATIONAL_EVENT_IDENTITY_REGISTRY) != set(OperationalEventKind):
    raise RuntimeError("operational event identity registry is not exhaustive")


#: CLOSED set of campaign-health OPERATOR DISPOSITIONS whose identity is the
#: incident's semantic fact key, clock excluded. CAMPAIGN_HEALTH stays
#: per-occurrence for observation rows (re-drive counts, lane telemetry), but
#: an operator DISPOSITION over one incident is one semantic fact: keying it
#: on wall-clock made every retried ``ratify``/``restore`` mint a fresh
#: identity and double-append the disposition (H1 F4). Widening this set is a
#: reviewed vocabulary change, exactly like the kind registry above. The
#: members mirror ``reducer.AWARD_RESTORATION_EVENT_DISPOSITION``, the ratify
#: tool's literal, and the models-ledger repair tool's disposition; the
#: strings are spelled here to keep this module free of tool/reducer imports.
CAMPAIGN_HEALTH_IDEMPOTENT_DISPOSITIONS: frozenset[str] = frozenset(
    {
        "award-restored-by-operator",
        "award-ratified",
        "models-ledger-tail-repaired",
    }
)


def _effective_identity_class(
    event_kind: OperationalEventKind, disposition: object
) -> OperationalEventIdentityClass:
    """Return the identity class governing one event, disposition-aware."""

    if (
        event_kind is OperationalEventKind.CAMPAIGN_HEALTH
        and isinstance(disposition, str)
        and disposition in CAMPAIGN_HEALTH_IDEMPOTENT_DISPOSITIONS
    ):
        return OperationalEventIdentityClass.IDEMPOTENT_FIRST_CLOCK
    return OPERATIONAL_EVENT_IDENTITY_REGISTRY[event_kind].identity_class


def mint_operational_event(
    event: Mapping[str, object],
    *,
    identity_payload: Mapping[str, object] | None = None,
) -> JsonObject:
    """Mint one operational event through its registered identity class.

    Idempotent-first-clock kinds require an explicit semantic fact key and
    intentionally exclude ``created_at`` from their ID preimage. Per-occurrence
    kinds derive the ID from the complete event, including their mandatory
    ``created_at`` occurrence axis.

    Parameters
    ----------
    event:
        Complete operational event except for ``event_id`` and append-assigned
        ledger fields.
    identity_payload:
        Explicit semantic key for idempotent-first-clock kinds. Must be absent
        for per-occurrence kinds.

    Returns
    -------
    dict[str, object]
        A defensive event copy carrying the typed derived ``event_id``.

    Raises
    ------
    ValueError
        If the kind is unknown, an ID was already supplied, or the constructor
        arguments contradict the registered identity class.
    """

    minted = dict(event)
    if "event_id" in minted:
        raise ValueError("typed operational-event mint requires event_id to be absent")
    raw_kind = minted.get("event_kind")
    try:
        event_kind = OperationalEventKind(raw_kind)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown operational event kind: {raw_kind!r}") from exc
    details = minted.get("details")
    disposition = details.get("disposition") if isinstance(details, Mapping) else None
    effective_class = _effective_identity_class(event_kind, disposition)
    if effective_class is OperationalEventIdentityClass.IDEMPOTENT_FIRST_CLOCK:
        if not identity_payload:
            raise ValueError(f"{event_kind.value} requires a non-empty idempotent identity payload")
        if (
            event_kind is OperationalEventKind.CAMPAIGN_HEALTH
            and identity_payload.get("disposition") != disposition
        ):
            raise ValueError(
                "idempotent campaign-health identity payload must name its own "
                f"disposition: {disposition!r}"
            )
        minted["event_id"] = idempotent_operational_event_id(event_kind, identity_payload)
        return minted
    else:
        if identity_payload is not None:
            raise ValueError(f"{event_kind.value} forbids an idempotent identity payload")
        created_at = minted.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError(f"{event_kind.value} requires created_at as its occurrence axis")
        preimage: Mapping[str, object] = {
            "event_kind": event_kind.value,
            "occurrence": minted,
        }
    digest = stable_hash(preimage).removeprefix("sha256:")[:24]
    minted["event_id"] = f"{event_kind.value}-{digest}"
    return minted


def idempotent_operational_event_id(
    event_kind: OperationalEventKind,
    identity_payload: Mapping[str, object],
) -> str:
    """Derive a registered idempotent-first-clock event ID.

    Parameters
    ----------
    event_kind:
        Closed operational event kind.
    identity_payload:
        Non-empty semantic fact key that intentionally omits observation time.

    Returns
    -------
    str
        Typed durable operational-event identity.

    Raises
    ------
    ValueError
        If the kind is per-occurrence or the semantic key is empty.
    """

    if not identity_payload:
        raise ValueError(f"{event_kind.value} requires a non-empty identity payload")
    effective_class = _effective_identity_class(event_kind, identity_payload.get("disposition"))
    if effective_class is not OperationalEventIdentityClass.IDEMPOTENT_FIRST_CLOCK:
        raise ValueError(f"{event_kind.value} is registered per-occurrence")
    digest = stable_hash(
        {"event_kind": event_kind.value, "identity": dict(identity_payload)}
    ).removeprefix("sha256:")[:24]
    return f"{event_kind.value}-{digest}"
