"""Chargeable author-session accounting derived from append-only history.

Docket C-AUTHOR-SPEND (T2-07) and the T1-13 allowance half. The declared
``author_sessions_max`` was enforced only by in-memory per-leg counters that
every rehydration reset, and the canonical ``budget.author_sessions_used``
field was hardcoded to 0/1 -- the pilot recorded 31 work generations over the
3-session cap (worst: eleven staged results on one m8650 work_id) while every
head read ``used=1``. BL-11 corrected the unit: the provider bills a bought
session whether or not it publishes. Every pre-dispatch admission now emits
``author-session-bought`` on the operational ledger. Historical
``staged-private`` artifact rows remain a fallback floor for legs predating
that event; they are never allowed to lower the bought-session truth.

One work generation's allowance is :data:`AUTHOR_SESSIONS_MAX`; a consumed
requeue grant EXTENDS its generation by the grant's ``attempts``
(``WorkItem.granted_attempts``). Exhaustion raises the existing
:class:`~menagerie.crawler.driver_contracts.AuthorEffortCapExceeded` lane, so
it terminalizes as ``failed:author`` / ``effort-exhausted:author-sessions``
for exactly this model, like every other cap.

The census and valve share one money truth: every distinct durable buy for the
exact ``(stable_id, work_id)`` remains charged to that work generation. A
disposable cache reset, stale cache, or supersession observation cannot mint a
new allowance. Only a new durable work generation or an operator grant can
provide new author-session capacity.

UNIT (PSLRETRY round 2): the allowance unit here is the ATTEMPT -- the
historical "author session".  One attempt now legitimately buys up to two paid
GUARD sessions (source-request, then author), each its own paid-ledger
reservation; those reservations share one durable ``attempt_group`` identity,
and every counter in this module counts attempt groups, never raw guard
sessions.  Raw session truth stays on the paid ledger's own QUOTA projection.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Optional

from menagerie.crawler.artifact_transactions import ArtifactEventKind
from menagerie.crawler.constants import AUTHOR_SESSIONS_MAX
from menagerie.crawler.driver_contracts import AuthorEffortCapExceeded, WorkItem
from menagerie.crawler.paid_session_accounting import (
    ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES,
    settled_attempt_weight,
    settled_session_weight,
)

__all__ = [
    "author_sessions_exhaustion_dimension",
    "author_sessions_used_for_allowance",
    "author_session_allowance",
    "author_spend_census",
    "chargeable_author_sessions",
    "enforce_author_session_allowance",
    "retired_author_result_ids",
]


def _chargeable_attempt_group_ids(
    materialized: tuple[Mapping[str, Any], ...], stable_id: str, work_id: str
) -> set[str]:
    """Return chargeable ATTEMPT-GROUP ids from merged paid-ledger rows.

    One author attempt is one attempt GROUP of guard-session reservations: the
    head (``attempt_group`` absent/null) plus its joins.  The historical
    "author session" unit IS the attempt, so a merged event stream counts
    groups, never raw guard sessions -- a two-guard attempt would otherwise
    read as two allowance units.  Group members settle strictly in sequence,
    so a member settled at zero attempt weight (provider decline, or a closed
    machine infrastructure outcome the driver derived from the harness
    envelope) is the attempt's TERMINAL disposition and refunds the whole
    group, matching the paid authority's own refund rule; an unsettled member
    stays chargeable in the conservative direction.
    """

    settled_attempt_weights = {
        str(event["spend_id"]): settled_attempt_weight(event)
        for event in materialized
        if event.get("event") == "session_settled"
        and isinstance(event.get("spend_id"), str)
    }
    members: dict[str, list[str]] = defaultdict(list)
    for event in materialized:
        if event.get("event") != "session_reserved" or event.get("lane") != "author":
            continue
        axis = event.get("axis")
        if not isinstance(axis, Mapping) or not any(
            isinstance(member, Mapping)
            and member.get("stable_id") == stable_id
            and member.get("work_id") == work_id
            for member in axis.get("work_generations", ())
        ):
            continue
        spend_id = str(event["spend_id"])
        head = event.get("attempt_group")
        group = str(head) if isinstance(head, str) and head else spend_id
        members[group].append(spend_id)
    return {
        f"attempt-group:{group}"
        for group, spend_ids in members.items()
        if all(settled_attempt_weights.get(spend_id, 1) == 1 for spend_id in spend_ids)
    }


def _recorded_author_session_total(
    events: Iterable[Mapping[str, Any]], stable_id: str, work_id: str
) -> int:
    """Return the lifetime ATTEMPT total asserted by durable buy evidence.

    Legacy ``author-session-bought`` operational rows predate the guard-session
    split, so each is one attempt; merged paid-ledger reservation rows count as
    attempt GROUPS through their durable ``attempt_group`` identity.
    """

    materialized = tuple(events)
    event_ids = {
        str(event.get("event_id") or event["details"].get("author_session_id"))
        for event in materialized
        if isinstance(event.get("details"), Mapping)
        and event["details"].get("disposition") == "author-session-bought"
        and event["details"].get("stable_id") == stable_id
        and event["details"].get("work_id") == work_id
    }
    event_ids.update(_chargeable_attempt_group_ids(materialized, stable_id, work_id))
    ordinals = [
        int(event["details"]["session_ordinal"])
        for event in materialized
        if isinstance(event.get("details"), Mapping)
        and event["details"].get("disposition") == "author-session-bought"
        and event["details"].get("stable_id") == stable_id
        and event["details"].get("work_id") == work_id
        and isinstance(event["details"].get("session_ordinal"), int)
        and not isinstance(event["details"].get("session_ordinal"), bool)
    ]
    return max(len(event_ids), max(ordinals, default=0))


def chargeable_author_sessions(
    events: Iterable[Mapping[str, Any]],
    stable_id: str,
    work_id: str,
    *,
    paid_attempts: int = 0,
) -> int:
    """Return the chargeable author ATTEMPTS one work generation has bought.

    The unit is the attempt (the historical "author session"): one staged
    result, one legacy bought event, or one paid attempt GROUP -- never a raw
    guard session, of which one attempt legitimately buys up to two.  This is
    the same unit as ``author_session_allowance``, so canonical
    ``budget.author_sessions_used`` and ``author_sessions_max`` stay
    comparable.

    Parameters
    ----------
    events:
        Artifact-ledger events in append order.
    stable_id, work_id:
        Exact model and work-generation identities.
    paid_attempts:
        O(1) chargeable attempt-group projection from the paid-session
        authority (``AccountingState.quota_attempt_groups``).

    Returns
    -------
    int
        Replay-derived attempt spend, independent of any in-memory counter.
    """

    if isinstance(paid_attempts, bool) or not isinstance(paid_attempts, int) or paid_attempts < 0:
        raise ValueError("paid_attempts must be a non-negative integer")
    materialized = tuple(events)
    staged = {
        str(event.get("author_result_id"))
        for event in materialized
        if event.get("event_kind") == ArtifactEventKind.STAGED_PRIVATE.value
        and event.get("stable_id") == stable_id
        and event.get("work_id") == work_id
    }
    return max(
        len(staged),
        _recorded_author_session_total(materialized, stable_id, work_id) + paid_attempts,
    )


def author_sessions_used_for_allowance(
    events: Iterable[Mapping[str, Any]],
    item: WorkItem,
    *,
    operational_events: Iterable[Mapping[str, Any]] = (),
) -> int:
    """Return durable per-generation spend using the valve's sole arithmetic.

    Parameters
    ----------
    events:
        Combined artifact and operational history.
    item:
        Exact scheduled work generation.
    operational_events:
        Retained for call-site compatibility. Cache-discard observations never
        subtract durable spend from the same work generation.

    Returns
    -------
    int
        Sessions already consumed by the durable work generation.
    """

    del operational_events
    return chargeable_author_sessions(events, item.stable_id, item.active_work_id)


def retired_author_result_ids(
    operational_events: Iterable[Mapping[str, Any]], stable_id: str, work_id: str
) -> frozenset[str]:
    """Return staged author results a durable supersession discard has retired.

    A supersession discard is the driver-attested lineage break: the resume
    found its author-result cache anchored to bytes the artifact ledger had
    already superseded, appended one ``author-cache-discarded`` operational
    event naming the EXACT ``superseded_author_result_ids`` it retired, and
    re-authored under current authority. Only rows carrying that explicit list
    retire anything -- a discard for any other reason (corrupt cache, ambiguous
    authority) names none and keeps full TW-H charging, and historical rows
    without the field are likewise inert. The retired identities stay in every
    census; they only stop counting against the CURRENT lineage's valve.

    Parameters
    ----------
    operational_events:
        Operational-ledger rows in append order.
    stable_id, work_id:
        Exact model and work-generation identities.

    Returns
    -------
    frozenset[str]
        Union of every retired ``author_result_id`` for the exact pair.
    """

    retired: set[str] = set()
    for event in operational_events:
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        if (
            details.get("disposition") != "author-cache-discarded"
            or details.get("stable_id") != stable_id
            or details.get("work_id") != work_id
        ):
            continue
        superseded = details.get("superseded_author_result_ids")
        if isinstance(superseded, (list, tuple)):
            retired.update(str(result_id) for result_id in superseded)
    return frozenset(retired)


def author_sessions_exhaustion_dimension(
    events: Iterable[Mapping[str, Any]], stable_id: str, work_id: str
) -> str:
    """Return the honest closed effort dimension for one exhaustion terminal.

    A sustained provider storm burns a work generation's session allowance on
    infrastructure noise; terminalizing that under the same
    ``effort-exhausted:author-sessions`` as genuine effort exhaustion pools
    restorable storm damage with real exhaustion in every triage table. The
    split is replay-derived from settlement evidence ONLY: over this exact
    work generation's author-lane reservations, a consumed (provider-charged)
    session is machine-classified infrastructure exactly when its settlement's
    attempt weight is zero while its session weight is one AND its local
    outcome is in the closed ``ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES``
    machine vocabulary, never any producer-supplied field. The
    ``author-blocked-prerequisite`` refund (AL-6) also zeroes the attempt
    weight but is an AUTHOR VERDICT, not machine weather, so it never counts
    toward the storm majority. A strict MAJORITY of consumed sessions with the
    infrastructure
    classification reads ``provider-storm``; everything else -- including an
    unreadable settlement, an unsettled reservation, or zero evidence -- fails
    toward plain ``author-sessions``, so uncertainty can never upgrade a
    terminal into the restorable class.

    Parameters
    ----------
    events:
        Merged event stream including paid-ledger reservation and settlement
        rows, exactly as ``enforce_author_session_allowance`` receives it.
    stable_id, work_id:
        Exact model and work-generation identities.

    Returns
    -------
    str
        ``"provider-storm"`` or ``"author-sessions"``; with the ``author``
        stage these spell the closed ``effort-exhausted:<dimension>`` reasons.
    """

    materialized = tuple(events)
    spend_ids: set[str] = set()
    for event in materialized:
        if event.get("event") != "session_reserved" or event.get("lane") != "author":
            continue
        axis = event.get("axis")
        if not isinstance(axis, Mapping) or not any(
            isinstance(member, Mapping)
            and member.get("stable_id") == stable_id
            and member.get("work_id") == work_id
            for member in axis.get("work_generations", ())
        ):
            continue
        if isinstance(event.get("spend_id"), str):
            spend_ids.add(str(event["spend_id"]))
    consumed = 0
    infrastructure = 0
    for event in materialized:
        if event.get("event") != "session_settled":
            continue
        spend_id = event.get("spend_id")
        if not isinstance(spend_id, str) or spend_id not in spend_ids:
            continue
        try:
            session_weight = settled_session_weight(event)
            attempt_weight = settled_attempt_weight(event)
        except (ValueError, KeyError, TypeError):
            # An unreadable settlement counts as consumed, never as storm
            # evidence: producer influence fails toward the plain reason.
            consumed += 1
            continue
        if session_weight != 1:
            # Provider-declined: never charged, never part of what the fuse
            # consumed, so it neither supports nor dilutes the storm reading.
            continue
        consumed += 1
        if (
            attempt_weight == 0
            and event.get("local_outcome") in ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES
        ):
            infrastructure += 1
    if consumed and infrastructure * 2 > consumed:
        return "provider-storm"
    return "author-sessions"


def author_session_allowance(granted_attempts: Optional[int]) -> int:
    """Return one work generation's total author-session allowance.

    Parameters
    ----------
    granted_attempts:
        The consumed requeue grant's ``attempts`` extension for the ACTIVE
        granted generation, or ``None`` for organic work.

    Returns
    -------
    int
        Base allowance plus the operator-granted extension.
    """

    return AUTHOR_SESSIONS_MAX + (granted_attempts or 0)


def enforce_author_session_allowance(
    events: Iterable[Mapping[str, Any]],
    item: WorkItem,
    *,
    operational_events: Iterable[Mapping[str, Any]] = (),
    staging_author_result_id: Optional[str] = None,
    admitted_in_flight: bool = False,
) -> None:
    """Refuse to buy a session past the work generation's derived allowance.

    Called at every author buy site BEFORE dispatch, so a rehydrated leg can
    never re-buy spend its predecessors already staged, and again at the ONE
    staging chokepoint (``CrawlerDriver._stage_author_result``) BEFORE the
    ``staged-private`` event lands -- the m4851 overshoot (6 staged of a
    3-session allowance) came from bounded repair loops that staged without
    re-consulting this valve after their buy. Cache-discard and supersession
    rows never retire money truth: a stale or cleared cache retains the same
    generation's durable spend and cannot buy a fresh allowance.

    Parameters
    ----------
    events:
        Artifact-ledger events in append order.
    item:
        Exact scheduled work generation about to buy a session.
    operational_events:
        Operational-ledger rows in append order. Retained for compatibility;
        producer-influenced cache state cannot lower the allowance count.
    staging_author_result_id:
        The exact result identity ABOUT to stage, when called from the staging
        chokepoint. A rehydration replay that re-stages an id the ledger
        already holds buys nothing under the distinct-id counting rule and is
        never refused; only a fresh id charges.
    admitted_in_flight:
        True when the pre-dispatch admission valve has ALREADY charged the
        session this call guards (``_admit_author_spend`` enforces and then
        records the buy before the lane body runs). The recorded in-flight
        session then counts inside ``used``, so re-checking ``used >=
        allowance`` here would refuse the exact final session the allowance
        grants: the rung-12 repair trajectory bought its 3rd (last) repair
        session at the admission valve and this re-check terminalized the
        model ``failed:author`` before the metadata gate could adjudicate its
        review-held terminal. With the flag, this call only refuses a state
        that EXCEEDS the allowance -- an unadmitted overshoot stays loud.

    Raises
    ------
    AuthorEffortCapExceeded
        When the derived chargeable count has reached the allowance. Routes
        model-local as ``failed:author`` / ``effort-exhausted:author-sessions``.
    """

    materialized = tuple(events)
    staged = {
        str(event.get("author_result_id"))
        for event in materialized
        if event.get("event_kind") == ArtifactEventKind.STAGED_PRIVATE.value
        and event.get("stable_id") == item.stable_id
        and event.get("work_id") == item.active_work_id
    }
    if staging_author_result_id is not None and staging_author_result_id in staged:
        return
    if staging_author_result_id is not None:
        lifetime_total = chargeable_author_sessions(
            materialized, item.stable_id, item.active_work_id
        )
        if lifetime_total > len(staged):
            # The pre-dispatch valve already charged this exact in-flight
            # session. Staging its result consumes no additional allowance;
            # only an unbacked fresh staging at the cap is an overshoot.
            return
    used = author_sessions_used_for_allowance(
        materialized,
        item,
        operational_events=operational_events,
    )
    allowance = author_session_allowance(item.granted_attempts)
    if admitted_in_flight:
        # The admission valve already enforced and charged the guarded
        # session; only a count STRICTLY past the allowance is an overshoot.
        if used > allowance:
            raise AuthorEffortCapExceeded(
                f"work generation {item.active_work_id} for {item.stable_id} already "
                f"bought {used} author sessions of its {allowance}-session allowance",
                stage="author",
                dimension=author_sessions_exhaustion_dimension(
                    materialized, item.stable_id, item.active_work_id
                ),
            )
        return
    if used >= allowance:
        raise AuthorEffortCapExceeded(
            f"work generation {item.active_work_id} for {item.stable_id} already "
            f"bought {used} author sessions of its {allowance}-session allowance",
            stage="author",
            dimension=author_sessions_exhaustion_dimension(
                materialized, item.stable_id, item.active_work_id
            ),
        )


def author_spend_census(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return replay-derived author spend over artifact and operational rows.

    The reader-side pairing for the checker-spend census: recomputable over any
    historical shard (it is how the pilot's 31 over-cap generations were
    adjudicated), and consumed by the throughput report so a leg's author and
    checker spend read side by side.

    Parameters
    ----------
    events:
        Combined artifact and operational events in append order.

    Returns
    -------
    dict[str, Any]
        Work-generation count, staged/bought/chargeable session totals, and
        generations above the base allowance. Grant extensions remain
        per-generation facts this census cannot infer.
    """

    materialized = tuple(events)
    sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    keys: set[tuple[str, str]] = set()
    for event in materialized:
        if event.get("event") == "session_reserved" and event.get("lane") == "author":
            axis = event.get("axis")
            if isinstance(axis, Mapping):
                for member in axis.get("work_generations", ()):
                    if isinstance(member, Mapping):
                        keys.add((str(member.get("stable_id")), str(member.get("work_id"))))
            continue
        if event.get("event_kind") != ArtifactEventKind.STAGED_PRIVATE.value:
            details = event.get("details")
            if isinstance(details, Mapping) and details.get("disposition") == (
                "author-session-bought"
            ):
                keys.add((str(details.get("stable_id")), str(details.get("work_id"))))
            continue
        key = (str(event.get("stable_id")), str(event.get("work_id")))
        keys.add(key)
        sessions[key].add(str(event.get("author_result_id")))
    counts = {
        key: chargeable_author_sessions(materialized, key[0], key[1]) for key in keys
    }
    bought_counts = {
        key: len(
            {
                str(event.get("event_id") or event["details"].get("author_session_id"))
                for event in materialized
                if isinstance(event.get("details"), Mapping)
                and event["details"].get("disposition") == "author-session-bought"
                and event["details"].get("stable_id") == key[0]
                and event["details"].get("work_id") == key[1]
            }
            # Paid-ledger rows count as attempt GROUPS, the same unit as the
            # legacy bought events (one attempt each), never raw guard sessions.
            | _chargeable_attempt_group_ids(materialized, key[0], key[1])
        )
        for key in keys
    }
    over = {
        f"{stable_id}/{work_id}": count
        for (stable_id, work_id), count in sorted(counts.items())
        if count > AUTHOR_SESSIONS_MAX
    }
    return {
        "work_generations": len(keys),
        "sessions_staged": sum(len(result_ids) for result_ids in sessions.values()),
        "sessions_bought": sum(bought_counts.values()),
        "sessions_chargeable": sum(counts.values()),
        "generations_over_allowance": len(over),
        "over_allowance": over,
    }
