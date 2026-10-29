"""The ONE canonical requeue grant-identity preimage (docket C-GRANT-ID, T1-08).

Every authority that mints or verifies a requeue ``grant_id`` — the checkpoint
builder (:func:`menagerie.crawler.checkpoint.build_canonical_requeue_grant`),
driver rehydration (``_validated_requeue_grants``), and the reducer's
consumption-side lineage validator (``_validate_persisted_requeue_lineage``) —
imports its preimage from this module. Grant identity forked here once: the
builder and driver folded ``target_intent`` into the hash while the reducer
hashed six fields without it, so every routed ``--intent`` grant was refused at
the canonical append AFTER its consumption event was durable (spend burned,
grant dangling active forever). Never re-spell these dicts inline.

Two preimages exist and only two:

- The tool preimage, for rows minted by the requeue tool (marked by
  ``new_work_generation``): generation, stable_id, stage, reason, attempts,
  granted_by, and — only when present — target_intent.
- The legacy preimage, for pre-tool handwritten rows: stable_id, stage, reason,
  and attempts under the historical ``grant`` key. It is intent-blind by
  construction, so a legacy row carrying ``target_intent`` has no single
  canonical identity and MUST refuse typed everywhere; silent acceptance would
  let routing ride a hash that never committed to it.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from menagerie.crawler.identity import stable_hash

__all__ = [
    "GrantIdentityError",
    "LEGACY_AWARD_REFUSAL",
    "LEGACY_CAPABILITY_REFUSAL",
    "LEGACY_INTENT_REFUSAL",
    "expected_grant_id",
    "legacy_grant_identity",
    "requeue_adjudication_key",
    "tool_grant_identity",
]

#: The single spelling of the legacy two-identity refusal, shared so driver
#: and reducer diagnostics stay greppable as one class.
LEGACY_INTENT_REFUSAL = "legacy requeue grant cannot carry an environment intent"

#: Same shape for the AW-1 award-naming field: the legacy preimage is
#: award-blind by construction, so a legacy row carrying
#: ``blocked_award_revision`` has no single canonical identity and must
#: refuse typed everywhere.
LEGACY_AWARD_REFUSAL = "legacy requeue grant cannot name a blocked award revision"

#: The same refusal for a typed capability: the legacy preimage cannot cover it, so a
#: hand-written legacy row naming one is refused rather than honoured unauthenticated.
LEGACY_CAPABILITY_REFUSAL = "legacy requeue grant cannot carry a typed capability"


class GrantIdentityError(ValueError):
    """A persisted grant row cannot carry one canonical identity."""


def tool_grant_identity(
    *,
    generation: Any,
    stable_id: Any,
    stage: Any,
    reason: Any,
    attempts: Any,
    granted_by: Any,
    target_intent: Optional[Any] = None,
    blocked_award_revision: Optional[Any] = None,
    capability: Optional[Any] = None,
) -> str:
    """Return the canonical identity of one tool-minted requeue grant.

    Parameters
    ----------
    generation, stable_id, stage, reason, attempts, granted_by:
        Operator authorization facts, exactly as persisted on the grant row.
    target_intent:
        Optional routed environment correction; folded into the identity only
        when present, keeping pre-routing tool grants' hashes stable.
    blocked_award_revision:
        Optional exact ``runs`` record revision this grant authorizes
        superseding (AW-1). Retry permission and downgrade permission are
        separate authorities: an ordinary grant authorizes WORK, and only a
        grant naming the exact award revision releases the award-custody
        tripwire. Folded into the identity only when present, keeping every
        existing grant hash stable -- and therefore no legacy or pre-AW-1
        grant can ever retroactively authorize a downgrade.
    capability:
        Optional typed capability the grant authorizes (WS-2:
        ``host-capability-routing`` is the only member, and the only thing that lets a
        grant re-open a host-capability head). Folded into the identity ONLY when
        present, so existing grants' hashes are untouched -- and folded in at all
        because the field STEERS that re-opening: outside the preimage, a capability
        could be appended to an already-authorized generic grant without changing its
        ``grant_id``, which is an authorization upgrade nobody authorized.

    Returns
    -------
    str
        The ``grant_id`` hash every authority must agree on.
    """

    identity: dict[str, Any] = {
        "generation": generation,
        "stable_id": stable_id,
        "stage": stage,
        "reason": reason,
        "attempts": attempts,
        "granted_by": granted_by,
    }
    if target_intent is not None:
        identity["target_intent"] = target_intent
    if blocked_award_revision is not None:
        identity["blocked_award_revision"] = blocked_award_revision
    if capability is not None:
        identity["capability"] = capability
    return stable_hash(identity)


def requeue_adjudication_key(
    *,
    stable_id: Any,
    stage: Any,
    reason: Any,
    target_intent: Optional[Any] = None,
    blocked_award_revision: Optional[Any] = None,
    capability: Optional[Any] = None,
) -> str:
    """Return the identity of one adjudicated requeue DECISION (W31 schema v2).

    The adjudication key names WHAT the operator decided -- which member, which
    failure stage, why, and any typed routing/award/capability scope -- and
    deliberately excludes WHO recorded it (``granted_by``), the allowance size
    (``attempts``), and the per-row ``generation``. Two live grants sharing this
    key are therefore two mints of ONE decision (the storm-grant overlap class
    the logical-twin check misses whenever attempts or attribution differ), and
    the append door refuses the second while the first is live. Optional scopes
    fold in only when present, mirroring ``tool_grant_identity``: a capability
    grant is a DIFFERENT authorization from a generic grant, and pre-existing
    keys never move when a new optional axis ships.

    Parameters
    ----------
    stable_id, stage, reason:
        The adjudicated member, failure stage, and operator rationale.
    target_intent, blocked_award_revision, capability:
        Optional typed decision scopes, exactly as persisted on the grant row.

    Returns
    -------
    str
        The ``adjudication_key`` hash every authority recomputes on read.
    """

    identity: dict[str, Any] = {
        "kind": "requeue-grant-adjudication",
        "stable_id": stable_id,
        "stage": stage,
        "reason": reason,
    }
    if target_intent is not None:
        identity["target_intent"] = target_intent
    if blocked_award_revision is not None:
        identity["blocked_award_revision"] = blocked_award_revision
    if capability is not None:
        identity["capability"] = capability
    return stable_hash(identity)


def legacy_grant_identity(
    *,
    stable_id: Any,
    stage: Any,
    reason: Any,
    attempts: Any,
) -> str:
    """Return the canonical identity of one pre-tool handwritten grant.

    Parameters
    ----------
    stable_id, stage, reason, attempts:
        Grant facts; ``attempts`` hashes under the historical ``grant`` key.

    Returns
    -------
    str
        The legacy ``grant_id`` hash.
    """

    return stable_hash(
        {
            "stable_id": stable_id,
            "stage": stage,
            "reason": reason,
            "grant": attempts,
        }
    )


def expected_grant_id(row: Mapping[str, Any]) -> str:
    """Return the single canonical identity for one persisted grant row.

    Parameters
    ----------
    row:
        Persisted grant row; ``new_work_generation`` marks the tool schema.

    Returns
    -------
    str
        The ``grant_id`` the row must carry.

    Raises
    ------
    GrantIdentityError
        If the row is a legacy two-identity row: the legacy preimage is
        intent-blind, so a legacy row carrying ``target_intent`` must refuse
        typed rather than validate as if the routing did not exist. The same holds for a
        legacy row carrying a typed ``capability`` -- a hand-written row cannot smuggle
        an authorization the identity does not cover.
    """

    target_intent = row.get("target_intent")
    blocked_award_revision = row.get("blocked_award_revision")
    capability = row.get("capability")
    if "new_work_generation" in row:
        return tool_grant_identity(
            generation=row.get("new_work_generation"),
            stable_id=row.get("stable_id"),
            stage=row.get("stage"),
            reason=row.get("reason"),
            attempts=row.get("attempts"),
            granted_by=row.get("granted_by"),
            target_intent=target_intent,
            blocked_award_revision=blocked_award_revision,
            capability=capability,
        )
    if target_intent is not None:
        raise GrantIdentityError(LEGACY_INTENT_REFUSAL)
    if blocked_award_revision is not None:
        raise GrantIdentityError(LEGACY_AWARD_REFUSAL)
    if capability is not None:
        raise GrantIdentityError(LEGACY_CAPABILITY_REFUSAL)
    return legacy_grant_identity(
        stable_id=row.get("stable_id"),
        stage=row.get("stage"),
        reason=row.get("reason"),
        attempts=row.get("attempts"),
    )
