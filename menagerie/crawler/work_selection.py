"""THE one work-selection policy (WS-1, AUDIT100 SYNAUD-A §3).

The step-100 campaign had no single answer to "what is schedulable work", and
the answers it had read the wrong authority (the dependency-currency
projection) at the wrong time (after side effects). The measured consequence:
eight finishing legs re-executed 357 forwards on settled members, re-authored
19 award holders, destroyed three verified awards, and chased an
unsatisfiable halt condition for ~23-28 hours of wall clock.

This module is the ONE policy. It is consumed identically by the pipelined
scheduler, the waves driver, grant reconciliation, and the wake-resume path:

* **Clause 1 — unsettled is RAW.** A member is schedulable iff it has no raw
  model-ledger head. Raw-head existence -- ``CanonicalReducer`` construction's
  :func:`~menagerie.crawler.reducer._select_current` over the models shard,
  exposed as :attr:`CanonicalReducer.raw_head_view` -- is the ONLY settlement
  authority. The dependency-currency projection is routing data consumed
  AFTER selection, never existence authority. "A stale award is still an
  award" (``award_blocked_supersession``, reducer.py): the reducer already
  believed this; now the scheduler does.
* **Clause 3/4 (S2 corrections, opus + sol, binding).** A raw head re-enters
  only through a closed, disposition-aware exception: a live grant of the
  correct type and scope; a NAMED re-drive mode whose enumerated target list
  IS an admission authority (non-``runs`` heads only); the named gate-catchup
  mode for pending-metadata ``runs`` records; or a host-capability deferral
  whose recorded refusal no longer holds. A generic requeue grant never
  reopens an award or a capability head.
* **Clause 5 — parks are absolute, with ONE bounded machine exit (D-48).** A
  live park row (checker-lane, terminal-refusal, grant-pending) makes the
  member not-work in every mode -- except that a member whose every live park
  is in the closed MACHINERY-FAULT class re-admits for ONE fresh generation
  per change of the running code tree, on ordinary legs only
  (:func:`~menagerie.crawler.park_release.derive_park_machine_readmission`;
  the re-park under the new tree closes the lane again). Operator parks
  (grant-pending, award) and operator releases/voids are untouched.
* **Clause 6 — identity discharge.** Work generations carry a closed
  lifecycle vocabulary (:class:`GenerationDischarge`); ``--resume`` reads
  heads and discharge state, never raw identity backlogs.
* **Clause 7 — one implementation.** The predicate is pure; the
  ``SelectionPlan`` records, per member, exactly which exception admitted it;
  ``invocation_origin`` has zero selection effect.

Naming the raw-head authority (S2-opus correction 2): there are two
``_select_current`` implementations. The reducer's
(:func:`menagerie.crawler.reducer._select_current`) validates public
supersession lineage and THEN delegates to the parent-chain selector in
``state.py``; it is therefore the strict superset and the ONE selection
authority. Every selection consumer reads it through
:attr:`CanonicalReducer.raw_head_view` / :func:`raw_model_heads`; the
``state.py`` spelling remains only as its inner mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.identity import stable_hash

#: Versioned policy identity persisted with every SelectionPlan and bound into
#: the wake continuation identity: a resumed leg proves it selects work under
#: the same policy the launching leg did.
SELECTION_POLICY_VERSION = "menagerie.crawler.work-selection.v1"


class SelectionMode(str, Enum):
    """Closed leg-level selection modes (clause 4: named modes, never defaults)."""

    #: Ordinary work: never-recorded members plus granted re-drives.
    ORDINARY = "ordinary"
    #: Operator scope cut: never-recorded members ONLY (absolute, 2026-08-15).
    ONLY_UNSETTLED = "only-unsettled"
    #: Named re-drive: the enumerated target list is an admission authority
    #: for non-``runs`` raw heads (S2-opus correction 1 -- without this, the
    #: 12 non-review-held failure terminals are permanently unschedulable).
    REDRIVE = "redrive"
    #: Named gate catch-up: enumerated pending-metadata ``runs`` records
    #: re-enter for their catch-up gate round only.
    GATE_CATCHUP = "gate-catchup"
    #: The Linux deferred-status handoff (existing ``--only-status`` lane).
    ONLY_STATUS = "only-status"
    #: RC-2 named re-check: the targets are the recheck census's
    #: ``needs_recheck`` members -- acceptances recorded before a measured
    #: SOUNDNESS rotation on one of their policy axes -- derived from the raw
    #: ledger heads plus ``records/control/rotations.jsonl`` at leg start,
    #: never typed on a command line. Award custody is unchanged: a ``runs``
    #: target still needs the AW-1 award-naming grant the census prints
    #: (clause 3 admits it once minted); a non-``runs`` target admits on the
    #: soundness rotation itself as its work authority.
    RECHECK = "recheck"


class SelectionDecision(str, Enum):
    """Closed per-member selection decisions (admitting exception or exclusion)."""

    # Admissions.
    NEVER_RECORDED = "never-recorded"
    GRANTED_REDRIVE = "granted-redrive"
    ACTIVE_GRANT_GENERATION = "active-granted-generation"
    NAMED_REDRIVE = "named-redrive"
    GATE_CATCHUP = "gate-catchup"
    STATUS_MODE = "status-mode"
    #: RC-2: a needs_recheck member admitted under the named RECHECK mode on
    #: the authority of the soundness rotation row(s) that impugn its basis.
    RECHECK = "recheck"
    #: WS-2: a host-capability head whose recorded refusal no longer holds on
    #: THIS host re-opens automatically -- the remedy the deferral row itself
    #: printed, no operator involved -- in every mode EXCEPT the absolute
    #: ``--only-unsettled`` cut (2026-08-16: the arm moved below the cut). The
    #: typed capability GRANT route flows through GRANTED_REDRIVE instead, so
    #: its consumption stays atomic with the plan.
    CAPABILITY_REOPENED = "capability-reopened"
    #: D-48 machine release lane: a member whose only live parks are
    #: machinery-fault rows re-enters for ONE fresh generation per tree
    #: change, on ordinary legs only. The admitting authority is the governing
    #: park row itself (its digest is the ``work_authority``); the recording
    #: ladder re-parks idempotently if the refusal still holds.
    PARK_MACHINE_RETRY = "park-machine-retry"
    # Exclusions.
    SETTLED_HEAD = "settled-head"
    STALE_EXCLUDED = "settled-head-dependency-stale"
    PARKED = "parked"
    DEFERRED = "machine-deferred"
    AWARD_CUSTODY = "award-custody-refusal"
    #: RC-2: a needs_recheck target held for human review is not admitted by
    #: the census alone -- a held terminal's supersession requires the explicit
    #: review grant (clause 3 admits it once minted), exactly like an award.
    REVIEW_CUSTODY = "review-custody-refusal"
    DISCHARGED = "generation-discharged"
    GRANT_UNBINDABLE = "grant-unbindable"
    NOT_IN_MODE_TARGETS = "not-in-mode-targets"

    @property
    def admits(self) -> bool:
        """Return whether this decision admits the member into the leg."""

        return self in _ADMITTING_DECISIONS


_ADMITTING_DECISIONS = frozenset(
    {
        SelectionDecision.NEVER_RECORDED,
        SelectionDecision.GRANTED_REDRIVE,
        SelectionDecision.ACTIVE_GRANT_GENERATION,
        SelectionDecision.NAMED_REDRIVE,
        SelectionDecision.GATE_CATCHUP,
        SelectionDecision.STATUS_MODE,
        SelectionDecision.RECHECK,
        SelectionDecision.CAPABILITY_REOPENED,
        SelectionDecision.PARK_MACHINE_RETRY,
    }
)


class GenerationDischarge(str, Enum):
    """Closed lifecycle vocabulary for durable in-flight work generations.

    Clause 6 (the vector-5 exhibit): every minted work generation must end in
    exactly one of these states or remain LIVE (``None``); nothing replays
    indefinitely, and the 1-vs-38-vs-262 count ambiguity is resolved by this
    vocabulary, never by picking a convenient authority.
    """

    SATISFIED = "satisfied"
    FAILED_TERMINAL = "failed-terminal"
    PARKED_POLICY_REFUSAL = "parked-policy-refusal"
    VOIDED_BEFORE_START = "voided-before-start"
    SUPERSEDED_BY_GENERATION = "superseded-by-generation"
    LEG_EXIT_ABANDONED = "leg-exit-abandoned"


@dataclass(frozen=True)
class MemberFacts:
    """Pure, side-effect-free selection inputs for one intake member.

    This is the (item, raw_head, capability, grants, parks) tuple of the
    chartered predicate signature, materialized before ANY side effect.
    """

    stable_id: str
    #: Raw model-ledger head (THE settlement authority), or ``None``.
    raw_head: Optional[Mapping[str, Any]]
    #: Routing data only: whether the head replays dependency-current. Never
    #: consulted for existence; feeds the stale census and the operator ack.
    dependency_current: bool
    #: The projection's stale reason, for the census histogram.
    stale_reason: Optional[str]
    #: A live park row exists in ANY park ledger (checker-lane,
    #: terminal-refusal, grant-pending), keyed by member -- not by work_id.
    parked: bool
    #: A machine-deferral valve row exists whose recorded refusal still holds
    #: on this host (capability input; a released row reads ``False``).
    deferral_live: bool
    #: A live (unconsumed, unvoided) grant for this member, or ``None``.
    live_grant: Optional[Mapping[str, Any]]
    #: Whether that grant can bind this head (review-held terminal whose
    #: stage the grant names) -- sol's "correct type and scope".
    grant_bindable: bool
    #: A prior leg's consumed-but-unsettled generation binding, or ``None``.
    active_generation: Optional[Mapping[str, Any]]
    #: Whether that generation is already discharged (clause 6).
    active_generation_discharged: bool
    #: WS-2: the member's raw head is a host-capability deferral whose
    #: recorded refusal no longer holds on the host we are actually on
    #: (:func:`~menagerie.crawler.driver_admission.host_capability_head_reopens`
    #: over the HOST FACT only -- the typed-grant route is carried by
    #: ``live_grant``/``grant_bindable`` so consumption stays in the plan).
    capability_reopened: bool = False
    #: Closed release authority behind ``grant_bindable`` (one of
    #: :data:`~menagerie.crawler.infra_release.GRANT_RELEASE_AUTHORITIES`), or
    #: ``None`` for the capability-grant OR-arm. Replay-derived by grant
    #: reconciliation each leg (four-part waiver doctrine part 3) -- pure
    #: DISCLOSURE for the selection census, never an admission input.
    grant_release_authority: Optional[str] = None
    #: D-48: the machine-readmission fact for a parked member, replay-derived
    #: per leg from the governing durable park rows plus the running tree
    #: identity (:func:`~menagerie.crawler.park_release.derive_park_machine_readmission`),
    #: or ``None`` when the member stays parked. Never persisted; nothing a
    #: producer writes can steer it.
    park_machine_readmission: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class MemberSelection:
    """One member's selection outcome, recorded in the persisted plan."""

    stable_id: str
    admitted: bool
    decision: SelectionDecision
    #: Exact authority for a granted admission (the grant id), else ``None``.
    work_authority: Optional[str] = None
    #: Closed release-authority class behind a granted admission (censused
    #: disclosure; recomputed every leg, never read back as authority).
    release_authority: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        """Return the persisted plan row for this member."""

        return {
            "stable_id": self.stable_id,
            "admitted": self.admitted,
            "decision": self.decision.value,
            "work_authority": self.work_authority,
            "release_authority": self.release_authority,
        }


@dataclass(frozen=True)
class SelectionPlan:
    """The ONE persisted selection decision for a leg, computed pre-side-effect."""

    mode: SelectionMode
    members: tuple[MemberSelection, ...]
    stale_reasons: Mapping[str, str] = field(default_factory=dict)
    policy_version: str = SELECTION_POLICY_VERSION

    @property
    def admitted_ids(self) -> frozenset[str]:
        """Return the exact scheduled set."""

        return frozenset(member.stable_id for member in self.members if member.admitted)

    def decision_of(self, stable_id: str) -> Optional[SelectionDecision]:
        """Return one member's decision, or ``None`` for a non-member."""

        for member in self.members:
            if member.stable_id == stable_id:
                return member.decision
        return None

    def census(self) -> dict[str, Any]:
        """Return the leg-start selection census (clause 7 disclosure)."""

        counts = {
            "never_recorded": 0,
            "stale_excluded": 0,
            "parked": 0,
            "deferred": 0,
            "granted": 0,
            # Four-part waiver doctrine part 4: infra-waiver releases are a
            # SUBSET of ``granted``, censused separately so a waived release
            # is never counted as review-released.
            "granted_infra_waiver": 0,
            # W14 signed class releases: the same subset discipline, one
            # counter for the whole ``machine-class:*`` family (the per-class
            # split lives in the grant-release census).
            "granted_machine_class": 0,
            "discharged": 0,
            "settled": 0,
            "named_redrive": 0,
            "gate_catchup": 0,
            "capability_reopened": 0,
            # D-48: machine readmissions of machinery-fault parks, censused
            # separately from operator-granted re-drives -- a machine retry is
            # never counted as a reviewed release.
            "park_machine_retry": 0,
        }
        by_decision = {
            SelectionDecision.NEVER_RECORDED: "never_recorded",
            SelectionDecision.STALE_EXCLUDED: "stale_excluded",
            SelectionDecision.PARKED: "parked",
            SelectionDecision.DEFERRED: "deferred",
            SelectionDecision.GRANTED_REDRIVE: "granted",
            SelectionDecision.ACTIVE_GRANT_GENERATION: "granted",
            SelectionDecision.DISCHARGED: "discharged",
            SelectionDecision.SETTLED_HEAD: "settled",
            SelectionDecision.NAMED_REDRIVE: "named_redrive",
            SelectionDecision.GATE_CATCHUP: "gate_catchup",
            SelectionDecision.CAPABILITY_REOPENED: "capability_reopened",
            SelectionDecision.PARK_MACHINE_RETRY: "park_machine_retry",
        }
        for member in self.members:
            bucket = by_decision.get(member.decision)
            if bucket is not None:
                counts[bucket] += 1
            if (
                member.decision is SelectionDecision.GRANTED_REDRIVE
                and member.release_authority == "infra-waiver"
            ):
                counts["granted_infra_waiver"] += 1
            if member.decision is SelectionDecision.GRANTED_REDRIVE and str(
                member.release_authority or ""
            ).startswith("machine-class:"):
                counts["granted_machine_class"] += 1
        return {
            "mode": self.mode.value,
            "policy_version": self.policy_version,
            "admitted": sorted(self.admitted_ids),
            **counts,
        }

    def to_json(self) -> dict[str, Any]:
        """Return the complete persisted plan."""

        return {
            "policy_version": self.policy_version,
            "mode": self.mode.value,
            "members": [member.to_json() for member in self.members],
            "stale_reasons": dict(self.stale_reasons),
        }


def _head_kind(raw_head: Mapping[str, Any]) -> str:
    """Return one raw head's status kind, fail-closed to an empty string."""

    status = raw_head.get("status")
    if not isinstance(status, Mapping):
        return ""
    return str(status.get("kind", ""))


def _head_code(raw_head: Mapping[str, Any]) -> str:
    """Return one raw head's status code, fail-closed to an empty string."""

    status = raw_head.get("status")
    if not isinstance(status, Mapping):
        return ""
    return str(status.get("code", ""))


def _pending_metadata_runs(raw_head: Mapping[str, Any]) -> bool:
    """Return whether one raw head is the R1/R2 catch-up shape (runs@pending)."""

    return raw_head.get("authored_metadata_state") == "pending" and _head_kind(raw_head) == "runs"


def _award_naming_grant(live_grant: Mapping[str, Any], raw_head: Mapping[str, Any]) -> bool:
    """Return whether a grant names this exact award revision (AW-1's lane).

    An ordinary requeue grant authorizes WORK, never a downgrade -- retry
    permission and downgrade permission are separate authorities (§2.1). Only
    a grant that explicitly names the blocked award revision may reopen a
    ``runs`` head. The grant schema field is AW-1's seam; this predicate
    recognizes it fail-closed (absent field == not award-naming) and reads
    identically from a live grant row or a consumption receipt's details
    (the receipt copies the durable row's named revision), so a consumed
    generation carries exactly the authority its grant had.
    """

    named = live_grant.get("blocked_award_revision")
    return isinstance(named, str) and named == str(raw_head.get("record_revision"))


def _review_held(head: Optional[Mapping[str, Any]]) -> bool:
    status = (head or {}).get("status")
    if not isinstance(status, Mapping):
        return False
    review = status.get("human_review")
    return isinstance(review, Mapping) and bool(review.get("required"))


def _recheck_selection(
    facts: MemberFacts, recheck_authorities: Optional[Mapping[str, str]]
) -> MemberSelection:
    """Decide one census-derived RECHECK target (RC-2).

    The census is a ledger-derived admission authority, but award CUSTODY is
    unchanged -- superseding a ``runs`` head still takes the AW-1 award-naming
    grant the census prints, and clause 3 of the predicate admits the member
    once that grant is live; without it the award stays operator property
    (typed, never silent). Every other head kind admits on the soundness
    rotation row(s) that impugn its basis (or the unmeasured-move label).
    """

    if _head_kind(facts.raw_head) == "runs":
        return MemberSelection(facts.stable_id, False, SelectionDecision.AWARD_CUSTODY)
    if _review_held(facts.raw_head):
        return MemberSelection(facts.stable_id, False, SelectionDecision.REVIEW_CUSTODY)
    return MemberSelection(
        facts.stable_id,
        True,
        SelectionDecision.RECHECK,
        work_authority=(recheck_authorities or {}).get(facts.stable_id),
    )


def work_selection_predicate(
    facts: MemberFacts,
    *,
    mode: SelectionMode,
    mode_targets: FrozenSet[str] = frozenset(),
    only_status_codes: FrozenSet[str] = frozenset(),
    recheck_authorities: Optional[Mapping[str, str]] = None,
) -> MemberSelection:
    """Decide one member's schedulability from pure facts (THE policy).

    Parameters
    ----------
    facts:
        Materialized selection inputs; nothing here reads a ledger.
    mode:
        Closed leg selection mode.
    mode_targets:
        Enumerated stable IDs for the named ``redrive``/``gate-catchup``
        modes (the list IS the admission authority for those modes) and the
        census-derived ``needs_recheck`` members for ``recheck`` mode.
    only_status_codes:
        Exact raw-head status codes admitted by ``only-status`` mode.
    recheck_authorities:
        RC-2: per-member soundness rotation authority (rotation ids) the
        recheck census derived, recorded as the admitted member's
        ``work_authority`` in ``recheck`` mode.

    Returns
    -------
    MemberSelection
        The member's decision, including the admitting exception.
    """

    stable_id = facts.stable_id

    # Clause 5: parks are absolute and generation-independent, in every mode --
    # with D-48's ONE bounded machine exit: a member whose every live park is a
    # machinery-fault row re-enters for one fresh generation per tree change,
    # on ORDINARY legs only. Named modes keep their own authorities and the
    # absolute --only-unsettled cut stays absolute (the CAPABILITY_REOPENED
    # precedent: a machine remedy never widens an operator scope cut).
    if facts.parked:
        if mode is SelectionMode.ORDINARY and facts.park_machine_readmission is not None:
            return MemberSelection(
                stable_id,
                True,
                SelectionDecision.PARK_MACHINE_RETRY,
                work_authority=str(facts.park_machine_readmission.get("row_sha256")),
            )
        return MemberSelection(stable_id, False, SelectionDecision.PARKED)
    # Clause 2 (capability): a still-holding machine deferral withholds the
    # member; a released one falls through to ordinary admission (sol row g).
    if facts.deferral_live:
        return MemberSelection(stable_id, False, SelectionDecision.DEFERRED)

    if mode is SelectionMode.ONLY_STATUS:
        if facts.raw_head is not None and _head_code(facts.raw_head) in only_status_codes:
            return MemberSelection(stable_id, True, SelectionDecision.STATUS_MODE)
        return MemberSelection(stable_id, False, SelectionDecision.NOT_IN_MODE_TARGETS)

    # Clause 1: unsettled is RAW. No raw head == work, in every remaining mode.
    if facts.raw_head is None:
        return MemberSelection(stable_id, True, SelectionDecision.NEVER_RECORDED)

    if mode is SelectionMode.ONLY_UNSETTLED:
        # Absolute operator scope cut: a settled head -- current OR stale --
        # is never work; granted re-drives AND capability re-opens are
        # excluded too. The WS-2 arm sat ABOVE this cut until 2026-08-16
        # (red-team break 7): a recorded head re-entered the "absolute" scope
        # with work_authority=None and no acknowledgment gate -- exactly the
        # silent-readmission shape vectors 4/6 were sold this flag against.
        if not facts.dependency_current:
            return MemberSelection(stable_id, False, SelectionDecision.STALE_EXCLUDED)
        return MemberSelection(stable_id, False, SelectionDecision.SETTLED_HEAD)

    # WS-2 (clause 3/4's capability exception): a host-capability deferral
    # head whose recorded refusal no longer holds HERE is schedulable work
    # again, on every scope EXCEPT the absolute --only-unsettled cut above --
    # without this the head is absorbing forever, including on the very host
    # that should discharge it. No operator is involved: this is the remedy
    # the deferral row itself printed. A generic requeue grant never takes
    # this arm.
    if facts.capability_reopened:
        return MemberSelection(stable_id, True, SelectionDecision.CAPABILITY_REOPENED)

    # Clause 3: explicitly granted re-drives. Consumption is a CONSEQUENCE of
    # this plan (atomic with the generation mint), never a precondition.
    if facts.live_grant is not None:
        if _head_kind(facts.raw_head) == "runs" and not _award_naming_grant(
            facts.live_grant, facts.raw_head
        ):
            return MemberSelection(stable_id, False, SelectionDecision.AWARD_CUSTODY)
        if facts.grant_bindable or _award_naming_grant(facts.live_grant, facts.raw_head):
            return MemberSelection(
                stable_id,
                True,
                SelectionDecision.GRANTED_REDRIVE,
                work_authority=str(facts.live_grant.get("grant_id")),
                release_authority=facts.grant_release_authority,
            )
        return MemberSelection(stable_id, False, SelectionDecision.GRANT_UNBINDABLE)

    # A prior leg's consumed generation stays admitted until discharged
    # (clause 6): its work identity is durable and must settle, not replay
    # forever nor vanish silently. The generation carries EXACTLY its
    # originating grant's authority -- consuming a grant never upgrades it
    # (red-team break 6): this arm is reached precisely when the admitting
    # grant is consumed and dead, and a generic grant's generation used to
    # re-admit a ``runs`` head the live grant would have refused. Over an
    # award, the consumption receipt must name the exact current revision
    # (``blocked_award_revision``, carried from the durable grant row);
    # otherwise the member stays operator property until an operator lane
    # re-admits or discharges the generation.
    if facts.active_generation is not None:
        if facts.active_generation_discharged:
            return MemberSelection(stable_id, False, SelectionDecision.DISCHARGED)
        if _head_kind(facts.raw_head) == "runs" and not _award_naming_grant(
            facts.active_generation, facts.raw_head
        ):
            return MemberSelection(stable_id, False, SelectionDecision.AWARD_CUSTODY)
        return MemberSelection(
            stable_id,
            True,
            SelectionDecision.ACTIVE_GRANT_GENERATION,
            work_authority=str(facts.active_generation.get("grant_id")),
        )

    # Clause 4: named re-validation modes, never a default.
    if mode is SelectionMode.REDRIVE and stable_id in mode_targets:
        if _head_kind(facts.raw_head) == "runs":
            # A CLI target list is not an award-work authority (S2-sol B-04):
            # reopening an award requires a durable authority naming its
            # current raw revision.
            return MemberSelection(stable_id, False, SelectionDecision.AWARD_CUSTODY)
        return MemberSelection(stable_id, True, SelectionDecision.NAMED_REDRIVE)
    if mode is SelectionMode.GATE_CATCHUP and stable_id in mode_targets:
        if _pending_metadata_runs(facts.raw_head):
            return MemberSelection(stable_id, True, SelectionDecision.GATE_CATCHUP)
        return MemberSelection(stable_id, False, SelectionDecision.NOT_IN_MODE_TARGETS)
    if mode is SelectionMode.RECHECK and stable_id in mode_targets:
        return _recheck_selection(facts, recheck_authorities)

    if not facts.dependency_current:
        # The incident axis (vector 8b): recorded-but-stale used to read as
        # never-recorded. A stale settled head is EXCLUDED and CENSUSED; an
        # ordinary leg refuses to start over a nonzero count without an
        # explicit operator acknowledgment.
        return MemberSelection(stable_id, False, SelectionDecision.STALE_EXCLUDED)
    return MemberSelection(stable_id, False, SelectionDecision.SETTLED_HEAD)


def build_selection_plan(
    members: Sequence[MemberFacts],
    *,
    mode: SelectionMode,
    mode_targets: FrozenSet[str] = frozenset(),
    only_status_codes: FrozenSet[str] = frozenset(),
    recheck_authorities: Optional[Mapping[str, str]] = None,
) -> SelectionPlan:
    """Evaluate the predicate over every member and freeze the plan.

    Parameters
    ----------
    members:
        Materialized facts for the whole intake, in intake order.
    mode, mode_targets, only_status_codes, recheck_authorities:
        Exactly as :func:`work_selection_predicate`.

    Returns
    -------
    SelectionPlan
        The frozen, persistable selection decision for this leg.
    """

    selections = tuple(
        work_selection_predicate(
            facts,
            mode=mode,
            mode_targets=mode_targets,
            only_status_codes=only_status_codes,
            recheck_authorities=recheck_authorities,
        )
        for facts in members
    )
    stale_reasons = {
        facts.stable_id: str(facts.stale_reason)
        for facts in members
        if facts.stale_reason is not None
    }
    return SelectionPlan(mode=mode, members=selections, stale_reasons=stale_reasons)


def selection_mode_of(
    config: Any, *, recheck_targets: Iterable[str] = ()
) -> tuple[SelectionMode, frozenset[str]]:
    """Derive the closed selection mode and targets from a driver config.

    Parameters
    ----------
    config:
        :class:`~menagerie.crawler.driver_contracts.DriverConfig` (duck-typed
        for tests). A truthy ``recheck`` attribute selects RC-2's named
        re-check mode.
    recheck_targets:
        The recheck census's ``needs_recheck`` members, derived by the caller
        at leg start (:func:`menagerie.crawler.recheck_census.recheck_targets`);
        the config itself carries no target list for this mode.

    Returns
    -------
    tuple[SelectionMode, frozenset[str]]
        The leg's mode and its enumerated targets (empty outside named modes).
    """

    if getattr(config, "recheck", False):
        return SelectionMode.RECHECK, frozenset(str(item) for item in recheck_targets)
    if getattr(config, "only_status", None) is not None:
        return SelectionMode.ONLY_STATUS, frozenset()
    if getattr(config, "only_unsettled", False):
        return SelectionMode.ONLY_UNSETTLED, frozenset()
    redrive = tuple(getattr(config, "redrive_targets", ()) or ())
    if redrive:
        return SelectionMode.REDRIVE, frozenset(redrive)
    catchup = tuple(getattr(config, "gate_catchup_targets", ()) or ())
    if catchup:
        return SelectionMode.GATE_CATCHUP, frozenset(catchup)
    return SelectionMode.ORDINARY, frozenset()


def selection_continuation_identity(config: Any) -> str:
    """Return the byte-stable continuation identity of a launching config.

    Wake callbacks persist and replay the COMPLETE launching config; a
    resumed leg compares this identity byte-for-byte and refuses to continue
    under a different selection world (DELTA F5/D8: the step-100 wake argv
    would have resumed UNSCOPED under the default scheduler).
    """

    mode, targets = selection_mode_of(config)
    payload = {
        "kind": "work-selection-continuation",
        "policy_version": SELECTION_POLICY_VERSION,
        "mode": mode.value,
        "mode_targets": sorted(targets),
        "scheduler": str(getattr(config, "scheduler", "waves")),
        "target": str(getattr(config, "target", "")),
        "phase": getattr(config, "phase", None),
        "only_status": getattr(config, "only_status", None),
        "only_unsettled": bool(getattr(config, "only_unsettled", False)),
        "recheck": bool(getattr(config, "recheck", False)),
        "ack_stale_excluded": bool(getattr(config, "ack_stale_excluded", False)),
        "acknowledge_award_holds": bool(getattr(config, "acknowledge_award_holds", False)),
        "award_holds_digest": getattr(config, "award_holds_digest", None),
        "stale_excluded_digest": getattr(config, "stale_excluded_digest", None),
        "author_concurrency": int(getattr(config, "author_concurrency", 1)),
        "checker_concurrency": int(getattr(config, "checker_concurrency", 1)),
    }
    # Versioned-monotone (shape 4): the declared purpose joins the sealed
    # identity ONLY when set, so every pre-purpose config -- including wake
    # argv persisted before this field existed -- keeps its exact identity.
    leg_purpose = getattr(config, "leg_purpose", None)
    if leg_purpose is not None:
        payload["leg_purpose"] = str(leg_purpose)
    return stable_hash(payload)


def raw_model_heads(reducer: Any) -> Mapping[str, Mapping[str, Any]]:
    """Return THE raw settlement authority: validated model-ledger heads.

    This is the reducer's construction-time
    :func:`~menagerie.crawler.reducer._select_current` map (public
    supersession lineage validated, maintained on every append) -- never the
    dependency-currency projection and never a replay-filtered surface.
    """

    return reducer.raw_head_view


def derive_generation_discharge(
    details: Mapping[str, Any],
    *,
    model_revisions: Sequence[Mapping[str, Any]],
    live_park_work_ids: FrozenSet[str],
    discharge_events: Sequence[Mapping[str, Any]],
    all_consumptions: Sequence[Mapping[str, Any]],
    voided_grant_ids: FrozenSet[str],
) -> tuple[Optional[GenerationDischarge], Optional[str]]:
    """Derive one consumed generation's discharge state from ledger evidence.

    A VOIDED grant discharges every identity it minted (2026-08-16 red-team
    round 2, break 5). Voiding removed the grant from the live-grant arm and
    from nothing else, so the same withdrawn authority kept re-admitting its
    member through its consumed GENERATION on every leg -- one wasted
    author/execution purchase per generation per member, and an operator lever
    that silently did not do what it said. The void is honoured HERE, at the
    one derivation selection consults, so the live arm and the generation arm
    can never disagree about whether a grant is dead. A LANDED revision still
    wins: work that actually settled is ``satisfied``, and relabelling it
    "voided" would be the dishonest half of the same mistake.

    Parameters
    ----------
    details:
        The ``requeue-grant-consumed`` event details (grant_id, stable_id,
        new_work_id, new_work_generation).
    model_revisions:
        Canonical model revisions, append order.
    live_park_work_ids:
        Work IDs held by live park rows across every park ledger.
    discharge_events:
        ``requeue-generation-discharged`` events (explicit operator or
        leg-exit discharges).
    all_consumptions:
        Every consumption event, used for supersession detection.
    voided_grant_ids:
        Grant ids the canonical void ledger has withdrawn. Required, not
        defaulted: a caller that cannot state the voids must not be able to
        derive a LIVE verdict by omission.

    Returns
    -------
    tuple[GenerationDischarge | None, str | None]
        The discharge state and its evidence (``None`` == LIVE: the
        generation is still schedulable work and, per the migration, may
        need an operator decision).
    """

    grant_id = str(details.get("grant_id", ""))
    stable_id = str(details.get("stable_id", ""))
    work_id = str(details.get("new_work_id", ""))
    generation = details.get("new_work_generation")

    for event in discharge_events:
        event_details = event.get("details", {})
        if event_details.get("stable_id") == stable_id and event_details.get("work_id") == work_id:
            state = GenerationDischarge(str(event_details.get("discharge")))
            return state, f"explicit discharge event {event.get('event_id')}"

    landed = [
        revision
        for revision in model_revisions
        if revision.get("stable_id") == stable_id
        and grant_id in revision.get("budget", {}).get("explicit_grants", [])
    ]
    if landed:
        final = landed[-1]
        kind = _head_kind(final)
        state = (
            GenerationDischarge.SATISFIED if kind == "runs" else GenerationDischarge.FAILED_TERMINAL
        )
        return state, f"landed revision {final.get('record_revision')}"

    if grant_id in voided_grant_ids:
        return (
            GenerationDischarge.VOIDED_BEFORE_START,
            f"grant {grant_id} voided: the authority that minted this generation was withdrawn",
        )

    if work_id in live_park_work_ids:
        return GenerationDischarge.PARKED_POLICY_REFUSAL, "live park row"

    later = [
        event
        for event in all_consumptions
        if event.get("details", {}).get("stable_id") == stable_id
        and isinstance(event.get("details", {}).get("new_work_generation"), int)
        and isinstance(generation, int)
        and event["details"]["new_work_generation"] > generation
    ]
    if later:
        return (
            GenerationDischarge.SUPERSEDED_BY_GENERATION,
            f"superseded by generation {later[-1]['details']['new_work_generation']}",
        )
    return None, None
