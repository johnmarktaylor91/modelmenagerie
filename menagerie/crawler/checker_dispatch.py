"""Fresh Codex checker envelopes, atomic results, and usage backoff classification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Collection, Mapping, Optional, Sequence, TypeVar, Union

from menagerie.crawler.constants import (
    checker_provider_for_model,
    fidelity_item_accepted,
    metadata_item_accepted,
    AUTHOR_DISPATCHER_COMPONENT,
    AUTHOR_RESULT_SCHEMA_COMPONENT,
    AccuracyVerdict,
    CHECKER_PROMPT_NAME,
    CheckerPauseReason,
    FidelityVerdict,
    GATE_ROUND_CEILINGS,
    GATE_SCHEMA_VERSION_V3,
    GateKind,
    METADATA_BATCH_MAX,
    METADATA_BATCH_MIN,
    METADATA_FINAL_TAIL_MIN,
    SourceRung,
    USAGE_LIMIT_PROVIDERS,
)
from menagerie.crawler.environment_inventory import ROUTED_INVENTORY_SOURCE_KIND
from menagerie.crawler.evidence import (
    LOCATOR_ATTESTATION_INVALID,
    LOCATOR_ATTESTATION_VALID,
    LOCATOR_AUTHORITY_TEXT_LINE,
    attest_excerpt_locators,
)
from menagerie.crawler.identity import (
    SHADOW_REHEARSAL_MARKER_FIELD,
    compute_evidence_identity,
    hash_bytes,
    stable_hash,
    stamp_shadow_rehearsal,
)
from menagerie.crawler.metadata import (
    MetadataValidationError,
    validate_authored_facts_for_write,
)
from menagerie.crawler.models import JsonObject, bounded_json_repr
from menagerie.crawler.operator_protocol import OperatorExitCode, build_operator_fields
from menagerie.crawler.proposal import (
    ProposalValidationError,
    required_metadata_field_checks,
    required_verified_hash_keys,
)
from menagerie.crawler.schema import (
    PayloadValidationError,
    RequiredFieldProjection,
    required_field_projection_spec,
    validate_payload,
)
from menagerie.crawler.terminal_evidence import (
    GROUNDED as TERMINAL_EVIDENCE_GROUNDED,
    SOURCE_CAS_DIRNAME,
    TERMINAL_EVIDENCE_RESOLUTIONS,
    UNRESOLVED as TERMINAL_EVIDENCE_UNRESOLVED,
)

PROMPT_PATH = Path(__file__).with_name("prompts") / f"{CHECKER_PROMPT_NAME}.txt"

#: BL-10 (e): the leg-scoped checker prompt pin. One driver process is one leg,
#: so pinning at first use and refusing every later divergence makes mid-leg
#: prompt rotation structurally impossible at the dispatch seam. ``None`` means
#: no checker dispatch has happened yet this leg.
_LEG_CHECKER_PROMPT_PIN: "Optional[str]" = None

_MetadataItemT = TypeVar("_MetadataItemT")

#: One checker provider-weather storm budget, shared with the driver retry seam.
#: The first retry distinguishes a transient edge; the remaining waits reproduce
#: the W2-D author-lane weather tier. Each entry is one storm interval, not one
#: Codex attempt inside the wrapper.
CHECKER_WEATHER_BACKOFF_SECONDS: tuple[float, ...] = (
    5.0,
    300.0,
    900.0,
    900.0,
    900.0,
)

#: Binding per-wait ceiling for checker provider-weather parking.
CHECKER_WEATHER_BACKOFF_CEILING_SECONDS: float = 900.0

#: Total bounded time one frozen checker request may remain parked.
CHECKER_WEATHER_PARK_BUDGET_SECONDS: float = sum(CHECKER_WEATHER_BACKOFF_SECONDS)

#: Total severity order over the closed checker verdicts.
#:
#: The frozen checker prompt's DECISION RULE is a maximum over this order: any
#: inaccurate component makes the top level inaccurate, otherwise any
#: cannot-verify makes it cannot-verify, otherwise accurate. Naming the order
#: once lets every lane state that one rule instead of each spelling out a
#: different comparison.
VERDICT_SEVERITY: Mapping[AccuracyVerdict, int] = MappingProxyType(
    {
        AccuracyVerdict.ACCURATE: 0,
        AccuracyVerdict.CANNOT_VERIFY: 1,
        AccuracyVerdict.INACCURATE: 2,
    }
)

#: Terminal-lane verdict lockstep. On a ``terminal_disposition`` item the
#: disposition decides the item's top-level ``verdict``, which carries its exact
#: counterpart below. ``integrity`` is scored on its own evidence and is bounded,
#: not pinned: it may be no MORE severe than the verdict under
#: :data:`VERDICT_SEVERITY`.
#:
#: The integrity half used to be an equality, and that mandated a falsehood. An
#: item rejected purely on the merits, with zero hash mismatches, zero excerpt
#: discrepancies and zero locator failures, still had to declare
#: ``integrity: "inaccurate"``, erasing the only signal separating "rejected
#: because the evidence is forged" from "rejected because the argument fails".
#: The same 10-model pilot rung on 2026-07-30 lost two of six terminal verdicts
#: to that clause: both agreed with the disposition at the top level and both
#: scored ``integrity`` independently (once ``accurate``, once ``cannot-verify``)
#: -- which the monotone rule now ADMITS, because in both the verdict was at
#: least as severe as integrity. The equality had no safety content: it could
#: only force integrity UP to match a severe disposition, never restrain a
#: lenient one. What refuses an unsafe item is the monotone bound, which still
#: rejects every acceptance over degraded integrity -- as
#: :func:`menagerie.crawler.gates.validate_terminal_disposition_gate`
#: independently does.
#:
#: This is a named constant rather than a dict literal inside the check because
#: it is a CONTRACT THE CHECKER MUST BE TOLD; the pilot's rejections were not
#: defiance, they were a rule the prompt never stated.
#: ``test_terminal_verdict_lockstep_is_stated_in_the_frozen_prompt`` reads this
#: mapping and requires the prompt to spell every arrow, so changing the rule
#: here fails until the prompt is re-stated and re-pinned.
TERMINAL_VERDICT_LOCKSTEP: Mapping[str, AccuracyVerdict] = MappingProxyType(
    {
        "accepted": AccuracyVerdict.ACCURATE,
        "rejected": AccuracyVerdict.INACCURATE,
        "cannot-verify": AccuracyVerdict.CANNOT_VERIFY,
    }
)

# Machine-owned gate scaffold. ``gate_id`` and the two component identities are
# IDENTITIES, not judgments: the checker cannot observe them and has no authority
# over them, so asking a model to author them turns a complete, well-reasoned
# verdict into a discarded contract rejection the first time it forgets one. That
# is exactly what killed a live campaign on 2026-07-29 (``'gate_id' is a required
# property``). The machine derives them and the wrapper stamps them; the checker
# owns only its verdict.
GATE_ID_PREFIX = "gate-"

# LEDGER-assigned, not wrapper-stamped. gate.v3's own authority-ownership table
# says ``$.ledger_seq -> reducer-derived``, and ``payload_sha256`` binds the exact
# line the ledger writes. NEITHER value exists until the append happens, so the
# stamped gate OMITS both and ``JsonlLedger.append`` assigns them from the
# ledger's own state.
#
# This used to be a pair of constants stamped into the scaffold
# (``PLACEHOLDER_LEDGER_SEQ = 1``). ``payload_sha256`` survived that because the
# ledger overwrites it unconditionally; ``ledger_seq`` did not, because the ledger
# only fills it when ABSENT and otherwise refuses a value that is not the next
# local sequence. A constant 1 is the next sequence exactly once, so the defect
# was invisible until a campaign recorded its SECOND gate and died with
# ``LedgerConflictError: ledger_seq must be next local sequence 2`` (2026-07-30,
# rung 10, 10/10 authors published). The stamped constant also poisoned the FIRST
# gate: ``result_envelope_sha256`` was computed over a body containing
# ``ledger_seq``, while ``authority.load_current_gate_proof`` recomputes it with
# ``ledger_seq`` excluded, so every recorded gate failed its own self-hash with
# "current gate result-envelope self-hash is invalid". Omitting the fields makes
# both hashes agree by construction, because an absent key and an excluded key
# canonicalize identically.
LEDGER_ASSIGNED_GATE_FIELDS = ("ledger_seq", "payload_sha256")

# Schema-shape stand-ins used ONLY to satisfy gate.v3's ``required`` list when a
# not-yet-appended gate is validated. They are never published, never hashed, and
# never handed to the ledger -- see ``_with_pending_ledger_fields``.
_PENDING_LEDGER_GATE_FIELDS: Mapping[str, Any] = {
    "ledger_seq": 1,
    "payload_sha256": "sha256:" + "0" * 64,
}
DETERMINISTIC_GATE_SCAFFOLD_FIELDS = (
    "schema_version",
    "gate_id",
    "gate_kind",
    "batch_size",
    "gate_round",
    "gate_identity",
    "author_result_schema_identity",
    "dispatcher_identity",
)


def metadata_fidelity_state() -> JsonObject:
    """Return one fresh machine-owned ``fidelity`` block for a metadata gate item.

    On a ``metadata_batch`` gate the fidelity block carries ZERO checker
    judgment: ``gate.v3`` pins ``required`` to ``false`` and ``verdict`` to
    ``"not-applicable"``, and the check arrays have nothing legal to hold. The
    whole block is a gate-kind constant, so it is MACHINE-OWNED and stamped by
    the wrapper -- the checker's native output schema for metadata gates omits
    it entirely, exactly like the gate scaffold.

    Returns
    -------
    dict[str, Any]
        Fresh typed not-applicable fidelity block (fresh lists, never shared).
    """

    return {
        "required": False,
        "verdict": "not-applicable",
        "material_checks": [],
        "unsupported_choices": [],
        "contradictions": [],
        "omissions": [],
        "permanent_scar": False,
    }


def metadata_fidelity_normalizations(
    result: Mapping[str, Any], envelope: Mapping[str, Any]
) -> tuple[str, ...]:
    """Name the items whose supplied fidelity diverges from the machine constant.

    The stamp in :func:`apply_machine_owned_gate_fields` is unconditional, and
    for THIS field that is safe -- see the laundering boundary documented there
    -- but overwriting must never be silent: a supplied divergent block is
    evidence the checker misunderstood its lane, and rung 8 proved exactly that
    (batch ``metadata-c001cad863d95a67``: a real ``fidelity.verdict`` on a
    metadata gate killed m10517 and m9666 in the same second). The wrapper calls
    this BEFORE stamping and emits the returned IDs as telemetry, so the
    evidence survives the normalization that keeps the batch alive.

    Parameters
    ----------
    result:
        Decoded candidate gate, before stamping.
    envelope:
        Hash-bound request envelope.

    Returns
    -------
    tuple[str, ...]
        ``stable_id`` values (in item order) whose supplied ``fidelity`` is
        present and not the machine-owned constant. Empty for non-metadata
        gates and for compliant results.
    """

    if envelope.get("gate_kind") != GateKind.METADATA_BATCH.value:
        return ()
    items = result.get("items")
    if not isinstance(items, list):
        return ()
    machine = metadata_fidelity_state()
    return tuple(
        str(item.get("stable_id"))
        for item in items
        if isinstance(item, Mapping) and "fidelity" in item and item["fidelity"] != machine
    )


def metadata_item_rejected(item: Mapping[str, Any]) -> bool:
    """Return whether one metadata gate item is a rejected (non-accepted) round.

    This is the ONE acceptance predicate the repair-cap accounting runs on,
    shared with the driver's routing and the authority's terminal proof so the
    three lanes can never drift: an item is accepted only when its top-level
    verdict, integrity verdict, and rung-check verdict are ALL accurate.

    Parameters
    ----------
    item:
        Persisted or candidate metadata gate item.

    Returns
    -------
    bool
        True for any adverse round.
    """

    integrity = item.get("integrity")
    rung_check = item.get("rung_check")
    return not (
        item.get("verdict") == AccuracyVerdict.ACCURATE.value
        and isinstance(integrity, Mapping)
        and integrity.get("verdict") == AccuracyVerdict.ACCURATE.value
        and isinstance(rung_check, Mapping)
        and rung_check.get("verdict") == AccuracyVerdict.ACCURATE.value
    )


def metadata_complaint_fields(item: Mapping[str, Any]) -> frozenset[str]:
    """Return the complaint keys one metadata gate item actually disclosed.

    A COMPLAINT is any component verdict the checker scored non-accurate: each
    ``field_checks`` row keyed by its claim path (the tolerated
    ``proposed_facts.`` prefix stripped exactly as the write-time validator
    strips it), plus the two non-claim components that reject an item on their
    own -- ``integrity`` and ``rung_check``, under their bare component names,
    which are disjoint from every gated claim path by construction.

    The key is the CLAIM, not the verdict class: a field flagged inaccurate in
    one round and cannot-verify in the next is the same complaint re-raised,
    so a checker cannot launder a repeat through verdict reclassification.
    A malformed component block counts as a complaint (fail-closed toward the
    stricter, chargeable reading rather than silently blessing it).

    Parameters
    ----------
    item:
        Persisted or candidate metadata gate item.

    Returns
    -------
    frozenset[str]
        Complaint keys disclosed by this round. Empty for an accepted round.
    """

    complaints: set[str] = set()
    checks = item.get("field_checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            if check.get("verdict") == AccuracyVerdict.ACCURATE.value:
                continue
            field = check.get("field")
            if isinstance(field, str):
                complaints.add(field.removeprefix("proposed_facts."))
    for component in ("integrity", "rung_check"):
        block = item.get(component)
        if not isinstance(block, Mapping) or block.get("verdict") != (
            AccuracyVerdict.ACCURATE.value
        ):
            complaints.add(component)
    return frozenset(complaints)


def metadata_rejection_chargeable(
    item: Mapping[str, Any], prior_disclosed: Optional[Collection[str]]
) -> bool:
    """Decide whether one rejected round charges the author's bounded repair cap.

    The complaint-disclosure rule (wave-3, rung-9b census docket #4): a repair
    round may only be charged against complaints already DISCLOSED to the
    author. A rejected round is therefore chargeable exactly when it re-raises
    at least one complaint SOME earlier round in the lineage disclosed -- the
    author was told and failed. A rejection consisting ONLY of complaints on
    claims no prior round ever raised is a FIRST DISCLOSURE: it still blocks
    acceptance (no verdict is weakened), but it measures checker disclosure
    debt, not author incorrigibility, so it grants a fresh repair round instead
    of consuming the cap. Rung 9b burned both m10517 and m7362 to
    ``cannot-verify-cap-exhausted`` on exactly this churn: every round's
    complaints were fixed, and the terminal cause (``paradigm``) was first
    flagged in round 3 after being blessed twice.

    The disclosure baseline is the UNION of every prior round's complaints, not
    the immediately preceding round's set (BL-10 b / audit B-1). Comparing
    against one round made the rule walkable: a checker alternating disjoint
    complaint sets ``{A}, {B}, {A}, {B}`` was never chargeable after round one
    -- pilot lineage m8650 ran 10 rounds and 9 rejections against a cap of 2,
    charging exactly 1. Under the union rule a claim grants a free round at
    most ONCE per lineage, ever, which is what the boundedness statement below
    always claimed.

    The first round of a lineage has no prior disclosure to answer and is
    always chargeable. A lineage whose only prior round ACCEPTED disclosed zero
    complaints, so a rejection following it re-raises nothing and is never
    chargeable -- consistent with the poisoned-accurate-gate doctrine that a
    fresh checker round on the same proposal counts zero repairs.

    Parameters
    ----------
    item:
        The rejected metadata gate item being accounted.
    prior_disclosed:
        The union of complaint keys disclosed by EVERY earlier round in the
        same lineage, or ``None`` when this is the lineage's first gate round.

    Returns
    -------
    bool
        True when this rejection consumes one bounded repair allowance.
    """

    if prior_disclosed is None:
        return True
    return bool(metadata_complaint_fields(item) & frozenset(prior_disclosed))


def disclosed_metadata_complaint_union(
    history: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> Optional[frozenset[str]]:
    """Return the union of complaints every round of one lineage disclosed.

    ``None`` means the lineage has no persisted round at all, which the routing
    reads as "nothing was previously disclosed, so a fresh failure is a
    chargeable first round". An empty set means rounds exist but disclosed
    nothing (all accepted).

    Parameters
    ----------
    history:
        Complete ledger-ordered ``(gate, item)`` pairs for ONE model lineage.

    Returns
    -------
    frozenset[str] | None
        Union of every disclosed complaint key, or ``None`` for no history.
    """

    if not history:
        return None
    union: frozenset[str] = frozenset()
    for _gate, item in history:
        union |= metadata_complaint_fields(item)
    return union


def chargeable_metadata_rejections(
    history: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    """Filter one lineage's gate history down to its cap-chargeable rejections.

    Boundedness is preserved without the raw-count cap: an exact repeat of an
    earlier complaint state still fast-terminals through the unchanged
    repeated-fingerprint rule, every re-raised complaint is chargeable, and --
    because the disclosure baseline is the UNION of all prior rounds -- a claim
    can grant a free round at most once per lineage, so a churning checker
    cannot spin a model forever: it can only grant the author the repair rounds
    its own late disclosures owe, bounded by the finite claim inventory, and
    the absolute :data:`menagerie.crawler.constants.METADATA_GATE_ROUND_CEILING`
    parks anything that still churns past that.

    Parameters
    ----------
    history:
        Complete ledger-ordered ``(gate, item)`` pairs for ONE model lineage,
        accepted and rejected rounds alike.

    Returns
    -------
    tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]
        The rejected pairs that charge the bounded repair cap, in order.
    """

    chargeable: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    disclosed: Optional[frozenset[str]] = None
    for gate, item in history:
        if metadata_item_rejected(item) and metadata_rejection_chargeable(item, disclosed):
            chargeable.append((gate, item))
        disclosed = (disclosed or frozenset()) | metadata_complaint_fields(item)
    return tuple(chargeable)


def fidelity_item_rejected(item: Mapping[str, Any]) -> bool:
    """Return whether one fidelity gate item is a rejected (non-accepted) round.

    The fidelity twin of :func:`metadata_item_rejected`, matching the driver's
    ``_fidelity_item_accepted`` exactly: an item is accepted only when its
    five-way fidelity verdict is ``match``/``minor-drift`` AND its rung check is
    accurate. A malformed component reads as rejected (fail-closed).

    Parameters
    ----------
    item:
        Persisted or candidate fidelity gate item.

    Returns
    -------
    bool
        True for any adverse round.
    """

    fidelity = item.get("fidelity")
    rung_check = item.get("rung_check")
    return not (
        isinstance(fidelity, Mapping)
        and fidelity.get("verdict") in {"match", "minor-drift"}
        and isinstance(rung_check, Mapping)
        and rung_check.get("verdict") == AccuracyVerdict.ACCURATE.value
    )


def fidelity_complaint_fields(item: Mapping[str, Any]) -> frozenset[str]:
    """Return the complaint keys one fidelity gate item actually disclosed.

    The metadata complaint keys (non-accurate ``field_checks`` claims plus the
    bare ``integrity``/``rung_check`` component names) apply verbatim, and the
    fidelity item adds ONE more rejecting component: the five-way ``fidelity``
    verdict itself, under its bare component name -- disjoint from every gated
    claim path by construction, exactly like the other two. Without it a
    round rejected purely on implementation drift (all field checks accurate)
    would disclose an empty complaint set, and the next drift rejection would
    read as a cost-free first disclosure forever.

    Parameters
    ----------
    item:
        Persisted or candidate fidelity gate item.

    Returns
    -------
    frozenset[str]
        Complaint keys disclosed by this round. Empty for an accepted round.
    """

    complaints = set(metadata_complaint_fields(item))
    fidelity = item.get("fidelity")
    if not isinstance(fidelity, Mapping) or fidelity.get("verdict") not in {
        "match",
        "minor-drift",
    }:
        complaints.add("fidelity")
    return frozenset(complaints)


def fidelity_rejection_chargeable(
    item: Mapping[str, Any], prior_disclosed: Optional[Collection[str]]
) -> bool:
    """Decide whether one rejected fidelity round charges the bounded repair cap.

    The complaint-disclosure rule, extended to the fidelity lane: a rejected
    round charges only when it re-raises at least one complaint SOME earlier
    round in the lineage disclosed. A rejection consisting ONLY of complaints
    on claims no prior round ever raised is a FIRST DISCLOSURE -- it still
    blocks acceptance (no verdict is weakened) but measures checker disclosure
    debt, not author incorrigibility, so it grants a fresh repair round instead
    of consuming the cap. The checker prompt has promised this accounting for
    every ``gate_round`` since the wave-3 docket; only the metadata lane
    implemented it, so a fidelity checker that blessed ``input_contract`` twice
    and first flagged it in the terminal round (pilot m1359) still consumed the
    author's last round on its own late disclosure.

    The baseline is the UNION of all prior rounds' disclosures, exactly as in
    :func:`metadata_rejection_chargeable` (BL-10 b): one-round comparison made
    the rule walkable by disjoint alternation.

    Parameters
    ----------
    item:
        The rejected fidelity gate item being accounted.
    prior_disclosed:
        Union of complaint keys disclosed by every earlier round in the same
        lineage, or ``None`` when this is the lineage's first gate round
        (always chargeable).

    Returns
    -------
    bool
        True when this rejection consumes one bounded repair allowance.
    """

    if prior_disclosed is None:
        return True
    return bool(fidelity_complaint_fields(item) & frozenset(prior_disclosed))


def chargeable_fidelity_rejections(
    history: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    """Filter one lineage's fidelity gate history to its cap-chargeable rejections.

    Boundedness holds exactly as it does for
    :func:`chargeable_metadata_rejections`: every re-raised complaint charges,
    an exact repeat of an earlier rejected round still fast-terminals through
    the unchanged repeated-fingerprint rule, and each claim can be newly
    disclosed at most once per lineage, so a churning checker only grants the
    author the rounds its own late disclosures owe.

    Parameters
    ----------
    history:
        Complete ledger-ordered ``(gate, item)`` pairs for ONE model lineage,
        accepted and rejected rounds alike.

    Returns
    -------
    tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]
        The rejected pairs that charge the bounded repair cap, in order.
    """

    chargeable: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    disclosed: Optional[frozenset[str]] = None
    for gate, item in history:
        if fidelity_item_rejected(item) and fidelity_rejection_chargeable(item, disclosed):
            chargeable.append((gate, item))
        disclosed = (disclosed or frozenset()) | fidelity_complaint_fields(item)
    return tuple(chargeable)


def _gate_item_accepted(gate_kind: str, item: Mapping[str, Any]) -> bool:
    """Return one gate item's acceptance under its lane's shared kernel."""

    if gate_kind == GateKind.METADATA_BATCH.value:
        return metadata_item_accepted(item)
    if gate_kind == GateKind.FIDELITY.value:
        return fidelity_item_accepted(item)
    if gate_kind == GateKind.TERMINAL_DISPOSITION.value:
        disposition = item.get("terminal_disposition")
        return isinstance(disposition, Mapping) and disposition.get("verdict") == "accepted"
    raise CheckerDispatchError(f"unknown gate kind for acceptance: {gate_kind!r}")


def gate_round_ceiling_state(
    gates: Sequence[Mapping[str, Any]],
    *,
    gate_kind: str,
    stable_id: str,
    campaign_root_work_id: Optional[str] = None,
) -> tuple[tuple[tuple[JsonObject, JsonObject], ...], int]:
    """Return one lineage's non-accepted rounds and its lane's absolute ceiling.

    AG-6: every gate lane carries the metadata mechanism's ABSOLUTE round
    ceiling (:data:`~menagerie.crawler.constants.GATE_ROUND_CEILINGS`), checked
    BEFORE a session is bought. Only NON-ACCEPTED rounds count: re-gates that
    re-confirm an acceptance (the 08-23 sweep re-adjudicated 37 already
    accepted arms) must never park a healthy lineage, while the m4257 shape --
    12 all-rejected terminal-disposition rounds -- parks at the ceiling as
    checker disclosure debt instead of burning forever. An unknown gate kind
    fails closed (:class:`CheckerDispatchError` / ``KeyError``), so a new
    GateKind cannot ship uncapped.

    Parameters
    ----------
    gates:
        Persisted gate rows (ledger order).
    gate_kind:
        One closed :class:`~menagerie.crawler.constants.GateKind` value.
    stable_id:
        Model stable identity.
    campaign_root_work_id:
        Optional lineage scope; ``None`` matches every lineage.

    Returns
    -------
    tuple[tuple[tuple[JsonObject, JsonObject], ...], int]
        ``(non_accepted_history, ceiling)`` -- the matched ``(gate, item)``
        pairs and this lane's absolute ceiling.
    """

    ceiling = GATE_ROUND_CEILINGS[gate_kind]
    history: list[tuple[JsonObject, JsonObject]] = []
    for gate in gates:
        if gate.get("gate_kind") != gate_kind:
            continue
        for item in gate.get("items", []):
            if item.get("stable_id") != stable_id:
                continue
            if (
                campaign_root_work_id is not None
                and item.get("campaign_root_work_id") != campaign_root_work_id
            ):
                continue
            if not _gate_item_accepted(gate_kind, item):
                history.append((dict(gate), dict(item)))
            break
    return tuple(history), ceiling


def gate_adjudication_identity(gate: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return one gate row's full adjudication identity, fail-closed to "".

    AG-11: the grouping key for every per-identity telemetry derivation --
    (provider, model, version, prompt_sha256), exactly the components the
    reuse machinery binds. Derived from the persisted row on every call;
    nothing mutable is consulted.
    """

    checker = gate.get("checker")
    checker = checker if isinstance(checker, Mapping) else {}
    return (
        str(checker.get("provider", "")),
        str(checker.get("model", "")),
        str(checker.get("version", "")),
        str(checker.get("prompt_sha256", "")),
    )


def gate_acceptance_timeline(
    gates: Sequence[Mapping[str, Any]],
    *,
    gate_kind: str,
    stable_id: str,
) -> dict[str, Any]:
    """Derive one member's per-lane acceptance telemetry from persisted rows.

    AG-11's gate-side kernel, shared so the monitor (WS9) and any census
    cannot drift from the proof kernels' acceptance semantics: rounds are
    ledger-ordered; ``rounds_after_first_acceptance`` counts the adjudications
    bought after the first accepted round (the 200a re-gate bill: 509/678
    metadata rounds); ``accept_to_reject_flips`` counts ACCEPT->REJECT
    transitions (119 in 200a; the sol-proven verdict-instability signal).

    Parameters
    ----------
    gates:
        Persisted gate rows (ledger order).
    gate_kind:
        One closed :class:`~menagerie.crawler.constants.GateKind` value.
    stable_id:
        Member identity.

    Returns
    -------
    dict[str, Any]
        ``rounds``, ``accepted_rounds``, ``rounds_after_first_acceptance``,
        ``accept_to_reject_flips``, and the ordered per-round
        ``(identity, accepted)`` trail.
    """

    trail: list[dict[str, Any]] = []
    for gate in gates:
        if gate.get("gate_kind") != gate_kind:
            continue
        for item in gate.get("items", []):
            if item.get("stable_id") != stable_id:
                continue
            trail.append(
                {
                    "identity": gate_adjudication_identity(gate),
                    "accepted": _gate_item_accepted(gate_kind, item),
                }
            )
            break
    first_accept_index: Optional[int] = None
    flips = 0
    previous_accepted: Optional[bool] = None
    for index, round_row in enumerate(trail):
        if round_row["accepted"] and first_accept_index is None:
            first_accept_index = index
        if previous_accepted is True and not round_row["accepted"]:
            flips += 1
        previous_accepted = round_row["accepted"]
    return {
        "rounds": len(trail),
        "accepted_rounds": sum(1 for row in trail if row["accepted"]),
        "rounds_after_first_acceptance": (
            len(trail) - first_accept_index - 1 if first_accept_index is not None else 0
        ),
        "accept_to_reject_flips": flips,
        "trail": trail,
    }


def gate_verdict_arm_census(gates: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Count per-lane verdict-arm firings with EVERY closed arm present.

    AG-11's never-silently-0 property: the census enumerates the closed
    verdict vocabulary per lane up front, so an arm that never fired appears
    with an explicit 0 (the monitor prints it as ``not-exercised``) instead
    of silently vanishing -- the 200a ``slop``/``permanent_scar`` arms ran an
    entire campaign at 0 firings with nothing able to say so.
    """

    arms: dict[str, dict[str, int]] = {
        GateKind.METADATA_BATCH.value: {member.value: 0 for member in AccuracyVerdict},
        GateKind.FIDELITY.value: {member.value: 0 for member in FidelityVerdict},
        GateKind.TERMINAL_DISPOSITION.value: {"accepted": 0, "rejected": 0, "cannot-verify": 0},
    }
    for gate in gates:
        gate_kind = str(gate.get("gate_kind"))
        lane = arms.get(gate_kind)
        if lane is None:
            continue
        for item in gate.get("items", []):
            if gate_kind == GateKind.FIDELITY.value:
                verdict = item.get("fidelity", {}).get("verdict")
            elif gate_kind == GateKind.TERMINAL_DISPOSITION.value:
                verdict = item.get("terminal_disposition", {}).get("verdict")
            else:
                verdict = item.get("verdict")
            if isinstance(verdict, str) and verdict in lane:
                lane[verdict] += 1
    return arms


class CheckerDispatchError(ValueError):
    """Raised when a checker request or atomic result violates its contract."""


class CheckerEnvelopeIncompleteError(CheckerDispatchError):
    """An envelope declares read roots its own items cannot dereference.

    AG-9 (200a: m8590 seq 296, m8635, m11578, m5445, m4334 x2): seven
    ``cannot-verify`` complaints were machinery faults -- the declared
    ``allowed_read_roots`` excluded or no longer contained the bytes the
    checker was asked to recompute -- yet they were charged to the MODEL as
    repair rounds. This refusal fires at envelope build, BEFORE any provider
    session is bought: an incomplete envelope produces no session, no durable
    gate row, and therefore no chargeable round (a round that never becomes a
    durable gate can never charge -- the complaint-disclosure doctrine's
    existing boundary). The fault parks on the checker lane as a lane
    contract failure, never a model complaint.

    Parameters
    ----------
    message:
        Human-readable refusal.
    stable_ids:
        Item identities whose roots failed the probe.
    unreadable_paths:
        The exact declared paths that did not dereference.
    """

    def __init__(
        self,
        message: str,
        *,
        stable_ids: tuple[str, ...] = (),
        unreadable_paths: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.stable_ids = stable_ids
        self.unreadable_paths = unreadable_paths


class CheckerCitationJoinError(CheckerDispatchError):
    """A checker purchase cites a source-manifest era it is not shipping.

    D-51 (m6484 class): a checker adjudicating against a re-frozen or rotated
    source manifest renders a wrong-era verdict -- the facts it validates were
    cited from bytes that are no longer the bytes in its envelope. The join is
    validated BEFORE the session is bought: the identity of the manifest
    actually shipped is recomputed from its rows and must equal the identity
    the adjudicated artifact BINDS. The refusal names both identities (cited
    and current), so a rotated-fact rig produces a named diff rather than a
    silent wrong-era adjudication.

    Parameters
    ----------
    message:
        Human-readable refusal.
    stable_id:
        Item whose citation era failed the join.
    bound_identity, current_identity:
        The identity the artifact binds vs the shipped manifest's recompute.
    """

    def __init__(
        self,
        message: str,
        *,
        stable_id: str = "",
        bound_identity: str = "",
        current_identity: str = "",
    ) -> None:
        super().__init__(message)
        self.stable_id = stable_id
        self.bound_identity = bound_identity
        self.current_identity = current_identity


def _shipped_manifest_identity(source_manifest: Mapping[str, Any]) -> str:
    """Return the shipped manifest's identity under the shared derivation rule.

    Byte-identical to the author-dispatch/gates derivation: an explicit
    ``manifest_sha256``/``source_manifest_identity`` field wins, else the
    stable hash of the frozen ``sources`` rows.
    """

    explicit = source_manifest.get("manifest_sha256") or source_manifest.get(
        "source_manifest_identity"
    )
    return (
        str(explicit) if explicit is not None else stable_hash(source_manifest.get("sources", []))
    )


def validate_citation_source_join(
    *,
    stable_id: str,
    source_manifest: Mapping[str, Any],
    bound_identity: str,
) -> None:
    """D-51 era pin: refuse a purchase whose manifest era moved under it.

    Parameters
    ----------
    stable_id:
        Adjudicated item identity (for the refusal).
    source_manifest:
        The manifest actually shipped in the envelope.
    bound_identity:
        The source-manifest identity the adjudicated artifact binds.

    Raises
    ------
    CheckerCitationJoinError
        The shipped manifest's recomputed identity differs from the bound one.
    """

    current_identity = _shipped_manifest_identity(source_manifest)
    if current_identity != bound_identity:
        raise CheckerCitationJoinError(
            f"gate-citation era mismatch for {stable_id}: the artifact binds "
            f"source-manifest identity {bound_identity} but the envelope ships a "
            f"manifest whose identity is {current_identity}; adjudicating cited "
            "facts against a different manifest era is the m6484 class -- refused "
            "before a session is bought",
            stable_id=stable_id,
            bound_identity=bound_identity,
            current_identity=current_identity,
        )


def _require_citation_source_joins(items: Sequence[Mapping[str, Any]]) -> None:
    """Validate every item's manifest era pin, where the item binds one.

    An item that ships a manifest without binding an identity (older
    fixtures, lanes whose binding travels elsewhere) is outside this pin's
    contract; the result-time joins still govern it.
    """

    for item in items:
        source_manifest = item.get("source_manifest")
        if not isinstance(source_manifest, Mapping):
            continue
        proposal = item.get("proposal")
        bound = None
        if isinstance(proposal, Mapping):
            bound = proposal.get("source_manifest_identity")
        if bound is None:
            bound = item.get("source_manifest_identity")
        if not isinstance(bound, str) or not bound:
            continue
        validate_citation_source_join(
            stable_id=str(item.get("stable_id")),
            source_manifest=source_manifest,
            bound_identity=bound,
        )


def _require_readable_envelope_roots(items: Sequence[Mapping[str, Any]]) -> None:
    """Refuse, pre-purchase, an envelope whose declared roots cannot be read.

    The machine-derived completeness floor over the same inputs
    :func:`_source_read_roots` declares: every item's author ``model_dir``
    and its frozen ``source-cas`` sibling must exist as readable directories
    at build time, or every recomputable digest in the item is guaranteed to
    die as a machinery ``cannot-verify``. Zero producer input steers this --
    the probe re-derives the paths from the items and asks the filesystem.

    Parameters
    ----------
    items:
        Normalized envelope items (each already carries ``model_dir``).

    Raises
    ------
    CheckerEnvelopeIncompleteError
        At least one declared root does not dereference to a readable
        directory.
    """

    failed_ids: list[str] = []
    unreadable: list[str] = []
    for item in items:
        declared = item.get("model_dir")
        if not isinstance(declared, str) or not declared.strip():
            # A MISSING declaration is a different (older) contract breach:
            # _source_read_roots refuses it with its established message.
            continue
        model_dir = Path(declared)
        cas_root = model_dir.parent / SOURCE_CAS_DIRNAME
        missing = [str(path) for path in (model_dir, cas_root) if not path.is_dir()]
        if missing:
            failed_ids.append(str(item.get("stable_id")))
            unreadable.extend(missing)
    if failed_ids:
        raise CheckerEnvelopeIncompleteError(
            "checker envelope is incomplete before purchase: items "
            f"{failed_ids} declare read roots that do not dereference "
            f"({sorted(set(unreadable))}); an unreadable preimage root is a "
            "machinery fault -- refused before a session is bought, never "
            "charged to the model as a repair round",
            stable_ids=tuple(failed_ids),
            unreadable_paths=tuple(sorted(set(unreadable))),
        )


class CheckerPromptRotatedError(CheckerDispatchError):
    """Raised when the checker prompt file changed under a running leg.

    BL-10 (e) / audit CH-F3: prompt identity used to be read LIVE from the
    working tree at every seam, so a mid-campaign prompt edit split the brain --
    already-accepted gates stayed durable while the runs-award door demanded the
    NEW live hash, converting every gated-but-unawarded model into full-cost
    unawardable work (the pilot carries THREE prompt digests, and prompt
    rotation was the proximate cause of the award-eating incident). The prompt
    is now PINNED once per driver leg at the first envelope build; a live file
    that later disagrees is this typed refusal at dispatch, so rotation can only
    happen by starting a new leg -- exactly the LADDER_PROTOCOL "flips only at
    leg boundaries" contract, now enforced by the machine instead of stated as
    a convention.
    """


class CheckerItemContractError(CheckerDispatchError):
    """A result refusal caused by ONE named item's author-side artifact.

    BL-10 (c) / audit B-3: 67% of all pilot verdict-contract refusals (156/233)
    were the write-contract replay refusing ONE fully-accurate item because its
    AUTHOR bound evidence that does not support the claim -- a fact about that
    model's frozen author artifact, not about the checker or its siblings. As a
    bare :class:`CheckerDispatchError` this destroyed the whole envelope: the
    wrapper spent its contract re-ask (provably futile -- the checker re-answers
    from the same frozen author bytes), exited 64, and the driver downshifted
    all five members through the halving ladder, mis-attributing the park to the
    checker lane. The typed subclass carries the offending ``stable_ids`` so the
    wrapper can skip the futile re-ask and the driver can scope the failure to
    exactly the named models while the innocent siblings re-batch.
    """

    def __init__(self, message: str, *, stable_ids: Sequence[str], author_side: bool) -> None:
        """Bind the item-scoped refusal to the models whose artifacts caused it.

        Parameters
        ----------
        message:
            Exact validator refusal text.
        stable_ids:
            The models whose items caused the refusal.
        author_side:
            ``True`` when the defect is a fact about the item's frozen AUTHOR
            artifact (the write-contract replay), where a contract re-ask is
            provably futile; ``False`` for a checker-authored per-item defect
            (complaint-disclosure enumeration), which a fresh bounded re-ask
            can legitimately repair.
        """

        super().__init__(message)
        self.stable_ids: tuple[str, ...] = tuple(str(value) for value in stable_ids)
        self.author_side = bool(author_side)


class CheckerDigestAttestationError(CheckerItemContractError):
    """The machine digest attestation refutes a bound digest BEFORE any session.

    F51 checker-lifecycle D8 (probe 4.3.2): the rung-adverse golden's bound
    ``verified_hashes.code`` does not match the staged ``model_dir`` bytes; sol
    caught it 3/3, Fable 1/3, and a miss makes no claim, so no tripwire saw
    the class. The acceptrate replay found it on 9 of 81 packs. A bound
    digest is the frozen pack's own claim about its bytes, so a disagreement
    with the staged bytes is a fact about that item's author artifact (never
    a checker judgment and never repairable by a re-ask): the envelope
    refuses pre-spend as an item-scoped ``author_side`` contract failure,
    naming exactly the mismatched keys, and the innocent siblings re-batch.
    """

    def __init__(
        self, message: str, *, stable_ids: Sequence[str], mismatched_keys: Sequence[str]
    ) -> None:
        """Bind the refusal to the item and the digest keys the machine refuted."""

        super().__init__(message, stable_ids=stable_ids, author_side=True)
        self.mismatched_keys: tuple[str, ...] = tuple(str(key) for key in mismatched_keys)


class CheckerBindingMismatchError(CheckerDispatchError):
    """Raised when a result item's echoed identity binding differs from its envelope.

    The envelope-bound identity fields are machine-known constants the checker is
    told to copy verbatim, so a mismatch here is the one contract violation with a
    stochastic cause the checker can repair on a fresh attempt: a generative model
    mis-transcribing one identity string (rung 3, batch ``0f44a52d3e324c4c``:
    ``vet_identity`` flubbed for one of two items while every sibling echo in the
    same run round-tripped exactly). The comparison itself is never relaxed -- a
    mismatched result is refused unconditionally -- but the operator wrapper is
    allowed to distinguish this refusal from a structural contract violation and
    spend its remaining bounded attempts on a re-ask instead of permanently
    failing every batch member on the first flub.
    """


class CheckerWallBudgetExceeded(CheckerDispatchError):
    """A named model exhausted its per-model checker wall budget (BL-10 g).

    Raised BEFORE a session is bought, from the command lane's pre-dispatch
    budget check, when an envelope contains a model whose accumulated checker
    wall this leg already exceeds
    :data:`menagerie.crawler.constants.CHECKER_MODEL_WALL_BUDGET_SECONDS`.
    Routed down the halving ladder like every other lane failure so innocent
    batch siblings re-dispatch; at the floor the over-budget model TERMINALIZES
    ``failed:accuracy-gate / checker-session-exhaustion`` under mandatory human
    review. Generation-scoped exhaustion is permanent for the generation, so
    the historical run-scoped ``checker-wall-exhausted`` park re-entered and
    re-parked forever with no reviewable head a requeue grant could bind (the
    pilot's five no-exit parks, 2026-08-21); only TRANSIENT lane weather parks.
    """

    def __init__(self, message: str, *, stable_ids: Sequence[str]) -> None:
        """Name the models whose budgets are exhausted."""

        super().__init__(message)
        self.stable_ids: tuple[str, ...] = tuple(str(value) for value in stable_ids)


class CheckerPartialResultError(CheckerDispatchError):
    """Raised when a checker result omits or invents batch members.

    This typed refusal is the shared seam for the checker failure route: callers
    can downshift the batch before spending the bounded contract re-ask without
    parsing validator prose or weakening the atomic result contract.
    """


class CheckerMechanicalClaimError(CheckerDispatchError):
    """A checker result asserts a claim the machine authority refutes (W-8).

    The rule-2 evidence: the one root-caused acceptance flip was a draw
    asserting 21 mechanically false claims about frozen bytes, and it
    MASS-REJECTED a clean pack -- a bad reject is invisible because it looks
    exactly like the gate working. Production therefore machine-checks every
    checker claim that is machine-decidable at the write door; a refuted claim
    is a TYPED CHECKER DEFECT that discards the whole result (never a member
    verdict, in either direction) and the driver parks the named items as
    ``checker-verdict-conflict`` -- instability is operator property, never a
    silent re-roll and never a model-blaming terminal.

    The closed write-time claim-class registry
    (:func:`_validate_item_mechanical_claims`):

    * ``hash-claim-binding`` -- every ``integrity.hash_mismatches`` entry is
      the closed structured claim ``{"digest_key": <str>, "detail":
      <optional str>}`` with no other keys (gate.v3 types the array as
      strings, so the object travels as its JSON serialization; a decoded
      mapping is adjudicated identically). ``digest_key`` must be an EXACT
      member of the envelope's machine-frozen ``verified_hashes`` keys bound
      to a non-null digest: free prose, unknown keys, and
      superstring/paraphrase digest names are each a typed refusal, because
      free text does not bind a claim to its mechanical authority (an
      incidental bound key occurring as a substring of unrelated prose
      licensed nothing). (Semantic mismatch adjudication over the staged
      bytes stays checker-owned -- AUTHORITY clause 1 -- and is NOT refuted
      here.)
    * ``locator-citation-license`` -- when the machine attestation discloses
      non-valid rows, every ``locator_failures`` / ``excerpt_discrepancies``
      entry is the closed structured claim ``{"evidence_id": <str>,
      "status": <str>, "claim_kind": <str>, "detail": <optional str>}`` with
      no other keys (serialized as above). ``claim_kind`` is the closed
      vocabulary ``locator-failure`` / ``excerpt-discrepancy`` and must match
      the array the entry sits in. The ALLEGED row is ``evidence_id``
      itself: it must name a present NON-valid attestation row, and
      ``status`` must equal that row's attested status exactly. An
      allegation about a ``valid`` row, an absent row, or with a wrong
      status disputes machine arithmetic -- a false allegation about one row
      can no longer be laundered by citing a different, genuinely non-valid
      row. (The all-valid arm is :class:`CheckerLocatorAuthorityError`'s.)
    * ``r4-search-marker-correlation`` -- every ``search-attested:`` /
      ``search-receipt-attested:`` marker in ``rung_check.findings`` must name
      a link of the frozen closed set ``search_report.links_checked`` with the
      route matching the entry form (plain URL re-fetched vs typed
      broker-receipt outcome; a swapped route is mechanically refused, as the
      prompt has always promised), and an ``accurate`` rung check on an
      R4_REIMPLEMENT item must attest EVERY link -- the R4 claim is a negative
      proof over exactly that closed set, and a spot-checked accurate verdict
      blesses the members never examined.

    Era: every class binds through the LIVE envelope's machine attestation
    stamp (``locator_attestation``), which recorded envelopes never carry --
    append-only history keeps rehydrating at its recorded basis.
    """

    def __init__(self, message: str, *, stable_ids: Sequence[str]) -> None:
        """Name the items whose machine-refuted claims discard the result."""

        super().__init__(message)
        self.stable_ids: tuple[str, ...] = tuple(str(value) for value in stable_ids)


class CheckerLocatorAuthorityError(CheckerMechanicalClaimError):
    """A checker disputed the machine locator attestation (rule-2, 2026-08-29).

    Excerpt-locator and excerpt-equivalence adjudication is MACHINE-DECIDABLE
    and withdrawn from model judgment: the envelope's coordinator-frozen
    ``locator_attestation`` rows are the only authority on whether an excerpt
    sits at its declared locator under the pipeline's byte equivalence. When
    every row for an item is ``valid``, the checker has no output field
    capable of creating a locator failure -- a result alleging one in
    ``integrity.locator_failures`` or ``integrity.excerpt_discrepancies`` is
    disputing machine arithmetic (the m10470 se2 shape: 21 fabricated line
    claims mass-rejecting a clean pack; the m10759 terminal se2 shape: byte
    identity silently imposed on the documented whitespace equivalence). The
    refusal is a TYPED CHECKER DEFECT that discards the whole result -- it is
    never published as a member verdict, so a fabricated locator claim can no
    longer steer any acceptance bit in either direction.
    """


class CheckerDigestAuthorityError(CheckerMechanicalClaimError):
    """A checker disputed the machine digest attestation (F51 D8, 2026-09-01).

    Digest verification is MACHINE-DECIDABLE and withdrawn from model
    judgment: the envelope's coordinator-frozen ``digest_attestation`` rows
    are the only authority on whether each ``verified_hashes`` key matches
    the frozen bytes it binds. A ``mismatch`` never reaches a session (it
    refuses at build), so at dispatch every row is ``valid`` or
    ``unresolvable``. An ``integrity.hash_mismatches`` entry naming a
    ``valid`` row (or a key the attestation never disclosed as non-valid)
    disputes machine arithmetic -- shell shasum by a model against the
    coordinator's recompute -- and is a TYPED CHECKER DEFECT discarding the
    whole result, never a member verdict in either direction. Citing an
    ``unresolvable`` row is licensed: that row is the attested evidence of
    exactly what could not be adjudicated.
    """


class CheckerDeviationAuthorityError(CheckerMechanicalClaimError):
    """A checker disputed the machine declaration anchor (rule-2 round 3).

    Whether a proposal DECLARES a deviation is a machine fact frozen into the
    fidelity envelope's ``declared_deviation_basis`` stamp (every
    ``fidelity.deviations`` row plus every author-chosen
    ``implementation.declared_choices`` row -- the closed derivation that ended
    the m10759 which-field-owns-deviations split). The match/minor-drift
    precedence is closed on that anchor: when ``any_declared`` is true, no
    reading of the pack satisfies match's "declares NO deviation of any kind".
    A result returning ``match`` against a nonempty stamped basis is disputing
    machine arithmetic; like the locator authority, the refusal is a TYPED
    CHECKER DEFECT discarding the whole result, never a member verdict.
    """

    def __init__(self, message: str, *, stable_ids: Sequence[str]) -> None:
        """Name the items whose machine declaration anchor was disputed."""

        super().__init__(message, stable_ids=stable_ids)


def downshift_metadata_items(
    items: Sequence[_MetadataItemT],
) -> tuple[tuple[_MetadataItemT, ...], ...]:
    """Split one refused metadata batch along the halving ladder.

    Parameters
    ----------
    items:
        Metadata item packs from the refused checker envelope.

    Returns
    -------
    tuple[tuple[_MetadataItemT, ...], ...]
        Stable-order halves. A singleton remains a singleton so callers can
        identify the floor without manufacturing an empty batch.
    """

    batch = tuple(items)
    if len(batch) <= 1:
        return (batch,) if batch else ()
    midpoint = (len(batch) + 1) // 2
    return (batch[:midpoint], batch[midpoint:])


def checker_usage_tokens(stdout: str) -> tuple[Optional[int], Optional[int]]:
    """Extract the final completed turn's input and output token counts.

    Parameters
    ----------
    stdout:
        Codex JSONL event stream for one checker attempt.

    Returns
    -------
    tuple[Optional[int], Optional[int]]
        ``(input_tokens, output_tokens)`` from the last well-formed
        ``turn.completed`` event, or ``(None, None)`` when none is available.
    """

    measured: tuple[Optional[int], Optional[int]] = (None, None)
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("type") != "turn.completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            continue
        raw_input = usage.get("input_tokens")
        raw_output = usage.get("output_tokens")
        measured = (
            raw_input if isinstance(raw_input, int) and raw_input >= 0 else None,
            raw_output if isinstance(raw_output, int) and raw_output >= 0 else None,
        )
    return measured


@dataclass(frozen=True)
class CheckerBackoffSignal:
    """Typed rate/quota pause signal for the later wakeup layer.

    Parameters
    ----------
    reason:
        Closed rate-limit or quota-exhaustion reason.
    retry_after_seconds:
        Provider retry delay when supplied.
    reset_at:
        Provider reset timestamp when supplied.
    response_excerpt:
        Bounded diagnostic response text.
    provider:
        Closed usage-limit provider vocabulary member owning the pause: the
        provider of the seated checker model the paused session ran under
        (``anthropic`` for the Fable primary seat, ``openai`` for the sol
        confirmation seat). Explicit rather than assumed by the scheduler; the
        default is the pre-swap era value.
    """

    reason: CheckerPauseReason
    retry_after_seconds: Optional[int]
    reset_at: Optional[str]
    response_excerpt: str
    provider: str = "openai"


def build_metadata_vet_envelope(
    items: Sequence[Mapping[str, Any]],
    *,
    gate_round: int,
    output_path: Union[str, Path],
    checker_model: str,
    checker_version: str,
    request_nonce: str,
    final_tail: bool = False,
) -> JsonObject:
    """Build a fresh metadata-vetting envelope, including an explicit final tail.

    Parameters
    ----------
    items:
        Independent item packs containing proposal/source/evidence artifacts and
        the exact identity fields expected in each result.
    gate_round:
        Bounded metadata repair round.
    output_path:
        Exact final checker result path.
    checker_model, checker_version:
        Expected independent checker identity.
    request_nonce:
        Caller-created fresh request identity.
    final_tail:
        Whether this is the end-of-queue flush that may contain one to nine items.

    Returns
    -------
    dict[str, Any]
        Hash-bound metadata checker envelope.

    Raises
    ------
    CheckerDispatchError
        If batch cardinality or item identities are invalid.
    """

    minimum = METADATA_FINAL_TAIL_MIN if final_tail else METADATA_BATCH_MIN
    if not minimum <= len(items) <= METADATA_BATCH_MAX:
        raise CheckerDispatchError(
            f"metadata batch must contain {minimum}--{METADATA_BATCH_MAX} items"
        )
    return _build_envelope(
        GateKind.METADATA_BATCH,
        items,
        gate_round=gate_round,
        output_path=output_path,
        checker_model=checker_model,
        checker_version=checker_version,
        request_nonce=request_nonce,
        final_tail=final_tail,
    )


def build_fidelity_envelope(
    item: Mapping[str, Any],
    *,
    gate_round: int,
    output_path: Union[str, Path],
    checker_model: str,
    checker_version: str,
    request_nonce: str,
) -> JsonObject:
    """Build one fresh per-model fidelity checker envelope.

    Parameters
    ----------
    item:
        Complete per-model proposal/source/evidence/code pack.
    gate_round:
        Fidelity review round.
    output_path:
        Exact final checker result path.
    checker_model, checker_version:
        Expected independent checker identity.
    request_nonce:
        Caller-created fresh request identity.

    Returns
    -------
    dict[str, Any]
        Hash-bound fidelity checker envelope.
    """

    return _build_envelope(
        GateKind.FIDELITY,
        [item],
        gate_round=gate_round,
        output_path=output_path,
        checker_model=checker_model,
        checker_version=checker_version,
        request_nonce=request_nonce,
    )


def build_terminal_disposition_envelope(
    item: Mapping[str, Any],
    *,
    gate_round: int,
    output_path: Union[str, Path],
    checker_model: str,
    checker_version: str,
    request_nonce: str,
) -> JsonObject:
    """Build one exact typed terminal-recommendation checker envelope.

    Parameters
    ----------
    item:
        Complete author-result/source/evidence/license pack.
    gate_round, output_path, checker_model, checker_version, request_nonce:
        Fresh checker request bindings.

    Returns
    -------
    dict[str, Any]
        Hash-bound terminal-disposition envelope.
    """

    return _build_envelope(
        GateKind.TERMINAL_DISPOSITION,
        [item],
        gate_round=gate_round,
        output_path=output_path,
        checker_model=checker_model,
        checker_version=checker_version,
        request_nonce=request_nonce,
    )


def machine_owned_gate_fields(envelope: Mapping[str, Any]) -> JsonObject:
    """Derive every gate field the machine owns and a checker must never invent.

    Parameters
    ----------
    envelope:
        Hash-bound metadata, fidelity, or terminal request envelope.

    Returns
    -------
    dict[str, Any]
        Deterministic gate scaffold. The two ``LEDGER_ASSIGNED_GATE_FIELDS`` are
        deliberately absent: the ledger owns them and assigns them at append time.

    Raises
    ------
    CheckerDispatchError
        If the envelope binding is invalid or incomplete.
    """

    _validate_envelope_hash(envelope)
    items = envelope.get("items")
    if not isinstance(items, list) or not items:
        raise CheckerDispatchError("checker envelope has no items")
    envelope_sha256 = str(envelope.get("envelope_sha256"))
    gate_seed = stable_hash(
        {"envelope_sha256": envelope_sha256, "request_nonce": envelope.get("request_nonce")}
    )
    return {
        "schema_version": str(envelope.get("required_result_schema")),
        "gate_id": GATE_ID_PREFIX + gate_seed.removeprefix("sha256:")[:32],
        "gate_kind": envelope.get("gate_kind"),
        "batch_size": len(items),
        "gate_round": envelope.get("gate_round"),
        "gate_identity": envelope_sha256,
        "author_result_schema_identity": component_identity(AUTHOR_RESULT_SCHEMA_COMPONENT),
        "dispatcher_identity": component_identity(AUTHOR_DISPATCHER_COMPONENT),
    }


def apply_machine_owned_gate_fields(
    result: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    started_at: str,
    finished_at: str,
) -> JsonObject:
    """Stamp the machine-owned scaffold onto one decoded checker verdict.

    The checker's authority is its verdict, its findings, and its per-field
    reasoning. Identities and wall timings are the machine's. Stamping happens
    BEFORE validation so a substantively complete verdict is never discarded for
    a scaffold field the model had no way to observe, and it is unconditional so
    a model-supplied identity can never be believed. On a ``metadata_batch``
    gate the per-item ``fidelity`` block is machine-owned too -- a gate-kind
    constant with no judgment in it -- and is machine-filled on every item.

    OMITTING a machine-owned field is free and always has been. SUPPLYING one
    with a conflicting value is refused, because the stamp cannot both discard
    a fabricated identity and leave evidence that one was fabricated.

    The ``LEDGER_ASSIGNED_GATE_FIELDS`` are stricter still: they are refused on
    ANY value, because there is no value the checker could legitimately hold. The
    ledger has not run yet, so the correct sequence and payload hash do not exist
    at stamp time -- anything present in those slots was invented.

    Parameters
    ----------
    result:
        Decoded candidate gate carrying the checker's own verdict.
    envelope:
        Hash-bound request envelope owning the scaffold.
    started_at, finished_at:
        Wrapper-observed UTC timestamps for the checker round trip.

    Returns
    -------
    dict[str, Any]
        Scaffolded gate whose ``result_envelope_sha256`` binds the stamped body.

    Raises
    ------
    CheckerDispatchError
        If the envelope binding is invalid, the candidate is not an object, or
        the candidate supplies a machine-owned field with a conflicting value.
    """

    if not isinstance(result, Mapping):
        raise CheckerDispatchError("checker result must contain exactly one JSON object")
    stamped = dict(result)
    scaffold = machine_owned_gate_fields(envelope)
    expected_checker = _required_mapping(envelope.get("checker"), "envelope checker")
    checker_scaffold = {
        "provider": expected_checker.get("provider"),
        "model": expected_checker.get("model"),
        "version": expected_checker.get("version"),
        "prompt_sha256": expected_checker.get("prompt_sha256"),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    supplied_checker = stamped.get("checker")
    if supplied_checker is not None and not isinstance(supplied_checker, Mapping):
        raise CheckerDispatchError("checker result checker field must be one JSON object")
    # Refuse BEFORE stamping. Omission stays free -- that tolerance is the
    # 2026-07-29 fix and is not being narrowed. What is refused is a CONFLICTING
    # supplied value, and the distinction matters because the stamp below is
    # unconditional: it used to overwrite a model-supplied scaffold silently,
    # which meant the two checks that already exist downstream in
    # ``validate_checker_result_mapping`` -- the
    # ``DETERMINISTIC_GATE_SCAFFOLD_FIELDS`` comparison and the four-field
    # ``checker`` comparison -- could never fail, because they ran against
    # values the stamp had just made correct by construction. Those checks were
    # structurally dead. Refusing here restores them, and it also closes the
    # near-miss they were written for: a checker that templates its answer from
    # a repository test fixture arrives carrying that fixture's placeholder
    # verification values, and silently rewriting them to the machine's real
    # ones launders a fabricated gate into a well-formed one.
    _refuse_conflicting_machine_owned(scaffold, stamped, prefix="")
    if isinstance(supplied_checker, Mapping):
        _refuse_conflicting_machine_owned(checker_scaffold, supplied_checker, prefix="checker.")
    # AFTER the scaffold comparison, deliberately: that check names the offending
    # field and both values, and running the ledger refusal first would shadow it
    # for any candidate that happens to carry both kinds of fabrication.
    _refuse_ledger_assigned_fields(stamped, "checker result")
    stamped.update(scaffold)
    # Per-item machine fill: on a metadata gate the whole ``fidelity`` block is a
    # gate-kind CONSTANT (gate.v3 pins required=false and verdict=not-applicable;
    # nothing in it is a judgment), so it is stamped unconditionally, like the
    # scaffold. Unlike the identity scaffold, a supplied divergent value is
    # NORMALIZED rather than refused: refusing is what turned one information-free
    # slip into a whole-batch ``failed:runner/protocol-violation`` on rung 8
    # (m10517 + m9666, batch ``metadata-c001cad863d95a67``), and overwriting a
    # value that depends on nothing the checker observed cannot launder anything
    # -- the machine value is correct independent of what was supplied. The
    # divergence itself is still surfaced: the wrapper records
    # ``metadata_fidelity_normalizations`` as telemetry before this stamp.
    if scaffold.get("gate_kind") == GateKind.METADATA_BATCH.value:
        raw_items = stamped.get("items")
        if isinstance(raw_items, list):
            stamped["items"] = [
                {**item, "fidelity": metadata_fidelity_state()}
                if isinstance(item, Mapping)
                else item
                for item in raw_items
            ]
    checker = dict(supplied_checker) if isinstance(supplied_checker, Mapping) else {}
    checker.update(checker_scaffold)
    stamped["checker"] = checker
    stamped = _seal_shadow_rehearsal_marker(stamped)
    stamped["result_envelope_sha256"] = compute_result_envelope_sha256(stamped)
    return stamped


def _seal_shadow_rehearsal_marker(stamped: JsonObject) -> JsonObject:
    """Seal the machine-owned W3 shadow marker before the envelope digest.

    The shadow-rehearsal marker is derived from the leg's declared mode,
    never from the checker's output -- so a checker SUPPLYING it is refused
    on presence alone, like the ledger-assigned fields. Sealing happens
    BEFORE ``result_envelope_sha256`` is computed, so the marker sits inside
    the gate's own self-hash and cannot be stripped without invalidating it.

    Parameters
    ----------
    stamped:
        Scaffolded gate about to receive its result-envelope digest.

    Returns
    -------
    dict[str, Any]
        The gate, marker-stamped when the shadow-rehearsal mode is active.

    Raises
    ------
    CheckerDispatchError
        If the checker output itself supplied the machine-owned marker.
    """

    if SHADOW_REHEARSAL_MARKER_FIELD in stamped:
        raise CheckerDispatchError(
            f"checker result supplies the machine-owned {SHADOW_REHEARSAL_MARKER_FIELD!r} "
            "marker: the shadow-rehearsal marker is stamped from the leg's declared "
            "mode, never accepted from checker output"
        )
    return stamp_shadow_rehearsal(stamped)


def _refuse_ledger_assigned_fields(supplied: Mapping[str, Any], origin: str) -> None:
    """Refuse one gate that carries a field only the ledger may assign.

    Parameters
    ----------
    supplied:
        Candidate gate object.
    origin:
        Human-readable description of where the candidate came from.

    Raises
    ------
    CheckerDispatchError
        If any ``LEDGER_ASSIGNED_GATE_FIELDS`` entry is present. Presence alone is
        the offence: the append has not happened, so no correct value exists yet
        and whatever is in the slot was invented. Refusing rather than overwriting
        keeps the ledger's own next-sequence check reachable -- silently rewriting
        the value is what made the constant-``1`` stamp survive five rungs.
    """

    for field in LEDGER_ASSIGNED_GATE_FIELDS:
        if field not in supplied:
            continue
        raise CheckerDispatchError(
            f"{origin} supplies the ledger-assigned field {field}="
            f"{bounded_json_repr(supplied[field])}; {field} is assigned by the ledger from its "
            "own state at append time and does not exist before then, so any value here was "
            "invented rather than derived"
        )


def _with_pending_ledger_fields(gate: Mapping[str, Any]) -> JsonObject:
    """Return one schema-shaped copy of a gate that has not been appended yet.

    Parameters
    ----------
    gate:
        Stamped gate without its ledger-assigned fields.

    Returns
    -------
    dict[str, Any]
        Copy whose ABSENT ledger fields carry stand-in values, so ``gate.v3``'s
        ``required`` list is satisfiable before the append that will supply the
        real ones. A field the candidate actually supplied is left as supplied, so
        an out-of-range value still fails its own schema constraint rather than
        being masked by the stand-in. The copy is validated and discarded; it is
        never hashed, published, or appended.
    """

    return {**_PENDING_LEDGER_GATE_FIELDS, **gate}


def _refuse_conflicting_machine_owned(
    scaffold: Mapping[str, Any], supplied: Mapping[str, Any], *, prefix: str
) -> None:
    """Refuse one checker result that supplies a conflicting machine-owned field.

    Parameters
    ----------
    scaffold:
        Machine-owned field values for this exact envelope.
    supplied:
        Model-authored candidate object at the same nesting level.
    prefix:
        Dotted path prefix used to name the offending field.

    Raises
    ------
    CheckerDispatchError
        If any machine-owned field is present with a value that is not the
        machine's. The message names the field and BOTH values, because the
        supplied value is the evidence: a fixture-templated gate carries the
        fixture's placeholders there, and a message that only said "invalid"
        would discard the one fact that identifies what went wrong.
    """

    for field, machine_value in scaffold.items():
        if field not in supplied:
            continue
        if supplied[field] == machine_value:
            continue
        raise CheckerDispatchError(
            f"checker supplied the machine-owned field {prefix}{field}="
            f"{bounded_json_repr(supplied[field])} but the machine-owned value for this "
            f"envelope is {bounded_json_repr(machine_value)}; the checker's authority is its "
            "verdict, not the gate scaffold, so a conflicting scaffold value is evidence the "
            "gate was templated rather than derived"
        )


def component_identity(relative: str) -> str:
    """Hash one exact shipped crawler component by its package-relative path.

    Parameters
    ----------
    relative:
        Package-relative path of a shipped authority component.

    Returns
    -------
    str
        Content identity, computed identically to the authority context.

    Raises
    ------
    CheckerDispatchError
        If the component is unavailable.
    """

    try:
        return hash_bytes((Path(__file__).parent / relative).read_bytes())
    except OSError as exc:
        raise CheckerDispatchError(
            f"checker authority component is unavailable: {relative}"
        ) from exc


def validate_checker_result(
    result_path: Union[str, Path], envelope: Mapping[str, Any]
) -> JsonObject:
    """Validate one complete gate result and every independently bound item.

    Parameters
    ----------
    result_path:
        Expected final result path.
    envelope:
        Metadata or fidelity request envelope.

    Returns
    -------
    dict[str, Any]
        Schema-valid immutable ``gate.v2`` record for reducer emission.

    Raises
    ------
    CheckerDispatchError
        If the result is partial, stale, mismatched, or logically inconsistent.
    """

    _validate_envelope_hash(envelope)
    path = Path(result_path).resolve()
    if (
        path != Path(str(envelope.get("required_output_path"))).resolve()
        or path.name != "result.json"
    ):
        raise CheckerDispatchError("checker result is not at the exact atomic output path")
    return validate_checker_result_mapping(_read_json_object(path), envelope)


def validate_checker_result_mapping(
    result: Mapping[str, Any], envelope: Mapping[str, Any]
) -> JsonObject:
    """Validate one in-memory gate result before atomic publication.

    Parameters
    ----------
    result:
        Candidate complete gate result.
    envelope:
        Metadata, fidelity, or terminal request envelope.

    Returns
    -------
    dict[str, Any]
        Schema-valid immutable gate record safe to publish.

    Raises
    ------
    CheckerDispatchError
        If the result is partial, stale, mismatched, or logically inconsistent.
    """

    _validate_envelope_hash(envelope)
    normalized = dict(result)
    try:
        # Validate against a PENDING copy: gate.v3 requires the ledger-assigned
        # fields, but this gate has not been appended, so the real values do not
        # exist yet. The stand-ins prove the shape; ``normalized`` -- the object
        # actually returned, hashed, and handed to the reducer -- stays free of
        # them so the ledger can assign from its own state.
        validate_payload(_with_pending_ledger_fields(normalized), GATE_SCHEMA_VERSION_V3)
    except PayloadValidationError as exc:
        raise CheckerDispatchError(str(exc)) from exc
    scaffold = machine_owned_gate_fields(envelope)
    for field in DETERMINISTIC_GATE_SCAFFOLD_FIELDS:
        if normalized.get(field) != scaffold[field]:
            raise CheckerDispatchError(
                f"checker result {field} is not the machine-owned value for its envelope"
            )
    # After the scaffold comparison, so a candidate carrying both kinds of
    # fabrication is still named by the more specific message first.
    _refuse_ledger_assigned_fields(normalized, "checker result")
    result_checker = _required_mapping(normalized.get("checker"), "result checker")
    expected_checker = _required_mapping(envelope.get("checker"), "envelope checker")
    for field in ("provider", "model", "version", "prompt_sha256"):
        if result_checker.get(field) != expected_checker.get(field):
            raise CheckerDispatchError(f"checker result {field} does not match its envelope")
    result_items = normalized.get("items")
    expected_items = envelope.get("items")
    if not isinstance(result_items, list) or not isinstance(expected_items, list):
        raise CheckerDispatchError("checker items are incomplete")
    if (
        normalized.get("gate_kind") == GateKind.METADATA_BATCH.value
        and len(expected_items) < METADATA_BATCH_MIN
        and envelope.get("final_tail") is not True
    ):
        raise CheckerDispatchError("short metadata result is not bound to a final-tail envelope")
    if normalized.get("batch_size") != len(expected_items) or len(result_items) != len(
        expected_items
    ):
        raise CheckerPartialResultError("checker result is partial or has extra items")
    expected_by_id = {
        str(item["stable_id"]): item for item in expected_items if isinstance(item, Mapping)
    }
    if len(expected_by_id) != len(expected_items):
        raise CheckerDispatchError("checker envelope contains duplicate stable IDs")
    seen: set[str] = set()
    for result_item in result_items:
        if not isinstance(result_item, Mapping):
            raise CheckerDispatchError("checker result item must be an object")
        stable_id = str(result_item.get("stable_id"))
        expected = expected_by_id.get(stable_id)
        if expected is None or stable_id in seen:
            raise CheckerDispatchError(f"unexpected or duplicate checker item: {stable_id}")
        seen.add(stable_id)
        _validate_item_binding(result_item, expected)
        _validate_item_decision(result_item, GateKind(str(normalized["gate_kind"])))
        _validate_item_locator_authority(result_item, expected, stable_id=stable_id)
        _validate_item_digest_authority(result_item, expected, stable_id=stable_id)
        _validate_item_deviation_authority(result_item, expected, stable_id=stable_id)
        _validate_item_mechanical_claims(result_item, expected, stable_id=stable_id)
        if normalized.get("gate_kind") == GateKind.METADATA_BATCH.value:
            _validate_item_write_contract(result_item, expected)
            _validate_item_complaint_disclosure(result_item, expected)
    expected_hash = compute_result_envelope_sha256(normalized)
    if normalized.get("result_envelope_sha256") != expected_hash:
        raise CheckerDispatchError("result_envelope_sha256 does not bind the complete gate result")
    return normalized


def _validate_item_locator_authority(
    result_item: Mapping[str, Any],
    envelope_item: Mapping[str, Any],
    *,
    stable_id: str,
) -> None:
    """Adjudicate a result item's locator claims against the machine authority.

    On an attested envelope item whose rows are ALL ``valid`` (invalid rows
    refuse at build; only ``unattestable`` rows can survive to dispatch), the
    checker has no output field capable of creating a locator or excerpt
    discrepancy: those adjudications are machine arithmetic. A nonempty
    ``integrity.locator_failures`` or ``integrity.excerpt_discrepancies``
    there is a typed authority conflict discarding the whole result. When any
    row is ``unattestable``, the arrays stay open -- the attestation itself
    is the licensed evidence of exactly what could not be adjudicated, and
    the checker must judge those rows. ``hash_mismatches`` is NOT this
    function's: on an envelope stamped with the machine digest attestation it
    is adjudicated by :func:`_validate_item_digest_authority`; on a legacy
    envelope without that stamp it stays checker-owned (hash re-verification
    was the checker's standing AUTHORITY clause 1). A legacy envelope without
    the locator attestation keeps historical semantics untouched.
    """

    attestation = envelope_item.get("locator_attestation")
    if not isinstance(attestation, Mapping):
        return
    rows = attestation.get("rows")
    rows = rows if isinstance(rows, list) else []
    all_valid = all(
        isinstance(row, Mapping) and row.get("status") == LOCATOR_ATTESTATION_VALID for row in rows
    )
    if not all_valid:
        return
    integrity = result_item.get("integrity")
    integrity = integrity if isinstance(integrity, Mapping) else {}
    disputes = [
        *(integrity.get("locator_failures") or []),
        *(integrity.get("excerpt_discrepancies") or []),
    ]
    if disputes:
        raise CheckerLocatorAuthorityError(
            f"checker item {stable_id} alleges {len(disputes)} locator/excerpt "
            "discrepancies against an envelope whose machine locator attestation "
            "verified every excerpt; excerpt-locator adjudication is machine-owned "
            "and a claim disputing it is a typed checker defect, never a member "
            f"verdict (first allegation: {str(disputes[0])[:200]!r})",
            stable_ids=[stable_id],
        )


def _validate_item_digest_authority(
    result_item: Mapping[str, Any],
    envelope_item: Mapping[str, Any],
    *,
    stable_id: str,
) -> None:
    """Adjudicate a result item's ``hash_mismatches`` against the digest attestation.

    On an envelope item stamped with ``digest_attestation`` (F51 D8), every
    entry must name a digest key whose attested status is NON-valid; an entry
    naming a ``valid`` row, or a key the attestation never disclosed as
    non-valid, disputes machine arithmetic and is a typed authority conflict
    discarding the whole result. Structural conformance of each entry (closed
    object, exact bound key) is the ``hash-claim-binding`` class in
    :func:`_validate_hash_mismatch_claims`; this function adds the citation
    license, mirroring ``locator-citation-license``. A legacy envelope without
    the stamp keeps ``hash_mismatches`` checker-owned, untouched.
    """

    attestation = envelope_item.get("digest_attestation")
    if not isinstance(attestation, Mapping):
        return
    rows = attestation.get("rows")
    non_valid = {
        str(row.get("digest_key")): str(row.get("status"))
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, Mapping) and row.get("status") != DIGEST_ATTESTATION_VALID
    }
    integrity = result_item.get("integrity")
    integrity = integrity if isinstance(integrity, Mapping) else {}
    for entry in integrity.get("hash_mismatches") or []:
        claim = _structured_integrity_claim(
            entry,
            stable_id=stable_id,
            array_name="hash_mismatches",
            allowed_keys=_HASH_MISMATCH_CLAIM_KEYS,
            required_keys=("digest_key",),
        )
        key = str(claim["digest_key"])
        if key not in non_valid:
            raise CheckerDigestAuthorityError(
                f"checker item {stable_id} alleges a hash mismatch on digest_key {key!r} "
                "against an envelope whose machine digest attestation verified that key "
                f"(non-valid rows: {sorted(non_valid) or 'none'}); digest verification is "
                "machine-owned and a claim disputing it is a typed checker defect, never "
                "a member verdict",
                stable_ids=[stable_id],
            )


def _validate_item_deviation_authority(
    result_item: Mapping[str, Any],
    envelope_item: Mapping[str, Any],
    *,
    stable_id: str,
) -> None:
    """Adjudicate a fidelity verdict against the machine declaration anchor.

    On a fidelity envelope item stamped with a nonempty
    ``declared_deviation_basis``, the checker has no verdict capable of
    asserting that the proposal declares nothing: ``match`` there disputes the
    machine-frozen anchor (the m10759 se3 shape: ``fidelity.deviations: []``
    read as the whole declaration while author-chosen ``declared_choices``
    rows sat beside it). WHICH non-match verdict applies -- minor-drift for
    demonstrably nonmaterial declared choices, major-drift for material ones,
    cannot-verify for undecidable ones -- remains entirely the checker's
    judgment. A legacy envelope without the stamp keeps historical semantics
    untouched.
    """

    basis = envelope_item.get("declared_deviation_basis")
    if not isinstance(basis, Mapping) or not basis.get("any_declared"):
        return
    fidelity = result_item.get("fidelity")
    fidelity = fidelity if isinstance(fidelity, Mapping) else {}
    if fidelity.get("verdict") == FidelityVerdict.MATCH.value:
        declared = int(basis.get("fidelity_deviation_entries") or 0) + len(
            basis.get("source_unspecified_choices") or []
        )
        raise CheckerDeviationAuthorityError(
            f"checker item {stable_id} returns fidelity verdict 'match' against an "
            f"envelope whose machine declaration anchor records {declared} declared "
            "deviation/source-unspecified-choice row(s); the match/minor-drift anchor "
            "is machine-owned and a verdict disputing it is a typed checker defect, "
            "never a member verdict",
            stable_ids=[stable_id],
        )


#: Closed structured shapes for LIVE integrity-array claims (W-8, P5-SOL-06).
#: gate.v3 types the integrity arrays as string arrays, so each structured row
#: travels as the JSON serialization of its closed object; free prose cannot
#: bind a claim to its mechanical authority and is refused typed.
_HASH_MISMATCH_CLAIM_KEYS = frozenset({"digest_key", "detail"})
_LOCATOR_CLAIM_KEYS = frozenset({"evidence_id", "status", "claim_kind", "detail"})
#: Closed claim_kind vocabulary, keyed by the exact array each kind may sit in.
_LOCATOR_CLAIM_KIND_BY_ARRAY = MappingProxyType(
    {
        "locator_failures": "locator-failure",
        "excerpt_discrepancies": "excerpt-discrepancy",
    }
)

#: The two R4 search-attestation routes (INTEGRITY/RUNG clause 2): a plain URL
#: entry claims re-fetched bytes; a typed ``{url, disposition}`` entry claims a
#: broker-proven non-byte outcome. The marker names its own route.
_SEARCH_MARKER_PATTERN = re.compile(r"(search-receipt-attested|search-attested):(\S+)")
_SEARCH_MARKER_ROUTE_PLAIN = "search-attested"
_SEARCH_MARKER_ROUTE_RECEIPT = "search-receipt-attested"


def _structured_integrity_claim(
    entry: Any,
    *,
    stable_id: str,
    array_name: str,
    allowed_keys: Collection[str],
    required_keys: Sequence[str],
) -> JsonObject:
    """Decode one LIVE integrity-array entry to its closed structured claim.

    gate.v3 carries the integrity arrays as strings, so a structured row is
    the JSON serialization of its closed object; an already-decoded mapping is
    adjudicated identically. Free prose, JSON non-objects, unknown keys,
    missing required keys, and non-string values are each a typed refusal
    naming the defect -- W-8: free text does not bind a claim to its
    mechanical authority.
    """

    claim: Any = entry
    if isinstance(entry, str):
        try:
            claim = json.loads(entry)
        except ValueError:
            claim = None
    if not isinstance(claim, Mapping):
        raise CheckerMechanicalClaimError(
            f"checker item {stable_id} integrity.{array_name} entry is not the "
            "closed structured claim object; free prose cannot bind a claim to "
            f"its mechanical authority (entry: {str(entry)[:200]!r})",
            stable_ids=[stable_id],
        )
    unknown = sorted(set(map(str, claim)) - set(allowed_keys))
    if unknown:
        raise CheckerMechanicalClaimError(
            f"checker item {stable_id} integrity.{array_name} entry carries "
            f"unknown key(s) {', '.join(unknown)}; the claim object is closed to "
            f"exactly {{{', '.join(sorted(allowed_keys))}}}",
            stable_ids=[stable_id],
        )
    for key in required_keys:
        if not isinstance(claim.get(key), str):
            raise CheckerMechanicalClaimError(
                f"checker item {stable_id} integrity.{array_name} entry is missing "
                f"the required string field {key!r}",
                stable_ids=[stable_id],
            )
    if "detail" in claim and not isinstance(claim["detail"], str):
        raise CheckerMechanicalClaimError(
            f"checker item {stable_id} integrity.{array_name} entry field 'detail' "
            "must be a string when present",
            stable_ids=[stable_id],
        )
    return dict(claim)


def _validate_hash_mismatch_claims(
    integrity: Mapping[str, Any],
    envelope_item: Mapping[str, Any],
    *,
    stable_id: str,
) -> None:
    """hash-claim-binding: each mismatch claim names an exactly bound digest key.

    The prose-era substring test admitted an unbound-digest allegation whose
    text merely CONTAINED a bound key ("evidence" as an incidental word) and a
    superstring paraphrase ("not_source_manifest_weights"); exact membership
    of the structured ``digest_key`` kills both shapes.
    """

    verified_hashes = envelope_item.get("verified_hashes")
    if not isinstance(verified_hashes, Mapping):
        return
    bound_keys = {str(key) for key, digest in verified_hashes.items() if isinstance(digest, str)}
    for entry in integrity.get("hash_mismatches") or []:
        claim = _structured_integrity_claim(
            entry,
            stable_id=stable_id,
            array_name="hash_mismatches",
            allowed_keys=_HASH_MISMATCH_CLAIM_KEYS,
            required_keys=("digest_key",),
        )
        if claim["digest_key"] not in bound_keys:
            raise CheckerMechanicalClaimError(
                f"checker item {stable_id} claims a hash mismatch for digest_key "
                f"{str(claim['digest_key'])!r}, which is not exactly a digest the "
                f"envelope binds (bound keys: {', '.join(sorted(bound_keys))}); a "
                "mismatch claim about an unbound digest is machine-refuted",
                stable_ids=[stable_id],
            )


def _validate_locator_claim_bindings(
    integrity: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    stable_id: str,
) -> None:
    """locator-citation-license: each entry alleges its OWN non-valid row.

    The prose-era rule validated only the cited license, so a mechanically
    false allegation about a different, VALID row passed as long as its text
    cited some unrelated non-valid row. The structured ``evidence_id`` IS the
    alleged row: it must be present and non-valid, ``status`` must equal the
    row's attested status exactly, and ``claim_kind`` must match the array the
    entry sits in. The all-valid arm stays
    :func:`_validate_item_locator_authority`'s.
    """

    rows = attestation.get("rows")
    rows = rows if isinstance(rows, list) else []
    non_valid_rows = {
        str(row.get("evidence_id")): str(row.get("status"))
        for row in rows
        if isinstance(row, Mapping) and row.get("status") != LOCATOR_ATTESTATION_VALID
    }
    if not non_valid_rows:
        return
    for array_name, claim_kind in _LOCATOR_CLAIM_KIND_BY_ARRAY.items():
        for entry in integrity.get(array_name) or []:
            _validate_one_locator_claim(
                entry,
                non_valid_rows,
                stable_id=stable_id,
                array_name=array_name,
                claim_kind=claim_kind,
            )


def _validate_one_locator_claim(
    entry: Any,
    non_valid_rows: Mapping[str, str],
    *,
    stable_id: str,
    array_name: str,
    claim_kind: str,
) -> None:
    """Adjudicate one structured locator/excerpt claim against its own row."""

    claim = _structured_integrity_claim(
        entry,
        stable_id=stable_id,
        array_name=array_name,
        allowed_keys=_LOCATOR_CLAIM_KEYS,
        required_keys=("evidence_id", "status", "claim_kind"),
    )
    if claim["claim_kind"] != claim_kind:
        raise CheckerMechanicalClaimError(
            f"checker item {stable_id} integrity.{array_name} entry carries "
            f"claim_kind {str(claim['claim_kind'])!r}, but entries in this array "
            f"must carry exactly {claim_kind!r}",
            stable_ids=[stable_id],
        )
    attested_status = non_valid_rows.get(str(claim["evidence_id"]))
    if attested_status is None:
        raise CheckerMechanicalClaimError(
            f"checker item {stable_id} alleges a {claim_kind} against attestation "
            f"row {str(claim['evidence_id'])!r}, which the machine attestation "
            "records as valid or does not record at all; an allegation "
            "contradicting machine arithmetic is refuted and is never licensed by "
            "a different row's non-valid status",
            stable_ids=[stable_id],
        )
    if claim["status"] != attested_status:
        raise CheckerMechanicalClaimError(
            f"checker item {stable_id} alleges status {str(claim['status'])!r} for "
            f"attestation row {str(claim['evidence_id'])!r}, but the machine "
            f"attestation records {attested_status!r}; a claim contradicting "
            "machine arithmetic is refuted",
            stable_ids=[stable_id],
        )


def _validate_item_mechanical_claims(
    result_item: Mapping[str, Any],
    envelope_item: Mapping[str, Any],
    *,
    stable_id: str,
) -> None:
    """W-8: machine-check every machine-decidable checker claim at the write door.

    The closed claim-class registry is documented on
    :class:`CheckerMechanicalClaimError`. Every class binds through the LIVE
    envelope's machine attestation stamp; a recorded (legacy) envelope keeps
    its historical semantics untouched. A refuted claim discards the whole
    result typed -- never a member verdict in either direction.
    """

    attestation = envelope_item.get("locator_attestation")
    if not isinstance(attestation, Mapping):
        return
    integrity = result_item.get("integrity")
    integrity = integrity if isinstance(integrity, Mapping) else {}
    _validate_hash_mismatch_claims(integrity, envelope_item, stable_id=stable_id)
    _validate_locator_claim_bindings(integrity, attestation, stable_id=stable_id)

    # -- r4-search-marker-correlation ----------------------------------------
    rung_check = result_item.get("rung_check")
    rung_check = rung_check if isinstance(rung_check, Mapping) else {}
    findings = [str(entry) for entry in (rung_check.get("findings") or [])]
    markers = [
        (route, url) for text in findings for route, url in _SEARCH_MARKER_PATTERN.findall(text)
    ]
    proposal = envelope_item.get("proposal")
    facts = proposal.get("proposed_facts") if isinstance(proposal, Mapping) else None
    resolution = facts.get("source_resolution") if isinstance(facts, Mapping) else None
    resolution = resolution if isinstance(resolution, Mapping) else {}
    search_report = resolution.get("search_report")
    links = search_report.get("links_checked") if isinstance(search_report, Mapping) else None
    routes_by_url: dict[str, str] = {}
    for link in links if isinstance(links, list) else []:
        if isinstance(link, str) and link.strip():
            routes_by_url[link] = _SEARCH_MARKER_ROUTE_PLAIN
        elif isinstance(link, Mapping) and isinstance(link.get("url"), str):
            routes_by_url[str(link["url"])] = _SEARCH_MARKER_ROUTE_RECEIPT
    for route, url in markers:
        expected_route = routes_by_url.get(url)
        if expected_route is None:
            raise CheckerMechanicalClaimError(
                f"checker item {stable_id} attests search link {url!r} which is not in "
                "the frozen search_report.links_checked closed set; an attestation "
                "outside the closed set is machine-refuted",
                stable_ids=[stable_id],
            )
        if route != expected_route:
            raise CheckerMechanicalClaimError(
                f"checker item {stable_id} attests {url!r} via {route!r} but the frozen "
                f"links_checked entry form requires {expected_route!r}; a swapped "
                "attestation route is mechanically refused",
                stable_ids=[stable_id],
            )
    if (
        resolution.get("rung") == SourceRung.REIMPLEMENT.value
        and rung_check.get("verdict") == AccuracyVerdict.ACCURATE.value
    ):
        attested_urls = {url for _route, url in markers}
        missing = sorted(set(routes_by_url) - attested_urls)
        if missing:
            raise CheckerMechanicalClaimError(
                f"checker item {stable_id} renders an accurate R4 rung check attesting "
                f"{len(attested_urls)} of {len(routes_by_url)} frozen checked links "
                f"(unattested: {', '.join(missing[:5])}); the R4 claim is a negative "
                "proof over exactly that closed set and a spot-checked accurate "
                "verdict blesses the members never examined",
                stable_ids=[stable_id],
            )


def classify_checker_response(
    status_code: int,
    response_body: str,
    *,
    retry_after_seconds: Optional[int] = None,
    reset_at: Optional[str] = None,
    provider: str = "openai",
) -> Optional[CheckerBackoffSignal]:
    """Classify rate/quota responses without treating them as gate verdicts.

    Parameters
    ----------
    status_code:
        Provider HTTP or process response status.
    response_body:
        Provider response text.
    retry_after_seconds:
        Parsed Retry-After delay, if supplied.
    reset_at:
        Provider reset timestamp, if supplied.
    provider:
        Closed usage-limit provider the session ran under (the caller derives
        it from the envelope's seated model via ``checker_provider_for_model``);
        the pause episode is scoped to it. Refused outside the closed vocabulary.

    Returns
    -------
    CheckerBackoffSignal | None
        Typed pause signal, or None for non-rate/quota responses.
    """

    if provider not in USAGE_LIMIT_PROVIDERS:
        raise CheckerDispatchError(f"unsupported usage-limit provider {provider!r}")
    lowered = response_body.lower()
    quota_markers = ("quota", "usage limit")
    rate_markers = ("rate limit", "too many requests", "retry after", "tokens per minute")
    auth_markers = (
        "not logged in",
        "codex login",
        "401 unauthorized",
        "unauthorized",
        "authentication failed",
        "token expired",
        "invalid api key",
    )
    reason: Optional[CheckerPauseReason] = None
    if any(marker in lowered for marker in quota_markers):
        reason = CheckerPauseReason.QUOTA_EXHAUSTED
    elif status_code == 429 or any(marker in lowered for marker in rate_markers):
        reason = CheckerPauseReason.RATE_LIMIT
    elif status_code != 0 and any(marker in lowered for marker in auth_markers):
        # T4-T3: an expired login is provider weather that pauses the LANE like
        # quota -- never a per-model park. The arm is gated on a failed exit so a
        # successful gate whose bounded diagnostics merely mention auth text can
        # never be demoted into a pause.
        reason = CheckerPauseReason.AUTH_FAILURE
    elif status_code == int(OperatorExitCode.RATE_OR_QUOTA_PAUSE):
        # The wrapper's typed pause exit is authoritative even when its bounded
        # text tail matches no marker: falling through would hand a declared
        # pause to `_raise_for_checker_exit`'s generic arm, which turns lane
        # weather into a batch failure and, at the ladder floor, a wrong-blame
        # park.
        reason = CheckerPauseReason.QUOTA_EXHAUSTED
    if reason is None:
        return None
    if retry_after_seconds is None:
        try:
            response_payload = json.loads(response_body)
        except (json.JSONDecodeError, TypeError):
            response_payload = None
        raw_retry_after = (
            response_payload.get("retry_after_seconds")
            if isinstance(response_payload, Mapping)
            else None
        )
        if (
            isinstance(raw_retry_after, (int, float))
            and not isinstance(raw_retry_after, bool)
            and 0 <= raw_retry_after <= 8 * 24 * 60 * 60
        ):
            retry_after_seconds = int(raw_retry_after)
    return CheckerBackoffSignal(
        reason=reason,
        retry_after_seconds=retry_after_seconds,
        reset_at=reset_at,
        response_excerpt=response_body[:1_500],
        provider=provider,
    )


def compute_result_envelope_sha256(result: Mapping[str, Any]) -> str:
    """Compute the non-self-referential checker result-envelope digest.

    Parameters
    ----------
    result:
        Gate result with absent result hash and absent ledger-assigned fields.

    Returns
    -------
    str
        Canonical result digest.

    Notes
    -----
    The excluded set is exactly the one ``authority.load_current_gate_proof`` and
    ``driver_models._normalize_gate_generation`` recompute over. It MUST stay
    aligned with them: this digest is written once, before the append, and
    re-derived afterwards from the persisted gate, so any key the ledger touches
    between those two moments has to be outside the digest or no gate can ever
    replay its own self-hash. ``ledger_seq`` was inside here and outside there,
    which made every recorded gate fail with "current gate result-envelope
    self-hash is invalid" the moment anything tried to read one back.
    """

    excluded = {"result_envelope_sha256", *LEDGER_ASSIGNED_GATE_FIELDS}
    return stable_hash({key: value for key, value in result.items() if key not in excluded})


#: Closed per-item inventory-mode vocabulary (Sol H-2.2): the coordinator
#: stamps whether the frozen ``env-routed-inventory`` machine row exists in
#: this item's manifest, so the checker never re-derives that scope fact from
#: author prose (the m10759 fidelity flip: a negative/meta inventory statement
#: read as an unverifiable inventory assertion by some draws and as a machine
#: scope fact by others).
INVENTORY_MODE_DISCLOSED = "disclosed"
INVENTORY_MODE_NOT_DISCLOSED = "not-disclosed"


def _derive_inventory_mode(source_manifest: Any) -> str:
    """Return the machine-derived inventory disclosure mode for one item."""

    sources = source_manifest.get("sources") if isinstance(source_manifest, Mapping) else None
    for row in sources if isinstance(sources, list) else ():
        if isinstance(row, Mapping) and row.get("source_kind") == ROUTED_INVENTORY_SOURCE_KIND:
            return INVENTORY_MODE_DISCLOSED
    return INVENTORY_MODE_NOT_DISCLOSED


#: The coordinator-frozen match/minor-drift declaration anchor (rule-2 round
#: 3, cell 1): WHICH pack vocabulary owns "the proposal declares a deviation"
#: was split across ``fidelity.deviations`` and
#: ``implementation.declared_choices`` and measured flipping on m10759
#: (match <-> minor-drift, 2-1). The anchor fact is machine-derivable from the
#: same frozen proposal bytes ``verified_hashes`` binds, so it is stamped like
#: ``inventory_mode`` and withdrawn from per-draw reading choices.
DEVIATION_BASIS_AUTHORITY = "declared-deviation-basis-v1"

#: ``declared_choices.source_status`` value that makes a row a
#: source-unspecified choice by definition (a ``source-specified`` row
#: restates the source and is never a deviation from it).
DECLARED_CHOICE_AUTHOR_CHOSEN = "author-chosen"


def _derive_declared_deviation_basis(proposed_facts: Any) -> JsonObject:
    """Freeze one fidelity item's machine declaration anchor.

    The closed derivation: every ``fidelity.deviations`` row plus every
    ``implementation.declared_choices`` row whose ``source_status`` is
    ``author-chosen``. ``any_declared`` is the anchor fact the match verdict
    keys on; the listed choice rows carry the author's own ``material`` flag
    verbatim (a build-disclosure flag -- the checker still judges fidelity
    materiality, the stamp only settles WHAT was declared).
    """

    facts = proposed_facts if isinstance(proposed_facts, Mapping) else {}
    fidelity = facts.get("fidelity")
    deviations = fidelity.get("deviations") if isinstance(fidelity, Mapping) else None
    deviations = deviations if isinstance(deviations, list) else []
    implementation = facts.get("implementation")
    choices = (
        implementation.get("declared_choices") if isinstance(implementation, Mapping) else None
    )
    unspecified = [
        {
            "field": str(row.get("field")),
            "source_status": DECLARED_CHOICE_AUTHOR_CHOSEN,
            "author_material_flag": bool(row.get("material")),
        }
        for row in (choices if isinstance(choices, list) else [])
        if isinstance(row, Mapping) and row.get("source_status") == DECLARED_CHOICE_AUTHOR_CHOSEN
    ]
    return {
        "authority": DEVIATION_BASIS_AUTHORITY,
        "fidelity_deviation_entries": len(deviations),
        "source_unspecified_choices": unspecified,
        "any_declared": bool(deviations or unspecified),
    }


#: The coordinator-frozen weights-convention anchor (rule-2 round 4, the last
#: cell): whether the catalog's UNIVERSAL weights-not-used convention
#: (pretrained loading disabled, random construction, weights license
#: ``not-used``) applies to a fidelity item was split across per-draw readings
#: of the pack's const fields and measured flipping on m3353
#: (cannot-verify <-> major-drift, 2-1: one draw charged the bundle's defining
#: checkpoint as a sourced material item the random-init recipe contradicts).
#: The anchor fact is machine-derivable from the same frozen proposal bytes
#: ``verified_hashes`` binds, so it is stamped like ``declared_deviation_basis``
#: and withdrawn from per-draw reading choices.
WEIGHTS_CONVENTION_AUTHORITY = "weights-convention-basis-v1"

#: ``implementation.initialization.policy`` const value under the catalog
#: convention: every recipe constructs at random initialization.
INITIALIZATION_POLICY_RANDOM = "random"

#: ``licenses.weights.status`` const value under the catalog convention: no
#: pretrained weights are used, so no weights license applies.
WEIGHTS_LICENSE_NOT_USED = "not-used"


def _derive_weights_convention_basis(proposed_facts: Any) -> JsonObject:
    """Freeze one fidelity item's machine weights-convention anchor.

    The closed derivation: ``convention_declared`` is true exactly when the
    pack's three const declarations all hold --
    ``implementation.initialization.policy`` is ``random``,
    ``implementation.initialization.pretrained_disabled`` is true, and
    ``licenses.weights.status`` is ``not-used``. Anything missing, malformed,
    or different fails closed to false (historical semantics untouched). The
    stamp settles WHAT the pack declares about weights; the checker still
    judges the constructed architecture on the merits.
    """

    facts = proposed_facts if isinstance(proposed_facts, Mapping) else {}
    implementation = facts.get("implementation")
    initialization = (
        implementation.get("initialization") if isinstance(implementation, Mapping) else None
    )
    initialization = initialization if isinstance(initialization, Mapping) else {}
    licenses = facts.get("licenses")
    weights = licenses.get("weights") if isinstance(licenses, Mapping) else None
    weights = weights if isinstance(weights, Mapping) else {}
    policy = initialization.get("policy")
    pretrained_disabled = initialization.get("pretrained_disabled")
    status = weights.get("status")
    return {
        "authority": WEIGHTS_CONVENTION_AUTHORITY,
        "initialization_policy": policy if isinstance(policy, str) else None,
        "pretrained_disabled": (
            pretrained_disabled if isinstance(pretrained_disabled, bool) else None
        ),
        "weights_license_status": status if isinstance(status, str) else None,
        "convention_declared": (
            policy == INITIALIZATION_POLICY_RANDOM
            and pretrained_disabled is True
            and status == WEIGHTS_LICENSE_NOT_USED
        ),
    }


def _derive_locator_attestation(
    excerpts: Sequence[Mapping[str, Any]],
    source_manifest: Any,
    *,
    model_dir: Any,
) -> JsonObject:
    """Freeze the machine locator attestation for one envelope item.

    The coordinator-owned ``env-routed-inventory`` idiom applied to locators
    (Opus W-1): derived here at the one boundary every checker request passes
    through, from the same frozen bytes ``verified_hashes`` binds, never
    authored and never re-derived by the checker. ``all_valid`` is a
    convenience disclosure; every consumer recomputes it from the rows.
    """

    manifest = source_manifest if isinstance(source_manifest, Mapping) else {"sources": []}
    cas_root = (
        Path(str(model_dir)).parent / SOURCE_CAS_DIRNAME
        if isinstance(model_dir, str) and model_dir.strip()
        else None
    )
    rows = attest_excerpt_locators(list(excerpts), manifest, cas_root=cas_root)
    return {
        "authority": LOCATOR_AUTHORITY_TEXT_LINE,
        "rows": rows,
        "all_valid": all(row.get("status") == LOCATOR_ATTESTATION_VALID for row in rows),
    }


#: The coordinator-frozen digest attestation (F51 checker-lifecycle D8 /
#: DECIDED_PLAN item 20, 2026-09-01): the receipt2 live-epoch probe measured
#: the rung-adverse golden's bound ``verified_hashes.code`` disagreeing with the
#: staged ``model_dir`` bytes -- sol caught it 3/3, Fable 1/3 -- and the
#: acceptrate replay found the same class on 9 of 81 packs. Digest verification
#: is machine-decidable and was delegated to model shell arithmetic in both
#: lanes; it is stamped here like ``locator_attestation`` and withdrawn from
#: per-draw model work. Statuses are a closed three-value vocabulary.
DIGEST_ATTESTATION_AUTHORITY = "verified-hashes-digest-v1"
DIGEST_ATTESTATION_VALID = "valid"
DIGEST_ATTESTATION_MISMATCH = "mismatch"
DIGEST_ATTESTATION_UNRESOLVABLE = "unresolvable"
DIGEST_ATTESTATION_STATUSES = frozenset(
    {DIGEST_ATTESTATION_VALID, DIGEST_ATTESTATION_MISMATCH, DIGEST_ATTESTATION_UNRESOLVABLE}
)

#: Closed per-key recompute basis, per gate kind: WHICH frozen bytes each
#: ``verified_hashes`` key binds and how the coordinator recomputes it. These
#: are the producers' own recipes (``author_executor`` stamps
#: ``proposal_sha256`` over the proposal minus itself; ``driver_admission``
#: binds ``code``/``code_manifest`` from the staged closure;
#: ``driver_models._terminal_checker_item`` binds the terminal keys), restated
#: at the one boundary every checker request passes through.
_DIGEST_BASIS_PROPOSAL = "canonical-json-object-minus-self-digest-v1"
_DIGEST_BASIS_SOURCES = "canonical-json-source-manifest-sources-v1"
_DIGEST_BASIS_EVIDENCE_EXCERPTS = "evidence-identity-over-proposal-excerpts-v1"
_DIGEST_BASIS_EVIDENCE_PREIMAGE = "canonical-json-evidence-pack-identity-preimage-v1"
_DIGEST_BASIS_SOURCE_MAP = "canonical-json-implementation-source-to-code-map-v1"
_DIGEST_BASIS_CHECKED_SOURCES = "canonical-json-evidence-pack-checked-source-ids-v1"
_DIGEST_BASIS_CODE = "staged-entry-point-bytes-v1"
_DIGEST_BASIS_CODE_MANIFEST = "canonical-json-code-manifest-and-staged-row-bytes-v1"
_DIGEST_BASIS_TEMPLATE = "proposal-website-template-hash-v1"
_METADATA_DIGEST_BASES: Mapping[str, str] = MappingProxyType(
    {
        "proposal": _DIGEST_BASIS_PROPOSAL,
        "source_manifest": _DIGEST_BASIS_SOURCES,
        "evidence": _DIGEST_BASIS_EVIDENCE_EXCERPTS,
        "source_to_code_map": _DIGEST_BASIS_SOURCE_MAP,
        "code": _DIGEST_BASIS_CODE,
        "code_manifest": _DIGEST_BASIS_CODE_MANIFEST,
        "family_template": _DIGEST_BASIS_TEMPLATE,
    }
)
DIGEST_BASIS_BY_KEY: Mapping[GateKind, Mapping[str, str]] = MappingProxyType(
    {
        GateKind.METADATA_BATCH: _METADATA_DIGEST_BASES,
        GateKind.FIDELITY: _METADATA_DIGEST_BASES,
        GateKind.TERMINAL_DISPOSITION: MappingProxyType(
            {
                "proposal": _DIGEST_BASIS_PROPOSAL,
                "source_manifest": _DIGEST_BASIS_SOURCES,
                "evidence": _DIGEST_BASIS_EVIDENCE_PREIMAGE,
                "source_to_code_map": _DIGEST_BASIS_CHECKED_SOURCES,
                "code": _DIGEST_BASIS_CODE,
                "family_template": _DIGEST_BASIS_TEMPLATE,
            }
        ),
    }
)


def _digest_row(
    key: str, basis: str, bound: str, recomputed: Optional[str], detail: Optional[str] = None
) -> JsonObject:
    """Freeze one attestation row; status is DERIVED from bound vs recomputed."""

    if recomputed is None:
        status = DIGEST_ATTESTATION_UNRESOLVABLE
    elif recomputed == bound:
        status = DIGEST_ATTESTATION_VALID
    else:
        status = DIGEST_ATTESTATION_MISMATCH
    return {
        "digest_key": key,
        "basis": basis,
        "bound": bound,
        "recomputed": recomputed,
        "status": status,
        "detail": detail,
    }


def _recompute_object_digest(container: Any, self_key: str) -> Optional[str]:
    """Canonical digest of a mapping minus its own self-digest field."""

    if not isinstance(container, Mapping):
        return None
    return stable_hash({key: value for key, value in container.items() if key != self_key})


def _recompute_list_digest(value: Any) -> Optional[str]:
    """Canonical digest of a list-valued binding; a non-list is unresolvable."""

    return stable_hash(list(value)) if isinstance(value, list) else None


def _recompute_staged_code(model_dir: Any, code_path: Any) -> tuple[Optional[str], Optional[str]]:
    """Digest the staged entry point ``model_dir/code_path``; name why when absent."""

    if not isinstance(model_dir, str) or not model_dir.strip():
        return None, "item declares no model_dir"
    if not isinstance(code_path, str) or not code_path.strip() or Path(code_path).is_absolute():
        return None, f"implementation.code_path {code_path!r} is not a relative staged path"
    staged = Path(model_dir) / code_path
    if not staged.is_file():
        return None, f"staged bytes absent: {code_path}"
    return hash_bytes(staged.read_bytes()), None


def _recompute_code_manifest(model_dir: Any, manifest: Any) -> tuple[Optional[str], Optional[str]]:
    """Digest the manifest list and re-hash every row's staged bytes.

    A row whose staged bytes hash differently is a positive contradiction and
    wins (the recomputed value is forced to differ from any bound digest by
    reporting the row's live digest); a row whose bytes are absent makes the
    manifest unresolvable when nothing contradicts it.
    """

    if not isinstance(manifest, list) or not isinstance(model_dir, str) or not model_dir.strip():
        return None, "code_manifest or model_dir is not a staged closure"
    drifted: list[str] = []
    absent: list[str] = []
    for row in manifest:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            return None, "code_manifest row is malformed"
        staged = Path(model_dir) / str(row["path"])
        if not staged.is_file():
            absent.append(str(row["path"]))
            continue
        live = hash_bytes(staged.read_bytes())
        if live != row.get("sha256"):
            drifted.append(f"{row['path']} staged bytes hash to {live}")
    if drifted:
        return stable_hash({"drifted_rows": drifted}), "; ".join(drifted)
    if absent:
        return None, "staged bytes absent: " + ", ".join(absent)
    return stable_hash(manifest), None


_Recompute = Callable[[Mapping[str, Any]], tuple[Optional[str], Optional[str]]]


def _mapping_at(container: Any, *path: str) -> Mapping[str, Any]:
    """Walk nested mappings; any non-mapping hop yields an empty mapping."""

    current: Any = container
    for hop in path:
        current = current.get(hop) if isinstance(current, Mapping) else None
    return current if isinstance(current, Mapping) else {}


def _recompute_proposal(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    return _recompute_object_digest(item.get("proposal"), "proposal_sha256"), None


def _recompute_author_result(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    return _recompute_object_digest(item.get("author_result"), "result_sha256"), None


def _recompute_sources(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    return _recompute_list_digest(_mapping_at(item, "source_manifest").get("sources")), None


def _recompute_excerpt_evidence(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    excerpts = _mapping_at(item, "proposal", "proposed_facts", "evidence").get("excerpts")
    if not isinstance(excerpts, list):
        return None, "proposal evidence.excerpts is not a list"
    return compute_evidence_identity(excerpts), None


def _recompute_preimage_evidence(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    preimage = _mapping_at(item, "evidence_pack").get("identity_preimage")
    return _recompute_list_digest(preimage), None


def _recompute_source_map(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    implementation = _mapping_at(item, "proposal", "proposed_facts", "implementation")
    return _recompute_list_digest(implementation.get("source_to_code_map")), None


def _recompute_checked_sources(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    checked = _mapping_at(item, "evidence_pack").get("checked_source_ids")
    return _recompute_list_digest(checked), None


def _recompute_code(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    implementation = _mapping_at(item, "proposal", "proposed_facts", "implementation")
    return _recompute_staged_code(item.get("model_dir"), implementation.get("code_path"))


def _recompute_manifest(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    implementation = _mapping_at(item, "proposal", "proposed_facts", "implementation")
    return _recompute_code_manifest(item.get("model_dir"), implementation.get("code_manifest"))


def _recompute_template(item: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    template_hash = _mapping_at(item, "proposal", "proposed_facts", "website").get("template_hash")
    return (template_hash if isinstance(template_hash, str) else None), None


#: Closed recompute dispatch, one callable per (gate kind, bound key); the
#: keyset is exactly :data:`DIGEST_BASIS_BY_KEY`'s so every disclosed basis has
#: exactly one recompute and vice versa.
_METADATA_RECOMPUTERS: Mapping[str, _Recompute] = MappingProxyType(
    {
        "proposal": _recompute_proposal,
        "source_manifest": _recompute_sources,
        "evidence": _recompute_excerpt_evidence,
        "source_to_code_map": _recompute_source_map,
        "code": _recompute_code,
        "code_manifest": _recompute_manifest,
        "family_template": _recompute_template,
    }
)
_DIGEST_RECOMPUTERS: Mapping[GateKind, Mapping[str, _Recompute]] = MappingProxyType(
    {
        GateKind.METADATA_BATCH: _METADATA_RECOMPUTERS,
        GateKind.FIDELITY: _METADATA_RECOMPUTERS,
        GateKind.TERMINAL_DISPOSITION: MappingProxyType(
            {
                "proposal": _recompute_author_result,
                "source_manifest": _recompute_sources,
                "evidence": _recompute_preimage_evidence,
                "source_to_code_map": _recompute_checked_sources,
                "code": _recompute_code,
                "family_template": _recompute_template,
            }
        ),
    }
)
assert all(
    set(_DIGEST_RECOMPUTERS[kind]) == set(DIGEST_BASIS_BY_KEY[kind]) for kind in _DIGEST_RECOMPUTERS
), "digest recompute dispatch and basis table must name the same keys"


def _recompute_digest(
    key: str, item: Mapping[str, Any], gate_kind: GateKind
) -> tuple[Optional[str], Optional[str]]:
    """Recompute one bound key from the frozen bytes it names (closed dispatch)."""

    recompute = _DIGEST_RECOMPUTERS[gate_kind].get(key)
    if recompute is None:
        return None, f"no closed recompute basis for digest key {key!r}"
    return recompute(item)


def _derive_digest_attestation(item: Mapping[str, Any], gate_kind: GateKind) -> JsonObject:
    """Freeze the machine digest attestation for one envelope item.

    One row per NON-null ``verified_hashes`` key, recomputed from the frozen
    bytes that key binds under the closed basis table; null keys are disclosed
    in ``unbound_keys``. ``all_valid`` is a convenience disclosure; every
    consumer recomputes it from the rows.
    """

    hashes = item.get("verified_hashes")
    hashes = hashes if isinstance(hashes, Mapping) else {}
    bases = DIGEST_BASIS_BY_KEY[gate_kind]
    rows: list[JsonObject] = []
    unbound: list[str] = []
    for key in sorted(str(name) for name in hashes):
        bound = hashes[key]
        if not isinstance(bound, str):
            unbound.append(key)
            continue
        recomputed, detail = _recompute_digest(key, item, gate_kind)
        rows.append(_digest_row(key, bases.get(key, "unknown-key"), bound, recomputed, detail))
    return {
        "authority": DIGEST_ATTESTATION_AUTHORITY,
        "rows": rows,
        "unbound_keys": unbound,
        "all_valid": all(row["status"] == DIGEST_ATTESTATION_VALID for row in rows),
    }


def _stamp_digest_attestation(
    normalized: dict[str, Any], *, stable_id: str, gate_kind: GateKind
) -> None:
    """Stamp the machine digest attestation, refusing conflicts and mismatches.

    Mirrors the locator stamp: stamping is unconditional, a caller-supplied
    conflicting block is refused rather than overwritten, a ``mismatch`` row
    refuses the envelope BEFORE any session exists (a stale or drifted staged
    closure is a coordinator fact about the frozen pack, never a checker
    judgment, and buying a session to rediscover it is never useful), and
    ``unresolvable`` rows are disclosed so the checker adjudicates exactly the
    insufficiency they name.
    """

    attestation = _derive_digest_attestation(normalized, gate_kind)
    supplied = normalized.get("digest_attestation")
    if supplied is not None and supplied != attestation:
        raise CheckerDispatchError(
            f"checker item {stable_id} supplies a digest_attestation that is not the "
            "machine derivation for its own frozen bytes; the attestation is "
            "machine-owned and cannot be authored"
        )
    mismatched = [
        row for row in attestation["rows"] if row["status"] == DIGEST_ATTESTATION_MISMATCH
    ]
    if mismatched:
        named = "; ".join(
            f"{row['digest_key']} bound {row['bound']} but the frozen bytes recompute to "
            f"{row['recomputed']}" + (f" ({row['detail']})" if row.get("detail") else "")
            for row in mismatched
        )
        raise CheckerDigestAttestationError(
            f"checker item {stable_id} binds digests the machine digest attestation "
            f"refutes: {named}; a drifted or stale frozen pack is adjudicated before any "
            "session is bought",
            stable_ids=[stable_id],
            mismatched_keys=[row["digest_key"] for row in mismatched],
        )
    normalized["digest_attestation"] = attestation


def _stamp_machine_item_fields(
    normalized: dict[str, Any],
    *,
    stable_id: str,
    excerpts: Sequence[Mapping[str, Any]],
    source_manifest: Any,
    inventory_mode: Optional[str],
    deviation_basis: Optional[JsonObject] = None,
    weights_convention: Optional[JsonObject] = None,
) -> None:
    """Stamp the machine-owned per-item envelope fields, refusing conflicts.

    Mirrors the ``required_field_checks`` rule: stamping is unconditional and
    a caller-supplied conflicting value is refused rather than overwritten,
    because silently rewriting it would hide the only evidence that a caller
    invented a machine fact instead of deriving one. An attestation row the
    machine proves ``invalid`` refuses the envelope BEFORE any session exists
    (Sol H-5: a mislocated excerpt is an author-side pack defect the write
    door already refuses for new packs; buying a session to re-discover it is
    never useful) -- ``unattestable`` rows are disclosed and dispatch
    proceeds, because they are exactly what the checker must adjudicate (the
    component-drift adverse golden cites sources its manifest never froze).
    """

    attestation = _derive_locator_attestation(
        excerpts, source_manifest, model_dir=normalized.get("model_dir")
    )
    supplied_attestation = normalized.get("locator_attestation")
    if supplied_attestation is not None and supplied_attestation != attestation:
        raise CheckerDispatchError(
            f"checker item {stable_id} supplies a locator_attestation that is not the "
            "machine derivation for its own frozen bytes; the attestation is "
            "machine-owned and cannot be authored"
        )
    invalid_rows = [
        str(row.get("evidence_id"))
        for row in attestation["rows"]
        if row.get("status") == LOCATOR_ATTESTATION_INVALID
    ]
    if invalid_rows:
        raise CheckerItemContractError(
            f"checker item {stable_id} carries excerpts the machine locator authority "
            f"refutes ({', '.join(invalid_rows)}); a mislocated excerpt is an author-side "
            "pack defect adjudicated before any session is bought",
            stable_ids=[stable_id],
            author_side=True,
        )
    normalized["locator_attestation"] = attestation
    if inventory_mode is not None:
        supplied_mode = normalized.get("inventory_mode")
        if supplied_mode is not None and supplied_mode != inventory_mode:
            raise CheckerDispatchError(
                f"checker item {stable_id} supplies an inventory_mode that is not the "
                "machine derivation for its own manifest; the mode is machine-owned "
                "and cannot be authored"
            )
        normalized["inventory_mode"] = inventory_mode
    if deviation_basis is not None:
        supplied_basis = normalized.get("declared_deviation_basis")
        if supplied_basis is not None and supplied_basis != deviation_basis:
            raise CheckerDispatchError(
                f"checker item {stable_id} supplies a declared_deviation_basis that is "
                "not the machine derivation for its own proposal bytes; the anchor is "
                "machine-owned and cannot be authored"
            )
        normalized["declared_deviation_basis"] = deviation_basis
    if weights_convention is not None:
        supplied_convention = normalized.get("weights_convention")
        if supplied_convention is not None and supplied_convention != weights_convention:
            raise CheckerDispatchError(
                f"checker item {stable_id} supplies a weights_convention that is not "
                "the machine derivation for its own proposal bytes; the anchor is "
                "machine-owned and cannot be authored"
            )
        normalized["weights_convention"] = weights_convention


def _build_envelope(
    gate_kind: GateKind,
    items: Sequence[Mapping[str, Any]],
    *,
    gate_round: int,
    output_path: Union[str, Path],
    checker_model: str,
    checker_version: str,
    request_nonce: str,
    final_tail: bool = False,
) -> JsonObject:
    """Build the shared metadata/fidelity envelope body.

    Parameters
    ----------
    gate_kind:
        Metadata batch or fidelity.
    items:
        Independent item artifact packs.
    gate_round:
        Review round.
    output_path:
        Exact final result path.
    checker_model, checker_version:
        Expected checker identity.
    request_nonce:
        Fresh caller-created identity.
    final_tail:
        Whether a metadata request is the explicitly authorized final short batch.

    Returns
    -------
    dict[str, Any]
        Hash-bound request envelope.

    Raises
    ------
    CheckerDispatchError
        If item identity bindings are incomplete or duplicated.
    """

    if gate_round < 1 or not request_nonce:
        raise CheckerDispatchError("gate_round and request_nonce must be positive/non-empty")
    normalized_items: list[JsonObject] = []
    seen: set[str] = set()
    for item in items:
        required = required_field_projection_spec(
            RequiredFieldProjection.GATE_ITEM_BINDING
        ).field_order
        if any(field not in item for field in required):
            raise CheckerDispatchError("checker item is missing identity/hash bindings")
        verified_hashes = item.get("verified_hashes")
        if gate_kind is GateKind.TERMINAL_DISPOSITION:
            if not isinstance(verified_hashes, Mapping):
                raise CheckerDispatchError("terminal checker item lacks verified hashes")
            if not isinstance(item.get("author_result"), Mapping):
                raise CheckerDispatchError("terminal checker item lacks its author result")
            if not isinstance(item.get("source_manifest"), Mapping):
                raise CheckerDispatchError("terminal checker item lacks its source manifest")
            evidence_pack = item.get("evidence_pack")
            if not isinstance(evidence_pack, Mapping):
                raise CheckerDispatchError("terminal checker item lacks its evidence pack")
            _validate_terminal_evidence_pack(evidence_pack)
            if not isinstance(item.get("license_disposition"), Mapping):
                raise CheckerDispatchError(
                    "terminal checker item lacks the license disposition its license_identity binds"
                )
            if not isinstance(item.get("recommendation_preimage"), Mapping):
                raise CheckerDispatchError(
                    "terminal checker item lacks the recommendation preimage its "
                    "recommendation_sha256 binds"
                )
            stable_id = str(item["stable_id"])
            if not stable_id or stable_id in seen:
                raise CheckerDispatchError(
                    "checker envelope stable IDs must be non-empty and unique"
                )
            seen.add(stable_id)
            normalized_terminal = dict(item)
            _stamp_machine_item_fields(
                normalized_terminal,
                stable_id=stable_id,
                excerpts=list(evidence_pack.get("excerpts") or []),
                source_manifest=item.get("source_manifest"),
                # Terminal advisory arms select no implementation rung, so the
                # R1 inventory mode is not a question their checker answers.
                inventory_mode=None,
            )
            _stamp_digest_attestation(normalized_terminal, stable_id=stable_id, gate_kind=gate_kind)
            normalized_items.append(normalized_terminal)
            continue
        proposal = item.get("proposal")
        if not isinstance(proposal, Mapping):
            raise CheckerDispatchError("checker item proposal must be a complete object")
        try:
            required_hash_keys = required_verified_hash_keys(proposal, include_proposal=True)
        except ProposalValidationError as exc:
            raise CheckerDispatchError(str(exc)) from exc
        if not isinstance(verified_hashes, Mapping) or set(verified_hashes) != required_hash_keys:
            raise CheckerDispatchError(
                "checker item verified_hashes must bind the exact proposal/artifact pack"
            )
        stable_id = str(item["stable_id"])
        if not stable_id or stable_id in seen:
            raise CheckerDispatchError("checker envelope stable IDs must be non-empty and unique")
        seen.add(stable_id)
        normalized = dict(item)
        # The required-check inventory is MACHINE-DERIVED here, at the one
        # boundary every checker request passes through, from the same proposal
        # bytes ``verified_hashes`` binds. The checker was previously asked to
        # reconstruct the required coverage set by inspecting the proposal JSON,
        # which no model does reliably, so complete well-reasoned verdicts died
        # at canonical write on ``ungated authored facts``. Stamping is
        # unconditional; a caller-supplied conflicting value is refused rather
        # than overwritten, mirroring the gate-scaffold rule, because silently
        # rewriting it would hide the only evidence that a caller invented an
        # inventory instead of deriving one.
        facts = proposal.get("proposed_facts")
        if not isinstance(facts, Mapping):
            raise CheckerDispatchError("checker item proposal lacks proposed_facts")
        derived_checks = list(required_metadata_field_checks(facts))
        supplied_checks = normalized.get("required_field_checks")
        if supplied_checks is not None and supplied_checks != derived_checks:
            raise CheckerDispatchError(
                f"checker item {stable_id} supplies a required_field_checks list that is "
                "not the machine derivation for its own proposal; the inventory is "
                "machine-owned and cannot be authored"
            )
        normalized["required_field_checks"] = derived_checks
        evidence_block = facts.get("evidence")
        excerpts = evidence_block.get("excerpts") if isinstance(evidence_block, Mapping) else None
        _stamp_machine_item_fields(
            normalized,
            stable_id=stable_id,
            excerpts=excerpts if isinstance(excerpts, list) else [],
            source_manifest=item.get("source_manifest"),
            inventory_mode=_derive_inventory_mode(item.get("source_manifest")),
            # The match/minor-drift declaration anchor is a FIDELITY-verdict
            # question; metadata items never author a fidelity block.
            deviation_basis=(
                _derive_declared_deviation_basis(facts) if gate_kind is GateKind.FIDELITY else None
            ),
            # The weights-convention anchor is likewise a FIDELITY-verdict
            # question; metadata items never author a fidelity block.
            weights_convention=(
                _derive_weights_convention_basis(facts) if gate_kind is GateKind.FIDELITY else None
            ),
        )
        _stamp_digest_attestation(normalized, stable_id=stable_id, gate_kind=gate_kind)
        normalized_items.append(normalized)
    prompt_sha256 = leg_checker_prompt_sha256()
    # AG-9 envelope completeness: refuse BEFORE any session exists when a
    # declared read root cannot be dereferenced (see the error's docstring).
    _require_readable_envelope_roots(normalized_items)
    # D-51 era pin: refuse BEFORE any session exists when the shipped
    # manifest's identity is not the identity the artifact binds.
    _require_citation_source_joins(normalized_items)
    resolved_output_path = Path(output_path).resolve()
    operator_fields = build_operator_fields(
        work_generation_identity=request_nonce,
        model=checker_model,
        allowed_read_roots=(resolved_output_path.parent, *_source_read_roots(normalized_items)),
        allowed_write_root=resolved_output_path.parent,
        required_output_path=resolved_output_path,
    )
    # A checker retry reuses the exact request bytes. Its ordinary operator
    # deadline therefore has to cover the bounded weather park too; otherwise a
    # 50-minute capacity storm would wake to an already-expired frozen request.
    # This extends only checker envelopes and does not enlarge any attempt grant.
    ordinary_deadline = datetime.fromisoformat(
        str(operator_fields["deadline_at"]).replace("Z", "+00:00")
    )
    weather_deadline = ordinary_deadline.astimezone(timezone.utc) + timedelta(
        seconds=CHECKER_WEATHER_PARK_BUDGET_SECONDS
    )
    operator_fields["deadline_at"] = weather_deadline.isoformat().replace("+00:00", "Z")
    body: JsonObject = {
        "envelope_version": "menagerie.crawler.checker-envelope.v3",
        **operator_fields,
        "gate_kind": gate_kind.value,
        "gate_round": gate_round,
        "request_nonce": request_nonce,
        "checker": {
            # DERIVED from the seated model (seat swap, 2026-09-01): the
            # provider is never a caller declaration. `openai` for the sol
            # confirmation seat and every pre-swap label, `anthropic` for the
            # Fable primary; the wrapper re-derives and refuses a mismatch.
            "provider": checker_provider_for_model(checker_model),
            "model": checker_model,
            "version": checker_version,
            "prompt_sha256": prompt_sha256,
        },
        "prompt": {
            "name": CHECKER_PROMPT_NAME,
            "path": str(PROMPT_PATH),
            "sha256": prompt_sha256,
        },
        "items": normalized_items,
        "required_output_path": str(resolved_output_path),
        "allowed_output_root": str(resolved_output_path.parent),
        "required_result_schema": GATE_SCHEMA_VERSION_V3,
        "final_tail": final_tail,
        # The envelope-wide locator-authority era stamp: every item above
        # carries its machine attestation, so a checker result disputing a
        # valid row is a typed authority conflict, never a member verdict.
        "locator_authority": LOCATOR_AUTHORITY_TEXT_LINE,
        # The envelope-wide digest-authority era stamp (F51 D8): every item
        # above carries its machine digest attestation, so a hash-mismatch
        # claim against a valid row is likewise a typed authority conflict.
        "digest_authority": DIGEST_ATTESTATION_AUTHORITY,
    }
    return {**body, "envelope_sha256": stable_hash(body)}


def _source_read_roots(items: Sequence[Mapping[str, Any]]) -> tuple[Path, ...]:
    """Return the content-addressed source roots the checker actually reads.

    The declaration is derived from the items themselves rather than passed in by
    a caller, because a read root that a call site can forget to declare is a
    declaration that will eventually be wrong. It is also the only honest
    declaration available: the checker's job is to re-derive every literal
    excerpt from FROZEN BYTES, and those bytes live in each item's author
    ``source-cas`` tree.

    The previous declaration named this module's own ``prompts`` directory and
    nothing else. That was wrong in both directions. The wrapper inlines the
    frozen prompt text into the argv, so the checker never opens ``prompts/``;
    and the one tree it must open to do its job -- ``source-cas`` -- was not
    declared at all. A declaration that names an unused directory while omitting
    the used one cannot be audited against behavior, which is exactly how a
    checker wandering the repository went unnoticed.

    Parameters
    ----------
    items:
        Normalized envelope items, each carrying its author ``model_dir``.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Deduplicated absolute ``source-cas`` roots in first-seen order.

    Raises
    ------
    CheckerDispatchError
        If an item does not name the author directory whose frozen bytes it
        expects the checker to dereference. Refusing is deliberate: silently
        declaring nothing would restore the untrue declaration this replaces.
    """

    roots: list[Path] = []
    for item in items:
        model_dir = item.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir.strip():
            raise CheckerDispatchError(
                f"checker item {item.get('stable_id')!r} does not name its author model_dir, so "
                "the source-cas root the checker must read cannot be declared"
            )
        cas_root = Path(model_dir).parent / SOURCE_CAS_DIRNAME
        if cas_root not in roots:
            roots.append(cas_root)
    return tuple(roots)


def _validate_terminal_evidence_pack(evidence_pack: Mapping[str, Any]) -> None:
    """Require a terminal evidence pack to be inspectable or honestly unresolved.

    A checker asked to rule on a terminal recommendation must be able to read the
    literal excerpt behind every evidence ID it is handed. The envelope carries
    two separate things and they must never be confused: ``identity_preimage``
    is the machine-derived citation table whose hash IS ``evidence_identity``,
    and ``excerpts`` holds only text the machine re-derived from frozen bytes.

    Resolution is per record, so the structural invariant this enforces is a
    PARTITION: the IDs carrying excerpts and the IDs named unresolved are
    disjoint, and together they are exactly ``declared_evidence_ids``. Every
    declared ID therefore lands in exactly one bucket, and an unverified record
    has no shape in which it can be presented as verified -- it is either an
    excerpt the machine re-derived, or a named gap.

    Parameters
    ----------
    evidence_pack:
        Terminal evidence pack supplied by the driver.

    Raises
    ------
    CheckerDispatchError
        If the pack does not partition its declared evidence IDs into
        re-derived excerpts and explicitly named gaps.
    """

    resolution = evidence_pack.get("resolution")
    declared = evidence_pack.get("declared_evidence_ids")
    excerpts = evidence_pack.get("excerpts")
    if resolution not in TERMINAL_EVIDENCE_RESOLUTIONS:
        raise CheckerDispatchError("terminal evidence pack must declare its resolution")
    if not isinstance(declared, list):
        raise CheckerDispatchError("terminal evidence pack must declare its evidence IDs")
    if not isinstance(evidence_pack.get("identity_preimage"), list):
        raise CheckerDispatchError(
            "terminal evidence pack must carry the preimage of its evidence_identity"
        )
    if not isinstance(excerpts, list):
        raise CheckerDispatchError("terminal evidence pack excerpts must be a list")
    unresolved = evidence_pack.get("unresolved_evidence_ids")
    if resolution == TERMINAL_EVIDENCE_UNRESOLVED:
        if excerpts:
            raise CheckerDispatchError(
                "unresolved terminal evidence pack cannot ship excerpts it did not verify"
            )
        if not isinstance(unresolved, list):
            raise CheckerDispatchError(
                "unresolved terminal evidence pack must name its unresolved evidence IDs"
            )
        if not isinstance(evidence_pack.get("unresolved_reason"), str):
            raise CheckerDispatchError(
                "unresolved terminal evidence pack must explain why it is unresolved"
            )
        return
    if not declared:
        raise CheckerDispatchError("a terminal evidence pack citing nothing cannot be grounded")
    grounded = {
        str(excerpt.get("evidence_id")): excerpt
        for excerpt in excerpts
        if isinstance(excerpt, Mapping)
    }
    for evidence_id, excerpt in grounded.items():
        if any(
            not isinstance(excerpt.get(field), str) or not excerpt.get(field)
            for field in ("source_id", "locator", "text")
        ):
            raise CheckerDispatchError(
                f"grounded terminal evidence pack has no literal excerpt for {evidence_id}"
            )
    if resolution == TERMINAL_EVIDENCE_GROUNDED:
        missing = [str(value) for value in declared if str(value) not in grounded]
        if missing:
            raise CheckerDispatchError(
                f"grounded terminal evidence pack has no literal excerpt for {missing[0]}"
            )
        if unresolved:
            raise CheckerDispatchError(
                "a grounded terminal evidence pack cannot also name unresolved evidence IDs"
            )
        return
    if not grounded:
        raise CheckerDispatchError(
            "a partially grounded terminal evidence pack must carry at least one excerpt"
        )
    if not isinstance(unresolved, list) or not unresolved:
        raise CheckerDispatchError(
            "a partially grounded terminal evidence pack must name its unresolved evidence IDs"
        )
    if not isinstance(evidence_pack.get("unresolved_reason"), str):
        raise CheckerDispatchError(
            "a partially grounded terminal evidence pack must explain each gap"
        )
    named = {str(value) for value in unresolved}
    if named & set(grounded):
        raise CheckerDispatchError(
            "a terminal evidence pack cannot both ground and disclaim the same evidence ID"
        )
    if named | set(grounded) != {str(value) for value in declared}:
        raise CheckerDispatchError(
            "a partially grounded terminal evidence pack must account for every declared "
            "evidence ID exactly once"
        )


def _validate_item_binding(result_item: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Validate one result item's independent request binding.

    Parameters
    ----------
    result_item:
        Checker-produced gate item.
    expected:
        Exact envelope item pack.

    Raises
    ------
    CheckerBindingMismatchError
        If identity or artifact hashes differ. The typed subclass lets the
        operator wrapper retry what is empirically a stochastic transcription
        fault without ever publishing, or comparing more leniently, a
        mismatched result.
    """

    # ``campaign_root_work_id`` is required by the gate.v3 item schema, so it is
    # always PRESENT, but its VALUE was verified only by
    # ``driver_models._require_gate_bindings`` -- which the metadata and fidelity
    # lanes reach and the terminal lane does not. On the terminal lane a checker
    # could therefore return any lineage it liked and nothing compared it to the
    # envelope. The prompt now tells the checker to copy this field verbatim from
    # its envelope item; a copy instruction is only worth giving when the copy is
    # checked, so it is checked here, for every gate kind, before publication.
    for field in (
        "work_id",
        "campaign_root_work_id",
        "stable_id",
        "family_representative_id",
        "fidelity_identity",
        "vet_identity",
        "verified_hashes",
    ):
        if result_item.get(field) != expected.get(field):
            raise CheckerBindingMismatchError(
                f"checker item {expected.get('stable_id')} mismatched binding: {field}"
            )


def _validate_item_decision(result_item: Mapping[str, Any], gate_kind: GateKind) -> None:
    """Enforce top-level verdict precedence and fidelity lane separation.

    Every lane enforces the same rule: the top-level verdict is the maximum of its
    components under :data:`VERDICT_SEVERITY`. The metadata and fidelity lanes take that
    maximum over integrity plus every scoped field check. The terminal lane's components
    are integrity and the disposition, so the same rule reads as two clauses: the verdict
    equals the disposition restated (:data:`TERMINAL_VERDICT_LOCKSTEP`), and the verdict
    is at least as severe as integrity.

    The terminal lane previously required ``integrity == verdict``. That mandated a false
    statement: an item rejected purely on the merits, with zero hash mismatches, zero
    excerpt discrepancies and zero locator failures, still had to declare
    ``integrity: "inaccurate"``. It erased the only signal separating "rejected because the
    evidence is forged" from "rejected because the argument fails", and it contradicted how
    this same function treats integrity in the metadata lane. Dropping the equality removes
    no protection: the equality could only ever force integrity *up* to match a severe
    disposition, never restrain a lenient one. What protects the gate is the monotone
    clause, which still refuses every cell where the verdict is more lenient than integrity
    -- above all an acceptance over degraded integrity, which
    :func:`menagerie.crawler.gates.validate_terminal_disposition_gate` independently
    refuses as well.

    Parameters
    ----------
    result_item:
        Schema-valid gate item.
    gate_kind:
        Metadata or fidelity lane.

    Raises
    ------
    CheckerDispatchError
        If a top-level verdict contradicts its atomic checks.
    """

    verdict = AccuracyVerdict(str(result_item.get("verdict")))
    integrity = _required_mapping(result_item.get("integrity"), "integrity")
    if gate_kind is GateKind.TERMINAL_DISPOSITION:
        terminal = _required_mapping(
            result_item.get("terminal_disposition"), "terminal_disposition"
        )
        expected_verdict = TERMINAL_VERDICT_LOCKSTEP[str(terminal.get("verdict"))]
        if verdict is not expected_verdict:
            raise CheckerDispatchError("terminal top-level verdict contradicts the disposition")
        integrity_verdict = AccuracyVerdict(str(integrity.get("verdict")))
        if VERDICT_SEVERITY[verdict] < VERDICT_SEVERITY[integrity_verdict]:
            raise CheckerDispatchError(
                "terminal verdict is less severe than its own integrity verdict"
            )
        return
    checks = result_item.get("field_checks")
    fidelity = _required_mapping(result_item.get("fidelity"), "fidelity")
    if not isinstance(checks, list) or not checks:
        raise CheckerDispatchError("every gate item requires scoped field checks")
    component_verdicts = [str(integrity.get("verdict"))]
    component_verdicts.extend(
        str(check.get("verdict")) for check in checks if isinstance(check, Mapping)
    )
    if AccuracyVerdict.INACCURATE.value in component_verdicts:
        expected = AccuracyVerdict.INACCURATE
    elif AccuracyVerdict.CANNOT_VERIFY.value in component_verdicts:
        expected = AccuracyVerdict.CANNOT_VERIFY
    else:
        expected = AccuracyVerdict.ACCURATE
    if gate_kind is GateKind.METADATA_BATCH:
        if fidelity.get("required") is not False or fidelity.get("verdict") != "not-applicable":
            raise CheckerDispatchError("metadata gates cannot issue fidelity verdicts")
    else:
        if fidelity.get("required") is not True:
            raise CheckerDispatchError("fidelity envelope requires a fidelity decision")
        fidelity_verdict = FidelityVerdict(str(fidelity.get("verdict")))
        if fidelity_verdict in {FidelityVerdict.MAJOR_DRIFT, FidelityVerdict.SLOP}:
            expected = AccuracyVerdict.INACCURATE
        elif (
            fidelity_verdict is FidelityVerdict.CANNOT_VERIFY
            and expected is AccuracyVerdict.ACCURATE
        ):
            expected = AccuracyVerdict.CANNOT_VERIFY
    if verdict is not expected:
        raise CheckerDispatchError(
            f"top-level verdict {verdict.value!r} contradicts component verdict {expected.value!r}"
        )


def _validate_item_write_contract(
    result_item: Mapping[str, Any], envelope_item: Mapping[str, Any]
) -> None:
    """Replay the canonical-write contract on a write-eligible metadata item.

    A fully accurate metadata item is a promise that canonical write will accept
    the authored facts against this exact gate item. Rung 9 broke that promise:
    one batch checker emitted every ``field_checks`` row with EMPTY
    ``checked_source_ids``, the gate schema's plain ``string_array`` let it
    publish, and the refusal surfaced only inside
    :func:`menagerie.crawler.metadata.validate_authored_facts_for_write` at
    terminal recording -- where the exact same poisoned gate also poisons the
    terminal fallback, so three consecutive models became unrecordable and the
    campaign halted with nine models stranded.

    The enforcement here IS the write-time validator, replayed against the same
    proposal bytes the envelope binds -- not a parallel re-implementation that
    could drift. A refusal is a structural contract violation, so the operator
    lane's bounded re-ask feeds the exact error back to the checker (observed
    live: the same checker emitted complete ``checked_source_ids`` on its
    single-item repair rounds), and a second refusal exits as the same permanent
    contract rejection as every other shape violation.

    Items carrying ANY adverse verdict are exempt on purpose: they are never
    write-eligible (``metadata_accepted`` requires all three verdicts accurate),
    they route to the bounded repair loop, and the write-time validator itself
    refuses them as "metadata gate is not accurate" -- replaying it here would
    turn honest adverse publications into contract rejections.

    Parameters
    ----------
    result_item:
        Schema-valid, machine-stamped metadata gate item.
    envelope_item:
        Matching envelope item carrying the bound proposal.

    Raises
    ------
    CheckerDispatchError
        If a fully accurate item would refuse at canonical write.
    """

    if not (
        result_item.get("verdict") == AccuracyVerdict.ACCURATE.value
        and result_item.get("integrity", {}).get("verdict") == AccuracyVerdict.ACCURATE.value
        and result_item.get("rung_check", {}).get("verdict") == AccuracyVerdict.ACCURATE.value
    ):
        return
    proposal = envelope_item.get("proposal")
    facts = proposal.get("proposed_facts") if isinstance(proposal, Mapping) else None
    if not isinstance(facts, Mapping):
        raise CheckerDispatchError(
            "metadata envelope item lacks the proposal facts its write contract binds"
        )
    try:
        validate_authored_facts_for_write(facts, result_item)
    except MetadataValidationError as exc:
        # BL-10 (c): the write-contract replay is per-item by construction, and
        # the typed error names the item so the driver scopes the failure
        # instead of destroying four innocent siblings' verdicts through the
        # halving ladder. WHOSE defect it is comes from the validator's own
        # owner attribution, derived per refusal from the frozen proposal --
        # blanket ``author_side=True`` here mislabeled the pilot's dominant
        # refusal families (wrong-excerpt citations and checked-source
        # bookkeeping, which the same checker repaired on its fed-back re-asks:
        # only 21 of 243 refusal events ever recurred identically) as futile
        # author defects, parking repairable batches as author-evidence-defect.
        raise CheckerItemContractError(
            f"fully accurate metadata item {result_item.get('stable_id')} would refuse "
            f"at canonical write: {exc}",
            stable_ids=(str(result_item.get("stable_id")),),
            author_side=exc.owner == "author",
        ) from exc


def _validate_item_complaint_disclosure(
    result_item: Mapping[str, Any], envelope_item: Mapping[str, Any]
) -> None:
    """Require an adverse metadata item to enumerate its COMPLETE complaint set.

    The write-contract replay above already proves exact one-to-one coverage
    for fully accurate items by replaying the write-time validator. Adverse
    items never reach that validator, so until now their coverage promise --
    stated in the frozen prompt and in ``gate.v3``'s own ``field`` description
    -- had no machine mirror: a checker could reject on two claims, stay silent
    on the other thirty-four, and publish. Silence is the channel the rung-9b
    cap churn rode in on (m10517/m7362: a new complaint surfaced per round on
    previously unexamined-looking claims until the cap burned), so this closes
    it the same way rung 9 closed the accurate side: every published round
    carries exactly one explicit verdict per machine-derived inventory entry.

    Nothing here judges merits. A refusal is a structural contract violation:
    the operator wrapper feeds the exact error back on its ONE bounded re-ask,
    the refused round is never published, and -- per the disclosure rule --
    a round that cannot enumerate completely is never chargeable, because it
    never becomes a durable gate at all.

    Parameters
    ----------
    result_item:
        Schema-valid metadata gate item carrying at least one adverse verdict.
    envelope_item:
        Matching envelope item carrying the machine-derived
        ``required_field_checks`` inventory.

    Raises
    ------
    CheckerDispatchError
        If coverage of the claim inventory is not exactly one-to-one.
    """

    if not metadata_item_rejected(result_item):
        return
    required = envelope_item.get("required_field_checks")
    if not isinstance(required, list) or not all(isinstance(value, str) for value in required):
        raise CheckerDispatchError(
            "checker envelope item lacks its machine-derived required_field_checks "
            "inventory, so complaint-set completeness cannot be validated"
        )
    stable_id = result_item.get("stable_id")
    checks = result_item.get("field_checks")
    if not isinstance(checks, list):
        raise CheckerItemContractError(
            f"adverse metadata item {stable_id} has no per-field checks to disclose",
            stable_ids=(str(stable_id),),
            author_side=False,
        )
    seen: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping) or not isinstance(check.get("field"), str):
            raise CheckerItemContractError(
                f"adverse metadata item {stable_id} carries a field check without a claim name",
                stable_ids=(str(stable_id),),
                author_side=False,
            )
        seen.append(str(check["field"]).removeprefix("proposed_facts."))
    required_set = set(required)
    duplicates = sorted({field for field in seen if seen.count(field) > 1})
    if duplicates:
        raise CheckerItemContractError(
            f"adverse metadata item {stable_id} repeats field checks {duplicates}; "
            "the claim inventory is covered exactly once per claim",
            stable_ids=(str(stable_id),),
            author_side=False,
        )
    extraneous = sorted(set(seen) - required_set)
    if extraneous:
        raise CheckerItemContractError(
            f"adverse metadata item {stable_id} names field checks outside the "
            f"machine-derived claim inventory: {extraneous}; each check must name "
            "exactly one claim from the envelope item's required_field_checks list",
            stable_ids=(str(stable_id),),
            author_side=False,
        )
    missing = sorted(required_set - set(seen))
    if missing:
        raise CheckerItemContractError(
            f"adverse metadata item {stable_id} complaint set is incomplete over the "
            f"machine-derived claim inventory: unchecked claims {missing}; every "
            "round must return one explicit verdict per required_field_checks entry "
            "-- disclose the complete complaint set, never a partial one",
            stable_ids=(str(stable_id),),
            author_side=False,
        )


def _read_prompt() -> bytes:
    """Read exact frozen checker prompt bytes.

    Returns
    -------
    bytes
        Frozen checker prompt.

    Raises
    ------
    CheckerDispatchError
        If the prompt is absent.
    """

    try:
        return PROMPT_PATH.read_bytes()
    except OSError as exc:
        raise CheckerDispatchError(f"cannot read frozen checker prompt: {exc}") from exc


def leg_checker_prompt_sha256() -> str:
    """Return this leg's PINNED checker prompt identity, pinning on first use.

    This is the ONE sanctioned prompt-identity read for every checker decision
    seam in this process (envelope build, lane adjudication identity, catch-up
    currency checks). The first call of a leg hashes the frozen prompt file and
    pins it; every later call re-reads the file and REFUSES typed when the live
    bytes disagree with the pin, so an operator editing the prompt under a
    running leg gets a loud ``CheckerPromptRotatedError`` at the next dispatch
    instead of a brain split between accepted gates and the award door
    (BL-10 e / CH-F3 -- the pilot's award-eating incident). Rotation is a
    leg-boundary act: a fresh driver process re-pins from the file.

    Returns
    -------
    str
        Pinned ``sha256:`` prompt identity for this leg.

    Raises
    ------
    CheckerPromptRotatedError
        If the live prompt file no longer matches this leg's pin.
    CheckerDispatchError
        If the prompt file cannot be read at all.
    """

    global _LEG_CHECKER_PROMPT_PIN
    live = hash_bytes(_read_prompt())
    if _LEG_CHECKER_PROMPT_PIN is None:
        _LEG_CHECKER_PROMPT_PIN = live
    elif live != _LEG_CHECKER_PROMPT_PIN:
        raise CheckerPromptRotatedError(
            "checker prompt rotated mid-leg: the live prompt file hashes to "
            f"{live} but this leg pinned {_LEG_CHECKER_PROMPT_PIN} at its first "
            "checker dispatch. Prompt rotation is a leg-boundary act: finish or "
            "stop this leg, then relaunch so the new prompt pins on a fresh leg"
        )
    return _LEG_CHECKER_PROMPT_PIN


def begin_checker_prompt_leg() -> None:
    """Open a fresh prompt-pin scope at one leg boundary.

    BL-8 (f): the pin's CONTRACT is leg-scoped ("rotation is a leg-boundary
    act ... a fresh driver process re-pins from the file"), but the state was
    process-global and never reset, so any process running more than one leg
    -- every resume test, and any future in-process relaunch -- treated a
    legitimate between-legs rotation as a mid-leg rotation and refused. The
    driver calls this at each locked leg start; the first checker dispatch of
    the NEW leg re-pins from the live file exactly as a fresh process would.
    The mid-leg tripwire is untouched: within one leg the pin still refuses
    any live-byte drift.
    """

    global _LEG_CHECKER_PROMPT_PIN
    _LEG_CHECKER_PROMPT_PIN = None


def _reset_checker_prompt_leg_pin_for_tests() -> None:
    """Clear the leg prompt pin (test isolation spelling of the leg boundary)."""

    begin_checker_prompt_leg()


def _validate_envelope_hash(envelope: Mapping[str, Any]) -> None:
    """Validate a checker envelope's self-hash.

    Parameters
    ----------
    envelope:
        Candidate envelope.

    Raises
    ------
    CheckerDispatchError
        If its exact request binding changed.
    """

    expected = stable_hash(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )
    if envelope.get("envelope_sha256") != expected:
        raise CheckerDispatchError("checker envelope hash mismatch")


def _read_json_object(path: Path) -> JsonObject:
    """Read one complete checker result object.

    Parameters
    ----------
    path:
        Final result path.

    Returns
    -------
    dict[str, Any]
        Parsed result.

    Raises
    ------
    CheckerDispatchError
        If the file is absent, partial, symlinked, or not one object.
    """

    try:
        raw = path.read_bytes()
        if not raw or path.is_symlink():
            raise CheckerDispatchError("checker result must be a non-empty regular file")
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckerDispatchError(f"partial or invalid checker result: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CheckerDispatchError("checker result must contain exactly one JSON object")
    return parsed


def _required_mapping(value: object, field: str) -> Mapping[str, Any]:
    """Return a required checker mapping.

    Parameters
    ----------
    value:
        Candidate object.
    field:
        Field name used in errors.

    Returns
    -------
    Mapping[str, Any]
        Valid mapping.

    Raises
    ------
    CheckerDispatchError
        If the value is not an object.
    """

    if not isinstance(value, Mapping):
        raise CheckerDispatchError(f"checker {field} must be an object")
    return value
