"""Derived-constants census (AB-5): derivations must stay COMPUTED.

A constant that is a function of another constant but hand-set to today's
value drifts silently the day its input changes — the fuse/floor class. This
census pins the exact set of module-level constants in ``constants.py`` whose
defining expression references other constants. Severing a derivation
(replacing the expression with a literal) fails the census; introducing a new
derived constant requires adding it here in the same reviewed change, so
every derivation is a deliberate, visible relationship.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_CONSTANTS_PATH = Path(__file__).resolve().parents[1] / "constants.py"

#: The committed derivation census: constant -> the exact upper-case constant
#: names its defining expression references. Regenerate deliberately; never
#: satisfy a failure by deleting a row (that severs the derivation guard).
_DERIVED_CONSTANTS: dict[str, frozenset[str]] = {
    # WS1 (PLAN_v6): closed per-gate-kind absolute round ceilings; derived from
    # GateKind and the metadata ceiling constant. Registered at the WS1 merge.
    "GATE_ROUND_CEILINGS": frozenset({"METADATA_GATE_ROUND_CEILING"}),
    "CURRENT_SCHEMA_VERSIONS": frozenset(
        {
            "ARTIFACT_EVENT_SCHEMA_VERSION",
            "ATTEMPT_SCHEMA_VERSION_V3",
            "AUTHOR_PROPOSAL_SCHEMA_VERSION_V3",
            "AUTHOR_RESULT_SCHEMA_VERSION",
            "GATE_SCHEMA_VERSION_V3",
            "MODEL_SCHEMA_VERSION_V3",
            "OPERATIONAL_EVENT_SCHEMA_VERSION",
        }
    ),
    "LEGACY_UNTRUSTED_SCHEMA_VERSIONS": frozenset(
        {
            "ATTEMPT_SCHEMA_VERSION",
            "AUTHOR_PROPOSAL_SCHEMA_VERSION",
            "AUTHOR_RESULT_SCHEMA_VERSION_V3",
            "GATE_SCHEMA_VERSION",
            "MODEL_SCHEMA_VERSION",
        }
    ),
    "CHECKER_LANE_PARK_STREAK_LIMIT": frozenset({"METADATA_BATCH_MAX"}),
    # Seat swap (SEAT_RULING.md 2026-09-01, item 32): the per-(gate kind, role)
    # checker seat table is derived from the two seat constants and the terminal
    # confirmation sample rate. Registered at the checker-bench merge closeout.
    "CHECKER_SEATS": frozenset(
        {"_FABLE_PRIMARY", "_SOL_CONFIRMATION", "TERMINAL_CONFIRMATION_SAMPLE_RATE"}
    ),
    "SCHEDULED_RECHECK_PROVIDERS": frozenset({"USAGE_LIMIT_PROVIDERS"}),
    "TIER_CAMPAIGN_IDS": frozenset({"TIER_CAMPAIGN_AUTHOR_MODELS"}),
    "SOURCE_RESOLUTION_RUNG_VALUES": frozenset({"NO_RUNG_SELECTED"}),
    "_MARKER_ALIASES_BY_SPECIFICITY": frozenset({"_FRAMEWORK_MARKER_ALIASES"}),
    # W13 (Wave 2): the disposition-review hold's closed reason pair, derived
    # from the two pre-existing terminal-disposition reason codes -- no new
    # reason spellings. Registered at the w2-ledger merge.
    "BLOCKED_DISPOSITION_REVIEW_REASON_CODES": frozenset(
        {
            "TERMINAL_DISPOSITION_REJECTED_REASON_CODE",
            "TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE",
        }
    ),
    "CAPABILITY_DEFERRAL_STATUS_CODES": frozenset(
        {
            "ACCESS_BLOCKED_STATUS_CODE",
            # EN-6: the CUDA-prerequisite deferral joins the accepted-BLOCKED
            # capability proof rule through this set.
            "PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE",
        }
    ),
    "HOST_CAPABILITY_DEFERRAL_STATUS_CODES": frozenset({"HOST_CAPABILITY_DEFERRAL_STATUS_CODE"}),
    "BACKEND_PARITY_DEFERRAL_STATUS_CODES": frozenset({"BACKEND_PARITY_DEFERRAL_STATUS_CODE"}),
    "TERMINAL_STATUS_CODES": frozenset(
        {
            # EN-6: the two capability deferrals stopped being restated as
            # literals; the whole closed set is the derivation input now.
            "CAPABILITY_DEFERRAL_STATUS_CODES",
            "BACKEND_PARITY_DEFERRAL_STATUS_CODES",
            # AL-6: the environment-adjacent routing terminal.
            "ENVIRONMENT_ADJACENT_STATUS_CODE",
            "HOST_CAPABILITY_DEFERRAL_STATUS_CODES",
            "SKIPPED_STATUS_CODES",
        }
    ),
    # AL-6: the stage-suffixed blocked-prerequisite reason family and the
    # environment-adjacent status spelling are derivations by design -- the
    # suffix set can never drift from the blocked-arm stage set.
    "BLOCKED_PREREQUISITE_STAGE_REASON_CODES": frozenset(
        {"BLOCKED_ARM_STAGES", "BLOCKED_PREREQUISITE_REASON_CODE"}
    ),
    "ENVIRONMENT_ADJACENT_STATUS_CODE": frozenset({"ENVIRONMENT_ADJACENT_FAILURE_STAGE"}),
    "TERMINAL_DISPOSITION_FAILURE_REASONS": frozenset(
        {
            "TERMINAL_DISPOSITION_REJECTED_REASON_CODE",
            "TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE",
        }
    ),
    "GATE_CAP_CHARGEABLE_REJECTIONS": frozenset({"GATE_MAX_REPAIRS"}),
    "AUTHOR_SESSIONS_MAX": frozenset({"GATE_MAX_REPAIRS"}),
    "GRANT_CAPABILITIES": frozenset({"HOST_CAPABILITY_ROUTING_CAPABILITY"}),
    "FAILURE_REASON_CODES": frozenset(
        {
            "BLOCKED_PREREQUISITE_REASON_CODE",
            # AL-6: the per-stage suffixed members and the environment-adjacent
            # stage key join the reason vocabulary as derivations.
            "BLOCKED_PREREQUISITE_STAGE_REASON_CODES",
            "CHECKER_SESSION_EXHAUSTION_REASON_CODE",
            "ENVIRONMENT_ADJACENT_FAILURE_STAGE",
            "TERMINAL_DISPOSITION_REJECTED_REASON_CODE",
            "TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE",
        }
    ),
    "EFFORT_EXHAUSTION_REASON_CODES": frozenset({"FAILURE_REASON_CODES"}),
    "EXHAUSTION_TERMINAL_REASON_BY_STAGE": frozenset(
        {"EFFORT_EXHAUSTION_REASON_CODES", "FAILURE_REASON_CODES"}
    ),
    "DEFAULT_REVIEW_CHECKPOINT_AT": frozenset({"DEFAULT_REVIEW_CHECKPOINT_MODELS"}),
    "DEFAULT_PROGRESS_MILESTONES": frozenset({"DEFAULT_PROGRESS_NOTIFICATION_MILESTONES"}),
    # Wave-2 sibling lanes (registered at the shared-tripwire boundary; the
    # derivations themselves are those lanes' work): the grant schema ladder and
    # the granted-attempt session multiplier.
    "REQUEUE_GRANT_SCHEMA_LADDER": frozenset(
        {"REQUEUE_GRANT_SCHEMA_VERSION", "REQUEUE_GRANT_SCHEMA_VERSION_V1"}
    ),
    "EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT": frozenset(
        {"AUTHOR_PAID_SESSIONS_PER_ATTEMPT"}
    ),
}


def _observed_derivations() -> dict[str, frozenset[str]]:
    """AST-derive the constant -> referenced-constant map from source."""

    tree = ast.parse(_CONSTANTS_PATH.read_text(encoding="utf-8"))
    observed: dict[str, frozenset[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            targets = [node.target.id]
            value = node.value
        else:
            continue
        for name in targets:
            if not name.isupper():
                continue
            references = frozenset(
                descendant.id
                for descendant in ast.walk(value)
                if isinstance(descendant, ast.Name)
                and isinstance(descendant.ctx, ast.Load)
                and descendant.id.isupper()
            )
            if references:
                observed[name] = references
    return observed


@pytest.mark.smoke
def test_derived_constants_stay_computed() -> None:
    """Every censused derivation survives in source as a computed expression."""

    observed = _observed_derivations()
    severed = {
        name: sorted(inputs)
        for name, inputs in _DERIVED_CONSTANTS.items()
        if observed.get(name) != inputs
    }
    assert not severed, (
        "derived-constants census: these derivations were severed or changed "
        "(a derived constant hand-set to a literal drifts the day its input "
        f"changes; update the census deliberately instead): {severed}"
    )


@pytest.mark.smoke
def test_new_derivations_join_the_census() -> None:
    """A new computed constant must be added to the census in the same change."""

    unlisted = sorted(set(_observed_derivations()) - set(_DERIVED_CONSTANTS))
    assert not unlisted, (
        "constants.py gained computed constants outside the derivation census; "
        f"add them to _DERIVED_CONSTANTS deliberately: {unlisted}"
    )
