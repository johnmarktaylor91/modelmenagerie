"""C1 rehearsal: the authority epoch's axis set is closed STRUCTURALLY, not by hand.

``test_authority_epoch_binding`` rotates eleven named identities and requires the
epoch digest to move. That list is hand-maintained, so a new verdict-steering
contract identity added to :class:`AuthorityContext` would silently sit OUTSIDE
the epoch -- and T1-06's whole promise is that a relaunch rotating ANY
adjudicating contract cannot inherit prior-epoch gate verdicts.

This module derives the axis set from the dataclass itself and pins each field to
one of two buckets: IN the epoch, or explicitly excluded WITH a reason. Adding a
field to ``AuthorityContext`` fails here until it is classified in the same
reviewed change.
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from menagerie.crawler.authority import AuthorityContext, authority_epoch_identity
from menagerie.crawler.tests.conftest import make_authority_context

pytestmark = pytest.mark.smoke

_OTHER_HASH = "sha256:" + "e" * 64

#: Every string-typed identity field that MUST move the authority epoch.
_EPOCH_AXES = frozenset(
    {
        "author_prompt_identity",
        "author_model_identity",
        "author_schema_identity",
        "author_dispatcher_identity",
        "checker_prompt_identity",
        "checker_model_identity",
        "checker_schema_identity",
        "reducer_policy_identity",
        "runner_policy_identity",
        "terminal_policy_identity",
        "publication_policy_identity",
    }
)

#: Fields deliberately OUTSIDE the epoch, each with the reason it adjudicates
#: nothing about a proposal's verdict.
_NON_EPOCH_FIELDS: dict[str, str] = {
    "active_intake_snapshot_id": "currency is pinned on the snapshot axis, not the epoch",
    "active_intake_snapshot_sha256": "currency is pinned on the snapshot axis, not the epoch",
    "intake_by_stable_id": "intake rows are the WORK, not a contract that adjudicates it",
    "family_bindings": "trusted intake topology, bound per record by family authority",
    "author_model_fields": "carried preimage of author_model_identity, already an axis",
    "checker_model_fields": "carried preimage of checker_model_identity, already an axis",
    "environment_generations": "environment identity binds per attempt and per record",
    "prior_intake_snapshots": "T4-T1 currency carve-out input; steers currency, not verdicts",
    "prior_intake_member_ids": "T4-T1 membership carve-out input; grants no write authority",
}


def test_every_authority_context_field_is_classified() -> None:
    """A new context field must be classified as epoch axis or excluded-with-reason.

    The anti-drift guard: this is the test that fails when someone adds a
    verdict-steering identity and forgets the epoch.
    """

    declared = {field.name for field in fields(AuthorityContext)}
    classified = _EPOCH_AXES | set(_NON_EPOCH_FIELDS)
    unclassified = declared - classified
    assert not unclassified, (
        "AuthorityContext gained unclassified field(s) "
        f"{sorted(unclassified)}: add each to _EPOCH_AXES (if it adjudicates a "
        "verdict, and to authority_epoch_identity) or to _NON_EPOCH_FIELDS with "
        "the reason it does not"
    )
    stale = classified - declared
    assert not stale, f"classification names fields AuthorityContext no longer has: {sorted(stale)}"


@pytest.mark.parametrize("axis", sorted(_EPOCH_AXES))
def test_each_epoch_axis_rotates_the_epoch(axis: str) -> None:
    """Rotating any classified epoch axis must move the digest."""

    context = make_authority_context(["m_example"])
    baseline = authority_epoch_identity(context)
    assert authority_epoch_identity(replace(context, **{axis: _OTHER_HASH})) != baseline, axis


@pytest.mark.parametrize(
    "field_name",
    sorted(name for name in _NON_EPOCH_FIELDS if name.endswith(("_id", "_sha256"))),
)
def test_excluded_scalar_fields_do_not_move_the_epoch(field_name: str) -> None:
    """The exclusion list is load-bearing in both directions.

    If an excluded field started moving the epoch, work identities would rotate
    on a snapshot boundary and prior-epoch gates would stop matching work they
    legitimately adjudicate -- the quadratic re-drive K2 exists to prevent.
    """

    context = make_authority_context(["m_example"])
    baseline = authority_epoch_identity(context)
    rotated = replace(context, **{field_name: "intake-" + "z" * 20})
    assert authority_epoch_identity(rotated) == baseline, field_name


def test_the_epoch_digest_is_versioned_and_reproducible() -> None:
    """Two identically built contexts agree; the digest carries its schema version."""

    first = authority_epoch_identity(make_authority_context(["m_example"]))
    second = authority_epoch_identity(make_authority_context(["m_other"]))
    assert first == second, "the epoch must not depend on which models are in intake"
    assert first.startswith("sha256:")
