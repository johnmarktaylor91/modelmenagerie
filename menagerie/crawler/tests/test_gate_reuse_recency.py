"""Reuse must respect RECENCY, not just identity: no stale verdict resurrection.

RETIREMENT NOTE (BL-10 h): live semantic reuse is disabled
(``GATE_VERDICT_REUSE_RETIRED``), so every entry point refuses everything.
These tests are kept as the fail-closed regression suite for any deliberate
future re-arming: every refusal case below must STILL refuse then, and the
former positive controls now assert the retirement instead.

``gate_reuse._latest_reuse_match`` documents the rule it means to enforce:

    Only the latest fully matching row is ever eligible: if it fails a
    reuse-side eligibility replay (the canonical-write tripwire), the answer is
    a live session, never a fall-back to an older adjudication the checker may
    since have superseded.

The implementation applied that rule to only TWO of the three ways a row can be
ineligible. A prior row whose payload no longer matches the proposal returns
None (live), and an ACCEPTED prior returns None (live) -- but a row that fails
the CHECKER-IDENTITY match was skipped with ``continue``, so the scan walked
PAST it into older history and replayed a superseded verdict.

That is the K1 defect class -- a stale rejection beating a later acceptance --
reintroduced through the reuse door, where K1's kernel fix (which hardened
``authority``) does not reach. Two ways the ledger legitimately interleaves
adjudications under different checker identities:

* **A rotated then reverted checker prompt or model.** PROMPT1 rotated three
  prompt digests; a rotation that is rolled back restores the earlier identity,
  and every adjudication bought in between becomes invisible to the identity
  filter while remaining the LATEST word on that work.
* **Merged host shards.** ``tools/merge_host_shards.py`` interleaves rows from
  hosts whose configs pin different checker models, so at 20k scale the newest
  adjudication of a model routinely carries a sibling host's identity.

In both cases the pre-fix scan replayed the older same-identity row. When that
row was a REJECTION and the newer one an ACCEPTANCE, reuse resurrected a verdict
the checker had already reversed -- for free, with no session to notice it, and
charged it against the model's bounded repair cap.

The fix makes recency structural: the LATEST adjudication of this work in this
lane decides, and if it is not reusable for ANY reason -- payload drift,
acceptance, or an identity the current lane cannot claim -- the answer is a live
session. Nothing is weakened; the change only removes fall-backs.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.gate_reuse import (
    CheckerAdjudicationIdentity,
    plan_metadata_gate_reuse,
    reusable_fidelity_gate,
    reusable_metadata_gate,
)
from menagerie.crawler.tests.test_gate_verdict_reuse import (
    _artifact,
    _fidelity_artifact,
    _identity,
    _prior_fidelity_gate,
    _prior_metadata_gate,
    _reseal,
)

pytestmark = pytest.mark.smoke


def _foreign_identity_row(gate: dict[str, Any], *, gate_id: str) -> dict[str, Any]:
    """Re-seal one prior row under a checker identity the current lane never claims.

    Models the rotated-then-reverted prompt and the merged-shard cases alike: a
    real adjudication of the same work, durably recorded, whose checker identity
    does not equal the identity the lane would buy today.
    """

    foreign = deepcopy(gate)
    foreign["gate_id"] = gate_id
    foreign["checker"]["prompt_sha256"] = "sha256:" + "b" * 64
    return _reseal(foreign)


# ---------------------------------------------------------------------------
# The load-bearing case: a later ACCEPTANCE must never be undone by reuse.
# ---------------------------------------------------------------------------


def test_stale_rejection_is_never_replayed_over_a_later_foreign_acceptance(
    tmp_path: Path,
) -> None:
    """An older rejection must not outrank the latest adjudication.

    Ledger order: the item was rejected under the identity the lane claims
    today, then ACCEPTED under a rotated identity. Replaying the rejection
    resurrects a verdict the checker reversed, and does it for free.
    """

    artifacts = (_artifact(tmp_path),)
    rejection = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    later_acceptance = _foreign_identity_row(
        _prior_metadata_gate(tmp_path, artifacts), gate_id="gate-later-accept"
    )

    replay = reusable_metadata_gate(artifacts, (rejection, later_acceptance), checker=_identity())

    assert replay is None, "reuse replayed a rejection the latest adjudication had already reversed"


def test_stale_rejection_is_never_replayed_over_any_later_foreign_row(
    tmp_path: Path,
) -> None:
    """Recency is structural, not verdict-dependent.

    Even when the newer foreign-identity adjudication also rejected, its
    findings and repair demands -- the payload the author must answer -- are the
    current word. Replaying the older row's stale complaint set would charge the
    bounded repair cap against a disclosure the checker has since superseded, so
    the rule is simply: the latest adjudication decides, or a session is bought.
    """

    artifacts = (_artifact(tmp_path),)
    older = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    newer = _foreign_identity_row(
        _prior_metadata_gate(tmp_path, artifacts, verdict="cannot-verify"),
        gate_id="gate-later-reject",
    )

    assert reusable_metadata_gate(artifacts, (older, newer), checker=_identity()) is None


def test_fidelity_reuse_honors_the_same_recency_rule(tmp_path: Path) -> None:
    """The fidelity lane shares the scan, so it shares the guarantee."""

    artifact = _fidelity_artifact(tmp_path)
    rejection = _prior_fidelity_gate(tmp_path, artifact)
    later_acceptance = _foreign_identity_row(
        _prior_fidelity_gate(tmp_path, artifact, rejected=False),
        gate_id="gate-later-fidelity",
    )

    replay = reusable_fidelity_gate(artifact, (rejection, later_acceptance), checker=_identity())

    assert replay is None


def test_planner_also_refuses_the_superseded_member(tmp_path: Path) -> None:
    """The pre-batching planner must route the same model to live work.

    The planner resolves reuse groups with the same scan; if it disagreed with
    ``reusable_metadata_gate`` the member would be extracted into a reuse group
    that then refuses at pop time, and the batch it was removed from has already
    been formed -- a silently under-packed live batch.
    """

    artifacts = (_artifact(tmp_path),)
    rejection = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    later_acceptance = _foreign_identity_row(
        _prior_metadata_gate(tmp_path, artifacts), gate_id="gate-later-accept"
    )

    groups, live = plan_metadata_gate_reuse(
        artifacts, (rejection, later_acceptance), checker=_identity()
    )

    assert groups == ()
    assert live == artifacts


# ---------------------------------------------------------------------------
# The control: recency must not become a blanket refusal.
# ---------------------------------------------------------------------------


def test_matching_latest_row_still_buys_live_under_retirement(
    tmp_path: Path,
) -> None:
    """The former positive control now asserts the retirement.

    Pre-retirement this pinned "a foreign row OLDER than the match must not
    block reuse" (the guard against over-tightening); with reuse retired the
    perfectly claimable latest row must ALSO refuse. On re-arming, restore the
    original positive assertion from history.
    """

    artifacts = (_artifact(tmp_path),)
    older_foreign = _foreign_identity_row(
        _prior_metadata_gate(tmp_path, artifacts, verdict="cannot-verify"),
        gate_id="gate-older-foreign",
    )
    latest_claimable = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")

    replay = reusable_metadata_gate(
        artifacts, (older_foreign, latest_claimable), checker=_identity()
    )

    assert replay is None


def test_foreign_rows_for_other_models_still_buy_live_under_retirement(
    tmp_path: Path,
) -> None:
    """Former per-model recency control: everything refuses under retirement."""

    artifacts = (_artifact(tmp_path, "m_target"),)
    others = (_artifact(tmp_path, "m_other"),)
    rejection = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    unrelated_newer = _foreign_identity_row(
        _prior_metadata_gate(tmp_path, others), gate_id="gate-other-model"
    )

    assert reusable_metadata_gate(artifacts, (rejection, unrelated_newer), checker=_identity()) is None


def test_a_foreign_row_of_another_kind_still_buys_live_under_retirement(
    tmp_path: Path,
) -> None:
    """Former per-lane recency control: everything refuses under retirement."""

    artifacts = (_artifact(tmp_path),)
    rejection = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    other_kind = deepcopy(rejection)
    other_kind["gate_id"] = "gate-other-kind"
    other_kind["gate_kind"] = "terminal_disposition"
    other_kind = _reseal(other_kind)

    assert reusable_metadata_gate(artifacts, (rejection, other_kind), checker=_identity()) is None


def test_unreadable_newer_row_forces_a_live_session(tmp_path: Path) -> None:
    """A torn newest adjudication is unreadable history, not permission to fall back.

    A row whose self-hash no longer replays is exactly the tamper/torn-row class
    the gate-level matcher refuses. Refusing it must not promote the row behind
    it: the truth is that the latest word on this work cannot be read, so the
    verdict must be re-bought.
    """

    artifacts = (_artifact(tmp_path),)
    rejection = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    torn = deepcopy(_prior_metadata_gate(tmp_path, artifacts, verdict="cannot-verify"))
    torn["gate_id"] = "gate-torn"
    torn["result_envelope_sha256"] = "sha256:" + "c" * 64  # deliberately NOT resealed

    assert reusable_metadata_gate(artifacts, (rejection, torn), checker=_identity()) is None


def test_replaying_a_replay_is_refused(tmp_path: Path) -> None:
    """A marked replay row is not a fresh verdict source.

    Chaining replays would let one bought adjudication spawn an unbounded chain
    of marked rows whose ``reused_from`` bindings point at each other rather than
    at the session that was actually paid for, and each link charges the bounded
    repair cap again. The verdict source must always be a row a checker session
    actually produced.
    """

    from menagerie.crawler.tests.test_gate_verdict_reuse import _historical_replay_row

    artifacts = (_artifact(tmp_path),)
    rejection = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    # Minted the way pre-retirement code minted replays; the live entry points
    # refuse everything now, so the marked row is built directly.
    first = _historical_replay_row(tmp_path, artifacts[0], rejection)

    persisted_replay = dict(deepcopy(first))
    persisted_replay["ledger_seq"] = 2
    persisted_replay["payload_sha256"] = "sha256:" + "d" * 64

    second = reusable_metadata_gate(artifacts, (rejection, persisted_replay), checker=_identity())

    assert second is None, (
        "a replay row became a verdict source, so reused_from no longer names "
        "the session that was bought"
    )


# ---------------------------------------------------------------------------
# Payload identity completeness: every verified hash, one at a time.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hash_key",
    ("proposal", "source_manifest", "evidence", "code", "source_to_code_map", "family_template"),
)
def test_each_verified_hash_independently_forces_a_live_session(
    tmp_path: Path, hash_key: str
) -> None:
    """All SIX bound hashes are load-bearing, not just the two W4 parametrized.

    ``_gate_item_matches_proposal`` compares the verified-hash mapping by full
    key SET and by every value, so each key is pinned today. This asserts it key
    by key so a future refactor that compares a fixed subset (or drops a key
    from the expectation) fails here instead of silently widening the reuse key.
    """

    artifacts = (_artifact(tmp_path),)
    prior = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    # (Retirement: the positive control -- reuse succeeding on the unmutated
    # prior -- is gone; every arm refuses. The per-key drift refusal below is
    # what must survive a future re-arming.)

    drifted = deepcopy(prior)
    hashes = drifted["items"][0]["verified_hashes"]
    assert hash_key in hashes, f"{hash_key} is not a bound verified hash"
    hashes[hash_key] = "sha256:" + "e" * 64 if hashes[hash_key] is None else None
    drifted = _reseal(drifted)

    assert reusable_metadata_gate(artifacts, (drifted,), checker=_identity()) is None


def test_a_dropped_verified_hash_key_forces_a_live_session(tmp_path: Path) -> None:
    """Removing a bound hash key entirely is drift, not a smaller payload."""

    artifacts = (_artifact(tmp_path),)
    prior = deepcopy(_prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate"))
    del prior["items"][0]["verified_hashes"]["evidence"]
    prior = _reseal(prior)

    assert reusable_metadata_gate(artifacts, (prior,), checker=_identity()) is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider", "anthropic"),
        ("model", "some-other-checker"),
        ("version", "channel-next"),
        ("prompt_sha256", "sha256:" + "f" * 64),
    ),
)
def test_each_checker_identity_component_independently_forces_live(
    tmp_path: Path, field: str, value: str
) -> None:
    """Every declared checker-identity component is individually load-bearing."""

    artifacts = (_artifact(tmp_path),)
    prior = deepcopy(_prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate"))
    prior["checker"][field] = value
    prior = _reseal(prior)

    assert reusable_metadata_gate(artifacts, (prior,), checker=_identity()) is None


def test_a_lane_claiming_a_partially_matching_identity_never_reuses(
    tmp_path: Path,
) -> None:
    """Reuse compares the WHOLE declared identity, never a majority of it."""

    artifacts = (_artifact(tmp_path),)
    prior = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    baseline = _identity()
    partial = CheckerAdjudicationIdentity(
        provider=baseline.provider,
        model=baseline.model,
        version=baseline.version,
        prompt_sha256="sha256:" + "0" * 64,
    )

    assert reusable_metadata_gate(artifacts, (prior,), checker=partial) is None
