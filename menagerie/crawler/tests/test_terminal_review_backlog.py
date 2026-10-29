"""Unit pins for the terminal-review backlog sweep's pure halves.

The sweep's effectful half lives on the driver
(``CrawlerDriver._review_terminal_backlog``; end-to-end coverage in
``test_terminal_review_backlog_e2e.py``). This module pins the pure halves in
``menagerie.crawler.terminal_review`` plus the reducer's replay-derived
work-generation read: eligibility is exactly the grant lane's
``unreviewed-terminal`` hold predicate read from the other side, the hold
candidate is the containment ladder's marking over a verbatim head (the
ORIGINAL status code survives so ``_grant_stage_binds_terminal`` still binds),
and the gate pre-filter trusts only a current-identity newest-row adjudication.
"""

from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest

from menagerie.crawler.checker_dispatch import component_identity
from menagerie.crawler.constants import (
    AUTHOR_DISPATCHER_COMPONENT,
    AUTHOR_RESULT_SCHEMA_COMPONENT,
    GATE_SCHEMA_VERSION_V3,
)
from menagerie.crawler.gate_reuse import CheckerAdjudicationIdentity
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.terminal_review import (
    TERMINAL_REVIEW_BACKLOG_PER_LEG,
    TERMINAL_REVIEW_CENSUS_CLASSES,
    TERMINAL_UNADJUDICATED_DECISION,
    current_identity_terminal_gate,
    terminal_review_hold_candidate,
    unreviewed_failed_terminal_heads,
)

IDENTITY = CheckerAdjudicationIdentity(
    provider="codex",
    model="gpt-5.6-sol",
    version="v1",
    prompt_sha256="sha256:" + "a" * 64,
)


def _head(
    stable_id: str,
    *,
    kind: str = "failed",
    code: str = "failed:author",
    review_required: Any = False,
) -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "record_seq": 7,
        "record_revision": f"sha256:rev-{stable_id}",
        "ledger_seq": 9,
        "payload_sha256": f"sha256:payload-{stable_id}",
        "parent_revision": f"sha256:parent-{stable_id}",
        "status": {
            "kind": kind,
            "code": code,
            "stage": code.split(":", 1)[1] if kind == "failed" else None,
            "attempt_ids": [f"sha256:attempt-{stable_id}"],
            "human_review": {
                "required": review_required,
                "reason": None,
                "queue": None,
                "requested_at": None,
            },
        },
        "provenance": {
            "author_model": "claude-fable-5-1",
            "checker_model": "gpt-5.6-sol",
            "producer_run_id": "crawler-run",
            "machine_id": "legacy-machine",
        },
        "budget": {"explicit_grants": []},
    }


def _terminal_gate(
    *,
    stable_id: str,
    work_id: str,
    identity: CheckerAdjudicationIdentity = IDENTITY,
    schema_version: str = GATE_SCHEMA_VERSION_V3,
) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "schema_version": schema_version,
        "gate_kind": "terminal_disposition",
        "gate_id": f"gate-{stable_id}-{work_id}",
        "checker": {
            "provider": identity.provider,
            "model": identity.model,
            "version": identity.version,
            "prompt_sha256": identity.prompt_sha256,
        },
        "author_result_schema_identity": component_identity(AUTHOR_RESULT_SCHEMA_COMPONENT),
        "dispatcher_identity": component_identity(AUTHOR_DISPATCHER_COMPONENT),
        "items": [{"stable_id": stable_id, "work_id": work_id, "verdict": "accepted"}],
    }
    gate["result_envelope_sha256"] = stable_hash(gate)
    return gate


# ---------------------------------------------------------------------------
# unreviewed_failed_terminal_heads


@pytest.mark.smoke
def test_eligibility_is_the_hold_predicate_read_from_the_other_side() -> None:
    heads: dict[str, Mapping[str, Any]] = {
        "m9": _head("m9"),
        "m1": _head("m1"),
        "m2": _head("m2", review_required=True),
        "m3": _head("m3", kind="runs", code="runs"),
        "m4": _head("m4", kind="deferred", code="deferred:needs-cuda"),
        "m5": {"stable_id": "m5", "status": "not-a-mapping"},
    }
    eligible = unreviewed_failed_terminal_heads(heads)
    assert [head["stable_id"] for head in eligible] == ["m1", "m9"]


@pytest.mark.smoke
def test_absent_and_null_review_blocks_are_eligible() -> None:
    bare = _head("m1")
    del bare["status"]["human_review"]
    nulled = _head("m2")
    nulled["status"]["human_review"] = {}
    eligible = unreviewed_failed_terminal_heads({"m1": bare, "m2": nulled})
    assert [head["stable_id"] for head in eligible] == ["m1", "m2"]


# ---------------------------------------------------------------------------
# terminal_review_hold_candidate


@pytest.mark.smoke
def test_hold_candidate_is_the_head_verbatim_with_only_the_containment_marking() -> None:
    head = _head("m1", code="failed:runner")
    candidate = terminal_review_hold_candidate(
        head,
        requested_at="2026-08-21T00:00:00Z",
        producer_run_id="run-current",
        machine_id="current-machine",
    )
    # Ledger-assigned fields are stripped for re-assignment.
    for stripped in ("record_seq", "record_revision", "ledger_seq", "payload_sha256"):
        assert stripped not in candidate
    # Supersession parentage binds the exact head revision.
    assert candidate["parent_revision"] == head["record_revision"]
    assert candidate["status"]["supersedes_revision"] == head["record_revision"]
    # The ORIGINAL status code and stage survive: rewriting them would flip the
    # grant's `unreviewed-terminal` hold into a permanent `stage-mismatch`.
    assert candidate["status"]["code"] == "failed:runner"
    assert candidate["status"]["stage"] == "runner"
    # The marking is exactly the fresh producer's review-held block.
    assert candidate["status"]["human_review"] == {
        "required": True,
        "reason": None,
        "queue": "crawler-human-review",
        "requested_at": "2026-08-21T00:00:00Z",
    }
    # Position is provenance: the re-append cites the run that appends it,
    # never the superseded head's stale citation (ordinary leg 2, 2026-08-27:
    # inherited retired citations orphaned the rows in the open generation's
    # window and the partition proof refused the whole leg).
    assert candidate["provenance"]["producer_run_id"] == "run-current"
    assert candidate["provenance"]["machine_id"] == "current-machine"
    # Authorship provenance -- the CONTENT's producers -- stays verbatim.
    assert candidate["provenance"]["author_model"] == head["provenance"]["author_model"]
    assert candidate["provenance"]["checker_model"] == head["provenance"]["checker_model"]
    # Everything else is byte-preserved.
    assert candidate["budget"] == head["budget"]
    assert candidate["status"]["attempt_ids"] == head["status"]["attempt_ids"]


@pytest.mark.smoke
def test_hold_candidate_never_mutates_the_head() -> None:
    head = _head("m1")
    import copy

    frozen = copy.deepcopy(head)
    terminal_review_hold_candidate(
        head,
        requested_at="2026-08-21T00:00:00Z",
        producer_run_id="run-current",
        machine_id="current-machine",
    )
    assert head == frozen


@pytest.mark.smoke
def test_hold_candidate_requires_a_status_block() -> None:
    head = _head("m1")
    head["status"] = "torn"
    with pytest.raises(ValueError):
        terminal_review_hold_candidate(
            head,
            requested_at="2026-08-21T00:00:00Z",
            producer_run_id="run-current",
            machine_id="current-machine",
        )


@pytest.mark.smoke
def test_hold_candidate_requires_a_provenance_block() -> None:
    # The citation stamps are the law, not a convenience: a head without a
    # provenance block cannot be re-cited and must refuse loudly rather than
    # mint a row whose in-window citation the partition proof would orphan.
    head = _head("m1")
    del head["provenance"]
    with pytest.raises(ValueError):
        terminal_review_hold_candidate(
            head,
            requested_at="2026-08-21T00:00:00Z",
            producer_run_id="run-current",
            machine_id="current-machine",
        )


# ---------------------------------------------------------------------------
# current_identity_terminal_gate


@pytest.mark.smoke
def test_gate_prefilter_requires_a_live_identity() -> None:
    gates = [_terminal_gate(stable_id="m1", work_id="work-m1")]
    assert (
        current_identity_terminal_gate(
            gates, stable_id="m1", work_id="work-m1", checker_identity=None
        )
        is None
    )


@pytest.mark.smoke
def test_gate_prefilter_matches_only_the_exact_generation() -> None:
    gates = [_terminal_gate(stable_id="m1", work_id="work-m1")]
    found = current_identity_terminal_gate(
        gates, stable_id="m1", work_id="work-m1", checker_identity=IDENTITY
    )
    assert found is not None and found["gate_id"] == "gate-m1-work-m1"
    assert (
        current_identity_terminal_gate(
            gates, stable_id="m1", work_id="sha256:granted", checker_identity=IDENTITY
        )
        is None
    )
    assert (
        current_identity_terminal_gate(
            gates, stable_id="m2", work_id="work-m2", checker_identity=IDENTITY
        )
        is None
    )


@pytest.mark.smoke
def test_gate_prefilter_newest_row_decides_and_identity_mismatch_refuses() -> None:
    rotated = CheckerAdjudicationIdentity(
        provider="codex",
        model="gpt-6-terra",
        version="v2",
        prompt_sha256="sha256:" + "b" * 64,
    )
    old = _terminal_gate(stable_id="m1", work_id="work-m1")
    new = _terminal_gate(stable_id="m1", work_id="work-m1", identity=rotated)
    # The newest row containing the head decides; its identity mismatch must
    # refuse to a live review rather than fall back into superseded history.
    assert (
        current_identity_terminal_gate(
            [old, new], stable_id="m1", work_id="work-m1", checker_identity=IDENTITY
        )
        is None
    )
    assert (
        current_identity_terminal_gate(
            [new, old], stable_id="m1", work_id="work-m1", checker_identity=IDENTITY
        )
        is not None
    )


# ---------------------------------------------------------------------------
# CanonicalReducer.terminal_work_generation (unbound over a facade)


def _reducer_facade(
    heads: Mapping[str, Mapping[str, Any]], attempts: Mapping[str, Mapping[str, Any]]
) -> SimpleNamespace:
    return SimpleNamespace(_current=dict(heads), _attempt_index=lambda: dict(attempts))


@pytest.mark.smoke
def test_terminal_work_generation_reads_the_newest_cited_attempt() -> None:
    head = _head("m1")
    head["status"]["attempt_ids"] = ["sha256:a1", "sha256:a2"]
    facade = _reducer_facade(
        {"m1": head},
        {
            "sha256:a1": {"work_id": "work-m1"},
            "sha256:a2": {"work_id": "sha256:granted-generation"},
        },
    )
    assert (
        CanonicalReducer.terminal_work_generation(cast(CanonicalReducer, facade), "m1")
        == "sha256:granted-generation"
    )


@pytest.mark.smoke
def test_terminal_work_generation_skips_uncited_and_generationless_history() -> None:
    head = _head("m1")
    head["status"]["attempt_ids"] = ["sha256:missing", "sha256:no-work-id"]
    facade = _reducer_facade({"m1": head}, {"sha256:no-work-id": {"work_id": None}})
    assert CanonicalReducer.terminal_work_generation(cast(CanonicalReducer, facade), "m1") is None
    assert CanonicalReducer.terminal_work_generation(cast(CanonicalReducer, facade), "m-absent") is None


# ---------------------------------------------------------------------------
# Closed vocabularies


@pytest.mark.smoke
def test_census_classes_are_the_closed_disclosure_vocabulary() -> None:
    assert TERMINAL_REVIEW_CENSUS_CLASSES == (
        "confirmed",
        "review_held",
        "already_adjudicated",
        "no_consumer",
        "claim_unrehydratable",
        "revision_refused",
        "dormant",
        "parked",
        "beyond_bound",
    )
    assert TERMINAL_REVIEW_BACKLOG_PER_LEG == 128


@pytest.mark.smoke
def test_unadjudicated_decision_constant_is_the_producer_wording() -> None:
    # driver_models imports this exact constant as its only source of the
    # ungated-verdict wording; freezing the bytes here makes any rewording a
    # reviewed change on both sides at once.
    assert TERMINAL_UNADJUDICATED_DECISION == (
        "author published a typed terminal verdict; its disposition gate "
        "was never obtained, so the verdict is unadjudicated"
    )
