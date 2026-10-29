"""W13: ``failed:accuracy-gate`` is split from disposition rejection.

R24's census-mislabel class: 200a carries 27 ``failed:accuracy-gate`` heads
whose ``accuracy_gate.verdict`` is ``None`` -- the checker refused the AUTHOR'S
advisory terminal CLAIM (a disposition review), yet the head asserts an
accuracy adjudication of authored metadata that never happened, and ~60% of
the "5% run rate" statistic was machine artifact. The split:

* a refused/unverifiable terminal disposition mints ``blocked:disposition-review``
  (kind ``blocked``, stage-less, closed reason pair, mandatory human review);
* a FRESH ``failed:accuracy-gate`` claim carrying a terminal-disposition
  reason refuses at reduction (the accuracy-gate terminal requires a RENDERED
  adjudication: the metadata-repair lineage's rejected verdicts, or the closed
  machine-witnessed ``checker-session-exhaustion``);
* era honesty (PD-13, forward-only): historical heads stay readable, and the
  review sweep's original-code re-appends still pass (parent-head continuity);
  the census discloses the historical shape
  (``accuracy_gate_disposition_mislabel``), never silently normal;
* era REPLAY passage is bound to the frozen realized-history cohort
  (``DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS``): semantic replay of a
  threaded persisted row whose RECOMPUTED bytes digest is a member and whose
  ``accuracy_gate.verdict`` is ``None``. Replay mode alone is caller identity,
  never era authority -- a self-hashed current-era row written straight into
  the ledger (the reconciliation's R1 attack) refuses identically to a fresh
  live mint.

Wave placement: Wave 2 by law (extends the closed status kind/code
vocabularies and the model schema enums).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    _derive_terminal_disposition_failure,
)
from menagerie.crawler.constants import (
    BLOCKED_DISPOSITION_REVIEW_REASON_CODES,
    BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
    NO_RUNG_SELECTED_STATUS_KINDS,
    TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
    TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE,
    TERMINAL_STATUS_CODES,
    StatusKind,
)
import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.constants import DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS
from menagerie.crawler.identity import payload_hash
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.status import _status_completeness_failures, funnel_counts


def _status(
    code: str,
    *,
    kind: str | None = None,
    stage: str | None = None,
    reason: str | None = None,
    human_review: bool = True,
) -> dict:
    """Build one status block for the validator fixtures."""

    return {
        "code": code,
        "kind": kind if kind is not None else code.split(":", 1)[0],
        "stage": stage,
        "reason_code": reason,
        "traceback": None,
        "no_traceback_reason": "checker decision produced no Python traceback",
        "human_review": {"required": human_review},
    }


def _validate(
    model: dict,
    *,
    current: dict | None = None,
    semantic_replay: bool | None = None,
    source_record: dict | None = None,
) -> None:
    """Run the reducer's closed status validation over one candidate.

    ``semantic_replay=None`` mirrors the live reducer exactly: the attribute is
    absent, as on every ``CanonicalReducer`` the driver constructs. ``True``
    mirrors the write-free replay reducers ``_replay_reducer`` builds over
    persisted ledger bytes; ``False`` pins that an explicit falsy attribute
    grants nothing. ``source_record`` mirrors the persisted ledger row the
    replay caller threads (``None`` = a non-resident candidate, exactly what
    the award-restoration replay declares).
    """

    duck = SimpleNamespace(_current=current or {}, _replay_source_record=source_record)
    if semantic_replay is not None:
        duck._semantic_replay = semantic_replay
    CanonicalReducer._validate_status(duck, model)  # noqa: SLF001


def _era_row(reason: str, *, verdict: object = None, **overrides: object) -> tuple[dict, dict]:
    """Build one R24-shape candidate plus the persisted row it replays from.

    The source row is the candidate with the append-assigned ``record_seq``
    stamped back on, exactly the relationship between a persisted ledger row
    and the stripped replay candidate the projection validates.
    """

    model = {
        "stable_id": "m-era",
        "accuracy_gate": {"verdict": verdict},
        "status": _status(
            "failed:accuracy-gate", stage="accuracy-gate", reason=reason
        ),
        **overrides,
    }
    return model, {**model, "record_seq": 7}


@pytest.mark.smoke
def test_vocabulary_extensions_are_exact() -> None:
    """The closed vocabularies gained exactly the disposition-review members."""

    assert BLOCKED_DISPOSITION_REVIEW_STATUS_CODE == "blocked:disposition-review"
    assert BLOCKED_DISPOSITION_REVIEW_STATUS_CODE in TERMINAL_STATUS_CODES
    assert StatusKind.BLOCKED.value == "blocked"
    assert StatusKind.BLOCKED.value in NO_RUNG_SELECTED_STATUS_KINDS
    assert BLOCKED_DISPOSITION_REVIEW_REASON_CODES == frozenset(
        {
            TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
            TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE,
        }
    )
    schema = json.loads(
        Path("menagerie/crawler/schemas/model-common.schema.json").read_text(encoding="utf-8")
    )
    assert "blocked" in schema["$defs"]["status"]["properties"]["kind"]["enum"]
    assert BLOCKED_DISPOSITION_REVIEW_STATUS_CODE in schema["$defs"]["status_code"]["enum"]


@pytest.mark.smoke
def test_fresh_accuracy_gate_claim_with_a_disposition_reason_refuses() -> None:
    """The RED core: reduction refuses the pre-W13 mislabel on a FRESH mint."""

    mislabeled = {
        "stable_id": "m-fresh",
        "status": _status(
            "failed:accuracy-gate",
            stage="accuracy-gate",
            reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        ),
    }
    with pytest.raises(ReductionError, match="blocked:disposition-review"):
        _validate(mislabeled)
    # ... for BOTH disposition reasons.
    mislabeled["status"]["reason_code"] = TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(mislabeled)


@pytest.mark.smoke
def test_era_reappend_of_a_historical_mislabel_still_passes() -> None:
    """PD-13 forward-only: the review sweep re-appends the ORIGINAL code.

    A candidate whose CURRENT head already carries the exact historical
    code+reason (the 27-head 200a shape) passes through unchanged -- the
    refusal fires only on fresh claims.
    """

    historical = _status(
        "failed:accuracy-gate",
        stage="accuracy-gate",
        reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
    )
    _validate(
        {"stable_id": "m-era", "status": dict(historical)},
        current={"m-era": {"status": dict(historical)}},
    )
    # A DIFFERENT-reason head grants no era passage: continuity is exact.
    with pytest.raises(ReductionError, match="blocked:disposition-review"):
        _validate(
            {"stable_id": "m-era", "status": dict(historical)},
            current={
                "m-era": {
                    "status": _status(
                        "failed:accuracy-gate",
                        stage="accuracy-gate",
                        reason="inaccurate-cap-exhausted",
                    )
                }
            },
        )


@pytest.mark.smoke
def test_era_frozen_cohort_is_exact() -> None:
    """The frozen realized-history cohort is exactly the derived 52 digests.

    Anti-substitution pin: 52 members, every one a canonical ``sha256:`` bytes
    digest, and the census hash over the sorted joined members matches the
    derivation recorded in the constant's docstring. Widening the set is a
    reviewed change that must update this pin in the same commit (the
    four-part doctrine's exact-identity-bindings clause) -- a runtime or
    producer-driven widening cannot.
    """

    import hashlib
    import re

    assert isinstance(DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS, frozenset)
    assert len(DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS) == 52
    pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    assert all(pattern.match(d) for d in DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS)
    census = hashlib.sha256(
        "\n".join(sorted(DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS)).encode("utf-8")
    ).hexdigest()
    assert census == "57b755eb94c6e64affcd9b5150663decc77628204548cfc012eb00cfda34cece"


@pytest.mark.smoke
def test_semantic_replay_of_a_frozen_cohort_member_is_era_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Era honesty is replay-symmetric for the FROZEN realized cohort only.

    The R24 deadlock: the projection replays each member's head through a
    replay reducer positioned immediately BEFORE it, so a pre-split first mint
    has no same-shape prior head to match, the head-continuity carve-out never
    fires, and the fresh-claim refusal re-litigated already-attested history
    under later law. The passage re-attests exactly the reviewed historical
    bytes: membership is the RECOMPUTED ``record_revision`` digest of the
    threaded persisted row (this test registers its synthetic rows the same
    way a real era boundary would -- a reviewed change to the frozen set).
    """

    for reason in (
        TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        TERMINAL_DISPOSITION_UNVERIFIABLE_REASON_CODE,
    ):
        # The single-revision class: no prior head at all.
        model, source = _era_row(reason)
        monkeypatch.setattr(
            reducer_module,
            "DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS",
            frozenset({payload_hash(source)}),
        )
        _validate(model, semantic_replay=True, source_record=source)
        # The prior-head class: a prior head under a DIFFERENT failure code.
        _validate(
            model,
            current={
                "m-era": {
                    "status": _status(
                        "failed:author", stage="author", reason="malformed-result"
                    )
                }
            },
            semantic_replay=True,
            source_record=source,
        )


@pytest.mark.smoke
def test_semantic_replay_without_frozen_membership_refuses() -> None:
    """The reconciliation's R1 kill, retired at the unit door.

    A current-law row of the historical shape -- honest sequence, valid
    self-hash, threaded as its own source -- refuses under semantic replay
    because its recomputed bytes digest is not in the frozen realized-history
    cohort. Replay-mode residency alone is caller identity, never era
    authority. A replay over a NON-resident candidate (``source_record=None``,
    the award-restoration declaration) refuses identically.
    """

    model, source = _era_row(TERMINAL_DISPOSITION_REJECTED_REASON_CODE)
    assert payload_hash(source) not in DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(model, semantic_replay=True, source_record=source)
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(model, semantic_replay=True, source_record=None)


@pytest.mark.smoke
def test_era_membership_binds_bytes_not_stable_id_or_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Membership is a recomputed content hash over exact row bytes.

    Same stable id with different bytes refuses; a row self-labeling a frozen
    member's ``record_revision`` refuses (the digest is recomputed, never
    read); and a candidate that differs from its threaded source row refuses
    (no laundering a novel candidate through a resident row's era).
    """

    member_model, member_source = _era_row(TERMINAL_DISPOSITION_REJECTED_REASON_CODE)
    member_digest = payload_hash(member_source)
    monkeypatch.setattr(
        reducer_module,
        "DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS",
        frozenset({member_digest}),
    )
    # Same stable id, one changed producer field: different bytes, no passage.
    tampered_model, tampered_source = _era_row(
        TERMINAL_DISPOSITION_REJECTED_REASON_CODE, notes="tampered"
    )
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(tampered_model, semantic_replay=True, source_record=tampered_source)
    # Self-labeling the frozen digest changes nothing: the recompute ignores
    # the row's own ``record_revision`` field.
    labeled_source = {**tampered_source, "record_revision": member_digest}
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(tampered_model, semantic_replay=True, source_record=labeled_source)
    # A genuine member row cannot vouch for a DIFFERENT candidate.
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(tampered_model, semantic_replay=True, source_record=member_source)
    # Control: the exact member bytes still vouch.
    _validate(member_model, semantic_replay=True, source_record=member_source)


@pytest.mark.smoke
def test_era_passage_refuses_a_rendered_verdict_contradiction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rendered verdict beside a disposition reason is a CONTRADICTION.

    Even a row whose digest is registered in the frozen set refuses when its
    ``accuracy_gate.verdict`` is rendered: the era passage re-attests the
    mislabel class (verdict ``None`` by definition), never a row that claims
    both an accuracy adjudication and a disposition-review outcome.
    """

    model, source = _era_row(
        TERMINAL_DISPOSITION_REJECTED_REASON_CODE, verdict="rejected"
    )
    monkeypatch.setattr(
        reducer_module,
        "DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS",
        frozenset({payload_hash(source)}),
    )
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(model, semantic_replay=True, source_record=source)


@pytest.mark.smoke
def test_live_append_never_gains_era_passage() -> None:
    """The era passage is bound to the replay reducer, never the live door.

    An absent ``_semantic_replay`` attribute (every live reducer) and an
    explicit falsy one both keep the exact fresh-claim refusal.
    """

    mislabeled = {
        "stable_id": "m-fresh",
        "status": _status(
            "failed:accuracy-gate",
            stage="accuracy-gate",
            reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        ),
    }
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(dict(mislabeled))
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(dict(mislabeled), semantic_replay=False)


@pytest.mark.smoke
def test_producer_row_content_cannot_claim_the_era() -> None:
    """Nothing a producer writes steers the era decision.

    The era authority is the reducer's OWN replay disposition -- module-private
    ``_replay_reducer`` construction over hash-validated ledger bytes -- so a
    live candidate carrying forged era-claim fields refuses identically.
    """

    forged = {
        "stable_id": "m-forged",
        "era": "200a",
        "semantic_replay": True,
        "_semantic_replay": True,
        # Self-labeling a REAL frozen member's digest names nothing: the era
        # digest is recomputed from bytes, never read off the row.
        "record_revision": min(DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS),
        "status": {
            **_status(
                "failed:accuracy-gate",
                stage="accuracy-gate",
                reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
            ),
            "era": "200a",
        },
    }
    with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
        _validate(forged)


@pytest.mark.smoke
def test_semantic_replay_is_not_a_general_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay accepts exactly the disclosed historical class -- nothing wider.

    Every OTHER closed-status check keeps full strength on replay, and this
    is proven under MAXIMAL era pressure: each malformed row is threaded as
    its own source AND its digest is registered in the (patched) frozen set,
    so the only thing standing between it and admission is the sibling check
    itself. Era passage can never launder an out-of-vocabulary reason, a
    stage drift, a missing review hold, or a malformed blocked head.
    """

    malformed = (
        (
            "not allowed for failure stage",
            _status(
                "failed:accuracy-gate", stage="accuracy-gate", reason="reason-from-nowhere"
            ),
        ),
        (
            "failure stage must match",
            _status(
                "failed:accuracy-gate",
                stage="author",
                reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
            ),
        ),
        (
            "requires human review",
            _status(
                "failed:accuracy-gate",
                stage="accuracy-gate",
                reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
                human_review=False,
            ),
        ),
        (
            "cannot carry a failure stage",
            _status(
                BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
                stage="accuracy-gate",
                reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
            ),
        ),
    )
    for message, status in malformed:
        model = {
            "stable_id": "m-replay",
            "accuracy_gate": {"verdict": None},
            "status": status,
        }
        source = {**model, "record_seq": 7}
        monkeypatch.setattr(
            reducer_module,
            "DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS",
            frozenset({payload_hash(source)}),
        )
        with pytest.raises(ReductionError, match=message):
            _validate(model, semantic_replay=True, source_record=source)


@pytest.mark.smoke
def test_rendered_accuracy_gate_terminals_still_validate() -> None:
    """The legitimate accuracy-gate reasons are untouched by the split."""

    for reason in ("inaccurate-cap-exhausted", "checker-session-exhaustion"):
        _validate(
            {
                "stable_id": "m-real",
                "status": _status("failed:accuracy-gate", stage="accuracy-gate", reason=reason),
            }
        )


@pytest.mark.smoke
def test_blocked_disposition_review_shape_is_closed() -> None:
    """Stage-less, closed reason pair, mandatory human review -- or refusal."""

    good = {
        "stable_id": "m-blocked",
        "status": _status(
            BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
            reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        ),
    }
    _validate(good)
    with pytest.raises(ReductionError, match="cannot carry a failure stage"):
        _validate(
            {
                "stable_id": "m-blocked",
                "status": _status(
                    BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
                    stage="accuracy-gate",
                    reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
                ),
            }
        )
    with pytest.raises(ReductionError, match="not allowed for blocked"):
        _validate(
            {
                "stable_id": "m-blocked",
                "status": _status(
                    BLOCKED_DISPOSITION_REVIEW_STATUS_CODE, reason="inaccurate-cap-exhausted"
                ),
            }
        )
    with pytest.raises(ReductionError, match="requires human review"):
        _validate(
            {
                "stable_id": "m-blocked",
                "status": _status(
                    BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
                    reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
                    human_review=False,
                ),
            }
        )
    with pytest.raises(ReductionError, match="unknown terminal status code"):
        _validate(
            {
                "stable_id": "m-blocked",
                "status": _status(
                    "blocked:never-registered",
                    kind="blocked",
                    reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
                ),
            }
        )


def _completeness_record(status: dict) -> dict:
    """Wrap one status block in the minimal completeness-check record shape."""

    return {
        "stable_id": str(status.get("code")),
        "status": {**status, "root_cause_fingerprint": "0" * 16},
        "source_resolution": {"rung": "R1_LIBRARY"},
        "discovery_probes": [],
    }


@pytest.mark.smoke
def test_status_completeness_enforces_the_blocked_shape() -> None:
    """The close-side completeness tripwire covers the new kind."""

    good = _completeness_record(
        _status(
            BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
            reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        )
    )
    assert _status_completeness_failures([good]) == []
    staged = _completeness_record(
        _status(
            BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
            stage="accuracy-gate",
            reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        )
    )
    assert any("blocked-carries-failure-stage" in entry for entry in _status_completeness_failures([staged]))
    unreviewed = _completeness_record(
        _status(
            BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
            reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
            human_review=False,
        )
    )
    assert any("missing-human-review" in entry for entry in _status_completeness_failures([unreviewed]))
    foreign_reason = _completeness_record(
        _status(BLOCKED_DISPOSITION_REVIEW_STATUS_CODE, reason="inaccurate-cap-exhausted")
    )
    assert any("blocked-reason-not-closed" in entry for entry in _status_completeness_failures([foreign_reason]))


def _census_record(status: dict, *, verdict: object = None) -> dict:
    """Build one minimal funnel-census record."""

    return {
        "stable_id": str(status.get("code")) + str(status.get("reason_code")),
        "status": status,
        "authored_metadata_state": "pending",
        "source_resolution": {"rung": "R1_LIBRARY"},
        "implementation": {"run_framework": "torch"},
        "modes": {"meaningful_modes": []},
        "completeness": {"source_read_fields_complete": False},
        "fidelity": {"required": False, "current": False, "gate_id": None, "verdict": None},
        "execution": {"accepted_attempt_ids": [], "current": False},
        "accuracy_gate": {"verdict": verdict},
    }


@pytest.mark.smoke
def test_census_discloses_the_historical_mislabel_class() -> None:
    """W13 census labels: the 27-head shape is counted, never silently normal."""

    mislabel = _census_record(
        _status(
            "failed:accuracy-gate",
            stage="accuracy-gate",
            reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        )
    )
    exhaustion = _census_record(
        _status(
            "failed:accuracy-gate", stage="accuracy-gate", reason="checker-session-exhaustion"
        )
    )
    anomaly = _census_record(
        _status(
            "failed:accuracy-gate", stage="accuracy-gate", reason="inaccurate-cap-exhausted"
        )
    )
    blocked = _census_record(
        _status(
            BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
            reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        )
    )
    counts = funnel_counts([mislabel, exhaustion, anomaly, blocked])
    assert counts["accuracy_gate_disposition_mislabel"] == 1
    # The machine-witnessed exhaustion class is legitimate and NOT an anomaly;
    # a cap-family failure with no rendered verdict is.
    assert counts["accuracy_gate_failure_without_rendered_verdict"] == 1
    assert counts["status:blocked:disposition-review"] == 1


@pytest.mark.smoke
def test_disposition_review_proof_rule_is_the_single_adjudication_rule() -> None:
    """The authority proof moves with the code: last adjudication decides."""

    def gate(seq: int, verdict: str, findings: list[str]) -> tuple[dict, dict]:
        return (
            {"gate_id": f"g-{seq}", "ledger_seq": seq, "gate_round": 1},
            {"terminal_disposition": {"verdict": verdict, "findings": findings}},
        )

    gate_id, reason, root = _derive_terminal_disposition_failure(
        [gate(1, "accepted", []), gate(2, "rejected", ["claim contradicts the frozen source"])],
        claim=BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
    )
    assert gate_id == "g-2"
    assert reason == TERMINAL_DISPOSITION_REJECTED_REASON_CODE
    assert root
    # The recency guard: a LAST adjudication that accepted contradicts the hold.
    with pytest.raises(AuthorityDerivationError, match="blocked:disposition-review is contradicted"):
        _derive_terminal_disposition_failure(
            [gate(1, "rejected", ["stale refusal"]), gate(2, "accepted", [])],
            claim=BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
        )
    # No findings, no evidence, no hold.
    with pytest.raises(AuthorityDerivationError, match="unevidenced"):
        _derive_terminal_disposition_failure(
            [gate(1, "rejected", [])], claim=BLOCKED_DISPOSITION_REVIEW_STATUS_CODE
        )


@pytest.mark.smoke
def test_deferred_terminal_grant_sentinel_binds_a_blocked_head() -> None:
    """A stage-less blocked head is grant-re-drivable exactly like its siblings."""

    from menagerie.crawler.constants import DEFERRED_TERMINAL_GRANT_STAGE
    from menagerie.crawler.driver_admission import _grant_stage_binds_terminal

    head = {
        "status": _status(
            BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
            reason=TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
        )
    }
    assert _grant_stage_binds_terminal({"stage": DEFERRED_TERMINAL_GRANT_STAGE}, head)
    assert not _grant_stage_binds_terminal({"stage": "accuracy-gate"}, head)
