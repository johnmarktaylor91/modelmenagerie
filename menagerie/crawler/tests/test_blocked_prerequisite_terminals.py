"""AL-6: honest BLOCKED-prerequisite terminals (JMT-signed vocabulary change).

The pilot's 41 ``AuthorBlockedPrerequisite`` witnesses split 22/19 across two
eras of one mislabel family: a checked-ACCURATE BLOCKED verdict either collapsed
onto ``failed:author``/``malformed-result`` (asserting an authoring defect that
never happened) or landed on ``failed:<stage>`` with the bare, stage-blind
``blocked-prerequisite`` member. Both spellings hide WHAT blocked the stage, and
the runtime-dependency cohort (19 of 20 reclassified members) records a fact
about the ENVIRONMENT ECOSYSTEM as a model defect while charging the author's
attempt allowance for an honest verdict.

Pinned here (the signed AL-6 contract):

* NEW MINTS carry the stage-suffixed reason ``blocked-prerequisite:<stage>``;
  the bare member stays read-valid for historical rows only.
* A runtime-dependency-class prerequisite (typed prerequisite IDs, never free
  text) routes ``failed:environment-adjacent`` -- the NEW closed status code --
  with the blocking stage preserved in the reason suffix.
* A CUDA-capability-class prerequisite routes the EXISTING
  ``deferred:needs-cuda`` valve; the reducer proves that head through the same
  accepted-BLOCKED capability rule that already serves the opus-tier and
  source-access deferrals.
* The witnessing attempt group refunds its ATTEMPT projection through the new
  closed ``author-blocked-prerequisite`` member; sessions and wall stay charged,
  and the provider-storm fuse and exhaustion dimension never count the refund
  as infrastructure weather.
* Injectivity census: on every new-mint row the ``(status_code, reason_code)``
  pair recovers the blocking stage exactly; a leg-close census proves no
  generic ``failed:author`` head is witnessed by ``AuthorBlockedPrerequisite``.
* The 20 reclassified pilot members' real shapes replay through the new
  routing (``fixtures/blocked_prerequisite_pilot_shapes.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest

from menagerie.crawler.author_dispatch import AuthorResultBinding, BlockedRecommendation
from menagerie.crawler.constants import (
    BLOCKED_PREREQUISITE_REASON_CODE,
    FAILURE_REASON_CODES,
    TERMINAL_STATUS_CODES,
)
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.tests.conftest import HASH

pytestmark = pytest.mark.smoke

_CREATED_AT = "2026-08-24T12:00:00Z"

#: The closed stage enum of the author-result blocked arm (author-result-v3/v4).
_BLOCKED_ARM_STAGES = ("source", "fetch", "author", "evidence", "environment", "policy", "runner")

_SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"

_PILOT_SHAPES = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "blocked_prerequisite_pilot_shapes.json")
    .read_text(encoding="utf-8")
)

#: The closed CUDA-capability prerequisite IDs observed in the pilot cohort.
_PILOT_CUDA_IDS = frozenset({"cuda-platform", "cuda-device", "cuda-runtime"})


def _blocked(
    stage: str,
    reason_code: str,
    prerequisite_ids: tuple[str, ...] = ("upstream-weights-release",),
) -> BlockedRecommendation:
    """Build one minimal checked BLOCKED recommendation for the mapping seam."""

    raw_fields = {
        "result_id": "result-blocked",
        "result_sha256": HASH,
        "stable_id": "m1777",
        "work_id": "work-m1777",
        "campaign_id": "work-m1777",
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": HASH,
        "intake_snapshot_id": "intake-1",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": HASH,
        "created_at": _CREATED_AT,
    }
    binding = AuthorResultBinding(
        raw_result={**raw_fields, "kind": "BLOCKED", "payload": {}}, **raw_fields
    )
    return BlockedRecommendation(
        binding=binding,
        stage=stage,
        reason_code=reason_code,
        prerequisite_ids=prerequisite_ids,
        evidence_ids=("ev-blocked-1",),
        evidence_identity=HASH,
        license_identity=HASH,
        recommendation_sha256=HASH,
    )


def _map(
    stage: str,
    reason_code: str,
    prerequisite_ids: tuple[str, ...] = ("upstream-weights-release",),
) -> tuple[str, str | None]:
    """Run the REAL total mapping with the default campaign config."""

    host = cast(CrawlerDriver, SimpleNamespace(config=DriverConfig()))
    return CrawlerDriver._blocked_terminal(host, _blocked(stage, reason_code, prerequisite_ids))


def _schema_defs(name: str) -> dict[str, Any]:
    return json.loads((_SCHEMAS / name).read_text(encoding="utf-8"))["$defs"]


def _model_row(
    stable_id: str,
    status_code: str,
    reason_code: Optional[str],
    *,
    work_id: str = "work-1",
) -> dict[str, Any]:
    """Build one minimal census-consumable terminal head row."""

    kind = status_code.split(":", 1)[0]
    stage = status_code.split(":", 1)[1] if kind == "failed" else None
    return {
        "stable_id": stable_id,
        "work_id": work_id,
        "status": {
            "kind": kind,
            "code": status_code,
            "stage": stage,
            "reason_code": reason_code,
        },
    }


def _witness_attempt(
    stable_id: str,
    stage: str,
    reason_code: str,
    *,
    work_id: str = "work-1",
    exception_type: str = "menagerie.crawler.driver_contracts.AuthorBlockedPrerequisite",
) -> dict[str, Any]:
    """Build one minimal witnessing failed-attempt row (pilot v3 shape)."""

    return {
        "attempt_id": f"attempt-{stable_id}-{stage}",
        "stable_id": stable_id,
        "work_id": work_id,
        "stage": stage,
        "result": "failed",
        "error": {
            "stage": stage,
            "reason_code": reason_code,
            "exception_type": exception_type,
        },
    }


# ---------------------------------------------------------------------------
# 1. Stage-suffixed reasons on every new-mint path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", _BLOCKED_ARM_STAGES)
def test_new_mints_carry_stage_suffixed_reason(stage: str) -> None:
    """An off-vocabulary reason with a neutral prerequisite keeps the stage,
    and the reason now NAMES that stage."""

    status_code, reason_code = _map(stage, "some-reason-the-record-cannot-express")
    assert status_code == f"failed:{stage}"
    assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:{stage}"
    assert reason_code in FAILURE_REASON_CODES[stage]


def test_exactly_recordable_reasons_still_map_verbatim() -> None:
    """The classification split changes nothing for already-recordable reasons."""

    assert _map("source", "missing-material-source") == (
        "failed:source",
        "missing-material-source",
    )
    assert _map("environment", "solve-failed") == ("failed:environment", "solve-failed")
    assert _map("author", "needs-higher-tier") == ("failed:author", "needs-higher-tier")


# ---------------------------------------------------------------------------
# 2. Routing split: runtime-dependency and CUDA prerequisite classes.
# ---------------------------------------------------------------------------


def test_m1777_runtime_dependency_routes_environment_adjacent() -> None:
    """The signed exemplar: an installable-runtime-dep BLOCKED is an ecosystem
    fact, not a model defect at the blocked stage."""

    status_code, reason_code = _map(
        "environment",
        "missing-runtime-dependency",
        ("runtime-dependency-minkowskiengine",),
    )
    assert status_code == "failed:environment-adjacent"
    assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:environment"


def test_runtime_dependency_class_preserves_the_blocking_stage_in_the_reason() -> None:
    """m4746 shape: a runtime-dep block at the AUTHOR stage keeps `author` in
    the reason suffix while the status names the adjacent-environment class."""

    status_code, reason_code = _map(
        "author",
        "missing-runtime-dependency",
        ("runtime-dependency", "espnet-distribution"),
    )
    assert status_code == "failed:environment-adjacent"
    assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:author"


@pytest.mark.parametrize(
    "prerequisite_ids",
    [
        ("runtime-dependency", "cuda-platform"),  # m4388
        (
            "runtime-dependency-knn-cuda",
            "runtime-dependency-pointnet2-ops",
            "runtime-dependency-chamfer-dist-cuda-extension",
            "runtime-dependency-matplotlib",
            "cuda-device",
        ),  # m8215
        (
            "runtime-dependency-torch-scatter",
            "runtime-dependency-plyfile",
            "runtime-dependency-simple-knn",
            "runtime-dependency-diff-gaussian-rasterization",
            "cuda-runtime",
        ),  # m9073
    ],
)
def test_cuda_class_routes_the_existing_needs_cuda_valve(
    prerequisite_ids: tuple[str, ...],
) -> None:
    """A CUDA-capability prerequisite outranks its runtime-dep siblings: the
    capability is the binding constraint, and the head is the EXISTING
    deferral, not a new failure spelling."""

    assert _map("environment", "missing-runtime-dependency", prerequisite_ids) == (
        "deferred:needs-cuda",
        None,
    )


def test_runtime_dep_package_name_containing_cuda_is_not_cuda_class() -> None:
    """`runtime-dependency-knn-cuda` names an installable package, not the
    CUDA capability; without a capability ID it stays runtime-dep class."""

    status_code, reason_code = _map(
        "environment",
        "missing-runtime-dependency",
        ("runtime-dependency-knn-cuda", "runtime-dependency-pointnet2-ops"),
    )
    assert status_code == "failed:environment-adjacent"
    assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:environment"


# ---------------------------------------------------------------------------
# 3. Vocabulary and schema lockstep for the new closed members.
# ---------------------------------------------------------------------------


def test_stage_suffixed_reason_vocabulary_is_closed_and_schema_representable() -> None:
    """Every blocked-arm stage owns exactly its own suffixed member, the
    environment-adjacent stage admits all seven, and the attempt schema can
    record each one."""

    reason_enum = set(_schema_defs("attempt-common.schema.json")["reason_code"]["enum"])
    for stage in _BLOCKED_ARM_STAGES:
        suffixed = f"{BLOCKED_PREREQUISITE_REASON_CODE}:{stage}"
        assert suffixed in FAILURE_REASON_CODES[stage], stage
        assert suffixed in FAILURE_REASON_CODES["environment-adjacent"], stage
        assert suffixed in reason_enum, stage
        # The legacy bare member stays read-valid exactly where it always was.
        assert BLOCKED_PREREQUISITE_REASON_CODE in FAILURE_REASON_CODES[stage], stage
        # And of no third stage: suffix injectivity is structural.
        for other, codes in FAILURE_REASON_CODES.items():
            if other not in (stage, "environment-adjacent"):
                assert suffixed not in codes, (stage, other)


def test_environment_adjacent_status_code_is_closed_everywhere() -> None:
    """The new status code and its stage are members of every closed surface a
    terminal head crosses: constants, model schema, attempt schema."""

    assert "failed:environment-adjacent" in TERMINAL_STATUS_CODES
    assert "environment-adjacent" in FAILURE_REASON_CODES
    assert FAILURE_REASON_CODES["environment-adjacent"] == frozenset(
        f"{BLOCKED_PREREQUISITE_REASON_CODE}:{stage}" for stage in _BLOCKED_ARM_STAGES
    )
    model_defs = _schema_defs("model-common.schema.json")
    assert "failed:environment-adjacent" in model_defs["status_code"]["enum"]
    assert "environment-adjacent" in model_defs["failure_stage"]["enum"]
    attempt_defs = _schema_defs("attempt-common.schema.json")
    assert "environment-adjacent" in attempt_defs["stage"]["enum"]


# ---------------------------------------------------------------------------
# 4. The terminal proof accepts the new head; the CUDA head reuses the
#    accepted-BLOCKED capability rule.
# ---------------------------------------------------------------------------


def test_terminal_proof_accepts_environment_adjacent_head() -> None:
    """A failed:environment-adjacent head is witnessed by an exact same-stage
    failed attempt carrying the suffixed reason."""

    from menagerie.crawler.authority import derive_terminal_proof
    from menagerie.crawler.identity import stable_hash

    attempt = {
        "attempt_id": "failed-environment-adjacent-1",
        "attempt_no": 1,
        "ledger_seq": 1,
        "stable_id": "m1777",
        "work_id": "work-m1777",
        "stage": "environment-adjacent",
        "mode": None,
        "result": "failed",
        "worker_receipt": {"present": False},
        "supervisor_observation": {"peak_rss_bytes": 7},
        "error": {
            "stage": "environment-adjacent",
            "reason_code": f"{BLOCKED_PREREQUISITE_REASON_CODE}:environment",
            "root_cause_fingerprint": stable_hash(["al6", "environment-adjacent"]),
        },
    }
    proof = derive_terminal_proof(
        "m1777",
        "work-m1777",
        "failed:environment-adjacent",
        attempts=[attempt],
        proof_rule_identity=HASH,
    )
    assert proof.failure_stage == "environment-adjacent"
    assert proof.reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:environment"
    assert proof.decisive_attempt_ids == ("failed-environment-adjacent-1",)


def test_blocked_capability_rule_proves_the_needs_cuda_head() -> None:
    """The BLOCKED-proved deferred:needs-cuda head settles through the SAME
    accepted-BLOCKED capability rule as the opus-tier/source-access pair --
    no handoff, no platform probe, only the checker's accepted gate."""

    from menagerie.crawler.reducer import _derive_blocked_capability_deferral_proof

    hash_a = "sha256:" + "a" * 64
    hash_b = "sha256:" + "b" * 64
    hash_c = "sha256:" + "c" * 64
    gate = {
        "gate_id": "gate-blocked-prerequisite",
        "gate_kind": "terminal_disposition",
        "gate_round": 1,
        "ledger_seq": 1,
        "items": [
            {
                "stable_id": "m4388",
                "work_id": "work-m4388",
                "verdict": "accurate",
                "integrity": {"verdict": "accurate"},
                "terminal_disposition": {
                    "kind": "BLOCKED",
                    "predicate": "blocked-prerequisite",
                    "handoff_proposal_id": None,
                    "handoff_sha256": None,
                    "verdict": "accepted",
                    "source_manifest_identity": hash_a,
                    "source_ids": ["source-1"],
                    "evidence_identity": hash_b,
                    "evidence_ids": ["evidence-1"],
                    "license_identity": hash_c,
                },
            }
        ],
    }
    proof = _derive_blocked_capability_deferral_proof(
        status_code="deferred:needs-cuda",
        stable_id="m4388",
        work_id="work-m4388",
        attempts=(),
        gates=(gate,),
        source_manifest=({"source_id": "source-1"},),
        evidence_excerpts=({"evidence_id": "evidence-1", "source_id": "source-1"},),
        source_manifest_identity=hash_a,
        evidence_identity=hash_b,
        license_identity=hash_c,
        meaningful_modes=("train", "eval"),
        proof_rule_identity=HASH,
    )
    assert proof.status_code == "deferred:needs-cuda"
    assert proof.gate_id == "gate-blocked-prerequisite"


# ---------------------------------------------------------------------------
# 5. Money: the author-blocked-prerequisite attempt refund.
# ---------------------------------------------------------------------------


def test_attempt_refund_vocabulary_gains_the_author_blocked_member() -> None:
    """The refund member is closed, derived nowhere from text, and segregated
    from the machine-infrastructure subset the storm surfaces consume."""

    from menagerie.crawler.paid_session_accounting import (
        ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES,
        ATTEMPT_REFUND_LOCAL_OUTCOMES,
        AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME,
        LOCAL_OUTCOMES,
    )
    from menagerie.crawler.storm_fuse import CROSS_MODEL_STORM_CLASSIFICATIONS

    assert AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME == "author-blocked-prerequisite"
    assert AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME in ATTEMPT_REFUND_LOCAL_OUTCOMES
    assert AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME in LOCAL_OUTCOMES
    # An author VERDICT is never machine weather: the infrastructure subset
    # excludes it, and the storm fuse's own closed set never counts it.
    assert (
        AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME
        not in ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES
    )
    assert ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES < ATTEMPT_REFUND_LOCAL_OUTCOMES
    assert AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME not in CROSS_MODEL_STORM_CLASSIFICATIONS


def test_refund_zeroes_the_attempt_projection_and_keeps_the_session_charged() -> None:
    """settled_attempt_weight == 0 (attempt refunds), settled_session_weight ==
    1 (sessions/wall stay charged), on a schema-valid settlement row."""

    from menagerie.crawler.paid_session_accounting import (
        AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME,
        PAID_SESSION_SETTLEMENT_SCHEMA,
        settled_attempt_weight,
        settled_session_weight,
        validate_settlement,
    )

    row = {
        "schema_version": PAID_SESSION_SETTLEMENT_SCHEMA,
        "event": "session_settled",
        "spend_id": "spend-1",
        "observed": {"wall_seconds": 5.0},
        "evidence": "provider-query",
        "custody_outcome": "exited",
        "local_outcome": AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME,
        "provider": {
            "accepted": True,
            "attested_wall_seconds": 5.0,
            "receipt_sha256": "sha256:" + "d" * 64,
            "adapter_identity": "provider-reconcile-evidence/2",
        },
        "machine": {"wall_seconds": 5.0, "source": "writer-monotonic"},
    }
    validate_settlement(row)
    assert settled_session_weight(row) == 1
    assert settled_attempt_weight(row) == 0


def test_refund_never_reads_as_provider_storm_exhaustion_evidence() -> None:
    """A generation whose consumed sessions are majority blocked-refunds still
    exhausts as plain author-sessions: an author verdict is not weather."""

    from menagerie.crawler.author_spend import author_sessions_exhaustion_dimension
    from menagerie.crawler.paid_session_accounting import (
        AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME,
        PAID_SESSION_SETTLEMENT_SCHEMA,
    )

    def _reserved(spend_id: str) -> dict[str, Any]:
        return {
            "event": "session_reserved",
            "lane": "author",
            "spend_id": spend_id,
            "axis": {"work_generations": [{"stable_id": "m1777", "work_id": "work-m1777"}]},
        }

    def _settled(spend_id: str, local_outcome: str) -> dict[str, Any]:
        return {
            "schema_version": PAID_SESSION_SETTLEMENT_SCHEMA,
            "event": "session_settled",
            "spend_id": spend_id,
            "observed": {"wall_seconds": 5.0},
            "evidence": "provider-query",
            "custody_outcome": "exited",
            "local_outcome": local_outcome,
            "provider": {
                "accepted": True,
                "attested_wall_seconds": 5.0,
                "receipt_sha256": "sha256:" + "d" * 64,
                "adapter_identity": "provider-reconcile-evidence/2",
            },
            "machine": {"wall_seconds": 5.0, "source": "writer-monotonic"},
        }

    blocked_majority = [
        _reserved("s1"),
        _reserved("s2"),
        _reserved("s3"),
        _settled("s1", AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME),
        _settled("s2", AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME),
        _settled("s3", "completed"),
    ]
    assert (
        author_sessions_exhaustion_dimension(blocked_majority, "m1777", "work-m1777")
        == "author-sessions"
    )
    # Control: a genuine machine-infrastructure majority still reads storm.
    infra_majority = [
        _reserved("s1"),
        _reserved("s2"),
        _reserved("s3"),
        _settled("s1", "failed-harness-api-error"),
        _settled("s2", "failed-machine-spawn"),
        _settled("s3", "completed"),
    ]
    assert (
        author_sessions_exhaustion_dimension(infra_majority, "m1777", "work-m1777")
        == "provider-storm"
    )


def test_author_settle_site_stamps_only_the_typed_blocked_arm() -> None:
    """The stamp keys on the schema-validated recommendation KIND -- the typed
    result class -- never on any text, and only for the author lane."""

    from menagerie.crawler.driver_admission import (
        _author_blocked_prerequisite_settlement_outcome,
    )
    from menagerie.crawler.paid_session_accounting import (
        AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME,
    )

    blocked = _blocked("environment", "missing-runtime-dependency")
    artifact = SimpleNamespace(author_result=blocked)
    item = SimpleNamespace(stable_id="m1777")
    assert (
        _author_blocked_prerequisite_settlement_outcome(("author", item), artifact)
        == AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME
    )
    # Non-BLOCKED arms, non-author lanes, and non-artifact results never stamp.
    assert (
        _author_blocked_prerequisite_settlement_outcome(
            ("author", item), SimpleNamespace(author_result=SimpleNamespace())
        )
        is None
    )
    assert _author_blocked_prerequisite_settlement_outcome(("checker", item), artifact) is None
    assert _author_blocked_prerequisite_settlement_outcome(None, artifact) is None
    assert _author_blocked_prerequisite_settlement_outcome(("author", item), "text") is None


# ---------------------------------------------------------------------------
# 6. Censuses: injectivity and the leg-close no-generic-failed:author tripwire.
# ---------------------------------------------------------------------------


def test_injectivity_census_recovers_the_blocking_stage_or_flags() -> None:
    from menagerie.crawler.status import blocked_prerequisite_injectivity_census

    agreeing = [
        _model_row("m1", "failed:environment", f"{BLOCKED_PREREQUISITE_REASON_CODE}:environment"),
        _model_row("m2", "failed:author", f"{BLOCKED_PREREQUISITE_REASON_CODE}:author"),
        _model_row(
            "m3",
            "failed:environment-adjacent",
            f"{BLOCKED_PREREQUISITE_REASON_CODE}:author",
        ),
        _model_row("m4", "failed:source", "missing-material-source"),
    ]
    census = blocked_prerequisite_injectivity_census(agreeing)
    assert census["flagged"] == ()
    assert census["legacy"] == ()

    disagreeing = agreeing + [
        _model_row("m5", "failed:source", f"{BLOCKED_PREREQUISITE_REASON_CODE}:environment"),
        _model_row(
            "m6",
            "failed:environment-adjacent",
            f"{BLOCKED_PREREQUISITE_REASON_CODE}:not-a-stage",
        ),
        _model_row("m7", "failed:environment", BLOCKED_PREREQUISITE_REASON_CODE),
    ]
    census = blocked_prerequisite_injectivity_census(disagreeing)
    assert census["flagged"] == ("m5", "m6")
    # Bare-member rows are the historical era: disclosed, never flagged.
    assert census["legacy"] == ("m7",)


def test_leg_close_census_flags_generic_failed_author_with_blocked_witness() -> None:
    from menagerie.crawler.status import author_blocked_prerequisite_terminal_census

    models = [
        # The historical mislabel shape: generic collapse, witnessed cause.
        _model_row("m4413", "failed:author", "malformed-result"),
        # Honest same-stage blocked terminal at the author stage: not generic.
        _model_row("m2", "failed:author", f"{BLOCKED_PREREQUISITE_REASON_CODE}:author"),
        # Top-tier escalation: its witness legitimately carries the cause.
        _model_row("m3", "failed:author", "needs-higher-tier"),
        # Generic reason but NO blocked witness: outside this census.
        _model_row("m4", "failed:author", "malformed-result"),
    ]
    attempts = [
        _witness_attempt("m4413", "author", "malformed-result"),
        _witness_attempt("m2", "author", f"{BLOCKED_PREREQUISITE_REASON_CODE}:author"),
        _witness_attempt("m3", "author", "needs-higher-tier"),
        _witness_attempt(
            "m4", "author", "malformed-result", exception_type="SomeOtherError"
        ),
    ]
    census = author_blocked_prerequisite_terminal_census(models, attempts)
    assert census["flagged"] == ("m4413",)
    assert census["escalation"] == ("m3",)
    assert census["stage_bound"] == ("m2",)


def test_leg_close_census_is_empty_over_new_mint_routing() -> None:
    """Everything the new total mapping can mint from a checked BLOCKED verdict
    produces ZERO generic failed:author heads: the tripwire's steady state."""

    from menagerie.crawler.status import author_blocked_prerequisite_terminal_census

    models = []
    attempts = []
    for shape in _PILOT_SHAPES:
        status_code, reason_code = _map(
            shape["blocked_stage"],
            shape["author_reason"],
            tuple(shape["prerequisite_ids"]),
        )
        models.append(_model_row(shape["stable_id"], status_code, reason_code))
        if status_code.startswith("failed:"):
            attempts.append(
                _witness_attempt(
                    shape["stable_id"],
                    status_code.removeprefix("failed:"),
                    str(reason_code),
                )
            )
    census = author_blocked_prerequisite_terminal_census(models, attempts)
    assert census["flagged"] == ()


# ---------------------------------------------------------------------------
# 7. The 20 reclassified pilot members replay through the new routing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", _PILOT_SHAPES, ids=[shape["stable_id"] for shape in _PILOT_SHAPES]
)
def test_pilot_member_shapes_route_honestly(shape: dict[str, Any]) -> None:
    """Each real reclassified member lands on its honest terminal: CUDA-class
    members defer on the capability, every other member records the ecosystem
    fact with its blocking stage recoverable from the reason suffix."""

    prerequisite_ids = tuple(shape["prerequisite_ids"])
    status_code, reason_code = _map(
        shape["blocked_stage"], shape["author_reason"], prerequisite_ids
    )
    if set(prerequisite_ids) & _PILOT_CUDA_IDS:
        assert (status_code, reason_code) == ("deferred:needs-cuda", None)
    elif not any(
        value.startswith("runtime-dependency-")
        or value.endswith("-distribution")
        or value.endswith("-package")
        for value in prerequisite_ids
    ):
        # EN-6 x AL-6 batch reconciliation: m1471/m4954 carry only
        # CONTENT-FREE or bare-name spellings the closed grammar refuses
        # (grammar-refused dominance -- a nameless claim can never steer into
        # the grantable environment-adjacent class; W14's seam translation
        # fails the same ids closed). They keep the honest blocked arm with
        # the stage-suffixed reason.
        assert status_code == f"failed:{shape['blocked_stage']}"
        assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:{shape['blocked_stage']}"
    else:
        assert status_code == "failed:environment-adjacent"
        assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:{shape['blocked_stage']}"
        assert reason_code in FAILURE_REASON_CODES["environment-adjacent"]


def test_pilot_fixture_set_matches_the_signed_bar() -> None:
    """Exactly the 20 reclassified members, m1777 included, none invented."""

    ids = [shape["stable_id"] for shape in _PILOT_SHAPES]
    assert len(ids) == len(set(ids)) == 20
    assert "m1777" in ids


def test_injectivity_census_over_pilot_derived_ledger() -> None:
    """The full pilot-derived fixture ledger yields zero injectivity flags."""

    from menagerie.crawler.status import blocked_prerequisite_injectivity_census

    models = []
    for shape in _PILOT_SHAPES:
        status_code, reason_code = _map(
            shape["blocked_stage"],
            shape["author_reason"],
            tuple(shape["prerequisite_ids"]),
        )
        models.append(_model_row(shape["stable_id"], status_code, reason_code))
    census = blocked_prerequisite_injectivity_census(models)
    assert census["flagged"] == ()
    assert census["legacy"] == ()
