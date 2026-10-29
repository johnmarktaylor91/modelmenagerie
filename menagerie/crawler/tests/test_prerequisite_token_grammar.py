"""EN-6: closed prerequisite-token grammar, known-19 aggregation, and the deferral crossing.

The pilot's 19 ``failed:environment`` / ``blocked-prerequisite`` terminals carried
their author-named prerequisites only as free text -- five spellings across 19
models, including ``m4954``'s content-free bare ``runtime-dependency`` token that
names nothing at all. This suite pins the EN-6 contract:

* the CANONICAL forward grammar is ``runtime-dependency:<pep503>``; a bare
  ``runtime-dependency`` marker is CONTENT-FREE and can never steer a disposition
  (the m4954 RED arm);
* every pilot spelling classifies deterministically (legacy hyphen form,
  ``-distribution``/``-package`` suffix forms, bare distribution names);
* the platform class is CLOSED (triton / flash-attn / spconv / knn-cuda /
  cuda-runtime); an all-platform prerequisite set aggregates to the
  needs-cuda-shaped deferral and NEVER to ``failed:environment``;
* the committed known-19 corpus (``pilot_known19_prerequisites.jsonl``, verbatim
  from the pilot attempt diagnostics) replays through the classifier against the
  REAL committed environment registry with pinned golden dispositions;
* ``driver._blocked_terminal`` routes an all-platform environment-stage BLOCKED
  verdict onto ``deferred:needs-cuda-prerequisites`` (a new closed member of
  ``CAPABILITY_DEFERRAL_STATUS_CODES``, proved by the existing accepted-BLOCKED
  capability rule), and leaves every other arm exactly as it was.

Census cross-check (the standing tripwire): zero ``failed:environment`` rows whose
prerequisites are ALL platform-class, over any records set.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from menagerie.crawler.author_dispatch import AuthorResultBinding, BlockedRecommendation
from menagerie.crawler.constants import (
    BLOCKED_PREREQUISITE_REASON_CODE,
    CAPABILITY_DEFERRAL_STATUS_CODES,
    PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE,
    TERMINAL_STATUS_CODES,
)
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.environment_coverage import (
    PLATFORM_CLASS_DISTRIBUTIONS,
    PrerequisiteAggregateDisposition,
    PrerequisiteTokenKind,
    aggregate_prerequisite_tokens,
    canonical_prerequisite_token,
    census_platform_class_environment_failures,
    parse_prerequisite_token,
)
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, load_environment_registry
from menagerie.crawler.tests.conftest import HASH

pytestmark = pytest.mark.smoke

_CORPUS_PATH = Path(__file__).resolve().parent / "pilot_known19_prerequisites.jsonl"

#: The four pilot models whose prerequisite lists carry a content-free bare
#: ``runtime-dependency`` marker token (m4954's list is ONLY that token).
_GRAMMAR_REFUSED_MODELS = frozenset({"m1471", "m4897", "m4954", "m8131"})


def _corpus() -> tuple[dict, ...]:
    """Load the committed known-19 corpus fixture."""

    rows = tuple(
        json.loads(line)
        for line in _CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    assert len(rows) == 19
    return rows


# ---------------------------------------------------------------------------
# Token grammar
# ---------------------------------------------------------------------------


def test_canonical_colon_form_parses_and_round_trips() -> None:
    """``runtime-dependency:<pep503>`` is the canonical forward grammar."""

    token = parse_prerequisite_token("runtime-dependency:torch-cluster")
    assert token.kind is PrerequisiteTokenKind.RUNTIME_DEPENDENCY
    assert token.distribution == "torch-cluster"
    assert canonical_prerequisite_token("torch-cluster") == "runtime-dependency:torch-cluster"
    # The renderer normalizes to PEP 503 before rendering.
    assert canonical_prerequisite_token("Torch_Cluster") == "runtime-dependency:torch-cluster"


def test_bare_marker_is_content_free_the_m4954_arm() -> None:
    """m4954's exact token names nothing and must never steer a disposition."""

    for raw in ("runtime-dependency", "runtime-dependency:", "runtime-dependency-", "  ", ""):
        token = parse_prerequisite_token(raw)
        assert token.kind is PrerequisiteTokenKind.CONTENT_FREE, raw
        assert token.distribution is None
        assert not token.platform_class


def test_legacy_pilot_spellings_classify_deterministically() -> None:
    """Every spelling observed in the pilot corpus extracts its distribution."""

    hyphen = parse_prerequisite_token("runtime-dependency-torch-scatter")
    assert hyphen.kind is PrerequisiteTokenKind.LEGACY_HYPHEN
    assert hyphen.distribution == "torch-scatter"

    suffixed = parse_prerequisite_token("spconv-distribution")
    assert suffixed.kind is PrerequisiteTokenKind.LEGACY_SUFFIXED
    assert suffixed.distribution == "spconv"
    assert suffixed.platform_class

    package = parse_prerequisite_token("pcdet-package")
    assert package.kind is PrerequisiteTokenKind.LEGACY_SUFFIXED
    assert package.distribution == "pcdet"

    bare = parse_prerequisite_token("cuda-runtime")
    assert bare.kind is PrerequisiteTokenKind.BARE_DISTRIBUTION
    assert bare.distribution == "cuda-runtime"
    assert bare.platform_class


def test_platform_class_set_is_exactly_the_closed_five() -> None:
    """The platform class is a closed vocabulary; extending it is a reviewed change."""

    assert PLATFORM_CLASS_DISTRIBUTIONS == frozenset(
        {"triton", "flash-attn", "spconv", "knn-cuda", "cuda-runtime"}
    )
    for name in sorted(PLATFORM_CLASS_DISTRIBUTIONS):
        assert parse_prerequisite_token(f"runtime-dependency:{name}").platform_class


# ---------------------------------------------------------------------------
# Aggregation ladder
# ---------------------------------------------------------------------------


def test_all_platform_set_aggregates_to_platform_deferral() -> None:
    """A prerequisite set naming only platform-class builds is CUDA-bound."""

    aggregation = aggregate_prerequisite_tokens(
        ("runtime-dependency:triton", "runtime-dependency-flash-attn", "cuda-runtime")
    )
    assert aggregation.disposition is PrerequisiteAggregateDisposition.PLATFORM_DEFERRAL
    assert set(aggregation.platform) == {"triton", "flash-attn", "cuda-runtime"}
    assert not aggregation.unrecoverable


def test_content_free_token_refuses_aggregation_even_beside_platform_tokens() -> None:
    """A content-free marker poisons the set: never a deferral from nothing."""

    aggregation = aggregate_prerequisite_tokens(("runtime-dependency", "runtime-dependency:triton"))
    assert aggregation.disposition is PrerequisiteAggregateDisposition.GRAMMAR_REFUSED


def test_empty_token_set_never_defers() -> None:
    """Zero prerequisites assert nothing; the blocked arm keeps them."""

    aggregation = aggregate_prerequisite_tokens(())
    assert aggregation.disposition is PrerequisiteAggregateDisposition.BLOCKED_PREREQUISITE


def test_served_tokens_are_discounted_before_the_platform_decision() -> None:
    """A distribution the routed inventory provably installs is not a blocker."""

    routed_packages = ({"name": "transformers", "version": "4.44.0"},)
    aggregation = aggregate_prerequisite_tokens(
        ("runtime-dependency:transformers", "runtime-dependency:triton"),
        routed_packages=routed_packages,
    )
    assert aggregation.served == ("transformers",)
    assert aggregation.disposition is PrerequisiteAggregateDisposition.PLATFORM_DEFERRAL


def test_mixed_platform_and_unrecoverable_stays_blocked() -> None:
    """A CUDA host alone does not recover a set with a non-platform member."""

    aggregation = aggregate_prerequisite_tokens(
        ("runtime-dependency:triton", "runtime-dependency:performer-pytorch")
    )
    assert aggregation.disposition is PrerequisiteAggregateDisposition.BLOCKED_PREREQUISITE
    assert aggregation.unrecoverable == ("performer-pytorch",)


# ---------------------------------------------------------------------------
# Replay-all-19: the committed corpus against the real committed registry
# ---------------------------------------------------------------------------


def test_known19_corpus_replays_with_pinned_dispositions() -> None:
    """Every pilot diagnostic classifies deterministically; goldens are pinned."""

    registry = load_environment_registry(DEFAULT_ENVS_ROOT, target="osx-arm64")
    by_id = {}
    for row in _corpus():
        intent = registry.intents.get(row["routed_intent"])
        packages: tuple = ()
        if intent is not None:
            from menagerie.crawler.driver_admission import _routed_environment_packages

            packages = _routed_environment_packages(registry, row["routed_intent"])
        aggregation = aggregate_prerequisite_tokens(
            tuple(row["prerequisite_ids"]),
            routed_intent=row["routed_intent"],
            routed_packages=packages,
            registry=registry,
        )
        by_id[row["stable_id"]] = aggregation

    # The four content-free rows refuse aggregation -- m4954 is the RED arm.
    for stable_id in sorted(_GRAMMAR_REFUSED_MODELS):
        assert (
            by_id[stable_id].disposition is PrerequisiteAggregateDisposition.GRAMMAR_REFUSED
        ), stable_id
    # No pilot row is all-platform: every remaining row stays on the blocked arm.
    for stable_id, aggregation in by_id.items():
        if stable_id in _GRAMMAR_REFUSED_MODELS:
            continue
        assert (
            aggregation.disposition is PrerequisiteAggregateDisposition.BLOCKED_PREREQUISITE
        ), stable_id
    # transformers rows routed to oddballs name a covering intent, and core is
    # among the locked-inventory intents that prove it (audio also carries
    # transformers with the same basis; the recorded pair is strongest-first).
    covered_m1035 = dict(by_id["m1035"].covered)
    assert covered_m1035.get("transformers") in {"audio", "core"}
    from menagerie.crawler.environment_coverage import find_covering_intents

    covering_names = {
        item.intent for item in find_covering_intents("transformers", registry, exclude="oddballs")
    }
    assert "core" in covering_names
    # torch-geometric names graph (pyg's locked inventory provides it).
    covered_m1239 = dict(by_id["m1239"].covered)
    assert covered_m1239.get("torch-geometric") == "graph"
    # CUDA-bound tokens classify platform wherever they appear.
    assert "knn-cuda" in by_id["m2221"].platform
    assert "cuda-runtime" in by_id["m9073"].platform
    assert "triton" in by_id["m4905"].platform


def test_census_zero_all_platform_failed_environment_rows() -> None:
    """The standing cross-check: no failed:environment row is all-platform."""

    rows = tuple(
        (row["terminal_code"], tuple(row["prerequisite_ids"])) for row in _corpus()
    )
    assert census_platform_class_environment_failures(rows) == ()
    # A synthetic violation is detected: the census is a tripwire, not a formality.
    violation = (("failed:environment", ("runtime-dependency:triton", "cuda-runtime")),)
    assert census_platform_class_environment_failures(violation) == (0,)


# ---------------------------------------------------------------------------
# The terminal crossing (driver._blocked_terminal)
# ---------------------------------------------------------------------------


def _blocked(stage: str, reason_code: str, prerequisites: tuple[str, ...]) -> BlockedRecommendation:
    """Build one minimal checked BLOCKED recommendation for the crossing."""

    raw_fields = {
        "result_id": "result-blocked",
        "result_sha256": HASH,
        "stable_id": "m-en6",
        "work_id": "work-en6",
        "campaign_id": "work-en6",
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": HASH,
        "intake_snapshot_id": "intake-1",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": HASH,
        "created_at": "2026-08-24T00:00:00Z",
    }
    binding = AuthorResultBinding(
        raw_result={**raw_fields, "kind": "BLOCKED", "payload": {}}, **raw_fields
    )
    return BlockedRecommendation(
        binding=binding,
        stage=stage,
        reason_code=reason_code,
        prerequisite_ids=prerequisites,
        evidence_ids=("ev-blocked-1",),
        evidence_identity=HASH,
        license_identity=HASH,
        recommendation_sha256=HASH,
    )


def _map(
    stage: str, reason_code: str, prerequisites: tuple[str, ...]
) -> tuple[str, str | None]:
    """Run the REAL total mapping with the default campaign config."""

    host = cast(CrawlerDriver, SimpleNamespace(config=DriverConfig()))
    return CrawlerDriver._blocked_terminal(host, _blocked(stage, reason_code, prerequisites))


def test_all_platform_environment_blocked_mints_the_deferral() -> None:
    """An all-platform environment-stage BLOCKED verdict is CUDA-bound, not failed."""

    status_code, reason_code = _map(
        "environment",
        "missing-runtime-dependency",
        ("runtime-dependency:triton", "runtime-dependency:flash-attn"),
    )
    assert status_code == PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE
    assert reason_code is None


def test_content_free_token_never_reaches_the_deferral() -> None:
    """m4954's shape keeps its honest failed:environment terminal."""

    status_code, reason_code = _map(
        "environment", "missing-runtime-dependency", ("runtime-dependency",)
    )
    # AL-6 (batch): new mints spell the stage-suffixed reason; the bare
    # member stays read-side legacy. The content-free token still never
    # steers the STATUS off the honest blocked arm (EN-6 grammar law).
    assert status_code == "failed:environment"
    assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:environment"


def test_mixed_set_keeps_the_blocked_arm() -> None:
    """The crossing only defers what a CUDA host alone provably recovers."""

    status_code, reason_code = _map(
        "environment",
        "missing-runtime-dependency",
        ("runtime-dependency:triton", "runtime-dependency:performer-pytorch"),
    )
    assert status_code == "failed:environment"
    # AL-6 (batch): stage-suffixed mint; bare member is read-side legacy.
    assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:environment"


def test_non_environment_stages_are_untouched_by_the_deferral() -> None:
    """Only the environment stage can assert an installable-prerequisite claim."""

    for stage in ("source", "fetch", "author", "evidence", "policy", "runner"):
        status_code, reason_code = _map(
            stage, "missing-runtime-dependency", ("runtime-dependency:triton",)
        )
        assert status_code == f"failed:{stage}", stage
        # AL-6 (batch): stage-suffixed mint; bare member is read-side legacy.
        assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:{stage}"


def test_in_vocabulary_environment_reasons_still_defer_when_all_platform() -> None:
    """The decision keys on the PREREQUISITES, never on the reason spelling."""

    status_code, reason_code = _map(
        "environment", "blocked-prerequisite", ("runtime-dependency:spconv",)
    )
    assert status_code == PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE
    assert reason_code is None


# ---------------------------------------------------------------------------
# Vocabulary membership (the closed-set extension is total)
# ---------------------------------------------------------------------------


def test_new_deferral_is_a_closed_vocabulary_member_everywhere() -> None:
    """The needs-cuda-shaped code joins every consumer of the closed sets."""

    assert PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE == "deferred:needs-cuda-prerequisites"
    assert PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE in CAPABILITY_DEFERRAL_STATUS_CODES
    assert PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE in TERMINAL_STATUS_CODES
    # Kind splits cleanly for the status displays.
    assert PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE.split(":", 1)[0] == "deferred"
    # The cross-campaign worklist assembles it for the host that has CUDA.
    from menagerie.crawler.campaign_merge import HOST_ROUTED_DEFERRED_STATUS_CODES

    assert PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE in HOST_ROUTED_DEFERRED_STATUS_CODES
    # The model schema can spell it.
    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas" / "model-common.schema.json"
    )
    enum = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["status_code"]["enum"]
    assert PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE in enum
