"""AB-30 matrix: env-prerequisite-token-grammar seam (EN-6, lane w2-environments).

Readers x eras over the seam's committed prior-era artifact. Per the declaration
registry note, "the replay-all-19 classifier fixture IS this seam's matrix": the
19 pilot ``failed:environment`` / ``blocked-prerequisite`` terminals with their
VERBATIM author-named tokens live in ``pilot_known19_prerequisites.jsonl``, and
every declared reader of the closed grammar
(``menagerie.crawler.environment-prerequisite-token-grammar.v1``) must replay
those prior-era rows deterministically -- parsed with era-scoped semantics or
refused typed, never an unclassified crash.

This module frames the reader x era MATRIX over the committed corpus; the
lane's own unit suite (``test_prerequisite_token_grammar.py``) owns the
per-spelling and per-ladder-rung unit assertions and is not duplicated here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.environment_coverage import (
    PLATFORM_CLASS_DISTRIBUTIONS,
    PrerequisiteAggregateDisposition,
    PrerequisiteTokenKind,
    aggregate_prerequisite_tokens,
    parse_prerequisite_token,
)
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, load_environment_registry
from menagerie.crawler.package_namespace import canonical_distribution_name
from menagerie.crawler.tests.mixed_era import classify_reader

pytestmark = pytest.mark.smoke

_CORPUS_PATH = Path(__file__).resolve().parent / "pilot_known19_prerequisites.jsonl"

#: Closed per-row key vocabulary of the committed prior-era corpus.
_CORPUS_ROW_KEYS = frozenset(
    {
        "attempt_id",
        "message",
        "prerequisite_ids",
        "routed_intent",
        "stable_id",
        "terminal_code",
        "terminal_stage",
    }
)

#: The exact prior-era census: the four pilot rows carrying a content-free bare
#: ``runtime-dependency`` marker refuse the grammar; every other row stays on
#: the blocked arm; NO pilot row is all-platform.
_GRAMMAR_REFUSED_IDS = frozenset({"m1471", "m4897", "m4954", "m8131"})
_EXPECTED_GRAMMAR_REFUSED = 4
_EXPECTED_BLOCKED_PREREQUISITE = 15
_EXPECTED_PLATFORM_DEFERRALS = 0

#: Foreign spellings OUTSIDE the five pilot forms (canonical colon, legacy
#: hyphen, -distribution/-package suffix, bare name, content-free marker).
#: The grammar is TOTAL: every one of these must classify, never crash.
_FOREIGN_SPELLINGS = (
    "runtime-dependency::torch",  # doubled separator
    "Runtime_Dependency:Torch_Cluster",  # foreign casing/underscores on the marker
    "runtime-dependency:torch cluster",  # embedded whitespace in the name
    "torch>=2.1",  # a requirement specifier, not a name
    "package==1.2.3",  # pinned-spec spelling
    "runtime-dependency:\x00weird",  # control byte in the name
    "需要-torch",  # non-ASCII prefix
    "-distribution",  # suffix with no name
    "   runtime-dependency:triton   ",  # padded canonical form
    "a" * 512,  # absurd length
    "runtime-dependency-",  # dangling legacy hyphen marker
    "!!!",  # pure punctuation
)


def _corpus() -> tuple[dict[str, Any], ...]:
    """Load the committed prior-era corpus verbatim."""

    return tuple(
        json.loads(line)
        for line in _CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _routed_packages(registry: Any, intent: str) -> tuple[Mapping[str, Any], ...]:
    """Resolve one routed intent's exact resolved-export inventory rows."""

    from menagerie.crawler.driver_admission import _routed_environment_packages

    return _routed_environment_packages(registry, intent)


def _replay_all_19(registry: Any) -> dict[str, Any]:
    """Replay every committed corpus row through the REAL classifier + ladder."""

    by_id: dict[str, Any] = {}
    for row in _corpus():
        by_id[row["stable_id"]] = aggregate_prerequisite_tokens(
            tuple(row["prerequisite_ids"]),
            routed_intent=row["routed_intent"],
            routed_packages=_routed_packages(registry, row["routed_intent"]),
            registry=registry,
        )
    return by_id


# ---------------------------------------------------------------------------
# The prior-era artifact itself: the corpus shape is pinned
# ---------------------------------------------------------------------------


def test_committed_corpus_is_the_frozen_prior_era_matrix() -> None:
    """The 19-row pilot corpus is byte-committed with a closed row shape.

    The seam declares no separate frozen fixture because this corpus IS the
    matrix; a drifted row count, key set, or terminal head would silently
    re-scope every cell below, so the artifact shape is adjudicated first.
    """

    rows = _corpus()
    assert len(rows) == 19
    stable_ids = [row["stable_id"] for row in rows]
    assert len(set(stable_ids)) == 19, "corpus stable_ids are not unique"
    assert _GRAMMAR_REFUSED_IDS <= set(stable_ids)
    for row in rows:
        assert frozenset(row) == _CORPUS_ROW_KEYS, row["stable_id"]
        assert row["terminal_code"] == "failed:environment", row["stable_id"]
        assert row["terminal_stage"] == "environment", row["stable_id"]
        prerequisites = row["prerequisite_ids"]
        assert isinstance(prerequisites, list) and prerequisites, row["stable_id"]
        assert all(isinstance(token, str) and token for token in prerequisites), row[
            "stable_id"
        ]


# ---------------------------------------------------------------------------
# Cell 1: replay ALL 19 rows; the era census is EXACT
# ---------------------------------------------------------------------------


def test_replay_all_19_era_census_is_exact() -> None:
    """Every prior-era row lands classified and the census matches exactly.

    4 grammar-refused (m1471/m4897/m4954/m8131), 15 blocked-prerequisite,
    0 platform-deferrals -- replayed through the real committed registry and
    each row's real routed-intent inventory, exactly as the driver aggregates.
    """

    registry_outcome = classify_reader(
        lambda: load_environment_registry(DEFAULT_ENVS_ROOT, target="osx-arm64")
    )
    assert registry_outcome[0] == "parsed", (
        f"the committed environment registry refused to load: {registry_outcome[1]!r}"
    )
    registry = registry_outcome[1]

    outcome = classify_reader(lambda: _replay_all_19(registry))
    # A grammar-refused DISPOSITION is a parsed, era-scoped verdict; a typed
    # refusal (let alone an unclassified crash) on the committed corpus would
    # mean the batch reader dropped the prior era.
    assert outcome[0] == "parsed", (
        f"replaying the committed prior-era corpus refused: {outcome[1]!r}"
    )
    by_id = outcome[1]
    assert len(by_id) == 19

    census: dict[PrerequisiteAggregateDisposition, set[str]] = {}
    for stable_id, aggregation in by_id.items():
        # Totality: every row lands on a member of the CLOSED disposition enum.
        assert isinstance(
            aggregation.disposition, PrerequisiteAggregateDisposition
        ), stable_id
        census.setdefault(aggregation.disposition, set()).add(stable_id)

    refused = census.get(PrerequisiteAggregateDisposition.GRAMMAR_REFUSED, set())
    blocked = census.get(PrerequisiteAggregateDisposition.BLOCKED_PREREQUISITE, set())
    deferred = census.get(PrerequisiteAggregateDisposition.PLATFORM_DEFERRAL, set())
    assert refused == _GRAMMAR_REFUSED_IDS, sorted(refused)
    assert len(refused) == _EXPECTED_GRAMMAR_REFUSED
    assert len(blocked) == _EXPECTED_BLOCKED_PREREQUISITE, sorted(blocked)
    assert len(deferred) == _EXPECTED_PLATFORM_DEFERRALS, sorted(deferred)
    assert refused | blocked | deferred == set(by_id)


# ---------------------------------------------------------------------------
# Cell 2: m4954's content-free token can NEVER steer a disposition
# ---------------------------------------------------------------------------


def test_m4954_content_free_token_never_steers_any_corpus_row() -> None:
    """Grammar-refused dominance holds over the whole committed matrix.

    m4954's verbatim bare marker dominates the ladder: injected into EVERY
    other corpus row's verbatim token set, the aggregate is grammar-refused
    with EMPTY evidence fields -- the content-free token neither earns a
    deferral, discounts a served token, nor contributes platform evidence.
    """

    rows = {row["stable_id"]: row for row in _corpus()}
    m4954_tokens = tuple(rows["m4954"]["prerequisite_ids"])
    assert m4954_tokens == ("runtime-dependency",)
    marker = m4954_tokens[0]
    token = parse_prerequisite_token(marker)
    assert token.kind is PrerequisiteTokenKind.CONTENT_FREE
    assert token.distribution is None and not token.platform_class

    for stable_id, row in rows.items():
        poisoned = aggregate_prerequisite_tokens(
            tuple(row["prerequisite_ids"]) + (marker,)
        )
        assert (
            poisoned.disposition is PrerequisiteAggregateDisposition.GRAMMAR_REFUSED
        ), stable_id
        assert poisoned.served == (), stable_id
        assert poisoned.platform == (), stable_id
        assert poisoned.covered == (), stable_id
        assert poisoned.unrecoverable == (), stable_id

    # m4954's own row -- the marker ALONE -- refuses identically: it never
    # falls through to the empty-set blocked arm as if nothing were asserted.
    own = aggregate_prerequisite_tokens(m4954_tokens)
    assert own.disposition is PrerequisiteAggregateDisposition.GRAMMAR_REFUSED


# ---------------------------------------------------------------------------
# Cell 3: the platform class is EXACTLY the closed five
# ---------------------------------------------------------------------------


def test_platform_class_is_exactly_the_closed_five_and_corpus_witnessed() -> None:
    """Set equality on the closed platform vocabulary, witnessed by the corpus.

    The production docstring claims every member is census-evidenced in the
    committed pilot corpus; the matrix holds it to that: each of the five is
    extracted from at least one VERBATIM prior-era token, and no corpus token
    extracts a platform-class name outside the closed set.
    """

    assert PLATFORM_CLASS_DISTRIBUTIONS == frozenset(
        {"triton", "flash-attn", "spconv", "knn-cuda", "cuda-runtime"}
    )

    witnessed: set[str] = set()
    for row in _corpus():
        for raw in row["prerequisite_ids"]:
            token = parse_prerequisite_token(raw)
            if token.platform_class:
                assert token.distribution in PLATFORM_CLASS_DISTRIBUTIONS, raw
                witnessed.add(token.distribution)
    assert witnessed == set(PLATFORM_CLASS_DISTRIBUTIONS), sorted(witnessed)


# ---------------------------------------------------------------------------
# Cell 4: foreign spellings classify TOTALLY -- typed or parsed, never a crash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", _FOREIGN_SPELLINGS, ids=repr)
def test_foreign_spellings_classify_totally(raw: str) -> None:
    """A spelling outside the five pilot forms still lands classified.

    ``classify_reader`` propagates any non-crawler exception as a matrix
    FAILURE, so a parsed token on a closed kind (or a crawler-typed refusal)
    is the only lawful landing; an unclassified crash can never be one.
    """

    parse_outcome = classify_reader(lambda: parse_prerequisite_token(raw))
    assert parse_outcome[0] in {"parsed", "refused-typed"}
    if parse_outcome[0] == "parsed":
        token = parse_outcome[1]
        assert isinstance(token.kind, PrerequisiteTokenKind)
        # An extraction, when one exists, is already PEP 503 canonical.
        if token.distribution is not None:
            assert token.distribution == canonical_distribution_name(token.distribution)
        else:
            assert token.kind is PrerequisiteTokenKind.CONTENT_FREE
            assert not token.platform_class

    ladder_outcome = classify_reader(lambda: aggregate_prerequisite_tokens((raw,)))
    assert ladder_outcome[0] in {"parsed", "refused-typed"}
    if ladder_outcome[0] == "parsed":
        assert isinstance(
            ladder_outcome[1].disposition, PrerequisiteAggregateDisposition
        )
