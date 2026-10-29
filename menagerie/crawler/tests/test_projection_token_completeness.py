"""IC-5: the projection-cache token counts EVERY mutable projection input.

Third recurrence of token incompleteness (T1-15 was the second): the token
covered the authority-bytes digest and the four ledger append generations, but
`project_dependency_current`'s outcome ALSO depends on the reducer's
`historical_generation_families` map (mutated additively mid-leg by
`register_attested_generation_families` / `refresh_attested_generation_families`)
(and, before RC-2, on the rotation registry file). A mid-leg family
registration therefore kept serving a projection computed over the
pre-mutation world. Since RC-2 the projection reads NO rotation ledger -- every
settled head replays at its recorded policy basis unconditionally and the
rotation class feeds the recheck census -- so the registry is no longer a
projection input and no longer a token component.

The pins here are behavioral (each mutable input, when moved, must change the
token) plus the declared-inputs structural check binding the token arity to
the named input list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import _PROJECTION_TOKEN_MUTABLE_INPUTS
from menagerie.crawler.tests.test_reducer import CanonicalReducer

pytestmark = pytest.mark.smoke


def _paths(tmp_path: Path) -> LedgerPaths:
    records = tmp_path / "records"
    records.mkdir(parents=True, exist_ok=True)
    return LedgerPaths(
        models=records / "models.jsonl",
        attempts=records / "attempts.jsonl",
        gates=records / "gates.jsonl",
    )


def test_family_map_registration_moves_the_token(tmp_path: Path) -> None:
    """A NEW attested generation binding must invalidate the cached
    projection; re-registering identical bindings must NOT (cache stays
    warm for the wave discipline)."""

    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        before = reducer._projection_token()  # noqa: SLF001 -- the pinned surface
        reducer.register_attested_generation_families({"env-gen-x": "family-y"})
        after = reducer._projection_token()  # noqa: SLF001
        assert before != after, (
            "historical_generation_families mutated without moving the "
            "projection token: the cache serves a pre-mutation projection"
        )
        reducer.register_attested_generation_families({"env-gen-x": "family-y"})
        assert reducer._projection_token() == after  # noqa: SLF001


def test_token_arity_matches_the_declared_mutable_inputs(tmp_path: Path) -> None:
    """The declared mutable-input list is the token's own documentation: a new
    mutable input must be added to BOTH, in the same reviewed change."""

    assert _PROJECTION_TOKEN_MUTABLE_INPUTS == (
        "authority_bytes_digest",
        "models_append_generation",
        "attempts_append_generation",
        "gates_append_generation",
        "artifact_events_append_generation",
        "historical_generation_families_epoch",
    )
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        token = reducer._projection_token()  # noqa: SLF001
        assert len(token) == len(_PROJECTION_TOKEN_MUTABLE_INPUTS)
