"""Projection-cache correctness: ledger-generation invalidation + replayed patches.

T1-15 (C-PROJ-CACHE, M3 precondition): the in-leg projection cache token was
code/env bytes ONLY, so gate/attempt appends between two ``current_records``
reads kept serving the pre-append projection, and ``append_model`` patched its
own revision in as dependency-current with no replay. These tests pin the two
fixed contracts:

1. the cache token incorporates the append generation of EVERY ledger the
   projection reads, while evidence-only appends advance that token by replaying
   only the stable IDs the new evidence names; and
2. the post-append cache patch REPLAYS the appended record's currency exactly
   as a fresh :func:`project_dependency_current` would, never asserting it.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
from typing import Any, Mapping

import pytest

import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import (
    CanonicalReducer,
    ReductionError,
    project_dependency_current,
)
from menagerie.crawler.tests.conftest import (
    make_attempt,
    make_authority_context,
    make_gate,
    make_model,
)
from menagerie.crawler.tests.test_reducer import (
    CanonicalReducer as AdapterCanonicalReducer,
    _attempt_for_model,
    _variant_from_representative,
)

#: Sub-second load-bearing coverage promoted into the mandatory smoke tier (T5-7).
pytestmark = pytest.mark.smoke


def _paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths."""

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _counting_projection(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Route the module projection through a call counter.

    Returns
    -------
    list[int]
        Single-element mutable counter of full projection computations.
    """

    calls = [0]
    original = reducer_module.project_dependency_current

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reducer_module, "project_dependency_current", counted)
    return calls


def _assert_cached_equals_fresh(reducer: CanonicalReducer) -> None:
    """Assert the served projection equals a fresh one over the same ledgers."""

    cached = reducer._current_projection()
    fresh = project_dependency_current(
        reducer.ledger_paths,
        context=reducer.context,
        model_records=tuple(reducer._models.iter_records()),
        attempt_records=tuple(reducer._attempts.iter_records()),
        gate_records=tuple(reducer._gates.iter_records()),
        artifact_records=reducer._artifacts.events,
    )
    assert dict(cached.current_records) == dict(fresh.current_records)
    assert dict(cached.stale_reasons) == dict(fresh.stale_reasons)


def test_gate_and_attempt_appends_advance_projection_without_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence appends advance the token without an O(history) recompute."""

    stable_ids = [f"m_{index}" for index in range(10)]
    context = make_authority_context(stable_ids)
    with CanonicalReducer(_paths(tmp_path), context) as reducer:
        calls = _counting_projection(monkeypatch)
        assert reducer.current_records == {}
        assert calls[0] == 1
        # Unchanged ledgers: the cache serves without recomputation.
        assert reducer.current_records == {}
        assert calls[0] == 1

        reducer.append_gate(make_gate(stable_ids))
        assert reducer.current_records == {}
        assert calls[0] == 1, "a gate append must not reproject model history"

        reducer.append_attempt(make_attempt("m_0", attempt_id="attempt-1"))
        assert reducer.current_records == {}
        assert calls[0] == 1, "an attempt append must not reproject model history"

        # Idempotent re-appends leave every append generation unchanged.
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(make_attempt("m_0", attempt_id="attempt-1"))
        assert reducer.current_records == {}
        assert calls[0] == 1


def test_projection_token_binds_every_ledger_generation(tmp_path: Path) -> None:
    """The cache token counts every declared mutable input: authority bytes,
    all four append generations, and the attested-family epoch (IC-5; RC-2
    removed the rotation registry from the projection's inputs)."""

    stable_ids = [f"m_{index}" for index in range(10)]
    context = make_authority_context(stable_ids)
    with CanonicalReducer(_paths(tmp_path), context) as reducer:
        token = reducer._projection_token()
        assert token == (
            reducer_module._dependency_bytes_token(reducer.ledger_paths),
            reducer._models.record_count,
            reducer._attempts.record_count,
            reducer._gates.record_count,
            reducer.artifact_ledger.event_count,
            reducer._generation_families_epoch,
        )
        reducer.append_gate(make_gate(stable_ids))
        assert reducer._projection_token() != token


def test_artifact_appends_advance_projection_without_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact evidence advances the fourth generation without history replay."""

    from menagerie.crawler.tests.test_artifact_transactions import _commit_private, _mirrors

    paths = _paths(tmp_path)
    context = make_authority_context(["m_0"])
    with CanonicalReducer(paths, context) as reducer:
        calls = _counting_projection(monkeypatch)
        assert reducer.current_records == {}
        assert calls[0] == 1
        _commit_private(
            "m_0",
            content=b"projection-cache artifact mutation",
            context=context,
            mirrors=_mirrors(tmp_path),
            ledger=reducer.artifact_ledger,
            canonical_root=tmp_path / "canonical",
            repository_root=tmp_path,
            evidence_id="ev-projection-cache",
        )
        assert reducer.artifact_ledger.event_count == 4
        assert reducer.current_records == {}
        assert calls[0] == 1
        _assert_cached_equals_fresh(reducer)
    with CanonicalReducer(paths, context) as reopened:
        assert reopened.artifact_ledger.event_count == 4
        _assert_cached_equals_fresh(reopened)


def test_append_model_patch_replays_currency_instead_of_asserting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-leg projection after an append must agree with a fresh projection.

    Before the fix, ``append_model`` patched the new revision in as
    dependency-current with no replay: a revision a fresh projection calls
    stale stayed served as current for the whole leg (O9's optimistic-close
    hazard). The live dependency failure is simulated exactly as
    ``test_dependency_stale_representative_cascades_to_current_variant`` does.
    """

    paths = _paths(tmp_path)
    stable_ids = [f"m_{index}" for index in range(10)]
    with AdapterCanonicalReducer(paths, stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(make_attempt("m_0", attempt_id="attempt-1"))
        persisted = reducer.append_model(
            make_model("m_0", accepted=True, attempt_id="attempt-1")
        ).record
        assert "m_0" in reducer.current_records

        original = reducer_module._referenced_evidence_repasses

        def stale_m0(
            replay: CanonicalReducer,
            record: Mapping[str, Any],
            attempts: tuple[dict[str, Any], ...],
            gates: tuple[dict[str, Any], ...],
        ) -> None:
            """Simulate a live dependency failure for m_0 only."""

            if record.get("stable_id") == "m_0":
                raise ReductionError("m_0 dependency changed")
            original(replay, record, attempts, gates)

        monkeypatch.setattr(reducer_module, "_referenced_evidence_repasses", stale_m0)

        superseding = deepcopy(persisted)
        superseding.pop("record_seq", None)
        superseding.pop("record_revision", None)
        superseding["parent_revision"] = persisted["record_revision"]
        superseding["status"]["supersedes_revision"] = persisted["record_revision"]
        superseding["notes"] = "metadata-only re-vet"
        reducer.append_model(superseding)

        fresh = project_dependency_current(paths, context=reducer.context)
        assert "m_0" not in fresh.current_records
        assert "m_0" not in reducer.current_records, (
            "append_model patched itself in as current without replaying currency"
        )
        assert "m_0 dependency changed" in reducer._current_projection().stale_reasons["m_0"]
        assert not reducer.has_current("m_0")


def test_patched_projection_matches_fresh_after_every_append(tmp_path: Path) -> None:
    """Differential: cached projection == fresh projection after each append."""

    paths = _paths(tmp_path)
    gate_ids = ["m_rep", *(f"m_{index}" for index in range(9))]
    stable_ids = [*gate_ids, "m_variant"]
    with AdapterCanonicalReducer(
        paths,
        stable_ids,
        intake_variant_bindings={"m_variant": ("m_rep", "Large")},
    ) as reducer:
        reducer.append_gate(make_gate(gate_ids))
        _assert_cached_equals_fresh(reducer)
        reducer.append_attempt(make_attempt("m_rep", attempt_id="attempt-rep"))
        _assert_cached_equals_fresh(reducer)
        persisted_representative = reducer.append_model(
            make_model("m_rep", accepted=True, attempt_id="attempt-rep")
        ).record
        _assert_cached_equals_fresh(reducer)

        variant = _variant_from_representative(persisted_representative)
        variant_attempt = _attempt_for_model(variant, "attempt-variant")
        variant_attempt.pop("ledger_seq")
        variant_attempt.pop("payload_sha256")
        reducer.append_attempt(variant_attempt)
        reducer.append_model(variant)
        _assert_cached_equals_fresh(reducer)
        assert reducer.has_current("m_variant")

        superseding = deepcopy(persisted_representative)
        superseding.pop("record_seq", None)
        superseding.pop("record_revision", None)
        superseding["parent_revision"] = persisted_representative["record_revision"]
        superseding["status"]["supersedes_revision"] = persisted_representative["record_revision"]
        superseding["notes"] = "metadata-only representative re-vet"
        reducer.append_model(superseding)
        # The representative moved to the ledger tail: its variant replays
        # before it in fresh revision order and must be stale in BOTH views.
        _assert_cached_equals_fresh(reducer)
        assert reducer.has_current("m_rep")
        assert not reducer.has_current("m_variant")


@pytest.mark.parametrize("seed", (7, 29, 101, 997))
def test_randomized_append_interleavings_reopen_to_identical_projection(
    tmp_path: Path, seed: int
) -> None:
    """Seeded topological interleavings stay equal live, fresh, and reopened.

    Each mutation receives a defensive copy. Attempts may land before or after
    the gate, copied idempotent attempt/gate replays are mixed in, and a cold
    reducer reopen checks the projection after every accepted mutation.
    """

    rng = random.Random(seed)
    paths = _paths(tmp_path)
    stable_ids = tuple(f"m_{index}" for index in range(10))
    gate = make_gate(stable_ids)
    attempts = {
        stable_id: make_attempt(stable_id, attempt_id=f"attempt-{stable_id}")
        for stable_id in stable_ids[:4]
    }
    for attempt in attempts.values():
        attempt.pop("ledger_seq", None)
        attempt.pop("payload_sha256", None)
    models = {
        stable_id: make_model(
            stable_id,
            accepted=True,
            attempt_id=f"attempt-{stable_id}",
        )
        for stable_id in stable_ids[:4]
    }
    for model in models.values():
        model.pop("record_seq", None)
        model.pop("record_revision", None)
    pending = {
        "gate",
        "gate-replay",
        *(f"attempt:{stable_id}" for stable_id in attempts),
        *(f"attempt-replay:{stable_id}" for stable_id in attempts),
        *(f"model:{stable_id}" for stable_id in models),
    }
    completed: set[str] = set()
    while pending:
        available = []
        for action in sorted(pending):
            if action == "gate":
                available.append(action)
            elif action == "gate-replay" and "gate" in completed:
                available.append(action)
            elif action.startswith("attempt:"):
                available.append(action)
            elif action.startswith("attempt-replay:"):
                stable_id = action.rsplit(":", 1)[1]
                if f"attempt:{stable_id}" in completed:
                    available.append(action)
            elif action.startswith("model:"):
                stable_id = action.split(":", 1)[1]
                if "gate" in completed and f"attempt:{stable_id}" in completed:
                    available.append(action)
        action = rng.choice(available)
        with AdapterCanonicalReducer(paths, stable_ids) as reducer:
            token_before = reducer._projection_token()
            if action in {"gate", "gate-replay"}:
                result = reducer.append_gate(deepcopy(gate))
            elif action.startswith(("attempt:", "attempt-replay:")):
                stable_id = action.rsplit(":", 1)[1]
                result = reducer.append_attempt(deepcopy(attempts[stable_id]))
            else:
                stable_id = action.split(":", 1)[1]
                result = reducer.append_model(deepcopy(models[stable_id]))
            if result.appended:
                assert reducer._projection_token() != token_before
            else:
                assert reducer._projection_token() == token_before
            _assert_cached_equals_fresh(reducer)
            live_current = deepcopy(dict(reducer._current_projection().current_records))
            live_stale = dict(reducer._current_projection().stale_reasons)
        with AdapterCanonicalReducer(paths, stable_ids) as reopened:
            _assert_cached_equals_fresh(reopened)
            assert dict(reopened.current_records) == live_current
            assert dict(reopened._current_projection().stale_reasons) == live_stale
        pending.remove(action)
        completed.add(action)


def test_dependency_bytes_token_is_memoized_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authority bytes hash once per process; ledger generations carry liveness."""

    paths = _paths(tmp_path)
    reducer_module._reset_dependency_bytes_token_memo()
    try:
        token = reducer_module._dependency_bytes_token(paths)

        def refuse(payload: bytes) -> str:
            raise AssertionError("memoized token must not rehash authority bytes")

        monkeypatch.setattr(reducer_module, "hash_bytes", refuse)
        assert reducer_module._dependency_bytes_token(paths) == token
        reducer_module._reset_dependency_bytes_token_memo()
        with pytest.raises(AssertionError, match="must not rehash"):
            reducer_module._dependency_bytes_token(paths)
    finally:
        reducer_module._reset_dependency_bytes_token_memo()


def test_current_view_is_immutable_and_matches_current_records(tmp_path: Path) -> None:
    """``current_view`` equals ``current_records`` without allowing mutation."""

    stable_ids = [f"m_{index}" for index in range(10)]
    with AdapterCanonicalReducer(_paths(tmp_path), stable_ids) as reducer:
        reducer.append_gate(make_gate(stable_ids))
        reducer.append_attempt(make_attempt("m_0", attempt_id="attempt-1"))
        reducer.append_model(make_model("m_0", accepted=True, attempt_id="attempt-1"))

        view = reducer.current_view
        assert dict(view) == dict(reducer.current_records)
        assert reducer.has_current("m_0")
        assert not reducer.has_current("m_1")
        with pytest.raises(TypeError):
            view["m_0"] = {}  # type: ignore[index]
        # A captured view is a stable snapshot: appends replace the cached
        # projection instead of mutating it in place.
        persisted = view["m_0"]
        superseding = deepcopy(persisted)
        superseding.pop("record_seq", None)
        superseding.pop("record_revision", None)
        superseding["parent_revision"] = persisted["record_revision"]
        superseding["status"]["supersedes_revision"] = persisted["record_revision"]
        superseding["notes"] = "metadata-only re-vet"
        appended = reducer.append_model(superseding).record
        assert view["m_0"]["record_revision"] == persisted["record_revision"]
        assert reducer.current_view["m_0"]["record_revision"] == appended["record_revision"]
