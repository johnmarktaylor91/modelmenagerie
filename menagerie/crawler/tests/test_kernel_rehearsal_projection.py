"""C1 rehearsal: randomized append interleavings against the honesty kernel.

The kernel's read path serves an INCREMENTALLY PATCHED projection
(:meth:`CanonicalReducer._patched_projection_after_append`) stamped with a
freshly recomputed PC1 token, so every read between appends trusts the patch to
equal a full :func:`project_dependency_current` replay. The shipped suites pin
that equality on hand-built scenarios; this module rehearses it under
randomized-but-seeded interleavings of every writer surface (attempts, gates,
models, refused appends) across several stable IDs, which is the shape weeks of
unattended 20k crawling actually produces.

Three projections must agree after EVERY step, including steps whose append was
refused:

* the live reducer's cached/patched projection;
* a fresh in-memory replay over the same scanned rows; and
* a fresh replay read back from the persisted ledger BYTES (the
  ``open -> replay -> identical projection`` property).

Divergence in any of them is a silent-corruption class: the driver would award,
publish, or skip against a projection no rebuild can reproduce.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
from typing import Any, Optional

import pytest

from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import SingleWriterError, scan_jsonl
from menagerie.crawler.reducer import (
    CanonicalReducer,
    ReductionError,
    project_dependency_current,
)
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_attempt,
    make_authority_context,
    make_failed_attempt,
    make_gate,
    make_model,
)

_STABLE_IDS = ("m_r0", "m_r1", "m_r2")

#: Seeds are fixed so a failure reproduces exactly; the sequences they draw are
#: what varies. Randomized ORDER, deterministic TEST.
#:
#: Each step compares THREE full projections, so a seed costs ~20s. Only the
#: first few carry the smoke mark: they are enough to catch a broken patch on the
#: per-step gate, and the whole set still runs in the full corpus (the wave/merge
#: gate). Keeping every seed on smoke would add ~50% to the mandatory job.
_SMOKE_SEEDS = (1, 2, 3)
_SEEDS = (*_SMOKE_SEEDS, 5, 8, 13, 21, 34)
_SEED_PARAMS = tuple(
    pytest.param(seed, marks=(pytest.mark.smoke,) if seed in _SMOKE_SEEDS else ())
    for seed in _SEEDS
)


def _ledger_paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths."""

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _models_tail(paths: LedgerPaths, stable_id: str) -> Optional[dict[str, Any]]:
    """Return the persisted models-ledger tail row for one stable ID."""

    rows = [row for row in scan_jsonl(paths.models) if row.get("stable_id") == stable_id]
    return dict(rows[-1]) if rows else None


def _projection_triple(
    reducer: CanonicalReducer, paths: LedgerPaths, context: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (cached-or-patched, fresh-in-memory, fresh-from-bytes) projections.

    The three are compared by current membership plus the revision governing
    each member -- the disposition the driver actually consumes.
    """

    def summarize(projection: Any) -> dict[str, Any]:
        return {
            stable_id: record.get("record_revision")
            for stable_id, record in projection.current_records.items()
        }

    live = reducer._current_projection()
    in_memory = project_dependency_current(
        paths,
        context=context,
        model_records=tuple(reducer._models.iter_records()),
        attempt_records=tuple(reducer._attempts.iter_records()),
        gate_records=tuple(reducer._gates.iter_records()),
        artifact_records=tuple(reducer._artifacts.iter_events()),
    )
    from_bytes = project_dependency_current(paths, context=context)
    return summarize(live), summarize(in_memory), summarize(from_bytes)


def _assert_projections_agree(
    reducer: CanonicalReducer, paths: LedgerPaths, context: Any, step: str
) -> None:
    """Require the patched, in-memory, and on-disk projections to be identical."""

    live, in_memory, from_bytes = _projection_triple(reducer, paths, context)
    assert live == in_memory, (
        f"patched projection diverged from a fresh in-memory replay after {step}: "
        f"patched={live} fresh={in_memory}"
    )
    assert live == from_bytes, (
        f"patched projection diverged from a replay of the persisted bytes after {step}: "
        f"patched={live} bytes={from_bytes}"
    )
    stale = dict(reducer._current_projection().stale_reasons)
    overlap = set(live) & set(stale)
    assert not overlap, f"stable ids are both current and stale after {step}: {sorted(overlap)}"


def _append_succeeded_attempt(reducer: CanonicalReducer, stable_id: str, ordinal: int) -> None:
    """Append one authenticated succeeded forward attempt."""

    attempt = make_attempt(stable_id, attempt_id=f"{stable_id}-ok-{ordinal}")
    attempt.pop("ledger_seq", None)
    reducer.append_attempt(attempt)


def _append_failed_attempt(reducer: CanonicalReducer, stable_id: str, ordinal: int) -> str:
    """Append one failed source-stage attempt and return its attempt ID."""

    attempt_id = f"{stable_id}-bad-{ordinal}"
    attempt = make_failed_attempt(stable_id, attempt_id=attempt_id)
    attempt.pop("ledger_seq", None)
    reducer.append_attempt(attempt)
    return attempt_id


def _append_failure_terminal(
    reducer: CanonicalReducer, paths: LedgerPaths, stable_id: str, ordinal: int
) -> None:
    """Append one failed:source terminal over whatever revision currently governs."""

    attempt_id = f"{stable_id}-bad-{ordinal}"
    attempt = make_failed_attempt(stable_id, attempt_id=attempt_id)
    attempt.pop("ledger_seq", None)
    try:
        reducer.append_attempt(attempt)
    except ReductionError:
        pass
    model = bind_terminal_attempts(
        make_model(stable_id, accepted=False, status_code="failed:source", attempt_id=attempt_id),
        [attempt],
    )
    model["execution"]["accepted_attempt_ids"] = []
    previous = _models_tail(paths, stable_id)
    if previous is None:
        model.pop("record_seq", None)
    else:
        model["parent_revision"] = previous["record_revision"]
        model["status"]["supersedes_revision"] = previous["record_revision"]
        model["record_seq"] = int(previous["record_seq"]) + 1
    reducer.append_model(reducer.prepare_model(model))


def _append_runs_terminal(
    reducer: CanonicalReducer, paths: LedgerPaths, stable_id: str, ordinal: int
) -> None:
    """Append one accepted runs revision over whatever revision currently governs."""

    attempt = make_attempt(stable_id, attempt_id=f"{stable_id}-ok-{ordinal}")
    attempt.pop("ledger_seq", None)
    try:
        reducer.append_attempt(attempt)
    except ReductionError:
        pass
    model = make_model(stable_id, accepted=True, attempt_id=f"{stable_id}-ok-{ordinal}")
    previous = _models_tail(paths, stable_id)
    if previous is None:
        model.pop("record_seq", None)
    else:
        model["parent_revision"] = previous["record_revision"]
        model["status"]["supersedes_revision"] = previous["record_revision"]
        model["record_seq"] = int(previous["record_seq"]) + 1
    reducer.append_model(reducer.prepare_model(model))


def _append_gate(reducer: CanonicalReducer, ordinal: int) -> None:
    """Append one metadata-batch gate covering every rehearsal stable ID."""

    gate = make_gate(list(_STABLE_IDS), gate_id=f"gate-{ordinal}")
    gate.pop("ledger_seq", None)
    reducer.append_gate(gate)


@pytest.mark.parametrize("seed", _SEED_PARAMS)
def test_patched_projection_equals_a_fresh_replay_under_random_interleavings(
    tmp_path: Path, seed: int
) -> None:
    """Every append surface, interleaved at random, keeps the three projections equal.

    This is the PC1 correctness property: the cache is stamped with a token
    recomputed AFTER the append, so a wrong patch is served silently until the
    next append. Refused appends are exercised deliberately -- a refusal that
    leaves a patched projection behind is the same corruption class.
    """

    rng = random.Random(seed)
    paths = _ledger_paths(tmp_path)
    context = make_authority_context(list(_STABLE_IDS))
    operations = ("gate", "attempt_ok", "attempt_bad", "terminal_failed", "terminal_runs")
    with CanonicalReducer(paths, context) as reducer:
        _assert_projections_agree(reducer, paths, context, "construction")
        for ordinal in range(1, 19):
            operation = rng.choice(operations)
            stable_id = rng.choice(_STABLE_IDS)
            try:
                if operation == "gate":
                    _append_gate(reducer, ordinal)
                elif operation == "attempt_ok":
                    _append_succeeded_attempt(reducer, stable_id, ordinal)
                elif operation == "attempt_bad":
                    _append_failed_attempt(reducer, stable_id, ordinal)
                elif operation == "terminal_failed":
                    _append_failure_terminal(reducer, paths, stable_id, ordinal)
                else:
                    _append_runs_terminal(reducer, paths, stable_id, ordinal)
            except ReductionError:
                # A refusal is a legitimate outcome of a random sequence; the
                # projection contract must survive it unchanged.
                _assert_projections_agree(
                    reducer, paths, context, f"refused {operation} on {stable_id} (step {ordinal})"
                )
                continue
            _assert_projections_agree(
                reducer, paths, context, f"{operation} on {stable_id} (step {ordinal})"
            )


@pytest.mark.parametrize("seed", _SEED_PARAMS[:4])
def test_reopened_reducer_reproduces_the_live_projection(tmp_path: Path, seed: int) -> None:
    """Closing and reopening the reducer must reproduce the identical projection.

    The rebuild path a weeks-long crawl depends on: whatever the live process
    served, a cold reducer opened over the same bytes must serve too.
    """

    rng = random.Random(seed)
    paths = _ledger_paths(tmp_path)
    context = make_authority_context(list(_STABLE_IDS))
    with CanonicalReducer(paths, context) as reducer:
        for ordinal in range(1, 13):
            stable_id = rng.choice(_STABLE_IDS)
            operation = rng.choice(("gate", "attempt_ok", "terminal_failed", "terminal_runs"))
            try:
                if operation == "gate":
                    _append_gate(reducer, ordinal)
                elif operation == "attempt_ok":
                    _append_succeeded_attempt(reducer, stable_id, ordinal)
                elif operation == "terminal_failed":
                    _append_failure_terminal(reducer, paths, stable_id, ordinal)
                else:
                    _append_runs_terminal(reducer, paths, stable_id, ordinal)
            except ReductionError:
                continue
        live = {
            stable_id: record.get("record_revision")
            for stable_id, record in reducer.current_records.items()
        }
        live_stale = dict(reducer._current_projection().stale_reasons)
    with CanonicalReducer(paths, context) as reopened:
        cold = {
            stable_id: record.get("record_revision")
            for stable_id, record in reopened.current_records.items()
        }
        cold_stale = dict(reopened._current_projection().stale_reasons)
    assert live == cold, f"reopened projection differs from the live one: {live} vs {cold}"
    assert live_stale.keys() == cold_stale.keys(), (
        f"reopened stale set differs: {sorted(live_stale)} vs {sorted(cold_stale)}"
    )


@pytest.mark.smoke
def test_pc1_token_moves_on_every_writer_surface(tmp_path: Path) -> None:
    """Each ledger's append generation must move the projection token.

    T1-15 shipped this for models/attempts/gates/artifacts; the rehearsal pins
    that no surface silently keeps a token (and therefore a pre-append
    projection) alive.
    """

    paths = _ledger_paths(tmp_path)
    context = make_authority_context(list(_STABLE_IDS))
    with CanonicalReducer(paths, context) as reducer:
        seen: set[tuple[Any, ...]] = {reducer._projection_token()}

        _append_gate(reducer, 1)
        token = reducer._projection_token()
        assert token not in seen, "a gate append did not move the projection token"
        seen.add(token)

        _append_succeeded_attempt(reducer, _STABLE_IDS[0], 1)
        token = reducer._projection_token()
        assert token not in seen, "an attempt append did not move the projection token"
        seen.add(token)

        _append_runs_terminal(reducer, paths, _STABLE_IDS[0], 1)
        token = reducer._projection_token()
        assert token not in seen, "a model append did not move the projection token"
        seen.add(token)


@pytest.mark.smoke
def test_writer_exclusivity_is_what_makes_the_in_memory_token_sound(
    tmp_path: Path,
) -> None:
    """The PC1 token counts IN-MEMORY appends, so exclusivity is load-bearing.

    ``_projection_token`` pairs the authority-bytes digest with each ledger's
    in-process append generation. An append that bypasses this reducer therefore
    canNOT move the token -- the cached projection would be served over changed
    bytes. That is safe only because the canonical ledgers are single-writer.

    This test pins BOTH halves so the coupling is explicit: a second writer is
    refused, and an out-of-band byte append is invisible to the token. If writer
    exclusivity is ever relaxed (sharded or multi-process writers), the
    projection cache must be re-keyed on file identity before that lands.
    """

    paths = _ledger_paths(tmp_path)
    context = make_authority_context(list(_STABLE_IDS))
    with CanonicalReducer(paths, context) as reducer:
        _append_gate(reducer, 1)
        _append_runs_terminal(reducer, paths, _STABLE_IDS[0], 1)

        with pytest.raises(SingleWriterError) as refused:
            CanonicalReducer(paths, context)
        assert str(paths.models) in str(refused.value), (
            "the single-writer refusal must name the ledger an operator has to release"
        )

        token_before = reducer._projection_token()
        with paths.models.open("ab") as handle:
            handle.write(b'{"schema_version": "menagerie.crawler.model.v3"}\n')
        assert reducer._projection_token() == token_before, (
            "the token counts in-memory appends: this documents WHY the single-writer "
            "lock is a correctness dependency of the projection cache, not just hygiene"
        )


@pytest.mark.smoke
def test_a_refused_model_append_leaves_the_projection_untouched(tmp_path: Path) -> None:
    """A refused append must not disturb the cached projection or its token.

    The award-guard and grant-pending refusals fire AFTER most validation, so a
    refusal path that had already patched the cache would serve a revision that
    was never persisted.
    """

    paths = _ledger_paths(tmp_path)
    context = make_authority_context(list(_STABLE_IDS))
    with CanonicalReducer(paths, context) as reducer:
        _append_gate(reducer, 1)
        _append_runs_terminal(reducer, paths, _STABLE_IDS[0], 1)
        before = deepcopy(dict(reducer.current_records))
        token_before = reducer._projection_token()

        # Bad parentage: refused deep inside admission.
        broken = make_model(_STABLE_IDS[0], accepted=True, attempt_id=f"{_STABLE_IDS[0]}-ok-1")
        broken["parent_revision"] = "sha256:" + "f" * 64
        broken["status"]["supersedes_revision"] = "sha256:" + "f" * 64
        with pytest.raises(ReductionError):
            reducer.append_model(reducer.prepare_model(broken))

        assert reducer._projection_token() == token_before, (
            "a refused append moved the projection token"
        )
        assert dict(reducer.current_records) == before, (
            "a refused append changed the current projection"
        )
        _assert_projections_agree(reducer, paths, context, "a refused model append")
