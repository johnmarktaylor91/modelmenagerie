"""T4-T1 membership half: a subset rung's reducer opens over registered superset history.

K2's currency carve-out (``prior_intake_snapshots``) let terminals recorded
under a PRIOR registered snapshot of the same campaign stay dependency-current
under a SUPERSET rung. The MEMBERSHIP half was still missing: the reducer's
construction-time check ``set(self._current) - self.intake_ids`` hard-stopped
any SUBSET rung whose ledgers carry revisions for prior-rung members outside
the new intake ("model revisions exist outside intake") -- the mini-rung leg
died in under a second against the step-100 history.

The carve-out is membership-only and lineage-gated: stable ids in the union of
REGISTERED prior same-campaign snapshot members (read driver-side, plumbed via
``AuthorityContext.prior_intake_member_ids``) are known read-only HISTORY at
construction. Every WRITE-side check keyed on ``intake_ids`` stays
CURRENT-intake only, and a stable id in NEITHER the intake NOR the registered
lineage still hard-stops with the exact original message.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.tests.conftest import (
    make_authority_context,
    make_failed_attempt,
    make_gate,
    make_model,
)

pytestmark = pytest.mark.smoke

_OLD_ID = "intake-" + "a" * 20
_OLD_SHA = "sha256:" + "a" * 64
_NEW_ID = "intake-" + "b" * 20
_NEW_SHA = "sha256:" + "b" * 64

#: The live repro shape: a 10-member mini-rung staged over a step-100 history
#: carrying 53 recorded model revisions. The mini members are themselves prior
#: members (a rung is a subset of its lineage), so the prior-member union
#: covers both populations.
_PRIOR_TERMINAL_IDS = tuple(f"m_prior_{index:02d}" for index in range(53))
_MINI_INTAKE_IDS = tuple(f"m_mini_{index}" for index in range(10))
_PRIOR_MEMBER_UNION = frozenset(_PRIOR_TERMINAL_IDS) | frozenset(_MINI_INTAKE_IDS)


def _ledger_paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths."""

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _seed_terminals_under_old_snapshot(paths: LedgerPaths, stable_ids: Sequence[str]) -> None:
    """Append one failed:source terminal per stable id under the OLD snapshot."""

    context = make_authority_context(stable_ids, snapshot_id=_OLD_ID, snapshot_sha256=_OLD_SHA)
    with CanonicalReducer(paths, context) as reducer:
        for index, stable_id in enumerate(stable_ids):
            attempt = make_failed_attempt(stable_id, attempt_id=f"attempt-{index}")
            attempt.pop("ledger_seq", None)
            model = make_model(stable_id, accepted=False, status_code="failed:source")
            model["status"]["attempt_ids"] = [attempt["attempt_id"]]
            model["execution"]["accepted_attempt_ids"] = []
            model.pop("record_seq", None)
            reducer.append_attempt(attempt)
            reducer.append_model(reducer.prepare_model(model))


@pytest.fixture(scope="module")
def superset_history(tmp_path_factory: pytest.TempPathFactory) -> LedgerPaths:
    """Seed the 53-terminal step-100-shaped history once for the module.

    Every consumer only OPENS reducers over these ledgers (each refused write
    raises before any append), so sharing the seeded bytes is safe.
    """

    tmp = tmp_path_factory.mktemp("superset-history")
    paths = _ledger_paths(tmp)
    _seed_terminals_under_old_snapshot(paths, _PRIOR_TERMINAL_IDS)
    return paths


def _mini_rung_context(prior_member_ids: frozenset[str]):
    """Return the 10-member subset-rung context with one prior-member union."""

    return replace(
        make_authority_context(_MINI_INTAKE_IDS, snapshot_id=_NEW_ID, snapshot_sha256=_NEW_SHA),
        prior_intake_snapshots={_OLD_ID: _OLD_SHA},
        prior_intake_member_ids=prior_member_ids,
    )


def test_subset_rung_reducer_opens_over_registered_superset_history(
    superset_history: LedgerPaths,
) -> None:
    """RED shape (a): 10-member intake over 53 prior-member revisions opens.

    On the pre-fix reducer this construction raised ``model revisions exist
    outside intake: [53 step-100 models]`` -- the live mini-rung launch blocker.
    """

    with CanonicalReducer(superset_history, _mini_rung_context(_PRIOR_MEMBER_UNION)) as reducer:
        assert set(reducer._current) == set(_PRIOR_TERMINAL_IDS)
        # Write authority is untouched: the current intake stays 10 members.
        assert reducer.intake_ids == frozenset(_MINI_INTAKE_IDS)


@pytest.mark.parametrize("active_width", (1, 2, 9, 11, 17, 52, 53, 61))
def test_registered_history_membership_is_width_independent(
    superset_history: LedgerPaths, active_width: int
) -> None:
    """Subset, overlap, exact, and superset active widths all open identically."""

    overlap = _PRIOR_TERMINAL_IDS[: min(active_width, len(_PRIOR_TERMINAL_IDS))]
    novel = tuple(f"m_active_novel_{index}" for index in range(max(0, active_width - len(overlap))))
    active_ids = (*overlap, *novel)
    context = replace(
        make_authority_context(
            active_ids,
            snapshot_id=_NEW_ID,
            snapshot_sha256=_NEW_SHA,
        ),
        prior_intake_snapshots={_OLD_ID: _OLD_SHA},
        prior_intake_member_ids=frozenset(_PRIOR_TERMINAL_IDS) | frozenset(active_ids),
    )
    with CanonicalReducer(superset_history, context) as reducer:
        assert set(reducer._current) == set(_PRIOR_TERMINAL_IDS)
        assert reducer.intake_ids == frozenset(active_ids)


def test_prior_snapshot_member_still_refuses_new_writes(
    superset_history: LedgerPaths,
) -> None:
    """RED shape (b): prior-member history is read-only in the subset leg.

    A registered prior member outside the CURRENT intake must refuse new
    attempts, model revisions, and gate items exactly as any foreign id.
    """

    prior_id = _PRIOR_TERMINAL_IDS[0]
    with CanonicalReducer(superset_history, _mini_rung_context(_PRIOR_MEMBER_UNION)) as reducer:
        attempt = make_failed_attempt(prior_id, attempt_id="attempt-new")
        attempt.pop("ledger_seq", None)
        with pytest.raises(ReductionError, match="attempt stable_id is outside intake"):
            reducer.append_attempt(attempt)
        model = make_model(prior_id, accepted=False, status_code="failed:source")
        model.pop("record_seq", None)
        with pytest.raises(ReductionError, match="model stable_id is outside intake"):
            reducer.append_model(model)
        with pytest.raises(ReductionError, match="gate item stable_id is outside intake"):
            reducer.append_gate(make_gate([prior_id]))


def test_unregistered_stable_id_still_hard_stops_with_the_exact_message(
    tmp_path: Path,
) -> None:
    """Shape (c): the foreign-records tripwire is byte-identical and precise.

    A ledger carrying one registered prior member AND one id in neither the
    intake nor the registered lineage refuses at construction, naming ONLY the
    unregistered id in the exact original message.
    """

    paths = _ledger_paths(tmp_path)
    _seed_terminals_under_old_snapshot(paths, (_PRIOR_TERMINAL_IDS[0], "m_unregistered"))
    with pytest.raises(
        ReductionError,
        match=r"model revisions exist outside intake: \['m_unregistered'\]$",
    ):
        CanonicalReducer(paths, _mini_rung_context(_PRIOR_MEMBER_UNION))


def test_empty_prior_member_union_keeps_the_exact_pin_world(tmp_path: Path) -> None:
    """Shape (c) control: without lineage the refusal covers every foreign id."""

    paths = _ledger_paths(tmp_path)
    _seed_terminals_under_old_snapshot(paths, (_PRIOR_TERMINAL_IDS[0],))
    with pytest.raises(
        ReductionError,
        match=rf"model revisions exist outside intake: \['{_PRIOR_TERMINAL_IDS[0]}'\]$",
    ):
        CanonicalReducer(paths, _mini_rung_context(frozenset()))


def test_prior_member_reader_is_campaign_scoped_and_lineage_bound(
    tmp_path: Path,
) -> None:
    """Shape (d): the driver-side member union licenses same-campaign lineage only.

    A FOREIGN campaign's registered rung never contributes members, the union
    is the campaign root plus strictly earlier same-campaign rungs, and an
    unregistered or root snapshot yields the empty (exact-pin) union.
    """

    from menagerie.crawler.identity import canonical_json_bytes
    from menagerie.crawler.partitioner import emit_campaign_partitions
    from menagerie.crawler.rung_staging import registered_prior_intake_members
    from menagerie.crawler.tests.test_rung_staging import (
        _ROSTER_ROWS,
        _close_staged_rung,
        _write_models_shard,
        _root_binding,
        _root_snapshot,
        _stage,
    )

    repo_root = tmp_path / "repo"
    records_root = repo_root / "menagerie" / "crawler" / "records"
    roster_path = tmp_path / "roster.jsonl"
    roster_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in _ROSTER_ROWS))
    emit_campaign_partitions(roster_path, records_root)
    # AW-2 fails CLOSED on an absent models ledger; the empty shard states
    # explicitly that no awards exist when the rungs stage.
    _write_models_shard(records_root, [])

    root_binding = _root_binding(repo_root, "c3-classics")
    classics_ids = tuple(item.stable_id for item in _root_snapshot(repo_root, "c3-classics").items)
    mech_ids = tuple(item.stable_id for item in _root_snapshot(repo_root, "c1-mech").items)
    rung_a = _stage(repo_root, "c3-classics", stable_ids=classics_ids[:1], label="rung-a")
    # BD-1: rung-b derives from rung-a, so rung-a must hold an immutable close.
    _close_staged_rung(repo_root, rung_a, "c3-classics")
    rung_b = _stage(
        repo_root,
        "c3-classics",
        stable_ids=classics_ids[:2],
        derived_from=rung_a.registration.snapshot_id,
        label="rung-b",
    )
    # A registered rung of ANOTHER campaign must never license members here.
    foreign = _stage(repo_root, "c1-mech", stable_ids=mech_ids[:2], label="foreign-rung")

    members_for_b = registered_prior_intake_members(
        records_root / "intake" / rung_b.registration.snapshot_id,
        rung_b.registration.snapshot_id,
    )
    # Lineage for rung B: the c3 campaign root plus the earlier rung A.
    assert members_for_b == frozenset(classics_ids)
    assert not members_for_b & set(mech_ids)

    members_for_a = registered_prior_intake_members(
        records_root / "intake" / rung_a.registration.snapshot_id,
        rung_a.registration.snapshot_id,
    )
    # Rung B is LATER, so only the root licenses A's history.
    assert members_for_a == frozenset(classics_ids)

    # The root snapshot itself is unregistered as a rung: exact-pin world.
    root_intake = records_root / root_binding.intake_path
    assert (
        registered_prior_intake_members(root_intake, root_binding.intake_snapshot_id) == frozenset()
    )
    assert foreign.registration.campaign_id == "c1-mech"


def test_prior_member_union_requires_matching_registered_bytes(
    tmp_path: Path,
) -> None:
    """Shape (d) fail-closed arm: tampered snapshot bytes license nothing.

    The reader re-verifies every prior snapshot's content identity against its
    REGISTERED digest; a mutated members file silently contributes no members
    instead of laundering foreign ids into the union.
    """

    from menagerie.crawler.identity import canonical_json_bytes
    from menagerie.crawler.partitioner import emit_campaign_partitions
    from menagerie.crawler.rung_staging import registered_prior_intake_members
    from menagerie.crawler.tests.test_rung_staging import (
        _ROSTER_ROWS,
        _close_staged_rung,
        _write_models_shard,
        _root_snapshot,
        _stage,
    )

    repo_root = tmp_path / "repo"
    records_root = repo_root / "menagerie" / "crawler" / "records"
    roster_path = tmp_path / "roster.jsonl"
    roster_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in _ROSTER_ROWS))
    emit_campaign_partitions(roster_path, records_root)
    # AW-2 fails CLOSED on an absent models ledger; the empty shard states
    # explicitly that no awards exist when the rungs stage.
    _write_models_shard(records_root, [])

    classics_ids = tuple(item.stable_id for item in _root_snapshot(repo_root, "c3-classics").items)
    rung_a = _stage(repo_root, "c3-classics", stable_ids=classics_ids[:1], label="rung-a")
    # BD-1: rung-b derives from rung-a, so rung-a must hold an immutable close.
    _close_staged_rung(repo_root, rung_a, "c3-classics")
    rung_b = _stage(
        repo_root,
        "c3-classics",
        stable_ids=classics_ids[:2],
        derived_from=rung_a.registration.snapshot_id,
        label="rung-b",
    )

    # Corrupt rung A's durable snapshot bytes after registration.
    items_path = records_root / "intake" / rung_a.registration.snapshot_id / "items.jsonl"
    items_path.write_bytes(items_path.read_bytes() + b'{"stable_id": "m_smuggled"}\n')

    members_for_b = registered_prior_intake_members(
        records_root / "intake" / rung_b.registration.snapshot_id,
        rung_b.registration.snapshot_id,
    )
    assert "m_smuggled" not in members_for_b
    # The campaign root still licenses its members; only the tampered rung is dropped.
    assert members_for_b == frozenset(classics_ids)
