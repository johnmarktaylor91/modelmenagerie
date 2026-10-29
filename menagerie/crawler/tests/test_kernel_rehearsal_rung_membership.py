"""C1 rehearsal: subset/superset rung membership at widths the RC6/RC7 suites skipped.

``test_subset_rung_reducer_membership`` pins the T4-T1 membership carve-out at
one width: a 10-member mini rung over a 53-terminal history. The ladder a 20k
crawl actually walks visits the degenerate and boundary widths too -- a
single-model re-drive, a rung identical to its lineage, a rung that adds brand
new members alongside prior history, and an empty intake. Each must either open
cleanly or hard-stop with the exact actionable message; none may open a reducer
whose write side accepts revisions for stable IDs outside the CURRENT intake.

The invariant under rehearsal: the carve-out is READ-side (construction) only.
Every write surface -- models, attempts, and gate items -- stays pinned to the
current intake at every width.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.tests.conftest import (
    make_attempt,
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


def _ledger_paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths."""

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _seed_history(paths: LedgerPaths, stable_ids: Sequence[str]) -> None:
    """Append one failed:source terminal per stable ID under the OLD snapshot."""

    context = make_authority_context(
        list(stable_ids), snapshot_id=_OLD_ID, snapshot_sha256=_OLD_SHA
    )
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


def _subset_context(intake_ids: Sequence[str], prior_member_ids: Sequence[str]):
    """Return a NEW-snapshot context whose lineage registers the prior members."""

    return replace(
        make_authority_context(list(intake_ids), snapshot_id=_NEW_ID, snapshot_sha256=_NEW_SHA),
        prior_intake_snapshots={_OLD_ID: _OLD_SHA},
        prior_intake_member_ids=frozenset(prior_member_ids),
    )


@pytest.mark.parametrize(
    ("history_width", "intake_width"),
    # Only genuine subset shapes (intake <= history) are generated: the former
    # grid emitted an intake-2-over-history-1 combination that runtime-skipped
    # as "not a subset rung", and a permanent skip is exactly what the smoke
    # execution floor exists to refuse.
    [
        (history_width, intake_width)
        for history_width in (1, 2, 7)
        for intake_width in (1, 2)
        if intake_width <= history_width
    ],
)
def test_narrow_rungs_open_over_wider_registered_history(
    tmp_path: Path, history_width: int, intake_width: int
) -> None:
    """A subset rung of any width opens over its registered superset history.

    RC6 proved width 10-over-53; the degenerate widths (a single-model re-drive
    over a one- or two-model history) are where an off-by-one in the membership
    difference would surface.
    """
    history_ids = tuple(f"m_hist_{index}" for index in range(history_width))
    intake_ids = history_ids[:intake_width]
    paths = _ledger_paths(tmp_path)
    _seed_history(paths, history_ids)
    with CanonicalReducer(paths, _subset_context(intake_ids, history_ids)) as reducer:
        assert set(reducer.current_records) <= set(history_ids)
        # Read side serves the whole registered history; write side is pinned.
        outsider = history_ids[-1]
        if outsider not in intake_ids:
            with pytest.raises(ReductionError) as refused:
                reducer.append_attempt(make_attempt(outsider, attempt_id="attempt-outsider"))
            assert outsider in str(refused.value), (
                "the write refusal must name the stable id that is outside intake"
            )


def test_a_rung_identical_to_its_lineage_opens(tmp_path: Path) -> None:
    """Width equality: intake == the registered prior member union."""

    history_ids = tuple(f"m_hist_{index}" for index in range(4))
    paths = _ledger_paths(tmp_path)
    _seed_history(paths, history_ids)
    with CanonicalReducer(paths, _subset_context(history_ids, history_ids)) as reducer:
        assert set(reducer.current_records) == set(history_ids)


def test_a_rung_adding_new_members_alongside_prior_history_opens(tmp_path: Path) -> None:
    """Mixed width: intake carries brand new members the lineage never had.

    The membership difference must be computed against the UNION of intake and
    registered lineage, so new members (no records yet) and prior members
    (records, outside intake) coexist.
    """

    history_ids = tuple(f"m_hist_{index}" for index in range(3))
    intake_ids = (history_ids[0], "m_brand_new_0", "m_brand_new_1")
    paths = _ledger_paths(tmp_path)
    _seed_history(paths, history_ids)
    with CanonicalReducer(paths, _subset_context(intake_ids, history_ids)) as reducer:
        # Construction succeeds: the whole registered history is readable.
        assert set(reducer._current) == set(history_ids)
        # A brand new member is writable; a prior-only member is not.
        with pytest.raises(ReductionError) as refused:
            reducer.append_attempt(make_attempt(history_ids[1], attempt_id="attempt-prior"))
        assert history_ids[1] in str(refused.value)


def test_an_empty_intake_still_opens_read_only_over_registered_history(
    tmp_path: Path,
) -> None:
    """Degenerate width 0: an empty intake may inspect history but never write.

    A zero-member rung is a real operator state (every candidate filtered out).
    Opening must not hard-stop on the membership difference, and every write
    surface must refuse.
    """

    history_ids = tuple(f"m_hist_{index}" for index in range(3))
    paths = _ledger_paths(tmp_path)
    _seed_history(paths, history_ids)
    with CanonicalReducer(paths, _subset_context((), history_ids)) as reducer:
        assert set(reducer._current) == set(history_ids)
        with pytest.raises(ReductionError):
            reducer.append_attempt(make_attempt(history_ids[0], attempt_id="attempt-x"))
        with pytest.raises(ReductionError):
            gate = make_gate([history_ids[0]], gate_id="gate-x")
            gate.pop("ledger_seq", None)
            reducer.append_gate(gate)
        with pytest.raises(ReductionError):
            reducer.append_model(
                reducer.prepare_model(
                    make_model(history_ids[0], accepted=False, status_code="failed:source")
                )
            )


def test_prior_only_members_are_dependency_stale_under_a_narrow_rung(
    tmp_path: Path,
) -> None:
    """Characterization + OPEN QUESTION: the read side stales prior-only members.

    RC6 widened the CONSTRUCTION check, so a subset rung opens over its superset
    history and ``_current`` carries every historical revision. The PROJECTION
    still stales the members outside the active intake, because
    ``_referenced_evidence_repasses`` replays each record's attempts through a
    replay reducer whose ``intake_ids`` is the NARROW current intake -- so the
    reason reads ``attempt stable_id is outside intake``.

    Consequence for a laddered 20k crawl: while a narrow rung is active, every
    prior-corpus terminal disappears from ``current_records``, which is what
    :func:`menagerie.crawler.status.funnel_counts` and the release census
    consume. The direction is FAIL-CLOSED (records are dropped, never wrongly
    published), so this is a reporting-fidelity question and not an honesty
    violation -- but whether K2's currency carve-out was meant to cover the
    membership axis too is an OWNER DECISION, deliberately not silently changed
    here. This test pins today's behavior so either answer is a reviewed one.
    """

    history_ids = tuple(f"m_hist_{index}" for index in range(3))
    paths = _ledger_paths(tmp_path)
    _seed_history(paths, history_ids)
    with CanonicalReducer(paths, _subset_context((history_ids[0],), history_ids)) as reducer:
        assert set(reducer._current) == set(history_ids)
        projection = reducer._current_projection()
        assert set(projection.current_records) == {history_ids[0]}
        for stable_id in history_ids[1:]:
            reason = projection.stale_reasons[stable_id]
            assert "outside intake" in reason and stable_id in reason, (
                "the stale reason must name the member and why it was dropped"
            )


def test_a_stable_id_in_neither_set_hard_stops_at_every_width(tmp_path: Path) -> None:
    """The foreign-records tripwire keeps its exact message at degenerate widths."""

    history_ids = ("m_hist_0", "m_hist_1")
    paths = _ledger_paths(tmp_path)
    _seed_history(paths, history_ids)
    # Register only ONE of the two historical members as lineage.
    with pytest.raises(ReductionError) as refused:
        CanonicalReducer(paths, _subset_context(("m_hist_0",), ("m_hist_0",)))
    message = str(refused.value)
    assert "model revisions exist outside intake" in message
    assert "m_hist_1" in message, "the tripwire must name the foreign stable id"


def test_family_bindings_stay_scoped_to_the_current_intake(tmp_path: Path) -> None:
    """A binding for a prior-only member hard-stops, naming the offending id.

    The membership carve-out deliberately does NOT widen the family-binding
    check: bindings describe the CURRENT intake's variant topology, so a binding
    for a stable id outside it is an authority-assembly defect, not history.
    """

    history_ids = ("m_hist_0", "m_hist_1")
    paths = _ledger_paths(tmp_path)
    _seed_history(paths, history_ids)
    context = _subset_context(("m_hist_0",), history_ids)
    bindings = dict(context.family_bindings)
    bindings["m_hist_1"] = {
        "binding_state": "variant",
        "representative_stable_id": "m_hist_0",
        "variant_token": "variant-a",
    }
    with pytest.raises(ReductionError) as refused:
        CanonicalReducer(paths, replace(context, family_bindings=bindings))
    message = str(refused.value)
    assert "family variant bindings exist outside intake" in message
    assert "m_hist_1" in message
