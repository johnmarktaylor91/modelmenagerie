"""AB-30 matrix: scheduler-quarantine ledger ladder seam (W12, lane w2-scheduler).

The Wave-2 batch extends the scheduler-quarantine ledger with two NEW row
families: ``menagerie.crawler.scheduler-quarantine-void.v2`` (remedied void,
REQUIRED remedy triple ``fixing_sha + test_node + canary``) and
``menagerie.crawler.scheduler-quarantine-drain.v1`` (machine drain, REQUIRED
``fixed_in_tree_sha``). Budget restoration is a closed, replay-derived set
(:func:`budget_restoring_boundaries`): a remedied void ALWAYS restores; a
machine drain restores ONLY when exhaustion held at its ledger position; bare
v1 void and v1 park rows restore nothing but stay READABLE UNCHANGED -- v1
park bytes are prior-era history whose legacy requeue-every-run release is
preserved era-scoped. Releasing an already-released row refuses typed (SD-7);
re-parks emit the ``-reused`` disclosure and charge nothing.

Every cell drives the REAL production readers: the ledger loaders and
validators in :mod:`menagerie.crawler.scheduler_parks` plus the budget and
census derivations, never re-implemented shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.identity import canonical_json_bytes, stamp_row_digest
from menagerie.crawler.scheduler_parks import (
    SCHEDULER_QUARANTINE_DISPOSITION,
    SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION,
    SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT,
    SCHEDULER_QUARANTINE_REMEDY_V1,
    SCHEDULER_QUARANTINE_REUSED_DISPOSITION,
    SCHEDULER_QUARANTINE_SCHEMA_VERSION_V1,
    SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION,
    SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2,
    SchedulerQuarantineError,
    append_scheduler_quarantine_row,
    append_scheduler_quarantine_void,
    budget_restoring_boundaries,
    build_scheduler_quarantine_drain,
    build_scheduler_quarantine_remedied_void,
    build_scheduler_quarantine_row,
    build_scheduler_quarantine_void,
    current_scheduler_quarantine_rows,
    load_scheduler_quarantine_ledger,
    scheduler_drain_census,
    scheduler_quarantine_path,
    scheduler_quarantine_released,
    validate_scheduler_quarantine_void,
)
from menagerie.crawler.tests.mixed_era import assert_era_outcome, classify_reader

pytestmark = pytest.mark.smoke

_LIMIT = SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT


def _v1_park_row(stable_id: str, *, created_at: str) -> dict[str, Any]:
    """Assemble one prior-era (pre-batch v1 schema) park row, digest-stamped.

    There is deliberately no production builder for the v1 shape anymore --
    these bytes exist only as append-only ledger history, so the fixture is
    assembled field-for-field in the historical shape (v1 schema version, v1
    remedy token) and content-addressed exactly as the era's writer did.
    """

    row: dict[str, Any] = {
        "schema_version": SCHEDULER_QUARANTINE_SCHEMA_VERSION_V1,
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "name": stable_id,
        "campaign_id": "c1-mixed-era",
        "run_id": "run-prior-era",
        "machine_id": "machine-1",
        "created_at": created_at,
        "disposition": SCHEDULER_QUARANTINE_DISPOSITION,
        "scheduler_quarantine": {
            "site": "scheduler-round",
            "reason_code": "no-durable-routing-progress",
            "fault_evidence": "scheduler made no durable routing progress",
            "exception_type": None,
            "remedy": SCHEDULER_QUARANTINE_REMEDY_V1,
        },
        "recheck_hint": (
            f"{SCHEDULER_QUARANTINE_REMEDY_V1}: quarantined member is requeued "
            "on the next run"
        ),
    }
    return stamp_row_digest(row)


def _v2_park_row(stable_id: str, *, created_at: str) -> dict[str, Any]:
    """Build one current-era park row through the real production builder."""

    return build_scheduler_quarantine_row(
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        name=stable_id,
        campaign_id="c1-mixed-era",
        run_id="run-1",
        machine_id="machine-1",
        created_at=created_at,
        site="scheduler-round",
        reason_code="no-durable-routing-progress",
        fault_evidence="scheduler made no durable routing progress",
    )


def _park_event(
    stable_id: str,
    *,
    created_at: str,
    disposition: str = "scheduler-member-quarantined",
) -> dict[str, Any]:
    """Build one operational park event in the real campaign-health shape."""

    return {
        "event_kind": "campaign-health",
        "created_at": created_at,
        "details": {
            "disposition": disposition,
            "stable_id": stable_id,
            "work_id": f"work-{stable_id}",
        },
    }


def _ledger(tmp_path: Path) -> Path:
    """Return the canonical quarantine ledger path under a tmp records root."""

    return scheduler_quarantine_path(tmp_path / "records")


def _write_raw(ledger: Path, row: dict[str, Any]) -> bytes:
    """Append one row's canonical bytes as a prior-era writer would have."""

    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json_bytes(row) + b"\n"
    with ledger.open("ab") as handle:
        handle.write(line)
    return line


def _owner(stable_id: str) -> tuple[str, str]:
    """Return the derivation's (stable_id, work_id) key for one member."""

    return (stable_id, f"work-{stable_id}")


# --- Cell 1: prior-era v1 park + v1 void rows: parsed era-scoped, bytes unchanged ---


def test_v1_park_and_void_rows_stay_readable_bytes_unchanged(tmp_path: Path) -> None:
    """Prior-era v1 park and bare-void bytes parse era-scoped and unchanged."""

    ledger = _ledger(tmp_path)
    park = _v1_park_row("m_v1", created_at="2026-08-01T00:00:00Z")
    park_line = _write_raw(ledger, park)

    outcome = classify_reader(lambda: load_scheduler_quarantine_ledger([ledger]))
    assert_era_outcome(
        reader_id="scheduler-quarantine-ledger-load-v1-park",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    rows = outcome[1]
    assert len(rows) == 1
    assert rows[0]["schema_version"] == SCHEDULER_QUARANTINE_SCHEMA_VERSION_V1
    # Bytes unchanged: the validated row round-trips to the exact prior-era
    # canonical line, and the ledger file itself is untouched by the read.
    assert canonical_json_bytes(rows[0]) + b"\n" == park_line
    assert ledger.read_bytes() == park_line

    # The bare v1 void (still a production builder) releases the live row...
    void = build_scheduler_quarantine_void(
        void_of=str(park["row_sha256"]),
        reason="operator cleared the prior-era row",
        voided_by="operator",
        created_at="2026-08-01T01:00:00Z",
    )
    assert void["schema_version"] == SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION
    append_scheduler_quarantine_void(ledger, void)
    outcome = classify_reader(lambda: load_scheduler_quarantine_ledger([ledger]))
    assert_era_outcome(
        reader_id="scheduler-quarantine-ledger-load-v1-void",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    rows = outcome[1]
    assert [row["schema_version"] for row in rows] == [
        SCHEDULER_QUARANTINE_SCHEMA_VERSION_V1,
        SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION,
    ]
    assert current_scheduler_quarantine_rows(rows) == ()

    # ...but restores NOTHING: bare v1 rows are outside the closed
    # budget-restoring set however many of them the history carries.
    events = [_park_event("m_v1", created_at="2026-08-01T00:00:00Z")]
    assert budget_restoring_boundaries(rows, events) == {}


def test_v1_park_row_keeps_legacy_requeue_every_run_release(tmp_path: Path) -> None:
    """Era-scoped semantics: v1 releases unconditionally where v2 holds."""

    ledger = _ledger(tmp_path)
    v1 = _v1_park_row("m_v1", created_at="2026-08-01T00:00:00Z")
    _write_raw(ledger, v1)
    v2 = append_scheduler_quarantine_row(
        ledger, _v2_park_row("m_v2", created_at="2026-08-01T00:00:00Z")
    )
    rows = load_scheduler_quarantine_ledger([ledger])
    assert len(rows) == 2

    # Both members exhausted the v2 budget in event history.
    events: list[dict[str, Any]] = []
    for member in ("m_v1", "m_v2"):
        for minute in range(_LIMIT + 1):
            events.append(
                _park_event(member, created_at=f"2026-08-01T00:0{minute + 1}:00Z")
            )

    # The real eligibility derivation treats the v1 row per the legacy rule:
    # unconditional next-run requeue, however many charged parks exist.
    assert scheduler_quarantine_released(rows[0], events, ledger_rows=rows) is True
    # The same history holds the v2 row: the era boundary is the schema, not
    # the member.
    assert scheduler_quarantine_released(v2, events, ledger_rows=rows) is False


# --- Cell 2: remedied void v2 with the full triple parses and restores budget ---


def test_remedied_void_v2_parses_and_restores_budget(tmp_path: Path) -> None:
    """A void.v2 with the full remedy triple validates and restores budget."""

    ledger = _ledger(tmp_path)
    park = append_scheduler_quarantine_row(
        ledger, _v2_park_row("m_rem", created_at="2026-08-01T00:01:00Z")
    )
    events = [
        _park_event("m_rem", created_at="2026-08-01T00:01:00Z"),
        _park_event("m_rem", created_at="2026-08-01T00:02:00Z"),
    ]
    rows = load_scheduler_quarantine_ledger([ledger])
    assert scheduler_quarantine_released(park, events, ledger_rows=rows) is False

    remedied = build_scheduler_quarantine_remedied_void(
        void_of=str(park["row_sha256"]),
        reason="root-caused: routing seam fixed",
        voided_by="operator",
        created_at="2026-08-01T00:03:00Z",
        fixing_sha="a" * 40,
        test_node=(
            "menagerie/crawler/tests/test_mixed_era_scheduler_quarantine.py::red_pin"
        ),
        canary="canary: replay of the parked generation passed",
    )
    assert remedied["schema_version"] == SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2
    outcome = classify_reader(lambda: append_scheduler_quarantine_void(ledger, remedied))
    assert_era_outcome(
        reader_id="scheduler-quarantine-remedied-void-append",
        outcome=outcome,
        prebatch_expected="parsed",
    )

    # The remedied void ALWAYS restores: the re-park releases on fresh budget.
    repark = append_scheduler_quarantine_row(
        ledger, _v2_park_row("m_rem", created_at="2026-08-01T00:04:00Z")
    )
    events.append(_park_event("m_rem", created_at="2026-08-01T00:04:00Z"))
    rows = load_scheduler_quarantine_ledger([ledger])
    boundaries = budget_restoring_boundaries(rows, events)
    assert boundaries.get(_owner("m_rem")) == "2026-08-01T00:03:00Z"
    assert scheduler_quarantine_released(repark, events, ledger_rows=rows) is True

    # A re-park's `-reused` disclosure charges nothing against that budget.
    events.append(
        _park_event(
            "m_rem",
            created_at="2026-08-01T00:05:00Z",
            disposition=SCHEDULER_QUARANTINE_REUSED_DISPOSITION,
        )
    )
    assert scheduler_quarantine_released(repark, events, ledger_rows=rows) is True

    # Disclosure census segregates the remedied release from the live classes.
    census = scheduler_drain_census(rows, events)
    assert "m_rem::work-m_rem" in census["remedied-void-released"]
    assert "m_rem::work-m_rem" in census["retry-eligible"]


def test_remedied_void_with_partial_remedy_refuses_typed(tmp_path: Path) -> None:
    """The remedy triple is REQUIRED: a partial remedy refuses typed."""

    # Empty member at build time.
    outcome = classify_reader(
        lambda: build_scheduler_quarantine_remedied_void(
            void_of="f" * 64,
            reason="partial remedy",
            voided_by="operator",
            created_at="2026-08-01T00:00:00Z",
            fixing_sha="a" * 40,
            test_node="tests::red_pin",
            canary="   ",
        )
    )
    assert outcome[0] == "refused-typed"
    assert isinstance(outcome[1], SchedulerQuarantineError)
    assert "restores nothing" in str(outcome[1])

    # Missing member at read time: a hand-forged v2 void without the full
    # DX-69 shape refuses through the same production validator.
    forged = stamp_row_digest(
        {
            "schema_version": SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2,
            "void_of": "f" * 64,
            "reason": "forged partial remedy",
            "voided_by": "operator",
            "created_at": "2026-08-01T00:00:00Z",
            "remedy": {"fixing_sha": "a" * 40, "canary": "canary text"},
        }
    )
    outcome = classify_reader(lambda: validate_scheduler_quarantine_void(forged))
    assert outcome[0] == "refused-typed"
    assert "fixing_sha, test_node, canary" in str(outcome[1])


# --- Cell 3: drain.v1 validates; an ineligible drain restores nothing ---


def test_drain_v1_validates_and_ineligible_drain_restores_nothing(tmp_path: Path) -> None:
    """A drain.v1 row parses; eligibility is replay-derived, never granted."""

    ledger = _ledger(tmp_path)

    # Member A: only ONE charged park -- exhaustion did NOT hold.
    park_a = append_scheduler_quarantine_row(
        ledger, _v2_park_row("m_early", created_at="2026-08-01T00:01:00Z")
    )
    events = [_park_event("m_early", created_at="2026-08-01T00:01:00Z")]

    # Member B: exhausted (LIMIT charged parks) before its drain.
    park_b = append_scheduler_quarantine_row(
        ledger, _v2_park_row("m_done", created_at="2026-08-01T00:01:00Z")
    )
    for minute in range(_LIMIT):
        events.append(_park_event("m_done", created_at=f"2026-08-01T00:0{minute + 1}:00Z"))

    ineligible = build_scheduler_quarantine_drain(
        void_of=str(park_a["row_sha256"]),
        reason="premature drain",
        drained_by="operator",
        created_at="2026-08-01T00:10:00Z",
        fixed_in_tree_sha="b" * 40,
    )
    assert ineligible["schema_version"] == SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION
    eligible = build_scheduler_quarantine_drain(
        void_of=str(park_b["row_sha256"]),
        reason="machine drain after exhausted retries",
        drained_by="operator",
        created_at="2026-08-01T00:10:00Z",
        fixed_in_tree_sha="c" * 40,
    )
    outcome = classify_reader(
        lambda: (
            append_scheduler_quarantine_void(ledger, ineligible),
            append_scheduler_quarantine_void(ledger, eligible),
        )
    )
    assert_era_outcome(
        reader_id="scheduler-quarantine-drain-append",
        outcome=outcome,
        prebatch_expected="parsed",
    )

    rows = load_scheduler_quarantine_ledger([ledger])
    boundaries = budget_restoring_boundaries(rows, events)
    # However the ineligible drain got appended, it restores NOTHING: the
    # closed proof rule recomputes exhaustion at the drain's position.
    assert _owner("m_early") not in boundaries
    # The exhausted member's drain restores, so the two are never conflated.
    assert boundaries.get(_owner("m_done")) == "2026-08-01T00:10:00Z"

    # And the re-parked early member is charged its full pre-drain history:
    # park 1 + re-park 2 with no boundary means the generation holds.
    repark_a = append_scheduler_quarantine_row(
        ledger, _v2_park_row("m_early", created_at="2026-08-01T00:11:00Z")
    )
    events.append(_park_event("m_early", created_at="2026-08-01T00:11:00Z"))
    rows = load_scheduler_quarantine_ledger([ledger])
    assert scheduler_quarantine_released(repark_a, events, ledger_rows=rows) is False


# --- Cell 4: unknown row family / schema version refuses typed ---


def test_unknown_row_family_refuses_typed(tmp_path: Path) -> None:
    """A future or foreign schema version never half-parses."""

    ledger = _ledger(tmp_path)
    unknown_family = stamp_row_digest(
        {
            "schema_version": "menagerie.crawler.scheduler-quarantine-park.v9",
            "stable_id": "m_future",
            "work_id": "work-m_future",
            "created_at": "2026-08-01T00:00:00Z",
        }
    )
    _write_raw(ledger, unknown_family)
    outcome = classify_reader(lambda: load_scheduler_quarantine_ledger([ledger]))
    assert outcome[0] == "refused-typed"
    assert isinstance(outcome[1], SchedulerQuarantineError)

    # A future VOID version refuses through the void validator's closed set.
    future_void = stamp_row_digest(
        {
            "schema_version": "menagerie.crawler.scheduler-quarantine-void.v3",
            "void_of": "f" * 64,
            "reason": "future ladder rung",
            "voided_by": "operator",
            "created_at": "2026-08-01T00:00:00Z",
        }
    )
    outcome = classify_reader(lambda: validate_scheduler_quarantine_void(future_void))
    assert outcome[0] == "refused-typed"
    assert "unknown schema" in str(outcome[1])

    # A drain declaring a future drain version refuses identically at the
    # ledger seam (never routed to a laxer validator).
    future_drain = stamp_row_digest(
        {
            "schema_version": "menagerie.crawler.scheduler-quarantine-drain.v2",
            "void_of": "f" * 64,
            "reason": "future drain",
            "drained_by": "operator",
            "created_at": "2026-08-01T00:00:00Z",
            "fixed_in_tree_sha": "d" * 40,
        }
    )
    _write_raw(ledger := _ledger(tmp_path / "second"), future_drain)
    outcome = classify_reader(lambda: load_scheduler_quarantine_ledger([ledger]))
    assert outcome[0] == "refused-typed"
    assert isinstance(outcome[1], SchedulerQuarantineError)


# --- Cell 5 (SD-7): releasing an already-released row refuses typed ---


def test_releasing_already_released_row_refuses_typed(tmp_path: Path) -> None:
    """The prior release stands: a second, different release refuses typed."""

    ledger = _ledger(tmp_path)
    park = append_scheduler_quarantine_row(
        ledger, _v2_park_row("m_sd7", created_at="2026-08-01T00:01:00Z")
    )
    bare = build_scheduler_quarantine_void(
        void_of=str(park["row_sha256"]),
        reason="operator cleared the row without a remedy",
        voided_by="operator",
        created_at="2026-08-01T00:02:00Z",
    )
    append_scheduler_quarantine_void(ledger, bare)

    # Byte-identical re-release is idempotent, never a double release.
    outcome = classify_reader(lambda: append_scheduler_quarantine_void(ledger, bare))
    assert outcome[0] == "parsed"
    assert outcome[1]["row_sha256"] == bare["row_sha256"]

    # A DIFFERENT release of the same park -- here a drain trying to forge a
    # budget restoration out of a bare-voided row -- refuses typed.
    drain = build_scheduler_quarantine_drain(
        void_of=str(park["row_sha256"]),
        reason="drain over a bare-voided row",
        drained_by="operator",
        created_at="2026-08-01T00:03:00Z",
        fixed_in_tree_sha="e" * 40,
    )
    outcome = classify_reader(lambda: append_scheduler_quarantine_void(ledger, drain))
    assert outcome[0] == "refused-typed"
    assert isinstance(outcome[1], SchedulerQuarantineError)
    assert "already released" in str(outcome[1])
    # And the ledger bytes still parse: the refusal appended nothing.
    rows = load_scheduler_quarantine_ledger([ledger])
    assert [row["schema_version"] for row in rows] == [
        park["schema_version"],
        SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION,
    ]
