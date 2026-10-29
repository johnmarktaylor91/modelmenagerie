"""A reducer refusal parks the MODEL; it never holes the partition or softens the rule.

Pilot evidence (T1-03 / O4): 19 ``terminal-unrecordable`` events across 8 models.
Only the grant class parked (W3); every OTHER refusal class -- the m9617
env-generation contradiction, the m3669/m10759 cap-not-reached shapes -- left
the model with no record and no claim, so the leg paid its full compute cost and
then died at close with ``PartitionError``, permanently, every leg. The streak
tripwire (3 DISTINCT consecutive models) never fired because successes
interleaved.

The adjudicated fix generalizes the W3 grant-park into a typed TERMINAL REFUSAL
park: when a terminal cannot be recorded (intended append refused AND the
minimal fallback refused), a durable typed row -- stable id + work generation +
refusal class + evidence digest -- claims the model as machine-withheld, the
close completes honestly under a never-"complete" status, and the refusal is
disclosed loudly at refusal time. NOTHING is softened: the reducer refusal
stays a refusal, nothing publishes, and the systemic streak backstop still
stops the campaign at three distinct consecutive unrecordable models.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.driver import (
    TERMINAL_UNRECORDABLE_STREAK_LIMIT,
    TerminalRecordingUnavailable,
)
from menagerie.crawler.driver_admission import (
    _deferral_row_fact_still_holds,
    _persisted_deferral_rows,
)
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_DISPOSITION,
    RefusalParkError,
    append_refusal_park_row,
    append_refusal_park_void,
    build_refusal_park_row,
    build_refusal_park_void,
    load_refusal_park_rows,
    refusal_park_path,
    refusal_park_released,
    validate_refusal_park_row,
)
from menagerie.crawler.state import scan_jsonl
from menagerie.crawler.status import (
    MACHINE_DEFERRAL_DISPOSITIONS,
    PartitionError,
    assert_partition,
    machine_deferral_rejection,
    partition_report,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    ForwardScript,
    InaccurateChecker,
    ScriptedForward,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke

REVISION = "sha256:" + "a" * 64
OTHER_REVISION = "sha256:" + "b" * 64

CAP_NOT_REACHED = "failed:accuracy-gate has not reached its bounded cap"
ENV_CONTRADICTION = "model environment generation contradicts its attempt proof"


def _park_row(**overrides: object) -> dict:
    """Build one valid terminal-refusal park row with optional field overrides."""

    values: dict = {
        "stable_id": "m_refused",
        "work_id": "work-m_refused",
        "name": "Refused",
        "campaign_id": "campaign-1",
        "run_id": "run-1",
        "machine_id": "machine-1",
        "created_at": "2026-08-11T00:00:00Z",
        "refusal_class": "ReductionError",
        "record_error_type": "ReductionError",
        "record_error": CAP_NOT_REACHED,
        "fallback_error_type": "ReductionError",
        "fallback_error": CAP_NOT_REACHED,
        "intended_status_code": "failed:accuracy-gate",
        "intended_reason_code": "inaccurate-cap-exhausted",
        "blocked_parent_revision": None,
        "candidate_sha256": REVISION,
    }
    values.update(overrides)
    return build_refusal_park_row(**values)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The typed row itself: identity, evidence digest, tamper refusal, remedy.
# --------------------------------------------------------------------------


def test_refusal_park_row_roundtrips_and_names_the_remedy() -> None:
    """A built row validates, is partition-claimable, and names the release lane."""

    row = _park_row()
    assert row["disposition"] == REFUSAL_PARK_DISPOSITION
    assert REFUSAL_PARK_DISPOSITION in MACHINE_DEFERRAL_DISPOSITIONS
    assert validate_refusal_park_row(row) == row
    refusal = row["refusal"]
    assert refusal["refusal_class"] == "ReductionError"
    assert refusal["record_error"] == CAP_NOT_REACHED
    assert refusal["intended_status_code"] == "failed:accuracy-gate"
    # The typed park identity the docket demands: stable id + generation +
    # refusal class + evidence digest.
    assert isinstance(refusal["evidence_sha256"], str) and refusal["evidence_sha256"]
    hint = row["recheck_hint"]
    assert "operator" in hint
    assert "void" in hint
    # The remedy is a decision or a fix, never silence and never auto-release.
    assert "root-cause" in hint or "requeue" in hint
    # The partition's fail-closed claim validation accepts the persisted row.
    assert machine_deferral_rejection("m_refused", row) is None
    assert machine_deferral_rejection("m_other", row) is not None


def test_refusal_park_row_rejects_tampering_and_missing_evidence() -> None:
    """Digest, identity, and refusal-evidence validation stay fail-closed."""

    row = _park_row()
    tampered = dict(row)
    tampered["name"] = "Renamed"
    with pytest.raises(RefusalParkError, match="digest"):
        validate_refusal_park_row(tampered)
    # Evidence tampering is caught by the EVIDENCE digest, not only the row's.
    forged = json.loads(json.dumps(row))
    forged["refusal"]["record_error"] = "a different refusal entirely"
    with pytest.raises(RefusalParkError, match="digest"):
        validate_refusal_park_row(forged)
    stripped = {key: value for key, value in row.items() if key != "refusal"}
    with pytest.raises(RefusalParkError, match="missing"):
        validate_refusal_park_row(stripped)
    with pytest.raises(RefusalParkError, match="record_error"):
        _park_row(record_error="")
    with pytest.raises(RefusalParkError, match="refusal_class"):
        _park_row(refusal_class="")
    with pytest.raises(RefusalParkError, match="intended_status_code"):
        _park_row(intended_status_code="")
    with pytest.raises(RefusalParkError, match="stable_id"):
        _park_row(stable_id="")


def test_refusal_park_append_is_idempotent_and_void_reopens(tmp_path: Path) -> None:
    """The same refusal fact replays as a no-op; a void releases and reopens."""

    ledger = refusal_park_path(tmp_path)
    first = append_refusal_park_row(ledger, _park_row())
    # A resume re-derives the SAME refusal with a fresh timestamp and run id:
    # the same fact, so a no-op returning the first persisted row.
    replay = append_refusal_park_row(
        ledger, _park_row(created_at="2026-08-11T01:00:00Z", run_id="run-2")
    )
    assert replay == first
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1

    # A DIFFERENT refusal for the same generation is a new fact; BOTH facts
    # stay live (RULING W7 clause 9: per-row liveness -- the newer fact never
    # silently erases the earlier unvoided one, the O A3 defect verified by
    # execution). The lane's governing selection still keys on the newest.
    changed = append_refusal_park_row(
        ledger,
        _park_row(
            record_error=ENV_CONTRADICTION,
            fallback_error=ENV_CONTRADICTION,
            created_at="2026-08-11T02:00:00Z",
        ),
    )
    assert changed["row_sha256"] != first["row_sha256"]
    current = load_refusal_park_rows([ledger])
    assert [row["row_sha256"] for row in current] == [
        first["row_sha256"],
        changed["row_sha256"],
    ]

    # An operator void releases exactly the row it names; the earlier,
    # never-released refusal fact STAYS live accounting (clause 9: releasing
    # a generation means releasing every live row, each by name).
    void = build_refusal_park_void(
        void_of=str(changed["row_sha256"]),
        reason="root cause fixed; model will be re-driven",
        voided_by="operator",
        created_at="2026-08-11T03:00:00Z",
    )
    append_refusal_park_void(ledger, void)
    assert [row["row_sha256"] for row in load_refusal_park_rows([ledger])] == [first["row_sha256"]]
    append_refusal_park_void(
        ledger,
        build_refusal_park_void(
            void_of=str(first["row_sha256"]),
            reason="root cause fixed; model will be re-driven",
            voided_by="operator",
            created_at="2026-08-11T03:30:00Z",
        ),
    )
    assert load_refusal_park_rows([ledger]) == ()
    # Voiding is idempotent (byte-identical) and never rewrites history.
    append_refusal_park_void(ledger, void)
    # A fresh park after the void is a NEW fact the voided row cannot swallow.
    reopened = append_refusal_park_row(
        ledger,
        _park_row(
            record_error=ENV_CONTRADICTION,
            fallback_error=ENV_CONTRADICTION,
            created_at="2026-08-11T04:00:00Z",
        ),
    )
    assert reopened["row_sha256"] != changed["row_sha256"]
    # A void must reference a persisted row.
    with pytest.raises(RefusalParkError, match="unknown"):
        append_refusal_park_void(
            ledger,
            build_refusal_park_void(
                void_of="sha256:" + "d" * 64,
                reason="nothing to void",
                voided_by="operator",
                created_at="2026-08-11T05:00:00Z",
            ),
        )


# --------------------------------------------------------------------------
# Partition accounting: claim, coexist, release -- all revision-aware.
# --------------------------------------------------------------------------


def test_partition_accounts_refusal_parks_by_revision() -> None:
    """A live park claims an absent model, coexists with its unmoved predecessor,
    and reads as superseded once any newer revision lands."""

    # First-terminal refusal: no predecessor revision existed.
    row = _park_row()
    claiming = partition_report(["m_refused"], [], machine_deferrals={"m_refused": row})
    assert claiming.valid
    assert "m_refused" in claiming.machine_deferred.get(REFUSAL_PARK_DISPOSITION, ())
    assert "m_refused" in claiming.refusal_parked_ids
    assert "m_refused" not in claiming.missing_ids
    # ANY landed record releases a parent-less park.
    landed = {
        "stable_id": "m_refused",
        "record_revision": REVISION,
        "status": {"code": "failed:forward"},
    }
    released = partition_report(["m_refused"], [landed], machine_deferrals={"m_refused": row})
    assert released.valid
    assert "m_refused" in released.superseded_deferral_ids
    assert "m_refused" not in released.refusal_parked_ids
    assert refusal_park_released(row, landed) is True
    assert refusal_park_released(row, None) is False

    # Supersession refusal: the predecessor still governs, so the pair
    # coexists -- the terminal satisfies the partition, the park stays LIVE.
    superseding_row = _park_row(blocked_parent_revision=REVISION)
    held = {
        "stable_id": "m_refused",
        "record_revision": REVISION,
        "status": {"code": "failed:runner"},
    }
    coexisting = partition_report(
        ["m_refused"], [held], machine_deferrals={"m_refused": superseding_row}
    )
    assert coexisting.valid
    assert "m_refused" not in coexisting.superseded_deferral_ids
    assert "m_refused" in coexisting.refusal_parked_ids
    assert refusal_park_released(superseding_row, held) is False
    moved = dict(held, record_revision=OTHER_REVISION)
    consumed = partition_report(
        ["m_refused"], [moved], machine_deferrals={"m_refused": superseding_row}
    )
    assert "m_refused" in consumed.superseded_deferral_ids
    assert "m_refused" not in consumed.refusal_parked_ids
    assert refusal_park_released(superseding_row, moved) is True


def test_partition_still_raises_without_a_valid_park_row() -> None:
    """The tripwire's real target is untouched: no record and no valid claim raises."""

    with pytest.raises(PartitionError, match="m_refused"):
        assert_partition(["m_refused"], [])
    # A tampered park row accounts for nothing -- fail-closed.
    forged = dict(_park_row())
    forged["run_id"] = "forged"
    with pytest.raises(PartitionError, match="m_refused"):
        assert_partition(["m_refused"], [], machine_deferrals={"m_refused": forged})


def test_refusal_park_never_short_circuits_a_resume_attempt(tmp_path: Path) -> None:
    """A persisted refusal park never claims work before it is re-attempted.

    The valves err fail-open toward attempting; a refusal park's fact (the
    recording path refused) can only be re-verified by re-running, so the next
    leg attempts the model normally and the ladder re-parks idempotently if the
    refusal still holds.
    """

    records_root = tmp_path / "records"
    row = append_refusal_park_row(refusal_park_path(records_root), _park_row())
    driver = SimpleNamespace(
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=records_root / "models" / "current.jsonl"),
        ),
        config=SimpleNamespace(machine_id="machine-1"),
    )
    item = SimpleNamespace(route=SimpleNamespace(intent="core"))
    assert _deferral_row_fact_still_holds(driver, item, row) is False  # type: ignore[arg-type]
    # The durable row IS read back for supersession reporting.
    persisted = _persisted_deferral_rows(driver)  # type: ignore[arg-type]
    assert [found["row_sha256"] for found in persisted.get("m_refused", ())] == [row["row_sha256"]]


# --------------------------------------------------------------------------
# The driver ladder: park instead of PartitionError, disclose, release.
# --------------------------------------------------------------------------


def _refuse_model_appends(monkeypatch: pytest.MonkeyPatch, stable_id: str, message: str) -> None:
    """Refuse every canonical model append for ONE model with one message.

    Models the pilot's deterministic non-grant reducer refusals: the intended
    terminal append and the minimal fallback refuse identically (19/19 in the
    pilot), while every other model records normally.
    """

    original = reducer_module.CanonicalReducer.append_model

    def refusing(self: Any, model: Mapping[str, Any]) -> Any:
        """Refuse the target model's appends; delegate everything else."""

        if str(model.get("stable_id")) == stable_id:
            raise reducer_module.ReductionError(message)
        return original(self, model)

    monkeypatch.setattr(reducer_module.CanonicalReducer, "append_model", refusing)


def test_unrecordable_terminal_parks_then_releases_across_legs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap-not-reached refusal class parks, replays idempotently, releases.

    Leg 1: one model's terminal refuses on both rungs -> the model parks and
    the close completes honestly instead of raising PartitionError. Leg 2 (same
    still-broken world) re-derives the same park as a no-op. Leg 3 (refusal
    gone) records the real terminal, releases the park, and closes complete.
    """

    snapshot = _snapshot(tmp_path, count=2)
    failed_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    records_root = paths.ledgers.models.parent.parent
    ledger = refusal_park_path(records_root)

    with monkeypatch.context() as patched:
        _refuse_model_appends(patched, failed_id, CAP_NOT_REACHED)
        first = _driver(
            tmp_path,
            snapshot,
            forward=ScriptedForward(ForwardScript(failed_id=failed_id)),
        ).run()

        # THE fix: a distinct, never-"complete" close instead of PartitionError.
        assert first.status == "terminal-partition-complete-with-refusal-parks"
        rows = load_refusal_park_rows([ledger])
        assert [str(row["stable_id"]) for row in rows] == [failed_id]
        park = rows[0]
        assert park["disposition"] == REFUSAL_PARK_DISPOSITION
        refusal = park["refusal"]
        assert refusal["refusal_class"] == "ReductionError"
        assert refusal["record_error"] == CAP_NOT_REACHED
        assert refusal["fallback_error"] == CAP_NOT_REACHED
        assert refusal["intended_status_code"] == "failed:forward"
        # First terminal: no predecessor revision to name.
        assert refusal["blocked_parent_revision"] is None
        # The refused candidate is identified by digest, so an operator can
        # match the park to the exact rejected append.
        assert isinstance(refusal["candidate_sha256"], str)
        assert machine_deferral_rejection(failed_id, park) is None

        # The refusal stays a refusal: no record was minted for the model,
        # while the healthy sibling recorded normally.
        models = {record["stable_id"] for record in scan_jsonl(paths.ledgers.models)}
        assert failed_id not in models
        assert len(models) == len(snapshot.items) - 1

        # Fail-fast disclosure at refusal time: the terminal-unrecordable
        # event still fires and now NAMES the park it minted.
        events = scan_jsonl(paths.operational_ledger)
        unrecordable = [
            event
            for event in events
            if event.get("event_kind") == OperationalEventKind.TERMINAL_UNRECORDABLE.value
        ]
        assert len(unrecordable) == 1
        details = unrecordable[0]["details"]
        assert details["stable_id"] == failed_id
        assert details["refusal_parked"] is True
        assert details["park_row_sha256"] == park["row_sha256"]

        # The close state names the pending operator decision.
        state = json.loads(Path(paths.driver_state).read_text(encoding="utf-8"))
        assert state["status"] == "terminal-partition-complete-with-refusal-parks"
        pending = state["refusal_parks"]["pending"]
        assert [entry["stable_id"] for entry in pending] == [failed_id]
        assert pending[0]["refusal_class"] == "ReductionError"
        assert pending[0]["row_sha256"] == park["row_sha256"]

        # LEG 2: an identical resume re-derives the SAME fact -- one row, no
        # duplicates, same honest close.
        second = _driver(
            tmp_path,
            snapshot,
            forward=ScriptedForward(ForwardScript(failed_id=failed_id)),
        ).run()
        assert second.status == "terminal-partition-complete-with-refusal-parks"
        assert (
            len([line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()])
            == 1
        )

    # LEG 3: the refusal is gone (root cause fixed). Under WS-1 clause 5 a
    # LIVE park is absolute -- selection excludes the member in every mode
    # until an operator releases or voids it (the step-100 treadmill re-ran
    # 17 park rows' identical failures leg after leg precisely because the
    # ready set ignored them). The row's own recheck hint has said it all
    # along: "nothing is ever released automatically". The operator drain is
    # the void; the retried terminal then records and the clean close clears
    # the disclosure block.
    blocked = _driver(
        tmp_path,
        snapshot,
        forward=ScriptedForward(ForwardScript(failed_id=failed_id)),
    ).run()
    assert blocked.status == "terminal-partition-complete-with-refusal-parks"
    append_refusal_park_void(
        ledger,
        build_refusal_park_void(
            void_of=str(park["row_sha256"]),
            reason="root cause fixed; operator releases the park for retry",
            voided_by="operator",
            created_at="2026-08-16T00:00:00.000000Z",
        ),
    )
    third = _driver(
        tmp_path,
        snapshot,
        forward=ScriptedForward(ForwardScript(failed_id=failed_id)),
    ).run()
    assert third.status == "complete"
    final = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert final[failed_id]["status"]["code"] == "failed:forward"
    for row in load_refusal_park_rows([ledger]):
        assert refusal_park_released(row, final[str(row["stable_id"])]) is True
    state = json.loads(Path(paths.driver_state).read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert "refusal_parks" not in state
    # The released row is never silently dropped: the operator VOID is the
    # durable disclosure (append-only, next to the park it releases), and the
    # governing view holds no live row.
    raw_lines = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [row.get("void_of") is not None for row in raw_lines] == [False, True]
    assert raw_lines[1]["void_of"] == park["row_sha256"]
    assert load_refusal_park_rows([ledger]) == ()


def test_env_generation_contradiction_class_parks_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The m9617 refusal class (env-generation contradiction) also parks."""

    snapshot = _snapshot(tmp_path, count=2)
    failed_id = snapshot.items[0].stable_id
    _refuse_model_appends(monkeypatch, failed_id, ENV_CONTRADICTION)

    result = _driver(
        tmp_path,
        snapshot,
        forward=ScriptedForward(ForwardScript(failed_id=failed_id)),
    ).run()

    assert result.status == "terminal-partition-complete-with-refusal-parks"
    paths = _paths(tmp_path, snapshot)
    rows = load_refusal_park_rows([refusal_park_path(paths.ledgers.models.parent.parent)])
    assert [str(row["stable_id"]) for row in rows] == [failed_id]
    assert rows[0]["refusal"]["record_error"] == ENV_CONTRADICTION


def test_parking_never_clears_the_systemic_streak_backstop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three distinct consecutive unrecordable models still stop the campaign.

    Parking bounds the damage of a PER-MODEL refusal; it must never absorb a
    SYSTEMIC recording failure into silent mass-parking. The evidence for the
    stop -- the parks and the unrecordable events -- is durable BEFORE the raise.
    """

    snapshot = _snapshot(tmp_path, count=6)

    def failing_append(self: Any, model: Mapping[str, Any]) -> Any:
        """Refuse every canonical model append."""

        del self, model
        raise reducer_module.ReductionError("synthetic systemic refusal")

    monkeypatch.setattr(reducer_module.CanonicalReducer, "append_model", failing_append)

    with pytest.raises(TerminalRecordingUnavailable):
        _driver(tmp_path, snapshot, checker=InaccurateChecker()).run()

    paths = _paths(tmp_path, snapshot)
    rows = load_refusal_park_rows([refusal_park_path(paths.ledgers.models.parent.parent)])
    assert len(rows) == TERMINAL_UNRECORDABLE_STREAK_LIMIT
