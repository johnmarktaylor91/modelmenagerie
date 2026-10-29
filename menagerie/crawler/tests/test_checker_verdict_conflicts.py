"""AG-4: checker conflict authority + the terminal duplicate-accept policy.

200a evidence (WS1 first-task (d), 2026-08-24 recount): the ONLY proven checker
nondeterminism is same-prompt disagreement on byte-identical content -- m4976
flipped accept->reject under the IDENTICAL prompt digest -- and the measured
bad-pass floor on the bless direction is 6.7% (sol U-S1). These tests pin the
two AG-4 mechanisms (PLAN_v6 §2 WS1, Wave-2 rotation batch):

* CONFLICT AUTHORITY: a fresh terminal-disposition verdict that DISAGREES with
  any persisted verdict for the same ``(stable_id, author_result_sha256)``
  under the SAME full adjudication identity (the
  :func:`gate_adjudication_identity` 4-tuple) is a typed conflict: a durable
  ``verdict-conflict`` row on the ``intake-extensions`` conflict ledger plus a
  ``checker-verdict-conflict`` park. Cross-identity disagreement (the
  m3222/m8215 shape) is a LAWFUL rotation effect: recorded as an
  ``identity-attributed-flip`` disclosure row, never parked.
* DUPLICATE-ACCEPT: terminal-disposition ACCEPTANCE requires two blind
  consistent accepts -- after a fresh accept the lane buys one more live
  session over the same envelope (never a pool replay, never a reused
  verdict), records the confirmation durably, and a disagreement parks as a
  conflict. Rejections are never duplicated. The second accept is NEVER a
  second gates-ledger row: a duplicate accepted item row would violate the
  exact-membership anti-substitution rule at publication
  (``resolve_exact_gate_item_membership``), so the confirmation lives on the
  conflict ledger, bound to the confirmed row's ``gate_id``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import pytest

from menagerie.crawler.checker_conflicts import (
    CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION,
    CONFLICT_ROW_KINDS,
    DUPLICATE_ACCEPT_STATUSES,
    CheckerVerdictConflictError,
    append_checker_verdict_conflict_row,
    build_duplicate_accept_row,
    build_identity_flip_row,
    build_verdict_conflict_row,
    checker_verdict_conflict_path,
    duplicate_accept_state,
    terminal_acceptance_witnesses,
    load_checker_verdict_conflict_rows,
    terminal_gate_verdict,
    terminal_verdict_disagreements,
    unconfirmed_accepted_terminal_gate,
    validate_checker_verdict_conflict_row,
)
from menagerie.crawler.checker_lane import (
    CHECKER_LANE_FAILURE_CLASSES,
    CHECKER_LANE_NON_LADDER_FAILURE_CLASSES,
)
from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.driver_contracts import DriverConfig, WorkItem
from menagerie.crawler.gate_reuse import CheckerAdjudicationIdentity
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.routing import route_model
from menagerie.crawler.tests.test_checker_lane_failure_scoping import _valve_rows
from menagerie.crawler.tests.test_slice_f_driver import (
    CheckerOutcome,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke

PROMPT = "sha256:" + "a" * 64
OTHER_PROMPT = "sha256:" + "b" * 64
SHA = "sha256:" + "c" * 64
OTHER_SHA = "sha256:" + "d" * 64
ENVELOPE = "sha256:" + "e" * 64

IDENTITY = CheckerAdjudicationIdentity(
    provider="openai", model="gpt-5.6-sol", version="current", prompt_sha256=PROMPT
)


def _td_gate(
    verdict: str,
    *,
    stable_id: str = "m4976",
    work_id: str = "work-m4976",
    lineage: str = "work-m4976",
    gate_id: str = "gate-prior",
    prompt: str = PROMPT,
    model: str = "gpt-5.6-sol",
    sha: str = SHA,
    gate_identity: str = ENVELOPE,
    verdict_source: Optional[str] = None,
) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "gate_kind": "terminal_disposition",
        "gate_id": gate_id,
        "gate_identity": gate_identity,
        "result_envelope_sha256": "sha256:" + "f" * 64,
        "checker": {
            "provider": "openai",
            "model": model,
            "version": "current",
            "prompt_sha256": prompt,
        },
        "items": [
            {
                "stable_id": stable_id,
                "work_id": work_id,
                "campaign_root_work_id": lineage,
                "terminal_disposition": {
                    "verdict": verdict,
                    "author_result_sha256": sha,
                },
            }
        ],
    }
    if verdict_source is not None:
        gate["verdict_source"] = verdict_source
    return gate


# ---------------------------------------------------------------------------
# Conflict detection kernel: same identity = conflict, cross identity = flip.
# ---------------------------------------------------------------------------


def test_same_identity_disagreement_is_a_conflict() -> None:
    """The m4976 exact pair: identical sha + identical prompt digest, flipped."""

    prior = _td_gate("accepted", gate_id="gate-prior")
    fresh = _td_gate("rejected", gate_id="gate-fresh")
    split = terminal_verdict_disagreements(fresh, [prior, fresh])
    assert [row["gate_id"] for row in split["conflicts"]] == ["gate-prior"]
    assert split["flips"] == []


def test_cross_identity_disagreement_is_an_identity_flip_not_a_conflict() -> None:
    """The m3222/m8215 shape: rotation is a lawful cause; the census watches it."""

    prior = _td_gate("rejected", gate_id="gate-terra", model="gpt-5.6-terra")
    fresh = _td_gate("accepted", gate_id="gate-sol")
    split = terminal_verdict_disagreements(fresh, [prior, fresh])
    assert split["conflicts"] == []
    assert [row["gate_id"] for row in split["flips"]] == ["gate-terra"]


def test_agreeing_regate_is_neither_conflict_nor_flip() -> None:
    """The 08-23 sweep shape: 37/38 byte-identical re-gates of prior accepts."""

    prior = _td_gate("accepted", gate_id="gate-prior")
    fresh = _td_gate("accepted", gate_id="gate-fresh")
    split = terminal_verdict_disagreements(fresh, [prior, fresh])
    assert split["conflicts"] == [] and split["flips"] == []


def test_reworked_content_never_compares() -> None:
    """All 4 lifetime reject->accept transitions were author-improved arms."""

    prior = _td_gate("rejected", gate_id="gate-prior", sha=OTHER_SHA)
    fresh = _td_gate("accepted", gate_id="gate-fresh")
    split = terminal_verdict_disagreements(fresh, [prior, fresh])
    assert split["conflicts"] == [] and split["flips"] == []


def test_reused_rows_are_transparent_to_conflict_detection() -> None:
    prior = _td_gate("rejected", gate_id="gate-prior", verdict_source="reused")
    fresh = _td_gate("accepted", gate_id="gate-fresh")
    split = terminal_verdict_disagreements(fresh, [prior, fresh])
    assert split["conflicts"] == [] and split["flips"] == []


def test_terminal_gate_verdict_fails_closed() -> None:
    assert terminal_gate_verdict(_td_gate("cannot-verify")) == "cannot-verify"
    with pytest.raises(CheckerVerdictConflictError):
        terminal_gate_verdict({"gate_kind": "terminal_disposition", "items": []})
    with pytest.raises(CheckerVerdictConflictError):
        terminal_gate_verdict(_td_gate("blessed"))


# ---------------------------------------------------------------------------
# The conflict ledger: closed vocabulary, digest-stamped, idempotent appends.
# ---------------------------------------------------------------------------


def _identity_fields(**overrides: str) -> dict[str, str]:
    fields = {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "version": "current",
        "prompt_sha256": PROMPT,
    }
    fields.update(overrides)
    return fields


def _conflict_row(**overrides: Any) -> dict[str, Any]:
    row = build_verdict_conflict_row(
        stable_id="m4976",
        work_id="work-m4976",
        campaign_id="c1-mech",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-24T00:00:00Z",
        gate_kind="terminal_disposition",
        author_result_sha256=SHA,
        new_gate_id="gate-fresh",
        prior_gate_id="gate-prior",
        new_verdict="rejected",
        prior_verdict="accepted",
        new_identity=_identity_fields(),
        prior_identity=_identity_fields(),
    )
    row = dict(row)
    row.update(overrides)
    return row


def test_conflict_row_vocabulary_is_closed() -> None:
    assert CONFLICT_ROW_KINDS == frozenset(
        {"verdict-conflict", "duplicate-accept-confirmation", "identity-attributed-flip"}
    )
    assert DUPLICATE_ACCEPT_STATUSES == frozenset({"confirmed", "conflict"})
    good = _conflict_row()
    assert validate_checker_verdict_conflict_row(good) == good
    assert good["schema_version"] == CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION
    for mutation in (
        {"row_kind": "advisory-note"},
        {"new_verdict": "blessed"},
        {"prior_verdict": ""},
        {"row_sha256": "sha256:" + "0" * 64},
        {"author_result_sha256": None},
    ):
        with pytest.raises(CheckerVerdictConflictError):
            validate_checker_verdict_conflict_row(_conflict_row(**mutation))


def test_verdict_conflict_row_pins_identity_equality_and_disagreement() -> None:
    """A conflict row REQUIRES same identity + different verdicts; a flip row
    REQUIRES different identities -- the builder and validator both refuse the
    cross-wired shapes, so a lawful rotation flip can never be dressed as a
    same-identity conflict (or vice versa)."""

    with pytest.raises(CheckerVerdictConflictError):
        build_verdict_conflict_row(
            stable_id="m4976",
            work_id="work-m4976",
            campaign_id="c1-mech",
            run_id="run-1",
            machine_id="machine-1",
            created_at="2026-08-24T00:00:00Z",
            gate_kind="terminal_disposition",
            author_result_sha256=SHA,
            new_gate_id="gate-fresh",
            prior_gate_id="gate-prior",
            new_verdict="rejected",
            prior_verdict="accepted",
            new_identity=_identity_fields(),
            prior_identity=_identity_fields(model="gpt-5.6-terra"),
        )
    with pytest.raises(CheckerVerdictConflictError):
        build_identity_flip_row(
            stable_id="m4976",
            work_id="work-m4976",
            campaign_id="c1-mech",
            run_id="run-1",
            machine_id="machine-1",
            created_at="2026-08-24T00:00:00Z",
            gate_kind="terminal_disposition",
            author_result_sha256=SHA,
            new_gate_id="gate-fresh",
            prior_gate_id="gate-prior",
            new_verdict="rejected",
            prior_verdict="accepted",
            new_identity=_identity_fields(),
            prior_identity=_identity_fields(),
        )
    with pytest.raises(CheckerVerdictConflictError):
        _conflict_row_same_verdict()


def _conflict_row_same_verdict() -> dict[str, Any]:
    return build_verdict_conflict_row(
        stable_id="m4976",
        work_id="work-m4976",
        campaign_id="c1-mech",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-24T00:00:00Z",
        gate_kind="terminal_disposition",
        author_result_sha256=SHA,
        new_gate_id="gate-fresh",
        prior_gate_id="gate-prior",
        new_verdict="accepted",
        prior_verdict="accepted",
        new_identity=_identity_fields(),
        prior_identity=_identity_fields(),
    )


def _confirmation_row(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "stable_id": "m4976",
        "work_id": "work-m4976",
        "campaign_id": "c1-mech",
        "run_id": "run-1",
        "machine_id": "machine-1",
        "created_at": "2026-08-24T00:00:00Z",
        "gate_kind": "terminal_disposition",
        "author_result_sha256": SHA,
        "gate_id": "gate-accepted",
        "gate_identity": ENVELOPE,
        # Distinct by default: a confirmation is a SECOND blind session and
        # the builder refuses an equal pair (P5 fable-F16).
        "confirmation_gate_identity": "sha256:" + "5" * 64,
        "confirmation_verdict": "accepted",
        "confirmation_identity": _identity_fields(),
        "confirmation_result_envelope_sha256": "sha256:" + "f" * 64,
    }
    kwargs.update(overrides)
    return build_duplicate_accept_row(**kwargs)


def test_duplicate_accept_row_refuses_a_self_witnessing_gate_identity() -> None:
    """P5 fable-F16: distinct-session independence is row law, not call-path.

    RED at bd5c7837b: builder and v2 validator both accepted
    ``confirmation_gate_identity == gate_identity`` -- the K=2 door's
    independence lived only in the driver's buy-a-second-session call path.
    """

    with pytest.raises(CheckerVerdictConflictError, match="distinct gate"):
        _confirmation_row(confirmation_gate_identity=ENVELOPE)


def test_duplicate_accept_row_status_is_derived_never_declared() -> None:
    confirmed = _confirmation_row()
    assert confirmed["status"] == "confirmed"
    conflicted = _confirmation_row(confirmation_verdict="rejected")
    assert conflicted["status"] == "conflict"
    # Envelope digests lawfully differ across sessions (fresh deadline/output
    # path per dispatch): the row RECORDS both, and question equality is the
    # caller's pre-condition over terminal_question_binding.
    recorded = _confirmation_row(confirmation_gate_identity="sha256:" + "9" * 64)
    assert recorded["confirmation_gate_identity"] == "sha256:" + "9" * 64
    # A hand-flipped status refuses at validation: status must re-derive.
    forged = dict(_confirmation_row(confirmation_verdict="rejected"))
    forged["status"] = "confirmed"
    with pytest.raises(CheckerVerdictConflictError):
        validate_checker_verdict_conflict_row(forged)


def test_append_is_idempotent_first_wins(tmp_path: Path) -> None:
    ledger = checker_verdict_conflict_path(tmp_path)
    assert ledger == tmp_path / "intake-extensions" / "checker-verdict-conflicts.jsonl"
    first = append_checker_verdict_conflict_row(ledger, _confirmation_row())
    replay = append_checker_verdict_conflict_row(
        ledger, _confirmation_row(created_at="2026-08-25T00:00:00Z")
    )
    assert replay == first, "re-parking the same confirmation must be a no-op"
    other = append_checker_verdict_conflict_row(ledger, _conflict_row())
    rows = load_checker_verdict_conflict_rows((ledger,))
    assert [row["row_kind"] for row in rows] == [
        "duplicate-accept-confirmation",
        "verdict-conflict",
    ]
    assert other in rows
    assert duplicate_accept_state("gate-accepted", rows) == "confirmed"
    assert duplicate_accept_state("gate-unknown", rows) is None


def test_checker_verdict_conflict_is_a_closed_lane_failure_class() -> None:
    """The park rides the existing checker-lane valve, ladder-exempt: a
    conflict is a decision-minted park (like disclosure debt), never a
    batch-halving ladder outcome."""

    assert "checker-verdict-conflict" in CHECKER_LANE_FAILURE_CLASSES
    assert "checker-verdict-conflict" in CHECKER_LANE_NON_LADDER_FAILURE_CLASSES


# ---------------------------------------------------------------------------
# The resume guard: an unconfirmed accept resumes its confirmation, never a
# duplicate first buy (the crash window between accept append and settle).
# ---------------------------------------------------------------------------


def test_unconfirmed_accept_probe_binds_identity_content_and_state() -> None:
    accepted = _td_gate("accepted", gate_id="gate-accepted")
    probe = dict(
        stable_id="m4976",
        work_id="work-m4976",
        author_result_sha256=SHA,
        checker_identity=IDENTITY,
    )
    assert unconfirmed_accepted_terminal_gate([accepted], (), **probe) == accepted
    # No identity => fail-open to a live buy, never a guess.
    no_identity = dict(probe)
    no_identity["checker_identity"] = None
    assert unconfirmed_accepted_terminal_gate([accepted], (), **no_identity) is None
    # A settled confirmation (either status) never resumes.
    confirmed = _confirmation_row(gate_id="gate-accepted")
    assert unconfirmed_accepted_terminal_gate([accepted], (confirmed,), **probe) is None
    # A different identity, different content, or non-accept never matches.
    rotated = _td_gate("accepted", gate_id="gate-accepted", model="gpt-5.6-terra")
    assert unconfirmed_accepted_terminal_gate([rotated], (), **probe) is None
    reworked = _td_gate("accepted", gate_id="gate-accepted", sha=OTHER_SHA)
    assert unconfirmed_accepted_terminal_gate([reworked], (), **probe) is None
    rejected = _td_gate("rejected", gate_id="gate-rejected")
    assert unconfirmed_accepted_terminal_gate([rejected], (), **probe) is None


def test_distinct_consistent_accept_is_the_second_blind_witness() -> None:
    """A persisted same-identity, same-envelope accept from a SEPARATE session
    (distinct gate_id -- e.g. a review-sweep re-gate) already completes the
    two-blind-accepts pair; a reused replay or the row itself never does."""

    fresh = _td_gate("accepted", gate_id="gate-fresh")
    witness = _td_gate("accepted", gate_id="gate-sweep-regate")
    assert [row["gate_id"] for row in terminal_acceptance_witnesses(fresh, [witness, fresh])] == [
        "gate-sweep-regate"
    ]
    # Self, reused, rejected, rotated-identity, and foreign-envelope rows
    # never witness.
    replay = _td_gate("accepted", gate_id="gate-replayed", verdict_source="reused")
    rotated = _td_gate("accepted", gate_id="gate-rotated", model="gpt-5.6-terra")
    other_envelope = _td_gate(
        "accepted", gate_id="gate-other-env", gate_identity="sha256:" + "9" * 64
    )
    rejected = _td_gate("rejected", gate_id="gate-rejected")
    assert (
        terminal_acceptance_witnesses(fresh, [fresh, replay, rotated, other_envelope, rejected])
        == []
    )
    assert terminal_acceptance_witnesses(rejected, [witness]) == []


# ---------------------------------------------------------------------------
# Driver wiring: the shared terminal front half enforces both mechanisms.
# ---------------------------------------------------------------------------


class SequenceTerminalChecker:
    """Scripted terminal lane rendering one verdict per live session."""

    def __init__(
        self,
        verdicts: Sequence[str],
        *,
        stable_id: str,
        work_id: str,
        models: Optional[Sequence[str]] = None,
        identity: Optional[CheckerAdjudicationIdentity] = IDENTITY,
    ) -> None:
        self.verdicts = list(verdicts)
        self.models = list(models) if models is not None else None
        self.terminal_calls = 0
        self._stable_id = stable_id
        self._work_id = work_id
        self._identity = identity

    def adjudication_identity(
        self, gate_kind: str, config: DriverConfig
    ) -> Optional[CheckerAdjudicationIdentity]:
        del gate_kind, config
        return self._identity

    def check_terminal(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        del artifact, work_root, config
        self.terminal_calls += 1
        verdict = self.verdicts.pop(0)
        model = self.models.pop(0) if self.models else "gpt-5.6-sol"
        return CheckerOutcome(
            gate=_td_gate(
                verdict,
                stable_id=self._stable_id,
                work_id=self._work_id,
                lineage=self._work_id,
                gate_id=f"gate-td-{self.terminal_calls}",
                # Every real dispatch mints a fresh envelope (deadline/output
                # path), so per-call identities model production; the K=2 row
                # builder refuses a self-witnessing equal pair (P5 fable-F16).
                gate_identity="sha256:" + format(self.terminal_calls, "064x"),
                model=model,
            )
        )


class _RecordingReducer:
    """Gate-records stub: append is recorded, never schema-validated."""

    def __init__(self, persisted: Sequence[dict[str, Any]] = ()) -> None:
        self.gate_records = list(persisted)

    def append_gate(self, gate: dict[str, Any]) -> SimpleNamespace:
        self.gate_records.append(gate)
        return SimpleNamespace(record=gate)


def _seam(
    tmp_path: Path,
    *,
    verdicts: Sequence[str],
    persisted: Sequence[dict[str, Any]] = (),
    models: Optional[Sequence[str]] = None,
    identity: Optional[CheckerAdjudicationIdentity] = IDENTITY,
    conflict_parking: bool = True,
) -> dict[str, Any]:
    """Run the shared terminal front half once; return every observable."""

    snapshot = _snapshot(tmp_path, count=1)
    intake = snapshot.items[0]
    stable_id = intake.stable_id
    work_id = f"work-{stable_id}"
    checker = SequenceTerminalChecker(
        verdicts, stable_id=stable_id, work_id=work_id, models=models, identity=identity
    )
    driver = _driver(tmp_path, snapshot, checker=checker)
    # Paid-session admission requires the durable leg generation the full
    # driver run opens; the direct-seam rig binds a synthetic one.
    driver._leg_generation = SimpleNamespace(generation_id="leg-gen-test")  # noqa: SLF001
    item = WorkItem(intake, route_model(intake.to_model_requirements("torch")))
    persisted_rows = [
        _td_gate(
            row["items"][0]["terminal_disposition"]["verdict"],
            stable_id=stable_id,
            work_id=work_id,
            lineage=work_id,
            gate_id=row["gate_id"],
            gate_identity=row["gate_identity"],
            model=row["checker"]["model"],
            sha=row["items"][0]["terminal_disposition"]["author_result_sha256"],
        )
        for row in persisted
    ]
    reducer = _RecordingReducer(persisted_rows)
    artifact = SimpleNamespace(
        campaign_root_work_id=work_id,
        proposal={"work_id": work_id},
        author_result=SimpleNamespace(
            binding=SimpleNamespace(stable_id=stable_id, work_id=work_id, result_sha256=SHA)
        ),
    )
    paths = _paths(tmp_path, snapshot)
    with JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        gate, pause = driver._obtain_terminal_disposition_gate(  # noqa: SLF001
            item, artifact, reducer, operational, conflict_parking=conflict_parking
        )
    records_root = paths.ledgers.models.parent.parent
    ledger = checker_verdict_conflict_path(records_root)
    conflict_rows = load_checker_verdict_conflict_rows((ledger,)) if ledger.is_file() else ()
    return {
        "gate": gate,
        "pause": pause,
        "calls": checker.terminal_calls,
        "conflict_rows": conflict_rows,
        "valve_rows": _valve_rows(tmp_path, snapshot),
        "stable_id": stable_id,
        "reducer": reducer,
    }


def test_fresh_terminal_accept_requires_a_second_blind_accept(tmp_path: Path) -> None:
    """Two blind consistent accepts: the lane buys the confirmation live."""

    result = _seam(tmp_path, verdicts=["accepted", "accepted"])
    assert result["calls"] == 2, "a fresh accept must buy one blind confirmation session"
    assert result["gate"] is not None and result["pause"] is None
    rows = result["conflict_rows"]
    assert [row["row_kind"] for row in rows] == ["duplicate-accept-confirmation"]
    assert rows[0]["status"] == "confirmed"
    assert rows[0]["gate_id"] == result["gate"]["gate_id"]
    assert result["valve_rows"] == []
    # The confirmation is durable on the conflict ledger, never a second
    # gates-ledger row (exact-membership anti-substitution).
    appended = [
        gate for gate in result["reducer"].gate_records if gate["gate_id"].startswith("gate-td-")
    ]
    assert len(appended) == 1


def test_duplicate_accept_disagreement_parks_and_discloses(tmp_path: Path) -> None:
    result = _seam(tmp_path, verdicts=["accepted", "rejected"])
    assert result["calls"] == 2
    assert result["gate"] is None and result["pause"] is None
    rows = result["conflict_rows"]
    assert [row["row_kind"] for row in rows] == ["duplicate-accept-confirmation"]
    assert rows[0]["status"] == "conflict"
    valve = result["valve_rows"]
    assert len(valve) == 1
    assert valve[0]["checker_lane"]["failure_class"] == "checker-verdict-conflict"
    assert valve[0]["checker_lane"]["gate_kind"] == "terminal_disposition"


def test_rejection_is_never_duplicated(tmp_path: Path) -> None:
    result = _seam(tmp_path, verdicts=["rejected"])
    assert result["calls"] == 1, "rejections are never duplicated (B1-XQ5 spend shape)"
    assert result["gate"] is not None and result["pause"] is None
    assert result["conflict_rows"] == ()
    assert result["valve_rows"] == []


def test_same_identity_history_conflict_parks_before_confirmation(tmp_path: Path) -> None:
    """The m4976 pair live: a fresh same-identity flip parks, no second buy.

    Conflict authority runs BEFORE duplicate-accept: a fresh accept that
    contradicts a persisted same-identity rejection parks as a conflict and
    never buys a confirmation for the contested acceptance.
    """

    prior = _td_gate("rejected", gate_id="gate-prior")
    result = _seam(tmp_path, verdicts=["accepted"], persisted=[prior])
    assert result["calls"] == 1
    assert result["gate"] is None and result["pause"] is None
    rows = result["conflict_rows"]
    assert [row["row_kind"] for row in rows] == ["verdict-conflict"]
    assert rows[0]["prior_gate_id"] == "gate-prior"
    assert rows[0]["new_verdict"] == "accepted"
    valve = result["valve_rows"]
    assert len(valve) == 1
    assert valve[0]["checker_lane"]["failure_class"] == "checker-verdict-conflict"


def test_confirmation_on_a_different_question_parks(tmp_path: Path) -> None:
    """A confirmation session that adjudicated a DIFFERENT question (here: a
    rotated identity mid-pair) can never settle the acceptance -- the lane
    parks fail-closed instead of comparing two different adjudications."""

    result = _seam(
        tmp_path,
        verdicts=["accepted", "accepted"],
        models=["gpt-5.6-sol", "gpt-5.6-terra"],
    )
    assert result["calls"] == 2
    assert result["gate"] is None and result["pause"] is None
    assert result["conflict_rows"] == ()
    valve = result["valve_rows"]
    assert len(valve) == 1
    assert valve[0]["checker_lane"]["gate_kind"] == "terminal_disposition"


def test_terminal_question_binding_ignores_envelope_bytes() -> None:
    """Same item + content + identity = same question, whatever the envelope
    digest (each dispatch mints a fresh deadline/output path)."""

    from menagerie.crawler.checker_conflicts import terminal_question_binding

    first = _td_gate("accepted", gate_id="gate-a", gate_identity="sha256:" + "1" * 64)
    second = _td_gate("accepted", gate_id="gate-b", gate_identity="sha256:" + "2" * 64)
    assert terminal_question_binding(first) == terminal_question_binding(second)
    rotated = _td_gate("accepted", gate_id="gate-c", model="gpt-5.6-terra")
    assert terminal_question_binding(first) != terminal_question_binding(rotated)
    reworked = _td_gate("accepted", gate_id="gate-d", sha=OTHER_SHA)
    assert terminal_question_binding(first) != terminal_question_binding(reworked)


def test_sweep_regate_accept_satisfies_the_pair_without_a_third_session(
    tmp_path: Path,
) -> None:
    """The identity-less sweep shape: a fresh accept agreeing with a persisted
    same-identity accept from a separate session (distinct gate_id) is already
    the two-blind-accepts pair -- no confirmation session, no settlement row.

    The witness law binds the byte-identical envelope (equal ``gate_identity``
    with a distinct ``gate_id`` -- the frozen-request re-gate shape), so the
    persisted prior carries exactly the identity the fresh session mints.
    """

    prior = _td_gate(
        "accepted",
        gate_id="gate-prior-session",
        gate_identity="sha256:" + format(1, "064x"),
    )
    result = _seam(tmp_path, verdicts=["accepted"], persisted=[prior], identity=None)
    assert result["calls"] == 1, "a witnessed acceptance must never buy a third session"
    assert result["gate"] is not None and result["pause"] is None
    assert result["gate"]["gate_id"] == "gate-td-1"
    assert result["conflict_rows"] == ()
    assert result["valve_rows"] == []


def test_resumed_confirmation_disagreement_parks(tmp_path: Path) -> None:
    """A persisted UNCONFIRMED accept whose resumed blind confirmation
    disagrees settles ``conflict`` and parks -- the m4976 flip caught at the
    confirmation seam instead of a duplicate first buy."""

    prior = _td_gate("accepted", gate_id="gate-prior")
    result = _seam(tmp_path, verdicts=["rejected"], persisted=[prior])
    assert result["calls"] == 1, "the resume path buys only the confirmation"
    assert result["gate"] is None and result["pause"] is None
    rows = result["conflict_rows"]
    assert [row["row_kind"] for row in rows] == ["duplicate-accept-confirmation"]
    assert rows[0]["gate_id"] == "gate-prior"
    assert rows[0]["status"] == "conflict"
    valve = result["valve_rows"]
    assert len(valve) == 1
    assert valve[0]["checker_lane"]["failure_class"] == "checker-verdict-conflict"


def test_sweep_path_records_conflicts_without_parking(tmp_path: Path) -> None:
    """The review-backlog sweep contract (``conflict_parking=False``): the
    same-identity disagreement lands as a durable conflict row -- the
    instability census -- but the verdict flows on to the sweep's own
    review-held supersession machinery instead of a park."""

    prior = _td_gate("accepted", gate_id="gate-prior")
    result = _seam(
        tmp_path,
        verdicts=["rejected"],
        persisted=[prior],
        identity=None,
        conflict_parking=False,
    )
    assert result["calls"] == 1
    assert result["gate"] is not None and result["pause"] is None
    rows = result["conflict_rows"]
    assert [row["row_kind"] for row in rows] == ["verdict-conflict"]
    assert result["valve_rows"] == []


def test_cross_identity_flip_is_recorded_never_parked(tmp_path: Path) -> None:
    """Rotation disagreement is disclosure, not a park (m3222/m8215 shape)."""

    prior = _td_gate("accepted", gate_id="gate-terra", model="gpt-5.6-terra")
    result = _seam(tmp_path, verdicts=["rejected"], persisted=[prior])
    assert result["calls"] == 1
    assert result["gate"] is not None and result["pause"] is None
    rows = result["conflict_rows"]
    assert [row["row_kind"] for row in rows] == ["identity-attributed-flip"]
    assert rows[0]["prior_gate_id"] == "gate-terra"
    assert result["valve_rows"] == []


def test_unconfirmed_accept_resumes_confirmation_without_rebuying(tmp_path: Path) -> None:
    """The crash window: accept appended, confirmation never settled. On
    re-entry the lane buys ONLY the confirmation and the persisted accepted
    row stays the one verdict of record."""

    prior = _td_gate("accepted", gate_id="gate-prior")
    result = _seam(tmp_path, verdicts=["accepted"], persisted=[prior])
    assert result["calls"] == 1, "the resume path must never re-buy the first session"
    assert result["gate"] is not None and result["pause"] is None
    assert result["gate"]["gate_id"] == "gate-prior"
    rows = result["conflict_rows"]
    assert [row["row_kind"] for row in rows] == ["duplicate-accept-confirmation"]
    assert rows[0]["gate_id"] == "gate-prior"
    assert rows[0]["status"] == "confirmed"
    appended = [
        gate for gate in result["reducer"].gate_records if gate["gate_id"].startswith("gate-td-")
    ]
    assert appended == [], "no new gates-ledger row may be minted on the resume path"
