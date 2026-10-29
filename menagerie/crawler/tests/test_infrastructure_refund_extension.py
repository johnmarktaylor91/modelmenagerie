"""Infrastructure deaths refund the attempt (DECIDED_PLAN item 9, recovery F1).

Two ledgers measured the defect. Pilot ``operational/spend.jsonl``: 82 author
sessions settled ``failed`` after a machine wall under 10 seconds (the
``exited-error`` custody outcome -- the executor exited before any harness
ran) and every one was charged as a model attempt; 12 work generations were
100% such deaths and are ``failed:author / effort-exhausted:author-sessions``
today. The R6b shadow clone: 742 such rows, 116 generations burned, 52/52
members falsely terminalized, root cause an unset
``MENAGERIE_AUTHOR_CLAUDE_BIN`` -> executor exit 78 in ~2 s. 72 of 316
requeue grants exist only to undo this burn.

The classifier's own comment (``classify_author_exit``) already says a session
that died without publishing "is indistinguishable from broken infrastructure"
and that mis-terminalizing every model on a host whose provider CLI is broken
"would be the worse defect" -- yet its settlement charged the attempt. These
tests pin the refund-class EXTENSION that closes that contradiction:

* ``failed-executor-typed-outage`` -- stamped only when the executor exited
  with its typed retryable/unavailable code (75/78), printed its own
  pre-session typed line, published NO payload at the envelope's output path,
  and emitted NO harness session-failure notice. Every input is a
  driver-observed fact; no model text participates.
* the class joins the cross-model storm fuse, so five distinct exit-78 deaths
  inside one window HALT the lane instead of burning 52 allowances.
* machinery faults (``text_corpus.promote_author_root`` ledger conflicts; the
  relative-import authority derivation on vendored packages) classify as
  infrastructure -- never ``session-crashed`` / ``member-fault``.

Money invariants are untouched: provider-attested session weight, the machine
wall floor, and replay-derived accounting all stay. A refunded attempt is
simply not an attempt: the member stays open, nothing terminalizes, nothing
publishes.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from menagerie.crawler import driver_admission as driver_admission_module
from menagerie.crawler import scheduler as scheduler_module
from menagerie.crawler.attempt_owner_derivation import ATTEMPT_TRIPLE_OWNERS
from menagerie.crawler.author_dispatch import (
    AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME,
    AUTHOR_EXECUTOR_UNAVAILABLE_LINE_PREFIX,
    AUTHOR_EXIT_RETRYABLE,
    AUTHOR_EXIT_UNAVAILABLE,
    executor_typed_outage_outcome,
)
from menagerie.crawler.authority import _import_names
from menagerie.crawler.completion_line import (
    AuthorityDerivationError,
    RelativeImportUnbindableError,
)
from menagerie.crawler.driver_admission import (
    _author_lane_failure,
    _paid_session_infrastructure_local_outcome,
    classify_author_exit,
)
from menagerie.crawler.driver_contracts import RetryableOperatorError, WorkItem
from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    WORKER_FAILURE_REASON_OWNERS,
)
from menagerie.crawler.paid_session_accounting import (
    ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES,
    ATTEMPT_REFUND_LOCAL_OUTCOMES,
    LOCAL_OUTCOMES,
)
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    ProviderReconciliation,
    load_paid_session_rows,
)
from menagerie.crawler.recordio import JsonlLedger, LedgerConflictError, scan_jsonl
from menagerie.crawler.scheduler_parks import (
    MACHINERY_FAULT_QUARANTINE_TYPES,
    load_scheduler_quarantine_rows,
    member_quarantine_reason,
    scheduler_quarantine_path,
)
from menagerie.crawler.storm_fuse import (
    CROSS_MODEL_STORM_CLASSIFICATIONS,
    CROSS_MODEL_STORM_DISTINCT_MODELS,
    CrossModelProviderStorm,
    CrossModelStormFuse,
)

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The R6b clone's root cause, byte-exact as the real executor prints it with
#: the CLI pin unset (captured 2026-09-01 by running ``author_executor`` under
#: ``env -i``; ``test_the_real_executor_prints_the_typed_line`` re-derives it).
_R6B_EXIT_78_STDERR = (
    "author executor error: MENAGERIE_AUTHOR_CLAUDE_BIN is unset: the author "
    "harness must be pinned by absolute executable path (W32)\n"
)

#: The pilot's m5717 award-site quarantine (``scheduler-quarantines.jsonl``,
#: 2026-08-22T17:27:19Z): ``exception_type`` LedgerConflictError, ``site``
#: ``award``, ``reason_code`` ``member-fault``. Same message class as the
#: m5520 / m7749 / m279 ``failed:author / session-crashed`` attempt rows.
_M5717_FAULT_EVIDENCE = (
    "conflicting replay for immutable identity "
    "('menagerie.crawler.text-corpus-index.v1', "
    "'sha256:86a457e6e9712099b614bdcc1b5461ebc5bb5db5318f8444d124cc1c6e65df28')"
)


# ---------------------------------------------------------------------------
# rig: the production author retry ladder over a real money authority
# ---------------------------------------------------------------------------


def _authority(tmp_path: Path) -> PaidSessionAuthority:
    """Return one temporary paid-session authority with a deterministic clock."""

    return PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-20T10:35:32Z",
    )


def _author_scope(stable_id: str, work_id: str, attempt_group: str | None) -> dict[str, Any]:
    """Return one author reservation scope carrying both accounting axes."""

    return {
        "campaign_id": "c1-mech",
        "generation_id": "generation-open",
        "lane": "author",
        "dispatch_kind": "author_source",
        "attempt_group": attempt_group,
        "work_generations": ((stable_id, work_id),),
        "gate_kind": None,
        "checker_identity": None,
        "run_id": "run-one",
        "machine_id": "host-one",
    }


def _policy() -> BudgetPolicy:
    """Return an author-shaped policy sized for several sessions per model."""

    return BudgetPolicy(
        quota_sessions=100,
        quota_wall_seconds=100_000.0,
        runaway_sessions=100,
        runaway_wall_seconds=100_000.0,
        reservation_wall_seconds=100.0,
    )


def _constant_accept(key: str) -> ProviderReconciliation:
    """The pilot's observed adapter answer on every sub-10 s death: accepted."""

    return ProviderReconciliation(True, None, f"sha256:receipt-{key[-12:]}")


def _work_item(stable_id: str) -> WorkItem:
    """Duck-typed WorkItem: only the attributes the ladder reads exist."""

    return cast(
        WorkItem,
        SimpleNamespace(stable_id=stable_id, active_work_id=f"work-{stable_id}", granted_attempts=None),
    )


def _ledger() -> JsonlLedger:
    """Duck-typed operational ledger for a rig that never reads past ``records``."""

    return cast(JsonlLedger, SimpleNamespace(records=()))


class _LadderRig(driver_admission_module.AdmissionEnvironmentMixin):
    """The real ladder, settle, admission valve and storm fuse over real money.

    ``_retry_infrastructure_call``, ``_settle_paid_session``,
    ``_admit_author_spend`` and ``_cross_model_storm_fuse`` are inherited
    VERBATIM; only driver-configuration reads are pinned.
    """

    _admit_author_spend = (
        driver_admission_module.AdmissionEnvironmentMixin._admit_author_spend  # type: ignore[assignment]
    )

    def __init__(self, authority: PaidSessionAuthority, records_root: Path) -> None:
        self._authority = authority
        self.waited: list[float] = []
        self.config = SimpleNamespace(
            campaign_id="c1-mech", run_id="run-one", machine_id="host-one"
        )
        self._leg_generation = SimpleNamespace(generation_id="generation-open")
        self.dependencies = SimpleNamespace(
            boundary_hook=lambda *_args: None,
            provider_reconciler=_constant_accept,
            monotonic=time.monotonic,
            clock=lambda: "2026-08-20T10:35:32Z",
        )
        # A REAL records root: a tripped fuse mints the W1 deferral valve row
        # under it, and the D-62 pause sentinel is read from it (absent here).
        self.paths = SimpleNamespace(
            ledgers=SimpleNamespace(models=records_root / "models" / "current-shard.jsonl")
        )

    def _check_shutdown(self, *_args: object, **_kwargs: object) -> None:
        """No shutdown is pending in this rig."""

    def _paid_session_authority(self) -> PaidSessionAuthority:
        return self._authority

    def _wait_before_infrastructure_retry(self, seconds: float) -> None:
        self.waited.append(seconds)

    def _author_no_publication_class(self, _exc: BaseException) -> bool:
        return False

    def _note_author_lane_publication(self, _stable_id: str) -> None:
        """No lane-health bookkeeping in this rig."""


def _exit_78_death(stable_id: str, output_path: Path) -> RetryableOperatorError:
    """Return the R6b exit-78 death exactly as the production classifier types it."""

    with pytest.raises(RetryableOperatorError) as raised:
        classify_author_exit(
            "author",
            stable_id,
            AUTHOR_EXIT_UNAVAILABLE,
            "",
            _R6B_EXIT_78_STDERR,
            output_path=output_path,
        )
    return raised.value


def _settlements(authority: PaidSessionAuthority) -> list[dict[str, Any]]:
    """Return every settlement row in ledger order."""

    return [
        dict(row)
        for row in load_paid_session_rows(authority.path)
        if row["event"] == "session_settled"
    ]


# ---------------------------------------------------------------------------
# 1. the closed outcome and its closed proof rule
# ---------------------------------------------------------------------------


def test_the_vocabulary_grows_by_exactly_one_closed_outcome() -> None:
    """The new outcome is in every closed set that pins the refund vocabulary."""

    assert AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME == "failed-executor-typed-outage"
    assert AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME in ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES
    assert AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME in ATTEMPT_REFUND_LOCAL_OUTCOMES
    assert AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME in LOCAL_OUTCOMES
    # Both item-9 classes are storm-fuse classifications.
    assert AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME in CROSS_MODEL_STORM_CLASSIFICATIONS
    assert "failed-custody-reaped" in CROSS_MODEL_STORM_CLASSIFICATIONS


def test_the_real_executor_prints_the_typed_line_on_the_r6b_root_cause(tmp_path: Path) -> None:
    """The REAL executor with the CLI pin unset exits 78 and prints the pinned prefix.

    Binds the classifier's positive match to the executor's own output rather
    than to a remembered string: if the executor's pre-session refusal line
    ever moves, this fails here instead of silently charging every death.
    """

    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {"campaign_id": "c1-mech", "kind": "author", "stable_id": "m0", "work_id": "work-m0"}
        ),
        encoding="utf-8",
    )
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(_REPO_ROOT),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "menagerie.crawler.author_executor", str(request)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120.0,
        check=False,
    )
    assert completed.returncode == AUTHOR_EXIT_UNAVAILABLE
    lines = [line for line in completed.stderr.splitlines() if line.strip()]
    assert lines and lines[-1].startswith(AUTHOR_EXECUTOR_UNAVAILABLE_LINE_PREFIX), completed.stderr
    assert "MENAGERIE_AUTHOR_CLAUDE_BIN is unset" in lines[-1]
    assert not (tmp_path / "result.json").exists()
    # The pure rule classifies the real output as the typed outage.
    assert (
        executor_typed_outage_outcome(
            completed.returncode, completed.stderr, published_payload=False
        )
        == AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME
    )


def test_the_proof_rule_is_positive_closed_and_conservative() -> None:
    """Every leg of the rule is a driver-observed fact; ambiguity charges."""

    io_line = "author executor io error: [Errno 2] No such file or directory: '/x/claude'\n"
    # The two typed pre-session lines, each bound to its own exit code.
    assert (
        executor_typed_outage_outcome(78, _R6B_EXIT_78_STDERR, published_payload=False)
        == AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME
    )
    assert (
        executor_typed_outage_outcome(75, io_line, published_payload=False)
        == AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME
    )
    # Crossed codes never match: the line proves WHICH refusal the executor took.
    assert executor_typed_outage_outcome(75, _R6B_EXIT_78_STDERR, published_payload=False) is None
    assert executor_typed_outage_outcome(78, io_line, published_payload=False) is None
    # Other exit codes never match, whatever was printed.
    for code in (0, 1, 64, 76, 127, 143):
        assert executor_typed_outage_outcome(code, _R6B_EXIT_78_STDERR, published_payload=False) is None
    # Unknown publication state charges; a published payload charges.
    assert executor_typed_outage_outcome(78, _R6B_EXIT_78_STDERR, published_payload=None) is None
    assert executor_typed_outage_outcome(78, _R6B_EXIT_78_STDERR, published_payload=True) is None
    # A session that RAN -- any attempt-scoped stage/broker line or a harness
    # session-failure notice -- is never a pre-session outage, even beside the line.
    ran = _R6B_EXIT_78_STDERR + "author executor stage1 failed: session-crashed (attempt abc)\n"
    assert executor_typed_outage_outcome(78, ran, published_payload=False) is None
    broker = io_line + "author executor broker failed: forge-rate-limited (attempt abc)\n"
    assert executor_typed_outage_outcome(75, broker, published_payload=False) is None
    notice = json.dumps(
        {
            "type": "result",
            "subtype": "author-session-failure",
            "failure_class": "content-refusal",
            "harness_is_error": False,
            "returncode": 0,
        }
    )
    assert executor_typed_outage_outcome(78, notice + "\n" + _R6B_EXIT_78_STDERR, published_payload=False) is None
    # Producer prose cannot mint the line: the prefix must open the line.
    forged = "  note: author executor error: pretend\n"
    assert executor_typed_outage_outcome(78, forged, published_payload=False) is None
    assert executor_typed_outage_outcome(78, "", published_payload=False) is None


def test_the_classifier_stamps_the_outage_only_without_a_published_payload(tmp_path: Path) -> None:
    """``classify_author_exit`` binds the rule to the envelope's output path."""

    missing = tmp_path / "author" / "result.json"
    death = _exit_78_death("m12591", missing)
    assert _paid_session_infrastructure_local_outcome(death) == AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME
    assert death.paid_session_local_outcome == AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME

    published = tmp_path / "published" / "result.json"
    published.parent.mkdir()
    published.write_text('{"kind": "PROPOSED"}', encoding="utf-8")
    charged = _exit_78_death("m12591", published)
    assert _paid_session_infrastructure_local_outcome(charged) is None

    # Legacy callers that cannot name the output path stay chargeable.
    with pytest.raises(RetryableOperatorError) as raised:
        classify_author_exit("author", "m12591", AUTHOR_EXIT_UNAVAILABLE, "", _R6B_EXIT_78_STDERR)
    assert _paid_session_infrastructure_local_outcome(raised.value) is None

    # An empty file is not a publication (the executor's atomic-rename contract).
    empty = tmp_path / "empty" / "result.json"
    empty.parent.mkdir()
    empty.write_bytes(b"")
    assert (
        _paid_session_infrastructure_local_outcome(_exit_78_death("m12591", empty))
        == AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME
    )


def test_a_session_that_ran_and_crashed_still_charges(tmp_path: Path) -> None:
    """The executor's own stage line proves a session ran: unchanged, chargeable."""

    with pytest.raises(RetryableOperatorError) as raised:
        classify_author_exit(
            "author",
            "m12591",
            AUTHOR_EXIT_RETRYABLE,
            "",
            "author executor stage2 failed: session-crashed (attempt 0f3a)\n",
            output_path=tmp_path / "result.json",
        )
    assert _paid_session_infrastructure_local_outcome(raised.value) is None


# ---------------------------------------------------------------------------
# 2. the R6b shape through the production ladder: refund, not terminal
# ---------------------------------------------------------------------------


def test_r6b_exit_78_deaths_refund_the_attempt_and_leave_the_member_open(tmp_path: Path) -> None:
    """Three R6b deaths then a publication: ONE attempt consumed, three refunded.

    Before: every death settled ``failed`` at attempt weight 1, so three deaths
    consumed the member's allowance and the fourth session never ran -- the
    ``effort-exhausted:author-sessions`` terminal on 52/52 clone members.
    """

    authority = _authority(tmp_path)
    rig = _LadderRig(authority, tmp_path / "records")
    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] <= 3:
            raise _exit_78_death("m12591", tmp_path / "m12591" / "result.json")
        return "authored"

    result = rig._retry_infrastructure_call(  # noqa: SLF001
        operation, admission=("author", _work_item("m12591")), operational=_ledger()
    )
    assert result == "authored"
    assert calls["count"] == 4

    settlements = _settlements(authority)
    assert [row["local_outcome"] for row in settlements] == [
        AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME,
        AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME,
        AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME,
        "completed",
    ]
    # The pilot's exact custody shape on these rows: exited-error, provider accepted.
    assert {row["custody_outcome"] for row in settlements[:3]} == {"exited-error"}
    assert all(row["provider"]["accepted"] is True for row in settlements)

    state = authority.state()
    coordinate = ("c1-mech", "m12591", "work-m12591", "author")
    # Money invariants untouched: every provider-consummated session and its
    # machine wall stays charged to QUOTA and RUNAWAY.
    assert state.quota_by_lane[coordinate].sessions == 4
    assert state.runaway_by_lane[("c1-mech", "generation-open", "author")].sessions == 4
    # Only the content-bearing attempt consumes effort allowance.
    assert state.quota_attempt_groups[coordinate] == 1


def test_five_distinct_exit_78_deaths_trip_the_storm_fuse_and_refuse_the_next_buy(
    tmp_path: Path,
) -> None:
    """The R6b storm shape halts the lane at five models instead of 52 allowances."""

    authority = _authority(tmp_path)
    rig = _LadderRig(authority, tmp_path / "records")
    stable_ids = [f"m{index}" for index in range(CROSS_MODEL_STORM_DISTINCT_MODELS)]

    for stable_id in stable_ids:
        deaths = {"count": 0}

        def operation(stable_id: str = stable_id) -> str:
            deaths["count"] += 1
            if deaths["count"] == 1:
                raise _exit_78_death(stable_id, tmp_path / stable_id / "result.json")
            return "authored"

        if stable_id == stable_ids[-1]:
            # The fifth model's first death completes the correlation; its
            # session retry is the NEXT buy and the fuse refuses it typed.
            with pytest.raises(CrossModelProviderStorm):
                rig._retry_infrastructure_call(  # noqa: SLF001
                    operation,
                    admission=("author", _work_item(stable_id)),
                    operational=_ledger(),
                )
        else:
            assert (
                rig._retry_infrastructure_call(  # noqa: SLF001
                    operation,
                    admission=("author", _work_item(stable_id)),
                    operational=_ledger(),
                )
                == "authored"
            )

    witness = rig._cross_model_storm_fuse().tripped(time.monotonic())  # noqa: SLF001
    assert witness is not None
    assert witness.classification == AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME
    assert witness.stable_ids == tuple(sorted(stable_ids))
    # No model terminalized and no attempt was consumed by any death.
    state = authority.state()
    for stable_id in stable_ids[:-1]:
        assert state.quota_attempt_groups[("c1-mech", stable_id, f"work-{stable_id}", "author")] == 1
    fifth = ("c1-mech", stable_ids[-1], f"work-{stable_ids[-1]}", "author")
    assert state.quota_attempt_groups.get(fifth, 0) == 0
    assert authority.outstanding_spend_ids() == ()


def test_the_pure_fuse_counts_both_item_9_classes() -> None:
    """Direct fuse observation: five distinct models per class trip; four do not."""

    for classification in (AUTHOR_EXECUTOR_TYPED_OUTAGE_LOCAL_OUTCOME, "failed-custody-reaped"):
        fuse = CrossModelStormFuse()
        for index in range(CROSS_MODEL_STORM_DISTINCT_MODELS - 1):
            fuse.observe(stable_id=f"m{index}", lane="author", local_outcome=classification, at=100.0)
        assert fuse.tripped(101.0) is None
        fuse.observe(stable_id="m-last", lane="author", local_outcome=classification, at=100.0)
        witness = fuse.tripped(101.0)
        assert witness is not None and witness.classification == classification


# ---------------------------------------------------------------------------
# 3. machinery faults are infrastructure, never session-crashed / member-fault
# ---------------------------------------------------------------------------


def test_text_corpus_ledger_conflict_is_a_machinery_fault_not_a_session_crash() -> None:
    """m5520 / m7749 / m279: ``failed:author / session-crashed`` on a LedgerConflictError."""

    fault = LedgerConflictError(_M5717_FAULT_EVIDENCE)
    assert _author_lane_failure(fault) == ("runner", "internal-error")
    # The owner of the new pair comes from the WORKER reason map (machinery),
    # which ``_failure_owner`` consults first; the derivation triple table stays
    # disjoint from that map by doctrine, and the historical triple that
    # adjudicates the 200a rows (author / session-crashed) stays machinery too.
    assert WORKER_FAILURE_REASON_OWNERS["internal-error"] == "machinery"
    assert "machinery" in NEVER_MODEL_TERMINAL_OWNERS
    assert (
        "runner",
        "internal-error",
        "menagerie.crawler.recordio.LedgerConflictError",
    ) not in ATTEMPT_TRIPLE_OWNERS
    assert (
        ATTEMPT_TRIPLE_OWNERS[
            ("author", "session-crashed", "menagerie.crawler.recordio.LedgerConflictError")
        ]
        == "machinery"
    )


def test_member_quarantine_reason_is_closed_by_exception_type() -> None:
    """The closed machinery type parks ``non-model-fault``; everything else stays ``member-fault``.

    ``LedgerConflictError`` is deliberately outside the tuple: a member reaching
    a catch-all with published artifact rows can never carry a routing marker
    (RULING W7 clause 3 refused exactly that at the award site when this lane
    first tried it), so its two raise sites classify it where it fires.
    """

    assert MACHINERY_FAULT_QUARANTINE_TYPES == (RelativeImportUnbindableError,)
    assert member_quarantine_reason(LedgerConflictError(_M5717_FAULT_EVIDENCE)) == "member-fault"
    assert (
        member_quarantine_reason(
            RelativeImportUnbindableError(
                "relative import cannot be bound to a lookup package: "
                "/work/m243/author/model/bd3lm/modeling_bd3lm.py"
            )
        )
        == "non-model-fault"
    )
    # A plain derivation failure (a member's unparsable file) is NOT machinery.
    assert member_quarantine_reason(AuthorityDerivationError("executable Python member cannot be parsed")) == "member-fault"
    assert member_quarantine_reason(KeyError("x")) == "member-fault"
    assert member_quarantine_reason(RuntimeError("x")) == "member-fault"


def test_relative_import_on_a_vendored_package_raises_the_typed_machinery_error(tmp_path: Path) -> None:
    """The R6b m243 shape: a vendored PACKAGE file with a relative import.

    The derivation cannot bind ``from .config import X`` when the file sits
    outside every lookup root -- a limit of the crawler's own closure
    derivation, not a fact about the model. The raise keeps its
    ``AuthorityDerivationError`` identity and message for every existing
    ``except`` clause, and gains the typed subclass the quarantine sites read.
    """

    vendored = tmp_path / "author" / "model" / "bd3lm" / "modeling_bd3lm.py"
    vendored.parent.mkdir(parents=True)
    vendored.write_text("from .configuration_bd3lm import BD3LMConfig\n", encoding="utf-8")
    with pytest.raises(RelativeImportUnbindableError) as raised:
        _import_names(vendored, [tmp_path / "elsewhere"])
    assert isinstance(raised.value, AuthorityDerivationError)
    assert str(raised.value) == f"relative import cannot be bound to a lookup package: {vendored}"
    # Beyond-the-root depth is the same class.
    shallow = tmp_path / "root" / "pkg" / "mod.py"
    shallow.parent.mkdir(parents=True)
    shallow.write_text("from ...far import X\n", encoding="utf-8")
    with pytest.raises(RelativeImportUnbindableError):
        _import_names(shallow, [tmp_path / "root"])
    # A bindable relative import derives normally.
    bound = tmp_path / "root2" / "pkg" / "mod.py"
    bound.parent.mkdir(parents=True)
    bound.write_text("from .sibling import Y\n", encoding="utf-8")
    assert "pkg.sibling" in _import_names(bound, [tmp_path / "root2"])


def test_the_env_turn_catch_all_routes_through_the_closed_reason() -> None:
    """The env-turn member loop consults the machinery types; the award step does not.

    The award-site arm keeps the literal ``member-fault``: an award-ready
    member's custody window is never silent, so a routing marker there would
    refuse the close (RULING W7 clause 3).
    """

    scheduler_source = inspect.getsource(scheduler_module.run_pipelined_leg)
    award = scheduler_source.index('site="award"')
    assert 'reason_code="member-fault"' in scheduler_source[award : award + 120]
    assert "member_quarantine_reason(" not in scheduler_source
    admission_source = inspect.getsource(driver_admission_module)
    env_turn = admission_source.index('site="env-turn",\n                                reason_code=')
    assert admission_source[env_turn : env_turn + 200].count("member_quarantine_reason(exc)") == 1


def test_author_stage_ledger_conflict_parks_the_member_instead_of_terminalizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the promote-at-accept ledger conflict parks; no head mints.

    The m5520 / m7749 / m279 mechanism, replayed through the real driver: the
    author publishes, ``_stage_author_result`` reaches the BL14 fetch-once
    promotion, and ``text_corpus.promote_author_root`` raises the corpus
    index's immutable-identity conflict. 200a recorded ``failed:author /
    session-crashed`` -- a death that did not occur -- and burned the member.
    """

    from menagerie.crawler import driver as driver_module
    from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

    def _conflict(*_args: Any, **_kwargs: Any) -> Any:
        raise LedgerConflictError(_M5717_FAULT_EVIDENCE)

    monkeypatch.setattr(driver_module, "promote_author_root", _conflict, raising=True)
    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    result = _driver(tmp_path, snapshot).run()

    paths = _paths(tmp_path, snapshot)
    assert result.status == "complete-with-deferrals"
    assert scan_jsonl(paths.ledgers.models) == [], "a machinery fault minted a model terminal"
    records_root = paths.ledgers.models.parent.parent
    quarantines = load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)])
    assert {row["stable_id"] for row in quarantines} == {stable_id}
    assert {row["scheduler_quarantine"]["reason_code"] for row in quarantines} == {"non-model-fault"}
    parked = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "non-model-fault-parked"
    ]
    assert {event["details"]["stable_id"] for event in parked} == {stable_id}
    assert {event["details"]["owner"] for event in parked} == {"machinery"}
    # Classified at the arm, not caught by the mint-sink backstop.
    assert {event["details"]["sink_guard"] for event in parked} == {False}
    attempts = [
        attempt
        for attempt in scan_jsonl(paths.ledgers.attempts)
        if isinstance(attempt.get("error"), dict)
        and attempt["error"].get("exception_type") == "menagerie.crawler.recordio.LedgerConflictError"
    ]
    assert attempts, "the machinery fault must persist as attempt evidence"
    assert {attempt["error"]["reason_code"] for attempt in attempts} == {"internal-error"}
    assert {attempt["error"]["stage"] for attempt in attempts} == {"runner"}


def test_award_site_ledger_conflict_is_disclosed_and_the_award_stands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m5717's shape: the post-commit corpus conflict discloses; no member disposition.

    The pilot parked the AWARDED member ``member-fault`` at ``site="award"``
    beside its live ``runs`` head. The promotion is post-commit by the award
    seam's own contract ("never terminalize the awarded model"), so the
    machinery fault is now a durable RUNNER_FAILED disclosure: the award stands,
    no quarantine row exists, and the leg closes clean.
    """

    from menagerie.crawler import driver_receipts as driver_receipts_module
    from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

    def _conflict(*_args: Any, **_kwargs: Any) -> Any:
        raise LedgerConflictError(_M5717_FAULT_EVIDENCE)

    monkeypatch.setattr(driver_receipts_module, "promote_author_root", _conflict, raising=True)
    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    result = _driver(tmp_path, snapshot, scheduler="pipelined").run()

    paths = _paths(tmp_path, snapshot)
    assert result.status == "complete"
    heads = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert heads[stable_id]["status"]["code"] == "runs"
    records_root = paths.ledgers.models.parent.parent
    assert not load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)])
    disclosures = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "text-corpus-promotion-conflict"
    ]
    assert len(disclosures) == 1
    details = disclosures[0]["details"]
    assert details["stable_id"] == stable_id
    assert details["owner"] == "machinery"
    assert details["exception_type"] == "menagerie.crawler.recordio.LedgerConflictError"
    assert details["message"] == _M5717_FAULT_EVIDENCE
    assert disclosures[0]["status"] == "failed:runner"
