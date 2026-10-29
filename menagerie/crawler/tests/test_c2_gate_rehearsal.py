"""C2 rehearsal: adversarial recency, identity, and batch-tail coverage for gates.

Tri-lab C2 axis (gates + verdict machinery + W4 reuse internals). Every test
here rehearses a shape the launch road can actually produce -- crash-resume
adjudication interleaves, repair rounds that revert bytes, tampered or torn
ledger tails, family-variant gate trickle -- and pins the fail-closed answer:

* Reuse recency is scoped per model, never per byte-match: when a LATER
  same-kind, same-identity adjudication exists for a model, an older row must
  never replay past it, whatever its bindings say (the metadata scanner
  already refused this; the terminal scanner fell back).
* Unauthenticated same-kind history fails the reuse question closed: a row
  that CLAIMS the requested kind but cannot replay its own self-hash (or
  carries unreadable items) may be a later adjudication whose marking was
  stripped or whose write tore, so the answer is a live session -- never
  "skip it and replay something older".
* The batch accumulator's final-tail proof never strands or pads a session:
  packed batches stay packed, the representative-in-accumulator carve-out
  breaks the family deadlock, and settled-dead siblings stop blocking the
  tail flush.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from menagerie.crawler.constants import METADATA_BATCH_MAX
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    CheckerOutcome,
    DriverConfig,
    EnvironmentBinding,
)
from menagerie.crawler.gate_reuse import (
    plan_metadata_gate_reuse,
    reusable_metadata_gate,
    reusable_terminal_gate,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_gate_verdict_reuse import (
    ReuseAwareChecker,
    _artifact,
    _clear_model_history,
    _identity,
    _prior_metadata_gate,
    _reseal,
    _spend_events,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    AuthorScript,
    FakeChecker,
    ScriptedAuthor,
    ScriptedForward,
    _driver,
    _family_snapshot,
    _paths,
    _snapshot,
)

# Everything here is smoke EXCEPT the three pipelined-driver batch witnesses,
# which carry the heavy mark alone (like the sibling scheduler module) so the
# fail-closed crawler-smoke CI gate stays fast.


# ---------------------------------------------------------------------------
# Terminal reuse recency: the LATEST same-kind adjudication for a model
# decides; a stale row must never replay past it.
# ---------------------------------------------------------------------------


class TaggedTerminalChecker(ReuseAwareChecker):
    """Mint per-leg terminal gate ids so byte-different legs never collide.

    The scripted fake derives its terminal ``gate_id`` from the stable ID
    alone; three legs adjudicating DIFFERENT result bytes would otherwise
    collide on one id -- an artifact of the fake, not of the production id,
    which mixes in a fresh request nonce.
    """

    def __init__(self, tag: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tag = tag
        self.captured_terminal_artifacts: list[AuthorArtifact] = []

    def check_terminal(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        self.captured_terminal_artifacts.append(artifact)
        outcome = super().check_terminal(artifact, work_root, config)
        assert outcome.gate is not None
        gate = dict(outcome.gate)
        gate["gate_id"] = f"{gate['gate_id']}-{self.tag}"
        return CheckerOutcome(gate=_reseal(gate))


def _skip_author(status_code: str) -> ScriptedAuthor:
    """Return an author lane recommending one terminal skip predicate."""

    return ScriptedAuthor(AuthorScript(terminal_outcomes=(("status", status_code),)))


@pytest.mark.smoke
def test_terminal_reuse_never_replays_past_a_later_adjudication_of_moved_bytes(
    tmp_path: Path,
) -> None:
    """A stale terminal verdict must not replay when a later adjudication exists.

    The constructible history: leg A's skip recommendation (result bytes A) is
    REJECTED; leg B re-authors to different bytes B, also REJECTED -- the
    checker's actual last word for this model and work. Leg C then reverts to
    bytes A (a reaped work tree plus an author answering as it first did).
    Replaying the stale row-A rejection would mint ``failed:accuracy-gate``
    from a verdict the checker superseded -- the driver's routing would cite
    row A while the K1 recency kernel independently derives from row B, and a
    live session might simply ACCEPT bytes A (as this one does). The only
    fail-closed answer for leg C is a LIVE session.

    Two composed defenses this scenario documents: the acceptance-echo
    variant (row B accepted) is unreachable through a full re-drive, because
    ``_rehydrate_final_authority`` reconstructs the accepted terminal from
    the surviving ledgers before any author runs; and in THIS fixture the
    stale row-A replay is additionally refused by the reuse-side validation
    replay (the fake's rejected-row evidence does not re-resolve), so the
    strict recency rule itself is RED-pinned at the unit level by
    ``test_terminal_reuse_fails_closed_when_the_latest_row_binds_moved_bytes``.
    """

    import shutil

    from menagerie.crawler.tests.test_slice_f_driver import CheckerScript

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)

    # Leg A: result bytes A, checker rejects -> rejected terminal row for A.
    leg_a = TaggedTerminalChecker("a", script=CheckerScript(terminal_verdict="rejected"))
    result_a = _driver(
        tmp_path, snapshot, author=_skip_author("skipped:not-a-real-NN"), checker=leg_a
    ).run()
    assert result_a.status in {"complete", "terminal-partition-complete"}
    assert leg_a.terminal_calls == 1

    # Leg B: different result bytes B (a different skip predicate), rejected
    # too -- the checker's last word for this model now binds B.
    _clear_model_history(paths)
    shutil.rmtree(paths.work_root / stable_id)
    leg_b = TaggedTerminalChecker("b", script=CheckerScript(terminal_verdict="rejected"))
    result_b = _driver(
        tmp_path, snapshot, author=_skip_author("skipped:no-description"), checker=leg_b
    ).run()
    assert result_b.status in {"complete", "terminal-partition-complete"}
    assert leg_b.terminal_calls == 1

    # Leg C: bytes revert to A -- the work tree is disposable (a reaped host or
    # a cleaned re-drive), so the leg re-authors from scratch and the author
    # answers with bytes A again. Without the work-tree wipe the disposable
    # ``driver-author-artifact.json`` cache would rehydrate B instead; the
    # wipe is what makes the reverted-bytes history real. The latest terminal
    # adjudication for this model binds B, so nothing recorded is current for
    # A: a live session is owed, and this one ACCEPTS.
    _clear_model_history(paths)
    shutil.rmtree(paths.work_root / stable_id)
    leg_c = TaggedTerminalChecker("c")
    result_c = _driver(
        tmp_path, snapshot, author=_skip_author("skipped:not-a-real-NN"), checker=leg_c
    ).run()

    # A LIVE session ran (zero calls would mean the stale row-A rejection
    # replayed -- the original defect this test pins). AG-4 (Wave-2 batch)
    # then composes on top: the live ACCEPT contradicts the SAME checker
    # identity's recorded rejection of the SAME bytes (leg A), which is the
    # m4976 verdict-conflict shape -- the conflict authority appends the
    # disclosure row and PARKS the member instead of silently blessing the
    # flip, and no blind-confirmation session is bought for a conflicted
    # accept. The old contract's silent acceptance of a checker
    # self-contradiction is exactly the bad-pass direction AG-4 closes.
    assert leg_c.terminal_calls == 1, (
        "a stale terminal adjudication replayed past the checker's later last word"
    )
    assert result_c.status == "complete-with-deferrals"
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert stable_id not in models, (
        "a conflicted acceptance must park for review, never mint a terminal head"
    )
    conflict_ledger = (
        paths.ledgers.models.parent.parent
        / "intake-extensions"
        / "checker-verdict-conflicts.jsonl"
    )
    conflict_rows = [
        json.loads(line)
        for line in conflict_ledger.read_text().splitlines()
        if line.strip()
    ]
    assert any(
        row.get("row_kind") == "verdict-conflict" and row.get("stable_id") == stable_id
        for row in conflict_rows
    ), "the same-identity disagreement must settle a durable verdict-conflict row"
    _bought, reused_events = _spend_events(paths)
    assert [e["details"]["gate_kind"] for e in reused_events] == []


def _terminal_rows_and_artifact(
    tmp_path: Path,
) -> tuple[Any, dict[str, Any], AuthorArtifact]:
    """Run one accepted-skip leg; return (paths, persisted terminal row, artifact)."""

    snapshot = _snapshot(tmp_path, count=1)
    checker = TaggedTerminalChecker("unit")
    result = _driver(
        tmp_path, snapshot, author=_skip_author("skipped:not-a-real-NN"), checker=checker
    ).run()
    assert result.status == "complete"
    # AG-4 duplicate-accept (Wave-2 batch): a FRESH terminal ACCEPT buys
    # exactly one more LIVE blind confirmation session over the same
    # question, so the checker is dispatched twice; the confirmation
    # settles on the conflict ledger, never as a second gates-ledger row
    # (test_checker_verdict_conflicts.py pins both directions).
    assert len(checker.captured_terminal_artifacts) == 2
    paths = _paths(tmp_path, snapshot)
    rows = [
        row
        for row in scan_jsonl(paths.ledgers.gates)
        if row.get("gate_kind") == "terminal_disposition"
    ]
    assert len(rows) == 1
    return paths, dict(rows[0]), checker.captured_terminal_artifacts[0]


@pytest.mark.smoke
def test_terminal_reuse_control_replays_the_recorded_acceptance(
    tmp_path: Path, calibrated_checker: dict
) -> None:
    """Control: even the authentic prior terminal row still buys live.

    This is the strongest retirement tripwire: the exact artifact and full
    adjudication identity match, but semantic gate reuse remains disabled. The
    mutation arms below stay pinned for any future deliberate re-arming.
    """

    _paths_unused, row, artifact = _terminal_rows_and_artifact(tmp_path)
    assert reusable_terminal_gate(artifact, [row], checker=_identity()) is None


@pytest.mark.smoke
def test_terminal_reuse_fails_closed_when_the_latest_row_binds_moved_bytes(
    tmp_path: Path,
) -> None:
    """The latest terminal row decides; older rows never replay past it.

    When the checker's newest terminal adjudication for this model binds a
    DIFFERENT author result (a repair moved the bytes, then a later leg
    reverted them), nothing recorded is current: the older byte-matching row
    is superseded history and must not replay. This is the unit-level pin of
    the recency rule -- the driver-level scenario above additionally rides the
    validation-replay defense, which cannot be relied on to catch real rows
    (a genuine rejection's evidence re-resolves from the same frozen bytes).
    """

    _paths_unused, row, artifact = _terminal_rows_and_artifact(tmp_path)
    moved = deepcopy(row)
    moved["gate_id"] = f"{row['gate_id']}-moved"
    moved["items"][0]["terminal_disposition"]["author_result_id"] = "sha256:" + "4" * 64
    moved["items"][0]["terminal_disposition"]["author_result_sha256"] = "sha256:" + "5" * 64
    _reseal(moved)
    moved["ledger_seq"] = int(row.get("ledger_seq", 1)) + 1

    assert reusable_terminal_gate(artifact, [row, moved], checker=_identity()) is None


@pytest.mark.smoke
def test_terminal_reuse_fails_closed_on_a_tampered_later_row(tmp_path: Path) -> None:
    """A later same-kind row with a broken self-hash blocks reuse entirely.

    The tampered tail may BE a later adjudication whose honesty marking was
    stripped or whose write tore; scanning past it to replay an older verdict
    is the recency side-door. Unauthenticated same-kind history means a live
    session, fail-closed.
    """

    _paths_unused, row, artifact = _terminal_rows_and_artifact(tmp_path)
    tampered = deepcopy(row)
    tampered["gate_id"] = f"{row['gate_id']}-tampered"
    tampered["result_envelope_sha256"] = "sha256:" + "9" * 64
    tampered["ledger_seq"] = int(row.get("ledger_seq", 1)) + 1

    assert reusable_terminal_gate(artifact, [row, tampered], checker=_identity()) is None


@pytest.mark.smoke
def test_terminal_reuse_fails_closed_on_unreadable_items_in_a_later_row(
    tmp_path: Path,
) -> None:
    """A later same-kind row whose items cannot be read blocks reuse entirely."""

    _paths_unused, row, artifact = _terminal_rows_and_artifact(tmp_path)
    unreadable = deepcopy(row)
    unreadable["gate_id"] = f"{row['gate_id']}-unreadable"
    unreadable["items"] = "not-a-list"
    _reseal(unreadable)
    unreadable["ledger_seq"] = int(row.get("ledger_seq", 1)) + 1

    assert reusable_terminal_gate(artifact, [row, unreadable], checker=_identity()) is None


_TERMINAL_ITEM_FIELDS = ("stable_id", "work_id", "campaign_root_work_id")
_TERMINAL_DISPOSITION_FIELDS = (
    "author_result_id",
    "author_result_sha256",
    "source_manifest_identity",
    "evidence_identity",
    "license_identity",
)


@pytest.mark.parametrize(
    "owner, field",
    [("item", field) for field in _TERMINAL_ITEM_FIELDS]
    + [("terminal", field) for field in _TERMINAL_DISPOSITION_FIELDS],
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.smoke
def test_terminal_binding_field_mutation_forces_live_session(
    tmp_path: Path, owner: str, field: str
) -> None:
    """Every terminal result-binding component mismatch buys a live session.

    The W4 report enumerates rules 12--14 (result id/sha256, source-manifest,
    evidence, license identities, work and lineage bindings) as reuse
    preconditions; this pins each one individually against the real artifact.
    """

    _paths_unused, row, artifact = _terminal_rows_and_artifact(tmp_path)
    mutated = deepcopy(row)
    target = mutated["items"][0] if owner == "item" else mutated["items"][0]["terminal_disposition"]
    assert field in target
    target[field] = "moved-" + str(target[field])[:16]
    _reseal(mutated)

    assert reusable_terminal_gate(artifact, [mutated], checker=_identity()) is None


# ---------------------------------------------------------------------------
# Metadata reuse: unauthenticated same-kind tails fail the question closed,
# and the verified-hash set is matched as a COMPLETE set.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_metadata_reuse_fails_closed_on_a_tampered_later_row(tmp_path: Path) -> None:
    """A later same-kind metadata row with a broken self-hash blocks reuse."""

    artifact = _artifact(tmp_path)
    older = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")
    tampered = deepcopy(older)
    tampered["gate_id"] = "gate-metadata-tampered"
    tampered["result_envelope_sha256"] = "sha256:" + "8" * 64
    tampered["ledger_seq"] = 2

    assert reusable_metadata_gate((artifact,), [older, tampered], checker=_identity()) is None
    groups, live = plan_metadata_gate_reuse((artifact,), [older, tampered], checker=_identity())
    assert groups == ()
    assert live == (artifact,)


@pytest.mark.smoke
def test_metadata_reuse_fails_closed_on_unreadable_items_in_a_later_row(
    tmp_path: Path,
) -> None:
    """A later same-kind metadata row whose items cannot be read blocks reuse."""

    artifact = _artifact(tmp_path)
    older = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")
    unreadable = deepcopy(older)
    unreadable["gate_id"] = "gate-metadata-unreadable"
    unreadable["items"] = "not-a-list"
    _reseal(unreadable)
    unreadable["ledger_seq"] = 2

    assert reusable_metadata_gate((artifact,), [older, unreadable], checker=_identity()) is None


@pytest.mark.smoke
def test_metadata_verified_hash_set_is_matched_completely(
    tmp_path: Path, calibrated_checker: dict
) -> None:
    """A perfect hash set and every malformed variant each force live.

    Semantic reuse is retired, so the perfect full-identity control must also
    return no reuse. The former rule-8 mutation arms remain as fail-closed
    regression pins for any future deliberate re-arming.
    """

    artifact = _artifact(tmp_path)
    base = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")
    assert reusable_metadata_gate((artifact,), [base], checker=_identity()) is None

    hash_keys = sorted(base["items"][0]["verified_hashes"])
    assert set(hash_keys) >= {"proposal", "source_manifest", "evidence"}
    for key in hash_keys:
        mutated = deepcopy(base)
        mutated["items"][0]["verified_hashes"][key] = "sha256:" + "7" * 64
        _reseal(mutated)
        assert reusable_metadata_gate((artifact,), [mutated], checker=_identity()) is None, (
            f"mutated verified_hashes[{key!r}] must force a live session"
        )

    dropped = deepcopy(base)
    del dropped["items"][0]["verified_hashes"][hash_keys[0]]
    _reseal(dropped)
    assert reusable_metadata_gate((artifact,), [dropped], checker=_identity()) is None

    widened = deepcopy(base)
    widened["items"][0]["verified_hashes"]["surplus"] = "sha256:" + "6" * 64
    _reseal(widened)
    assert reusable_metadata_gate((artifact,), [widened], checker=_identity()) is None


# ---------------------------------------------------------------------------
# Batch tails under trickle arrivals: packed sessions, no deadlock, no
# hostage tails.
# ---------------------------------------------------------------------------


class BatchSizeChecker(FakeChecker):
    """Record the exact size and membership of every bought metadata session.

    ``primary_batch_sizes`` separates first-adjudication batches from the H-5
    duplicate-accept confirmation singletons (every member already
    adjudicated): the batching-shape pins below adjudicate the eager batcher,
    and the confirmation lane is a different spend with its own census.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.batch_sizes: list[int] = []
        self.batch_members: list[tuple[str, ...]] = []

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        self.batch_sizes.append(len(artifacts))
        self.batch_members.append(
            tuple(str(artifact.proposal["stable_id"]) for artifact in artifacts)
        )
        return super().check_metadata(artifacts, work_root, config)

    @property
    def primary_batch_sizes(self) -> list[int]:
        seen: set[str] = set()
        primary: list[int] = []
        for members in self.batch_members:
            if any(stable_id not in seen for stable_id in members):
                primary.append(len(members))
            seen.update(members)
        return primary


class SelectiveImportDeadForward(ScriptedForward):
    """Kill selected models with the worker's exact import-death attempt shape."""

    def __init__(self, dead_ids: frozenset[str]) -> None:
        super().__init__()
        self.dead_ids = dead_ids

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        attempts = [
            dict(attempt)
            for attempt in super().forward(artifact, environment, cold_runs, work_root)
        ]
        if str(artifact.proposal["stable_id"]) not in self.dead_ids:
            return attempts
        from menagerie.crawler.identity import stable_hash
        from menagerie.crawler.tests.conftest import HASH

        failed = attempts[0]
        failed["result"] = "failed"
        failed["stage"] = "import"
        failed["mode"] = None
        failed["worker_receipt"] = dict(failed["worker_receipt"])
        failed["worker_receipt"]["mode"] = None
        failed["worker_receipt"]["constructor_started"] = False
        failed["worker_receipt"]["constructor_completed"] = False
        failed["worker_receipt"]["input_completed"] = False
        failed["worker_receipt"]["forward_started"] = False
        failed["worker_receipt"]["forward_completed"] = False
        failed["error"] = {
            "stage": "import",
            "reason_code": "module-missing",
            "exception_type": "builtins.ModuleNotFoundError",
            "message": "No module named 'synthetic_dead_dependency'",
            "traceback": "Traceback: ModuleNotFoundError",
            "no_traceback_reason": None,
            "native_crash": False,
            "root_cause_fingerprint": HASH,
            "details": {"receipt_error": None},
        }
        failed["raw_award_receipt"] = None
        failed["raw_award_receipt_sha256"] = None
        parent_attestation = dict(failed["parent_attestation"])
        parent_attestation["completion_line_sha256"] = None
        parent_attestation["named_raw_award_receipt_sha256"] = None
        parent_attestation["attestation_sha256"] = stable_hash(
            {key: value for key, value in parent_attestation.items() if key != "attestation_sha256"}
        )
        failed["parent_attestation"] = parent_attestation
        failed["unattested_partial"] = {
            "state": "unattested-partial",
            "stage": "import",
            "reason_code": "module-missing",
            "diagnostic_sha256": None,
        }
        return [failed]


@pytest.mark.heavy
@pytest.mark.skipif(
    METADATA_BATCH_MAX < 2,
    reason=(
        "a multi-item eager batch cannot exist at the calibrated width-1 "
        "production config (every flush is a singleton by construction); this "
        "batching-shape pin re-arms with any future width raise"
    ),
)
def test_pipelined_seven_models_pack_one_full_batch_and_one_tail(tmp_path: Path) -> None:
    """Seven synchronized arrivals buy exactly [5, 2]: packed, then proven tail.

    The width-5 eager batcher must flush a packed batch immediately and the
    short remainder ONLY under the final-tail proof -- never as [5, 1, 1] or a
    padded re-buy. H-5 duplicate-accept confirmations ride separately: every
    confirmation is a singleton re-adjudication of an already-gated member.
    """

    snapshot = _snapshot(tmp_path, count=7)
    checker = BatchSizeChecker()
    result = _driver(tmp_path, snapshot, checker=checker, scheduler="pipelined").run()
    assert result.status == "complete"
    assert sorted(checker.primary_batch_sizes) == [2, 5]


@pytest.mark.heavy
@pytest.mark.skipif(
    METADATA_BATCH_MAX < 2,
    reason=(
        "the variants-share-one-session shape is a multi-item batch and cannot "
        "exist at the calibrated width-1 production config; this pin re-arms "
        "with any future width raise"
    ),
)
def test_pipelined_variant_tail_flush_breaks_the_representative_deadlock(
    tmp_path: Path,
) -> None:
    """The representative flushes alone; its variants then share one session.

    A family variant's only path to gate readiness passes through the
    representative's flush (rep gate -> rep award -> variant author), so the
    final-tail proof must treat an unsettled representative INSIDE the
    accumulator as flushable -- refusing would deadlock -- while the trailing
    variants still batch together instead of trickling one session each.
    """

    snapshot, representative_id, variant_ids = _family_snapshot(tmp_path)
    checker = BatchSizeChecker()
    result = _driver(tmp_path, snapshot, checker=checker, scheduler="pipelined").run()
    assert result.status == "complete"
    assert checker.primary_batch_sizes == [1, len(variant_ids)]


@pytest.mark.heavy
@pytest.mark.skipif(
    METADATA_BATCH_MAX < 2,
    reason=(
        "the survivors-share-one-tail shape is a multi-item batch and cannot "
        "exist at the calibrated width-1 production config; this pin re-arms "
        "with any future width raise"
    ),
)
def test_pipelined_dead_siblings_never_hold_the_tail_flush_hostage(tmp_path: Path) -> None:
    """Import-dead models settle gate-free and release the survivors' tail.

    Three of five models die at import (machine death, zero checker spend);
    the two survivors must flush as ONE tail session once the dead are
    settled -- not wait forever on siblings that can never arrive, and not
    trickle out as two single-item sessions.
    """

    snapshot = _snapshot(tmp_path, count=5)
    dead_ids = frozenset(item.stable_id for item in snapshot.items[:3])
    survivors = {item.stable_id for item in snapshot.items[3:]}
    checker = BatchSizeChecker()
    result = _driver(
        tmp_path,
        snapshot,
        checker=checker,
        forward=SelectiveImportDeadForward(dead_ids),
        scheduler="pipelined",
    ).run()
    # Failed terminals close the run's terminal partition; the survivors' runs
    # keep the leg honest either way.
    assert result.status in {"complete", "terminal-partition-complete"}
    assert checker.primary_batch_sizes == [len(survivors)]
    paths = _paths(tmp_path, snapshot)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    for stable_id in dead_ids:
        assert models[stable_id]["status"]["code"] == "failed:import"
    gate_rows = scan_jsonl(paths.ledgers.gates)
    gated_ids = {
        str(item.get("stable_id"))
        for row in gate_rows
        for item in row.get("items", [])
        if isinstance(item, Mapping)
    }
    assert gated_ids == survivors


# ---------------------------------------------------------------------------
# Cross-generation echoes against the recency kernel: an acceptance in a
# LATER work generation of the same campaign lineage contradicts a stale
# failure claim.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_cross_generation_final_acceptance_refuses_a_stale_cap_claim() -> None:
    """A later-generation acceptance is the lineage's last word.

    ``_derive_gate_failure`` builds its history per campaign lineage across
    work generations; a final accepted round from generation two must refuse a
    cap claim assembled from generation one's rejections, exactly like the
    same-work m10949 shape.
    """

    from menagerie.crawler.authority import AuthorityDerivationError, _derive_gate_failure

    def _row(gate_id: str, seq: int, work_id: str, verdict: str) -> dict[str, Any]:
        accepted = verdict == "accurate"
        return {
            "gate_id": gate_id,
            "gate_kind": "metadata_batch",
            "gate_round": seq,
            "ledger_seq": seq,
            "items": [
                {
                    "stable_id": "m_echo",
                    "work_id": work_id,
                    "campaign_root_work_id": "work-gen-1",
                    "verdict": verdict,
                    "integrity": {"verdict": "accurate"},
                    "rung_check": {"verdict": "accurate"},
                    "field_checks": [],
                    "required_repairs": [] if accepted else ["repair the stale claim"],
                }
            ],
        }

    rejected_one = _row("gate-gen1-r1", 1, "work-gen-1", "inaccurate")
    rejected_two = _row("gate-gen1-r2", 2, "work-gen-1", "inaccurate")
    accepted_late = _row("gate-gen2-r3", 3, "work-gen-2", "accurate")
    with pytest.raises(AuthorityDerivationError, match="final accepted metadata_batch"):
        _derive_gate_failure(
            [rejected_one, rejected_two, accepted_late],
            stable_id="m_echo",
            work_id="work-gen-1",
            stage="accuracy-gate",
        )
