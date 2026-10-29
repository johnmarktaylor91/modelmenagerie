"""Gate-identity verdict reuse is RETIRED: every dispatch buys a live session.

BL-10 (h): semantic verdict reuse recorded ZERO lifetime hits in 344 pilot gate
rows while adding four hot-path hooks and a second verdict provenance whose key
is not semantically closed. The live feature is disabled
(:data:`menagerie.crawler.gate_reuse.GATE_VERDICT_REUSE_RETIRED`); transport-
level completed-result adoption (frozen weather-request replay) is unaffected.

What this module now pins, permanently:

* **The retirement itself.** Even a PERFECT full-adjudication-identity match --
  the exact scenario the feature was built for -- returns "no reuse" from every
  entry point, and the driver buys a live session and records only
  ``checker-session-bought`` (never ``checker-verdict-reused``).
* **Every fail-closed refusal** the feature had (identity mismatch, tampered
  row, absent lane identity, accepted-prior) still refuses -- these are the
  regression suite for any future deliberate re-arming.
* **Historical read-side machinery.** Marked replay rows recorded by earlier
  code must keep validating, appending, routing, and refusing to masquerade as
  live adjudications: retirement disables MINTING, never reading history.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

import menagerie.crawler.driver as driver_module
from menagerie.crawler.constants import METADATA_BATCH_MAX
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.driver_contracts import CheckerOutcome, DriverConfig
from menagerie.crawler.driver_models import (
    _normalize_gate_generation,
    _prepare_ledger_record,
    _without_ledger_fields,
)
from menagerie.crawler.gate_reuse import (
    GATE_REUSED_FROM_FIELD,
    GATE_VERDICT_SOURCE_FIELD,
    GATE_VERDICT_SOURCE_REUSED,
    CheckerAdjudicationIdentity,
    plan_metadata_gate_reuse,
    reusable_fidelity_gate,
    reusable_metadata_gate,
)
from menagerie.crawler.gates import GateRoute, emit_gate_records, route_metadata_gate
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.paid_session_lifecycle import load_paid_session_rows
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.tests.conftest import (
    HASH,
    make_author_proposal,
    make_authority_context,
    make_proposed_artifact,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    AuthorScript,
    CheckerScript,
    ScriptedAuthor,
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke

CONFIG = DriverConfig(run_id="run-test", machine_id="machine-test")


def _identity(config: DriverConfig = CONFIG) -> CheckerAdjudicationIdentity:
    """Return the adjudication identity the scripted checker fakes stamp."""

    return CheckerAdjudicationIdentity(
        provider="openai",
        model=config.checker_model,
        version=config.checker_version,
        prompt_sha256=driver_module._checker_prompt_hash(),
    )


class CountingChecker(ScriptedChecker):
    """Scripted checker that counts terminal sessions like the other lanes."""

    def __init__(self, *, quota: bool = False, script: Optional[CheckerScript] = None) -> None:
        super().__init__(quota=quota, script=script)
        self.terminal_calls = 0

    def check_terminal(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        self.terminal_calls += 1
        return super().check_terminal(artifact, work_root, config)


class ReuseAwareChecker(CountingChecker):
    """Scripted checker that also advertises its adjudication identity."""

    def adjudication_identity(
        self, gate_kind: str, config: DriverConfig
    ) -> CheckerAdjudicationIdentity:
        """Advertise the exact identity this fake stamps onto its gates."""

        del gate_kind
        return _identity(config)


class NoSessionChecker(ReuseAwareChecker):
    """Refuse to sell any checker session: every dispatch is a test failure."""

    def check_metadata(
        self, artifacts: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        raise AssertionError("metadata session bought where a reused verdict was required")

    def check_fidelity(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        raise AssertionError("fidelity session bought where a reused verdict was required")

    def check_terminal(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        raise AssertionError("terminal session bought where a reused verdict was required")


def _artifact(tmp_path: Path, stable_id: str = "m_example") -> Any:
    """Build one proposed author artifact whose identities are self-coherent."""

    proposal = make_author_proposal(stable_id)
    manifest = {"manifest_sha256": HASH, "sources": []}
    return make_proposed_artifact(proposal, manifest, tmp_path / stable_id)


def _prior_metadata_gate(
    tmp_path: Path, artifacts: tuple[Any, ...], *, verdict: str = "accurate"
) -> dict[str, Any]:
    """Produce one persisted-shaped prior metadata gate echoing the artifacts."""

    script = CheckerScript(metadata_verdict=verdict) if verdict != "accurate" else None
    checker = ScriptedChecker(script=script)
    gate = checker.check_metadata(artifacts, tmp_path, CONFIG).gate
    assert gate is not None
    persisted = dict(deepcopy(gate))
    persisted["ledger_seq"] = 1
    persisted["payload_sha256"] = HASH
    # Persisted rows carry a coherent self-hash (generation normalization
    # recomputes it); the scripted fake's post-seal mutations leave it stale.
    return _reseal(persisted)


def _reseal(gate: dict[str, Any]) -> dict[str, Any]:
    """Recompute one gate's result-envelope self-hash after a test mutation."""

    gate["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in gate.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    return gate


def _ledger_paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths below one records root."""

    records = tmp_path / "records"
    return LedgerPaths(
        models=records / "models" / "current-shard.jsonl",
        attempts=records / "attempts" / "local.jsonl",
        gates=records / "gates" / "current-shard.jsonl",
    )


# ---------------------------------------------------------------------------
# RETIREMENT: a PERFECT full adjudication identity match still buys live.
# ---------------------------------------------------------------------------


def test_retirement_constant_is_armed() -> None:
    """Re-arming reuse is a deliberate reviewed change, not a drive-by flip."""

    from menagerie.crawler.gate_reuse import GATE_VERDICT_REUSE_RETIRED

    assert GATE_VERDICT_REUSE_RETIRED is True


def test_perfect_identity_match_still_buys_live_metadata_session(tmp_path: Path) -> None:
    """The strongest retirement tripwire: a full-identity match returns NO reuse.

    This is byte-for-byte the scenario the feature existed for -- the exact
    evidence payload identity, checker provider/model/version/prompt, component
    identities, kind, and schema all match the recorded rejection -- and under
    retirement it must STILL buy a live session. If this ever replays again,
    someone re-armed semantic reuse without the reviewed re-arming change.
    """

    artifact = _artifact(tmp_path)
    prior = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")

    assert reusable_metadata_gate((artifact,), [prior], checker=_identity()) is None
    groups, live = plan_metadata_gate_reuse((artifact,), [prior], checker=_identity())
    assert groups == ()
    assert live == (artifact,)


def test_accepted_prior_is_never_replayed(tmp_path: Path) -> None:
    """An accepted matching adjudication is upstream-skip territory, not reuse.

    If an accepted item legitimately re-reaches dispatch it is the rung-9
    write-refusing class or a repair reverted to previously accepted bytes;
    both must buy a live session, and replaying would also duplicate accepted
    item bytes on the ledger, which exact-membership resolution refuses.
    """

    artifact = _artifact(tmp_path)
    prior = _prior_metadata_gate(tmp_path, (artifact,))

    assert reusable_metadata_gate((artifact,), [prior], checker=_identity()) is None
    groups, live = plan_metadata_gate_reuse((artifact,), [prior], checker=_identity())
    assert groups == ()
    assert live == (artifact,)


def _historical_replay_row(tmp_path: Path, artifact: Any, prior: dict[str, Any]) -> dict[str, Any]:
    """Mint a marked replay row the way PRE-RETIREMENT code did.

    Retirement disables the live entry points, but ledgers recorded before it
    may carry marked replay rows; the read-side machinery must keep handling
    them. The internal builder is used directly, exactly as the retired entry
    points used it.
    """

    from menagerie.crawler.driver_models import _artifact_lineage
    from menagerie.crawler.gate_reuse import _build_reused_gate

    item = deepcopy(dict(prior["items"][0]))
    item["work_id"] = artifact.proposal["work_id"]
    item["campaign_root_work_id"] = _artifact_lineage(artifact)
    return _build_reused_gate(prior, [item], gate_kind="metadata_batch")


def test_historical_replay_row_survives_normalization_and_append(tmp_path: Path) -> None:
    """A PRE-RETIREMENT marked replay row still validates, appends, and cannot masquerade."""

    artifact = _artifact(tmp_path)
    stable_id = str(artifact.proposal["stable_id"])
    prior = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")
    reused = _historical_replay_row(tmp_path, artifact, prior)

    normalized = _normalize_gate_generation(reused, [prior], (stable_id,))
    assert normalized["gate_round"] == 2
    assert normalized[GATE_VERDICT_SOURCE_FIELD] == GATE_VERDICT_SOURCE_REUSED
    prepared = _prepare_ledger_record(normalized, 2)
    records = emit_gate_records(prepared)
    assert len(records) == 1

    with CanonicalReducer(_ledger_paths(tmp_path), make_authority_context([stable_id])) as reducer:
        reducer.append_gate(_without_ledger_fields(prior))
        result = reducer.append_gate(_without_ledger_fields(prepared))
        assert result.appended
        appended = result.record
        assert appended[GATE_VERDICT_SOURCE_FIELD] == GATE_VERDICT_SOURCE_REUSED
        assert appended[GATE_REUSED_FROM_FIELD]["gate_identity"] == prior["gate_identity"]
        # The current-proof self-hash covers the honesty marking, so a replayed
        # row cannot later masquerade as a live adjudication by dropping it.
        from menagerie.crawler.authority import load_current_gate_proof

        assert load_current_gate_proof(appended) is appended
        stripped = {
            key: value
            for key, value in appended.items()
            if key not in {GATE_VERDICT_SOURCE_FIELD, GATE_REUSED_FROM_FIELD}
        }
        assert (
            stable_hash(
                {
                    key: value
                    for key, value in stripped.items()
                    if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
                }
            )
            != appended["result_envelope_sha256"]
        )


def test_rows_without_reuse_fields_stay_valid(tmp_path: Path) -> None:
    """Additive schema: prior rows without the new fields remain schema-valid."""

    artifact = _artifact(tmp_path)
    prior = _prior_metadata_gate(tmp_path, (artifact,))
    assert GATE_VERDICT_SOURCE_FIELD not in prior
    validate_payload(prior)


# ---------------------------------------------------------------------------
# Fail-closed: EVERY identity component mismatch forces a full live session.
# ---------------------------------------------------------------------------


def _mutate_checker_model(gate: dict[str, Any]) -> None:
    gate["checker"]["model"] = "some-other-model"


def _mutate_checker_version(gate: dict[str, Any]) -> None:
    gate["checker"]["version"] = "next"


def _mutate_checker_provider(gate: dict[str, Any]) -> None:
    gate["checker"]["provider"] = "anthropic"


def _mutate_prompt(gate: dict[str, Any]) -> None:
    gate["checker"]["prompt_sha256"] = "sha256:" + "b" * 64


def _mutate_schema_component(gate: dict[str, Any]) -> None:
    gate["author_result_schema_identity"] = "sha256:" + "c" * 64


def _mutate_dispatcher(gate: dict[str, Any]) -> None:
    gate["dispatcher_identity"] = "sha256:" + "d" * 64


def _mutate_gate_kind(gate: dict[str, Any]) -> None:
    gate["gate_kind"] = "fidelity"


def _mutate_schema_version(gate: dict[str, Any]) -> None:
    gate["schema_version"] = "menagerie.crawler.gate.v2"


def _mutate_payload_hash(gate: dict[str, Any]) -> None:
    gate["items"][0]["verified_hashes"]["proposal"] = "sha256:" + "e" * 64


def _mutate_evidence_hash(gate: dict[str, Any]) -> None:
    gate["items"][0]["verified_hashes"]["evidence"] = "sha256:" + "f" * 64


def _mutate_vet_identity(gate: dict[str, Any]) -> None:
    gate["items"][0]["vet_identity"] = "sha256:" + "1" * 64


def _mutate_work_id(gate: dict[str, Any]) -> None:
    gate["items"][0]["work_id"] = "work-somebody-else"


def _mutate_rung(gate: dict[str, Any]) -> None:
    gate["items"][0]["rung_check"]["selected_rung"] = "R3_PORT"


def _mutate_fidelity_identity(gate: dict[str, Any]) -> None:
    gate["items"][0]["fidelity_identity"] = "sha256:" + "2" * 64


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_checker_model,
        _mutate_checker_version,
        _mutate_checker_provider,
        _mutate_prompt,
        _mutate_schema_component,
        _mutate_dispatcher,
        _mutate_gate_kind,
        _mutate_schema_version,
        _mutate_payload_hash,
        _mutate_evidence_hash,
        _mutate_vet_identity,
        _mutate_work_id,
        _mutate_rung,
        _mutate_fidelity_identity,
    ],
    ids=lambda mutate: mutate.__name__.removeprefix("_mutate_"),
)
def test_identity_component_mismatch_forces_live_session(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    """One mutated adjudication-identity component refuses reuse, fail-closed."""

    artifact = _artifact(tmp_path)
    prior = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")
    mutate(prior)
    _reseal(prior)

    assert reusable_metadata_gate((artifact,), [prior], checker=_identity()) is None
    groups, live = plan_metadata_gate_reuse((artifact,), [prior], checker=_identity())
    assert groups == ()
    assert live == (artifact,)


def test_tampered_prior_row_self_hash_forces_live_session(tmp_path: Path) -> None:
    """An unreadable/tampered prior row (broken self-hash) is never reused."""

    artifact = _artifact(tmp_path)
    prior = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")
    prior["result_envelope_sha256"] = "sha256:" + "9" * 64

    assert reusable_metadata_gate((artifact,), [prior], checker=_identity()) is None


def test_absent_adjudication_identity_forces_live_session(tmp_path: Path) -> None:
    """A lane that cannot name its checker identity never reuses anything."""

    artifact = _artifact(tmp_path)
    prior = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")

    assert reusable_metadata_gate((artifact,), [prior], checker=None) is None
    groups, live = plan_metadata_gate_reuse((artifact,), [prior], checker=None)
    assert groups == ()
    assert live == (artifact,)


def test_write_refusing_accepted_item_is_never_reused(tmp_path: Path) -> None:
    """The rung-9 poisoned gate class buys a fresh session instead of replaying.

    An all-accurate prior item that would REFUSE at canonical write (empty
    ``checked_source_ids``) must not be replayed: reusing it would recreate the
    poisoned-acceptance loop the write-time validator exists to break. It is
    refused by the accepted-verdict rule, which covers the whole class.
    """

    artifact = _artifact(tmp_path)
    prior = _prior_metadata_gate(tmp_path, (artifact,))
    for check in prior["items"][0]["field_checks"]:
        check["checked_source_ids"] = []
    _reseal(prior)

    assert reusable_metadata_gate((artifact,), [prior], checker=_identity()) is None


def test_accepted_latest_match_never_falls_back_to_older_rejection(tmp_path: Path) -> None:
    """When the LATEST matching adjudication accepted, nothing older replays."""

    artifact = _artifact(tmp_path)
    rejected = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")
    accepted = _prior_metadata_gate(tmp_path, (artifact,))
    accepted["ledger_seq"] = 2

    assert reusable_metadata_gate((artifact,), [rejected, accepted], checker=_identity()) is None


# ---------------------------------------------------------------------------
# Planner: reuse groups by owning prior row; everything else stays live.
# ---------------------------------------------------------------------------


def test_planner_returns_everything_live_under_retirement(tmp_path: Path) -> None:
    """The planner partitions NOTHING into reuse groups: all work stays live."""

    matched_a = _artifact(tmp_path, "m_reuse_a")
    matched_b = _artifact(tmp_path, "m_reuse_b")
    fresh = _artifact(tmp_path, "m_fresh")
    prior = _prior_metadata_gate(tmp_path, (matched_a, matched_b), verdict="inaccurate")

    groups, live = plan_metadata_gate_reuse(
        (matched_a, matched_b, fresh), [prior], checker=_identity()
    )
    assert groups == ()
    assert live == (matched_a, matched_b, fresh)


def test_latest_matching_prior_row_still_buys_live(tmp_path: Path) -> None:
    """Even the LATEST perfectly matching adjudication never replays."""

    artifact = _artifact(tmp_path)
    first = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")
    second = deepcopy(first)
    second["gate_id"] = "gate-metadata-later"
    second["gate_identity"] = "sha256:" + "3" * 64
    second["items"][0]["required_repairs"] = ["repair the LATER unsupported claim"]
    second["items"][0]["field_checks"][0]["required_repair"] = "repair the later claim"
    _reseal(second)
    second["ledger_seq"] = 2

    assert reusable_metadata_gate((artifact,), [first, second], checker=_identity()) is None


# ---------------------------------------------------------------------------
# Budgets and grants: a reused REJECTED verdict still routes through the
# existing bounded-repair lanes, and a repaired payload buys a live session.
# ---------------------------------------------------------------------------


def test_historical_replay_rejection_still_routes_to_repair(tmp_path: Path) -> None:
    """A PRE-RETIREMENT replayed rejection row still routes through bounded repair."""

    artifact = _artifact(tmp_path)
    stable_id = str(artifact.proposal["stable_id"])
    prior = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")
    reused = _historical_replay_row(tmp_path, artifact, prior)

    normalized = _normalize_gate_generation(reused, [prior], (stable_id,))
    prepared = _prepare_ledger_record(normalized, 2)
    decisions = route_metadata_gate(prepared, {stable_id: 0}, max_repairs=2)
    assert decisions[0].route is GateRoute.REQUEUE_NEXT_BATCH
    assert decisions[0].repair_count == 1
    assert not decisions[0].canonical_write_allowed


def test_repaired_payload_is_a_moved_identity_and_buys_live(tmp_path: Path) -> None:
    """A repair that changed the author result moves the payload identity."""

    artifact = _artifact(tmp_path)
    prior = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")

    repaired_proposal = deepcopy(artifact.proposal)
    repaired_proposal["proposal_sha256"] = "sha256:" + "4" * 64
    repaired = make_proposed_artifact(
        repaired_proposal, {"manifest_sha256": HASH, "sources": []}, tmp_path / "m_example"
    )

    assert reusable_metadata_gate((repaired,), [prior], checker=_identity()) is None


# ---------------------------------------------------------------------------
# Fidelity gates reuse through the same identity discipline.
# ---------------------------------------------------------------------------


def _fidelity_artifact(tmp_path: Path, stable_id: str = "m_example") -> Any:
    proposal = make_author_proposal(stable_id)
    proposal["fidelity_identity"] = HASH
    return make_proposed_artifact(
        proposal, {"manifest_sha256": HASH, "sources": []}, tmp_path / stable_id
    )


def _prior_fidelity_gate(tmp_path: Path, artifact: Any, *, rejected: bool = True) -> dict[str, Any]:
    script = CheckerScript(reject_fidelity=True) if rejected else None
    gate = ScriptedChecker(script=script).check_fidelity(artifact, tmp_path, CONFIG).gate
    assert gate is not None
    persisted = dict(deepcopy(gate))
    persisted["ledger_seq"] = 1
    persisted["payload_sha256"] = HASH
    return _reseal(persisted)


def test_perfect_identity_fidelity_match_still_buys_live(tmp_path: Path) -> None:
    """Retirement covers the fidelity lane: a full-identity match never replays."""

    artifact = _fidelity_artifact(tmp_path)
    prior = _prior_fidelity_gate(tmp_path, artifact)

    assert reusable_fidelity_gate(artifact, [prior], checker=_identity()) is None


def test_accepted_fidelity_prior_is_never_replayed(tmp_path: Path) -> None:
    """Accepted fidelity adjudications stay upstream-skip territory."""

    artifact = _fidelity_artifact(tmp_path)
    prior = _prior_fidelity_gate(tmp_path, artifact, rejected=False)

    assert reusable_fidelity_gate(artifact, [prior], checker=_identity()) is None


def test_fidelity_identity_mismatch_forces_live_session(tmp_path: Path) -> None:
    """A moved fidelity identity is a moved adjudication input: no reuse."""

    artifact = _fidelity_artifact(tmp_path)
    prior = _prior_fidelity_gate(tmp_path, artifact)
    prior["items"][0]["fidelity_identity"] = "sha256:" + "5" * 64
    _reseal(prior)

    assert reusable_fidelity_gate(artifact, [prior], checker=_identity()) is None


def test_metadata_row_never_serves_a_fidelity_gate(tmp_path: Path) -> None:
    """Gate kind is part of the adjudication identity."""

    artifact = _fidelity_artifact(tmp_path)
    prior = _prior_metadata_gate(tmp_path, (artifact,), verdict="inaccurate")

    assert reusable_fidelity_gate(artifact, [prior], checker=_identity()) is None


# ---------------------------------------------------------------------------
# Driver integration: sessions are only bought where no recorded adjudication
# replays, and the operational surface counts both sides.
# ---------------------------------------------------------------------------


def _spend_events(paths: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (sessions-bought, verdicts-reused) operational spend events."""

    events = scan_jsonl(paths.operational_ledger)
    bought = [
        {
            "details": {
                "gate_kind": row.get("gate_kind"),
                "batch_size": len(row["axis"]["work_generations"]),
                "checker_spend_id": row["spend_id"],
            }
        }
        for row in load_paid_session_rows(paths.paid_session_ledger)
        if row.get("event") == "session_reserved" and row.get("lane") == "checker"
    ]
    reused = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "checker-verdict-reused"
    ]
    return bought, reused


def test_verbatim_resubmission_buys_live_under_retirement(tmp_path: Path) -> None:
    """A repair that changed nothing buys a fresh live session under retirement.

    The scripted author resubmits byte-identical proposals on repair, and the
    scripted checker rejects the model every round. With reuse retired, the
    second round is a LIVE session; the repeated root-cause fast-terminal then
    fires on the two identical live rejections and the model terminalizes. No
    row is marked replayed and no ``checker-verdict-reused`` event exists.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    checker = ReuseAwareChecker(
        script=CheckerScript(metadata_verdict="inaccurate", selected_id=stable_id)
    )
    result = _driver(tmp_path, snapshot, checker=checker).run()

    # The single model terminalizes on the repeated root cause, so the run's
    # terminal partition is total.
    assert result.status == "terminal-partition-complete"
    assert checker.metadata_calls == 2, "retired reuse must buy the second round live"
    paths = _paths(tmp_path, snapshot)
    gates = scan_jsonl(paths.ledgers.gates)
    metadata_gates = [gate for gate in gates if gate["gate_kind"] == "metadata_batch"]
    assert len(metadata_gates) == 2
    assert all(GATE_VERDICT_SOURCE_FIELD not in gate for gate in metadata_gates)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert models[stable_id]["status"]["code"] == "failed:accuracy-gate"
    bought, reused_events = _spend_events(paths)
    # Distinct durable spend IDs keep both paid sessions even under the frozen
    # clock; economic truth must not collapse byte-similar rounds.
    assert len([e for e in bought if e["details"]["gate_kind"] == "metadata_batch"]) == 2
    assert reused_events == []


def test_healthy_run_buys_sessions_and_records_spend(tmp_path: Path) -> None:
    """Live dispatches are visible as sessions-bought operational events."""

    snapshot = _snapshot(tmp_path, count=2)
    checker = ReuseAwareChecker()
    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    bought, reused = _spend_events(paths)
    assert reused == []
    metadata_bought = [e for e in bought if e["details"]["gate_kind"] == "metadata_batch"]
    assert len(metadata_bought) == checker.metadata_calls
    assert metadata_bought[0]["details"]["batch_size"] == min(2, METADATA_BATCH_MAX)


def _terminal_driver(
    tmp_path: Path,
    snapshot: Any,
    checker: ScriptedChecker,
    *,
    checker_model: Optional[str] = None,
) -> CrawlerDriver:
    """Build one deterministic driver whose author recommends a terminal skip."""

    author = ScriptedAuthor(AuthorScript(terminal_outcomes=(("status", "skipped:not-a-real-NN"),)))
    return _driver(tmp_path, snapshot, author=author, checker=checker, checker_model=checker_model)


def _clear_model_history(paths: Any) -> None:
    """Simulate a fresh re-drive leg: models and driver state reset, gates kept."""

    for stale in (paths.ledgers.models, paths.driver_state):
        if stale.exists():
            stale.unlink()


def test_redrive_over_unchanged_terminal_result_buys_live_under_retirement(
    tmp_path: Path,
) -> None:
    """A re-drive leg over an unchanged author result buys a fresh live session.

    The scripted checker is deterministic, so the second live adjudication
    mints byte-identical gate content and the append stays idempotent -- the
    ledger keeps exactly one adjudication claim -- but the SESSION is bought
    live and no reuse event exists.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    leg_one = ReuseAwareChecker()
    assert _terminal_driver(tmp_path, snapshot, leg_one).run().status == "complete"
    # AG-4 duplicate-accept: a fresh acceptance costs the accept session plus
    # one blind confirmation session.
    assert leg_one.terminal_calls == 2

    paths = _paths(tmp_path, snapshot)
    _clear_model_history(paths)
    leg_two = ReuseAwareChecker()
    result = _terminal_driver(tmp_path, snapshot, leg_two).run()

    assert result.status == "complete"
    # The re-drive adjudicates LIVE (retired reuse), and its byte-identical
    # accept lands on the already-CONFIRMED gate row, so no second
    # confirmation is bought.
    assert leg_two.terminal_calls == 1, "retired reuse must adjudicate the re-drive live"
    gates = [
        gate
        for gate in scan_jsonl(paths.ledgers.gates)
        if gate["gate_kind"] == "terminal_disposition"
    ]
    assert len(gates) == 1
    (original,) = gates
    assert GATE_VERDICT_SOURCE_FIELD not in original
    assert original["items"][0]["terminal_disposition"]["verdict"] == "accepted"
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert models[stable_id]["status"]["code"] == "skipped:not-a-real-NN"
    assert models[stable_id]["status"]["human_review"]["required"] is False
    bought, reused = _spend_events(paths)
    assert reused == []
    # Leg one: accept + AG-4 blind confirmation; leg two: one live re-drive
    # session landing on the already-confirmed row.
    assert len([e for e in bought if e["details"]["gate_kind"] == "terminal_disposition"]) == 3


class FreshIdChecker(ReuseAwareChecker):
    """Mint per-checker-model terminal gate ids, like real nonce-seeded ids.

    The scripted fake derives its terminal ``gate_id`` from the stable ID
    alone, so two legs adjudicating under DIFFERENT checker models would
    collide on one identity with different bytes -- an artifact of the fake,
    not of the production id, which mixes in a fresh request nonce.
    """

    def check_terminal(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        outcome = super().check_terminal(artifact, work_root, config)
        assert outcome.gate is not None
        gate = dict(outcome.gate)
        gate["gate_id"] = f"{gate['gate_id']}-{config.checker_model}"
        return CheckerOutcome(gate=_reseal(gate))


def test_checker_model_change_forces_live_terminal_session(tmp_path: Path) -> None:
    """Reuse never crosses a checker model change: the re-drive buys live."""

    snapshot = _snapshot(tmp_path, count=1)
    leg_one = FreshIdChecker()
    assert _terminal_driver(tmp_path, snapshot, leg_one).run().status == "complete"
    # AG-4 duplicate-accept: accept session plus one blind confirmation.
    assert leg_one.terminal_calls == 2

    paths = _paths(tmp_path, snapshot)
    _clear_model_history(paths)
    leg_two = FreshIdChecker()
    result = _terminal_driver(
        tmp_path, snapshot, leg_two, checker_model="gpt-next-generation"
    ).run()

    # PRE-EXISTING behavior, not a reuse effect: the live cross-model
    # re-adjudication appends a SECOND gate row carrying the same accepted
    # item bytes, so publication's exact-membership resolution goes ambiguous
    # and routes the model to human review. Reuse neither causes nor masks it.
    assert result.status == "terminal-partition-complete"
    # Live cross-model adjudication, and its fresh accept under the NEW
    # identity carries its own AG-4 blind confirmation (2 sessions).
    assert leg_two.terminal_calls == 2, "a new checker model must adjudicate live"
    replays = [
        gate
        for gate in scan_jsonl(paths.ledgers.gates)
        if gate.get(GATE_VERDICT_SOURCE_FIELD) == GATE_VERDICT_SOURCE_REUSED
    ]
    assert replays == []


def test_throughput_census_counts_only_live_spend_under_retirement(tmp_path: Path) -> None:
    """The census records every session as bought and zero verdicts as reused."""

    from menagerie.crawler.tools.throughput_report import (
        _checker_spend_counts,
        _gate_timings,
    )

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    checker = ReuseAwareChecker(
        script=CheckerScript(metadata_verdict="inaccurate", selected_id=stable_id)
    )
    _driver(tmp_path, snapshot, checker=checker).run()

    paths = _paths(tmp_path, snapshot)
    spend = _checker_spend_counts(
        (
            *scan_jsonl(paths.operational_ledger),
            *load_paid_session_rows(paths.paid_session_ledger),
        )
    )
    assert spend["metadata_batch"]["sessions_bought"] == 2
    assert spend["metadata_batch"]["verdicts_reused"] == 0
    gates = scan_jsonl(paths.ledgers.gates)
    timings = _gate_timings(gates)
    metadata_rows = [g for g in gates if g["gate_kind"] == "metadata_batch"]
    assert len(metadata_rows) == 2
    assert timings["metadata_batch"]["count"] == 2


def test_lane_without_identity_never_reuses_in_the_driver(tmp_path: Path) -> None:
    """A checker lane that does not advertise identity always buys live sessions."""

    snapshot = _snapshot(tmp_path, count=1)
    leg_one = CountingChecker()
    author_script = AuthorScript(terminal_outcomes=(("status", "skipped:not-a-real-NN"),))
    assert (
        _driver(tmp_path, snapshot, author=ScriptedAuthor(author_script), checker=leg_one)
        .run()
        .status
        == "complete"
    )

    paths = _paths(tmp_path, snapshot)
    _clear_model_history(paths)
    leg_two = CountingChecker()
    result = _driver(
        tmp_path, snapshot, author=ScriptedAuthor(author_script), checker=leg_two
    ).run()

    assert result.status == "complete"
    assert leg_two.terminal_calls == 1
    replays = [
        gate
        for gate in scan_jsonl(paths.ledgers.gates)
        if gate.get(GATE_VERDICT_SOURCE_FIELD) == GATE_VERDICT_SOURCE_REUSED
    ]
    assert replays == []
