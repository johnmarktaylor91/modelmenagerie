"""RED suite for STRAT1: stratified rung staging with tail-class salting.

Pins the 2026-08-11 directive: every ladder rung from step-200 onward is
STRATIFIED -- deliberately salted with tail-class representatives so surprise
classes surface at rung width, never at full-run width. The stratifier reads
the campaign partition, the models ledger heads, the deferral valves, and the
attempt refusal fields (all read-only) and emits a deterministic member list
(``--stable-ids-file`` shape for ``stage_rung``) plus a JSON report naming
every member's stratum and evidence.

Covered here, RED-first:

* ladder-bulk ordering: next partition rows in order, dedup vs already-terminal
  runs-holders and prior staged state;
* every tail stratum's detector (R2/vendored closure, oddball/lockless intents,
  CUDA deferrals, prior content refusals, never-sampled zoos, hard-tail
  failures);
* empty-stratum refusal with a note instead of silent reweighting;
* determinism given (records state, seed, size);
* shortfall disclosure with loud bulk backfill;
* CLI end-to-end and staging compatibility of the emitted member list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

from menagerie.crawler import authority as authority_module
from menagerie.crawler.environment_coverage import (
    COVERAGE_DEFERRAL_DISPOSITION,
    COVERAGE_DEFERRAL_SCHEMA_VERSION,
    coverage_deferral_path,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.intake import IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.partitioner import emit_campaign_partitions, load_campaign_bindings
from menagerie.crawler.rung_staging import CARRIED_TIP_STRATUM
from menagerie.crawler.tests.close_object_support import record_fixture_close

pytestmark = pytest.mark.smoke

_CAMPAIGN = "c1-mech"

#: One roster covering every stratum detector at least once. All rows land in
#: c1-mech (no reserved zoo literal). Zoo strings are chosen for the ROUTING
#: facts they derive after the 2026-08-19 oddball graduation (``mmlab`` and
#: ``oddballs`` are locked families now, so the salted/lockless intents are
#: exactly ``tf-keras-arm64`` and ``jax-flax-arm64``): a ``flax`` marker ->
#: framework ``jax`` -> ``jax-flax-arm64``; a ``keras`` marker -> framework
#: ``tensorflow`` -> ``tf-keras-arm64`` (native-tail rows inside a pytorch
#: campaign, exactly like the committed ERNIE ``discovered-paddle`` precedent --
#: only the reserved ``discovered-{tensorflow,jax,keras}`` zoos partition out);
#: ``timm``/``recbole-zoo`` and the nova zoos derive no oddball intent.
_ROSTER_ROWS: tuple[dict[str, Any], ...] = (
    {"name": "conv-done", "zoo": "recbole-zoo", "variant": ""},
    {"name": "rec-shim", "zoo": "recbole-zoo", "variant": ""},
    {"name": "rec-fresh", "zoo": "recbole-zoo", "variant": ""},
    {"name": "odd-repo", "zoo": "discovered-flax", "variant": ""},
    {"name": "mm-model", "zoo": "keras-hub", "variant": ""},
    {"name": "parked-model", "zoo": "keras-hub", "variant": ""},
    {"name": "cuda-model", "zoo": "timm", "variant": ""},
    {"name": "refused-model", "zoo": "timm", "variant": ""},
    {"name": "hard-import", "zoo": "timm", "variant": ""},
    {"name": "hard-author", "zoo": "timm", "variant": ""},
    {"name": "nova-a1", "zoo": "zoo-nova-1", "variant": ""},
    {"name": "nova-a2", "zoo": "zoo-nova-1", "variant": ""},
    {"name": "nova-b1", "zoo": "zoo-nova-2", "variant": ""},
    {"name": "timm-b", "zoo": "timm", "variant": ""},
    {"name": "timm-c", "zoo": "timm", "variant": ""},
    {"name": "timm-d", "zoo": "timm", "variant": ""},
    {"name": "timm-e", "zoo": "timm", "variant": ""},
    {"name": "timm-f", "zoo": "timm", "variant": ""},
    # Filler rows so every frozen campaign has a non-empty partition.
    {"name": "disco-one", "zoo": "discovered-pytorch", "variant": ""},
    {"name": "classic-one", "zoo": "unregistered-classics-pytorch", "variant": ""},
    {"name": "native-one", "zoo": "discovered-keras", "variant": ""},
)

#: Members of the prior registered rung ("step-100" stand-in): every one holds
#: a head record or a valve row, so the fixture's prior world is coherent.
_PRIOR_RUNG_NAMES = (
    "conv-done",
    "rec-shim",
    "cuda-model",
    "refused-model",
    "hard-import",
    "hard-author",
    "parked-model",
)

#: Explicit integer-friendly shares used by most tests: size 10 apportions to
#: bulk 4 and one member for each of the six tail strata.
_TEST_SHARES = {
    "ladder-bulk": 0.4,
    "r2-vendored": 0.1,
    "oddball-intent": 0.1,
    "cuda-deferred": 0.1,
    "content-refusal": 0.1,
    "unsampled-zoo": 0.1,
    "hard-tail": 0.1,
}


def _records_root(repo_root: Path) -> Path:
    """Return the campaign records root for one repository."""

    return repo_root / "menagerie" / "crawler" / "records"


def _root_snapshot(repo_root: Path) -> IntakeSnapshot:
    """Load the c1-mech bound full-partition intake snapshot."""

    bindings = load_campaign_bindings(_records_root(repo_root) / "campaigns.json")
    binding = next(b for b in bindings if b.spec.campaign_id == _CAMPAIGN)
    return load_intake_snapshot(_records_root(repo_root) / binding.intake_path)


def _ids_by_name(repo_root: Path) -> dict[str, str]:
    """Map fixture model names to their intake stable ids."""

    return {item.name: item.stable_id for item in _root_snapshot(repo_root).items}


def _head_record(
    stable_id: str,
    status_code: str,
    *,
    rung: Optional[str] = None,
    attempted_rungs: tuple[str, ...] = (),
    intake_snapshot_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build one minimal current model revision for the synthetic ledger.

    The stratifier reads ``status.code`` and ``source_resolution`` and selects
    current revisions through the real parentage selector, so the row needs
    exactly the lineage and fact fields -- not the full model.v3 payload. The
    close-object v2 gate additionally reads ``dependency_vector`` (the
    recorded-under discriminant), so fixture heads that a close accounts must
    name the rung that recorded them.
    """

    body: dict[str, Any] = {
        "stable_id": stable_id,
        "record_seq": 1,
        "parent_revision": None,
        "status": {"kind": status_code.split(":")[0], "code": status_code},
        "source_resolution": {
            "rung": rung or "NO_RUNG_SELECTED",
            "attempted_rungs": [
                {"rung": attempted, "result": "failed"} for attempted in attempted_rungs
            ],
        },
        "provenance": {"producer_run_id": "pytest-close"},
    }
    if intake_snapshot_id is not None:
        body["dependency_vector"] = {"intake_snapshot_id": intake_snapshot_id}
    body["record_revision"] = stable_hash(body)
    return body


def _refusal_attempt(stable_id: str, attempt_id: str) -> dict[str, Any]:
    """Build one attempt row carrying the TK1 refusal classification."""

    return {
        "stable_id": stable_id,
        "attempt_id": attempt_id,
        "error": {
            "stage": "author",
            "reason_code": "author-session-failure",
            "failure_class": "content-refusal",
            "failure_class_basis": "session-diagnostic",
            "refusal_excerpt": "I can't help with building this model.",
        },
    }


def _valve_row(stable_id: str) -> dict[str, Any]:
    """Build one machine-deferral valve row the partition arm would accept."""

    body: dict[str, Any] = {
        "schema_version": COVERAGE_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "campaign_id": _CAMPAIGN,
        "disposition": COVERAGE_DEFERRAL_DISPOSITION,
        "run_id": "pytest-close",
        "recheck_hint": "re-run this model routed to the 'tf-keras-arm64' environment intent",
    }
    body["row_sha256"] = stable_hash(body)
    return body


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write canonical JSONL rows, creating parents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


def _register_prior_rung(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Register the fixture's prior rung under the durable registry.

    Returns
    -------
    StagedRung
        The registered prior rung, so the fixture can attest its close after
        the prior-rung records land (BD-1: staging step-200 consumes step-100's
        immutable close object, and that object binds the ledger bytes it
        observed -- so it must be written AFTER those bytes exist).
    """

    from menagerie.crawler.rung_staging import CURRENCY_CARVE_OUT_ATTRIBUTE, stage_rung

    monkeypatch.setattr(authority_module, CURRENCY_CARVE_OUT_ATTRIBUTE, True, raising=False)
    ids = _ids_by_name(repo_root)
    return stage_rung(
        _records_root(repo_root),
        campaign_id=_CAMPAIGN,
        stable_ids=tuple(ids[name] for name in _PRIOR_RUNG_NAMES),
        count=None,
        derived_from_snapshot_id=None,
        label="step-100",
        registered_by="pytest",
        created_at="2026-08-11T00:00:00Z",
        # The stratification universe DELIBERATELY carries lockless-intent
        # members (the oddball-intent stratum), so the fixture states the
        # EN-2 exemption explicitly, exactly as an operator would.
        allow_lockless=("jax-flax-arm64", "tf-keras-arm64"),
    )


def build_stratified_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_roster_rows: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Materialize a campaign repository with prior-rung records state.

    Exposed as a plain builder as well as the ``stratified_repo`` fixture so a
    sibling suite (the B-03 composition-attestation module) composes against
    the SAME records world rather than a divergent copy of it.

    Parameters
    ----------
    extra_roster_rows:
        Additional roster rows appended after :data:`_ROSTER_ROWS`, for a suite
        that needs more fresh partition width than one composition spends (the
        two-generation succession module). Rows land after every fixture row, so
        every existing stratum detector and partition-order pin is unchanged.

    Returns
    -------
    pathlib.Path
        Repository root whose records hold the emitted partitions, the prior
        registered rung, synthetic model heads, refusal attempts, and one
        coverage valve row.
    """

    repo_root = tmp_path / "repo"
    records_root = _records_root(repo_root)
    roster_path = tmp_path / "roster.jsonl"
    roster_path.write_bytes(
        b"".join(
            canonical_json_bytes(dict(row)) + b"\n"
            for row in (*_ROSTER_ROWS, *extra_roster_rows)
        )
    )
    emit_campaign_partitions(roster_path, records_root)
    # AW-2 fails CLOSED on an absent models ledger: the empty shard is the
    # explicit statement that no awards exist yet when the prior rung stages.
    _write_jsonl(records_root / "models" / "current-shard.jsonl", [])
    prior_rung = _register_prior_rung(repo_root, monkeypatch)
    ids = _ids_by_name(repo_root)
    # Open the prior rung's leg BEFORE its records land (close-object v2): the
    # generation's baselines are the evidence-window floor, and the fixture's
    # heads and valve row must sit INSIDE that window, recorded-under the
    # rung, for the close's sealed proof to account them.
    from menagerie.crawler.tests.close_object_support import open_fixture_leg

    open_fixture_leg(
        records_root,
        campaign_id=_CAMPAIGN,
        intake_snapshot_id=prior_rung.registration.snapshot_id,
        intake_snapshot_sha256=prior_rung.registration.snapshot_sha256,
    )
    recorded_under = prior_rung.registration.snapshot_id
    _write_jsonl(
        records_root / "models" / "current-shard.jsonl",
        [
            _head_record(
                ids["conv-done"], "runs", rung="R2_VENDOR", intake_snapshot_id=recorded_under
            ),
            _head_record(
                ids["rec-shim"],
                "failed:import",
                rung="R2_VENDOR",
                intake_snapshot_id=recorded_under,
            ),
            _head_record(
                ids["cuda-model"], "deferred:needs-cuda", intake_snapshot_id=recorded_under
            ),
            _head_record(ids["refused-model"], "failed:author", intake_snapshot_id=recorded_under),
            _head_record(ids["hard-import"], "failed:import", intake_snapshot_id=recorded_under),
            _head_record(ids["hard-author"], "failed:author", intake_snapshot_id=recorded_under),
        ],
    )
    _write_jsonl(
        records_root / "attempts" / "local.jsonl",
        [
            _refusal_attempt(ids["refused-model"], "attempt-refused-1"),
            {"stable_id": ids["hard-import"], "attempt_id": "attempt-plain-1", "error": None},
        ],
    )
    _write_jsonl(coverage_deferral_path(records_root), [_valve_row(ids["parked-model"])])
    # The prior rung CLOSED: step-200 may only be staged on step-100's
    # immutable close attestation (BD-1 / T1-sol-02). Written last so the
    # source identities it binds are the fixture's final bytes; the arms are
    # DERIVED from the sealed proof (six adjudicated run outcomes, one
    # valve-withheld member), never hand-asserted.
    record_fixture_close(
        records_root,
        campaign_id=_CAMPAIGN,
        intake_snapshot_id=prior_rung.registration.snapshot_id,
        intake_snapshot_sha256=prior_rung.registration.snapshot_sha256,
        member_ids=tuple(ids[name] for name in _PRIOR_RUNG_NAMES),
        completion_status="complete-with-deferrals",
    )
    return repo_root


@pytest.fixture()
def stratified_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the stratification fixture repository for one test.

    Returns
    -------
    pathlib.Path
        Repository root built by :func:`build_stratified_repo`.
    """

    return build_stratified_repo(tmp_path, monkeypatch)


#: The fixture's PD-7 disposition (SOL-02): fixtures model the composed
#: successor shape, and a successor stage demands a digest-bound SATISFIED
#: disposition, so the composer binds one by default.
_TEST_PD7_DISPOSITION = {
    "status": "SATISFIED",
    "source_attestation_sha256": "sha256:" + "7d" * 32,
}


def _stratify(
    repo_root: Path,
    *,
    size: int = 10,
    seed: int = 7,
    shares: Optional[Mapping[str, float]] = None,
    pd7_disposition: Optional[Mapping[str, str]] = _TEST_PD7_DISPOSITION,
):
    """Run the stratifier under test with the fixture's default knobs."""

    from menagerie.crawler.rung_stratification import stratify_rung_members

    return stratify_rung_members(
        _records_root(repo_root),
        campaign_id=_CAMPAIGN,
        size=size,
        seed=seed,
        shares=dict(_TEST_SHARES if shares is None else shares),
        pd7_disposition=pd7_disposition,
    )


def _composed_member_count(plan: Any) -> int:
    """Return the width one composition BOUGHT, excluding the carried tip.

    ``--size`` buys new width; the lineage tip is carried on top of it (C-2), so
    the number to compare against ``size`` is the composed count, not the total
    membership. Shared so every suite spells the distinction the same way.
    """

    return sum(1 for member in plan.members if member.stratum != CARRIED_TIP_STRATUM)


def _runs_award_authorities(repo_root: Path, member_ids: Sequence[str]) -> dict[str, str]:
    """Return the AW-2 authorities a stage of these members demands.

    Read straight from the fixture's ledger bytes, so a test that stages a
    CUMULATIVE rung (which carries the tip's converted models by construction)
    passes the authorities an operator would pass, and the gate under test is
    the one that fires.
    """

    path = _records_root(repo_root) / "models" / "current-shard.jsonl"
    heads: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            heads[str(row["stable_id"])] = row
    authorities: dict[str, str] = {}
    for stable_id in member_ids:
        head = heads.get(str(stable_id))
        if head is not None and dict(head.get("status") or {}).get("kind") == "runs":
            authorities[str(stable_id)] = str(head["record_revision"])
    return authorities


def _members_by_stratum(plan: Any) -> dict[str, list[str]]:
    """Group one plan's selected stable ids by stratum name."""

    grouped: dict[str, list[str]] = {}
    for member in plan.members:
        grouped.setdefault(member.stratum, []).append(member.stable_id)
    return grouped


def _partition_order(repo_root: Path) -> list[str]:
    """Return the campaign's partition-file stable ids in scheduling order."""

    bindings = load_campaign_bindings(_records_root(repo_root) / "campaigns.json")
    binding = next(b for b in bindings if b.spec.campaign_id == _CAMPAIGN)
    by_key = {
        (item.name, item.zoo, item.variant): item.stable_id
        for item in _root_snapshot(repo_root).items
    }
    ordered = []
    for line in (
        (_records_root(repo_root) / binding.partition_path).read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        ordered.append(by_key[(row["name"], row["zoo"], str(row.get("variant", "")))])
    return ordered


# ---------------------------------------------------------------------------
# Ladder bulk: next partition rows in order, dedup vs prior staged state.
# ---------------------------------------------------------------------------


def test_bulk_members_follow_partition_order_and_skip_prior_state(
    stratified_repo: Path,
) -> None:
    """Bulk takes fresh partition rows in order; prior state never re-enters."""

    ids = _ids_by_name(stratified_repo)
    plan = _stratify(stratified_repo)
    bulk = _members_by_stratum(plan)["ladder-bulk"]
    assert len(bulk) == 4
    prior_state = {ids[name] for name in (*_PRIOR_RUNG_NAMES, "parked-model")}
    assert not set(bulk) & prior_state
    order = _partition_order(stratified_repo)
    assert bulk == sorted(bulk, key=order.index)
    # Bulk is the "next fresh rows" prefix: nothing fresh that precedes a
    # selected bulk row in partition order may have been skipped silently.
    tail_selected = {m.stable_id for m in plan.members if m.stratum != "ladder-bulk"}
    fresh = [
        stable_id
        for stable_id in order
        if stable_id not in prior_state and stable_id not in tail_selected
    ]
    assert bulk == fresh[: len(bulk)]


def test_runs_holders_are_never_selected_as_work(stratified_repo: Path) -> None:
    """A converted model is never re-crawled, whatever classes it matches.

    Since C-2 it does reach the rung's MEMBERSHIP -- a successor carries its
    lineage tip, and the tip contains it -- but only under ``carried-tip``
    provenance, never as a selection any composing stratum made. The staging
    door demands its award authority either way (AW-2).
    """

    ids = _ids_by_name(stratified_repo)
    plan = _stratify(stratified_repo)
    composing = {
        member.stable_id for member in plan.members if member.stratum != CARRIED_TIP_STRATUM
    }
    assert ids["conv-done"] not in composing
    assert ids["conv-done"] in {
        member.stable_id for member in plan.members if member.stratum == CARRIED_TIP_STRATUM
    }


# ---------------------------------------------------------------------------
# Tail stratum detectors.
# ---------------------------------------------------------------------------


def test_r2_vendored_detector_selects_own_record_evidence(stratified_repo: Path) -> None:
    """A non-runs head that resolved R2_VENDOR is the R2 class's core member."""

    ids = _ids_by_name(stratified_repo)
    plan = _stratify(stratified_repo)
    selected = _members_by_stratum(plan)["r2-vendored"]
    assert selected == [ids["rec-shim"]]
    member = next(m for m in plan.members if m.stable_id == ids["rec-shim"])
    assert "R2_VENDOR" in member.evidence


def test_r2_vendored_zoo_propagation_reaches_fresh_zoo_mates(stratified_repo: Path) -> None:
    """A fresh member of an R2-evidenced zoo joins the class when width allows."""

    ids = _ids_by_name(stratified_repo)
    shares = dict(_TEST_SHARES)
    shares.update({"r2-vendored": 0.2, "ladder-bulk": 0.3})
    plan = _stratify(stratified_repo, shares=shares)
    selected = set(_members_by_stratum(plan)["r2-vendored"])
    assert selected == {ids["rec-shim"], ids["rec-fresh"]}
    member = next(m for m in plan.members if m.stable_id == ids["rec-fresh"])
    assert "zoo" in member.evidence and "R2_VENDOR" in member.evidence


def test_oddball_intent_detector_names_the_routed_intent(stratified_repo: Path) -> None:
    """Oddball/lockless-intent members are selected by their routed intent."""

    ids = _ids_by_name(stratified_repo)
    shares = dict(_TEST_SHARES)
    shares.update({"oddball-intent": 0.3, "ladder-bulk": 0.2})
    plan = _stratify(stratified_repo, shares=shares)
    selected = set(_members_by_stratum(plan)["oddball-intent"])
    assert selected == {ids["odd-repo"], ids["mm-model"], ids["parked-model"]}
    evidence = {m.stable_id: m.evidence for m in plan.members}
    assert "jax-flax-arm64" in evidence[ids["odd-repo"]]
    assert "tf-keras-arm64" in evidence[ids["mm-model"]]
    # The already-parked model discloses its valve so width proves the parking.
    assert COVERAGE_DEFERRAL_DISPOSITION in evidence[ids["parked-model"]]


def test_cuda_deferred_detector(stratified_repo: Path) -> None:
    """deferred:needs-cuda heads form the CUDA-deferred stratum."""

    ids = _ids_by_name(stratified_repo)
    plan = _stratify(stratified_repo)
    assert _members_by_stratum(plan)["cuda-deferred"] == [ids["cuda-model"]]
    member = next(m for m in plan.members if m.stable_id == ids["cuda-model"])
    assert "deferred:needs-cuda" in member.evidence


def test_refusal_history_detector_wins_over_hard_tail(stratified_repo: Path) -> None:
    """A content-refusal attempt assigns the refusal stratum, not hard-tail."""

    ids = _ids_by_name(stratified_repo)
    plan = _stratify(stratified_repo)
    grouped = _members_by_stratum(plan)
    assert grouped["content-refusal"] == [ids["refused-model"]]
    assert ids["refused-model"] not in grouped["hard-tail"]
    member = next(m for m in plan.members if m.stable_id == ids["refused-model"])
    assert "content-refusal" in member.evidence


def test_never_sampled_zoo_detector_maximizes_zoo_diversity(stratified_repo: Path) -> None:
    """With width for two, two DISTINCT never-sampled zoos are selected."""

    ids = _ids_by_name(stratified_repo)
    shares = dict(_TEST_SHARES)
    shares.update({"unsampled-zoo": 0.2, "ladder-bulk": 0.3})
    plan = _stratify(stratified_repo, shares=shares)
    selected = _members_by_stratum(plan)["unsampled-zoo"]
    assert len(selected) == 2
    zoos = {next(m.zoo for m in plan.members if m.stable_id == stable_id) for stable_id in selected}
    # Two DISTINCT never-sampled zoos (the fresh flax-zoo oddball counts as
    # one when the oddball stratum did not already claim it).
    assert len(zoos) == 2
    assert zoos <= {"zoo-nova-1", "zoo-nova-2", "discovered-flax"}
    # Sampled zoos (timm, recbole-zoo, keras-hub) are not eligible.
    assert ids["rec-fresh"] not in selected and ids["timm-b"] not in selected


def test_hard_tail_detector_selects_failed_author_and_import_heads(
    stratified_repo: Path,
) -> None:
    """failed:author / failed:import heads form the hard-tail difficulty sample."""

    ids = _ids_by_name(stratified_repo)
    shares = dict(_TEST_SHARES)
    shares.update({"hard-tail": 0.2, "ladder-bulk": 0.3})
    plan = _stratify(stratified_repo, shares=shares)
    selected = set(_members_by_stratum(plan)["hard-tail"])
    # rec-shim is claimed by the earlier R2 stratum; refused-model by the
    # refusal stratum; what remains is the pure difficulty sample.
    assert selected == {ids["hard-import"], ids["hard-author"]}
    evidence = {m.stable_id: m.evidence for m in plan.members}
    assert "failed:import" in evidence[ids["hard-import"]]
    assert "failed:author" in evidence[ids["hard-author"]]


# ---------------------------------------------------------------------------
# Refusal, disclosure, determinism.
# ---------------------------------------------------------------------------


def test_empty_stratum_refuses_with_a_note(
    stratified_repo: Path,
) -> None:
    """A vanished stratum refuses loudly instead of silently reweighting."""

    from menagerie.crawler.rung_stratification import EmptyStratumError

    records_root = _records_root(stratified_repo)
    ids = _ids_by_name(stratified_repo)
    models_path = records_root / "models" / "current-shard.jsonl"
    kept = [
        row
        for row in (
            json.loads(line)
            for line in models_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row["stable_id"] != ids["cuda-model"]
    ]
    _write_jsonl(models_path, kept)
    with pytest.raises(EmptyStratumError) as excinfo:
        _stratify(stratified_repo)
    assert "cuda-deferred" in str(excinfo.value)
    # An explicit zero share acknowledges the vanished stratum and proceeds.
    shares = dict(_TEST_SHARES)
    shares.update({"cuda-deferred": 0.0, "ladder-bulk": 0.5})
    plan = _stratify(stratified_repo, shares=shares)
    assert "cuda-deferred" not in _members_by_stratum(plan)
    assert _composed_member_count(plan) == 10


def test_shortfall_backfills_from_bulk_and_is_disclosed(stratified_repo: Path) -> None:
    """A non-empty short stratum takes what exists; bulk fills the gap loudly."""

    shares = dict(_TEST_SHARES)
    shares.update({"cuda-deferred": 0.3, "ladder-bulk": 0.2})
    plan = _stratify(stratified_repo, shares=shares)
    assert len(_members_by_stratum(plan)["cuda-deferred"]) == 1
    assert plan.shortfalls["cuda-deferred"] == {"target": 3, "available": 1}
    assert plan.backfilled_from_bulk == 2
    assert _composed_member_count(plan) == 10


def test_deterministic_given_records_state_seed_and_size(stratified_repo: Path) -> None:
    """Identical inputs yield byte-identical member lists and reports."""

    first = _stratify(stratified_repo)
    second = _stratify(stratified_repo)
    assert [m.stable_id for m in first.members] == [m.stable_id for m in second.members]
    assert json.dumps(first.to_report(), sort_keys=True) == json.dumps(
        second.to_report(), sort_keys=True
    )
    reseeded = _stratify(stratified_repo, seed=8)
    assert {m.stratum for m in reseeded.members} == {m.stratum for m in first.members}
    counts = lambda plan: {  # noqa: E731 - tiny local aggregation
        stratum: len(ids) for stratum, ids in _members_by_stratum(plan).items()
    }
    assert counts(reseeded) == counts(first)


def test_report_names_every_member_stratum_and_evidence(stratified_repo: Path) -> None:
    """The JSON report is complete: strata, targets, evidence, records state."""

    plan = _stratify(stratified_repo)
    report = plan.to_report()
    assert report["campaign_id"] == _CAMPAIGN
    assert report["size"] == 10 and report["seed"] == 7
    assert _composed_member_count(plan) == 10
    assert len(report["members"]) == len(plan.members)
    for member in report["members"]:
        assert member["stable_id"] and member["stratum"] and member["evidence"]
        assert "name" in member and "zoo" in member
    # Carrying is mandatory, so it draws no share and no apportioned target; it
    # is accounted in selected_counts and disclosed in its own block.
    assert set(report["targets"]) == set(report["shares"]) == set(_TEST_SHARES)
    assert set(report["selected_counts"]) == set(_TEST_SHARES) | {CARRIED_TIP_STRATUM}
    assert report["carried_tip"]["provenance"] == "lineage-tip"
    assert report["carried_tip"]["member_count"] == len(_PRIOR_RUNG_NAMES)
    assert report["records_state"]["models_ledger_sha256"].startswith("sha256:")


# ---------------------------------------------------------------------------
# CLI end-to-end and staging compatibility.
# ---------------------------------------------------------------------------


def test_cli_writes_member_list_and_report(stratified_repo: Path, tmp_path: Path) -> None:
    """The tool writes the stage_rung member-list shape plus the JSON report."""

    from menagerie.crawler.tools.stratify_rung import main

    members_path = tmp_path / "step-200-members.txt"
    report_path = tmp_path / "step-200-report.json"
    exit_code = main(
        [
            "--repo-root",
            str(stratified_repo),
            "--campaign",
            _CAMPAIGN,
            "--size",
            "10",
            "--seed",
            "7",
            "--out-members",
            str(members_path),
            "--out-report",
            str(report_path),
            *(
                part
                for name, share in sorted(_TEST_SHARES.items())
                for part in ("--share", f"{name}={share}")
            ),
        ]
    )
    assert exit_code == 0
    member_ids = [line for line in members_path.read_text(encoding="utf-8").splitlines() if line]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert member_ids == [member["stable_id"] for member in report["members"]]
    composed = [
        member["stable_id"]
        for member in report["members"]
        if member["stratum"] != CARRIED_TIP_STRATUM
    ]
    assert len(member_ids) == len(set(member_ids)) == 10 + len(report["carried_tip"]["carried_forward"])
    assert len(composed) == 10
    # The file the staging door consumes is the CUMULATIVE membership: every
    # carried tip member is in it, or the successor could not widen the tip.
    assert set(report["carried_tip"]["member_ids"]) <= set(member_ids)


def test_cli_refuses_empty_stratum_with_exit_code(
    stratified_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI surfaces the typed empty-stratum refusal as a non-zero exit."""

    from menagerie.crawler.tools.stratify_rung import main

    records_root = _records_root(stratified_repo)
    ids = _ids_by_name(stratified_repo)
    models_path = records_root / "models" / "current-shard.jsonl"
    kept = [
        row
        for row in (
            json.loads(line)
            for line in models_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row["stable_id"] != ids["cuda-model"]
    ]
    _write_jsonl(models_path, kept)
    exit_code = main(
        [
            "--repo-root",
            str(stratified_repo),
            "--campaign",
            _CAMPAIGN,
            "--size",
            "10",
            "--seed",
            "7",
            "--out-members",
            str(tmp_path / "members.txt"),
            *(
                part
                for name, share in sorted(_TEST_SHARES.items())
                for part in ("--share", f"{name}={share}")
            ),
        ]
    )
    assert exit_code == 2
    assert "cuda-deferred" in capsys.readouterr().err
    assert not (tmp_path / "members.txt").exists()


def test_emitted_members_stage_through_stage_rung(stratified_repo: Path) -> None:
    """The composer's own member list stages: it is a cumulative successor.

    C-2. The composition carries the lineage tip, so the widen rule is satisfied
    by construction and the two documented commands actually compose. A member
    list with the carried members stripped back out is still refused -- the
    round-2 break-1 property is unchanged, it is the COMPOSER that changed.
    """

    from menagerie.crawler.rung_staging import RungStagingError, stage_rung

    plan = _stratify(stratified_repo)
    tip_members = set(plan.to_report()["carried_tip"]["member_ids"])
    members = plan.member_ids()

    def stage(selected: tuple[str, ...], label: str, attestation: Any) -> Any:
        return stage_rung(
            _records_root(stratified_repo),
            campaign_id=_CAMPAIGN,
            stable_ids=selected,
            count=None,
            derived_from_snapshot_id=None,
            label=label,
            registered_by="pytest",
            created_at="2026-08-11T00:00:00Z",
            award_authorities=_runs_award_authorities(stratified_repo, selected),
            # EN-2 exemption, stated explicitly: the oddball-intent stratum
            # composes lockless-routing members by design.
            allow_lockless=("jax-flax-arm64", "tf-keras-arm64"),
            # Fixtures model the composed next-rung shape (W16/S-2 declaration),
            # so they present the SOL-02 PD-7-bearing attestation it demands.
            rung_class="successor_rung",
            composition_attestation=attestation,
        )

    from menagerie.crawler.tests.rt3_support import fixture_composition_attestation

    delta = tuple(stable_id for stable_id in members if stable_id not in tip_members)
    with pytest.raises(RungStagingError, match="strictly widen"):
        stage(
            delta,
            "step-200-delta",
            fixture_composition_attestation(
                _records_root(stratified_repo), _CAMPAIGN, stable_ids=delta
            ),
        )

    staged = stage(members, "step-200", plan.composition_attestation)
    assert staged.registration_created is True
    assert staged.registration.member_count == len(members)


# ---------------------------------------------------------------------------
# Membership reconciliation (C-ERNIE / T2-14): canonical records whose
# stable ids fall outside the campaign staging universe refuse composition
# until the operator acknowledges each id explicitly.
# ---------------------------------------------------------------------------


def _append_ledger_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one canonical JSONL row to an existing fixture ledger."""

    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(row)) + b"\n")


def test_foreign_head_refuses_composition(stratified_repo: Path) -> None:
    """A model head outside the partition universe is a typed refusal.

    The pilot's ERNIE 2.0 row (T2-14) lives in the intake snapshot, models
    ledger, gates, and grants but in neither the roster nor any campaign
    partition -- silently unstageable forever. Staging must surface it.
    """

    from menagerie.crawler.rung_stratification import MembershipReconciliationError

    records_root = _records_root(stratified_repo)
    ghost = _head_record("m-ghost-foreign", "failed:accuracy-gate")
    _append_ledger_row(records_root / "models" / "current-shard.jsonl", ghost)
    with pytest.raises(MembershipReconciliationError) as excinfo:
        _stratify(stratified_repo)
    assert "m-ghost-foreign" in str(excinfo.value)
    assert "foreign" in str(excinfo.value)


def test_foreign_attempt_subject_refuses_composition(stratified_repo: Path) -> None:
    """An attempt row for a never-a-member id also trips the census."""

    from menagerie.crawler.rung_stratification import MembershipReconciliationError

    records_root = _records_root(stratified_repo)
    _append_ledger_row(
        records_root / "attempts" / "local.jsonl",
        {"stable_id": "m-ghost-attempt", "attempt_id": "attempt-ghost-1", "error": None},
    )
    with pytest.raises(MembershipReconciliationError) as excinfo:
        _stratify(stratified_repo)
    assert "m-ghost-attempt" in str(excinfo.value)


def test_acknowledged_foreign_id_composes_and_is_disclosed(stratified_repo: Path) -> None:
    """Acknowledging the id lets staging proceed and discloses the census.

    The acknowledgment is disposition-invariant: the foreign id is never
    selected into any stratum; it is only disclosed.
    """

    from menagerie.crawler.rung_stratification import stratify_rung_members

    records_root = _records_root(stratified_repo)
    ghost = _head_record("m-ghost-foreign", "failed:accuracy-gate")
    _append_ledger_row(records_root / "models" / "current-shard.jsonl", ghost)
    plan = stratify_rung_members(
        records_root,
        campaign_id=_CAMPAIGN,
        size=10,
        seed=7,
        shares=dict(_TEST_SHARES),
        acknowledged_membership=("m-ghost-foreign",),
    )
    report = plan.to_report()
    census = report["membership_reconciliation"]
    assert census["foreign_to_partition"] == ["m-ghost-foreign"]
    assert census["acknowledged"] == ["m-ghost-foreign"]
    assert "m-ghost-foreign" not in {m.stable_id for m in plan.members}


def test_head_outside_registered_lineage_refuses(stratified_repo: Path) -> None:
    """A partition member with records but no registered-rung membership refuses.

    Records minted under a never-registered intake (the pilot's head-50 class)
    stay invisible to prior-rung dedup and to lineage-aware reads; staging
    surfaces them instead of composing around them.
    """

    from menagerie.crawler.rung_stratification import MembershipReconciliationError

    records_root = _records_root(stratified_repo)
    ids = _ids_by_name(stratified_repo)
    stray = _head_record(ids["timm-b"], "failed:import")
    _append_ledger_row(records_root / "models" / "current-shard.jsonl", stray)
    with pytest.raises(MembershipReconciliationError) as excinfo:
        _stratify(stratified_repo)
    assert ids["timm-b"] in str(excinfo.value)
    assert "registered" in str(excinfo.value)


def test_unmatched_acknowledgment_refuses(stratified_repo: Path) -> None:
    """An acknowledgment naming no censused id is operator error, not a no-op."""

    from menagerie.crawler.rung_stratification import (
        MembershipReconciliationError,
        stratify_rung_members,
    )

    with pytest.raises(MembershipReconciliationError) as excinfo:
        stratify_rung_members(
            _records_root(stratified_repo),
            campaign_id=_CAMPAIGN,
            size=10,
            seed=7,
            shares=dict(_TEST_SHARES),
            acknowledged_membership=("m-never-censused",),
        )
    assert "m-never-censused" in str(excinfo.value)


def test_clean_records_report_empty_reconciliation(stratified_repo: Path) -> None:
    """A healthy store discloses empty censuses (the field is always present)."""

    plan = _stratify(stratified_repo)
    census = plan.to_report()["membership_reconciliation"]
    assert census == {
        "foreign_to_partition": [],
        "records_outside_this_campaign": [],
        "outside_registered_lineage": [],
        "acknowledged": [],
    }


def test_cli_acknowledge_membership_flag(stratified_repo: Path, tmp_path: Path) -> None:
    """CLI: refusal exits 2 naming the id; the ack flag composes end to end."""

    from menagerie.crawler.tools.stratify_rung import main

    records_root = _records_root(stratified_repo)
    ghost = _head_record("m-ghost-foreign", "failed:accuracy-gate")
    _append_ledger_row(records_root / "models" / "current-shard.jsonl", ghost)
    base_args = [
        "--repo-root",
        str(stratified_repo),
        "--campaign",
        _CAMPAIGN,
        "--size",
        "10",
        "--seed",
        "7",
        "--out-members",
        str(tmp_path / "members.txt"),
        "--out-report",
        str(tmp_path / "report.json"),
        *(
            part
            for name, share in sorted(_TEST_SHARES.items())
            for part in ("--share", f"{name}={share}")
        ),
    ]
    assert main(base_args) == 2
    assert not (tmp_path / "members.txt").exists()
    assert main([*base_args, "--acknowledge-membership", "m-ghost-foreign"]) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["membership_reconciliation"]["foreign_to_partition"] == ["m-ghost-foreign"]
