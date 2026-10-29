"""H1 red-team operator-flow robustness fixes (2026-08-19), pinned RED->GREEN.

Each test distills one confirmed H1 finding's repro:

* F1 -- ``ratify`` selected its governing head from ONE layout's raw last row,
  attesting a superseded award as governing when the true current head lived
  in ``models.jsonl``.
* F2 -- the intake sweeps' claim rule: a manifest-less partial register copy
  is never a snapshot claim (so nothing bricks) and is disclosed by the
  residue census.
* F3 -- operator ``void`` then ``migrate --apply`` hard-conflicted on the
  same discharge identity and permanently bricked the migration.
* F4 -- ratify/restore disposition events were keyed on wall-clock, so
  operator retries double-appended.
* F5 -- ``launch_preflight`` silently skipped the launch-certificate assert
  when ``--repo-root`` was omitted.
* F6 -- the models ledger alone had no torn-tail recovery path; the typed
  repair tool quarantines the torn bytes with evidence and disclosure while
  the reader keeps refusing (no fail-open).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.operational_identity import (
    CAMPAIGN_HEALTH_IDEMPOTENT_DISPOSITIONS,
    mint_operational_event,
)
from menagerie.crawler.recordio import JsonlLedger


def _model_row(
    stable_id: str,
    revision: str,
    *,
    seq: int,
    parent: str | None,
    kind: str,
    code: str,
) -> str:
    return json.dumps(
        {
            "stable_id": stable_id,
            "record_revision": revision,
            "record_seq": seq,
            "parent_revision": parent,
            "status": {"kind": kind, "code": code},
        }
    )


def _campaign_root(tmp_path: Path) -> tuple[Path, Path]:
    """Return (repo_root, records_root) with the lock dir materialized."""

    repo_root = tmp_path / "campaign"
    records_root = repo_root / "records"
    (records_root / "models").mkdir(parents=True)
    (repo_root / ".crawl-local" / "locks").mkdir(parents=True)
    return repo_root, records_root


def _seed_eaten_then_leak_award(records_root: Path) -> None:
    """Shard layout: eaten failure -> leak runs award (the m4334 shape)."""

    (records_root / "models" / "current-shard.jsonl").write_text(
        _model_row(
            "m1", "sha256:eaten1", seq=1, parent=None, kind="failed", code="failed:accuracy-gate"
        )
        + "\n"
        + _model_row("m1", "sha256:run1", seq=2, parent="sha256:eaten1", kind="runs", code="runs")
        + "\n",
        encoding="utf-8",
    )


def _ratify_args(repo_root: Path, records_root: Path, *, reason: str = "x") -> list[str]:
    return [
        "ratify",
        "m1",
        "--eaten-revision",
        "sha256:eaten1",
        "--leak-award-revision",
        "sha256:run1",
        "--reason",
        reason,
        "--records-root",
        str(records_root),
        "--repo-root",
        str(repo_root),
    ]


def _events(records_root: Path) -> list[dict[str, Any]]:
    path = records_root / "operational" / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# F1 -- ratify head selection through the canonical union projection
# ---------------------------------------------------------------------------


def test_f1_ratify_refuses_when_true_union_head_superseded_the_shard_head(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A later failure head in models.jsonl defeats the shard's raw last row.

    RED (pre-fix): ratify read only ``models/current-shard.jsonl`` and took
    ``rows[-1]``, so the guard passed on the superseded ``runs`` head and a
    durable ``award-ratified`` attestation named it governing (exit 0).
    """

    from menagerie.crawler.tools.restore_superseded_award import main

    repo_root, records_root = _campaign_root(tmp_path)
    _seed_eaten_then_leak_award(records_root)
    # The TRUE current head lives in the flat layout: run1 was superseded.
    (records_root / "models.jsonl").write_text(
        _model_row(
            "m1",
            "sha256:fail2",
            seq=3,
            parent="sha256:run1",
            kind="failed",
            code="failed:accuracy-gate",
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(_ratify_args(repo_root, records_root)) == 1
    err = capsys.readouterr().err
    assert "requires a governing runs head" in err
    assert _events(records_root) == [], "a refused ratification must append nothing"


def test_f1_ratify_still_ratifies_a_genuine_governing_runs_head(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from menagerie.crawler.tools.restore_superseded_award import main

    repo_root, records_root = _campaign_root(tmp_path)
    _seed_eaten_then_leak_award(records_root)
    assert main(_ratify_args(repo_root, records_root)) == 0
    event = json.loads(capsys.readouterr().out)
    assert event["details"]["disposition"] == "award-ratified"
    assert event["details"]["governing_revision"] == "sha256:run1"


def test_f1_ratify_refuses_when_named_award_is_not_the_derived_current_head(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A further leak award in the flat layout outdates the named revision."""

    from menagerie.crawler.tools.restore_superseded_award import main

    repo_root, records_root = _campaign_root(tmp_path)
    _seed_eaten_then_leak_award(records_root)
    (records_root / "models.jsonl").write_text(
        _model_row("m1", "sha256:run2", seq=3, parent="sha256:run1", kind="runs", code="runs")
        + "\n",
        encoding="utf-8",
    )
    assert main(_ratify_args(repo_root, records_root)) == 1
    assert "must be the current governing head" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# F2 -- manifest-less partial register copies: skipped by claims, disclosed
# ---------------------------------------------------------------------------


def test_f2_partial_dir_is_no_claim_and_is_disclosed_by_the_residue_census(
    tmp_path: Path,
) -> None:
    from menagerie.crawler.checkpoint import (
        canonical_intake_snapshot_roots,
        partial_intake_snapshot_residues,
    )

    intake_root = tmp_path / "intake"
    complete = intake_root / "intake-complete"
    complete.mkdir(parents=True)
    (complete / "manifest.json").write_text("{}", encoding="utf-8")
    partial = intake_root / "intake-crashmid"
    (partial / "sources").mkdir(parents=True)
    (partial / "items.jsonl").write_text("", encoding="utf-8")
    wrapper = intake_root / "wrapper-dir"
    nested = wrapper / "intake-nested"
    nested.mkdir(parents=True)
    (nested / "manifest.json").write_text("{}", encoding="utf-8")

    claims = canonical_intake_snapshot_roots(intake_root)
    assert partial not in claims, "a manifest-less partial dir must never claim"
    assert {complete, nested} == set(claims)
    residues = partial_intake_snapshot_residues(intake_root)
    assert residues == (partial,), "the crashed register copy must be disclosed"


def test_f2_register_crash_writes_manifest_last_and_heals_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED (pre-fix): the durability copy wrote manifest.json BEFORE sources/,
    so a crash mid-copy left a durable dir that CLAIMED to be a snapshot and
    bricked the grant lane and the canonical checkpoint campaign-wide."""

    from menagerie.crawler import rung_staging
    from menagerie.crawler.checkpoint import (
        canonical_intake_snapshot_roots,
        partial_intake_snapshot_residues,
    )
    from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes
    from menagerie.crawler.intake import load_intake_snapshot
    from menagerie.crawler.partitioner import emit_campaign_partitions
    from menagerie.crawler.rung_staging import register_existing_snapshot
    from menagerie.crawler.tests import test_rung_staging as staging

    repo_root = tmp_path / "repo"
    records_root = repo_root / "menagerie" / "crawler" / "records"
    roster = tmp_path / "roster.jsonl"
    roster.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in staging._ROSTER_ROWS)
    )
    emit_campaign_partitions(roster, records_root)
    staging._remove_carve_out(monkeypatch)
    scratch = staging._scratch_subset_snapshot(repo_root, tmp_path / "scratch", "c1-mech", take=3)
    staging._write_models_shard(
        records_root,
        [
            staging._head_record(
                item.stable_id, "runs", "runs", intake_snapshot_id=scratch.snapshot_id
            )
            for item in scratch.items
        ],
    )

    register_kwargs: dict[str, Any] = dict(
        campaign_id="c1-mech",
        derived_from_snapshot_id=None,
        label="pilot-adopted",
        registered_by="pytest",
        created_at="2026-08-11T00:00:00Z",
    )

    def crash_on_manifest(target: Path, data: bytes) -> None:
        if target.name == "manifest.json":
            raise OSError("simulated crash during the durability copy")
        atomic_replace_bytes(target, data)

    monkeypatch.setattr(rung_staging, "atomic_replace_bytes", crash_on_manifest)
    with pytest.raises(OSError, match="simulated crash"):
        register_existing_snapshot(records_root, scratch.root, **register_kwargs)

    durable = records_root / "intake" / scratch.snapshot_id
    assert durable.is_dir(), "the crash happened mid-copy, after member writes"
    assert not (durable / "manifest.json").exists(), "the manifest must be written LAST"
    assert (durable / "items.jsonl").is_file()

    intake_root = records_root / "intake"
    claims = canonical_intake_snapshot_roots(intake_root)
    assert durable not in claims, "a manifest-less partial dir must never claim"
    for claim in claims:  # the sweeps stay un-bricked: every claim still loads
        load_intake_snapshot(claim)
    assert durable in partial_intake_snapshot_residues(intake_root)

    # Healing: re-running the register completes the copy and registers.
    monkeypatch.setattr(rung_staging, "atomic_replace_bytes", atomic_replace_bytes)
    adopted = register_existing_snapshot(records_root, scratch.root, **register_kwargs)
    assert adopted.snapshot_root == durable
    assert load_intake_snapshot(durable).snapshot_sha256 == scratch.snapshot_sha256
    assert partial_intake_snapshot_residues(intake_root) == ()


# ---------------------------------------------------------------------------
# F3 -- void then migrate --apply is idempotent, never a conflict
# ---------------------------------------------------------------------------


def _seed_consumed_generation(records_root: Path) -> None:
    event = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": "2026-08-19T00:00:00.000000Z",
        "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
        "status": OperationalEventStatus.HEALTHY.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": "seed",
        "machine_id": "seed",
        "details": {
            "stable_id": "m1",
            "grant_id": "sha256:g1",
            "new_work_id": "sha256:W1",
            "new_work_generation": 5,
        },
    }
    event = mint_operational_event(
        event, identity_payload={"grant_id": "sha256:g1", "new_work_id": "sha256:W1"}
    )
    path = records_root / "operational" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with JsonlLedger(path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        ledger.append(event)


def test_f3_void_then_migrate_apply_skips_idempotently_and_stays_rerunnable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """RED (pre-fix): migrate synthesized a same-identity/different-details
    discharge for the voided generation, hard-conflicted, and every future
    ``migrate --apply`` re-derived the same conflict -- a permanent brick."""

    from menagerie.crawler.tools.discharge_generation import main

    repo_root, records_root = _campaign_root(tmp_path)
    _seed_consumed_generation(records_root)
    common = ["--records-root", str(records_root), "--repo-root", str(repo_root)]

    assert (
        main([*common, "void", "--stable-id", "m1", "--work-id", "sha256:W1", "--reason", "op"])
        == 0
    )
    capsys.readouterr()
    for _ in range(2):  # re-runnable: the second apply is equally clean
        assert main([*common, "migrate", "--apply"]) == 0
        captured = capsys.readouterr()
        assert "skipped as idempotent" in captured.err
        report = json.loads(captured.out)
        assert report[0]["discharge"] == "voided-before-start"
        assert report[0]["explicit_discharge_event_id"] is not None

    discharges = [
        row
        for row in _events(records_root)
        if row["event_kind"] == OperationalEventKind.REQUEUE_GENERATION_DISCHARGED.value
    ]
    assert len(discharges) == 1, "the operator void must stay the ONLY discharge row"
    assert discharges[0]["details"]["source"] == "operator-void"


def test_f3_migrate_apply_still_appends_for_derivable_undischarged_generations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from menagerie.crawler.tools.discharge_generation import main

    repo_root, records_root = _campaign_root(tmp_path)
    _seed_consumed_generation(records_root)
    # A landed revision carrying the grant derives `satisfied`.
    (records_root / "models" / "current-shard.jsonl").write_text(
        json.dumps(
            {
                "stable_id": "m1",
                "record_revision": "sha256:landed",
                "record_seq": 1,
                "parent_revision": None,
                "status": {"kind": "runs", "code": "runs"},
                "budget": {"explicit_grants": ["sha256:g1"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    common = ["--records-root", str(records_root), "--repo-root", str(repo_root)]
    assert main([*common, "migrate", "--apply"]) == 0
    capsys.readouterr()
    discharges = [
        row
        for row in _events(records_root)
        if row["event_kind"] == OperationalEventKind.REQUEUE_GENERATION_DISCHARGED.value
    ]
    assert [row["details"]["discharge"] for row in discharges] == ["satisfied"]
    # And the follow-up apply replays without a second append or a crash.
    assert main([*common, "migrate", "--apply"]) == 0
    assert (
        sum(
            row["event_kind"] == OperationalEventKind.REQUEUE_GENERATION_DISCHARGED.value
            for row in _events(records_root)
        )
        == 1
    )


# ---------------------------------------------------------------------------
# F4 -- disposition events are content-keyed, so retries replay
# ---------------------------------------------------------------------------


def test_f4_ratify_retry_replays_into_the_persisted_event(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """RED (pre-fix): two identical ratify runs appended two rows whose
    event_ids differed only through sub-second ``created_at``."""

    from menagerie.crawler.tools.restore_superseded_award import main

    repo_root, records_root = _campaign_root(tmp_path)
    _seed_eaten_then_leak_award(records_root)
    assert main(_ratify_args(repo_root, records_root)) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(_ratify_args(repo_root, records_root)) == 0
    second = json.loads(capsys.readouterr().out)

    rows = [
        row
        for row in _events(records_root)
        if row.get("details", {}).get("disposition") == "award-ratified"
    ]
    assert len(rows) == 1, "an operator retry must never double-append"
    assert second == first, "the retry must return the persisted event unchanged"


def test_f4_conflicting_retry_refuses_instead_of_forking_the_incident(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from menagerie.crawler.tools.restore_superseded_award import main

    repo_root, records_root = _campaign_root(tmp_path)
    _seed_eaten_then_leak_award(records_root)
    assert main(_ratify_args(repo_root, records_root, reason="first")) == 0
    capsys.readouterr()
    assert main(_ratify_args(repo_root, records_root, reason="second")) == 1
    assert "conflicting replay" in capsys.readouterr().err
    assert len(_events(records_root)) == 1


def test_f4_identity_policy_is_disposition_scoped_and_closed() -> None:
    """The mint layer enforces the closed idempotent-disposition vocabulary."""

    def health_event(disposition: str) -> dict[str, Any]:
        return {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": "2026-08-19T00:00:00.000000Z",
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.DEGRADED.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": None,
            "run_id": "test",
            "machine_id": "test",
            "details": {"disposition": disposition, "stable_id": "m1"},
        }

    assert "award-ratified" in CAMPAIGN_HEALTH_IDEMPOTENT_DISPOSITIONS
    payload = {"disposition": "award-ratified", "stable_id": "m1"}
    one = mint_operational_event(health_event("award-ratified"), identity_payload=payload)
    two = mint_operational_event(
        {**health_event("award-ratified"), "created_at": "2027-01-01T00:00:00.000000Z"},
        identity_payload=payload,
    )
    assert one["event_id"] == two["event_id"], "identity must exclude the clock"

    with pytest.raises(ValueError, match="requires a non-empty idempotent identity payload"):
        mint_operational_event(health_event("award-ratified"))
    with pytest.raises(ValueError, match="must name its own disposition"):
        mint_operational_event(health_event("award-ratified"), identity_payload={"stable_id": "m1"})
    # A per-occurrence campaign-health disposition still forbids the payload.
    with pytest.raises(ValueError, match="forbids an idempotent identity payload"):
        mint_operational_event(
            health_event("observation-row"), identity_payload={"disposition": "observation-row"}
        )


# ---------------------------------------------------------------------------
# F9 -- twin-detection compares exactly the id-hashed operator facts
# ---------------------------------------------------------------------------


def test_f9_distinct_capability_grant_is_not_a_twin_of_a_generic_grant(
    tmp_path: Path,
) -> None:
    """RED (pre-fix): twin-detection omitted ``capability`` and
    ``blocked_award_revision`` -- both in the grant id hash -- so the ONLY
    grant kind that re-opens a memory-deferred head refused as a twin."""

    from menagerie.crawler.checkpoint import (
        append_canonical_requeue_grant,
        build_canonical_requeue_grant,
    )

    path = tmp_path / "records" / "operational" / "requeue-grants.jsonl"
    facts: dict[str, Any] = dict(
        stable_id="m1",
        stage="accuracy-gate",
        reason="needs more memory",
        attempts=1,
        granted_by="op",
    )
    append_canonical_requeue_grant(path, build_canonical_requeue_grant(path, **facts))

    capability_grant = build_canonical_requeue_grant(
        path, **facts, capability="host-capability-routing"
    )
    assert capability_grant["capability"] == "host-capability-routing"
    append_canonical_requeue_grant(path, capability_grant)

    award_grant = build_canonical_requeue_grant(
        path, **facts, blocked_award_revision="sha256:" + "a" * 64
    )
    assert award_grant["blocked_award_revision"] == "sha256:" + "a" * 64

    # A TRUE twin (identical operator facts, capability included) still refuses.
    with pytest.raises(ValueError, match="already outstanding"):
        build_canonical_requeue_grant(path, **facts, capability="host-capability-routing")
    with pytest.raises(ValueError, match="already outstanding"):
        build_canonical_requeue_grant(path, **facts)


# ---------------------------------------------------------------------------
# F6 -- models torn tail: reader refuses typed, repair quarantines evidence
# ---------------------------------------------------------------------------

_TORN = '{"stable_id":"m2","record_revision":"sha256:par'


def _seed_torn_models_ledger(records_root: Path) -> Path:
    path = records_root / "models" / "current-shard.jsonl"
    path.write_text(
        _model_row("m1", "sha256:run1", seq=1, parent=None, kind="runs", code="runs")
        + "\n"
        + _TORN,
        encoding="utf-8",
    )
    return path


def test_f6_reader_keeps_refusing_a_torn_tail_and_names_the_repair_tool(
    tmp_path: Path,
) -> None:
    from menagerie.crawler.leg_generation import current_model_heads
    from menagerie.crawler.recordio import TornTailError

    _repo_root, records_root = _campaign_root(tmp_path)
    _seed_torn_models_ledger(records_root)
    with pytest.raises(TornTailError, match="repair_models_ledger"):
        current_model_heads(records_root)


def test_f6_repair_quarantines_bytes_truncates_and_discloses_idempotently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from menagerie.crawler.leg_generation import current_model_heads
    from menagerie.crawler.tools.repair_models_ledger import main

    repo_root, records_root = _campaign_root(tmp_path)
    _seed_torn_models_ledger(records_root)
    common = ["--records-root", str(records_root), "--repo-root", str(repo_root)]

    assert main(common) == 0
    repaired = json.loads(capsys.readouterr().out)["repaired"]
    assert len(repaired) == 1
    facts = repaired[0]
    quarantine = Path(facts["quarantine_path"])
    assert quarantine.read_bytes() == _TORN.encode(), "evidence must hold the exact torn bytes"
    assert Path(facts["evidence_path"]).is_file()
    assert current_model_heads(records_root).keys() == {"m1"}, "the reader recovers post-repair"

    disclosures = [
        row
        for row in _events(records_root)
        if row.get("details", {}).get("disposition") == "models-ledger-tail-repaired"
    ]
    assert len(disclosures) == 1
    assert disclosures[0]["details"]["tail_sha256"] == facts["tail_sha256"]

    # Idempotent: a re-run repairs nothing and appends nothing.
    assert main(common) == 0
    assert json.loads(capsys.readouterr().out)["repaired"] == []
    assert (
        sum(
            row.get("details", {}).get("disposition") == "models-ledger-tail-repaired"
            for row in _events(records_root)
        )
        == 1
    )


def test_f6_repair_never_touches_a_complete_corrupt_mid_file_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """DOCTRINE: no fail-open. Mid-file corruption refuses; bytes stay put."""

    from menagerie.crawler.tools.repair_models_ledger import main

    repo_root, records_root = _campaign_root(tmp_path)
    ledger = records_root / "models" / "current-shard.jsonl"
    ledger.write_text("NOT-JSON-COMPLETE-ROW\n" + '{"torn', encoding="utf-8")
    before = ledger.read_bytes()
    assert (
        main(["--records-root", str(records_root), "--repo-root", str(repo_root)]) == 1
    )
    assert "repair-models-ledger failed" in capsys.readouterr().err
    assert ledger.read_bytes() == before, "a refused repair must not truncate"
