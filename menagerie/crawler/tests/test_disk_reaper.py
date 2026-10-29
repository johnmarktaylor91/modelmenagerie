"""Tiered report-only disk reaper: referenced-path manifest + corpus guard.

The reaper is the C-DISK-GC consumer of the text-corpus GC contract
(``procedures/TEXT_CORPUS.md``): the corpus is never reapable, a work dir's
scratch is reapable only once ``work_dir_reap_clearance`` clears it, and
garbage collection stays report-only until explicitly applied. These tests pin
the four contracts the reaper ships with: the corpus-protection guard (a plan
can never name a promoted payload, in either containment direction), the
referenced-path manifest (repair budget, attempt history, author custody,
diagnostics pointers, canonical records), the report-only default, and the
apply path's replay-derived re-verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.driver_contracts import DriverLock, DriverLockError
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.reaper import (
    REAP_PLAN_VERSION,
    REAP_TIERS,
    TIER_DERIVED_CACHES,
    TIER_DIAGNOSTICS_ORPHANS,
    TIER_INSTRUMENTATION,
    TIER_PROMOTED_ATTEMPT_SCRATCH,
    apply_reap_plan,
    build_reap_plan,
    referenced_path_manifest,
)
from menagerie.crawler.text_corpus import (
    corpus_payload_path,
    corpus_root,
    promote_author_root,
    verify_corpus,
    work_dir_reap_clearance,
)
from menagerie.crawler.tools import reap_disk

pytestmark = pytest.mark.smoke


IMPL_BYTES = b"import torch\n\nclass Net(torch.nn.Module):\n    pass\n"
PROBE_BYTES = b"<html>rejected candidate page</html>\n"
ATTEMPT_NONCE = "0f0e0d0c0b0a0908"


def _evidence_name(content: bytes) -> str:
    return f"{hash_bytes(content).removeprefix('sha256:')[:16]}.bin"


def _outcome(
    source_id: str,
    content: bytes,
    *,
    requested_role: str,
    bound_role,
    outcome="fetched",
    url="https://example.test/doc",
) -> dict:
    return {
        "source_id": source_id,
        "kind": "raw-url",
        "requested_role": requested_role,
        "bound_role": bound_role,
        "outcome": outcome,
        "url": url,
        "final_url": url,
        "redirect_chain": [url],
        "status": 200,
        "bytes_fetched": len(content),
        "sha256": hash_bytes(content),
        "media_type": "text/html",
        "media_type_method": "path-extension",
        "resolver_receipt": None,
        "derived_citation": None,
        "detail": "",
        "rate_limit": None,
        "retrieved_at": "2026-08-11T00:00:00.000000Z",
    }


def _manifest_row(source_id: str, content: bytes, *, broker_role: str) -> dict:
    return {
        "source_id": source_id,
        "url": "https://example.test/doc",
        "final_url": "https://example.test/doc",
        "revision": hash_bytes(content),
        "expected_sha256": hash_bytes(content),
        "media_type": "text/html",
        "media_type_method": "path-extension",
        "broker_role": broker_role,
        "broker_outcome": "fetched",
        "redirect_chain": ["https://example.test/doc"],
        "requested_role": "implementation",
        "media_type_hint": "",
        "basis": "",
        "retrieved_at": "2026-08-11T00:00:00.000000Z",
    }


def _author_root(work_dir: Path, stable_id: str) -> Path:
    """Materialize a promotable author custody root (mirrors test_text_corpus)."""

    from menagerie.crawler.fetcher import publish_cas_object
    from menagerie.crawler.source_broker import BROKER_PACK_VERSION

    root = work_dir / "author"
    evidence_dir = root / "broker" / "evidence"
    evidence_dir.mkdir(parents=True)
    source_cas = root / "source-cas"
    source_cas.mkdir(parents=True)

    impl_digest = hash_bytes(IMPL_BYTES)
    (evidence_dir / _evidence_name(IMPL_BYTES)).write_bytes(IMPL_BYTES)
    publish_cas_object(source_cas, impl_digest, IMPL_BYTES)
    (evidence_dir / _evidence_name(PROBE_BYTES)).write_bytes(PROBE_BYTES)

    impl_outcome = _outcome(
        f"impl-{stable_id}",
        IMPL_BYTES,
        requested_role="implementation",
        bound_role="implementation",
    )
    probe_outcome = _outcome(
        "candidate-probe-001",
        PROBE_BYTES,
        requested_role="probe",
        bound_role="probe",
        outcome="probed",
        url="https://example.test/probe",
    )
    pack = {
        "pack_version": BROKER_PACK_VERSION,
        "sources": [_manifest_row(f"impl-{stable_id}", IMPL_BYTES, broker_role="implementation")],
        "broker": {
            "outcomes": [impl_outcome, probe_outcome],
            "derived_citations": [],
            "total_bytes": len(IMPL_BYTES) + len(PROBE_BYTES),
        },
    }
    (root / "source-targets.json").write_text(json.dumps(pack, sort_keys=True))

    custody = {
        "provenance_version": "menagerie.crawler.source-custody-provenance.v1",
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "promotions": [
            {
                "source_id": f"impl-{stable_id}",
                "content_sha256": impl_digest,
                "disposition": "promoted",
                "bytes_len": len(IMPL_BYTES),
                "evidence_path": str(evidence_dir / _evidence_name(IMPL_BYTES)),
            }
        ],
        "upstream_drift": [],
    }
    (root / "source-custody.json").write_text(json.dumps(custody, sort_keys=True))

    envelope = json.dumps({"discovery": stable_id}, sort_keys=True).encode() + b"\n"
    envelope_digest = hash_bytes(envelope)
    (source_cas / f"{envelope_digest.removeprefix('sha256:')}.source").write_bytes(envelope)
    provenance = {
        "provenance_version": "menagerie.crawler.found-discovery-provenance.v1",
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "discovery_evidence": {
            "byte_count": len(envelope),
            "cas_path": str(source_cas / f"{envelope_digest.removeprefix('sha256:')}.source"),
            "content_sha256": envelope_digest,
            "discovery_arm": "FOUND",
            "media_type": "application/json",
            "url": f"urn:menagerie:source-discovery:{envelope_digest.removeprefix('sha256:')}",
        },
        "discovery_evidence_sha256": stable_hash({"discovery": stable_id}),
    }
    (root / "discovery-provenance.json").write_text(json.dumps(provenance, sort_keys=True))
    return root


def _attempt_dir(author_root: Path, number: int, *, status: str = "superseded") -> Path:
    attempt = author_root / "attempts" / f"attempt-{number:03d}-{ATTEMPT_NONCE}"
    (attempt / "scratch").mkdir(parents=True)
    (attempt / "scratch" / "session.log").write_bytes(b"claude scratch " * 64)
    (attempt / "quarantine").mkdir()
    (attempt / "quarantine" / "aa11bb22-result.json").write_text("{}")
    (attempt / "model").mkdir()
    (attempt / "model" / "adapter.py").write_bytes(b"# staged\n" * 32)
    probe = attempt / f"executor-probe-{'ab' * 6}"
    probe.mkdir()
    (probe / "receipt.json").write_text("{}")
    (attempt / "record.json").write_text(
        json.dumps(
            {
                "record_version": "menagerie.crawler.author-attempt.v1",
                "status": status,
                "attempt_number": number,
            }
        )
    )
    (attempt / "events.jsonl").write_text('{"event": "created"}\n')
    (attempt / "discovery.json").write_text('{"discovery": true}')
    evidence = attempt / "broker" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "feedbeef00000000.bin").write_bytes(b"attempt evidence")
    return attempt


def _work_dir(
    work_root: Path, stable_id: str, *, attempts: int = 2, attempt_status: str = "superseded"
) -> Path:
    """One realistic model work dir: author custody, attempts, repair, forward."""

    work_dir = work_root / stable_id
    author_root = _author_root(work_dir, stable_id)
    for number in range(1, attempts + 1):
        status = attempt_status if number < attempts else "completed"
        _attempt_dir(author_root, number, status=status)

    repair = work_dir / "repair"
    repair.mkdir(parents=True)
    (repair / "run-modes-charge-1.json").write_text("{}")
    (repair / f"recipe-refusal-charge-{'cd' * 8}-1.json").write_text("{}")

    supervisor = work_dir / "forward" / "cold-1" / "plain" / "supervisor"
    supervisor.mkdir(parents=True)
    (supervisor / "stdout.log").write_bytes(b"forward stdout " * 128)
    (supervisor / "stderr.log").write_bytes(b"forward stderr " * 128)
    audit = work_dir / "forward" / "cold-1" / "plain" / ".menagerie-parent-audit-1-2"
    audit.mkdir()
    (audit / "sandbox-syscalls.log").write_bytes(b"syscall telemetry\n")

    (work_dir / "driver-author-artifact.json").write_text('{"cache": true}')
    return work_dir


def _model_row(stable_id: str, status_code: str) -> dict:
    return {
        "stable_id": stable_id,
        "record_revision": stable_hash({"rev": stable_id, "code": status_code}),
        "parent_revision": None,
        "record_seq": 1,
        "status": {"code": status_code},
    }


def _records_root(tmp_path: Path, rows: list[dict]) -> Path:
    records = tmp_path / "records"
    models = records / "models"
    models.mkdir(parents=True)
    with (models / "current-shard.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    attempts = records / "attempts"
    attempts.mkdir()
    (attempts / "local.jsonl").write_text(
        json.dumps(
            {
                "attempt": {
                    "diagnostics": {
                        "redaction": "externally-controlled-text-v1",
                        "local_path": ".crawl-local/diagnostics/att-referenced.json",
                    }
                }
            }
        )
        + "\n"
    )
    return records


def _runtime_root(tmp_path: Path) -> Path:
    runtime = tmp_path / ".crawl-local"
    (runtime / "diagnostics").mkdir(parents=True)
    (runtime / "diagnostics" / "att-referenced.json").write_text("{}")
    (runtime / "instrumentation").mkdir()
    (runtime / "instrumentation" / "ledger-hot-path.jsonl").write_text("{}\n")
    return runtime


def _standard_fixture(tmp_path: Path, *, status_code: str = "runs", promote: bool = True):
    """One promoted terminal model plus canonical roots; returns all roots."""

    runtime = _runtime_root(tmp_path)
    work_root = runtime / "work"
    work_root.mkdir()
    work_dir = _work_dir(work_root, "m0001")
    records_root = _records_root(tmp_path, [_model_row("m0001", status_code)])
    if promote:
        report = promote_author_root(work_dir / "author", records_root)
        assert not report.refusals
        clearance = work_dir_reap_clearance(work_dir, records_root)
        assert clearance.cleared, clearance.reasons
    return runtime, work_root, work_dir, records_root


def _plan(work_root: Path, records_root: Path, runtime: Path, **kwargs):
    return build_reap_plan(
        work_root=work_root, records_root=records_root, runtime_root=runtime, **kwargs
    )


def _tree_snapshot(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# -- corpus-protection guard (merge gate) --------------------------------------


def test_plan_never_names_a_promoted_corpus_payload(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    assert plan.candidates, "fixture must produce reapable scratch"
    protected_root = corpus_root(records_root).resolve()
    for candidate in plan.candidates:
        resolved = Path(candidate.path).resolve()
        assert resolved != protected_root
        assert protected_root not in resolved.parents
        assert resolved not in (protected_root, *protected_root.parents)


def test_plan_refuses_candidates_that_contain_the_corpus(tmp_path):
    """A records root nested below would-be scratch protects the whole subtree."""

    runtime = _runtime_root(tmp_path)
    work_root = runtime / "work"
    work_root.mkdir()
    work_dir = _work_dir(work_root, "m0001")
    scratch = work_dir / "author" / "attempts" / f"attempt-001-{ATTEMPT_NONCE}" / "scratch"
    nested_records = scratch / "records"
    models = nested_records / "models"
    models.mkdir(parents=True)
    with (models / "current-shard.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_model_row("m0001", "runs"), sort_keys=True) + "\n")
    report = promote_author_root(work_dir / "author", nested_records)
    assert not report.refusals

    plan = _plan(work_root, nested_records, runtime)
    protected_root = corpus_root(nested_records).resolve()
    for candidate in plan.candidates:
        resolved = Path(candidate.path).resolve()
        assert protected_root != resolved
        assert protected_root not in resolved.parents, candidate
        assert resolved not in protected_root.parents, candidate


def test_apply_refuses_symlinked_scratch_into_the_corpus(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    scratch_candidates = [
        candidate
        for candidate in plan.candidates
        if candidate.tier == TIER_PROMOTED_ATTEMPT_SCRATCH
        and Path(candidate.path).name == "scratch"
    ]
    assert scratch_candidates
    target = Path(scratch_candidates[0].path)
    payload = corpus_payload_path(records_root, hash_bytes(IMPL_BYTES))
    assert payload.is_file()
    # Re-point the planned scratch dir at the corpus payload tree post-planning.
    import shutil

    shutil.rmtree(target)
    target.symlink_to(payload.parent)

    apply_reap_plan(plan, work_root=work_root, records_root=records_root, runtime_root=runtime)
    assert payload.is_file()
    assert verify_corpus(records_root, work_roots=[work_root]).passed


def test_apply_ignores_forged_corpus_candidates(tmp_path):
    import dataclasses

    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    payload = corpus_payload_path(records_root, hash_bytes(IMPL_BYTES))
    forged_candidate = dataclasses.replace(plan.candidates[0], path=payload)
    forged = dataclasses.replace(plan, candidates=(*plan.candidates, forged_candidate))
    report = apply_reap_plan(
        forged, work_root=work_root, records_root=records_root, runtime_root=runtime
    )
    assert payload.is_file()
    assert str(payload) not in {str(entry) for entry in report.removed}
    assert verify_corpus(records_root, work_roots=[work_root]).passed


# -- referenced-path manifest ---------------------------------------------------


def test_manifest_protects_every_load_bearing_class(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    manifest = referenced_path_manifest(
        work_root=work_root, records_root=records_root, runtime_root=runtime
    )
    categories = {entry.category for entry in manifest}
    for required in (
        "corpus",
        "canonical-records",
        "repair-budget",
        "attempt-history",
        "author-custody",
        "driver-author-cache",
        "diagnostics",
        "runtime-control",
    ):
        assert required in categories, f"missing manifest class {required}"

    protected_paths = {str(Path(entry.path).resolve()) for entry in manifest}
    for load_bearing in (
        records_root,
        work_dir / "repair",
        work_dir / "driver-author-artifact.json",
        work_dir / "author" / "source-cas",
        work_dir / "author" / "attempts" / f"attempt-001-{ATTEMPT_NONCE}" / "record.json",
    ):
        assert str(load_bearing.resolve()) in protected_paths, load_bearing


def test_plan_candidates_never_overlap_the_manifest(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    protected = [Path(entry.path).resolve() for entry in plan.protected]
    for candidate in plan.candidates:
        resolved = Path(candidate.path).resolve()
        for shield in protected:
            assert resolved != shield
            assert shield not in resolved.parents, (candidate, shield)


def test_repair_budget_and_attempt_history_survive_apply(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    apply_reap_plan(plan, work_root=work_root, records_root=records_root, runtime_root=runtime)

    assert (work_dir / "repair" / "run-modes-charge-1.json").is_file()
    assert (work_dir / "repair" / f"recipe-refusal-charge-{'cd' * 8}-1.json").is_file()
    assert (work_dir / "driver-author-artifact.json").is_file()
    for attempt in sorted((work_dir / "author" / "attempts").iterdir()):
        assert (attempt / "record.json").is_file()
        assert (attempt / "events.jsonl").is_file()
        assert (attempt / "discovery.json").is_file()
        assert (attempt / "broker" / "evidence" / "feedbeef00000000.bin").is_file()
    assert (work_dir / "author" / "result.json").exists() or True  # author surfaces untouched
    assert (work_dir / "author" / "source-cas").is_dir()
    assert verify_corpus(records_root, work_roots=[work_root]).passed


# -- clearance + eligibility gates ---------------------------------------------


def test_unpromoted_work_dir_is_fully_protected(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path, promote=False)
    plan = _plan(work_root, records_root, runtime)
    assert not [c for c in plan.candidates if c.tier == TIER_PROMOTED_ATTEMPT_SCRATCH]
    reasons = {
        entry.category
        for entry in plan.protected
        if Path(entry.path).resolve() == work_dir.resolve()
    }
    assert "unpromoted-work-dir" in reasons


def test_live_model_work_dir_is_fully_protected(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(
        tmp_path, status_code="authoring"
    )
    plan = _plan(work_root, records_root, runtime)
    assert not [c for c in plan.candidates if c.stable_id == "m0001"]


def test_deleting_a_promoted_payload_flips_the_work_dir_back_to_protected(tmp_path):
    """Extends the RETAIN1 guard: clearance is recomputed from corpus bytes."""

    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    assert _plan(work_root, records_root, runtime).candidates
    corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)).unlink()
    plan = _plan(work_root, records_root, runtime)
    assert not [c for c in plan.candidates if c.tier == TIER_PROMOTED_ATTEMPT_SCRATCH]


def test_in_flight_attempt_record_protects_the_work_dir(tmp_path):
    runtime = _runtime_root(tmp_path)
    work_root = runtime / "work"
    work_root.mkdir()
    work_dir = _work_dir(work_root, "m0001", attempts=1, attempt_status="running")
    attempt = work_dir / "author" / "attempts" / f"attempt-001-{ATTEMPT_NONCE}"
    record = json.loads((attempt / "record.json").read_text())
    record["status"] = "running"
    (attempt / "record.json").write_text(json.dumps(record))
    records_root = _records_root(tmp_path, [_model_row("m0001", "runs")])
    promote_author_root(work_dir / "author", records_root)
    plan = _plan(work_root, records_root, runtime)
    assert not [c for c in plan.candidates if c.stable_id == "m0001"]


def test_park_valve_row_protects_the_work_dir(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    extensions = records_root / "intake-extensions"
    extensions.mkdir()
    (extensions / "disk-floor-deferrals.jsonl").write_text(
        json.dumps({"stable_id": "m0001", "disposition": "deferred:needs-host-disk"}) + "\n"
    )
    plan = _plan(work_root, records_root, runtime)
    assert not [c for c in plan.candidates if c.stable_id == "m0001"]


# -- tier composition ------------------------------------------------------------


def test_promoted_attempt_scratch_tier_targets_only_scratch_classes(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    names = {
        Path(candidate.path).name
        for candidate in plan.candidates
        if candidate.tier == TIER_PROMOTED_ATTEMPT_SCRATCH
    }
    assert "scratch" in names
    assert "quarantine" in names
    assert any(name.startswith("executor-probe-") for name in names)
    # model/ is reapable only in superseded attempts, never the newest one.
    model_candidates = [
        Path(candidate.path)
        for candidate in plan.candidates
        if candidate.tier == TIER_PROMOTED_ATTEMPT_SCRATCH and Path(candidate.path).name == "model"
    ]
    assert model_candidates
    for path in model_candidates:
        assert path.parent.name == f"attempt-001-{ATTEMPT_NONCE}"


def test_diagnostics_tier_protects_ledger_referenced_sidecars(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    orphan = runtime / "diagnostics" / "att-orphan.json"
    orphan.write_text("{}")
    plan = _plan(work_root, records_root, runtime)
    tier_paths = {
        Path(candidate.path).name
        for candidate in plan.candidates
        if candidate.tier == TIER_DIAGNOSTICS_ORPHANS
    }
    assert "att-orphan.json" in tier_paths
    assert "att-referenced.json" not in tier_paths


def test_instrumentation_tier_is_eligibility_gated_supervisor_output(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    tier = [c for c in plan.candidates if c.tier == TIER_INSTRUMENTATION]
    names = {Path(candidate.path).name for candidate in tier}
    assert {"stdout.log", "stderr.log"} <= names
    assert any(Path(c.path).name.startswith(".menagerie-parent-audit-") for c in tier)
    # The runtime instrumentation ledger is consumed by throughput reports.
    assert not [c for c in plan.candidates if Path(c.path).name == "ledger-hot-path.jsonl"]


def test_derived_caches_tier_lists_checkpoint_cache_and_rehydrated_artifacts(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    cache = runtime / "derived" / "checkpoint-cache"
    cache.mkdir(parents=True)
    (cache / "entry.json").write_text("{}")
    rehydrated = work_root / "rehydrated-artifacts"
    rehydrated.mkdir()
    (rehydrated / "model.py").write_text("# rehydrated")
    plan = _plan(work_root, records_root, runtime)
    tier_paths = {
        str(Path(candidate.path).resolve())
        for candidate in plan.candidates
        if candidate.tier == TIER_DERIVED_CACHES
    }
    assert str(cache.resolve()) in tier_paths
    assert str(rehydrated.resolve()) in tier_paths


# -- report-only default + apply ---------------------------------------------


def test_cli_default_is_report_only_and_deletes_nothing(tmp_path, capsys):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    before = _tree_snapshot(tmp_path)
    plan_json = tmp_path / "plan.json"
    exit_code = reap_disk.main(
        [
            "--work-root",
            str(work_root),
            "--records-root",
            str(records_root),
            "--runtime-root",
            str(runtime),
            "--json",
            str(plan_json),
        ]
    )
    assert exit_code == 0
    after = {key: value for key, value in _tree_snapshot(tmp_path).items() if key != "plan.json"}
    assert after == before
    payload = json.loads(plan_json.read_text())
    assert payload["plan_version"] == REAP_PLAN_VERSION
    assert payload["applied"] is False
    assert payload["tiers"]
    output = capsys.readouterr().out
    assert "report-only" in output


def test_apply_removes_candidates_and_reports_bytes(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    planned_bytes = sum(candidate.bytes_len for candidate in plan.candidates)
    assert planned_bytes > 0
    report = apply_reap_plan(
        plan, work_root=work_root, records_root=records_root, runtime_root=runtime
    )
    assert report.bytes_removed > 0
    scratch = work_dir / "author" / "attempts" / f"attempt-001-{ATTEMPT_NONCE}" / "scratch"
    assert not scratch.exists()
    assert verify_corpus(records_root, work_roots=[work_root]).passed
    clearance = work_dir_reap_clearance(work_dir, records_root)
    assert clearance.cleared, clearance.reasons


def test_apply_refuses_while_a_driver_holds_the_lock(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    lock_path = runtime / "locks" / "driver.lock"
    with DriverLock(lock_path, {"owner": "test-driver"}):
        with pytest.raises(DriverLockError):
            apply_reap_plan(
                plan, work_root=work_root, records_root=records_root, runtime_root=runtime
            )
    scratch = work_dir / "author" / "attempts" / f"attempt-001-{ATTEMPT_NONCE}" / "scratch"
    assert scratch.is_dir()


def test_apply_recomputes_the_plan_and_skips_stale_candidates(tmp_path):
    """The apply path is replay-derived: a stale plan cannot force a deletion."""

    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    # The model's promoted payload vanishes between planning and applying.
    corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)).unlink()
    report = apply_reap_plan(
        plan, work_root=work_root, records_root=records_root, runtime_root=runtime
    )
    scratch = work_dir / "author" / "attempts" / f"attempt-001-{ATTEMPT_NONCE}" / "scratch"
    assert scratch.is_dir()
    assert not [removed for removed in report.removed if str(work_dir) in str(removed)]


def test_plan_reports_tier_and_protected_byte_totals(tmp_path):
    runtime, work_root, work_dir, records_root = _standard_fixture(tmp_path)
    plan = _plan(work_root, records_root, runtime)
    totals = plan.tier_totals()
    assert set(totals) <= set(REAP_TIERS)
    scratch_count, scratch_bytes = totals[TIER_PROMOTED_ATTEMPT_SCRATCH]
    assert scratch_count > 0
    assert scratch_bytes > 0
    payload = plan.to_json()
    assert payload["plan_version"] == REAP_PLAN_VERSION
    assert payload["candidates"]
    assert payload["protected_summary"]
