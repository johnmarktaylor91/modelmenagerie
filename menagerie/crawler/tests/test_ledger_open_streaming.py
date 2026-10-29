"""Ledger open/append data-plane costs pinned as behavioral contracts (TW-A).

C-LEDGER-OPEN (T1-12 + T4-T6): ``JsonlLedger`` startup ran ``recover_torn_tail``
(a whole-file ``read_bytes`` plus one full validated scan) and then a SECOND
full validated scan, so every open paid ~2x its own corpus; every append
re-walked complete history 2-3 times; and ``append_model`` materialized four
whole-ledger deepcopies per append. These tests pin the fixed contracts:

1. a clean open runs exactly ONE validated scan (each record validates once);
2. ``recover_torn_tail`` proves a clean tail by seeking, never by reading the
   whole shard into memory;
3. a reopen whose previously validated prefix still matches the recorded
   open-state digest re-validates ONLY rows appended since (streaming
   validation), and ANY anomaly -- tampered bytes, tampered or missing
   sidecar, changed schema environment -- fails closed to the full validated
   scan (never a silently trusted read);
4. torn-tail recovery evidence and truncation (V1) survive unchanged;
5. appends resolve replay/conflict/sequence through indexes, not whole-history
   rescans, and ``append_model`` performs no whole-ledger deepcopy.

C-CHECKPOINT-COST (T4-T12, verified real): checkpoint candidate derivation
re-loaded and re-hashed EVERY intake snapshot ever staged on EVERY call; the
fix verifies each snapshot once per process and re-verifies exactly when any
snapshot byte changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

import menagerie.crawler.checkpoint as checkpoint_module
import menagerie.crawler.recordio as recordio_module
import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.artifact_transactions import ArtifactEventLedger
from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import (
    JsonlLedger,
    LedgerConflictError,
    LedgerCorruptionError,
    open_state_path,
    recover_torn_tail,
    scan_jsonl,
)
from menagerie.crawler.tests.conftest import make_attempt, make_gate, make_model
from menagerie.crawler.tests.test_reducer import (
    CanonicalReducer as AdapterCanonicalReducer,
)

#: Sub-second load-bearing coverage promoted into the mandatory smoke tier (T5-7).
pytestmark = pytest.mark.smoke


def _write_attempt_ledger(path: Path, count: int) -> None:
    """Materialize a valid attempts ledger with ``count`` distinct rows."""

    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        for index in range(count):
            payload = make_attempt(attempt_id=f"attempt-{index}")
            payload.pop("ledger_seq", None)
            payload.pop("payload_sha256", None)
            ledger.append(payload)


def _count_validations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count ``validate_payload`` passes made through the recordio scan path."""

    counter = [0]
    real = recordio_module.validate_payload

    def counting(payload: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
        counter[0] += 1
        real(payload, *args, **kwargs)

    monkeypatch.setattr(recordio_module, "validate_payload", counting)
    return counter


def test_clean_open_validates_each_record_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold clean open is ONE validated scan, not the historical two."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 3)
    open_state_path(path).unlink(missing_ok=True)
    counter = _count_validations(monkeypatch)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert ledger.record_count == 3
    assert counter[0] == 3


def test_recover_torn_tail_never_reads_a_clean_ledger_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newline-terminated ledger proves its tail by seek, not read_bytes."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 2)
    real_read_bytes = Path.read_bytes
    whole_reads = [0]

    def counting_read_bytes(self: Path) -> bytes:
        if self == path:
            whole_reads[0] += 1
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    assert recover_torn_tail(path) is None
    assert whole_reads[0] == 0


def test_warm_reopen_skips_revalidating_digest_held_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reopen whose recorded prefix digest still holds validates zero rows."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 3)
    assert open_state_path(path).is_file()
    counter = _count_validations(monkeypatch)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert ledger.record_count == 3
        assert [record["attempt_id"] for record in ledger.iter_records()] == [
            "attempt-0",
            "attempt-1",
            "attempt-2",
        ]
    assert counter[0] == 0


def test_warm_reopen_validates_only_externally_appended_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open cost is O(new rows): only the suffix beyond the digest validates."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 3)
    spare = tmp_path / "spare.jsonl"
    with JsonlLedger(spare, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        payload = make_attempt(attempt_id="attempt-99")
        payload.pop("ledger_seq", None)
        payload.pop("payload_sha256", None)
        appended = ledger.append(payload).record
    appended["ledger_seq"] = 4
    appended.pop("payload_sha256", None)
    appended["payload_sha256"] = recordio_module.payload_hash(appended)
    with path.open("ab") as handle:
        handle.write(json.dumps(appended, sort_keys=True).encode() + b"\n")
    counter = _count_validations(monkeypatch)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert ledger.record_count == 4
    assert counter[0] == 1


def test_prefix_byte_tamper_fails_closed_on_reopen(tmp_path: Path) -> None:
    """A changed prefix byte voids the digest and the full rescan refuses."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 3)
    tampered = path.read_bytes().replace(b'"attempt-1"', b'"attempt-x"', 1)
    path.write_bytes(tampered)
    with pytest.raises(LedgerCorruptionError):
        JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)


def test_forged_sidecar_over_corrupt_bytes_still_refuses(tmp_path: Path) -> None:
    """A sidecar regenerated to match corrupted bytes cannot smuggle them in.

    The held prefix digest skips ONLY the jsonschema pass; every record's own
    content digest re-verifies live (the W5 ValidationMemo doctrine). Rehearsed
    live pre-fix: a corrupt row with an invalid ``payload_sha256`` was served
    as canonical history whenever the sidecar's whole-prefix digest was
    recomputed over the corrupted bytes.
    """

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 3)
    corrupted = path.read_bytes().replace(b'"attempt-1"', b'"attempt-x"', 1)
    path.write_bytes(corrupted)
    state = json.loads(open_state_path(path).read_bytes())
    state["byte_count"] = len(corrupted)
    state["line_count"] = 3
    state["prefix_sha256"] = recordio_module.hashlib.sha256(corrupted).hexdigest()
    open_state_path(path).write_bytes(json.dumps(state).encode())
    with pytest.raises(LedgerCorruptionError, match="payload_sha256"):
        JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)


def test_refused_open_publishes_no_open_state(tmp_path: Path) -> None:
    """A writer that refuses its open never overwrites the good sidecar."""

    from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 2)
    good = open_state_path(path).read_bytes()
    with pytest.raises(LedgerCorruptionError):
        JsonlLedger(path, ATTEMPT_SCHEMA_VERSION)  # v2 writer over v3 rows
    assert open_state_path(path).read_bytes() == good


def test_sidecar_corruption_falls_back_to_full_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed open-state sidecar degrades to the full validated scan."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 3)
    open_state_path(path).write_bytes(b"{not json")
    counter = _count_validations(monkeypatch)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert ledger.record_count == 3
    assert counter[0] == 3


def test_schema_environment_change_falls_back_to_full_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A memoized jsonschema pass never survives a schema-environment change."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 3)
    monkeypatch.setattr(recordio_module, "_open_state_environment", lambda: "other-environment")
    counter = _count_validations(monkeypatch)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert ledger.record_count == 3
    assert counter[0] == 3


def test_warm_reopen_reads_mixed_version_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest-held prefix containing legacy v2 rows reopens exactly once.

    Cross-version reads must survive the streaming open: old rows under the
    new writer parse from the held prefix (jsonschema skipped, content digest
    still verified), and the read-version tripwire still walks every record.
    """

    from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION

    path = tmp_path / "attempts.jsonl"
    legacy = make_attempt(attempt_id="attempt-legacy")
    legacy["schema_version"] = ATTEMPT_SCHEMA_VERSION
    for field in (
        "capability_observation",
        "execution_read_manifest_identity",
        "raw_award_receipt",
        "raw_award_receipt_sha256",
        "parent_attestation",
        "unattested_partial",
    ):
        legacy.pop(field, None)
    for field in (
        "authority_epoch",
        "base_environment_generation",
        "environment_content_sha256",
        "environment_authority_id",
        "selected_interpreter_relative_path",
        "selected_interpreter_digest",
        "external_escape_records",
    ):
        legacy["environment"].pop(field, None)
    legacy["worker_receipt"].pop("output_value_sha256", None)
    legacy.pop("ledger_seq", None)
    legacy.pop("payload_sha256", None)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION) as ledger:
        ledger.append(legacy)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        payload = make_attempt(attempt_id="attempt-current")
        payload.pop("ledger_seq", None)
        payload.pop("payload_sha256", None)
        ledger.append(payload)
    # third open: the v3 sidecar covers the mixed prefix; zero jsonschema passes
    counter = _count_validations(monkeypatch)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        versions = [record["schema_version"] for record in ledger.iter_records()]
    assert versions == [ATTEMPT_SCHEMA_VERSION, ATTEMPT_SCHEMA_VERSION_V3]
    assert counter[0] == 0


def test_torn_tail_recovery_survives_streaming_open(tmp_path: Path) -> None:
    """V1 recovery -- evidence then truncation -- is unchanged by the sidecar."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 3)
    good_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(b'{"torn": ')
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert ledger.recovery_evidence is not None
        assert ledger.recovery_evidence.byte_offset == good_size
        assert ledger.record_count == 3
    assert path.stat().st_size == good_size
    evidence_path = path.with_suffix(f"{path.suffix}.recovery.jsonl")
    assert evidence_path.is_file()


def test_append_replay_conflict_and_sequence_survive_trusted_prefix_open(
    tmp_path: Path,
) -> None:
    """Idempotency, conflict refusal, and seq allocation hold on a warm open."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 3)
    replay = make_attempt(attempt_id="attempt-1")
    replay.pop("ledger_seq", None)
    replay.pop("payload_sha256", None)
    conflicting = json.loads(json.dumps(replay))
    conflicting["created_at"] = "2030-01-01T00:00:00Z"
    fresh = make_attempt(attempt_id="attempt-3")
    fresh.pop("ledger_seq", None)
    fresh.pop("payload_sha256", None)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert not ledger.append(replay).appended
        with pytest.raises(LedgerConflictError):
            ledger.append(conflicting)
        result = ledger.append(fresh)
        assert result.appended
        assert result.record["ledger_seq"] == 4
    assert len(scan_jsonl(path)) == 4


def test_append_resolves_identity_without_rescanning_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One append derives identity keys O(1) times, not once per stored row."""

    path = tmp_path / "attempts.jsonl"
    _write_attempt_ledger(path, 40)
    real_identity_key = recordio_module._identity_key
    calls = [0]

    def counting_identity_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        calls[0] += 1
        return real_identity_key(record)

    monkeypatch.setattr(recordio_module, "_identity_key", counting_identity_key)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        calls[0] = 0
        payload = make_attempt(attempt_id="attempt-40")
        payload.pop("ledger_seq", None)
        payload.pop("payload_sha256", None)
        assert ledger.append(payload).appended
    assert calls[0] <= 8


def test_append_model_performs_no_whole_ledger_deepcopy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The append path never materializes four whole-ledger deepcopies."""

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    with AdapterCanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_gate(make_gate(["m_example"]))
        reducer.append_attempt(make_attempt("m_example", attempt_id="attempt-1"))

        def refusing_records(self: JsonlLedger) -> Any:
            raise AssertionError("append_model must not deepcopy a whole ledger")

        def refusing_events(self: ArtifactEventLedger) -> Any:
            raise AssertionError("append_model must not deepcopy the artifact ledger")

        monkeypatch.setattr(JsonlLedger, "records", property(refusing_records))
        monkeypatch.setattr(ArtifactEventLedger, "events", property(refusing_events))
        result = reducer.append_model(make_model("m_example", accepted=True))
        assert result.appended


def test_open_state_sidecar_is_never_a_checkpoint_candidate(tmp_path: Path) -> None:
    """The derived sidecar's final suffix stays outside the public allowlist."""

    sidecar = open_state_path(tmp_path / "records" / "models" / "current-shard.jsonl")
    assert sidecar.suffix == ".open-state"
    assert sidecar.suffix not in checkpoint_module._ALLOWLIST_SUFFIXES


def test_requeue_lineage_accepts_one_shot_evidence_iterators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence iterables are re-read once per grant revision, so they must
    be materialized inside the validator: the pilot corpus (two grant-bearing
    lineages) refused falsely when the second revision saw an already
    exhausted attempts iterator."""

    seen_evidence: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []

    def recording(revision: Mapping[str, Any], attempts: Any, gates: Any, events: Any = ()) -> Any:
        seen_evidence.append((tuple(attempts), tuple(gates)))

        class _Anything:
            def __contains__(self, item: object) -> bool:
                return True

        return _Anything()

    monkeypatch.setattr(reducer_module, "_revision_work_ids", recording)
    monkeypatch.setattr(reducer_module, "expected_grant_id", lambda grant: grant["grant_id"])
    grant_rows = tuple(
        {"grant_id": f"grant-{index}", "stable_id": f"m_{index}"} for index in range(2)
    )
    event_rows = tuple(
        {
            "event_kind": "requeue-grant-consumed",
            "details": {
                "grant_id": f"grant-{index}",
                "stable_id": f"m_{index}",
                "source_record_revision": f"rev-{index}",
                "new_work_generation": 1,
                "new_work_id": reducer_module.stable_hash(
                    {
                        "stable_id": f"m_{index}",
                        "grant_id": f"grant-{index}",
                        "parent_revision": f"rev-{index}",
                        "generation": 1,
                    }
                ),
            },
        }
        for index in range(2)
    )
    monkeypatch.setattr(
        reducer_module,
        "_scan_operational_evidence_cached",
        lambda path, validate=True: grant_rows if "grants" in path.name else event_rows,
    )
    revisions = [
        {
            "stable_id": f"m_{index}",
            "parent_revision": f"rev-{index}",
            "budget": {"explicit_grants": [f"grant-{index}"]},
        }
        for index in range(2)
    ]
    attempt_rows = ({"attempt_id": "attempt-1", "work_id": "work-1"},)
    reducer_module._validate_persisted_requeue_lineage(
        iter(revisions),
        LedgerPaths(
            models=Path("records/models/x.jsonl"),
            attempts=Path("records/attempts/x.jsonl"),
            gates=Path("records/gates/x.jsonl"),
        ),
        iter(attempt_rows),
        iter(()),
        iter(()),
    )
    assert len(seen_evidence) == 2
    assert seen_evidence[0][0] == attempt_rows
    assert seen_evidence[1][0] == attempt_rows


def _checkpoint_records_root(tmp_path: Path, snapshots: int) -> Path:
    """Materialize a canonical root holding ``snapshots`` real intake snapshots."""

    root = tmp_path / "repo"
    canonical_root = root / "menagerie" / "crawler"
    intake_root = canonical_root / "records" / "intake"
    intake_root.mkdir(parents=True)
    for index in range(snapshots):
        master = tmp_path / f"master-{index}.jsonl"
        deferred = tmp_path / f"deferred-{index}.jsonl"
        master.write_text(
            json.dumps({"name": f"Example-{index}", "zoo": "tests", "variant": "base"}) + "\n"
        )
        deferred.write_text("")
        create_intake_snapshot(master, deferred, intake_root)
    return root


def test_checkpoint_snapshot_verification_is_cached_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged staged snapshots verify once per process, not per checkpoint."""

    root = _checkpoint_records_root(tmp_path, snapshots=2)
    canonical_root = root / "menagerie" / "crawler"
    real_load = checkpoint_module.load_intake_snapshot
    loads = [0]

    def counting_load(snapshot_root: Path) -> Any:
        loads[0] += 1
        return real_load(snapshot_root)

    monkeypatch.setattr(checkpoint_module, "load_intake_snapshot", counting_load)
    checkpoint_module._reset_intake_snapshot_verification_memo()
    checkpoint_module._derive_candidate_paths(root, canonical_root, ())
    first = loads[0]
    assert first == 2
    checkpoint_module._derive_candidate_paths(root, canonical_root, ())
    assert loads[0] == first
    checkpoint_module._reset_intake_snapshot_verification_memo()


def test_checkpoint_snapshot_reverifies_on_any_byte_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any snapshot byte change voids the memo and verification refuses again."""

    root = _checkpoint_records_root(tmp_path, snapshots=1)
    canonical_root = root / "menagerie" / "crawler"
    checkpoint_module._reset_intake_snapshot_verification_memo()
    checkpoint_module._derive_candidate_paths(root, canonical_root, ())
    snapshot_root = next((canonical_root / "records" / "intake").iterdir())
    items = snapshot_root / "items.jsonl"
    items.write_bytes(items.read_bytes() + b'{"forged": true}\n')
    with pytest.raises(checkpoint_module.CheckpointValidationError):
        checkpoint_module._derive_candidate_paths(root, canonical_root, ())
    checkpoint_module._reset_intake_snapshot_verification_memo()
