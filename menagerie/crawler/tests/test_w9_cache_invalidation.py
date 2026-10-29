"""W9 composed cache invalidation: retirement row, retire-by-rename, eviction, fuse.

R16: the work-dir author-result scratch cache (C2) survived ledger
invalidation -- six call sites disclosed an identity-stale entry and did
nothing else, so the stale authority was reloaded and re-bought every leg
(m12646's paid loop). W9 composes ONE retirement authority in S-U3 crash
order: durable retirement row FIRST (replay-derived key axes, honest DEGRADED
status), atomic evidence-preserving ``.stale-<event_id>`` rename of C2
(ruling A20/IC-Q1: never unlink), eviction of every in-memory layer holding
the dead authority, and an N-repeat fuse that parks a member
``invalidation-nonconvergent`` on the third same-key retirement instead of
re-buying the loop. The cache-layer registry (``cache_registry``) refuses
consulting any admission-steering on-disk cache that is not registered with
an invalidation authority in IC-4's closed inventory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import menagerie.crawler.cache_registry as cache_registry_module
from menagerie.crawler.artifact_transactions import ArtifactTransactionId
from menagerie.crawler.cache_registry import (
    CacheRegistryError,
    author_result_cache_path,
    startup_cache_registration_check,
)
from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.driver_admission import (
    ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION,
)
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.tests.test_artifact_identity_recorded_basis import (
    _crafted_artifact,
    _patch_rehydrate_once,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    _driver,
    _paths,
    _snapshot,
)

import menagerie.crawler.driver_admission as driver_admission_module

pytestmark = pytest.mark.smoke

_SENTINEL_BYTES = b'{"sentinel": "stale scratch cache evidence"}\n'


def _retirement_rows(operational_path: Path) -> list[dict[str, Any]]:
    return [
        event
        for event in scan_jsonl(operational_path)
        if isinstance(event.get("details"), dict)
        and event["details"].get("disposition")
        == ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION
    ]


def _stale_evidence(work_root: Path, stable_id: str) -> list[Path]:
    work_dir = work_root / stable_id
    if not work_dir.is_dir():
        return []
    return sorted(work_dir.glob("driver-author-artifact.json.stale-*"))


def _patch_rehydrate_always(
    monkeypatch: pytest.MonkeyPatch, stable_id: str, crafted: Any
) -> None:
    """Serve the same stale crafted artifact on EVERY consult (R16 recurrence)."""

    real = driver_admission_module.AdmissionEnvironmentMixin._rehydrate_final_authority

    def rehydrate(self: Any, item: Any, reducer: Any, operational: Any) -> Any:
        if item.stable_id == stable_id:
            return crafted
        return real(self, item, reducer, operational)

    monkeypatch.setattr(
        driver_admission_module.AdmissionEnvironmentMixin,
        "_rehydrate_final_authority",
        rehydrate,
    )
    real_certain = (
        driver_admission_module.AdmissionEnvironmentMixin._author_session_is_certain
    )

    def certain(self: Any, item: Any, reducer: Any) -> bool:
        if item.stable_id == stable_id:
            return False
        return real_certain(self, item, reducer)

    monkeypatch.setattr(
        driver_admission_module.AdmissionEnvironmentMixin,
        "_author_session_is_certain",
        certain,
    )


def test_retirement_authority_composes_row_rename_eviction_and_fuse(tmp_path):
    """One call: durable row first, typed rename, in-memory eviction; the
    repeat ordinal derives from the ledger and the third call parks."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    stable_id = item.stable_id

    cache = author_result_cache_path(paths.work_root, stable_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(_SENTINEL_BYTES)
    driver._family_artifacts[stable_id] = object()
    driver._final_artifact_transactions[
        (stable_id, item.active_work_id, ArtifactTransactionId("tx-dead"))
    ] = object()

    operational = JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION)
    try:
        verdicts = [
            driver._report_identity_stale_artifact_authority(
                item,
                None,
                RuntimeError("identity recompute mismatch (unit)"),
                operational,
                transaction_id="tx-dead",
                recorded_basis_available=False,
            )
            for _ in range(3)
        ]
    finally:
        operational.close()

    assert verdicts == ["retired", "retired", "invalidation-nonconvergent"]
    rows = _retirement_rows(paths.operational_ledger)
    assert [row["details"]["repeat_ordinal"] for row in rows] == [1, 2, 3]
    # Honest status: an invalidation is never HEALTHY (DX-36 rule).
    assert {row["status"] for row in rows} == {"degraded"}
    assert {row["details"]["cache_identity"] for row in rows} == {"C1"}
    assert rows[-1]["details"]["park"] == "invalidation-nonconvergent"
    assert "park" not in rows[0]["details"]

    # Retire-by-rename: live path gone, evidence preserved byte-identically,
    # suffix names the retirement row that caused it.
    assert not cache.is_file()
    (evidence,) = _stale_evidence(paths.work_root, stable_id)
    assert evidence.read_bytes() == _SENTINEL_BYTES
    assert evidence.name.endswith(str(rows[0]["event_id"]).replace(":", "-"))

    # In-memory eviction of every layer that held the dead authority.
    assert stable_id not in driver._family_artifacts
    assert not [
        key for key in driver._final_artifact_transactions if key[0] == stable_id
    ]
    assert stable_id in driver._invalidation_nonconvergent_ids


def test_identity_stale_invalidation_retires_scratch_evidence_end_to_end(
    tmp_path, monkeypatch
):
    """The production leg: invalidation retires C2 by rename (never unlink),
    the retirement row carries the composed key axes, and the member is
    authored FRESH -- a new author result, not the retired bytes."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id
    fresh_author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=fresh_author)
    item = driver._ordered_work(snapshot, {})[0]
    crafted = _crafted_artifact(tmp_path, driver, snapshot, item)
    crafted.proposal["vet_identity"] = "sha256:" + "d" * 64
    _patch_rehydrate_once(monkeypatch, stable_id, crafted, stage=False)

    cache = author_result_cache_path(paths.work_root, stable_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(_SENTINEL_BYTES)

    result = driver.run()

    assert result.status == "complete"
    assert fresh_author.calls == {stable_id: 1}
    models = scan_jsonl(paths.ledgers.models)
    assert [model["status"]["code"] for model in models] == ["runs"]

    (row,) = _retirement_rows(paths.operational_ledger)
    details = row["details"]
    assert details["stable_id"] == stable_id
    assert details["repeat_ordinal"] == 1
    assert details["cache_identity"] == "C1"
    assert details["result_id"], "the rehydrated artifact's result id is named"
    assert row["status"] == "degraded"

    # Retire-by-rename: the sentinel survives as evidence; the live path, if
    # present at all, is the FRESH result's cache -- never the retired bytes.
    evidence = _stale_evidence(paths.work_root, stable_id)
    assert evidence, "retired scratch cache must survive as renamed evidence"
    assert any(entry.read_bytes() == _SENTINEL_BYTES for entry in evidence)
    if cache.is_file():
        assert cache.read_bytes() != _SENTINEL_BYTES


def test_recurring_invalidation_parks_nonconvergent_on_the_third_retirement(
    tmp_path, monkeypatch
):
    """R16's paid loop, fused: with two same-key retirement rows already
    durable on the operational ledger (prior legs), the leg whose invalidation
    mints the THIRD row parks the member -- no author session is bought and
    the member is deferred, disclosed by its own park row. The ordinal is
    replay-derived from the ledger: nothing in memory carries it across legs."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id

    seeder = _driver(tmp_path, snapshot)
    seed_item = seeder._ordered_work(snapshot, {})[0]
    operational = JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION)
    try:
        for _ in range(2):
            verdict = seeder._report_identity_stale_artifact_authority(
                seed_item,
                None,
                RuntimeError("identity recompute mismatch (prior leg)"),
                operational,
                transaction_id=None,
                recorded_basis_available=False,
            )
            assert verdict == "retired"
    finally:
        operational.close()

    author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=author)
    item = driver._ordered_work(snapshot, {})[0]
    crafted = _crafted_artifact(tmp_path, driver, snapshot, item)
    crafted.proposal["vet_identity"] = "sha256:" + "d" * 64
    _patch_rehydrate_always(monkeypatch, stable_id, crafted)

    driver.run()

    # The fuse: no third paid session for the same retirement key.
    assert author.calls.get(stable_id, 0) == 0
    rows = _retirement_rows(paths.operational_ledger)
    assert [row["details"]["repeat_ordinal"] for row in rows] == [1, 2, 3]
    assert rows[-1]["details"]["park"] == "invalidation-nonconvergent"
    # The park is the durable partition-accounted quarantine valve, typed.
    from menagerie.crawler.scheduler_parks import SCHEDULER_QUARANTINE_RELATIVE_PATH

    records_root = paths.ledgers.models.parent.parent
    quarantines = [
        row
        for row in scan_jsonl(records_root / SCHEDULER_QUARANTINE_RELATIVE_PATH)
        if row.get("stable_id") == stable_id
    ]
    assert quarantines
    assert (
        quarantines[-1]["scheduler_quarantine"]["reason_code"]
        == "invalidation-nonconvergent"
    )


def test_cache_registry_refuses_unregistered_or_authorityless_caches(monkeypatch):
    """The registry chokepoint: consulting an unregistered admission-steering
    cache refuses typed, and startup refuses an on-disk cache registered
    without an invalidation authority. The shipped inventory verifies."""

    verified = startup_cache_registration_check()
    assert "work-dir-author-result-scratch-cache" in verified

    monkeypatch.setattr(cache_registry_module, "_REGISTRY", {})
    with pytest.raises(CacheRegistryError, match="not registered"):
        author_result_cache_path(Path("/tmp"), "m0001")

    monkeypatch.setattr(
        cache_registry_module,
        "_REGISTRY",
        {
            "rogue-on-disk-cache": {
                "name": "rogue-on-disk-cache",
                "kind": "on-disk-file",
                "invalidation_authority": "",
            }
        },
    )
    with pytest.raises(CacheRegistryError, match="refusing startup"):
        startup_cache_registration_check()


def test_author_cache_discard_preserves_evidence_by_rename(tmp_path):
    """The author-cache-discarded handler also retires by rename (A20/IC-Q1):
    an unreadable or contradicted cache is evidence of its own discard cause,
    never unlinked."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id
    author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=author)

    # A cache that parses as JSON but lacks the staging inputs: the reload
    # path discards it (`author-cache-discarded`) and must retire it.
    cache = author_result_cache_path(paths.work_root, stable_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(_SENTINEL_BYTES)

    result = driver.run()

    assert result.status == "complete"
    discarded = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if isinstance(event.get("details"), dict)
        and event["details"].get("disposition") == "author-cache-discarded"
    ]
    assert discarded, "the malformed cache must be discarded loudly"
    evidence = _stale_evidence(paths.work_root, stable_id)
    assert evidence, "discarded cache must survive as renamed evidence"
    assert any(entry.read_bytes() == _SENTINEL_BYTES for entry in evidence)
    if cache.is_file():
        # A fresh authoring may lawfully write a NEW cache afterwards; the
        # retired bytes themselves are never back at the live path.
        assert cache.read_bytes() != _SENTINEL_BYTES
    json.loads(_SENTINEL_BYTES)  # the sentinel is honest JSON, so the discard
    # exercised the shape validation, not a parse failure.
