"""AT-10: the author-result scratch cache binds its generation, fail-closed.

m12646's surviving cache carried ``repair_generation: null`` and no
generation binding at all, so an upgrade-lane re-drive under a NEW work
generation kept reloading it (R16's paid loop had a second door). Cache v2
embeds an explicit generation binding -- stable_id, work_id, result id,
manifest identity, repair generation, staged transaction id, and the
capture-time lifecycle -- and the loader refuses typed on ANY divergence:

- a v1/legacy cache fails CLOSED for authority-bearing reuse (the version
  check; the archived m12646 shape below);
- a v2 cache serving a foreign work generation, stable id, or manifest
  identity refuses before any result validation;
- a cache may reference only a PENDING 1-event staging transaction: a
  referenced chain that advanced is owned by canonical authority.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorEngineFaultError,
    AuthorInfrastructureFaultError,
    serialize_author_result_cache,
    validate_author_result_cache,
)
from menagerie.crawler.cache_registry import author_result_cache_path
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import FakeAuthor, _driver, _paths, _snapshot

pytestmark = pytest.mark.smoke


_ENVELOPE = {
    "stable_id": "m_v2",
    "work_id": "work-m_v2",
    "source_manifest_identity": "sha256:" + "a" * 64,
}


def _stub_result(**overrides) -> SimpleNamespace:
    binding = dict(
        stable_id="m_v2",
        work_id="work-m_v2",
        result_id="result-1",
        source_manifest_identity="sha256:" + "a" * 64,
        raw_result={"result_id": "result-1"},
    )
    binding.update(overrides)
    return SimpleNamespace(binding=SimpleNamespace(**binding))


def _sealed(cache: dict) -> dict:
    body = {key: value for key, value in cache.items() if key != "cache_identity"}
    return {**body, "cache_identity": stable_hash(body)}


def _v2_cache(**binding_overrides) -> dict:
    binding = {
        "stable_id": "m_v2",
        "work_id": "work-m_v2",
        "result_id": "result-1",
        "source_manifest_identity": "sha256:" + "a" * 64,
        "repair_generation": None,
        "staged_transaction_id": None,
        "lifecycle": "unstaged",
    }
    binding.update(binding_overrides)
    return _sealed(
        {
            "cache_version": "menagerie.crawler.author-result-cache.v2",
            "result": {"result_id": "result-1"},
            "source_manifest": {"manifest_sha256": "sha256:" + "a" * 64},
            "model_dir": "/tmp/model",
            "repair_generation": None,
            "generation_binding": binding,
        }
    )


def test_legacy_v1_cache_fails_closed_for_authority_bearing_reuse():
    """The archived m12646 shape: v1, repair_generation null, no binding."""

    legacy = _sealed(
        {
            "cache_version": "menagerie.crawler.author-result-cache.v1",
            "result": {"result_id": "result-m12646"},
            "source_manifest": {},
            "model_dir": "/tmp/model",
            "repair_generation": None,
        }
    )
    with pytest.raises(AuthorInfrastructureFaultError, match="cache version"):
        validate_author_result_cache(legacy, _ENVELOPE)
    # Diagnostic-only reads stay possible: the JSON parses without the loader.
    assert json.loads(json.dumps(legacy))["repair_generation"] is None


def test_v2_cache_without_a_generation_binding_refuses():
    cache = _v2_cache()
    body = {
        key: value
        for key, value in cache.items()
        if key not in {"cache_identity", "generation_binding"}
    }
    with pytest.raises(AuthorInfrastructureFaultError, match="generation_binding"):
        validate_author_result_cache(_sealed(body), _ENVELOPE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("work_id", "work-m_v2-requeue-2"),  # old/future generation, uniformly
        ("stable_id", "m_other"),
        ("source_manifest_identity", "sha256:" + "b" * 64),
    ],
)
def test_v2_cache_binding_divergence_refuses_before_result_validation(field, value):
    with pytest.raises(AuthorInfrastructureFaultError, match=f"foreign {field}"):
        validate_author_result_cache(_v2_cache(**{field: value}), _ENVELOPE)


def test_v2_cache_lifecycle_outside_the_closed_vocabulary_refuses():
    with pytest.raises(AuthorInfrastructureFaultError, match="lifecycle"):
        validate_author_result_cache(_v2_cache(lifecycle="final"), _ENVELOPE)


def test_v2_cache_result_id_contradiction_refuses():
    with pytest.raises(AuthorInfrastructureFaultError, match="result id"):
        validate_author_result_cache(_v2_cache(result_id="result-9"), _ENVELOPE)


def test_serializer_refuses_incoherent_lifecycle_declarations():
    result = _stub_result()
    manifest = {"manifest_sha256": "sha256:" + "a" * 64}
    with pytest.raises(AuthorEngineFaultError, match="closed vocabulary"):
        serialize_author_result_cache(
            result, source_manifest=manifest, model_dir="/tmp/m", lifecycle="final"
        )
    with pytest.raises(AuthorEngineFaultError, match="staged-pending exactly"):
        serialize_author_result_cache(
            result,
            source_manifest=manifest,
            model_dir="/tmp/m",
            staged_transaction_id="tx-1",
            lifecycle="unstaged",
        )
    with pytest.raises(AuthorEngineFaultError, match="staged-pending exactly"):
        serialize_author_result_cache(
            result, source_manifest=manifest, model_dir="/tmp/m", lifecycle="staged-pending"
        )


def test_serializer_embeds_the_generation_binding():
    result = _stub_result()
    manifest = {"manifest_sha256": "sha256:" + "a" * 64}
    cache = serialize_author_result_cache(
        result,
        source_manifest=manifest,
        model_dir="/tmp/m",
        staged_transaction_id="tx-pending-1",
        lifecycle="staged-pending",
    )
    binding = cache["generation_binding"]
    assert binding == {
        "stable_id": "m_v2",
        "work_id": "work-m_v2",
        "result_id": "result-1",
        "source_manifest_identity": "sha256:" + "a" * 64,
        "repair_generation": None,
        "staged_transaction_id": "tx-pending-1",
        "lifecycle": "staged-pending",
    }


def test_cache_referencing_a_non_pending_transaction_is_discarded_and_retired(tmp_path):
    """A resumed leg whose cache names a staging transaction the ledger does
    not hold as a PENDING 1-event chain discards it typed (AT-10) and retires
    it by rename (W9); the member authors fresh instead of re-serving."""

    from menagerie.crawler.driver_contracts import DriverPaused

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id

    def _die_pre_award(boundary: str, _stable_id: str) -> None:
        if boundary == "post-attempt-pre-award":
            raise DriverPaused("crash before the award (fixture)")

    first = _driver(tmp_path, snapshot, author=FakeAuthor(), boundary=_die_pre_award)
    try:
        first.run()
    except DriverPaused:
        pass
    cache = author_result_cache_path(paths.work_root, stable_id)
    assert cache.is_file(), "leg 1 leaves the fresh result's cache behind"

    loaded = json.loads(cache.read_text(encoding="utf-8"))
    binding = dict(loaded["generation_binding"])
    binding["staged_transaction_id"] = "tx-nowhere-in-the-ledger"
    binding["lifecycle"] = "staged-pending"
    loaded["generation_binding"] = binding
    tampered = {key: value for key, value in loaded.items() if key != "cache_identity"}
    cache.write_text(
        json.dumps({**tampered, "cache_identity": stable_hash(tampered)}), encoding="utf-8"
    )

    second_author = FakeAuthor()
    second = _driver(tmp_path, snapshot, author=second_author)
    second.run()
    discarded = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if isinstance(event.get("details"), dict)
        and event["details"].get("disposition") == "author-cache-discarded"
    ]
    assert discarded
    assert any(
        "non-pending staging transaction" in str(event["details"].get("error", ""))
        for event in discarded
    )
    # W9 retire-by-rename applied to the AT-10 discard too, and the member
    # authored fresh instead of re-serving the refused cache.
    work_dir = paths.work_root / stable_id
    assert list(work_dir.glob("driver-author-artifact.json.stale-*"))
    assert second_author.calls.get(stable_id, 0) == 1