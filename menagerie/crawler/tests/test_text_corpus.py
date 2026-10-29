"""Durable text-corpus retention: promotion, GC contract, verification, union.

The corpus is the fetch-once retention seam: every document the crawl already
holds in a work dir is promoted into the append-only, content-addressed store
under ``records/text-corpus/`` so prose authoring and any future derived
writing never require a re-crawl. These tests pin the four contracts the store
ships with: idempotent digest-checked promotion, the reaper clearance guard,
fail-closed verification (payloads and excerpt resolution), and the
content-addressed multi-host union.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.campaign_merge import _canonical_input_paths
from menagerie.crawler.constants import TEXT_CORPUS_INDEX_SCHEMA_VERSION
from menagerie.crawler.fetcher import cas_path, publish_cas_object
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.recordio import JsonlLedger, LedgerConflictError, scan_jsonl
from menagerie.crawler.source_broker import BROKER_PACK_VERSION
from menagerie.crawler.text_corpus import (
    CORPUS_INDEX_FILENAME,
    TEXT_CORPUS_DIRNAME,
    CorpusIntegrityError,
    CorpusReapRefusal,
    assert_not_corpus_path,
    corpus_index_path,
    corpus_payload_path,
    corpus_root,
    corpus_row_id,
    promote_author_root,
    union_corpus,
    verify_corpus,
    work_dir_reap_clearance,
)

pytestmark = pytest.mark.smoke


IMPL_BYTES = b"import torch\n\nclass Net(torch.nn.Module):\n    pass\n"
PAPER_BYTES = b"<html><body>We introduce the Net architecture.</body></html>\n"
PROBE_BYTES = b"<html>rejected candidate page</html>\n"


def _evidence_name(content: bytes) -> str:
    return f"{hash_bytes(content).removeprefix('sha256:')[:16]}.bin"


def _outcome(
    source_id: str,
    content: bytes,
    *,
    requested_role: str,
    bound_role: str | None,
    outcome: str = "fetched",
    url: str = "https://example.test/doc",
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


def _author_root(
    tmp_path: Path,
    stable_id: str = "m0001",
    *,
    impl_bytes: bytes = IMPL_BYTES,
    corrupt_impl_evidence: bool = False,
    skip_custody_bytes: bool = False,
) -> Path:
    """Materialize a realistic author custody root under ``tmp_path``.

    One custody-promoted implementation document, one fetched-but-unmanifested
    probe document, and one frozen discovery envelope -- the three retention
    classes the corpus must cover.
    """

    root = tmp_path / "work" / stable_id / "author"
    evidence_dir = root / "broker" / "evidence"
    evidence_dir.mkdir(parents=True)
    source_cas = root / "source-cas"
    source_cas.mkdir(parents=True)

    impl_digest = hash_bytes(impl_bytes)
    stored_impl = b"corrupted " + impl_bytes if corrupt_impl_evidence else impl_bytes
    if not skip_custody_bytes:
        (evidence_dir / _evidence_name(impl_bytes)).write_bytes(stored_impl)
        if not corrupt_impl_evidence:
            publish_cas_object(source_cas, impl_digest, impl_bytes)
    (evidence_dir / _evidence_name(PROBE_BYTES)).write_bytes(PROBE_BYTES)

    impl_outcome = _outcome(
        f"impl-{stable_id}",
        impl_bytes,
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
        "sources": [_manifest_row(f"impl-{stable_id}", impl_bytes, broker_role="implementation")],
        "broker": {
            "outcomes": [impl_outcome, probe_outcome],
            "derived_citations": [],
            "total_bytes": len(impl_bytes) + len(PROBE_BYTES),
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
                "bytes_len": len(impl_bytes),
                "evidence_path": str(evidence_dir / _evidence_name(impl_bytes)),
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


def _records_root(tmp_path: Path) -> Path:
    root = tmp_path / "records"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _row_for(report, source_id):
    matches = [row for row in report.dispositions if row["source_id"] == source_id]
    assert matches, f"no disposition for {source_id!r}: {report.dispositions}"
    return matches[0]


# -- index ledger ------------------------------------------------------------


def test_corpus_index_rows_are_idempotent_and_conflict_checked(tmp_path):
    records_root = _records_root(tmp_path)
    path = corpus_index_path(records_root)
    assert path == records_root / TEXT_CORPUS_DIRNAME / CORPUS_INDEX_FILENAME
    digest = hash_bytes(IMPL_BYTES)
    row = {
        "schema_version": TEXT_CORPUS_INDEX_SCHEMA_VERSION,
        "corpus_row_id": corpus_row_id(
            stable_id="m0001",
            content_sha256=digest,
            source_id="impl-m0001",
            role="implementation",
            origin="custody-promotion",
            work_id="work-m0001",
        ),
        "stable_id": "m0001",
        "work_id": "work-m0001",
        "source_id": "impl-m0001",
        "role": "implementation",
        "requested_role": "implementation",
        "origin": "custody-promotion",
        "content_sha256": digest,
        "bytes_len": len(IMPL_BYTES),
        "media_type": "text/html",
        "url": "https://example.test/doc",
        "final_url": "https://example.test/doc",
        "retrieved_at": "2026-08-11T00:00:00.000000Z",
        "license_disposition": "manifest-only",
        "custody_disposition": "promoted",
    }
    ledger = JsonlLedger(path, TEXT_CORPUS_INDEX_SCHEMA_VERSION)
    try:
        first = ledger.append(dict(row))
        assert first.appended is True
        replay = dict(row)
        replay["ledger_seq"] = first.record["ledger_seq"]
        replay["payload_sha256"] = first.record["payload_sha256"]
        second = ledger.append(replay)
        assert second.appended is False
        conflicting = dict(replay)
        conflicting["bytes_len"] = len(IMPL_BYTES) + 1
        with pytest.raises(LedgerConflictError):
            ledger.append(conflicting)
        # The m10949 incident shape (2026-08-13): a RE-DRIVEN model re-fetches
        # the same document under a NEW work generation with a new
        # retrieved_at. That is new provenance -- a new row id, appended
        # cleanly -- never a conflicting replay of the prior generation's row.
        redrive = dict(row)
        redrive.pop("ledger_seq", None)
        redrive.pop("payload_sha256", None)
        redrive["work_id"] = "work-m0001-gen2"
        redrive["retrieved_at"] = "2026-08-13T00:00:00.000000Z"
        redrive["corpus_row_id"] = corpus_row_id(
            stable_id="m0001",
            content_sha256=digest,
            source_id="impl-m0001",
            role="implementation",
            origin="custody-promotion",
            work_id="work-m0001-gen2",
        )
        third = ledger.append(redrive)
        assert third.appended is True
    finally:
        ledger.close()


# -- promotion ---------------------------------------------------------------


def test_promotion_publishes_custody_documents_and_is_idempotent(tmp_path):
    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)

    first = promote_author_root(author_root, records_root)
    impl = _row_for(first, "impl-m0001")
    assert impl["disposition"] == "promoted"
    assert impl["role"] == "implementation"
    assert impl["origin"] == "custody-promotion"
    payload = corpus_payload_path(records_root, hash_bytes(IMPL_BYTES))
    assert payload.read_bytes() == IMPL_BYTES
    assert payload == cas_path(corpus_root(records_root), hash_bytes(IMPL_BYTES))

    second = promote_author_root(author_root, records_root)
    assert _row_for(second, "impl-m0001")["disposition"] == "replayed"
    rows = [json.loads(line) for line in corpus_index_path(records_root).read_text().splitlines()]
    keys = [row["corpus_row_id"] for row in rows]
    assert len(keys) == len(set(keys)), "replayed promotion appended duplicate index rows"


def test_promotion_widens_to_unmanifested_fetches_and_discovery(tmp_path):
    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)

    report = promote_author_root(author_root, records_root)

    probe = _row_for(report, "candidate-probe-001")
    assert probe["disposition"] == "promoted"
    assert probe["role"] == "probe"
    assert probe["origin"] == "broker-evidence"
    assert corpus_payload_path(records_root, hash_bytes(PROBE_BYTES)).read_bytes() == PROBE_BYTES

    discovery_rows = [row for row in report.dispositions if row["origin"] == "discovery-evidence"]
    assert len(discovery_rows) == 1
    assert discovery_rows[0]["role"] == "discovery"
    assert discovery_rows[0]["disposition"] == "promoted"
    envelope_digest = discovery_rows[0]["content_sha256"]
    assert corpus_payload_path(records_root, envelope_digest).is_file()


def test_promotion_refuses_wrong_bytes_without_publishing(tmp_path):
    author_root = _author_root(tmp_path, corrupt_impl_evidence=True)
    records_root = _records_root(tmp_path)

    report = promote_author_root(author_root, records_root)

    impl = _row_for(report, "impl-m0001")
    assert impl["disposition"] == "digest-mismatch-refused"
    assert not corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)).exists()
    assert report.refusals, "digest mismatch must surface as a typed refusal"


def test_promotion_never_reads_producer_paths(tmp_path):
    # The custody row's evidence_path is producer-written provenance. Bytes that
    # exist ONLY at that foreign path must not be admitted: promotion searches
    # the author root's own evidence dirs and CAS, nothing else.
    author_root = _author_root(tmp_path, skip_custody_bytes=True)
    foreign = tmp_path / "foreign.bin"
    foreign.write_bytes(IMPL_BYTES)
    custody_path = author_root / "source-custody.json"
    custody = json.loads(custody_path.read_text())
    custody["promotions"][0]["evidence_path"] = str(foreign)
    custody_path.write_text(json.dumps(custody, sort_keys=True))
    records_root = _records_root(tmp_path)

    report = promote_author_root(author_root, records_root)

    impl = _row_for(report, "impl-m0001")
    assert impl["disposition"] == "evidence-missing"
    assert not corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)).exists()


def test_promotion_dedups_shared_documents_across_models(tmp_path):
    root_a = _author_root(tmp_path, "m0001")
    root_b = _author_root(tmp_path, "m0002")
    records_root = _records_root(tmp_path)

    promote_author_root(root_a, records_root)
    promote_author_root(root_b, records_root)

    payload = corpus_payload_path(records_root, hash_bytes(PROBE_BYTES))
    assert payload.read_bytes() == PROBE_BYTES
    rows = [json.loads(line) for line in corpus_index_path(records_root).read_text().splitlines()]
    probe_rows = [row for row in rows if row["content_sha256"] == hash_bytes(PROBE_BYTES)]
    assert sorted(row["stable_id"] for row in probe_rows) == ["m0001", "m0002"]
    payload_files = [
        candidate
        for candidate in (corpus_root(records_root) / "sha256").rglob("*")
        if candidate.is_file()
    ]
    digests = {candidate.name for candidate in payload_files}
    assert len(payload_files) == len(digests), "shared bytes must be stored once per digest"


# -- GC contract -------------------------------------------------------------


def test_reap_clearance_requires_full_promotion(tmp_path):
    author_root = _author_root(tmp_path)
    work_dir = author_root.parent
    records_root = _records_root(tmp_path)

    before = work_dir_reap_clearance(work_dir, records_root)
    assert before.cleared is False
    assert before.reasons

    promote_author_root(author_root, records_root)
    after = work_dir_reap_clearance(work_dir, records_root)
    assert after.cleared is True
    assert after.reasons == ()


def test_reap_clearance_recomputes_from_corpus_bytes(tmp_path):
    # Clearance is replay-derived: deleting a promoted payload out of the corpus
    # must flip the work dir back to not-reapable, no matter what any producer
    # or prior report claimed.
    author_root = _author_root(tmp_path)
    work_dir = author_root.parent
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)
    corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)).unlink()

    clearance = work_dir_reap_clearance(work_dir, records_root)
    assert clearance.cleared is False


def test_clearance_is_generation_tolerant_when_bytes_are_durable(tmp_path: Path) -> None:
    """A redrive does not strand bytes already held under an exact source binding."""

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)
    custody_path = author_root / "source-custody.json"
    custody = json.loads(custody_path.read_text())
    custody["work_id"] = "work-m0001-gen2"
    custody_path.write_text(json.dumps(custody, sort_keys=True))

    clearance = work_dir_reap_clearance(author_root.parent, records_root)

    assert clearance.cleared is True
    assert clearance.reasons == ()


def test_reaper_guard_refuses_corpus_paths(tmp_path):
    records_root = _records_root(tmp_path)
    author_root = _author_root(tmp_path)
    promote_author_root(author_root, records_root)

    with pytest.raises(CorpusReapRefusal):
        assert_not_corpus_path(corpus_index_path(records_root), records_root)
    with pytest.raises(CorpusReapRefusal):
        assert_not_corpus_path(
            corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)), records_root
        )
    with pytest.raises(CorpusReapRefusal):
        assert_not_corpus_path(corpus_root(records_root), records_root)
    assert_not_corpus_path(author_root, records_root)
    assert_not_corpus_path(author_root / "broker" / "evidence", records_root)


# -- verification ------------------------------------------------------------


def test_verify_corpus_passes_on_clean_store(tmp_path):
    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)

    report = verify_corpus(records_root)
    assert report.passed is True
    assert report.findings == ()
    assert report.row_count >= 3


def test_verify_corpus_flags_missing_and_corrupt_payloads(tmp_path):
    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)

    corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)).unlink()
    corrupt = corpus_payload_path(records_root, hash_bytes(PROBE_BYTES))
    corrupt.write_bytes(b"not the recorded bytes")

    report = verify_corpus(records_root)
    assert report.passed is False
    kinds = sorted(finding["kind"] for finding in report.findings)
    assert kinds == ["payload-digest-mismatch", "payload-missing"]


def test_verify_corpus_resolves_excerpts_to_documents(tmp_path):
    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)

    excerpt_text = "class Net(torch.nn.Module):"
    start = IMPL_BYTES.index(excerpt_text.encode())
    result = {
        "stable_id": "m0001",
        "work_id": "work-m0001",
        "proposed_facts": {
            "evidence": {
                "excerpts": [
                    {
                        "evidence_id": "ev-impl-1",
                        "source_id": "impl-m0001",
                        "locator": f"bytes:{start}-{start + len(excerpt_text)}",
                        "text": excerpt_text,
                        "text_sha256": hash_bytes(excerpt_text.encode()),
                        "supports": ["architecture"],
                        "family_level": False,
                    }
                ]
            }
        },
    }
    (author_root / "result.json").write_text(json.dumps(result, sort_keys=True))

    clean = verify_corpus(records_root, work_roots=(tmp_path / "work",))
    assert clean.passed is True
    assert clean.excerpt_count == 1

    tampered = dict(result)
    tampered["proposed_facts"] = json.loads(json.dumps(result["proposed_facts"]))
    excerpt = tampered["proposed_facts"]["evidence"]["excerpts"][0]
    excerpt["text"] = "class Net(torch.nn.Sequential):"
    excerpt["text_sha256"] = hash_bytes(excerpt["text"].encode())
    (author_root / "result.json").write_text(json.dumps(tampered, sort_keys=True))

    broken = verify_corpus(records_root, work_roots=(tmp_path / "work",))
    assert broken.passed is False
    assert any(finding["kind"] == "excerpt-locator-unresolved" for finding in broken.findings)


def test_verify_corpus_flags_excerpts_from_unpromoted_sources(tmp_path):
    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    # Deliberately NOT promoted: the excerpt's source cannot resolve.
    result = {
        "stable_id": "m0001",
        "work_id": "work-m0001",
        "proposed_facts": {
            "evidence": {
                "excerpts": [
                    {
                        "evidence_id": "ev-impl-1",
                        "source_id": "impl-m0001",
                        "locator": "bytes:0-6",
                        "text": "import",
                        "text_sha256": hash_bytes(b"import"),
                        "supports": ["architecture"],
                        "family_level": False,
                    }
                ]
            }
        },
    }
    (author_root / "result.json").write_text(json.dumps(result, sort_keys=True))

    report = verify_corpus(records_root, work_roots=(tmp_path / "work",))
    assert report.passed is False
    assert any(finding["kind"] == "excerpt-source-unpromoted" for finding in report.findings)


def test_excerpt_verification_pins_its_work_generation(tmp_path: Path) -> None:
    """A current result cannot resolve through a historical generation's row."""

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)
    excerpt_text = "class Net(torch.nn.Module):"
    start = IMPL_BYTES.index(excerpt_text.encode())
    result = {
        "stable_id": "m0001",
        "work_id": "work-m0001-gen2",
        "proposed_facts": {
            "evidence": {
                "excerpts": [
                    {
                        "evidence_id": "ev-impl-1",
                        "source_id": "impl-m0001",
                        "locator": f"bytes:{start}-{start + len(excerpt_text)}",
                        "text": excerpt_text,
                        "text_sha256": hash_bytes(excerpt_text.encode()),
                        "supports": ["architecture"],
                        "family_level": False,
                    }
                ]
            }
        },
    }
    (author_root / "result.json").write_text(json.dumps(result, sort_keys=True))

    report = verify_corpus(records_root, work_roots=(tmp_path / "work",))

    assert report.passed is False
    assert any(finding["kind"] == "excerpt-source-unpromoted" for finding in report.findings)


def test_backfill_promotes_the_accepted_result_generation(tmp_path: Path) -> None:
    """A stale custody envelope cannot strand the accepted result's exact lineage."""

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    excerpt_text = "class Net(torch.nn.Module):"
    start = IMPL_BYTES.index(excerpt_text.encode())
    accepted_work_id = "work-m0001-accepted"
    result = {
        "stable_id": "m0001",
        "work_id": accepted_work_id,
        "payload": {
            "proposal": {
                "proposed_facts": {
                    "source_resolution": {
                        "sources": [
                            {
                                "source_id": "impl-m0001",
                                "content_sha256": hash_bytes(IMPL_BYTES),
                                "role": "implementation",
                                "media_type": "text/x-python",
                            }
                        ]
                    },
                    "evidence": {
                        "excerpts": [
                            {
                                "evidence_id": "ev-impl-1",
                                "source_id": "impl-m0001",
                                "locator": f"bytes:{start}-{start + len(excerpt_text)}",
                                "text": excerpt_text,
                                "text_sha256": hash_bytes(excerpt_text.encode()),
                            }
                        ]
                    },
                }
            }
        },
    }
    (author_root / "result.json").write_text(json.dumps(result, sort_keys=True))

    promote_author_root(author_root, records_root)
    report = verify_corpus(records_root, work_roots=(tmp_path / "work",))

    assert report.passed is True
    rows = scan_jsonl(corpus_index_path(records_root))
    assert any(
        row.get("work_id") == accepted_work_id
        and row.get("source_id") == "impl-m0001"
        for row in rows
    )


def test_excerpt_verification_requires_a_work_generation_binding(tmp_path: Path) -> None:
    """Missing result lineage fails closed instead of searching all history."""

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)
    result = {
        "stable_id": "m0001",
        "proposed_facts": {
            "evidence": {
                "excerpts": [
                    {
                        "evidence_id": "ev-impl-1",
                        "source_id": "impl-m0001",
                        "locator": "bytes:0-6",
                        "text": "import",
                        "text_sha256": hash_bytes(b"import"),
                    }
                ]
            }
        },
    }
    (author_root / "result.json").write_text(json.dumps(result, sort_keys=True))

    report = verify_corpus(records_root, work_roots=(tmp_path / "work",))

    assert report.passed is False
    assert any(finding["kind"] == "excerpt-work-id-missing" for finding in report.findings)


def test_promotion_repairs_a_corrupt_shared_payload_from_fresh_custody(tmp_path: Path) -> None:
    """One corrupt corpus object cannot abort every later model sharing its bytes."""

    first_root = _author_root(tmp_path, "m0001")
    second_root = _author_root(tmp_path, "m0002")
    records_root = _records_root(tmp_path)
    promote_author_root(first_root, records_root)
    payload = corpus_payload_path(records_root, hash_bytes(IMPL_BYTES))
    payload.write_bytes(b"corrupt shared bytes")

    report = promote_author_root(second_root, records_root)

    assert payload.read_bytes() == IMPL_BYTES
    assert not report.refusals


def test_promotion_contains_corpus_writer_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corpus lock contention is a typed model-local refusal, never a leg exception."""

    from menagerie.crawler import text_corpus as corpus_module
    from menagerie.crawler.recordio import SingleWriterError

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)

    def _busy(_records_root: Path):
        """Simulate a corpus writer that outlives the bounded wait."""

        raise SingleWriterError("busy corpus writer")

    monkeypatch.setattr(corpus_module, "open_corpus_index", _busy)

    report = promote_author_root(author_root, records_root)

    assert report.refusals
    assert {row["disposition"] for row in report.refusals} == {"index-busy-refused"}


def test_verify_corpus_clean_run_is_a_launch_gate(tmp_path: Path) -> None:
    """The launch gate rejects backlog and accepts a fully promoted work root."""

    from menagerie.crawler.text_corpus import TextCorpusError, verify_corpus_launch_gate

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    result = {
        "stable_id": "m0001",
        "work_id": "work-m0001",
        "proposed_facts": {
            "evidence": {
                "excerpts": [
                    {
                        "evidence_id": "ev-impl-1",
                        "source_id": "impl-m0001",
                        "locator": "bytes:0-6",
                        "text": "import",
                        "text_sha256": hash_bytes(b"import"),
                    }
                ]
            }
        },
    }
    (author_root / "result.json").write_text(json.dumps(result, sort_keys=True))

    with pytest.raises(TextCorpusError, match="excerpt-source-unpromoted"):
        verify_corpus_launch_gate(records_root, work_root=tmp_path / "work")

    promote_author_root(author_root, records_root)
    clean = verify_corpus_launch_gate(records_root, work_root=tmp_path / "work")
    assert clean.passed is True


def test_corpus_growth_surfaces_name_capacity_and_reclamation() -> None:
    """Every BL14 growth surface declares its measured rate and lifecycle owner."""

    from menagerie.crawler.text_corpus import CORPUS_GROWTH_SURFACES

    assert {surface.key for surface in CORPUS_GROWTH_SURFACES} == {
        "author-broker-evidence",
        "author-source-cas",
        "corpus-index",
        "corpus-payloads",
    }
    for surface in CORPUS_GROWTH_SURFACES:
        assert surface.bytes_per_model > 0
        assert surface.capacity_authority
        assert surface.reclaimer or surface.durable_retention


# -- multi-host union --------------------------------------------------------


def test_union_corpus_is_content_addressed_and_idempotent(tmp_path):
    shard_a = _records_root(tmp_path / "shard-a")
    shard_b = _records_root(tmp_path / "shard-b")
    promote_author_root(_author_root(tmp_path / "shard-a", "m0001"), shard_a)
    promote_author_root(_author_root(tmp_path / "shard-b", "m0002"), shard_b)

    merged = _records_root(tmp_path / "merged")
    first = union_corpus((shard_a, shard_b), merged)
    assert first.rows_appended >= 6
    assert corpus_payload_path(merged, hash_bytes(PROBE_BYTES)).read_bytes() == PROBE_BYTES

    second = union_corpus((shard_a, shard_b), merged)
    assert second.rows_appended == 0
    rows = [json.loads(line) for line in corpus_index_path(merged).read_text().splitlines()]
    keys = [row["corpus_row_id"] for row in rows]
    assert len(keys) == len(set(keys))
    assert {row["stable_id"] for row in rows} == {"m0001", "m0002"}


def test_union_corpus_refuses_conflicting_bytes_at_one_address(tmp_path):
    shard_a = _records_root(tmp_path / "shard-a")
    shard_b = _records_root(tmp_path / "shard-b")
    promote_author_root(_author_root(tmp_path / "shard-a", "m0001"), shard_a)
    promote_author_root(_author_root(tmp_path / "shard-b", "m0002"), shard_b)
    # Plant wrong bytes at an existing address in shard B, bypassing the
    # digest-checked writer, exactly the corruption the union must refuse.
    victim = corpus_payload_path(shard_b, hash_bytes(PROBE_BYTES))
    victim.write_bytes(b"substituted bytes")

    merged = _records_root(tmp_path / "merged")
    with pytest.raises(CorpusIntegrityError):
        union_corpus((shard_a, shard_b), merged)


def test_canonical_input_paths_cover_corpus_index(tmp_path):
    records_root = _records_root(tmp_path)
    promote_author_root(_author_root(tmp_path), records_root)

    covered = _canonical_input_paths(records_root)
    assert corpus_index_path(records_root) in covered


# -- operator tools ----------------------------------------------------------


def test_promote_corpus_tool_backfills_a_work_root(tmp_path, capsys):
    from menagerie.crawler.tools.promote_corpus import main as promote_main
    from menagerie.crawler.tools.verify_corpus import main as verify_main

    _author_root(tmp_path, "m0001")
    _author_root(tmp_path, "m0002")
    records_root = _records_root(tmp_path)

    rc = promote_main(["--work-root", str(tmp_path / "work"), "--records-root", str(records_root)])
    assert rc == 0
    output = capsys.readouterr().out
    assert "plan: promote 2 author custody trees" in output
    assert '"passed": true' in output
    assert corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)).is_file()

    rc = verify_main(["--records-root", str(records_root), "--work-root", str(tmp_path / "work")])
    assert rc == 0
    capsys.readouterr()

    corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)).unlink()
    rc = verify_main(["--records-root", str(records_root)])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert any(finding["kind"] == "payload-missing" for finding in payload["findings"])


def test_promote_corpus_tool_signals_refusals(tmp_path, capsys):
    from menagerie.crawler.tools.promote_corpus import main as promote_main

    _author_root(tmp_path, "m0001", corrupt_impl_evidence=True)
    records_root = _records_root(tmp_path)

    rc = promote_main(["--work-root", str(tmp_path / "work"), "--records-root", str(records_root)])
    assert rc == 1
    assert "m0001" in capsys.readouterr().out


def test_promote_corpus_backfill_refuses_while_driver_is_live(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The owed backfill cannot contend with a live campaign writer."""

    from menagerie.crawler.driver_contracts import DriverLock
    from menagerie.crawler.tools.promote_corpus import main as promote_main

    _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    lock_path = tmp_path / "locks" / "driver.lock"
    with DriverLock(lock_path, {"test": "live-driver"}):
        rc = promote_main(
            ["--work-root", str(tmp_path / "work"), "--records-root", str(records_root)]
        )

    assert rc == 2
    assert "driver-active" in capsys.readouterr().err
    assert not corpus_index_path(records_root).exists()


# -- award-path hook ---------------------------------------------------------


def _award_stub(tmp_path: Path):
    from types import SimpleNamespace

    from menagerie.crawler.driver_receipts import ReceiptDriverMixin

    records_root = _records_root(tmp_path)
    paths = SimpleNamespace(
        work_root=tmp_path / "work",
        ledgers=SimpleNamespace(records_root=records_root),
    )
    stub = SimpleNamespace(
        paths=paths,
        config=SimpleNamespace(run_id="run-1", machine_id="machine-1"),
        dependencies=SimpleNamespace(clock=lambda: "2026-08-11T00:00:00.000000Z"),
    )
    return ReceiptDriverMixin._promote_award_corpus, stub, records_root


def _award_reducer(stable_id: str, digest: str):
    from types import SimpleNamespace

    event = {
        "claims": [
            {
                "stable_id": stable_id,
                "revision": digest,
                "license_disposition": "public-compatible",
            }
        ]
    }
    return SimpleNamespace(artifact_ledger=SimpleNamespace(events=[event]))


def _operational_ledger(tmp_path: Path) -> JsonlLedger:
    from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION

    return JsonlLedger(
        tmp_path / "records" / "operational" / "events.jsonl",
        OPERATIONAL_EVENT_SCHEMA_VERSION,
    )


def test_award_hook_promotes_and_mirrors_license_decisions(tmp_path):
    from types import SimpleNamespace

    _author_root(tmp_path)
    hook, stub, records_root = _award_stub(tmp_path)
    reducer = _award_reducer("m0001", hash_bytes(IMPL_BYTES))
    operational = _operational_ledger(tmp_path)
    try:
        hook(
            stub,
            SimpleNamespace(stable_id="m0001", active_work_id="work-m0001"),
            reducer,
            operational,
        )
        assert operational.record_count == 0, "clean promotion must not emit health events"
    finally:
        operational.close()

    rows = [json.loads(line) for line in corpus_index_path(records_root).read_text().splitlines()]
    impl_rows = [row for row in rows if row["content_sha256"] == hash_bytes(IMPL_BYTES)]
    assert impl_rows and impl_rows[0]["license_disposition"] == "public-compatible"
    assert corpus_payload_path(records_root, hash_bytes(IMPL_BYTES)).is_file()


def test_award_hook_discloses_refusals_on_the_operational_ledger(tmp_path):
    from types import SimpleNamespace

    _author_root(tmp_path, corrupt_impl_evidence=True)
    hook, stub, records_root = _award_stub(tmp_path)
    reducer = _award_reducer("m0001", hash_bytes(IMPL_BYTES))
    operational = _operational_ledger(tmp_path)
    try:
        hook(
            stub,
            SimpleNamespace(stable_id="m0001", active_work_id="work-m0001"),
            reducer,
            operational,
        )
        events = list(operational.iter_records())
        assert len(events) == 1
        details = events[0]["details"]
        assert details["disposition"] == "text-corpus-promotion-refusals"
        assert details["stable_id"] == "m0001"
        assert details["refusals"]
    finally:
        operational.close()


# -- W10: within-generation idempotent replay over declared volatile fields ---


def _repromotion_pack_mutation(author_root: Path, **moved: str) -> None:
    """Re-observe the same sources within the same work generation.

    Mutates ONLY the named advisory observation fields in the machine-written
    broker pack (identity fields, digests, work_id untouched) -- the exact
    shape of a within-generation re-promotion whose fetch happened again.
    """

    pack_path = author_root / "source-targets.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    for row in pack["sources"]:
        row.update(moved)
    for outcome in pack["broker"]["outcomes"]:
        outcome.update(moved)
    pack_path.write_text(json.dumps(pack, sort_keys=True))


def test_within_generation_replay_with_moved_volatile_fields_replays(tmp_path):
    """R19 RED repro: a re-promotion whose retrieved_at/final_url moved must
    land as ``replayed`` with the ledger retaining the immutable FIRST
    observation byte-identically -- never a conflicting-replay driver abort."""

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    first = promote_author_root(author_root, records_root)
    assert {row["disposition"] for row in first.dispositions} == {"promoted"}
    first_bytes = corpus_index_path(records_root).read_bytes()

    _repromotion_pack_mutation(
        author_root,
        retrieved_at="2026-08-12T09:00:00.000000Z",
        final_url="https://example.test/doc?cache-bust=2",
    )

    second = promote_author_root(author_root, records_root)
    assert {row["disposition"] for row in second.dispositions} == {"replayed"}
    assert corpus_index_path(records_root).read_bytes() == first_bytes


def test_within_generation_replay_with_moved_identity_field_still_refuses(tmp_path):
    """Negative arm: the volatile carve-out must never widen -- a non-volatile
    field moving under the same corpus_row_id stays a conflicting replay."""

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)

    _repromotion_pack_mutation(author_root, media_type="application/pdf")

    with pytest.raises(LedgerConflictError):
        promote_author_root(author_root, records_root)


def test_corpus_row_volatile_fields_are_the_declared_closed_list():
    """The volatile-field list is a CLOSED vocabulary (Wave-2 extension law):
    widening it is a reviewed change that must touch this pin, and it may
    never overlap the row-identity inputs (never widen corpus_row_id)."""

    from menagerie.crawler.text_corpus import _CORPUS_ROW_VOLATILE_FIELDS

    assert _CORPUS_ROW_VOLATILE_FIELDS == (
        "license_disposition",
        "retrieved_at",
        "final_url",
        "locator",
        "custody_disposition",
    )
    identity_inputs = {"stable_id", "content_sha256", "source_id", "role", "origin", "work_id"}
    assert identity_inputs.isdisjoint(_CORPUS_ROW_VOLATILE_FIELDS)
    # corpus_row_id stays exactly its six identity inputs: same inputs, same
    # id; a volatile-shaped observation never participates.
    base = dict(
        stable_id="m0001",
        content_sha256=hash_bytes(b"x"),
        source_id="s1",
        role="implementation",
        origin="custody-promotion",
        work_id="work-m0001",
    )
    assert corpus_row_id(**base) == corpus_row_id(**base)


def test_union_retains_first_volatile_observation_across_shards(tmp_path):
    """The union appender replays the same row_id whose volatile observation
    moved between shards, retaining the output store's first observation."""

    shard_a = _records_root(tmp_path / "a")
    shard_b = _records_root(tmp_path / "b")
    root_a = _author_root(tmp_path / "wa")
    root_b = _author_root(tmp_path / "wb")
    promote_author_root(root_a, shard_a)
    _repromotion_pack_mutation(root_b, retrieved_at="2026-08-13T00:00:00.000000Z")
    promote_author_root(root_b, shard_b)

    merged = _records_root(tmp_path / "merged")
    report = union_corpus((shard_a, shard_b), merged)
    assert report.rows_replayed > 0
    rows = list(scan_jsonl(corpus_index_path(merged)))
    retrieved = {row["corpus_row_id"]: row["retrieved_at"] for row in rows if row["retrieved_at"]}
    # shard_a promoted first into the union output; its observations win.
    assert all(value == "2026-08-11T00:00:00.000000Z" for value in retrieved.values())
