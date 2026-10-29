"""AB-30 matrix: corpus-index-row-replay seam (W10/AL-5 volatile-field replay).

The Wave-2 batch generalizes the historical license-only replay carve-out to
the DECLARED closed volatile-field list ``_CORPUS_ROW_VOLATILE_FIELDS``
(``license_disposition``, ``retrieved_at``, ``final_url``, ``locator``,
``custody_disposition``): a within-generation re-promotion of an existing
``corpus_row_id`` retains the immutable FIRST observation for exactly those
fields, so the ledger replay is byte-identical and the idempotent-replay rule
lands it as ``replayed`` -- never a conflicting-replay driver abort (R19: 11
members, two aborts). A moved NON-volatile field under the same row id must
still refuse as a typed ledger conflict, and ``corpus_row_id`` derivation
never widens (the m10949 lesson).

Every cell runs the REAL writer/readers: ``promote_author_root`` (promotion
writer + its ledger idempotent-replay reader over the existing index) and
``scan_jsonl`` over the corpus index. The prior-era artifact is a promoted
corpus row minted through that writer from a pre-batch-shaped author root
(2026-08-11 observations, default ``manifest-only`` license era).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.recordio import LedgerConflictError, scan_jsonl
from menagerie.crawler.tests.mixed_era import assert_era_outcome, classify_reader
from menagerie.crawler.tests.test_text_corpus import (
    IMPL_BYTES,
    _author_root,
    _records_root,
    _repromotion_pack_mutation,
)
from menagerie.crawler.text_corpus import (
    _CORPUS_ROW_VOLATILE_FIELDS,
    corpus_index_path,
    corpus_row_id,
    promote_author_root,
)

pytestmark = pytest.mark.smoke

#: The five declared volatile members (task cell 3 pins set equality below).
_DECLARED_VOLATILE = {
    "license_disposition",
    "retrieved_at",
    "final_url",
    "locator",
    "custody_disposition",
}

#: First-era observation values minted by the shared author-root fixture.
_FIRST_RETRIEVED_AT = "2026-08-11T00:00:00.000000Z"
_FIRST_IMPL_FINAL_URL = "https://example.test/doc"
_FIRST_PROBE_FINAL_URL = "https://example.test/probe"


def _move_custody_dispositions(author_root: Path, value: str) -> None:
    """Re-observe the custody disposition strings within the same generation.

    Mutates ONLY the advisory ``disposition`` field of the machine-written
    custody promotions (source ids, digests, work_id untouched).
    """

    path = author_root / "source-custody.json"
    custody = json.loads(path.read_text(encoding="utf-8"))
    for row in custody["promotions"]:
        row["disposition"] = value
    path.write_text(json.dumps(custody, sort_keys=True))


def _reobserve_every_volatile_field(author_root: Path) -> None:
    """Move EVERY declared volatile observation in the author envelopes.

    ``retrieved_at`` and ``final_url`` move directly; ``locator`` moves from
    absent to present via a resolver receipt on the re-fetch; the custody
    ``disposition`` string moves in the custody envelope. The fifth member,
    ``license_disposition``, moves at the promotion call site (it is a
    promotion-time input, not an envelope field).
    """

    _repromotion_pack_mutation(
        author_root,
        retrieved_at="2026-08-24T00:00:00.000000Z",
        final_url="https://example.test/doc?cache-bust=w2",
        resolver_receipt={
            "forge": "github",
            "repo": "example/net",
            "ref": "main",
            "resolved_sha": "a" * 40,
        },
    )
    _move_custody_dispositions(author_root, "re-promoted")


def _rows(records_root: Path) -> list[dict[str, Any]]:
    return scan_jsonl(corpus_index_path(records_root))


def _row_for(rows: list[dict[str, Any]], source_id: str) -> dict[str, Any]:
    matches = [row for row in rows if row["source_id"] == source_id]
    assert matches, f"no corpus row for {source_id!r}"
    return matches[0]


def test_repromotion_with_every_volatile_field_moved_replays_first_observation(
    tmp_path: Path,
) -> None:
    """Cell 1: all five volatile fields moved -> ``parsed`` (replayed).

    The re-promotion classifies as a lawful parse, every disposition is
    ``replayed``, the ledger bytes are identical, and each retained row keeps
    the FIRST observation for every declared volatile field.
    """

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    first = promote_author_root(author_root, records_root)
    assert {row["disposition"] for row in first.dispositions} == {"promoted"}
    first_bytes = corpus_index_path(records_root).read_bytes()
    first_digests = {row["content_sha256"] for row in _rows(records_root)}

    _reobserve_every_volatile_field(author_root)
    moved_licenses = {digest: "public-compatible" for digest in first_digests}

    outcome = classify_reader(
        lambda: promote_author_root(
            author_root, records_root, license_dispositions=moved_licenses
        )
    )
    assert_era_outcome(
        reader_id="corpus-promotion-volatile-replay",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    report = outcome[1]
    assert {row["disposition"] for row in report.dispositions} == {"replayed"}
    assert corpus_index_path(records_root).read_bytes() == first_bytes

    rows = _rows(records_root)
    impl = _row_for(rows, "impl-m0001")
    assert impl["retrieved_at"] == _FIRST_RETRIEVED_AT
    assert impl["final_url"] == _FIRST_IMPL_FINAL_URL
    assert impl["license_disposition"] == "manifest-only"
    assert impl["custody_disposition"] == "promoted"
    probe = _row_for(rows, "candidate-probe-001")
    assert probe["retrieved_at"] == _FIRST_RETRIEVED_AT
    assert probe["final_url"] == _FIRST_PROBE_FINAL_URL
    # The first observation carried NO resolver locator: retained absence IS
    # the first observation (present iff present there).
    assert "locator" not in probe
    assert all(row["license_disposition"] == "manifest-only" for row in rows)


def test_repromotion_with_moved_identity_field_refuses_typed(tmp_path: Path) -> None:
    """Cell 2: a moved NON-volatile row field stays a typed conflict refusal.

    ``media_type`` is outside the volatile carve-out, so the same
    ``corpus_row_id`` re-presenting different row content must refuse as a
    crawler-typed conflicting replay in every era -- the tripwire stays armed.
    """

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)

    _repromotion_pack_mutation(author_root, media_type="application/pdf")

    outcome = classify_reader(lambda: promote_author_root(author_root, records_root))
    assert_era_outcome(
        reader_id="corpus-promotion-identity-conflict",
        outcome=outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(outcome[1], LedgerConflictError)


def test_volatile_field_list_is_exactly_the_declared_closed_five() -> None:
    """Cell 3: the volatile list is EXACTLY the five declared members.

    Closed vocabulary (Wave-2 extension law): widening it is a reviewed
    change that must touch this matrix pin in the same change, and it may
    never overlap the six ``corpus_row_id`` identity inputs.
    """

    assert set(_CORPUS_ROW_VOLATILE_FIELDS) == _DECLARED_VOLATILE
    assert len(_CORPUS_ROW_VOLATILE_FIELDS) == len(_DECLARED_VOLATILE)
    identity_inputs = {
        "stable_id",
        "content_sha256",
        "source_id",
        "role",
        "origin",
        "work_id",
    }
    assert identity_inputs.isdisjoint(_CORPUS_ROW_VOLATILE_FIELDS)


def test_corpus_row_id_is_byte_identical_across_eras(tmp_path: Path) -> None:
    """Cell 4: the replayed row's ``corpus_row_id`` never moves.

    Across the volatile re-observation the persisted row ids are
    byte-identical, and each one still recomputes from exactly its six
    identity inputs -- no volatile observation participates in derivation.
    """

    author_root = _author_root(tmp_path)
    records_root = _records_root(tmp_path)
    promote_author_root(author_root, records_root)
    first_rows = _rows(records_root)
    first_ids = sorted(str(row["corpus_row_id"]).encode("ascii") for row in first_rows)

    _reobserve_every_volatile_field(author_root)
    all_digests = {row["content_sha256"] for row in first_rows}
    promote_author_root(
        author_root,
        records_root,
        license_dispositions={digest: "public-compatible" for digest in all_digests},
    )

    second_rows = _rows(records_root)
    second_ids = sorted(str(row["corpus_row_id"]).encode("ascii") for row in second_rows)
    assert second_ids == first_ids
    for row in second_rows:
        assert row["corpus_row_id"] == corpus_row_id(
            stable_id=row["stable_id"],
            content_sha256=row["content_sha256"],
            source_id=row["source_id"],
            role=row["role"],
            origin=row["origin"],
            work_id=row["work_id"],
        )
    impl = _row_for(second_rows, "impl-m0001")
    assert impl["content_sha256"] == hash_bytes(IMPL_BYTES)
