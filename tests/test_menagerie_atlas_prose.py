"""Tests for the menagerie atlas-prose store, schema, and traceability gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from menagerie.atlas_prose import (
    MAX_SUMMARY_PARAGRAPHS,
    MIN_SUMMARY_CHARS,
    SCHEMA_FORMAT,
    AtlasProseGateError,
    append_entry,
    canonical_catalog_revision,
    catalog_row_is_trace_verified,
    crawler_record_is_verified,
    current_entry,
    load_store_file,
    store_path,
    validate_entry_against_record,
    validate_row_schema,
)

PROMPT_SHA = "sha256:" + hashlib.sha256(b"prompt-v1").hexdigest()


def verified_record() -> dict:
    """Build a minimal metadata-verified crawler model record.

    Returns
    -------
    dict
        Record with an accurate gate, accepted metadata, and evidence excerpts.
    """

    return {
        "stable_id": "m9430",
        "record_revision": "sha256:" + "c" * 64,
        "accuracy_gate": {"verdict": "accurate"},
        "authored_metadata_state": "accepted",
        "citation": {
            "title": "Feature Pyramid Networks for Object Detection",
            "authors": ["Tsung-Yi Lin", "Piotr Dollar"],
            "year": 2016,
            "venue": None,
        },
        "taxonomy": {"family": "fpn-segmentation", "domains": ["computer-vision"]},
        "identity": {"canonical_name": "smp_FPN_resnext101_32x16d"},
        "evidence": {
            "excerpts": [
                {"evidence_id": "ex-fpn-class", "text": "class FPN(SegmentationModel):"},
                {"evidence_id": "ex-fpn-meta", "text": "arXiv:1612.03144"},
            ]
        },
    }


def prose_row(**overrides: object) -> dict:
    """Build a schema-valid prose row bound to :func:`verified_record`.

    Parameters
    ----------
    overrides:
        Field overrides applied on top of the valid baseline.

    Returns
    -------
    dict
        Atlas-prose row.
    """

    paragraph = (
        "FPN is a feature-pyramid segmentation network built on a ResNeXt encoder, "
        "and this sentence pads the fixture out to a plausible summary length so the "
        "character floor does not trip in unrelated tests."
    )
    row = {
        "stable_id": "m9430",
        "summary_markdown": paragraph + "\n\n" + paragraph,
        "facts_used": ["citation.title", "taxonomy.family", "citation.authors.0"],
        "evidence_ids": ["ex-fpn-class", "ex-fpn-meta"],
        "author_model": "claude-fable-5",
        "prompt_sha256": PROMPT_SHA,
        "record_revision": "sha256:" + "c" * 64,
        "created_at": "2026-08-12T01:00:00Z",
        "source_kind": "crawler-record",
    }
    row.update(overrides)
    return row


class TestRowSchema:
    """Structural validation of a single prose row."""

    def test_valid_row_passes(self) -> None:
        """A well-formed row validates without error."""

        validate_row_schema(prose_row())

    @pytest.mark.parametrize(
        "field",
        [
            "stable_id",
            "summary_markdown",
            "facts_used",
            "evidence_ids",
            "author_model",
            "prompt_sha256",
            "record_revision",
            "created_at",
            "source_kind",
        ],
    )
    def test_missing_required_field_fails(self, field: str) -> None:
        """Dropping any required field is a schema failure."""

        row = prose_row()
        del row[field]
        with pytest.raises(AtlasProseGateError) as err:
            validate_row_schema(row)
        assert err.value.code == "schema-invalid"

    def test_bad_prompt_digest_fails(self) -> None:
        """The prompt digest must be a sha256: hex literal."""

        with pytest.raises(AtlasProseGateError) as err:
            validate_row_schema(prose_row(prompt_sha256="v1"))
        assert err.value.code == "schema-invalid"

    def test_bad_created_at_fails(self) -> None:
        """created_at must be an ISO-8601 UTC Z timestamp."""

        with pytest.raises(AtlasProseGateError) as err:
            validate_row_schema(prose_row(created_at="yesterday"))
        assert err.value.code == "schema-invalid"

    def test_unknown_source_kind_fails(self) -> None:
        """source_kind comes from the closed vocabulary."""

        with pytest.raises(AtlasProseGateError) as err:
            validate_row_schema(prose_row(source_kind="oral-tradition"))
        assert err.value.code == "schema-invalid"

    def test_empty_facts_used_fails(self) -> None:
        """facts_used must be non-empty: prose without facts is invention."""

        with pytest.raises(AtlasProseGateError) as err:
            validate_row_schema(prose_row(facts_used=[]))
        assert err.value.code == "schema-invalid"

    def test_summary_below_char_floor_fails(self) -> None:
        """A summary shorter than the character floor fails length bounds."""

        assert MIN_SUMMARY_CHARS > len("Too short.\n\nStill too short.")
        with pytest.raises(AtlasProseGateError) as err:
            validate_row_schema(prose_row(summary_markdown="Too short.\n\nStill too short."))
        assert err.value.code == "length-out-of-bounds"

    def test_too_many_paragraphs_fails(self) -> None:
        """More paragraphs than the ceiling fails the paragraph bound."""

        block = "One honest paragraph about the model, padded well past the floor " * 3
        summary = "\n\n".join([block] * (MAX_SUMMARY_PARAGRAPHS + 1))
        with pytest.raises(AtlasProseGateError) as err:
            validate_row_schema(prose_row(summary_markdown=summary))
        assert err.value.code == "paragraphs-out-of-bounds"


class TestTraceability:
    """Facts and evidence in a row must resolve against the bound record."""

    def test_valid_entry_passes(self) -> None:
        """A row whose facts and evidence all resolve validates cleanly."""

        validate_entry_against_record(prose_row(), verified_record())

    def test_unresolved_fact_path_fails(self) -> None:
        """A facts_used path absent from the record is a traceability failure."""

        row = prose_row(facts_used=["citation.title", "citation.nobel_prizes"])
        with pytest.raises(AtlasProseGateError) as err:
            validate_entry_against_record(row, verified_record())
        assert err.value.code == "facts-path-unresolved"
        assert "citation.nobel_prizes" in str(err.value)

    def test_fact_path_resolving_to_null_fails(self) -> None:
        """A path resolving to null is not a usable fact."""

        row = prose_row(facts_used=["citation.venue"])
        with pytest.raises(AtlasProseGateError) as err:
            validate_entry_against_record(row, verified_record())
        assert err.value.code == "facts-path-unresolved"

    def test_list_index_fact_path_resolves(self) -> None:
        """Integer segments index into record lists."""

        row = prose_row(facts_used=["citation.authors.1"])
        validate_entry_against_record(row, verified_record())

    def test_unknown_evidence_id_fails(self) -> None:
        """An evidence id absent from the record's excerpts fails."""

        row = prose_row(evidence_ids=["ex-fpn-class", "ex-invented"])
        with pytest.raises(AtlasProseGateError) as err:
            validate_entry_against_record(row, verified_record())
        assert err.value.code == "evidence-id-unresolved"

    def test_stable_id_mismatch_fails(self) -> None:
        """The row must describe the record it is bound to."""

        row = prose_row(stable_id="m9999")
        with pytest.raises(AtlasProseGateError) as err:
            validate_entry_against_record(row, verified_record())
        assert err.value.code == "stable-id-mismatch"

    def test_revision_mismatch_fails(self) -> None:
        """The row must bind the record revision it describes."""

        row = prose_row(record_revision="sha256:" + "d" * 64)
        with pytest.raises(AtlasProseGateError) as err:
            validate_entry_against_record(row, verified_record())
        assert err.value.code == "revision-mismatch"

    def test_unverified_record_refused(self) -> None:
        """Prose may only derive from metadata-verified records."""

        record = verified_record()
        record["accuracy_gate"]["verdict"] = "cannot-verify"
        record["authored_metadata_state"] = "failed"
        assert not crawler_record_is_verified(record)
        with pytest.raises(AtlasProseGateError) as err:
            validate_entry_against_record(prose_row(), record)
        assert err.value.code == "record-not-verified"


def catalog_row() -> dict:
    """Build a trace-verified catalog row fixture.

    Returns
    -------
    dict
        Master-catalog-shaped row for a classics model.
    """

    return {
        "name": "LeNet-5",
        "zoo": "historical-classics",
        "family": "cnn-classic",
        "domain": "vision/classification-backbone",
        "era": "1998",
        "input_shape": "(1,1,32,32)",
        "notes": "repaired 2026-06-18; verified instantiate+trace n_ops=14",
        "recipe": {"type": "exec-string", "body": "model = LeNet5()"},
    }


class TestCatalogRowSource:
    """Catalog-row sources: classics prose binds the catalog row itself."""

    def test_trace_verified_marker_detection(self) -> None:
        """Rows are eligible only with an explicit trace-verification stamp."""

        assert catalog_row_is_trace_verified(catalog_row())
        bare = catalog_row()
        bare["notes"] = "added 2026-06-18"
        assert not catalog_row_is_trace_verified(bare)

    def test_catalog_entry_validates(self) -> None:
        """A catalog-row entry with resolvable facts and no evidence ids passes."""

        record = catalog_row()
        row = prose_row(
            stable_id="m1573",
            source_kind="catalog-row",
            source_name="LeNet-5",
            facts_used=["name", "era", "recipe.body", "notes"],
            evidence_ids=[],
            record_revision=canonical_catalog_revision(record),
        )
        validate_entry_against_record(row, record)

    def test_catalog_entry_with_evidence_ids_fails(self) -> None:
        """Catalog rows carry no excerpt store, so evidence_ids must be empty."""

        record = catalog_row()
        row = prose_row(
            stable_id="m1573",
            source_kind="catalog-row",
            source_name="LeNet-5",
            facts_used=["name"],
            evidence_ids=["ex-anything"],
            record_revision=canonical_catalog_revision(record),
        )
        with pytest.raises(AtlasProseGateError) as err:
            validate_entry_against_record(row, record)
        assert err.value.code == "evidence-id-unresolved"

    def test_catalog_entry_requires_source_name(self) -> None:
        """catalog-row entries must name the row for later re-validation."""

        record = catalog_row()
        row = prose_row(
            stable_id="m1573",
            source_kind="catalog-row",
            facts_used=["name"],
            evidence_ids=[],
            record_revision=canonical_catalog_revision(record),
        )
        row.pop("source_name", None)
        with pytest.raises(AtlasProseGateError) as err:
            validate_row_schema(row)
        assert err.value.code == "schema-invalid"

    def test_untraced_catalog_row_refused(self) -> None:
        """A catalog row without the trace stamp is refused."""

        record = catalog_row()
        record["notes"] = "harvested, never traced"
        row = prose_row(
            stable_id="m1573",
            source_kind="catalog-row",
            source_name="LeNet-5",
            facts_used=["name"],
            evidence_ids=[],
            record_revision=canonical_catalog_revision(record),
        )
        with pytest.raises(AtlasProseGateError) as err:
            validate_entry_against_record(row, record)
        assert err.value.code == "row-not-trace-verified"

    def test_canonical_revision_is_stable(self) -> None:
        """The catalog revision digest is key-order independent."""

        row = catalog_row()
        shuffled = dict(reversed(list(row.items())))
        assert canonical_catalog_revision(row) == canonical_catalog_revision(shuffled)
        assert canonical_catalog_revision(row).startswith("sha256:")


class TestStore:
    """Append-only per-model store files."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """An appended entry loads back as the current entry."""

        row = prose_row()
        path = append_entry(tmp_path, row)
        assert path == store_path(tmp_path, "m9430")
        store = load_store_file(path)
        assert store["format"] == SCHEMA_FORMAT
        assert store["stable_id"] == "m9430"
        assert current_entry(store) == row

    def test_append_supersedes_and_preserves_history(self, tmp_path: Path) -> None:
        """A newer row supersedes; the prior row survives verbatim."""

        first = prose_row()
        second = prose_row(created_at="2026-08-12T02:00:00Z")
        append_entry(tmp_path, first)
        append_entry(tmp_path, second)
        store = load_store_file(store_path(tmp_path, "m9430"))
        assert store["entries"] == [first, second]
        assert current_entry(store) == second

    def test_append_validates_schema(self, tmp_path: Path) -> None:
        """The store refuses schema-invalid rows."""

        row = prose_row()
        del row["author_model"]
        with pytest.raises(AtlasProseGateError) as err:
            append_entry(tmp_path, row)
        assert err.value.code == "schema-invalid"
        assert not store_path(tmp_path, "m9430").exists()

    def test_load_rejects_malformed_store(self, tmp_path: Path) -> None:
        """A corrupt store file is a typed failure, never silently repaired."""

        path = store_path(tmp_path, "m9430")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        with pytest.raises(AtlasProseGateError) as err:
            load_store_file(path)
        assert err.value.code == "store-malformed"

    def test_load_rejects_wrong_format(self, tmp_path: Path) -> None:
        """A store file with a foreign format marker is refused."""

        path = store_path(tmp_path, "m9430")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": "v0", "stable_id": "m9430", "entries": []}))
        with pytest.raises(AtlasProseGateError) as err:
            load_store_file(path)
        assert err.value.code == "store-malformed"


class TestCommittedStore:
    """Every committed atlas-prose file must satisfy the structural gate."""

    def test_committed_entries_are_schema_valid(self) -> None:
        """All committed store files parse, validate, and match their filenames."""

        store_dir = Path(__file__).resolve().parents[1] / "menagerie" / "data" / "atlas_prose"
        if not store_dir.exists():
            pytest.skip("no committed atlas-prose store yet")
        files = sorted(store_dir.glob("*.json"))
        assert files, "atlas_prose store dir exists but is empty"
        catalog_path = store_dir.parent / "master_catalog.jsonl"
        catalog = {}
        with catalog_path.open() as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    catalog[record["name"]] = record
        for path in files:
            store = load_store_file(path)
            assert store["stable_id"] == path.stem
            for row in store["entries"]:
                validate_row_schema(row)
                assert row["stable_id"] == store["stable_id"]
                if row["source_kind"] == "catalog-row":
                    validate_entry_against_record(row, catalog[row["source_name"]])
