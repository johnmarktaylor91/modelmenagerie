"""Local release behavior, schema coverage, currency and Cloudflare boundary counts."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie import csv_dictionary
from menagerie.pipeline import (
    StoreLayout, append_jsonl, dataset, publish, read_json, tree_sha256, write_json,
)
from menagerie.pipeline.records import Hashes, LedgerRow, ModelRecord, RecordError
from test_publish import _data_root, _rows


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(publish, "torchlens_pin", lambda: "2.35.0")
    data = _data_root(tmp_path)
    store = StoreLayout(tmp_path / "store")
    monkeypatch.setenv("MENAGERIE_STORE", str(store.root))
    trace = next(r for r in _rows(data) if isinstance(r, LedgerRow) and r.id == "m8839")
    summary = {"n_params": 11_689_512, "graph_depth": 17, "pct_conv": 30.5,
               "is_recurrent": False, "output_shape": "(1, 1000)", "n_output_tensors": 1,
               "op_type_histogram": {"conv2d": 20}, "input_dtype": "float32",
               "trace_summary_version": "1.2.0", "stable_id": "spoofed-id"}
    blob = json.dumps(summary).encode()
    sha = hashlib.sha256(blob).hexdigest()
    store.blob(sha).parent.mkdir(parents=True)
    store.blob(sha).write_bytes(blob)
    passed = replace(trace, at="2026-10-05T07:00:00Z", summary=sha,
                     validation_state="passed", graph_hash="a" * 64,
                     hashes=Hashes(skeleton="b" * 64, architecture="c" * 64, schema=1),
                     n_ops=473, params=11_689_512)
    append_jsonl([passed], data.ledger("2026-10"))
    # One untraced row and one structure-only hypothesis are carried, never counted as validated.
    model = read_json(ModelRecord, data.model("m3631"))
    hypothesis = replace(passed, id=model.id, recipe_rev=model.recipe.rev,
                         capture="structure_only", validation_state="structure_only",
                         params_source="geometry")
    append_jsonl([hypothesis], data.ledger("2026-10"))
    untraced = replace(model, id="m99999", name="untraced")
    write_json(untraced, data.model(untraced.id))
    return data, store, passed


def _csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_release_integrity(released: Path):
    manifest = json.loads((released / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        file = released / name
        assert info == {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                        "bytes": file.stat().st_size}
    with zipfile.ZipFile(released / "bundles/menagerie_csv.zip") as archive:
        assert set(archive.namelist()) == {f"csv/{n}" for n in (
            *dataset.TABLES, "DATA_DICTIONARY.md")}


def test_release_six_tables_schema_types_checksums_and_bundle(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    data, _, _ = _fixture(tmp_path, monkeypatch)
    root = tmp_path / "dataset"
    report = dataset.build(data=data, out=root)
    released = root / "v1"
    flagship = _csv(released / "menagerie.csv")
    assert list(flagship[0]) == [c.name for c in csv_dictionary.parse_flagship_schema()]
    assert len(flagship[0]) == 78 and len(flagship) == 3
    rows = {r["stable_id"]: r for r in flagship}
    assert rows["m8839"]["is_trustworthy"] == "1"
    assert rows["m8839"]["graph_depth"] == "17"
    assert rows["m8839"]["publication_year"] == "2015"
    assert rows["m8839"]["output_type"] == "class-logits"
    assert rows["m3631"]["is_trustworthy"] == "0"
    assert rows["m99999"]["graph_depth"] == ""
    assert report["null_rates"]["menagerie.csv"]["graph_depth"]["rate"] == 1 / 3
    metrics = pq.read_table(released / "trace_metrics.parquet")
    assert metrics.column_names == list(csv_dictionary.TRACE_METRICS_COLUMNS)
    assert str(metrics.schema.field("graph_depth").type) == "int64"
    assert str(metrics.schema.field("pct_conv").type) == "double"
    assert metrics.to_pylist()[2]["graph_depth"] is None
    assert {r["stable_id"] for r in metrics.to_pylist()} == set(rows)
    for name in ("papers.csv", "artifacts.csv"):
        assert {r["stable_id"] for r in _csv(released / name)} == set(rows)
    _assert_release_integrity(released)
    assert json.loads((root / "latest.json").read_text())["release"] == "v1"
    before = tree_sha256(root)
    with pytest.raises(RecordError, match="immutable"):
        dataset.build(data=data, out=root)
    assert tree_sha256(root) == before


def test_dry_run_and_failed_bundle_do_not_publish(tmp_path, monkeypatch):
    from menagerie import bundle
    data, _, _ = _fixture(tmp_path, monkeypatch)
    root = tmp_path / "dataset"
    assert dataset.build(data=data, out=root, dry_run=True)["counts"]["menagerie.csv"] == 3
    assert not root.exists()
    def failed(**kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(bundle, "build_bundles", failed)
    with pytest.raises(OSError, match="disk full"):
        dataset.build(data=data, out=root)
    assert not (root / "v1").exists() and not (root / "latest.json").exists()
    assert not list(root.glob(".v1-*"))


def test_infra_preserves_pass_but_real_failure_and_recipe_change_retire_it(tmp_path, monkeypatch):
    data, _, passed = _fixture(tmp_path, monkeypatch)
    append_jsonl([replace(passed, at="2026-10-06T07:00:00Z", outcome="infra", cls="infra:worker",
                         validation_state=None)], data.ledger("2026-10"))
    raw, _ = dataset._tables(data, as_of="2026-10-08", commit="a" * 40)
    assert next(r for r in raw["menagerie.csv"] if r["stable_id"] == "m8839")["is_trustworthy"]
    append_jsonl([replace(passed, at="2026-10-07T07:00:00Z", outcome="fail", cls="capture_error",
                         validation_state="failed")], data.ledger("2026-10"))
    raw, _ = dataset._tables(data, as_of="2026-10-08", commit="a" * 40)
    failed = next(r for r in raw["menagerie.csv"] if r["stable_id"] == "m8839")
    assert not failed["is_trustworthy"] and failed["graph_depth"] is None
    artifact = next(r for r in raw["artifacts.csv"] if r["stable_id"] == "m8839")
    assert "no_artifact" in artifact["row_quality_flags_json"]
    model = read_json(ModelRecord, data.model("m8839"))
    write_json(replace(model, recipe=replace(model.recipe, rev="f" * 12)), data.model(model.id))
    raw, _ = dataset._tables(data, as_of="2026-10-08", commit="a" * 40)
    row = next(r for r in raw["menagerie.csv"] if r["stable_id"] == "m8839")
    assert row["n_ops"] is None and row["graph_depth"] is None


def test_missing_summary_is_a_hard_error(tmp_path, monkeypatch):
    data, store, passed = _fixture(tmp_path, monkeypatch)
    store.blob(passed.summary).unlink()
    with pytest.raises(FileNotFoundError):
        dataset.build(data=data, out=tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_corruption_and_compatible_stale_validation(tmp_path, monkeypatch):
    data, store, passed = _fixture(tmp_path, monkeypatch)
    model = read_json(ModelRecord, data.model(passed.id))
    write_json(replace(model, recipe=replace(model.recipe, rev="f" * 12)), data.model(model.id))
    raw, _ = dataset._tables(data, as_of="2026-10-08", commit="a" * 40)
    row = next(r for r in raw["menagerie.csv"] if r["stable_id"] == passed.id)
    assert row["is_trustworthy"] and not row["validated_on_current_release"]
    assert row["trust_tier"] == "stale_recipe" and row["graph_depth"] is None
    write_json(model, data.model(model.id))
    store.blob(passed.summary).write_text("{}")
    with pytest.raises(RecordError, match="checksum"):
        dataset.build(data=data, out=tmp_path / "dataset")


@pytest.mark.parametrize("refusal", ["standin", "fabrication"])
def test_refused_architecture_keeps_metadata_but_no_trace_values(tmp_path, monkeypatch, refusal):
    data, _, passed = _fixture(tmp_path, monkeypatch)
    if refusal == "standin":
        append_jsonl([replace(passed, at="2026-10-07T07:00:00Z", outcome="giveup",
                             cls="standin:dense")], data.ledger("2026-10"))
    else:
        review = LedgerRow(id=passed.id, stage="review", at="2026-10-07T07:00:00Z",
                           host=passed.host, env=passed.env, recipe_rev=passed.recipe_rev,
                           outcome="pass", verdict="fabrication")
        append_jsonl([review], data.ledger("2026-10"))
    raw, _ = dataset._tables(data, as_of="2026-10-08", commit="a" * 40)
    row = next(r for r in raw["menagerie.csv"] if r["stable_id"] == passed.id)
    assert row["display_name"] == "resnet18" and not row["is_trustworthy"]
    assert row["n_ops"] is None and row["n_params"] is None
    assert row["dedup_architecture_key"] is None and row["svg_url"] is None
    assert all(row[k] is None for k in publish.trace_metrics.FLAGSHIP)
    for name in ("trace_metrics.parquet", "trace_histograms.jsonl"):
        assert next(r for r in raw[name] if r["stable_id"] == passed.id) == {"stable_id": passed.id}
    artifact = next(r for r in raw["artifacts.csv"] if r["stable_id"] == passed.id)
    assert artifact["tlspec_path"] is None


def test_render_failure_preserves_trace_trust_and_removes_asset(tmp_path, monkeypatch):
    from menagerie.pipeline.records import RenderArtifact, RenderRow
    data, _, passed = _fixture(tmp_path, monkeypatch)
    artifacts = tuple(RenderArtifact(kind=k, sha256="d" * 64, bytes=10,
                                    variant="unrolled-none") for k in ("svg", "thumb"))
    rendered = RenderRow(id=passed.id, at="2026-10-06T07:00:00Z", trace=passed.trace,
                         renderer_rev=dataset.render.renderer_rev(), tl=passed.tl,
                         outcome="pass", renders=artifacts)
    append_jsonl([rendered], data.ledger("2026-10"))
    raw, _ = dataset._tables(data, as_of="2026-10-08", commit="a" * 40)
    assert next(r for r in raw["menagerie.csv"] if r["stable_id"] == passed.id)["svg_url"]
    append_jsonl([replace(rendered, at="2026-10-07T07:00:00Z", outcome="fail", renders=())],
                 data.ledger("2026-10"))
    raw, _ = dataset._tables(data, as_of="2026-10-08", commit="a" * 40)
    row = next(r for r in raw["menagerie.csv"] if r["stable_id"] == passed.id)
    assert row["is_trustworthy"] and row["svg_url"] is None


def test_site_counts_search_and_size_boundaries(tmp_path):
    dist = tmp_path / "dist"
    (dist / "pagefind").mkdir(parents=True)
    (dist / "index.html").write_text("hello")
    (dist / "pagefind/x.pf_fragment").write_text("fragment")
    asset = dist / "large.bin"
    with asset.open("wb") as handle:
        handle.truncate(25 * 1024 * 1024)
    result = dataset.site_count(dist, out=tmp_path / "count.json")
    assert (result["files"], result["html"], result["assets"], result["search_fragments"]) == (3, 1, 1, 1)
    assert result["free"]["fits"] and result["paid"]["fits"]
    with asset.open("ab") as handle:
        handle.write(b"x")
    assert not dataset.site_count(dist)["free"]["fits"]
    with pytest.raises(ValueError, match="outside"):
        dataset.site_count(dist, out=dist / "report.json")
    with pytest.raises(FileNotFoundError):
        dataset.site_count(tmp_path / "missing")
    (dist / "link").symlink_to(asset)
    with pytest.raises(ValueError, match="symlink"):
        dataset.site_count(dist)


def test_site_file_limit_inclusive_boundary(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").touch()
    for i in range(19_999):
        (dist / f"asset-{i}").touch()
    assert dataset.site_count(dist)["free"] == {"fits": True, "headroom": 0}
    (dist / "one-more").touch()
    report = dataset.site_count(dist)
    assert report["free"] == {"fits": False, "headroom": -1}
    assert report["paid"]["fits"] and report["limits"]["paid_files"] == 100_000
