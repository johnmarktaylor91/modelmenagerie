"""publish_public: renderer diagnostics out of the drawings, references only to published files, the
gzipped release; and the export's hidden-record drop and TorchLens split (publish.py)."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import StoreLayout, publish, publish_assets, publish_public
from menagerie.pipeline.records import RecordError

DIAG = ('<text text-anchor="start" x="8" y="-894.7" font-family="Helvetica,sans-Serif" '
        'font-weight="bold" font-size="14.00">collapse fell back to the floor plan (11 nodes '
        'visible) &#45;&#45; root owns 118 units, above K_CAP</text>\n')
SVG = ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
       '<svg width="300pt" height="900pt" viewBox="0.00 0.00 300.00 900.00" '
       'xmlns="http://www.w3.org/2000/svg">\n<g id="graph0" class="graph">'
       '<rect x="0" y="0" width="300" height="900" fill="white"/>\n'
       '<text text-anchor="start" x="8" y="-910" font-size="14.00">tiny</text>\n'
       f'{DIAG}<rect x="20" y="20" width="260" height="120" fill="#4a7"/></g>\n</svg>\n')


def _put(store: StoreLayout, data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    path = store.blob(sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha


def _records(export: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (export / "models.jsonl").read_text().splitlines()]


def _export(tmp_path: Path, svg_text: str = SVG, **record: Any) -> tuple[Path, StoreLayout, str]:
    store = StoreLayout(tmp_path / "store")
    svg = _put(store, svg_text.encode())
    pdf = _put(store, b"%PDF-1.4 tiny")
    variant = {"key": "unrolled-none", "label": "Unrolled, full", "monster": False,
               "svg": publish.asset(svg, "svg", len(svg_text.encode())),
               "thumb": publish.asset(svg, "svg", len(svg_text.encode())),
               "pdf": publish.asset(pdf, "pdf", 13), "tiles": None}
    row = {"stable_id": "m1", "slug": "tiny", "default_variant": "unrolled-none",
           "variants": [variant], "interactive": None, "netron": None, "model_explorer": None,
           "tlspec": None, **record}
    export = tmp_path / "export"
    export.mkdir()
    publish._write_jsonl(export / "models.jsonl", [row])
    publish.write_assets(export / "assets.tsv", [row])
    (export / "export_manifest.json").write_text(json.dumps({"as_of": "2026-11-03",
                                                             "counts": {"models": 1}}))
    return export, store, svg


def test_strip_diagnostic_removes_only_the_diagnostic_line() -> None:
    cleaned = publish_public.strip_diagnostic(SVG.encode())
    assert cleaned is not None and b"collapse fell back" not in cleaned
    assert b">tiny</text>" in cleaned and cleaned.count(b"\n") == SVG.count("\n") - 1
    assert publish_public.strip_diagnostic(cleaned) is None


def test_clean_points_the_record_at_a_copy_without_the_diagnostic(tmp_path: Path) -> None:
    export, store, svg = _export(tmp_path)
    report = publish_public.clean(export, store=store, workers=1)
    (variant,) = _records(export)[0]["variants"]
    new = variant["svg"]["sha256"]
    assert new != svg and variant["pdf"] is None
    copy = (export / "derived" / f"{new}.svg").read_bytes()
    assert b"collapse fell back" not in copy and hashlib.sha256(copy).hexdigest() == new
    assert variant["thumb"]["url"] == variant["svg"]["url"]  # the record's picture was the SVG
    assert report["diagnostic_svgs_cleaned"] == 1 and report["diagnostic_pdfs_removed"] == 1
    counts = json.loads((export / "export_manifest.json").read_text())["counts"]
    assert counts["diagnostic_variants_cleaned"] == 1
    assert new in (export / "assets.tsv").read_text()


def test_a_second_clean_reads_nothing_new_and_a_clean_svg_is_left_alone(tmp_path: Path) -> None:
    export, store, svg = _export(tmp_path, svg_text=SVG.replace(DIAG, ""))
    first = publish_public.clean(export, store=store, workers=1)
    assert first["diagnostic_svgs_cleaned"] == 0 and first["read"] == 1
    assert _records(export)[0]["variants"][0]["svg"]["sha256"] == svg
    assert publish_public.clean(export, store=store, workers=1)["read"] == 0


def test_derive_and_stage_read_the_cleaned_copy(tmp_path: Path) -> None:
    export, store, _ = _export(tmp_path)
    publish_public.clean(export, store=store, workers=1)
    publish_assets.derive(export, store=store, workers=1, memory_gb=2)
    (variant,) = _records(export)[0]["variants"]
    assert variant["svg_width_px"] == 400  # measured from the copy, which the store does not hold
    report = publish_assets.stage(export, tmp_path / "assets", store=store, kinds="svg,thumb,pdf")
    assert report["missing"] == []
    assert (tmp_path / "assets" / f"{variant['svg']['sha256']}.svg").is_file()


def _staged(tmp_path: Path, keys: list[str]) -> Path:
    assets = tmp_path / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "MANIFEST.tsv").write_text("key\tsha256\tbytes\tcontent_type\n"
                                         + "".join(f"{k}\tx\t1\tx\n" for k in keys))
    return assets


def test_restrict_removes_unpublished_kinds_and_named_missing_kinds(tmp_path: Path) -> None:
    html = publish.asset("ab" * 32, "html", 10)
    export, _, svg = _export(tmp_path, interactive=html)
    assets = _staged(tmp_path, [f"{svg}.svg"])
    report = publish_public.restrict(export, assets, kinds="svg,thumb,pdf", null_missing=["pdf"])
    record = _records(export)[0]
    assert record["interactive"] is None and record["variants"][0]["pdf"] is None
    assert report["refs_removed"] == {"missing:pdf": 1, "unpublished:html": 1}
    assert report["refs_unstaged"] == 0


def test_restrict_keeps_a_missing_svg_named_so_sanity_refuses_it(tmp_path: Path) -> None:
    export, _, _ = _export(tmp_path)
    report = publish_public.restrict(export, _staged(tmp_path, []), kinds="svg,thumb,pdf",
                                     null_missing=["pdf"])
    assert report["refs_unstaged"] == 2  # the SVG and the small picture (the same SVG)
    assert {kind for _, kind in report["unstaged"]} == {"svg"}
    with pytest.raises(RecordError):
        publish_public.restrict(export, _staged(tmp_path, []), kinds="svg", null_missing=["svg"])


def test_release_writes_gzipped_public_records_and_names_them(tmp_path: Path) -> None:
    export, _, svg = _export(tmp_path)
    assets = _staged(tmp_path, [f"{svg}.svg"])
    report = publish_public.release(export, assets, base="https://assets.example.org/")
    target = assets / report["path"]
    lines = gzip.decompress(target.read_bytes()).decode().splitlines()
    assert len(lines) == 1 and f"https://assets.example.org/{svg}.svg?v=1" in lines[0]
    assert "/export/assets/" not in lines[0]
    release = json.loads((export / "export_manifest.json").read_text())["dataset_release"]
    assert release["version"] == "2026-11-03" and release["files"][0]["path"] == report["path"]
    assert report["path"] in (assets / "MANIFEST.tsv").read_text()
    again = publish_public.release(export, assets, base="https://assets.example.org/")
    assert again["path"] == report["path"]  # the same records give the same file
    assert (assets / "MANIFEST.tsv").read_text().count("dataset/") == 1


@pytest.mark.parametrize(("record", "why"), [
    ({"fidelity": "fabrication", "validation_state": "passed", "forward_validated": True}, "fabrication"),
    ({"own_diagram": False}, "not_own_diagram"),
    ({"capture": "structure_only", "validation_state": "structure_only"}, None),
    ({}, None),
    ({"validation_state": "passed", "forward_validated": True}, None),
    ({"validation_state": "failed", "forward_validated": False}, "not_validated"),
])
def test_hidden_reason_mirrors_the_site_tiers(record: dict[str, Any], why: str | None) -> None:
    assert publish.hidden_reason(record) == why


def test_version_counts_split_checked_from_structure_only() -> None:
    rows = [{"torchlens_version": "2.36.0", "forward_validated": True, "capture": "full"},
            {"torchlens_version": "2.36.0", "forward_validated": False, "capture": "structure_only"},
            {"torchlens_version": "2.35.3", "forward_validated": True, "capture": "full"}]
    counts = publish.version_counts(rows)
    assert counts["by_torchlens_version"] == {"2.35.3": 1, "2.36.0": 2}
    assert counts["validated_by_torchlens_version"] == {"2.35.3": 1, "2.36.0": 1}
    assert counts["structure_only_by_torchlens_version"] == {"2.36.0": 1}
