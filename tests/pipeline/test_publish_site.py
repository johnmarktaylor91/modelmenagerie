"""Site export additions: shared diagrams, roster index, families, derived sizes and crops, staging."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import (
    StoreLayout, publish, publish_assets, publish_index, publish_names, read_json,
)
from menagerie.pipeline.records import GiveUp, ModelRecord, RecordError, RosterRow

from test_publish import AS_OF, FIXTURE_TL, FIXTURES, _data_root, stock_store

REPO = Path(__file__).resolve().parents[2]
SVG = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="300pt" height="900pt" viewBox="0.00 0.00 300.00 900.00"
 xmlns="http://www.w3.org/2000/svg">
<g id="graph0" class="graph"><rect x="0" y="0" width="300" height="900" fill="white"/>
<rect x="20" y="20" width="260" height="120" fill="#4a7"/>
<rect x="20" y="700" width="260" height="120" fill="#a47"/></g>
</svg>
"""


@pytest.fixture(autouse=True)
def _fixture_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(publish, "torchlens_pin", lambda: FIXTURE_TL)
    # Packed blobs unpack into a cache; keep it (and derive's libvips settings) in the test.
    monkeypatch.setenv("MENAGERIE_UNPACK_CACHE", str(tmp_path / "unpack"))
    monkeypatch.setenv("VIPS_CONCURRENCY", "1")
    monkeypatch.setenv("MALLOC_ARENA_MAX", "2")
    monkeypatch.delenv("MENAGERIE_STORE", raising=False)


# --- Shared diagrams, roster index, families ---------------------------------------------------


def _record(sid: str, svg: str, *, slug: str | None = None,
            family: str = "Fam") -> dict[str, Any]:
    variant = {"key": "unrolled-none", "svg": {"sha256": svg}}
    return {"stable_id": sid, "slug": slug or sid, "default_variant": "unrolled-none",
            "variants": [variant], "family_normalized": family}


def test_a_summary_shown_on_several_pages_is_counted() -> None:
    records: list[dict[str, Any]] = [{"summary_markdown": "A family."}, {"summary_markdown": None},
                                     {"summary_markdown": "A family."}, {"summary_markdown": "B."}]
    publish_index.mark_shared_summaries(records)
    assert [r["summary_shared_with"] for r in records] == [1, 0, 1, 0]


def test_own_diagram_is_set_and_shared_svgs_are_counted() -> None:
    records = [_record("m1", "a" * 64), _record("m2", "a" * 64), _record("m3", "a" * 64),
               _record("m4", "b" * 64)]
    publish_index.mark_diagrams(records)
    assert [r["own_diagram"] for r in records] == [True] * 4
    assert [r["diagram_shared_with"] for r in records] == [2, 2, 2, 0]


def _model(**changes: Any) -> ModelRecord:
    return replace(read_json(ModelRecord, FIXTURES / "m3631.model.json"), **changes)


def _row(**changes: Any) -> RosterRow:
    base = RosterRow(name="Some Net", zoo="discovered-pytorch", stable_id="m3631")
    return replace(base, **changes)


def _give_up(cls: str) -> GiveUp:
    return GiveUp(cls=cls, reason="internal words: oversized, research_gap",
                  at="2026-10-20T00:00:00Z")


@pytest.mark.parametrize(("cls", "status", "reason"), [
    ("not_found", "source_unavailable", "No public source code was found"),
    ("inaccessible", "source_unavailable", "The source code could not be reached"),
    ("insufficient_description", "paper_only", "Described in a paper with no public code"),
    ("oversized:beyond_allocation", "not_yet_traced",
     "Too large to trace on the machines available"),
    ("needs_gpu", "not_yet_traced", "Needs a GPU to trace"),
    ("standin:dense", "not_yet_traced", "No faithful build of the original code yet"),
    ("research_gap:hf-config", "not_yet_traced", None),
])
def test_listed_status_comes_from_the_typed_give_up_class(cls: str, status: str,
                                                          reason: str | None) -> None:
    got = publish_index.listed_status(_model(recipe=None, give_up=_give_up(cls)), _row())
    assert got == (status, reason)
    # The phrase never carries the raw class or another internal word.
    assert reason is None or not any(w in reason for w in ("_", ":", "unknown", "pending"))


def test_rows_without_a_record_are_not_yet_traced_unless_paper_only() -> None:
    assert publish_index.listed_status(None, _row()) == ("not_yet_traced", None)
    with_code = _row(triage_class="REIMPLEMENT", hint_urls=("https://github.com/a/b",))
    assert publish_index.listed_status(None, with_code)[0] == "not_yet_traced"
    paper = _row(triage_class="REIMPLEMENT", hint_paper="arxiv:2309.01782")
    assert publish_index.listed_status(None, paper) == (
        "paper_only", publish_index.ROSTER_PAPER_ONLY)
    name_only = _row(triage_class="REIMPLEMENT")
    assert publish_index.listed_status(None, name_only) == (
        "paper_only", publish_index.ROSTER_NAME_ONLY)
    # A record with a recipe is waiting for its trace, whatever the June triage said.
    assert publish_index.listed_status(_model(), paper) == ("not_yet_traced", None)


def test_roster_index_lists_unexported_active_rows_with_public_links() -> None:
    rows = [
        _row(stable_id="m1", name="Exported"),
        _row(stable_id="m2", name="Paper Net", triage_class="REIMPLEMENT", era="2019",
             source_url="arXiv:1901.00001", hint_urls=("arXiv:1901.00001",)),
        _row(stable_id="m3", name="Coded", source_url="https://github.com/x/y", family="coded"),
        _row(stable_id="m4", name="Dropped", state="not_architecture"),
        _row(stable_id="m5", name="Not a model"),
        _row(stable_id="m2", name="Paper Net again"),
    ]
    models = {"m5": _model(id="m5", recipe=None, give_up=_give_up("not_architecture"))}
    listed = publish_index.roster_index(
        rows, models=models, exported={"m1"}, packets={}, source_url=publish._source_url,
        paper_url=publish._paper_url,
        year=lambda packet, row: publish._era_fields(packet, row)["year"])
    assert [e["stable_id"] for e in listed] == ["m2", "m3"]
    paper, coded = listed
    assert paper == {
        "stable_id": "m2", "display_name": "Paper Net", "family_normalized": None,
        "zoo": "discovered-pytorch", "source_url": None,
        "paper_url": "https://arxiv.org/abs/1901.00001", "year": 2019,
        "listed_status": "paper_only", "listed_reason": publish_index.ROSTER_PAPER_ONLY,
    }
    assert coded["source_url"] == "https://github.com/x/y" and coded["paper_url"] is None
    assert coded["listed_status"] == "not_yet_traced" and coded["family_normalized"] == "coded"


def test_export_writes_roster_index_families_and_counts(tmp_path: Path) -> None:
    data = _data_root(tmp_path)
    out = publish.export_site(data=data, out_dir=tmp_path / "export", ids=["m8839"], as_of=AS_OF)
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["own_diagram"] is True and record["diagram_shared_with"] == 0
    (listed,) = [json.loads(line) for line in (out / "roster-index.jsonl").read_text().splitlines()]
    # beit has a recipe but no current trace: listed as a text entry, from its packet.
    assert listed["stable_id"] == "m3631" and listed["listed_status"] == "not_yet_traced"
    assert listed["display_name"] == "beit_base_patch16_224" and listed["year"] == 2021
    (family,) = [json.loads(line) for line in (out / "families.jsonl").read_text().splitlines()]
    assert family["family"] == "ResNet" and family["member_slugs"] == ["resnet18"]
    assert family["paper_title"] == "Deep Residual Learning for Image Recognition"
    assert family["paper_url"] == "https://arxiv.org/abs/1512.03385"
    counts = json.loads((out / "export_manifest.json").read_text())["counts"]
    assert (counts["roster_index"], counts["families"]) == (1, 1)
    assert counts["by_torchlens_version"] == {FIXTURE_TL: 1}
    with (out / "assets.tsv").open() as handle:
        kinds = {r["kind"] for r in csv.DictReader(handle, delimiter="\t")}
    assert kinds == {"svg", "pdf", "html", "netron", "model_explorer"}


def test_an_accepted_earlier_pin_stands_in_when_the_pin_has_no_deciding_trace(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data_root(tmp_path)
    monkeypatch.setattr(publish, "torchlens_pin", lambda: "2.36.0")
    with pytest.raises(RecordError, match="m8839.*no current trace"):
        publish.export_site(data=data, out_dir=tmp_path / "x", ids=["m8839"], as_of=AS_OF)
    out = publish.export_site(data=data, out_dir=tmp_path / "accepted", ids=["m8839"],
                              as_of=AS_OF, accept_pins=["2.35.3", FIXTURE_TL])
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["torchlens_version"] == FIXTURE_TL
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["torchlens_version"] == "2.36.0"
    assert manifest["counts"]["by_torchlens_version"] == {FIXTURE_TL: 1}


def test_an_export_read_at_an_earlier_pin_keeps_that_pins_trace_and_version(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data_root(tmp_path)
    monkeypatch.setattr(publish, "torchlens_pin", lambda: "3.0.2")  # the pin moved; no row there yet
    with pytest.raises(RecordError, match="m8839.*no current trace"):
        publish.export_site(data=data, out_dir=tmp_path / "x", ids=["m8839"], as_of=AS_OF)
    out = publish.export_site(data=data, out_dir=tmp_path / "viewed", ids=["m8839"], as_of=AS_OF,
                              pin=FIXTURE_TL)
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["torchlens_version"] == FIXTURE_TL
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["torchlens_version"] == FIXTURE_TL
    assert manifest["counts"]["by_torchlens_version"] == {FIXTURE_TL: 1}


def test_each_record_is_counted_once_in_the_manifest() -> None:
    base = {"variants": [], "monster_graph": False, "hashes": None, "graph_shape_hash": ""}
    records = [{**base, "forward_validated": True, "capture": "structure_only"},
               {**base, "forward_validated": True, "capture": "full"},
               {**base, "forward_validated": False, "capture": "full"},
               {**base, "forward_validated": False, "capture": "structure_only"}]
    _, manifest = publish._counts(records, roster_names=4, as_of=AS_OF, tl="x", renderer="y")
    counts = manifest["counts"]
    # A structure-only record is never also counted as validated: 1 + 2 + 1 records.
    assert (counts["validated"], counts["structure_only"], counts["preview_unvalidated"]) == (
        1, 2, 1)


def _one_record(out: Path) -> dict[str, Any]:
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    return record


def test_an_export_without_citation_shards_is_refused_unless_asked(tmp_path: Path) -> None:
    data = _data_root(tmp_path)
    for shard in (data.root / "cite").glob("*.jsonl"):
        shard.unlink()
    with pytest.raises(RecordError, match="cite.*holds no citation shard"):
        publish.export_site(data=data, out_dir=tmp_path / "x", ids=["m8839"], as_of=AS_OF)
    out = publish.export_site(data=data, out_dir=tmp_path / "bare", ids=["m8839"], as_of=AS_OF,
                              no_citations=True)
    assert _one_record(out)["citations"] is None
    assert json.loads((out / "export_manifest.json").read_text())["counts"]["with_citations"] == 0


def test_the_manifest_counts_records_with_citations(tmp_path: Path) -> None:
    out = publish.export_site(data=_data_root(tmp_path), out_dir=tmp_path / "export",
                              ids=["m8839"], as_of=AS_OF)
    assert json.loads((out / "export_manifest.json").read_text())["counts"]["with_citations"] == 1


def test_a_structure_only_record_is_never_flagged_validated(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data_root(tmp_path)
    real = publish._current

    def structure_only(*args: Any, **kwargs: Any) -> Any:
        found = real(*args, **kwargs)
        if isinstance(found, str):
            return found
        trace = replace(found.trace, capture="structure_only", validation_state="structure_only",
                        params_source="geometry")
        return replace(found, trace=trace, state="publishable")

    monkeypatch.setattr(publish, "_current", structure_only)
    out = publish.export_site(data=data, out_dir=tmp_path / "export", ids=["m8839"], as_of=AS_OF)
    record = _one_record(out)
    assert record["capture"] == "structure_only" and record["forward_validated"] is False
    counts = json.loads((out / "export_manifest.json").read_text())["counts"]
    funnel = json.loads((out / "funnel.json").read_text())
    assert (counts["validated"], counts["structure_only"], counts["preview_unvalidated"]) == (
        0, 1, 0)
    assert funnel["validated_records"] == 0


@pytest.mark.parametrize("raw, clean", [
    ("IEEE Conf/Proceedings - IEEE Conf/Proceedings", "IEEE Conf/Proceedings"),
    ("NeurIPS 2020NeurIPS 2020", "NeurIPS 2020"),
    ("Proceedings of the IEEE", "Proceedings of the IEEE"),
    ("ICML", "ICML"),
    (None, None),
])
def test_a_venue_written_twice_is_collapsed(raw: str | None, clean: str | None) -> None:
    assert publish_names.collapse_repeat(raw) == clean


@pytest.mark.parametrize("raw, clean", [
    ("mmagic (web; mmcv ABI-incompat torch2.8 -> source-catalog)", "mmagic"),
    ("act(repo,tonyzhaozh)", "act"),
    ("torch.nn (hand-build)", "torch.nn"),
    ("timm 1.0.30", "timm 1.0.30"),
    (None, None),
])
def test_a_zoo_note_is_dropped(raw: str | None, clean: str | None) -> None:
    assert publish_names.bare_zoo(raw) == clean


def test_venue_and_zoo_are_cleaned_in_the_export(tmp_path: Path) -> None:
    data = _data_root(tmp_path)
    packet_path = data.packet("f-resnet")
    packet = json.loads(packet_path.read_text())
    packet["citation"]["venue"] = "CVPR/Proceedings - CVPR/Proceedings"
    packet_path.write_text(json.dumps(packet))
    model_path = data.model("m8839")
    model = json.loads(model_path.read_text())
    model["zoo"] = "timm (web; note)"
    model_path.write_text(json.dumps(model))
    record = _one_record(publish.export_site(data=data, out_dir=tmp_path / "export",
                                             ids=["m8839"], as_of=AS_OF))
    assert record["venue"] == "CVPR/Proceedings"
    assert record["zoo"] == "timm" and "(" not in record["source_zoo"]
    assert "CVPR/Proceedings - CVPR" not in record["bibtex"]


def test_summary_basis_and_checked_come_from_the_describe_record(tmp_path: Path) -> None:
    data = _data_root(tmp_path)
    record = _one_record(publish.export_site(data=data, out_dir=tmp_path / "grounded",
                                             ids=["m8839"], as_of=AS_OF))
    assert record["summary_checked"] is True and record["summary_basis"] is None
    assert record["member_note"] == "The 18-layer configuration." and record["summary_shared_with"] == 0
    describe_path = data.describe("f-resnet")
    describe = json.loads(describe_path.read_text())
    describe.update(basis="abstract", grounding_state="unchecked")
    describe_path.write_text(json.dumps(describe))
    record = _one_record(publish.export_site(data=data, out_dir=tmp_path / "unchecked",
                                             ids=["m8839"], as_of=AS_OF))
    assert record["summary_checked"] is False and record["summary_basis"] == "abstract"


def test_curated_display_names_with_raw_fallback(tmp_path: Path) -> None:
    assert publish_names.model_name("gpt2-huggingface", "gpt2") == "GPT-2"
    assert publish_names.model_name("some-model", "some_model") == "some_model"
    assert publish_names.family_name("deeplabv3") == "DeepLabV3"
    assert publish_names.family_name("vit") == "ViT"
    assert publish_names.family_name("odd_family") == "odd_family"
    out = publish.export_site(data=_data_root(tmp_path), out_dir=tmp_path / "export",
                              ids=["m8839"], as_of=AS_OF)
    record = _one_record(out)
    assert record["slug"] == "resnet18" and record["display_name"] == "ResNet-18"
    (family,) = [json.loads(line) for line in (out / "families.jsonl").read_text().splitlines()]
    assert family["family"] == "ResNet" and family["display_name"] == "ResNet"


# --- Derived sizes and crops ------------------------------------------------------------------


def test_svg_sizes_in_css_px() -> None:
    assert publish_assets.svg_size_of_text(SVG) == (400.0, 1200.0, 300.0)
    px = '<svg xmlns="http://www.w3.org/2000/svg" width="640px" height="480">'
    assert publish_assets.svg_size_of_text(px) == (640.0, 480.0, 640.0)
    box = '<svg viewBox="0 0 120 60" xmlns="http://www.w3.org/2000/svg">'
    assert publish_assets.svg_size_of_text(box) == (120.0, 60.0, 120.0)
    inches = '<svg width="2in" height="1in">'
    assert publish_assets.svg_size_of_text(inches) == (192.0, 96.0, 144.0)
    with pytest.raises(RecordError):
        publish_assets.svg_size_of_text('<svg xmlns="x">')


def test_graphviz_svg_size_uses_the_renderers_points(tmp_path: Path) -> None:
    svg = tmp_path / "g.svg"
    svg.write_text(SVG.replace('width="300pt" height="900pt"', 'width="123.5pt" height="45pt"'))
    width, height, raster = publish_assets.svg_size(svg)
    assert (round(width, 3), height, raster) == (164.667, 60.0, 123.5)


def test_webp_header_sizes(tmp_path: Path) -> None:
    lossless = tmp_path / "l.webp"
    bits = (800 - 1) | ((600 - 1) << 14)
    body = b"VP8L" + (5).to_bytes(4, "little") + b"\x2f" + bits.to_bytes(4, "little") + b"\0" * 5
    lossless.write_bytes(b"RIFF" + (len(body) + 4).to_bytes(4, "little") + b"WEBP" + body)
    assert publish_assets.webp_size(lossless) == (800, 600)
    lossy = tmp_path / "v.webp"
    frame = b"\0\0\0" + b"\x9d\x01\x2a" + (400).to_bytes(2, "little") + (37).to_bytes(2, "little")
    body = b"VP8 " + len(frame).to_bytes(4, "little") + frame
    lossy.write_bytes(b"RIFF" + (len(body) + 4).to_bytes(4, "little") + b"WEBP" + body)
    assert publish_assets.webp_size(lossy) == (400, 37)
    (tmp_path / "x.webp").write_bytes(b"not a webp at all, not at all, no")
    assert publish_assets.webp_size(tmp_path / "x.webp") is None


def _put(store: StoreLayout, data: bytes, *, gzipped: bool = False) -> str:
    sha = hashlib.sha256(data).hexdigest()
    path = store.blob(sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzipped:
        path.with_name(path.name + ".gz").write_bytes(gzip.compress(data, mtime=0))
    else:
        path.write_bytes(data)
    return sha


def _tiny_export(tmp_path: Path) -> tuple[Path, StoreLayout, str, str]:
    store = StoreLayout(tmp_path / "store")
    svg = _put(store, SVG.encode(), gzipped=True)  # a packed view, read through the store
    pdf = _put(store, b"%PDF-1.4 tiny")
    variant = {"key": "unrolled-none", "vis_mode": "unrolled", "collapse": "none",
               "label": "Unrolled, full", "mode": "eval", "monster": False,
               "svg": publish.asset(svg, "svg", len(SVG.encode())),
               "thumb": publish.asset(svg, "svg", len(SVG.encode())),
               "pdf": publish.asset(pdf, "pdf", len(b"%PDF-1.4 tiny")), "tiles": None}
    record = {"stable_id": "m1", "slug": "tiny", "default_variant": "unrolled-none",
              "variants": [variant], "interactive": None, "netron": None,
              "model_explorer": None, "tlspec": None}
    export = tmp_path / "export"
    export.mkdir()
    publish._write_jsonl(export / "models.jsonl", [record])
    publish.write_assets(export / "assets.tsv", [record])
    (export / "export_manifest.json").write_text(json.dumps({"counts": {"models": 1}}))
    return export, store, svg, pdf


def test_derive_rewrites_the_export_with_sizes_and_is_incremental(tmp_path: Path) -> None:
    export, store, svg, _ = _tiny_export(tmp_path)
    report = publish_assets.derive(export, store=store, workers=1, memory_gb=2)
    (record,) = [json.loads(line) for line in (export / "models.jsonl").read_text().splitlines()]
    (variant,) = record["variants"]
    assert (variant["svg_width_px"], variant["svg_height_px"]) == (400, 1200)
    assert report["svgs"] == 1 and report["worked"] == 1 and report["missing_svgs"] == 0
    assert (export / "derived" / "by-svg" / f"{svg}.json").is_file()
    counts = json.loads((export / "export_manifest.json").read_text())["counts"]
    assert counts["svg_dimensions"] == 1 and counts["models"] == 1
    if not publish_assets._have_pyvips():
        assert variant["thumb_crop"] is None and record["share_image"] is None
        assert report["crop_skipped_by_reason"] == {"no-libvips": 1}
        return
    crop = variant["thumb_crop"]
    assert (crop["width"], crop["height"]) == (800, 600)
    assert crop["url"] == f"/export/assets/{crop['sha256']}.webp"
    assert (export / "derived" / f"{crop['sha256']}.webp").stat().st_size == crop["bytes"]
    assert publish_assets.webp_size(export / "derived" / f"{crop['sha256']}.webp") == (800, 600)
    share = record["share_image"]
    assert (share["width"], share["height"]) == (1200, 630) and share["url"].endswith(".jpg")
    with (export / "assets.tsv").open() as handle:
        kinds = {r["kind"] for r in csv.DictReader(handle, delimiter="\t")}
    assert kinds == {"svg", "pdf", "crop", "share"}
    again = publish_assets.derive(export, store=store, workers=1, memory_gb=2)
    assert (again["cached"], again["worked"]) == (1, 0)


def test_a_short_graph_gives_a_shorter_crop(tmp_path: Path) -> None:
    pytest.importorskip("pyvips")
    export, store, _, _ = _tiny_export(tmp_path)
    wide = SVG.replace('width="300pt" height="900pt"', 'width="1600pt" height="400pt"')
    sha = _put(store, wide.encode())
    out = tmp_path / "derived"
    out.mkdir()
    entry = publish_assets.derive_one({"svg": sha, "thumb": None, "crop": True, "share": False},
                                      str(store.root), str(out))
    assert (entry["crop"]["width"], entry["crop"]["height"]) == (800, 200)


def _px_svg(width: int, height: int) -> str:
    """A plain SVG in CSS px with a label at its top-left corner."""

    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px">'
            f'<rect width="{width}" height="{height}" fill="white"/>'
            f'<rect x="4" y="4" width="200" height="{max(1, height - 8)}" fill="#4a7"/></svg>')


@pytest.mark.parametrize(("width", "height", "rule"), [
    (20_000, 40, "corner"),  # fitted to 800 px it would be under 2 px tall
    (1200, 900, "fit"),
    (6000, 1125, "fit"),  # fitted height exactly 150 px: the threshold itself fits
    (1000, 100, "fit"),  # under 150 px fitted, but no wider than the readable width: fit is larger
])
def test_crop_rule_by_fitted_height(width: int, height: int, rule: str) -> None:
    got, scale = publish_assets.crop_plan(width, height, width)
    assert got == rule
    readable = publish_assets.CROP_W / publish_assets.READABLE_W
    assert scale == pytest.approx(readable if rule == "corner" else 800 / width)


def test_corner_scale_follows_graphviz_points() -> None:
    # A Graphviz SVG draws one pixel per point at scale 1; its CSS width is 4/3 of that.
    rule, scale = publish_assets.crop_plan(24_000 * 4 / 3, 40 * 4 / 3, 24_000)
    assert rule == "corner" and 24_000 * scale == pytest.approx(24_000 * 4 / 3 * 800 / 1200)


@pytest.mark.parametrize(("width", "height", "rule", "size"), [
    (20_000, 40, "corner", (800, 27)),
    (1200, 900, "fit", (800, 600)),
])
def test_a_very_wide_graph_gets_a_corner_crop(tmp_path: Path, width: int, height: int,
                                              rule: str, size: tuple[int, int]) -> None:
    pytest.importorskip("pyvips")
    store = StoreLayout(tmp_path / "store")
    sha = _put(store, _px_svg(width, height).encode())
    out = tmp_path / "derived"
    out.mkdir()
    entry = publish_assets.derive_one({"svg": sha, "thumb": None, "crop": True, "share": False},
                                      str(store.root), str(out))
    crop = entry["crop"]
    assert crop["rule"] == rule
    assert crop["width"] == size[0] and abs(crop["height"] - size[1]) <= 1
    assert publish_assets.webp_size(out / f"{crop['sha256']}.webp") == (crop["width"],
                                                                         crop["height"])


def _cached_entry(export: Path, svg: str, **fields: Any) -> dict[str, Any]:
    """Write derivative files and a cache entry for ``svg``; return the entry."""

    out = export / "derived"
    (out / "by-svg").mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {"rev": publish_assets.DERIVE_REV, "svg": svg,
                             "svg_width_px": 26_667, "svg_height_px": 53}
    for key, (name, ext, width, height) in fields.items():
        data = name.encode() * 50
        sha = hashlib.sha256(data).hexdigest()
        (out / f"{sha}{ext}").write_bytes(data)
        entry[key] = {"sha256": sha, "bytes": len(data), "width": width, "height": height}
    return entry


def test_the_crop_rule_round_trips_through_the_export(tmp_path: Path) -> None:
    export, store, svg, _ = _tiny_export(tmp_path)
    entry = _cached_entry(export, svg, crop=("crop", ".webp", 800, 35),
                          share=("share", ".jpg", 1200, 630))
    entry["crop"]["rule"] = "corner"
    (export / "derived" / "by-svg" / f"{svg}.json").write_text(json.dumps(entry))
    report = publish_assets.derive(export, store=store, workers=1, memory_gb=2)
    assert (report["cached"], report["worked"]) == (1, 0)
    assert report["crops_by_rule"] == {"corner": 1}
    (record,) = [json.loads(line) for line in (export / "models.jsonl").read_text().splitlines()]
    crop = record["variants"][0]["thumb_crop"]
    assert (crop["rule"], crop["width"], crop["height"]) == ("corner", 800, 35)
    counts = json.loads((export / "export_manifest.json").read_text())["counts"]
    assert counts["default_thumb_crops_by_rule"] == {"corner": 1}


def test_a_new_revision_redoes_the_crop_and_keeps_the_share_image(tmp_path: Path) -> None:
    export, store, svg, _ = _tiny_export(tmp_path)
    entry = _cached_entry(export, svg, crop=("old crop", ".webp", 800, 600),
                          share=("share", ".jpg", 1200, 630))
    entry["rev"] = "derive/1 crop=800x600 webp q78 share=1200x630 jpeg q85"
    (export / "derived" / "by-svg" / f"{svg}.json").write_text(json.dumps(entry))
    assert publish_assets.SHARE_SPEC in entry["rev"] and entry["rev"] != publish_assets.DERIVE_REV
    report = publish_assets.derive(export, store=store, workers=1, memory_gb=2)
    assert report["worked"] == 1
    (record,) = [json.loads(line) for line in (export / "models.jsonl").read_text().splitlines()]
    assert record["share_image"]["sha256"] == entry["share"]["sha256"]
    crop = record["variants"][0]["thumb_crop"]
    if publish_assets._have_pyvips():
        assert crop["sha256"] != entry["crop"]["sha256"] and crop["rule"] == "fit"
    else:
        assert crop is None


# --- Blobs the store does not hold ------------------------------------------------------------


def _blob_record(sid: str, default_svg: str, pdf: str) -> dict[str, Any]:
    variant = {"key": "unrolled-none", "svg": {"sha256": default_svg},
               "thumb": {"sha256": default_svg}, "pdf": {"sha256": pdf}}
    other = {"key": "rolled-none", "svg": {"sha256": "o" + sid}, "thumb": {"sha256": "o" + sid},
             "pdf": None}
    return {"stable_id": sid, "slug": sid, "default_variant": "unrolled-none",
            "variants": [variant, other]}


def test_a_missing_default_svg_drops_the_record_and_a_missing_pdf_keeps_it() -> None:
    records = [_blob_record("a", "svg-a", "pdf-a"), _blob_record("b", "svg-b", "pdf-b"),
               _blob_record("c", "svg-c", "pdf-c")]
    gone = {"svg-a", "pdf-a", "pdf-b"}
    kept, rows = publish_index.drop_missing_blobs(records, lambda sha: sha not in gone)
    assert [r["stable_id"] for r in kept] == ["b", "c"]
    assert [(r["stable_id"], r["kind"], r["action"]) for r in rows] == [
        ("a", "svg", "dropped_record"), ("a", "thumb", "dropped_record"),
        ("a", "pdf", "dropped_record"), ("b", "pdf", "kept")]


def _export_against(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
                    ) -> tuple[Path, list[dict[str, str]]]:
    """Export m8839 against a store missing one ``kind`` blob: the default variant's, or for a
    kind the default lacks (m8839's PDF is on its full view only), the first variant's that has it."""

    data = _data_root(tmp_path)
    plain = publish.export_site(data=data, out_dir=tmp_path / "plain", ids=["m8839"], as_of=AS_OF)
    record = _one_record(plain)
    variants = sorted(record["variants"], key=lambda v: v["key"] != record["default_variant"])
    blob = next(v[kind] for v in variants if v.get(kind))
    store = tmp_path / "store"
    stock_store(store, plain, leave_out={blob["sha256"]})
    monkeypatch.setenv("MENAGERIE_STORE", str(store))
    out = publish.export_site(data=data, out_dir=tmp_path / "export", ids=["m8839"], as_of=AS_OF)
    with (out / publish.MISSING_BLOBS_FILE).open() as handle:
        return out, list(csv.DictReader(handle, delimiter="\t"))


def test_an_export_drops_a_record_whose_default_svg_is_missing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out, rows = _export_against(tmp_path, monkeypatch, "svg")
    assert (out / "models.jsonl").read_text() == ""
    counts = json.loads((out / "export_manifest.json").read_text())["counts"]
    assert (counts["models"], counts["dropped_missing_blob"]) == (0, 1)
    assert {(r["stable_id"], r["action"]) for r in rows} == {("m8839", "dropped_record")}
    assert "svg" in {r["kind"] for r in rows}
    listed = [json.loads(line)["stable_id"]
              for line in (out / "roster-index.jsonl").read_text().splitlines()]
    assert "m8839" in listed  # still on the roster, as a text entry


def test_an_export_keeps_a_record_whose_only_missing_blob_is_a_pdf(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out, rows = _export_against(tmp_path, monkeypatch, "pdf")
    assert _one_record(out)["stable_id"] == "m8839"
    counts = json.loads((out / "export_manifest.json").read_text())["counts"]
    assert counts["dropped_missing_blob"] == 0
    assert [(r["stable_id"], r["kind"], r["action"]) for r in rows] == [("m8839", "pdf", "kept")]


def test_an_export_without_a_store_checks_no_blobs(tmp_path: Path) -> None:
    out = publish.export_site(data=_data_root(tmp_path), out_dir=tmp_path / "export",
                              ids=["m8839"], as_of=AS_OF)
    assert json.loads((out / "export_manifest.json").read_text())["counts"][
        "dropped_missing_blob"] is None
    assert not (out / publish.MISSING_BLOBS_FILE).exists()


# --- Staging --------------------------------------------------------------------------------


def test_stage_copies_the_named_kinds_and_writes_the_manifest(tmp_path: Path) -> None:
    export, store, svg, pdf = _tiny_export(tmp_path)
    derived = export / "derived"
    derived.mkdir()
    crop = b"RIFF fake crop bytes"
    crop_sha = hashlib.sha256(crop).hexdigest()
    (derived / f"{crop_sha}.webp").write_bytes(crop)
    records = [json.loads(line) for line in (export / "models.jsonl").read_text().splitlines()]
    records[0]["variants"][0]["thumb_crop"] = {**publish.asset(crop_sha, "crop", len(crop)),
                                               "width": 800, "height": 600}
    gone = "c" * 64
    records[0]["interactive"] = publish.asset(gone, "html", 10)
    publish.write_assets(export / "assets.tsv", records)

    dest = tmp_path / "assets"
    report = publish_assets.stage(export, dest, store=store, kinds="svg,crop,pdf")
    assert report["missing"] == [] and report["objects"] == 3
    assert (dest / f"{svg}.svg").read_text() == SVG  # gunzipped from the store's packed form
    with (dest / "MANIFEST.tsv").open() as handle:
        rows = {r["key"]: r for r in csv.DictReader(handle, delimiter="\t")}
    assert rows[f"{svg}.svg"]["content_type"] == "image/svg+xml"
    assert rows[f"{crop_sha}.webp"]["content_type"] == "image/webp"
    assert rows[f"{pdf}.pdf"] == {"key": f"{pdf}.pdf", "sha256": pdf,
                                  "bytes": str(len(b"%PDF-1.4 tiny")),
                                  "content_type": "application/pdf"}
    assert report["by_kind"]["svg"] == {"objects": 1, "files": 1, "bytes": len(SVG.encode())}

    # html is not a default kind; `all` adds it, and its missing blob is reported, not fatal.
    report = publish_assets.stage(export, dest, store=store, kinds="all")
    assert report["missing"] == [f"{gone}.html"] and report["objects"] == 3
    assert "not in the store" in (dest / "MISSING.tsv").read_text()


def test_a_day_folder_seeded_by_hard_links_never_writes_through_to_the_seed(
        tmp_path: Path) -> None:
    export, store, _, _ = _tiny_export(tmp_path)
    seed = tmp_path / "seed"
    (seed / "assets").mkdir(parents=True)
    (seed / "derived").mkdir()
    for name in ("assets/MANIFEST.tsv", "assets/MISSING.tsv", "derived/derive-report.json"):
        (seed / name).write_text("seed\n")
    day = tmp_path / "day"
    for sub in ("assets", "derived"):
        (day / sub).mkdir(parents=True)
        for f in (seed / sub).iterdir():
            (day / sub / f.name).hardlink_to(f)
    publish_assets.derive(export, store=store, derived=day / "derived", workers=1, memory_gb=2)
    publish_assets.stage(export, day / "assets", store=store, derived=day / "derived",
                         kinds=["svg"])
    for name in ("assets/MANIFEST.tsv", "assets/MISSING.tsv", "derived/derive-report.json"):
        assert (seed / name).read_text() == "seed\n"
        assert (day / name).read_text() != "seed\n"


def test_stage_rejects_an_unknown_kind(tmp_path: Path) -> None:
    export, store, _, _ = _tiny_export(tmp_path)
    with pytest.raises(RecordError, match="unknown asset kinds"):
        publish_assets.stage(export, tmp_path / "d", store=store, kinds="svg,gif")


def test_sync_site_data_copies_the_side_lists_when_present(tmp_path: Path,
                                                            capsys: pytest.CaptureFixture) -> None:
    spec = importlib.util.spec_from_file_location(
        "sync_site_data", REPO / "site" / "scripts" / "sync_site_data.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    export = tmp_path / "export"
    export.mkdir()
    for name in sync.DATA_FILES:
        (export / name).write_text("{}\n")
    copied = sync.sync_site_data(export, tmp_path / "site-data")
    assert len(copied) == len(sync.DATA_FILES)
    assert "roster-index.jsonl is not in the export" in capsys.readouterr().err
    for name in sync.OPTIONAL_FILES:
        (export / name).write_text("{}\n")
    copied = sync.sync_site_data(export, tmp_path / "site-data")
    assert {p.name for p in copied} == {*sync.DATA_FILES, *sync.OPTIONAL_FILES}
