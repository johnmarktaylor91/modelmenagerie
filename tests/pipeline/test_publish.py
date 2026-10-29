"""Site export: real fixture records export with variants, viewer refs and dated counts."""

from __future__ import annotations

import csv
import functools
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import (
    DataLayout,
    append_jsonl,
    month_of,
    publish,
    read_json,
    read_jsonl,
    render,
    standin,
    write_json,
)
from menagerie.pipeline.records import (
    GiveUp,
    LedgerRow,
    ModelRecord,
    Packet,
    RecordError,
    RenderRow,
    RosterRow,
    parse_ledger_line,
)

FIXTURES = Path(__file__).parent / "fixtures"
AS_OF = "2026-10-04"
SVG_NONE = "111336b3e474e0651fe34812e4724423135857b879b61397e207d37298265302"
FIXTURE_TL = "2.35.0"
JUNE_GRAPH_HASH = "909da2bb8dc29f34a8120d701c9c91eb0b0fe6a4cee6d3c49bb712a11ea03d88"


@pytest.fixture(autouse=True)
def _fixture_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fixture rows were captured at TorchLens 2.35.0; export them as current at that pin.
    monkeypatch.setattr(publish, "torchlens_pin", lambda: FIXTURE_TL)


def _data_root(tmp_path: Path) -> DataLayout:
    """Lay the step zero fixtures out as a committed data tree."""

    data = DataLayout(tmp_path / "data")
    for model in ("m8839", "m3631"):
        target = data.model(model)
        target.parent.mkdir(parents=True)
        shutil.copy(FIXTURES / f"{model}.model.json", target)
    for family, name in (("f-resnet", "f-resnet.packet.json"), ("f-beit", "f-beit.packet.json")):
        data.packet(family).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES / name, data.packet(family))
    data.describe("f-resnet").parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "f-resnet.describe.json", data.describe("f-resnet"))
    shutil.copy(FIXTURES / "roster.jsonl", data.roster)
    for row in read_jsonl(parse_ledger_line, FIXTURES / "ledger.jsonl"):
        append_jsonl([row], data.ledger(month_of(row.at)))
    for name, target in (("cite.jsonl", data.cite("2026-10")),
                         ("store_index.jsonl", data.store_index("2026-10"))):
        target.parent.mkdir()
        shutil.copy(FIXTURES / name, target)
    return data


def _rows(data: DataLayout) -> list:
    return read_jsonl(parse_ledger_line, data.ledger("2026-10"))


def test_preview_export_of_resnet18(tmp_path: Path) -> None:
    data = _data_root(tmp_path)
    out = publish.export_site(data=data, out_dir=tmp_path / "export", ids=["m8839"], as_of=AS_OF)

    assert sorted(p.name for p in out.iterdir()) == sorted(
        [*publish.DATA_FILES, "assets.tsv", "trace_metrics.jsonl", "trace_metrics.csv"])
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["slug"] == "resnet18"
    assert record["category"] == "vision_classification-backbone"
    assert record["family_normalized"] == "ResNet"
    assert record["paper_url"] == "https://arxiv.org/abs/1512.03385"
    assert record["source_url"] == "https://pypi.org/project/timm/1.0.30/"
    assert record["source_license"] == "Apache-2.0"
    assert record["paper_title"] == "Deep Residual Learning for Image Recognition"
    assert record["year"] == 2015 and record["authors"][0] == "Kaiming He"
    assert "Deep Residual Learning for Image Recognition" in record["bibtex"]
    assert record["citations"] == {"count": 4703, "as_of": AS_OF}
    assert record["recipe_text"] == "timm.create_model('resnet18', pretrained=False)"
    # The probe's trace replays only partly, so the preview carries it, unvalidated.
    assert record["forward_validated"] is False
    assert record["validation_state"] == "unverified" and record["capture"] == "full"
    assert (record["n_ops"], record["param_count"], record["torchlens_version"]) == (
        85, 11689512, FIXTURE_TL)
    assert record["hashes"]["skeleton"].startswith("8a4531bf") and record["hashes"]["schema"] == 1
    assert set(record["hashes"]) == {"skeleton", "schema"}

    assert [v["key"] for v in record["variants"]] == ["unrolled-none", "unrolled-auto"]
    # 85 ops is above the readable band, so the collapsed view is the default.
    assert record["default_variant"] == "unrolled-auto" and record["is_recurrent"] is False
    full = record["variants"][0]
    assert full["svg"] == {"url": f"/export/assets/{SVG_NONE}.svg", "sha256": SVG_NONE,
                           "bytes": 113915}
    assert full["thumb"] == full["svg"]  # the probe made no thumbnail
    assert full["pdf"]["sha256"].startswith("72b53781") and full["label"] == "Unrolled, full"
    assert record["variants"][1]["pdf"] is None
    assert record["interactive"]["url"].endswith(".html")
    assert record["interactive"]["sha256"].startswith("601f37ff")
    assert record["netron"]["sha256"].startswith("fc9fe034")
    assert record["model_explorer"]["url"].endswith(".model-explorer.json")
    assert record["tlspec"] is None  # the store index has no line for the probe's trace

    funnel = json.loads((out / "funnel.json").read_text())
    assert funnel == {
        "as_of": AS_OF, "roster_names": 2, "validated_records": 0, "structure_only_records": 0,
        "distinct_architectures": 0, "distinct_basis": "legacy graph_shape_hash",
    }
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["counts"]["preview_unvalidated"] == 1
    assert manifest["counts"]["thumbnail_fallbacks"] == 2
    assert manifest["torchlens_version"] == FIXTURE_TL
    with (out / "assets.tsv").open() as handle:
        assets = list(csv.DictReader(handle, delimiter="\t"))
    assert len(assets) == 6 and {a["key"].split(".", 1)[1] for a in assets} == {
        "svg", "pdf", "html", "netron.json", "model-explorer.json"}
    with (out / "catalog.csv").open() as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["stable_id"] == "m8839" and row["default_variant"] == "unrolled-auto"


def test_explicit_ids_without_a_current_render_are_refused(tmp_path: Path) -> None:
    data = _data_root(tmp_path)
    # beit's only trace row is Slice A's, at TorchLens 2.34.1 with no saved trace.
    with pytest.raises(RecordError, match="m3631.*no current trace"):
        publish.export_site(data=data, out_dir=tmp_path / "x", ids=["m8839", "m3631"])
    with pytest.raises(RecordError, match="m0.*no model record"):
        publish.export_site(data=data, out_dir=tmp_path / "x", ids=["m0"])


def test_default_export_is_validated_only(tmp_path: Path) -> None:
    data = _data_root(tmp_path)
    out = publish.export_site(data=data, out_dir=tmp_path / "before", as_of=AS_OF)
    assert (out / "models.jsonl").read_text() == ""

    # The same capture as a validated row with June's graph hash, redrawn at today's renderer
    # but still without a thumbnail: validated, yet pending its render, so not published.
    trace = next(r for r in _rows(data) if isinstance(r, LedgerRow) and r.id == "m8839")
    drawn = next(r for r in _rows(data) if isinstance(r, RenderRow))
    passed = replace(trace, at="2026-10-04T07:00:00Z", validation_state="passed",
                     graph_hash=JUNE_GRAPH_HASH, msg=None)
    redrawn = replace(drawn, at="2026-10-04T07:00:01Z", renderer_rev=render.renderer_rev())
    append_jsonl([passed, redrawn], data.ledger("2026-10"))
    out = publish.export_site(data=data, out_dir=tmp_path / "pending", as_of=AS_OF)
    assert (out / "models.jsonl").read_text() == ""

    # As a preview it exports, marked validated: the verdict never waits on a render.
    out = publish.export_site(data=data, out_dir=tmp_path / "preview", ids=["m8839"], as_of=AS_OF)
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["forward_validated"] is True and record["validation_state"] == "passed"
    assert record["renderer_version"] == render.renderer_rev()
    funnel = json.loads((out / "funnel.json").read_text())
    assert (funnel["validated_records"], funnel["distinct_architectures"]) == (1, 1)


def test_default_export_publishes_validated_rendered_researched_models(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # resnet18 traced and rendered end to end at TorchLens 2.35.2 by the lane's site check
    # (real rows, with WebP thumbnails); replay left it unverified.
    monkeypatch.setattr(publish, "torchlens_pin", lambda: "2.35.2")
    data = _data_root(tmp_path)
    trace, drawn = read_jsonl(parse_ledger_line, FIXTURES / "m8839.e2e.ledger.jsonl")
    append_jsonl([trace, replace(drawn, renderer_rev=render.renderer_rev())],
                 data.ledger("2026-10"))
    out = publish.export_site(data=data, out_dir=tmp_path / "unverified", as_of=AS_OF)
    assert (out / "models.jsonl").read_text() == ""

    passed = replace(trace, at="2026-10-04T07:00:00Z", validation_state="passed",
                     graph_hash=JUNE_GRAPH_HASH, msg=None)
    append_jsonl([passed], data.ledger("2026-10"))
    out = publish.export_site(data=data, out_dir=tmp_path / "validated", as_of=AS_OF)
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["forward_validated"] is True and record["torchlens_version"] == "2.35.2"
    default = next(v for v in record["variants"] if v["key"] == record["default_variant"])
    assert default["thumb"]["url"].endswith(".webp") and default["pdf"] is not None
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["counts"]["thumbnail_fallbacks"] == 0
    assert manifest["counts"]["validated"] == 1

    # Research comes before publishing (JMT 2026-10-03): no packet, no page.
    data.packet("f-resnet").unlink()
    out = publish.export_site(data=data, out_dir=tmp_path / "unresearched", as_of=AS_OF)
    assert (out / "models.jsonl").read_text() == ""


@pytest.mark.parametrize(("domain", "era", "category"), [
    ("vision/semantic-segmentation", "2021", "vision_segmentation"),
    ("NLP/LLM/text", "2023", "NLP_LLM_text"),
    ("text/llm", "2024", "NLP_LLM_text"),
    ("audio:asr", "2020", "audio_speech"),
    ("vision:classification:quantization", "2019", "vision_classification-backbone"),
    ("detection/pose", "2018", "vision_pose"),
    ("3D/geometry", "2020", "3D_geometry_NeRF_point-cloud"),
    ("time-series/dynamics", "2026-06-17", "time-series_dynamics"),
    ("rl", "2017", "RL_robotics_control"),
    ("paper/local-source reimplementation", "2022", "exotic_other"),
    ("graph/geometric", "1986", "history"),
    (None, None, "exotic_other"),
])
def test_categories_from_real_roster_domains(domain: str | None, era: str | None,
                                             category: str) -> None:
    row = RosterRow(name="x", zoo="z", domain=domain, era=era)
    assert publish.category_for(None, row) == category


def test_slugs_are_decided_over_every_record() -> None:
    resnet = read_json(ModelRecord, FIXTURES / "m8839.model.json")
    # resnet18 has three stable ids in `stable_ids.jsonl`: one name across zoos and variants.
    twins = [replace(resnet, id="m8840", zoo="torchvision.models"),
             replace(resnet, id="m8841", zoo="torchvision.models")]
    beit = read_json(ModelRecord, FIXTURES / "m3631.model.json")
    assert publish.assign_slugs([resnet, beit]) == {"m8839": "resnet18",
                                                    "m3631": "beit-base-patch16-224"}
    assert publish.assign_slugs([resnet, *twins, beit]) == {
        "m8839": "resnet18-timm", "m8840": "resnet18-torchvision-models-m8840",
        "m8841": "resnet18-torchvision-models-m8841", "m3631": "beit-base-patch16-224",
    }


def test_torchlens_pin_is_read_from_the_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    pin = publish.torchlens_pin()
    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    assert re.fullmatch(r"\d+\.\d+\.\d+", pin) and f'"torchlens=={pin}"' in pyproject


def test_a_shared_family_packet_gives_each_model_its_own_library_license() -> None:
    pub = publish
    data = DataLayout(Path(__file__).parents[2] / "menagerie" / "data")
    packet = read_json(Packet, data.packet("f-vit"))
    torchvision = read_json(ModelRecord, data.model("m10921"))
    transformers = read_json(ModelRecord, data.model("m10919"))
    assert pub._license(torchvision, packet) == "BSD-3-Clause"
    assert pub._license(transformers, packet) == "Apache-2.0"
    assert "torchvision" in (pub._source_url(torchvision, packet) or "")


def test_an_unsaved_structure_only_capture_exports_by_its_summary(tmp_path: Path) -> None:
    data = _data_root(tmp_path)
    trace, render_row = [r for r in _rows(data) if r.id == "m8839"][:2]
    summary = "5" * 64  # a giant no save level accepts: renders keyed on the summary sha
    giant = replace(trace, capture="structure_only", validation_state="structure_only",
                    params_source="geometry", trace=None, summary=summary, runnable="n/a")
    shard = data.ledger("2026-10")
    shard.unlink()
    append_jsonl([giant, replace(render_row, trace=summary)], shard)
    out = publish.export_site(data=data, out_dir=tmp_path / "export", ids=["m8839"], as_of=AS_OF)
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["capture"] == "structure_only" and record["tlspec"] is None
    assert json.loads((out / "funnel.json").read_text())["structure_only_records"] == 1


def test_a_projection_beyond_the_worker_exports_its_structure_only_capture(tmp_path: Path) -> None:
    """Stage 3 types the full trace ``oversized:projected`` without running it (fix 10); the
    structure-only capture taken at once is what the record is built from."""

    data = _data_root(tmp_path)
    trace, render_row = [r for r in _rows(data) if r.id == "m8839"][:2]
    capture = replace(trace, capture="structure_only", validation_state="structure_only",
                      params_source="geometry", runnable="n/a")
    typed = replace(trace, outcome="fail", cls="oversized:projected", trace=None, summary=None,
                    n_ops=None, graph_hash=None, hashes=None, validation_state=None,
                    runnable=None, at="2026-10-30T00:00:00Z")  # newer: it decides the full verdict
    shard = data.ledger("2026-10")
    shard.unlink()
    append_jsonl([capture, render_row, typed], shard)
    out = publish.export_site(data=data, out_dir=tmp_path / "export", ids=["m8839"], as_of=AS_OF)
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["capture"] == "structure_only" and not record["forward_validated"]


def test_a_standin_give_up_never_exports_under_the_real_name(tmp_path: Path) -> None:
    """A recipe that replaced a library with a hand-written equivalent passes its own trace;
    typed ``standin:dense``, that trace is refused by name and left out of the default export."""

    data = _data_root(tmp_path)
    trace = next(r for r in _rows(data) if isinstance(r, LedgerRow) and r.id == "m8839")
    drawn = next(r for r in _rows(data) if isinstance(r, RenderRow))
    append_jsonl([replace(trace, at="2026-10-04T07:00:00Z", validation_state="passed",
                          graph_hash=JUNE_GRAPH_HASH, msg=None),
                  replace(drawn, at="2026-10-04T07:00:01Z", renderer_rev=render.renderer_rev())],
                 data.ledger("2026-10"))
    publish.export_site(data=data, out_dir=tmp_path / "before", ids=["m8839"], as_of=AS_OF)
    model = read_json(ModelRecord, data.model("m8839"))
    give = GiveUp(cls="standin:dense", reason="dense stand-in for a sparse library",
                  at="2026-10-05T21:30:00Z")
    write_json(replace(model, give_up=give), data.model("m8839"))
    with pytest.raises(RecordError, match="m8839.*gave up: standin"):
        publish.export_site(data=data, out_dir=tmp_path / "named", ids=["m8839"], as_of=AS_OF)
    out = publish.export_site(data=data, out_dir=tmp_path / "default", as_of=AS_OF)
    assert (out / "models.jsonl").read_text() == ""

    # A later lane's give-up replaces the record's class; the typed recipe stays refused.
    typed = LedgerRow(id="m8839", stage="forward", at=give.at, host="worker", env=trace.env,
                      recipe_rev=model.recipe.rev, outcome="giveup", cls="standin:dense",
                      msg=give.reason)
    append_jsonl([typed], data.ledger("2026-10"))
    later = GiveUp(cls="budget", reason="the retry ran out of tries", at="2026-10-06T03:00:00Z")
    write_json(replace(model, give_up=later), data.model("m8839"))
    with pytest.raises(RecordError, match="m8839.*gave up: standin"):
        publish.export_site(data=data, out_dir=tmp_path / "later", ids=["m8839"], as_of=AS_OF)


def _exportable(data: DataLayout) -> None:
    """m8839's fixture trace made current, validated and drawn at today's renderer."""

    trace = next(r for r in _rows(data) if isinstance(r, LedgerRow) and r.id == "m8839")
    drawn = next(r for r in _rows(data) if isinstance(r, RenderRow))
    append_jsonl([replace(trace, at="2026-10-04T07:00:00Z", validation_state="passed",
                          graph_hash=JUNE_GRAPH_HASH, msg=None),
                  replace(drawn, at="2026-10-04T07:00:01Z", renderer_rev=render.renderer_rev())],
                 data.ledger("2026-10"))


def test_a_recipe_typed_standin_for_another_id_is_refused_and_swept(tmp_path: Path) -> None:
    """A catalog row can run another row's typed stand-in recipe under its own id (m1778 ran
    m12019's): the shared revision is refused by name, and the sweep types it once."""

    data = _data_root(tmp_path)
    _exportable(data)
    publish.export_site(data=data, out_dir=tmp_path / "before", ids=["m8839"], as_of=AS_OF)
    model = read_json(ModelRecord, data.model("m8839"))
    other = LedgerRow(id="m3631", stage="forward", at="2026-10-05T21:30:00Z", host="worker",
                      env=model.recipe.env, recipe_rev=model.recipe.rev, outcome="giveup",
                      cls="standin:dense", msg="dense stand-in for a sparse library")
    append_jsonl([other], data.ledger("2026-10"))
    with pytest.raises(RecordError, match="m8839.*gave up: standin.*typed standin:dense for m3631"):
        publish.export_site(data=data, out_dir=tmp_path / "named", ids=["m8839"], as_of=AS_OF)

    listed = standin.sweep(data=data)
    assert [(t["id"], t["class"]) for t in listed["typed"]] == [("m8839", "standin:dense")]
    assert read_json(ModelRecord, data.model("m8839")).give_up is None  # a listing writes nothing
    standin.sweep(data=data, apply=True)
    typed = read_json(ModelRecord, data.model("m8839")).give_up
    assert typed is not None and typed.cls == "standin:dense" and "m3631" in typed.reason
    row = _rows(data)[-1]
    assert (row.id, row.outcome, row.cls, row.recipe_rev) == (
        "m8839", "giveup", "standin:dense", model.recipe.rev)
    assert standin.sweep(data=data)["typed"] == []  # typed once


def test_a_recipe_that_calls_itself_a_standin_is_refused(tmp_path: Path) -> None:
    data = _data_root(tmp_path)
    _exportable(data)
    model = read_json(ModelRecord, data.model("m8839"))
    write_json(replace(model, name="ResNet-18 (dense stand-in for a sparse library)"),
               data.model("m8839"))
    with pytest.raises(RecordError, match="m8839.*gave up: standin.*its name calls it a stand-in"):
        publish.export_site(data=data, out_dir=tmp_path / "named", ids=["m8839"], as_of=AS_OF)
    # "standing in for" (a config object, a seeded input) is not a self-description.
    write_json(replace(model, name="ResNet-18 (seeded noise standing in for images)"),
               data.model("m8839"))
    publish.export_site(data=data, out_dir=tmp_path / "plain", ids=["m8839"], as_of=AS_OF)


def test_a_shared_dense_standin_adapter_is_flagged_for_a_reader_not_typed(tmp_path: Path) -> None:
    """One family adapter can build the stand-in for one member and the library's own path for
    another (f-schnet.py): sharing its text flags the other member; only a reader types it."""

    data = _data_root(tmp_path)
    adapter = data.root / "adapters" / "f-x.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("def build(cfg):\n    return None\n", encoding="utf-8")
    for sid, rev in (("m8839", "aaaaaaaaaaaa"), ("m3631", "bbbbbbbbbbbb")):
        model = read_json(ModelRecord, data.model(sid))
        recipe = replace(model.recipe, kind="adapter", adapter="adapters/f-x.py", call=None,
                         rev=rev)
        give = GiveUp(cls="standin:dense", reason="hand-written", at="2026-10-05T21:30:00Z")
        write_json(replace(model, recipe=recipe, give_up=give if sid == "m8839" else None),
                   data.model(sid))
    found = standin.sweep(data=data)
    assert [f["id"] for f in found["flagged"]] == ["m3631"] and found["typed"] == []
    assert "adapters/f-x.py" in found["flagged"][0]["why"]


def test_distinct_architectures_switch_to_the_architecture_hash_once_the_harness_passed() -> None:
    def record(graph: str, architecture: str, schema: int) -> dict[str, Any]:
        return {"forward_validated": True, "capture": "full", "graph_shape_hash": graph,
                "hashes": {"architecture": architecture, "schema": schema}, "variants": [],
                "backend": "torch",
                "monster_graph": False}

    # Two legacy hashes, one architecture at schema 2, and one row hashed at an older schema.
    records = [record("g1", "a1", 2), record("g2", "a1", 2), record("g2", "a0", 1)]
    count = functools.partial(publish._counts, records, roster_names=3, as_of=AS_OF, tl="t",
                              renderer="r")
    legacy, _ = count()
    assert (legacy["distinct_architectures"], legacy["distinct_basis"]) == (
        2, "legacy graph_shape_hash")
    switched, _ = count(basis={"basis": "architecture", "schema": 2, "backends": ["torch"]})
    assert (switched["distinct_architectures"], switched["distinct_basis"]) == (
        1, "architecture hash, schema 2")


def test_trace_statistics_come_from_the_summary_blob_in_the_store(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data_root(tmp_path)
    store = tmp_path / "store"
    monkeypatch.setenv("MENAGERIE_STORE", str(store))
    out = publish.export_site(data=data, out_dir=tmp_path / "bare", ids=["m8839"], as_of=AS_OF)
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["trace_stats"] is None and (out / "trace_metrics.jsonl").read_text() == ""
    summary = "5" * 64
    rows = [replace(r, summary=summary) if isinstance(r, LedgerRow) and r.stage == "trace"
            and r.id == "m8839" else r for r in _rows(data)]
    data.ledger("2026-10").unlink()
    append_jsonl(rows, data.ledger("2026-10"))
    blob = store / "sha256" / summary[:2] / summary
    blob.parent.mkdir(parents=True)
    blob.write_text(json.dumps({"n_params": 11689512, "graph_depth": 52,
                                "total_flops_forward": 3_600_000_000}))
    out = publish.export_site(data=data, out_dir=tmp_path / "export", ids=["m8839"], as_of=AS_OF)
    (record,) = [json.loads(line) for line in (out / "models.jsonl").read_text().splitlines()]
    assert record["trace_stats"]["graph_depth"] == 52
    assert record["year_source"] == "citation" and "era" in record
    (row,) = [json.loads(line) for line in (out / "trace_metrics.jsonl").read_text().splitlines()]
    assert (row["stable_id"], row["total_flops_forward"]) == ("m8839", 3_600_000_000)
