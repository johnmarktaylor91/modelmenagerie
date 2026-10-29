"""Lane r3-describe: ``describe write`` (layout, jev fields, the lane, the check, one return, the
typed outcomes), ``describe check`` and the ``describe drive`` scoring, with the store, the lane
launcher and jev stood in for."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    describe_jev,
    describe_text,
    describe_workers,
    describe_write,
    mini_relay,
    publish,
    review,
    tick_full,
)
from menagerie.pipeline import drive as drive_mod
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import DescribeRecord, from_dict

FIX = Path(__file__).parent / "fixtures"
REPO = Path(__file__).parents[2]
TL = "2.35.3"
TEXT_SHA, README_SHA, BIB_SHA, SUMMARY_SHA = (c * 64 for c in "abcd")
PAPER = ("Deep Residual Learning for Image Recognition. We present a residual learning framework to "
         "ease the training of networks that are substantially deeper than those used previously. "
         "We explicitly reformulate the layers as learning residual functions with reference to the "
         "layer inputs. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 "
         "layers. This work was done at the Visual Computing Group. It builds on the VGG nets and "
         "on BEiT in spirit.")
OPENALEX = {"authorships": [{"author": {"display_name": "Kaiming He"},
                             "institutions": [{"display_name": "Microsoft Research"}]},
                            {"author": {"display_name": "Jian Sun"},
                             "institutions": [{"display_name": "Microsoft Research"}]}]}
SUMMARY = {"output_shape": [1, 1000], "n_output_tensors": 1, "output_container_kind": "tensor"}


def _materialize(sha, dest, *, store):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if sha == README_SHA:
        dest.mkdir(exist_ok=True)
        (dest / "README.md").write_text("# timm\n\nPyTorch image models. resnet18 is a residual net.")
    else:
        dest.write_text({TEXT_SHA: PAPER, BIB_SHA: json.dumps(OPENALEX),
                         SUMMARY_SHA: json.dumps(SUMMARY)}[sha])
    return dest


def _data(tmp_path: Path, *, text: bool = True) -> DataLayout:
    data = DataLayout(tmp_path / "data")
    packet = json.loads((FIX / "f-resnet.packet.json").read_text())
    if text:
        packet["sources"][0]["text"] = TEXT_SHA
    packet["sources"] += [
        {"sid": "s4", "role": "code", "url": "https://github.com/huggingface/pytorch-image-models",
         "blob": README_SHA, "rev": "v1.0.30"},
        {"sid": "bib-openalex-1", "role": "bib", "api": "openalex", "blob": BIB_SHA}]
    for family, body in (("f-resnet", packet), ("f-beit", json.loads((FIX / "f-beit.packet.json").read_text()))):
        data.packet(family).parent.mkdir(parents=True, exist_ok=True)
        data.packet(family).write_text(json.dumps(body))
    model = json.loads((FIX / "m8839.model.json").read_text())
    data.model("m8839").parent.mkdir(parents=True, exist_ok=True)
    data.model("m8839").write_text(json.dumps(model))
    shutil.copyfile(FIX / "roster.jsonl", data.roster)
    row = {"schema": "ledger/1", "id": "m8839", "stage": "trace", "at": "2026-10-05T00:00:00Z",
           "host": "w", "env": "torch-main", "recipe_rev": model["recipe"]["rev"], "outcome": "pass",
           "mode": "eval", "capture": "full", "backend": "torch", "tl": TL, "n_ops": 85,
           "graph_hash": "1" * 64, "validation_state": "passed", "summary": SUMMARY_SHA}
    data.ledger("2026-10").parent.mkdir(parents=True, exist_ok=True)
    data.ledger("2026-10").write_text(json.dumps(row) + "\n")
    (data.root / "review").mkdir()
    shutil.copyfile(REPO / "menagerie" / "data" / "review" / "jev-describe-profile.json",
                    data.root / "review" / "jev-describe-profile.json")
    return data


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(store_mod, "materialize", _materialize)
    monkeypatch.setattr(publish, "torchlens_pin", lambda: TL)
    monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", "1")
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 4)
    launches, runs = [], {}

    def launch(kind, role, brief, **kw):
        run = f"run-{len(launches) + 1}"
        launches.append({"kind": kind, "role": role, "brief": Path(brief), "run": run, **kw})
        runs[run] = {"state": "RUNNING"}
        return run

    monkeypatch.setattr(drive_mod, "launch", launch)
    monkeypatch.setattr(drive_mod, "lanes", lambda **kw: [
        {"kind": lane["kind"], "run": lane["run"], "keys": lane["keys"]}
        for lane in launches if runs[lane["run"]]["state"] != "TERMINAL"])
    monkeypatch.setattr(review, "_fleet_run", lambda run: runs[run])
    return {"launches": launches, "runs": runs, "store": StoreLayout(tmp_path / "store")}


def _draft(**over):
    quote = "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously."
    s1 = "ResNet is a residual learning framework that eases the training of substantially deeper networks."
    s2 = "Its layers learn residual functions with reference to the layer inputs instead of unreferenced functions."
    s3 = "On ImageNet the authors evaluate residual nets up to 152 layers deep."
    draft = {"schema": "describe-draft/1", "family": "f-resnet", "domain": "vision/classification-backbone",
             "field": "computer vision", "subfield": "image classification", "task": "image-classification",
             "tasks": [], "modality": "image", "modalities": ["image"], "modality_output": "logits",
             "output_type": "logits", "paradigm": "CNN", "arch_class": "CNN", "arch_subclass": "backbone",
             "is_generative": False, "tags": ["residual"], "keywords": ["ResNet", "resnet18"],
             "labs": ["Visual Computing Group"],
             "lineage": [{"relation": "inspired_by", "target": "BEiT", "sid": "s1",
                          "quote": "It builds on the VGG nets and on BEiT in spirit."}],
             "members": {}, "tagline": "Residual learning for very deep image classifiers.",
             "summary_markdown": f"{s1} {s2}\n\n{s3} " * 1,
             "key_contribution": s1,
             "grounding": [{"sentence": s1, "sid": "s1", "quote": quote},
                           {"sentence": s2, "sid": "s1", "quote": "We explicitly reformulate the layers as learning residual functions with reference to the layer inputs."},
                           {"sentence": s3, "sid": "s1", "quote": "we evaluate residual nets with a depth of up to 152 layers"}]}
    draft["summary_markdown"] = f"{s1} {s2}\n\n{s3} {s1}"
    draft.update(over)
    return draft


def test_validated_families_and_due_order(tmp_path, world):
    data = _data(tmp_path)
    assert describe_text.validated_families(data=data, tl=TL) == {"f-resnet": ["m8839"]}
    assert describe_text.validated_families(data=data, tl="9.9.9") == {}
    assert describe_write.due(None, data=data, tl=TL) == ["f-resnet"]
    assert describe_write.due(["f-beit"], data=data, tl=TL) == []


def _earlier_family(data, pin):
    """f-beit's m3631 validated only at ``pin``."""

    model = {**json.loads(data.model("m8839").read_text()), "id": "m3631", "family": "f-beit"}
    data.model("m3631").parent.mkdir(parents=True, exist_ok=True)
    data.model("m3631").write_text(json.dumps(model))
    row = {**json.loads(data.ledger("2026-10").read_text()), "id": "m3631", "tl": pin}
    with data.ledger("2026-10").open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def test_earlier_pin_families_follow_the_current_pin_only_when_switched_on(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    _earlier_family(data, "2.35.2")
    assert describe_text.validated_pins(data=data, tl=TL) == {"f-resnet": TL}
    assert describe_text.validated_pins(data=data, tl=TL, earlier=True) == {
        "f-resnet": TL, "f-beit": "2.35.2"}
    assert describe_text.validated_pins(data=data, tl="2.35.1", earlier=True) == {}  # never later
    assert describe_write.due(None, data=data, tl=TL) == ["f-resnet"]
    monkeypatch.setenv(describe_write.EARLIER_PINS, "1")
    assert describe_write.due(None, data=data, tl=TL) == ["f-resnet", "f-beit"]
    seen = {}
    gather = describe_text.gather

    def spy(family, **kw):
        seen[family] = kw["tl"]
        return gather(family, **kw)

    monkeypatch.setattr(describe_text, "gather", spy)
    monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", "2")
    describe_write.write(None, data=data, store=world["store"])
    assert seen == {"f-resnet": TL, "f-beit": "2.35.2"}  # facts are read at the family's own pin


def test_a_hub_reserve_keeps_slots_free_for_other_lanes(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    monkeypatch.setenv(describe_write.HUB_RESERVE, "4")  # the stand-in hub has 4 free slots
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row["verdict"] == "waiting" and not world["launches"]
    monkeypatch.setenv(describe_write.HUB_RESERVE, "3")
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row["verdict"] == "launched" and len(world["launches"]) == 1


def test_write_lays_out_launches_returns_once_then_writes(tmp_path, world):
    data = _data(tmp_path)
    assert describe_write.write(None, data=data, store=world["store"], write=False) == [
        {"family": "f-resnet", "verdict": "due"}]
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert (row["verdict"], row["run"]) == ("launched", "run-1")
    lane = world["launches"][0]
    assert (lane["kind"], lane["role"], lane["lab"], lane["keys"]) == ("describe", "model-judgment", "claude", ("f-resnet",))
    folder = lane["brief"].parent
    family = folder / "families" / "f-resnet"
    facts = json.loads((family / "facts.json").read_text())
    assert (facts["mode"], facts["era"], facts["institutions"]) == ("full", "2015", ["Microsoft Research"])
    assert facts["output"]["m8839"]["output_shape"] == [1, 1000] and facts["jev"] == {}
    assert sorted(facts["sids"]) == ["s1", "s4"] and facts["vocab"]["paradigm"][0] == "CNN"
    assert (family / "source" / "s1-paper.txt").read_text() == PAPER
    assert "resnet18 is a residual net" in (family / "source" / "s4-readme.txt").read_text()
    assert lane["brief"].read_text().count("## The honesty rule") == 1
    assert describe_write.write(None, data=data, store=world["store"])[0]["verdict"] == "running"
    # An explicitly requested unvalidated family cannot launch.
    rows = describe_write.write(["f-resnet", "f-beit"], data=data, store=world["store"])
    assert {"family": "f-beit", "verdict": "not_validated"} in rows
    assert len(world["launches"]) == 1
    world["runs"]["run-1"]["state"] = "TERMINAL"
    bad = _draft(grounding=[{"sentence": "x", "sid": "s1", "quote": "a sentence the paper never says"},
                            {"sentence": "y", "sid": "s1", "quote": ""}])
    (folder / "out" / "f-resnet.json").write_text(json.dumps(bad))
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row["verdict"] == "returned" and row["run"] == "run-2" and len(world["launches"]) == 2
    assert any("not verbatim" in f for f in row["findings"]) and any("not grounded" in f for f in row["findings"])
    assert any(f.startswith("quote of 0 words") for f in row["findings"])
    assert (family / "RETURN.md").read_text().count("- ") == len(row["findings"])
    assert world["launches"][1]["label"].endswith("-r2") and world["launches"][1]["brief"].name == "BRIEF-r2.md"
    assert "BRIEF-r2.result.md" in world["launches"][1]["brief"].read_text()
    world["runs"]["run-2"]["state"] = "TERMINAL"
    (folder / "out" / "f-resnet.json").write_text(json.dumps(_draft()))
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row == {"family": "f-resnet", "verdict": "grounded", "written": True}
    record = from_dict(DescribeRecord, json.loads(data.describe("f-resnet").read_text()))
    assert record.grounding_state == "grounded" and record.cls is None and record.era == "2015"
    assert record.lineage[0].target_id == "f-beit" and record.institutions == ("Microsoft Research",)
    assert set(record.inputs) == {"s1", "s4", "bib-openalex-1"} and record.prompt == describe_write.prompt_version()
    assert (folder / "done.json").is_file()
    assert describe_write.write(None, data=data, store=world["store"]) == []  # nothing due now
    assert describe_write.check(None, data=data, store=world["store"]) == [
        {"family": "f-resnet", "verdict": "grounded", "findings": []}]


def test_a_lane_the_hub_never_admitted_starts_again_without_spending_the_return(tmp_path, world):
    data = _data(tmp_path)
    describe_write.write(None, data=data, store=world["store"])
    world["runs"]["run-1"] = {"state": "TERMINAL", "admission_state": "expired"}
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert (row["verdict"], row["run"]) == ("relaunched", "run-2")
    assert world["launches"][1]["keys"] == ("f-resnet",)
    folder = world["launches"][1]["brief"].parent
    assert world["launches"][1]["brief"].name == "BRIEF.md" and not (folder / "BRIEF-r2.md").exists()
    world["runs"]["run-2"] = {"state": "TERMINAL", "admission_state": "admitted"}
    (folder / "out" / "f-resnet.json").write_text(json.dumps(_draft()))
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row["verdict"] == "grounded" and len(world["launches"]) == 2


def test_a_second_failure_types_the_record_ungrounded(tmp_path, world):
    data = _data(tmp_path)
    describe_write.write(["f-resnet"], data=data, store=world["store"])
    folder = world["launches"][0]["brief"].parent
    world["runs"]["run-1"]["state"] = "TERMINAL"
    (folder / "out" / "f-resnet.json").write_text(json.dumps(_draft(labs=["Nowhere Lab"])))
    assert describe_write.write(["f-resnet"], data=data, store=world["store"])[0]["verdict"] == "returned"
    world["runs"]["run-2"]["state"] = "TERMINAL"
    [row] = describe_write.write(["f-resnet"], data=data, store=world["store"])
    assert row["verdict"] == "describe:ungrounded" and row["findings"] == ["lab 'Nowhere Lab' is not named in any saved text"]
    record = from_dict(DescribeRecord, json.loads(data.describe("f-resnet").read_text()))
    assert (record.cls, record.grounding_state, record.labs) == ("describe:ungrounded", "ungrounded", ("Nowhere Lab",))
    site = publish._describe_fields(from_dict(__import__("menagerie.pipeline.records", fromlist=["ModelRecord"]).ModelRecord,
                                              json.loads((FIX / "m8839.model.json").read_text())), record)
    assert site["summary_markdown"] is None and site["keywords"] == ["ResNet", "resnet18"]


def test_a_family_without_text_is_typed_at_once(tmp_path, world):
    data = _data(tmp_path, text=False)
    packet = json.loads(data.packet("f-resnet").read_text())
    packet["sources"] = [s for s in packet["sources"] if s["sid"] != "s4"]
    packet["sources"][0]["sid"] = "../escape"  # a sid that is not a file name is never written
    packet["sources"][0]["text"] = TEXT_SHA
    data.packet("f-resnet").write_text(json.dumps(packet))
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row["verdict"] == "describe:no_text" and not world["launches"]
    record = from_dict(DescribeRecord, json.loads(data.describe("f-resnet").read_text()))
    assert (record.cls, record.era, record.summary_markdown, record.by) == ("describe:no_text", "2015", None, "describe")
    assert describe_write.check(None, data=data, store=world["store"])[0]["verdict"] == "typed"
    assert describe_write.due(None, data=data, tl=TL) == []  # typed and still textless: settled
    packet["sources"][0] = {**packet["sources"][0], "sid": "s1", "text": TEXT_SHA}
    data.packet("f-resnet").write_text(json.dumps(packet))
    assert describe_write.due(None, data=data, tl=TL) == ["f-resnet"]  # a text arrived: due again


def test_readme_only_families_get_taxonomy_and_a_tagline(tmp_path, world):
    data = _data(tmp_path, text=False)
    [row] = describe_write.write(None, data=data, store=world["store"])
    folder = world["launches"][0]["brief"].parent
    assert json.loads((folder / "families" / "f-resnet" / "facts.json").read_text())["mode"] == "readme-only"
    world["runs"]["run-1"]["state"] = "TERMINAL"
    draft = _draft(summary_markdown=None, key_contribution=None, grounding=[], lineage=[], labs=[])
    (folder / "out" / "f-resnet.json").write_text(json.dumps(draft))
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row["verdict"] == "grounded"
    assert from_dict(DescribeRecord, json.loads(data.describe("f-resnet").read_text())).by == "readme-only"


S2_SHA, DOCS_SHA = "e" * 64, "f" * 64
DOCS = ("ResNet. We present a residual learning framework to ease the training of networks that are "
        "substantially deeper than those used previously. This model was trained by timm.")


def _short_texts(world, monkeypatch):
    """Stand-ins for an S2 record holding the cited paper's abstract and a library's docs."""

    def materialize(sha, dest, *, store):
        if sha not in (S2_SHA, DOCS_SHA):
            return _materialize(sha, dest, store=store)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(DOCS if sha == DOCS_SHA else json.dumps(
            {"title": "Deep residual learning for image recognition", "abstract": PAPER}))
        return dest

    monkeypatch.setattr(store_mod, "materialize", materialize)


def _cited(draft, sid):
    return {**draft, "lineage": [], "grounding": [{**g, "sid": sid} for g in draft["grounding"]]}


def test_an_abstract_grounds_a_textless_family_once_switched_on(tmp_path, world, monkeypatch):
    _short_texts(world, monkeypatch)
    data = _data(tmp_path, text=False)
    packet = json.loads(data.packet("f-resnet").read_text())
    packet["sources"] = [s for s in packet["sources"] if s["sid"] != "s4"]
    packet["sources"].append({"sid": "bib-s2-1", "role": "bib", "api": "s2", "blob": S2_SHA})
    data.packet("f-resnet").write_text(json.dumps(packet))
    [row] = describe_write.write(None, data=data, store=world["store"])
    record = json.loads(data.describe("f-resnet").read_text())
    assert (row["verdict"], record["basis"], record["inputs"]) == ("describe:no_text", "none", [])
    assert describe_write.due(None, data=data, tl=TL) == []  # settled while abstracts are off
    monkeypatch.setenv(describe_write.ABSTRACTS, "f-other,f-resnet")  # a trial naming the family
    assert describe_write.due(None, data=data, tl=TL) == ["f-resnet"]
    [row] = describe_write.write(None, data=data, store=world["store"])
    folder = world["launches"][0]["brief"].parent
    facts = json.loads((folder / "families" / "f-resnet" / "facts.json").read_text())
    assert (row["verdict"], facts["mode"], facts["sids"]) == ("launched", "abstract", ["bib-s2-1"])
    assert (folder / "families" / "f-resnet" / "source" / "bib-s2-1-abstract.txt").read_text() == PAPER
    world["runs"]["run-1"]["state"] = "TERMINAL"
    (folder / "out" / "f-resnet.json").write_text(json.dumps(_cited(_draft(), "bib-s2-1")))
    [row] = describe_write.write(None, data=data, store=world["store"])
    record = json.loads(data.describe("f-resnet").read_text())
    assert (row["verdict"], record["basis"], record["summary_markdown"] is not None) == ("grounded", "abstract", True)
    monkeypatch.delenv(describe_write.ABSTRACTS)
    assert describe_write.check(None, data=data, store=world["store"])[0]["verdict"] == "grounded"
    assert describe_write.due(None, data=data, tl=TL) == []
    packet["sources"][0]["text"] = TEXT_SHA  # the paper pass saves the full text: due again
    data.packet("f-resnet").write_text(json.dumps(packet))
    assert describe_write.due(None, data=data, tl=TL) == ["f-resnet"]


def test_library_docs_ground_a_summary_and_reopen_a_readme_record(tmp_path, world, monkeypatch):
    _short_texts(world, monkeypatch)
    data = _data(tmp_path, text=False)
    describe_write._write_record(describe_write._record(
        _draft(summary_markdown=None, key_contribution=None, grounding=[], lineage=[], labs=[]),
        {"sids": ["s4"], "mode": "readme-only"}, family="f-resnet", prompt="0" * 8,
        resolver=describe_write._Resolver(data)), "grounded", data=data, write=True)
    assert json.loads(data.describe("f-resnet").read_text())["basis"] == "readme"
    assert describe_write.due(None, data=data, tl=TL) == []
    packet = json.loads(data.packet("f-resnet").read_text())
    packet["sources"].append({"sid": "s5", "role": "code", "url": "https://github.com/x/y/blob/v1/docs.md",
                              "rev": "v1.0.30", "text": DOCS_SHA})
    data.packet("f-resnet").write_text(json.dumps(packet))
    assert describe_write.due(None, data=data, tl=TL) == ["f-resnet"]
    describe_write.write(None, data=data, store=world["store"])
    folder = world["launches"][0]["brief"].parent
    facts = json.loads((folder / "families" / "f-resnet" / "facts.json").read_text())
    assert (facts["mode"], facts["sids"]) == ("docs", ["s4", "s5"])
    world["runs"]["run-1"]["state"] = "TERMINAL"
    s1 = "ResNet is a residual learning framework that eases the training of substantially deeper networks."
    quote = "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously."
    draft = _draft(lineage=[], labs=[], summary_markdown=(s1 + " ") * 4, key_contribution=s1,
                   grounding=[{"sentence": s1, "sid": "s5", "quote": quote}])
    (folder / "out" / "f-resnet.json").write_text(json.dumps(draft))
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row["verdict"] == "grounded" and json.loads(data.describe("f-resnet").read_text())["basis"] == "docs"


def test_an_old_textless_record_is_looked_at_once_more_then_settles(tmp_path, world, monkeypatch):
    data = _data(tmp_path, text=False)
    record = describe_write._record(None, {"mode": "none", "sids": [], "institutions_sid": "bib-openalex-1"},
                                    family="f-resnet", prompt="0" * 8, resolver=describe_write._Resolver(data))
    body = {k: v for k, v in describe_write.to_dict(record).items() if k != "basis"}
    data.describe("f-resnet").parent.mkdir(parents=True, exist_ok=True)
    data.describe("f-resnet").write_text(json.dumps({**body, "class": "describe:no_text", "inputs": ["bib-openalex-1"]}))
    monkeypatch.setattr(describe_text, "_readme", lambda *a: None)  # the snapshot holds no README
    assert describe_write.due(None, data=data, tl=TL) == ["f-resnet"]
    [row] = describe_write.write(None, data=data, store=world["store"])
    record = json.loads(data.describe("f-resnet").read_text())
    assert (row["verdict"], record["basis"], record["inputs"]) == ("describe:no_text", "none", ["s4"])
    assert describe_write.due(None, data=data, tl=TL) == []  # no more looks every pass


def test_verify_names_each_failure(tmp_path, world):
    data = _data(tmp_path)
    resolver = describe_write._Resolver(data)
    record = from_dict(DescribeRecord, {"schema": "describe/1", "family": "f-resnet", "by": "t",
                                        "domain": "vision", "members": {"m9": {"output_type": "class-logits"}},
                                        "summary_markdown": "Short.", "inputs": ["s1"], "prompt": "0" * 8})
    findings = describe_write.verify(record, {"s1": PAPER}, mode="full", member_ids=["m8839"],
                                     resolves=resolver.resolves)
    assert [f.split(":")[0] for f in findings] == [
        "summary is 6 characters, not 250 to 4000", "sentence not grounded", "members names m9, not a packet member",
        "member m9 output_type 'class-logits' is outside its vocabulary", "domain 'vision' is outside its vocabulary"]
    assert describe_write.sentences("Fig. 1 shows it, e.g. here. Then 32x32 inputs? Yes.\n\nNext.") == [
        "Fig. 1 shows it, e.g. here.", "Then 32x32 inputs?", "Yes.", "Next."]
    assert describe_write._grounded("1 shows it.", [{"sentence": "Fig. 1 shows it.", "sid": "s1", "quote": "x"}])
    assert not describe_write._grounded("A long claim with many extra words added here.",
                                        [{"sentence": "A long claim.", "sid": "s1", "quote": "x"}])
    assert not describe_write._grounded("Anything.", [{"sentence": "", "sid": "s1", "quote": ""}])
    assert resolver.resolve("beit") == "f-beit" and resolver.resolve("ResNet-18") == "m8839"


def test_check_rewrites_the_state_when_a_quote_stops_holding(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    record = describe_write._record(_draft(), {"sids": ["s1", "s4"], "mode": "full", "era": "2015",
                                              "institutions": []}, family="f-resnet", prompt="0" * 8,
                                    resolver=describe_write._Resolver(data))
    describe_write._write_record(record, "grounded", data=data, write=True)
    assert describe_write.check(["f-resnet"], data=data, store=world["store"])[0]["verdict"] == "grounded"
    def changed(sha, dest, *, store):
        if sha != TEXT_SHA:
            return _materialize(sha, dest, store=store)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("Nothing here.")
        return dest

    monkeypatch.setattr(store_mod, "materialize", changed)
    [row] = describe_write.check(["f-resnet"], data=data, store=world["store"])
    assert row["verdict"] == "ungrounded" and len(row["findings"]) >= 4
    assert json.loads(data.describe("f-resnet").read_text())["grounding_state"] == "ungrounded"


def _journal(argv, answers_for):
    states = [json.loads(line) for line in Path(argv[argv.index("--in") + 1]).read_text().splitlines()]
    with open(argv[argv.index("--out") + 1], "w") as out:
        out.writelines(json.dumps({"id": row["id"], "status": "ok", "answers": answers_for(row["id"])}) + "\n"
                       for row in states)
    return describe_jev.subprocess.CompletedProcess(argv, 0, "", "")


def test_drive_runs_jev_labels_by_lane_and_enables_fields(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    calls = []

    def answers(fid):
        return {"domain": {"choice": "vision/classification-backbone",
                           "probabilities": {"vision/classification-backbone": 0.9, "unknown": 0.1}},
                "paradigm": {"choice": "Transformer", "probabilities": {"Transformer": 0.8, "CNN": 0.2}},
                "is_generative": {"noul": 0.05}}

    monkeypatch.setattr(describe_jev.subprocess, "run", lambda argv, **kw: calls.append(argv) or _journal(argv, answers))
    starts = []
    monkeypatch.setattr(review, "_start", lambda role, brief, ids, label, keys=None: starts.append((role, label))
                        or (brief.parent / "lane.json").write_text(json.dumps({"run": "run-l"})) or "run-l")
    world["runs"]["run-l"] = {"state": "RUNNING"}
    first = describe_jev.drive(50, data=data)
    assert (first["state"], first["families"], starts[0][0]) == ("launched", 1, "model-judgment")
    assert calls[0][calls[0].index("--profile") + 1] == "describe-taxonomy"
    folder = next(p for p in (Path(tmp_path / "work") / "describe-drive").iterdir())
    state = json.loads((folder / "states.jsonl").read_text())["state"]
    assert state["name"] == "ResNet" and state["abstract"].startswith("Deep Residual") and "resnet18" in state["readme"]
    assert state["output"]["m8839"]["n_output_tensors"] == 1
    assert describe_jev.drive(50, data=data)["state"] == "running" and len(calls) == 1
    world["runs"]["run-l"]["state"] = "TERMINAL"
    (folder / "labels.jsonl").write_text(json.dumps({"id": "f-resnet", "domain": "vision/classification-backbone",
                                                     "paradigm": "CNN", "is_generative": False}) + "\n")
    result = describe_jev.drive(50, data=data)
    assert result["fields"]["domain"] == {"agree": 1, "enabled": False}  # one family is not a drive
    assert (result["complete"], result["need"]) == (False, 45) and describe_jev.enabled_fields(data) == set()
    # A short drive can never enable a field, even if every available label agrees.
    scored = describe_jev.score(folder, ["f-resnet"], wanted=1)
    assert scored["fields"]["domain"] == {"agree": 1, "enabled": False}
    assert scored["fields"]["paradigm"] == {"agree": 0, "enabled": False}
    assert not scored["fields"]["is_generative"]["enabled"]
    (data.root / "review" / "describe-drive.jsonl").write_text(json.dumps({**result,
        "wanted": 50, "families": 50, "complete": True, "fields": {
            "domain": {"enabled": True, "agree": 45}, "is_generative": {"enabled": True, "agree": 50}}}) + "\n")
    assert describe_jev.enabled_fields(data) == {"domain", "is_generative"}
    describe_jev.drive(50, data=data)
    assert len((data.root / "review" / "describe-drive.jsonl").read_text().splitlines()) == 1
    # with fields enabled, `describe write` fills them by jev and the record copies them
    describe_write.write(["f-resnet"], data=data, store=world["store"])
    facts = json.loads((world["launches"][0]["brief"].parent / "families" / "f-resnet" / "facts.json").read_text())
    assert facts["jev"] == {"domain": "vision/classification-backbone", "is_generative": False}


def test_classify_keeps_only_confident_enabled_answers(tmp_path, monkeypatch):
    data = DataLayout(tmp_path / "data")
    (data.root / "review").mkdir(parents=True)
    (data.root / "review" / "describe-drive.jsonl").write_text(json.dumps(
        {"batch": "x", "wanted": 50, "families": 50, "complete": True,
         "fields": {"domain": {"enabled": True, "agree": 45}, "modality": {"enabled": True, "agree": 45},
                                  "paradigm": {"enabled": False}}}) + "\n")
    monkeypatch.setattr(describe_jev.subprocess, "run", lambda argv, **kw: _journal(argv, lambda fid: {
        "domain": {"choice": "audio/speech", "probabilities": {"audio/speech": 0.55, "unknown": 0.45}},
        "modality": {"choice": "audio", "probabilities": {"audio": 0.95, "text": 0.05}},
        "paradigm": {"choice": "CNN", "probabilities": {"CNN": 0.99}}}))
    got = describe_jev.classify(tmp_path / "jev", [{"id": "f-x", "state": {}}], data=data)
    assert got == {"f-x": {"modality": "audio"}}
    assert describe_jev.classify(tmp_path / "jev2", [], data=data) == {}
    assert describe_jev._call({"choice": "audio"}) == ("audio", 0.0, 0.0)  # no probabilities, no confidence
    assert describe_jev._call({"choice": "audio", "probabilities": {"text": 0.9}}) == ("audio", 0.0, 0.0)
    assert describe_jev._call({"choice": "unknown", "probabilities": {"unknown": 0.9, "audio": 0.1}})[0] is None
    assert describe_jev._call({"noul": 0.9}) == (True, 0.9, pytest.approx(0.8))
    assert describe_jev.stratified({"a": "v", "b": "v", "c": "nlp", "d": None}, 3).count("c") == 1


def test_jev_profile_tracks_the_vocabularies_and_the_prompt_is_short():
    profile = json.loads((REPO / "menagerie" / "data" / "review" / "jev-describe-profile.json").read_text())
    for name in describe_jev.VOCAB_FIELDS:
        assert set(profile["questions"][name]["criteria"]) == set(describe_jev.vocabularies()[name]) | {"unknown"}
    assert profile["questions"]["is_generative"]["type"] == "noul" and profile["name"] == describe_jev.PROFILE
    assert len(describe_write.PROMPT.read_text().splitlines()) < 80


def _many(data, monkeypatch, count):
    names = [f"f-seat-{n:02d}" for n in range(count)]
    original = json.loads(data.packet("f-resnet").read_text())
    for name in names:
        path = data.packet(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**original, "family": name}))
    monkeypatch.setattr(describe_text, "validated_families", lambda **kw: {f: ["m8839"] for f in names})
    return names


def test_four_seats_stage_disjoint_keys_and_a_fifth_waits(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    names = _many(data, monkeypatch, 50)
    monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", "4")
    result = describe_write.write(names, data=data, store=world["store"])
    assert len(world["launches"]) == 4
    keys = [set(lane["keys"]) for lane in world["launches"]]
    assert all(len(k) == 10 for k in keys) and len(set.union(*keys)) == 40
    assert sum(r["verdict"] == "waiting" for r in result) == 10
    assert len(describe_write.seats(data=data)) == 4
    describe_write.write(names, data=data, store=world["store"])
    assert len(world["launches"]) == 4


def test_default_cap_is_four_and_hub_room_limits_launches(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    names = _many(data, monkeypatch, 30)
    monkeypatch.delenv("MENAGERIE_LANES_DESCRIBE")
    assert describe_write._cap() == 4
    monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", "1")
    describe_write.write(names, data=data, store=world["store"])
    assert len(world["launches"]) == 1 and describe_write._cap() == 1
    monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", "4")
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 0)
    describe_write.write(names, data=data, store=world["store"])
    assert len(world["launches"]) == 1


def test_held_family_and_refused_launch_remain_waiting(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    monkeypatch.setattr(drive_mod, "lanes", lambda **kw: [{"kind": "2b", "run": "other", "keys": ["f-resnet"]}])
    result = describe_write.write(["f-resnet"], data=data, store=world["store"])
    assert result[0]["verdict"] == "waiting" and not world["launches"]
    monkeypatch.setattr(drive_mod, "lanes", lambda **kw: [])
    def refuse(*args, **kwargs):
        raise drive_mod.Refused("lease_held", "held")
    monkeypatch.setattr(drive_mod, "launch", refuse)
    assert describe_write.write(["f-resnet"], data=data, store=world["store"])[0]["verdict"] == "waiting"
    assert len(describe_write._folders()) == 1


def test_lowered_cap_also_holds_return_lanes(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    describe_write.write(["f-resnet"], data=data, store=world["store"])
    world["runs"]["run-1"]["state"] = "TERMINAL"
    monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", "0")
    [row] = describe_write.write(["f-resnet"], data=data, store=world["store"])
    assert row["verdict"] == "waiting" and len(world["launches"]) == 1
    monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", "1")
    [row] = describe_write.write(["f-resnet"], data=data, store=world["store"])
    assert row["verdict"] == "returned" and len(world["launches"]) == 2


def test_trial_keeps_classifier_values_out_of_prose_inputs(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    monkeypatch.setenv("MENAGERIE_DESCRIBE_TRIAL", "1")
    monkeypatch.setattr(describe_jev, "classify", lambda *a, **k: {"f-resnet": {"is_generative": True}})
    describe_write.write(["f-resnet"], data=data, store=world["store"])
    folder = world["launches"][0]["brief"].parent
    facts = json.loads((folder / "families/f-resnet/facts.json").read_text())
    assert facts["jev"] == {}
    assert json.loads((folder / "trial-values.json").read_text()) == {"f-resnet": {"is_generative": True}}
    assert "answer every vocabulary field yourself" in world["launches"][0]["brief"].read_text()


@pytest.mark.parametrize("missing_text", [0, 5])
def test_concurrent_writers_reserve_no_duplicate_seat_or_family(tmp_path, world, monkeypatch, missing_text):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    data = _data(tmp_path)
    names = _many(data, monkeypatch, 50)
    _without_text(data, names[:missing_text])
    monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", "4")
    started, release = threading.Event(), threading.Event()
    original = drive_mod.launch
    def slow_launch(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(drive_mod, "launch", slow_launch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(describe_write.write, names, data=data, store=world["store"])
        assert started.wait(10)
        second = pool.submit(describe_write.write, names, data=data, store=world["store"])
        release.set()
        first.result(timeout=20)
        second.result(timeout=20)
    leases = [key for lane in world["launches"] for key in lane["keys"]]
    assert len(world["launches"]) == 4 and len(leases) == len(set(leases)) == 40


def _without_text(data, names):
    for name in names:
        path = data.packet(name)
        packet = json.loads(path.read_text())
        path.write_text(json.dumps({**packet, "sources": []}))


@pytest.mark.parametrize("missing_text", [1, 13, 30])
def test_one_seat_fills_ten_text_families_after_typing_missing_text(tmp_path, world, monkeypatch, missing_text):
    data = _data(tmp_path)
    names = _many(data, monkeypatch, missing_text + 10)
    _without_text(data, names[:missing_text])
    rows = describe_write.write(names, data=data, store=world["store"])
    assert len(world["launches"]) == 1
    assert set(world["launches"][0]["keys"]) == set(names[missing_text:])
    assert {r["family"] for r in rows if r["verdict"] == "describe:no_text"} == set(names[:missing_text])
    assert len(rows) == len(names) and not any(r["verdict"] == "waiting" for r in rows)


def test_a_second_seat_never_types_a_family_the_first_already_typed(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    names = _many(data, monkeypatch, 45)
    _without_text(data, names[:25])
    monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", "2")
    rows = describe_write.write(names, data=data, store=world["store"])
    typed = [r["family"] for r in rows if r["verdict"] == "describe:no_text"]
    assert sorted(typed) == sorted(names[:25])  # each once
    keys = [set(lane["keys"]) for lane in world["launches"]]
    assert len(keys) == 2 and set.union(*keys) == set(names[25:45])


def test_busy_hub_types_missing_text_before_agent_admission(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    names = _many(data, monkeypatch, 12)
    _without_text(data, names[:2])
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 0)
    rows = describe_write.write(names, data=data, store=world["store"])
    assert not world["launches"]
    assert {r["family"] for r in rows if r["verdict"] == "describe:no_text"} == set(names[:2])
    assert {r["family"] for r in rows if r["verdict"] == "waiting"} == set(names[2:])
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 1)
    describe_write.write(names[2:], data=data, store=world["store"])
    assert len(world["launches"]) == 1 and len(world["launches"][0]["keys"]) == 10


def test_one_seat_targeted_call_leaves_an_unrelated_batch_alone(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    names = _many(data, monkeypatch, 20)
    describe_write.write(names[:10], data=data, store=world["store"])
    world["runs"]["run-1"]["state"] = "TERMINAL"
    rows = describe_write.write(names[10:], data=data, store=world["store"])
    assert len(world["launches"]) == 1
    assert all(r["family"] in names[10:] and r["verdict"] == "waiting" for r in rows)


# --- worker seats (``describe_workers``) ---------------------------------------------------------


def _on_workers(world, monkeypatch, *, placed=True):
    """Workers switched on; a worker start is stood in for (``None``: no worker had room)."""

    monkeypatch.setenv(describe_workers.WORKERS, "1")
    started = []

    def launch(brief, families, *, role):
        if not placed:
            return None
        run = f"wrun-{len(started) + 1}"
        started.append({"brief": Path(brief), "families": list(families), "role": role, "run": run})
        world["runs"][run] = {"state": "RUNNING", "admission_state": None}
        return run, "zm", f"describe-mini-{len(started)}"

    monkeypatch.setattr(describe_workers, "launch", launch)
    return started


def test_switched_off_never_asks_a_worker(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    monkeypatch.setattr(describe_workers, "launch", lambda *a, **kw: pytest.fail("asked a worker"))
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 0)
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row["verdict"] == "waiting" and row["why"]["result"] == "hub_full"
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 4)
    [row] = describe_write.write(None, data=data, store=world["store"])
    lane = json.loads((world["launches"][0]["brief"].parent / "lane.json").read_text())
    assert row["verdict"] == "launched" and "placement" not in lane


def test_a_worker_seat_skips_the_hub_gate_and_settles_after_ingest(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    started = _on_workers(world, monkeypatch)
    monkeypatch.setattr(tick_full, "hub_slots", lambda: pytest.fail("a worker seat read the hub gate"))
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert (row["verdict"], row["run"]) == ("launched", "wrun-1") and not world["launches"]
    assert started[0]["families"] == ["f-resnet"] and started[0]["role"] == "model-judgment"
    folder = started[0]["brief"].parent
    lane = json.loads((folder / "lane.json").read_text())
    assert (lane["placement"], lane["label"]) == ("zm", "describe-mini-1")
    # Ended on the worker but not ingested: the lease still holds it, so it is still running.
    world["runs"]["wrun-1"]["state"] = "TERMINAL"
    leases = describe_write.work_dir("leases")
    leases.mkdir(parents=True)
    (leases / "describe-mini-1.json").write_text("{}")
    assert describe_write.write(None, data=data, store=world["store"])[0]["verdict"] == "running"
    (folder / "out" / "f-resnet.json").write_text(json.dumps(_draft()))  # what the ingest keeps
    mini_relay.state_dir("describe-mini-1").mkdir(parents=True)
    (mini_relay.state_dir("describe-mini-1") / "ingested.json").write_text('{"status": "ingested"}')
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row == {"family": "f-resnet", "verdict": "grounded", "written": True}


def test_no_worker_room_falls_back_to_the_hub_behind_its_gate(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    _on_workers(world, monkeypatch, placed=False)
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 0)
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert row["verdict"] == "waiting" and row["why"]["result"] == "hub_full" and not world["launches"]
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 4)
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert (row["verdict"], row["run"]) == ("launched", "run-1")


def test_a_lost_worker_lane_is_returned_like_a_failed_hub_lane(tmp_path, world, monkeypatch):
    data = _data(tmp_path)
    started = _on_workers(world, monkeypatch)
    describe_write.write(None, data=data, store=world["store"])
    world["runs"]["wrun-1"]["state"] = "TERMINAL"  # no ingest, and its lease expired and is gone
    [row] = describe_write.write(None, data=data, store=world["store"])
    assert (row["verdict"], row["run"]) == ("returned", "wrun-2")
    assert row["findings"] == ["the lane wrote no draft"]
    assert started[1]["brief"].name == "BRIEF-r2.md"
