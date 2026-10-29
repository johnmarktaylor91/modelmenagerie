"""The harvester on the committed rung 0 records and the real June catalog.

Only records whose family packet the harvester wrote are its own: rung 1 research and
implementation lanes took some rung 0 families over (f-beit3, f-esmfold) and wrote others, and
their lanes test those. Rung 3's 2b lanes also take single members of the harvester's families
over (an accepted recipe, or a typed give-up that keeps the old recipe beside it); those records
are the lane's too. The record checks need only the repository. Reproducing the records and constructing every
recipe need the ``torch-main`` libraries (torch, timm, torchvision, transformers, smp,
ultralytics, diffusers), so those tests skip without them.
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import json
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, read_json, recipe_revision
from menagerie.pipeline.harvest import (
    RUNG0_NAMED,
    _adapter_bytes,
    _load_file,
    _lock,
    _parse_call,
    _weights_gib,
    construct,
    harvest,
)
from menagerie.pipeline.records import OFF_WORKER, ModelRecord, Packet, to_dict

DATA = DataLayout.from_env()


def _harvested(path: Path) -> bool:
    record = read_json(ModelRecord, path)
    packet = read_json(Packet, DATA.packet(record.family))
    return packet.by.startswith("harvester@") and not _taken_over(record)


def _taken_over(record: ModelRecord) -> bool:
    """A 2b lane's member: an accepted recipe records the lane's forward modes
    (``implement.done``), a lane's give-up is typed ``by: agent``; the harvester writes neither."""

    return bool((record.recipe and record.recipe.modes)
                or (record.give_up and record.give_up.by == "agent"))


RECORDS = sorted(p for p in DATA.root.glob("models/*/m*.json") if _harvested(p))
SLICE_A = DATA.root / "regression" / "slice_a_200.tsv"
needs_libraries = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("torch", "timm", "transformers", "ultralytics")),
    reason="needs the torch-main libraries",
)


def _records() -> list[ModelRecord]:
    return [read_json(ModelRecord, path) for path in RECORDS]


def _rung0_ids() -> set[str]:
    smoke = [json.loads(line) for line in (DATA.root / "smoke_manifest.jsonl").open()]
    slice_a = [row["stable_id"] for row in csv.DictReader(SLICE_A.open(), delimiter="\t")]
    return {*RUNG0_NAMED, *slice_a,
            *(r["stable_id"] for r in smoke if r["expected_env"] == "base" and not r["synthetic"])}


# The catalog pass's tier records (harvest_tiers) are checked by their own tests and by stage 3;
# building every one here would take hours. Construction and placement cover rung 0's records.
RUNG0_RECORDS = [p for p in RECORDS if p.stem in _rung0_ids()]


def test_every_rung0_row_has_a_record_and_a_packet() -> None:
    every = {p.stem for p in DATA.root.glob("models/*/m*.json")}
    assert _rung0_ids() <= every, sorted(_rung0_ids() - every)
    records = {r.id: r for r in _records()}
    for record in records.values():
        packet = read_json(Packet, DATA.packet(record.family))
        assert record.id in {m.id for m in packet.members}
        assert (record.recipe is None) == (record.give_up is not None)
        if record.recipe is None:
            continue
        sids = {s.sid: s for s in packet.sources}
        assert set(record.recipe.sources) <= set(sids)
        assert all(sids[s].role in ("library", "code") for s in record.recipe.sources)


def test_recipe_revisions_match_the_lock_and_adapters() -> None:
    lock = _lock("torch-main")
    for record in (r for r in _records() if r.recipe):
        recipe = to_dict(record.recipe)
        assert recipe["rev"] == recipe_revision(recipe, _adapter_bytes(recipe, DATA), lock)


def _own(records: dict[str, ModelRecord], sid: str) -> ModelRecord | None:
    """The harvester's record for ``sid``, or None once a lane has taken it over (never gone)."""

    if sid in records:
        return records[sid]
    record = records_named(sid)
    packet = read_json(Packet, DATA.packet(record.family))
    assert _taken_over(record) or not packet.by.startswith("harvester@"), sid
    return None


def test_record_facts_on_known_rows() -> None:
    records = {r.id: r for r in _records()}
    if resnet_row := _own(records, "m8839"):  # rung 3's f-resnet lane rewrote it (2026-10-21)
        resnet = resnet_row.recipe
        assert (resnet.call, resnet.args, resnet.kwargs) == ("timm.create_model", ("resnet18",),
                                                             {"pretrained": False})
        assert (resnet.input.preset, resnet.input.shape, resnet.input.asset) == (
            "image", (1, 3, 224, 224), "std-image-01")
        assert resnet_row.hf and resnet_row.hf.id.startswith("timm/resnet18")
    # Hugging Face rows build the library's reference config, never June's shrunken one.
    if bert := _own(records, "m3638"):
        assert bert.recipe.config == {"model_type": "bert"}
        assert bert.hf.id == "google-bert/bert-base-uncased" and bert.recipe.input.preset == "text"
    gemma = records["m5140"].recipe  # June ran a hand-written stand-in under this name
    assert (gemma.call, gemma.config) == ("transformers.AutoModel.from_config",
                                          {"model_type": "gemma"})
    # esmfold's Hub-config gap went to research at rung 1, which now owns its family.
    assert "m4742" not in records and read_json(Packet, DATA.packet("f-esmfold")).by.startswith("2a@")
    if longcat := _own(records, "m5651"):
        assert longcat.recipe.placement is None  # no measurement yet: stage 3 measures
    lstm = records["m2061"].recipe
    assert (lstm.kind, lstm.adapter) == ("classic", "menagerie.classics.orig_lstm_1997:build")
    assert lstm.input.preset == "example:menagerie.classics.orig_lstm_1997:example_input"
    if dcgan := _own(records, "m482"):
        assert dcgan.recipe.adapter == "adapters/f-dcgan.py"
    assert records["m11497"].recipe.call == "ultralytics.nn.tasks.DetectionModel"
    standin = records["m3488"].recipe  # June built a 64 px stand-in for aimv2_3b_patch14_224
    assert (standin.call, standin.args) == (
        "timm.create_model", ("aimv2_3b_patch14_224",))
    unet = read_json(Packet, DATA.packet("f-unet"))  # smp's `.. _Unet:` target, not a param link
    assert [s.url for s in unet.sources if s.role == "paper"] == ["https://arxiv.org/abs/1505.04597"]
    beit = read_json(Packet, DATA.packet("f-beit"))
    assert beit.citation and beit.citation.arxiv == "2106.08254"
    assert beit.availability.license == "found"
    assert {s.spdx for s in beit.sources if s.role == "license"} == {"Apache-2.0"}


def test_placement_is_stage3s_rule_on_the_ledger_or_the_plan() -> None:
    from menagerie.pipeline import stage3

    if not _taken_over(records_named("m3635")):  # a lane typed it oversized (2026-10-22)
        assert records_named("m3635").recipe.placement == "axon-reviewed"  # the plan's 128 GiB job
    assert records_named("m8839").recipe.placement is None  # resnet18 fits a worker
    rung0 = _rung0_ids()  # a catalog row's placement moves as stage 3 measures it
    measured = _traced_at_revision()  # and so does a rung 0 row's, once stage 3 traces it again
    for record in (r for r in _records() if r.recipe and r.id != "m3635" and r.id in rung0):
        if (record.id, record.recipe.rev) in measured:
            continue  # stage 3 measured it; the stored placement was the projection before that
        rule = stage3.placement(record.id, data=DATA)["placement"]
        assert record.recipe.placement == (None if rule == "worker" else rule), record.id
    # memory_gib is a forward's measured peak at the record's revision (fix 8), never a trace's.
    forwards = {}
    for path in (DATA.root / "ledger").glob("*.jsonl"):
        for row in map(json.loads, path.open()):
            if row.get("forward_peak_rss_mb"):
                key = (row["id"], row["recipe_rev"])
                forwards[key] = max(forwards.get(key, 0), round(row["forward_peak_rss_mb"] / 1024, 2))
    for record in (r for r in _records() if r.recipe and r.id in rung0):
        if (record.id, record.recipe.rev) not in measured:  # as above: stage 3 moves it
            assert record.recipe.memory_gib == forwards.get((record.id, record.recipe.rev)), record.id


def _traced_at_revision() -> set[tuple[str, str]]:
    """The (id, recipe revision) pairs with a trace row on the ledger."""

    return {(row["id"], row["recipe_rev"]) for path in (DATA.root / "ledger").glob("*.jsonl")
            for row in map(json.loads, path.open()) if row.get("stage") == "trace"}


def records_named(stable_id: str) -> ModelRecord:
    return read_json(ModelRecord, DATA.model(stable_id))


@pytest.mark.parametrize(("code", "expected"), [
    ("timm.create_model('resnet18', pretrained=False)",
     ("timm.create_model", ["resnet18"], {"pretrained": False}, {})),
    (("import segmentation_models_pytorch as smp; model=smp.Unet(encoder_name='resnet34', "
      "encoder_weights=None, in_channels=3, classes=21); model.eval()"),
     ("segmentation_models_pytorch.Unet", [], {"encoder_name": "resnet34",
      "encoder_weights": None, "in_channels": 3, "classes": 21}, {})),
    ("AutoModel.from_config(AutoConfig.for_model('gpt2', n_layer=2))",
     ("transformers.AutoModel.from_config", [], {}, {"model_type": "gpt2", "n_layer": 2})),
    ("from menagerie.classics.wave_d_reimpl import build_hf_text; model=build_hf_text('decoder')",
     ("menagerie.classics.wave_d_reimpl.build_hf_text", ["decoder"], {}, {})),
    ("import torch; model = torch.nn.Sequential(torch.nn.Linear(2, 2))", None),
    ("import torch\nclass A(torch.nn.Module):\n    pass\nmodel = A()", None),
])
def test_parse_call_translates_june_recipes(code: str, expected: tuple | None) -> None:
    parsed = _parse_call(code)
    assert (parsed[:4] if parsed else None) == expected


@needs_libraries
def test_harvest_reproduces_the_committed_records() -> None:
    mine = _rung0_ids() & {p.stem for p in RECORDS}
    fresh = {r.id: r for r in harvest("rung0", sorted(mine), data=DATA, write=False)}
    committed = {r.id: r for r in _records() if r.id in fresh}
    assert set(fresh) == mine
    measured = _traced_at_revision()
    assert ({k: _unmeasured(v, measured) for k, v in fresh.items()}
            == {k: _unmeasured(v, measured) for k, v in committed.items()})


def _unmeasured(record: ModelRecord, measured: set[tuple[str, str]]) -> dict:
    """A record as JSON, without the fields the harvester reads off the ledger (placement and
    memory_gib) once stage 3 has traced it at its revision: those move as stage 3 measures."""

    out = to_dict(record)
    if record.recipe and (record.id, record.recipe.rev) in measured:
        out["recipe"] = {k: v for k, v in out["recipe"].items()
                         if k not in ("placement", "memory_gib")}
    return out


@needs_libraries
@pytest.mark.parametrize("path", RUNG0_RECORDS, ids=[p.stem for p in RUNG0_RECORDS])
def test_every_recipe_constructs_and_takes_its_input(path: Path) -> None:
    import torch

    from menagerie.pipeline.inputs import make_input

    record = read_json(ModelRecord, path)
    if record.recipe is None:
        assert record.give_up.cls.startswith("research_gap")
        return
    # Built on meta: off-worker rows, and rows whose weights alone would crowd the test's memory.
    giant = record.recipe.placement in OFF_WORKER or _weights_gib(to_dict(record.recipe), DATA) > 8
    missing = [d for d in _extra_deps(record) if importlib.util.find_spec(d) is None]
    if missing:  # typed `missing_dependency` by the runner until an environment holds it
        with pytest.raises(ModuleNotFoundError, match=missing[0]):
            construct(record.recipe)
        return
    model = construct(record.recipe, device="meta" if giant else "cpu")
    params = sum(p.numel() for p in model.parameters())
    assert params > 0
    assert all(p.is_meta for p in model.parameters()) == giant
    args, kwargs = make_input(record.recipe.input)
    if record.recipe.input.shape:  # an example preset may return keyword inputs (t5's two)
        first = args[0] if args else next(iter(kwargs.values()))
        assert tuple(first.shape) == record.recipe.input.shape
    if record.id in RUNG0_NAMED and not giant:
        model.eval()
        with torch.no_grad():
            model(*args, **kwargs)
    del model
    gc.collect()


def _extra_deps(record: ModelRecord) -> list[str]:
    if record.recipe.kind != "adapter":
        return []
    return list(_load_file(DATA.root / record.recipe.adapter).META["extra_deps"])


@needs_libraries
def test_construct_is_deterministic_and_leaves_the_rng_alone() -> None:
    import torch

    recipe = records_named("m8839").recipe
    state = torch.random.get_rng_state()
    first, second = construct(recipe), construct(recipe)
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(),
                                                  second.state_dict().values()))


@pytest.mark.parametrize(("code", "kwargs"), [
    ("from spikingjelly.activation_based import neuron, surrogate; "
     "from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg19_bn; "
     "model = spiking_vgg19_bn(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan())",
     {"spiking_neuron": {"$ref": "spikingjelly.activation_based.neuron.IFNode"},
      "surrogate_function": {"$call": "spikingjelly.activation_based.surrogate.ATan",
                             "args": [], "kwargs": {}}}),
    ("from gigagan_pytorch import UnetUpsampler; "
     "model = UnetUpsampler(dim=64, style_network=dict(dim=64, depth=4))",
     {"dim": 64, "style_network": {"dim": 64, "depth": 4}}),
])
def test_parse_call_keeps_components_and_dicts_under_refs(code: str, kwargs: dict) -> None:
    assert _parse_call(code) is None  # without refs: literals only, as before
    assert _parse_call(code, refs=True)[2] == kwargs


@pytest.mark.parametrize("code", [
    "import torch; model = torch.nn.Sequential(torch.nn.Linear(2, 2))",  # a hand-written stand-in
    "from x import Net; model = Net(block=make_block())",  # a local name: never called
    "from x import Net; model = Net(cfg=dict(**sizes))",
    "import x; cfg = x.Config(n=2); model = x.Net(cfg)",  # a name the code binds itself
    "from x import Net; model = Net(block=sizes.width)",
])
def test_parse_call_refuses_what_is_not_a_library_component(code: str) -> None:
    assert _parse_call(code, refs=True) is None


def test_a_component_call_is_built_when_the_model_is() -> None:
    from fractions import Fraction

    from menagerie.pipeline.harvest_build import _deref, component

    value = {"$call": "fractions.Fraction", "args": [1, 3], "kwargs": {}}
    assert component(value) and component({"$ref": "fractions.Fraction"})
    assert _deref(value) == Fraction(1, 3)
    assert _deref([value, {"$ref": "fractions.Fraction"}]) == [Fraction(1, 3), Fraction]
    assert not component({"$call": "x"}) and _deref({"a": 1}) == {"a": 1}


def test_a_rerun_never_drops_a_packet_citation() -> None:
    """The drafts review's hazard: 20 harvester packets lost their citation to a rerun whose
    drafts carried no paper title."""
    import dataclasses

    from menagerie.pipeline.harvest import _carried
    from menagerie.pipeline.records import Citation, Search

    fixtures = Path(__file__).parent / "fixtures"
    old = dataclasses.replace(read_json(Packet, fixtures / "f-resnet.packet.json"),
                              citation=Citation(title="Deep Residual Learning"), amendments=())
    assert _carried(old, None, Search())["citation"] == old.citation
    found = Citation(title="Another title")
    assert _carried(old, found, Search())["citation"] == found


def test_a_rerun_keeps_an_agents_give_up_whole(tmp_path: Path) -> None:
    from menagerie.pipeline import DataLayout, write_json
    from menagerie.pipeline.harvest import Draft, Row, _record
    from menagerie.pipeline.records import GiveUp, ModelRecord

    data = DataLayout(tmp_path)
    give = GiveUp(cls="needs_adapter", reason="the agent's reading", at="2026-10-09T10:00:00Z",
                  by="agent", missing="an adapter for the two inputs")
    write_json(ModelRecord(id="m1", name="Net", zoo="z", variant="", family="f-net",
                           source_case="library", give_up=give), data.model("m1"))
    draft = Draft(Row("m1", "Net", "z", "", "Net"), "library", {},
                  give_up=("needs_adapter", "the script's text"))
    assert _record(draft, "f-net", [], data).give_up == give
    scripted = GiveUp(cls="needs_adapter", reason="old script text", at=give.at, by="script")
    write_json(ModelRecord(id="m1", name="Net", zoo="z", variant="", family="f-net",
                           source_case="library", give_up=scripted), data.model("m1"))
    kept = _record(draft, "f-net", [], data).give_up
    assert (kept.reason, kept.at, kept.by) == ("the script's text", give.at, "script")
