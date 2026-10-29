"""The catalog pass's plan and the library tier, on the committed June catalog and roster.

The plan is torch-free. Drafting and building recipes need the ``torch-main`` libraries, so those
tests skip without them; a zoo whose library is an overlay (monai) skips unless it is installed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, harvest_tiers, read_json
from menagerie.pipeline.harvest import Draft, NoRecipe, Row, _carried, _parse_call
from menagerie.pipeline.harvest_tiers import catalog_plan, group, joined, parse_model_call
from menagerie.pipeline.records import ModelRecord, Packet, Search

DATA = DataLayout.from_env()
needs_libraries = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("torch", "timm", "transformers", "diffusers",
                                                  "segmentation_models_pytorch")),
    reason="needs the torch-main libraries",
)
# The harvest-tier test's data root, frozen so the live catalog pass cannot move it (its README).
FROZEN = Path(__file__).parent / "fixtures" / "harvest-tier"
PLAN_INPUTS = ("crawl_roster.jsonl", "master_catalog.jsonl", "routing_manifest.tsv",
               "catalog/r2-rows.tsv")


def _plan_root(tmp_path: Path) -> DataLayout:
    data = DataLayout(tmp_path / "data")
    for name in PLAN_INPUTS:
        (data.root / name).parent.mkdir(parents=True, exist_ok=True)
        (data.root / name).symlink_to(DATA.root / name)
    return data


def _rows(*ids: str) -> dict[str, Row]:
    found = {ros["stable_id"]: Row(ros["stable_id"], cat["name"], cat["zoo"], cat["variant"],
                                   cat["family"], catalog=cat)
             for cat, ros in joined(DATA) if ros["stable_id"] in ids}
    assert set(found) == set(ids)
    return found


def test_catalog_plan_groups_every_joined_row(tmp_path: Path) -> None:
    data = _plan_root(tmp_path)
    counts = catalog_plan(data=data)
    assert counts["rows"] == 7699
    assert counts["group"] == {"A": 5205, "B": 1267, "X": 1227}
    with (data.root / "catalog" / "plan.tsv").open(encoding="utf-8") as handle:
        plan = list(csv.DictReader(handle, delimiter="\t"))
    assert [int(r["order"]) for r in plan] == list(range(1, 7700))
    frozen = {r["id"] for r in csv.DictReader((DATA.root / "catalog" / "r2-rows.tsv").open(),
                                               delimiter="\t")}
    assert {r["id"] for r in plan if r["wave"] == "skip"} == frozen & {r["id"] for r in plan}
    waves = [harvest_tiers.WAVES.index(r["wave"]) for r in plan]
    assert waves == sorted(waves)
    c1 = [r for r in plan if r["wave"] == "C1"]
    assert c1[0]["zoo"] == "timm" and c1[-1]["group"] == "B"
    assert all(r["group"] != "X" for r in plan if r["wave"] in ("C1", "C2"))
    assert {r["class"] for r in plan if r["wave"] == "typed"} == {"standin:exec",
                                                                  "standin:compact"}
    assert all(r["june"] == "failed:timeout" or r["june"] == "unrun"
               for r in plan if r["wave"] == "C2")
    deferred = {r["zoo"] for r in plan if r["wave"] == "rung3-env"}
    assert "open-mmlab/mmdetection" in deferred and "detectron2" in deferred


def test_catalog_plan_imports_no_torch() -> None:
    code = ("import sys; from menagerie.pipeline import DataLayout, harvest_tiers; "
            "harvest_tiers.catalog_plan(data=DataLayout.from_env(), write=False); "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_groups_follow_the_june_recipe_text() -> None:
    assert group({"zoo": "base-pytorch-compact", "recipe": {"type": "statement",
                                                            "code": "model = f()"}}) == "X"
    assert group({"zoo": "timm", "recipe": {"type": "exec-string", "body": "exec('')"}}) == "X"
    classic = "from menagerie.classics.grapnet import build; model=build()"
    assert group({"zoo": "timm", "recipe": {"type": "statement", "code": classic}}) == "B"
    assert group({"zoo": "timm", "recipe": {"expr": "timm.create_model('resnet18')"}}) == "A"


def test_parse_model_call_finds_the_library_constructor() -> None:
    strict = parse_model_call("import segmentation_models_pytorch as smp; "
                              "model=smp.FPN(encoder_name='mit_b0', classes=21)")
    assert strict[:3] == ("segmentation_models_pytorch.FPN", [],
                          {"encoder_name": "mit_b0", "classes": 21})
    loose = parse_model_call("import builtins, torch, diffusers; torch.set_num_threads(1); "
                             "_m=diffusers.AutoencoderKL(layers_per_block=1); "
                             "model=builtins.type('W', (torch.nn.Module,), {})(_m)")
    assert loose[0] == "diffusers.AutoencoderKL" and loose[4]["_wrapped"]
    evaled = parse_model_call("from pytorchvideo.models.hub import x3d_xs; "
                              "model=x3d_xs(pretrained=False).eval()")
    assert evaled[:3] == ("pytorchvideo.models.hub.x3d_xs", [], {"pretrained": False})
    assert parse_model_call("import torch.nn as nn; model=type('W', (nn.Module,), {})()") is None
    assert parse_model_call("import torch; _C = 3; model=_C(4)") is None  # bound in the code
    assert parse_model_call("import transformers; cfg=transformers.GPT2Config(n_layer=2); "
                            "model=transformers.GPT2Model(cfg)") is None  # sizes, then a model
    star = parse_model_call("from torch.nn import *; model=Linear(4, 2)")
    assert star[:2] == ("torch.nn.Linear", [4, 2])


def test_parse_model_call_keeps_object_references() -> None:
    parsed = parse_model_call("from spikingjelly.activation_based.model.parametric_lif_net import "
                              "MNISTNet; from spikingjelly.activation_based import neuron; "
                              "model=MNISTNet(spiking_neuron=neuron.ParametricLIFNode)")
    assert parsed[2] == {"spiking_neuron": {
        "$ref": "spikingjelly.activation_based.neuron.ParametricLIFNode"}}
    assert _parse_call("import torch; model=torch.nn.ReLU(x=torch.nn.Tanh)") is None  # rung 0


@needs_libraries
def test_a_referenced_object_is_resolved_when_built() -> None:
    import torch

    from menagerie.pipeline.harvest import _builder

    recipe = {"kind": "library", "call": "torch.nn.TransformerEncoderLayer",
              "kwargs": {"d_model": 8, "nhead": 2,
                         "activation": {"$ref": "torch.nn.functional.gelu"}}}
    layer = _builder(recipe, None)()
    assert layer.activation is torch.nn.functional.gelu


def test_rerun_keeps_the_packets_search() -> None:
    old = read_json(Packet, DATA.packet("f-fpn-segmentation"))
    assert old.search.note
    kept = _carried(old, None, Search())["search"]
    assert kept.note == old.search.note and kept.places == old.search.places
    grown = _carried(old, None, Search(places=("other docstrings",)))["search"]
    assert grown.places == (*old.search.places, "other docstrings")


def test_stand_in_rows_are_typed_without_libraries() -> None:
    compact = next(cat for cat, _ in joined(DATA) if cat["zoo"] == "base-pytorch-compact")
    row = Row("m1", compact["name"], compact["zoo"], "", compact["family"], catalog=compact)
    draft = harvest_tiers.tier_draft(row, DATA)
    assert isinstance(draft, Draft) and draft.give_up[0] == "standin:compact"
    assert draft.recipe == {}


@needs_libraries
def test_library_calls_keep_defaults_selectors_and_task_configuration() -> None:
    rows = _rows("m9413", "m138", "m4851")
    smp = harvest_tiers.tier_draft(rows["m9413"], DATA)
    assert smp.give_up is None and smp.delta == {"classes": 21}  # in_channels 3: default
    assert smp.recipe["kwargs"]["encoder_name"] == "mit_b0" and smp.recipe["extras"] == []
    vae = harvest_tiers.tier_draft(rows["m138"], DATA)  # June shrank AutoencoderKL
    assert vae.give_up[0] == "standin:kwargs" and "block_out_channels" in vae.give_up[1]
    rcnn = harvest_tiers.tier_draft(rows["m4851"], DATA)  # weight switches, June's wrapper dropped
    assert rcnn.give_up is None
    assert (rcnn.recipe["call"], rcnn.recipe["args"]) == ("torchvision.models.get_model",
                                                          ["fasterrcnn_resnet50_fpn"])
    with pytest.raises(NoRecipe, match="aux_logits"):
        harvest_tiers.vet(rows["m4851"], "torchvision.models.get_model", ["googlenet"],
                          {"weights": None, "aux_logits": False})


@needs_libraries
def test_rows_whose_wrapper_made_the_inputs_need_an_input_adapter() -> None:
    rows = _rows("m3785", "m8562")
    for row in rows.values():  # clip zero-shot (June wrapper), raft_small (two images)
        assert harvest_tiers.tier_draft(row, DATA).give_up[0] == "needs_adapter:inputs"


@needs_libraries
def test_t5_gets_decoder_input_and_runs() -> None:
    import torch

    from menagerie.pipeline.harvest import construct, write_drafts
    from menagerie.pipeline.inputs import make_input

    draft = harvest_tiers.tier_draft(_rows("m10470")["m10470"], DATA)
    assert draft.recipe["config"] == {"model_type": "t5"}
    assert draft.recipe["input"].preset == "example:menagerie.pipeline.harvest:seq2seq_text"
    assert draft.recipe["input"].asset == "std-text-01"
    record = write_drafts([draft], data=DATA, write=False)[0]
    args, kwargs = make_input(record.recipe.input)
    assert set(kwargs) == {"input_ids", "decoder_input_ids"} and not args
    with torch.no_grad():
        out = construct(record.recipe).eval()(**kwargs)
    assert out.last_hidden_state.shape[:2] == (1, 16)


@pytest.mark.skipif(importlib.util.find_spec("monai") is None, reason="monai is an overlay")
def test_template_classes_take_the_docstring_example() -> None:
    draft = harvest_tiers.tier_draft(_rows("m7466")["m7466"], DATA)
    assert draft.give_up is None and "template" in draft.delta
    assert "docstring" in draft.delta["template"]
    assert draft.recipe["extras"] == list(harvest_tiers.OVERLAYS["monai"])


@needs_libraries
def test_harvest_tier_writes_records_and_rebuild_candidates(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = DataLayout(tmp_path / "data")
    shutil.copytree(FROZEN, data.root, ignore=shutil.ignore_patterns("README.md"))
    (data.root / "stable_ids.jsonl").symlink_to(DATA.root / "stable_ids.jsonl")
    before = data.model("m9427").read_bytes()
    records = {r.id: r for r in harvest_tiers.harvest_tier(
        "segmentation_models_pytorch-0.5.0", ["m9427", "m9413"], data=data)}
    # The rung 0 record would gain `delta.classes` at its revision: kept as committed, and named.
    assert set(records) == {"m9413"} and data.model("m9427").read_bytes() == before
    assert "kept m9427 as committed" in capsys.readouterr().err
    packet = read_json(Packet, data.packet(records["m9413"].family))
    assert "m9413" in {m.id for m in packet.members}
    harvest_tiers.harvest_tier("diffusers", ["m138"], data=data)
    with (data.root / "catalog" / "standin-kwargs.tsv").open(encoding="utf-8") as handle:
        listed = {r["id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    assert listed["m138"]["call"] == "diffusers.AutoencoderKL"
    assert float(listed["m138"]["default_gib"]) > 0
    assert read_json(ModelRecord, data.model("m138")).give_up.cls == "standin:kwargs"


@needs_libraries
def test_hub_custom_code_builds_on_meta_in_a_fresh_process(tmp_path: Path) -> None:
    snapshot = tmp_path / "tiny-hub"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "tinyhub", "hidden": 8,
        "auto_map": {"AutoConfig": "configuration_tiny.TinyConfig",
                     "AutoModel": "modeling_tiny.TinyModel"}}))
    (snapshot / "configuration_tiny.py").write_text(textwrap.dedent("""
        from transformers import PretrainedConfig
        class TinyConfig(PretrainedConfig):
            model_type = "tinyhub"
            def __init__(self, hidden=8, **kwargs):
                self.hidden = hidden
                super().__init__(**kwargs)
    """))
    (snapshot / "modeling_tiny.py").write_text(textwrap.dedent("""
        import torch
        from transformers import PreTrainedModel
        from .configuration_tiny import TinyConfig
        class TinyModel(PreTrainedModel):
            config_class = TinyConfig
            def __init__(self, config):
                super().__init__(config)
                self.lin = torch.nn.Linear(config.hidden, config.hidden)
            def forward(self, x):
                return self.lin(x)
    """))
    code = ("import torch; from menagerie.pipeline.harvest import hub_model\n"
            f"with torch.device('meta'):\n    m = hub_model({str(snapshot)!r})\n"
            "assert next(m.parameters()).device.type == 'meta', m")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=Path.cwd())
