"""``hash_harness``: the case rules on recorded hashes, then a real model through the children.

The rules (what each case expects, the input-dependent-topology exception, the pass rule) are
checked on hand-made probe results; the variants, the kernel growth and the constructed cases on
real traces; one timm model goes through ``harness`` end to end, child processes included.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import DataLayout, hash_harness, write_json
from menagerie.pipeline.records import ModelRecord, RecipeInput, from_dict

FIXTURES = Path(__file__).parent / "fixtures"


def _variant(skeleton: str = "s", architecture: str = "a", model: str = "m", n: int = 3) -> dict:
    return {"hashes": {"skeleton": skeleton, "architecture": architecture, "model": model},
            "n_ops": n, "payload": [{"func": "conv2d", "args": [n]}]}


def _variants(**changed: dict) -> dict[str, Any]:
    base = {name: _variant() for name in ("default", "again", "fresh", "batch", "resolution",
                                          "meta")}
    base.update(wrap_a=_variant(model="ma"), wrap_b=_variant(model="mb"),
                kernel=_variant(architecture="a2", model="m2"))
    return {**base, **changed}


def test_every_case_passes_on_hashes_that_behave() -> None:
    cases = [hash_harness.compare(kind, "m1", _variants()) for kind in hash_harness.PAIRS]
    assert [c["case"] for c in cases if not c["ok"]] == []
    assert next(c for c in cases if c["case"] == "structure_only")["gates"] is False


@pytest.mark.parametrize(("kind", "changed", "level"), [
    ("same_process", {"again": _variant(model="other")}, "model"),
    ("fresh_process", {"fresh": _variant(skeleton="s2", architecture="a3")}, "skeleton"),
    ("wrapper", {"wrap_b": _variant(model="ma")}, "model"),
    ("wrapper", {"wrap_b": _variant(architecture="a9", model="mb")}, "architecture"),
    ("batch", {"batch": _variant(architecture="a4", n=3)}, "architecture"),
    ("kernel", {"kernel": _variant(model="m2")}, "architecture"),
])
def test_a_broken_level_fails_its_case(kind: str, changed: dict, level: str) -> None:
    case = hash_harness.compare(kind, "m1", _variants(**changed))
    assert (case["ok"], case["hash"]) == (False, level)


def test_an_invariance_failure_names_the_first_differing_op() -> None:
    batch = _variant(architecture="a4")
    batch["payload"] = [{"func": "conv2d", "args": [6]}]
    case = hash_harness.compare("resolution", "m1", _variants(resolution=batch))
    assert not case["ok"] and "op 0 (conv2d)" in case["why"] and '"args": [[3], [6]]' in case["why"]


def test_a_topology_that_follows_the_input_is_an_exception_not_a_collision() -> None:
    longer = _variant(skeleton="s2", architecture="a5", n=9)
    case = hash_harness.compare("resolution", "m1", _variants(resolution=longer))
    assert case["ok"] and case["exception"] and "3 ops, then 9" in case["why"]


def test_a_refused_variant_is_skipped_with_its_reason() -> None:
    case = hash_harness.compare("resolution", "m1", _variants(resolution={"skipped": "fixed size"}))
    assert case["ok"] and case["skipped"] and "fixed size" in case["why"]


def test_the_pass_rule_needs_every_gating_kind_checked_often_enough() -> None:
    ok = [{"case": kind, "ok": True, "gates": True} for kind in hash_harness.GATING]
    assert not hash_harness.passed(ok)
    enough = ok * hash_harness.MIN_CHECKED
    assert hash_harness.passed(enough)
    assert hash_harness.passed([*enough, {"case": "heads", "ok": False, "gates": False}])
    assert not hash_harness.passed([*enough, {"case": "width", "ok": False, "gates": True}])
    skipped = [{**c, "skipped": True} for c in enough if c["case"] == "kernel"]
    assert not hash_harness.passed([c for c in enough if c["case"] != "kernel"] + skipped)


@pytest.mark.parametrize(("preset", "shape", "name", "expected"), [
    ("image", (1, 3, 224, 224), "batch", (2, 3, 224, 224)),
    ("image", (1, 3, 224, 224), "resolution", (1, 3, 448, 448)),
    ("text", (1, 16), "resolution", (1, 32)),
    ("randn", (1, 4, 1, 32, 32), "resolution", (1, 4, 1, 64, 64)),
    ("randn", (1, 128), "resolution", None),
    ("example:m:f", None, "batch", None),
])
def test_varied_shapes(preset: str, shape: Any, name: str, expected: Any) -> None:
    spec = RecipeInput(preset=preset, kind="random-fallback", shape=shape)
    varied = hash_harness._varied_shape(spec, name)
    assert varied == expected if expected else isinstance(varied, str)


def test_headline_basis_follows_the_committed_verdict(tmp_path: Path) -> None:
    data = DataLayout(tmp_path)
    legacy = {"basis": "graph_shape_hash", "schema": None, "backends": []}
    assert hash_harness.headline_basis(data) == legacy
    path = tmp_path / hash_harness.VERDICT_FILE
    path.write_text(json.dumps({"passed": False, "hash_schema": 1}), encoding="utf-8")
    assert hash_harness.headline_basis(data) == legacy
    path.write_text(json.dumps({"passed": True, "hash_schema": 2, "backends": ["torch"]}),
                    encoding="utf-8")
    basis = hash_harness.headline_basis(data)
    assert basis == {"basis": "architecture", "schema": 2, "backends": ["torch"]}
    hashes = {"architecture": "a1", "schema": 2}
    assert hash_harness.dedup_key(hashes, "g1", basis) == "a1"
    assert hash_harness.dedup_key({**hashes, "schema": 1}, "g1", basis) == ""
    assert hash_harness.dedup_key(hashes, None, basis, "jax") == ""  # previews are not checked
    assert hash_harness.dedup_key(hashes, "g1", legacy) == "g1"


def test_the_committed_verdict_passes_and_names_its_backends() -> None:
    verdict = json.loads((Path(hash_harness.__file__).parents[1] / "data" /
                          hash_harness.VERDICT_FILE).read_text(encoding="utf-8"))
    assert verdict["passed"] is True and verdict["backends"] == ["torch"]
    assert verdict["models"] == len(hash_harness.HARNESS_IDS) == 30


# --- real traces ---------------------------------------------------------------------


def test_kernel_growth_keeps_the_topology_and_changes_the_architecture() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchlens")
    from torch import nn

    from menagerie.pipeline import worker

    def net() -> Any:
        torch.manual_seed(0)
        return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 1)).eval()

    x = (torch.ones(1, 3, 8, 8),)
    plain, grown = net(), net()
    assert hash_harness._grow_first_kernel(grown) and grown[0].kernel_size == (5, 5)
    assert grown(*x).shape == plain(*x).shape
    a = hash_harness._hash_subject(worker.Subject(plain, x, {}))
    b = hash_harness._hash_subject(worker.Subject(grown, x, {}))
    assert a["hashes"]["skeleton"] == b["hashes"]["skeleton"]
    assert a["hashes"]["architecture"] != b["hashes"]["architecture"]
    assert not hash_harness._grow_first_kernel(nn.Sequential(nn.Conv2d(3, 4, 2)))


def test_constructed_cases() -> None:
    pytest.importorskip("torchlens")
    cases = {c["case"]: c for c in hash_harness.constructed_cases()}
    assert {k for k, c in cases.items() if c["gates"] and not c["ok"]} == set()
    assert set(cases) == {"organisation", "width_mlp", "pool_kernel", "heads"}


def test_harness_end_to_end_on_one_timm_model(tmp_path: Path) -> None:
    """resnet10t at 64 px through both children: every gating case checked and passing."""

    pytest.importorskip("timm")
    pytest.importorskip("torchlens")
    raw = json.loads((FIXTURES / "m8839.model.json").read_text(encoding="utf-8"))
    raw["recipe"].update(args=["resnet10t"])
    raw["recipe"]["input"].update(shape=[1, 3, 64, 64])
    data = DataLayout(tmp_path / "data")
    write_json(from_dict(ModelRecord, raw), data.model("m8839"))
    report = hash_harness.harness(["m8839"], data=data, out=tmp_path / "out")
    by_case = {c["case"]: c for c in report["cases"] if c["id"] == "m8839"}
    assert set(by_case) == set(hash_harness.PAIRS)
    assert [k for k, c in by_case.items() if not c["ok"] or c.get("skipped")] == []
    assert report["passed"] is False  # one model is under MIN_CHECKED
    verdict = json.loads((tmp_path / "out" / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["models"] == 1 and verdict["counts"]["kernel"] == {"ok": 1}
    assert verdict["hash_schema"] and verdict["torchlens"]


_STORED_GELU = """
import torch, torchlens as tl, transformers
from menagerie.pipeline import worker

if WRAP:
    worker._wrap_torch()
config = transformers.BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=64, vocab_size=64)
torch.manual_seed(0)
model = transformers.BertModel(config).eval()  # its GELUActivation keeps F.gelu
trace = tl.trace(model, torch.ones(1, 8, dtype=torch.long))
print(sorted({layer.layer_type for layer in trace.layer_list}))
"""


@pytest.mark.parametrize("wrap", [True, False])
def test_a_build_before_the_first_capture_keeps_its_stored_torch_functions(wrap: bool) -> None:
    """Without ``_wrap_torch`` the first capture in a process loses BERT's stored ``F.gelu`` (an
    ``internalsource`` takes its place; TorchLens 2.35.3). Recheck the control when the pin moves."""

    pytest.importorskip("torchlens")
    pytest.importorskip("transformers")
    import subprocess
    import sys

    code = _STORED_GELU.replace("if WRAP:", f"if {wrap}:")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True, timeout=300)
    kinds = done.stdout.strip().splitlines()[-1]
    from packaging.version import Version
    import torchlens as tl

    rescued = wrap or Version(tl.__version__) >= Version("2.36.0")
    assert ("'gelu'" in kinds, "'internalsource'" in kinds) == (rescued, not rescued)


def test_summary_tolerates_the_collections_a_preview_trace_leaves_unset() -> None:
    """A ``function_root`` preview trace (JAX, TensorFlow, Paddle) has no ``buffers``; the summary
    reads every trace collection as empty when it is unset."""

    torch = pytest.importorskip("torch")
    tl = pytest.importorskip("torchlens")
    from menagerie import trace_summary

    class Unset:
        def __init__(self, trace: Any) -> None:
            self._trace = trace

        def __getattr__(self, name: str) -> Any:
            if name in ("buffers", "params", "modules", "ops", "output_ops"):
                return None
            return getattr(self._trace, name)

    trace = tl.trace(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()).eval(),
                     torch.ones(1, 4))
    row = trace_summary.summarize_trace("m1", Unset(trace), "r",
                                        hashes=trace_summary.hash_family(trace))
    assert row["buffer_memory_bytes"] == 0 and row["architecture_hash"]
