"""A row name looked up in the installed model registries (r3-coverage, rung 3)."""

import importlib.util
import re

import pytest

from menagerie.pipeline import registry_names


@pytest.mark.parametrize(("name", "zoo", "expected"), [
    ("monai.BasicUNetPlusPlus", "monai.networks.nets", {"monai.basicunetplusplus",
                                                        "basicunetplusplus"}),
    ("ganzoo_dcgan_discriminator", "ganzoo (pip pytorch-gan-zoo)",
     {"ganzoo_dcgan_discriminator", "dcgan_discriminator"}),
    ("qwen2_moe_hf", "transformers", {"qwen2_moe_hf", "qwen2_moe"}),
    ("udop:base-document-vlm", "base-pytorch-compact", {"udop:base_document_vlm", "udop"}),
    ("midas:MiDaS_small", "base-pytorch-compact", {"midas:midas_small"}),  # a namespace
    ("mmdet:convnext", "mmdet", {"mmdet:convnext"}),
    ("yoloe-26s-seg", "ultralytics", {"yoloe_26s_seg"}),
])
def test_candidates_drop_prefixes_and_suffixes(name: str, zoo: str, expected: set) -> None:
    assert set(registry_names.candidates(name, zoo)) == expected


@pytest.mark.parametrize(("stem", "name", "hit"), [
    ("yoloe_26_seg", "yoloe_26s_seg", True), ("yolo11", "yolo11n", True),
    ("yolo11", "yolo11", True), ("yolov8_seg", "yolov8x_seg", True),
    ("yolo11", "yolo11q", False), ("yoloe_26_seg", "yoloe_26s", False),
])
def test_ultralytics_stems_take_a_scale_letter(stem: str, name: str, hit: bool) -> None:
    assert bool(re.fullmatch(registry_names._scaled(stem), name)) is hit


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="needs torchvision")
def test_the_zoo_library_is_tried_first() -> None:
    assert registry_names.lookup("resnet18", "torchvision")[:2] == (
        "torchvision.models.get_model", ["resnet18"])
    if importlib.util.find_spec("timm") is not None:
        assert registry_names.lookup("resnet18", "timm")[:2] == ("timm.create_model", ["resnet18"])
    assert registry_names.lookup("no_such_model_r3cov", "torchvision") is None


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_a_transformers_model_type_is_its_default_config() -> None:
    found = registry_names.lookup("qwen2_moe_hf", "discovered-pytorch")
    assert found[0] == "transformers.AutoModel.from_config"
    assert found[3] == {"model_type": "qwen2_moe"}


@pytest.mark.skipif(importlib.util.find_spec("diffusers") is None, reason="needs diffusers")
def test_a_diffusers_model_class_is_its_default() -> None:
    assert registry_names.lookup("UNet2DModel", "diffusers")[0] == "diffusers.UNet2DModel"


def test_a_namespace_before_the_colon_is_never_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """``zoedepth:ZoeD_N`` must not build transformers' default ``zoedepth`` config."""

    monkeypatch.setattr(registry_names, "REGISTRIES",
                        (("re", lambda names: ("lib.Model", [names[0]], {}, {}, {})
                          if "zoedepth" in names else None),))
    assert registry_names.lookup("zoedepth:ZoeD_N", "base-pytorch-compact") is None
    assert registry_names.lookup("zoedepth:depth-estimation", "base-pytorch-compact") is not None


@pytest.mark.parametrize(("stem", "name", "model"), [
    ("yolov8_seg", "yolov8x_seg", "SegmentationModel"), ("yolov8_pose", "yolov8n_pose", "PoseModel"),
    ("yolo11_cls", "yolo11s_cls", "ClassificationModel"), ("yolo11", "yolo11n", "DetectionModel"),
])
def test_an_ultralytics_task_comes_from_the_stem(monkeypatch: pytest.MonkeyPatch, stem: str,
                                                 name: str, model: str) -> None:
    monkeypatch.setattr(registry_names, "_yaml_stems", lambda: (stem,))
    assert registry_names._ultralytics([name])[0] == f"ultralytics.nn.tasks.{model}"
