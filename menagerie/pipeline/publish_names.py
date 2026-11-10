"""Curated display names and small text cleanups for the site export.

A raw identifier stays the fallback: only names listed here change. This is a short list of
well-known models and families, not a general prettifier.
"""

from __future__ import annotations

import re

# Exported model slug to the model's usual written name.
MODEL_NAMES = {
    "alexnet": "AlexNet",
    "gpt2": "GPT-2",
    "gpt2-huggingface": "GPT-2",
    "lenet5": "LeNet-5",
    "lenet5-classic": "LeNet-5",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
    "vgg11": "VGG-11",
    "vgg13": "VGG-13",
    "vgg16": "VGG-16",
    "vgg19": "VGG-19",
    "wavenet": "WaveNet",
}

# Family name (``family_normalized``) to its usual written name, for families with five or more
# exported members whose name is a known acronym or coined spelling.
FAMILY_NAMES = {
    "beit": "BEiT", "bert": "BERT", "coatnet": "CoAtNet", "convnext": "ConvNeXt",
    "convnextv2": "ConvNeXt V2", "crossvit": "CrossViT", "davit": "DaViT",
    "deeplabv3": "DeepLabV3", "deit": "DeiT", "deit3": "DeiT III", "densenet": "DenseNet",
    "dpt": "DPT", "ecaresnet": "ECA-ResNet", "edsr": "EDSR", "efficientdet": "EfficientDet",
    "efficientnet": "EfficientNet", "efficientnetv2": "EfficientNetV2",
    "efficientvit": "EfficientViT", "eva": "EVA", "eva02": "EVA-02", "fastvit": "FastViT",
    "focalnet": "FocalNet", "gcvit": "GC ViT", "graphsage": "GraphSAGE", "hrnet": "HRNet",
    "hubert": "HuBERT", "levit": "LeViT", "linknet": "LinkNet", "manet": "MA-Net",
    "mask2former": "Mask2Former", "maxvit": "MaxViT", "mixnet": "MixNet", "mlp_mixer": "MLP-Mixer",
    "mnasnet": "MnasNet", "mobilenet": "MobileNet", "mobilenetv2": "MobileNetV2",
    "mobilenetv3": "MobileNetV3", "mobilenetv4": "MobileNetV4", "mobilevit": "MobileViT",
    "mobilevitv2": "MobileViTv2", "mvitv2": "MViTv2", "nfnet": "NFNet", "pit": "PiT",
    "poolformer": "PoolFormer", "pvt": "PVT", "regnetx": "RegNetX", "regnety": "RegNetY",
    "regnetz": "RegNetZ", "repvgg": "RepVGG", "repvit": "RepViT", "res2net": "Res2Net",
    "resnest": "ResNeSt", "resnet": "ResNet", "resnetv2": "ResNetV2", "resnext": "ResNeXt",
    "segformer": "SegFormer", "seresnet": "SE-ResNet", "seresnext": "SE-ResNeXt", "swin": "Swin",
    "swinv2": "Swin V2", "unet++": "UNet++", "vgg": "VGG", "vit": "ViT", "volo": "VOLO",
    "wavlm": "WavLM", "xception": "Xception", "xcit": "XCiT", "yolov5": "YOLOv5",
    "yolov6": "YOLOv6", "yolov8": "YOLOv8", "yolov9": "YOLOv9",
}

# A trailing parenthetical note on a zoo label: "torch.nn (hand-build)", "act(repo,owner)".
_ZOO_NOTE = re.compile(r"\s*\([^()]*\)\s*$")
_SEPARATORS = " -,;/|"
_MIN_REPEAT = 8
# A comma followed by one of these is a name suffix, not "Family, Given".
_NAME_SUFFIXES = frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "ph.d."})


def model_name(slug: str, raw: str) -> str:
    """The curated name for an exported model's slug, else its raw name."""

    return MODEL_NAMES.get(slug, raw)


def family_name(family: str) -> str:
    """The curated name for a family, else the family name itself."""

    return FAMILY_NAMES.get(family) or FAMILY_NAMES.get(family.lower(), family)


def bare_zoo(value: str | None) -> str | None:
    """Drop a trailing parenthetical note from a zoo label; keep the label otherwise."""

    if not value:
        return value
    return _ZOO_NOTE.sub("", value) or value


def collapse_repeat(value: str | None) -> str | None:
    """Collapse a string made of one half written twice ("A - A", "A/A", "AA") to that half."""

    if not value:
        return value
    text = value.strip()
    n = len(text)
    for k in range(n // 2, _MIN_REPEAT - 1, -1):
        half = text[:k]
        if text.endswith(half) and not text[k:n - k].strip(_SEPARATORS):
            return half.strip(_SEPARATORS + " ") or value
    return value


def author_names(authors: tuple[str, ...] | list[str]) -> list[str]:
    """Authors as "Given Family": a "Family, Given" entry is turned round, and an entry with
    no letter or digit in it (a stray ":" between a team name and its members) is dropped."""

    return [_given_first(a.strip()) for a in authors if any(ch.isalnum() for ch in a)]


def _given_first(name: str) -> str:
    family, comma, given = name.partition(", ")
    if not comma or not given or ", " in given or given.lower() in _NAME_SUFFIXES:
        return name
    return f"{given} {family}"
