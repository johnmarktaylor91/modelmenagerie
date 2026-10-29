"""The transformers class a roster row names (r3-coverage follow-up, rung 3)."""

import importlib.util

import pytest

from menagerie.pipeline import transformers_heads as heads


@pytest.mark.parametrize(("name", "zoo", "code", "head"), [
    ("wav2vec2", "transformers:AutoModelForCTC", "", "AutoModelForCTC"),
    ("udop:image-text-to-text", "base-pytorch-compact", "", "AutoModelForImageTextToText"),
    ("video_llava:vision-to-text", "base-pytorch-compact", "", "AutoModelForVision2Seq"),
    ("vilt:base-vision-language-fusion", "base-pytorch-compact", "", "AutoModel"),
    ("umt5", "huggingface_transformers",
     "from transformers import AutoConfig, AutoModelForSeq2SeqLM", "AutoModelForSeq2SeqLM"),
    ("qwen2_moe_hf", "transformers", "", "AutoModel"),
    ("detr", "huggingface", "m = transformers.DetrForObjectDetection(cfg)", "DetrForObjectDetection"),
])
def test_the_named_head(name: str, zoo: str, code: str, head: str) -> None:
    assert heads.named_head(name, zoo, code)[0] == head


@pytest.mark.parametrize(("name", "zoo", "code"), [
    ("x", "transformers:trust_remote_code", ""),
    ("x", "roboflow:roboflow/rf-detr", ""),
    ("x:frontier-segmentation", "base-pytorch-compact", ""),
    ("x", "huggingface", "AutoModelForCTC ... AutoModelForAudioClassification"),
])
def test_no_head_when_the_row_names_none_a_default_can_be(name: str, zoo: str, code: str) -> None:
    head, why = heads.named_head(name, zoo, code)
    assert head is None and why


def test_calls() -> None:
    assert heads.call_for("AutoModelForCTC") == "transformers.AutoModelForCTC.from_config"
    assert heads.call_for("DetrForObjectDetection") == "transformers.DetrForObjectDetection"


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_the_built_class_comes_from_the_auto_mapping() -> None:
    assert heads.built_class("AutoModelForCTC", "wav2vec2") == "Wav2Vec2ForCTC"
    assert heads.built_class("AutoModel", "bert") == "BertModel"
    assert heads.built_class("BertForMaskedLM", "bert") == "BertForMaskedLM"
    for head, kind in [("AutoModelForCTC", "bert"), ("NoSuchAutoClass", "bert"),
                       ("BertForMaskedLM", "gpt2"), ("AutoModel", "no_such_type")]:
        with pytest.raises(LookupError):
            heads.built_class(head, kind)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_a_renamed_auto_class_is_built_as_its_successor(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    assert heads.installed("AutoModelForCTC") == "AutoModelForCTC"
    monkeypatch.delattr(transformers, "AutoModelForVision2Seq", raising=False)
    assert heads.installed("AutoModelForVision2Seq") == "AutoModelForImageTextToText"
    assert heads.built_class("AutoModelForImageTextToText", "llava").startswith("Llava")

