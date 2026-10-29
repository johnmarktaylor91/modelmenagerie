"""The transformers class a roster row names, for a registry default built from a model type.

Rung 3 (r3-coverage follow-up, Crawler 2026-10-09): a registry default must build the model the
roster row names, not the bare backbone. A row names its class in one of three places, read in
this order: the zoo label (``transformers:AutoModelForCTC``), a task part of the name
(``udop:image-text-to-text``; a ``base-*`` part is the bare model), or June's own code
(``AutoModelForSeq2SeqLM``). A row that names none is the bare model (``AutoModel``). A named
class that is not a transformers class, a task with no known class, or a class with no model
for the registry's model type has no default: the row stays open for an agent. A June-era Auto
class the installed transformers renamed is built as its successor (``SUCCESSORS``).
"""

from __future__ import annotations

import re

# A ``name:<task>`` part as the transformers Auto class that builds that task's model.
TASK_HEADS = {
    "image-text-to-text": "AutoModelForImageTextToText",
    "vision-to-text": "AutoModelForVision2Seq",
    "image-classification": "AutoModelForImageClassification",
    "zero-shot-image-classification": "AutoModelForZeroShotImageClassification",
    "object-detection": "AutoModelForObjectDetection",
    "zero-shot-object-detection": "AutoModelForZeroShotObjectDetection",
    "semantic-segmentation": "AutoModelForSemanticSegmentation",
    "universal-segmentation": "AutoModelForUniversalSegmentation",
    "depth-estimation": "AutoModelForDepthEstimation",
    "visual-question-answering": "AutoModelForVisualQuestionAnswering",
    "document-question-answering": "AutoModelForDocumentQuestionAnswering",
    "audio-classification": "AutoModelForAudioClassification",
    "audio-frame-classification": "AutoModelForAudioFrameClassification",
    "speaker-xvector": "AutoModelForAudioXVector",
    "ctc-asr": "AutoModelForCTC",
    "speech-seq2seq": "AutoModelForSpeechSeq2Seq",
    "text-to-waveform": "AutoModelForTextToWaveform",
    "text-to-spectrogram": "AutoModelForTextToSpectrogram",
    "mask-generation": "AutoModelForMaskGeneration",
}
CLASS = re.compile(r"[A-Z][A-Za-z0-9_]*")
# Model classes June's code names: Auto model classes and concrete ``...For<Task>`` heads.
JUNE_CLASSES = re.compile(r"\b(AutoModel(?:For[A-Za-z0-9]+)?|[A-Z][A-Za-z0-9]*For[A-Z][A-Za-z0-9]+)\b")


def named_head(name: str, zoo: str, june_code: str) -> tuple[str | None, str]:
    """The class the row names and where it is named; ``None`` with the reason when no default
    can be that class."""

    label = zoo.partition(":")[2]
    if label:
        if CLASS.fullmatch(label):
            return label, f"zoo label {label}"
        return None, f"zoo label {label!r} is not a transformers class"
    part = name.partition(":")[2]
    if part.startswith("base-"):
        return "AutoModel", f"name part {part}"
    if part in TASK_HEADS:
        return TASK_HEADS[part], f"name part {part}"
    if part and "-" in part:
        return None, f"name part {part!r} is a task with no known transformers class"
    found = sorted(set(JUNE_CLASSES.findall(june_code)))
    if len(found) == 1:
        return found[0], f"June code {found[0]}"
    if len(found) > 1:
        return None, f"June code names several model classes ({', '.join(found)})"
    return "AutoModel", "no class named: the bare model"


# Auto classes a later transformers renamed: the row's June-era name -> the class that took its
# role (transformers 5 removed AutoModelForVision2Seq for AutoModelForImageTextToText).
SUCCESSORS = {"AutoModelForVision2Seq": "AutoModelForImageTextToText"}


def installed(head: str) -> str:
    """``head``, or its successor when the installed transformers no longer has it."""

    import transformers

    if hasattr(transformers, head) or head not in SUCCESSORS:
        return head
    return SUCCESSORS[head]


def built_class(head: str, model_type: str) -> str:
    """The class ``head`` builds for ``model_type``'s default config; ``LookupError`` if none."""

    import transformers
    from transformers import CONFIG_MAPPING

    cls = getattr(transformers, head, None)
    if cls is None:
        raise LookupError(f"transformers has no {head}")
    config_class = CONFIG_MAPPING[model_type]  # a KeyError is a LookupError
    mapping = getattr(cls, "_model_mapping", None)
    if mapping is None:  # a concrete class names its own config
        if getattr(cls, "config_class", None) is not config_class:
            raise LookupError(f"{head} is not a {model_type} model")
        return cls.__name__
    if config_class not in mapping:
        raise LookupError(f"{head} has no model for {model_type}")
    try:
        return mapping[config_class].__name__
    except Exception as exc:  # a lazily imported model module that does not import here
        raise LookupError(f"{head} for {model_type} does not import: {type(exc).__name__}") from exc


def call_for(head: str) -> str:
    """The recipe call that builds ``head`` from a config."""

    return f"transformers.{head}.from_config" if head.startswith("Auto") else f"transformers.{head}"
