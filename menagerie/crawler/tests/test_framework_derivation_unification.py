"""EN-1 -- ONE framework derivation, token-boundary ``tf``, positive torch-zoo veto.

200a shipped TWO divergent framework derivations: ``cli._framework`` (full-word
markers, used by ``plan`` and the routed-intake closure) and
``constants.framework_from_markers`` (bare ``"tf"`` substring, used by the driver).
The launch census, the ``--allow-lockless`` justification, and the readiness seal all
consulted the weak one, so m10502 (``tf_efficientnet_b5``, zoo ``timm``, a PyTorch
model) was driver-routed to lockless ``tf-keras-arm64`` and died
``CommittedLockFamilyError`` after buying a full paid author pipeline, and m4714
(``escnn_QuotientFourierELU`` -- ``tf`` inside "QuotientFourier") carried a lockless
env binding through its whole lifecycle. Measured blast radius before the fix:
105/8,533 master-catalog rows and 222/28,482 roster rows where plan and driver
disagreed (~2.4 expected hits per 200-member draw).

The remedy is structural, not tolerancing: exactly one derivation exists
(``framework_from_markers``), the ``tf`` marker matches tokens only, and a closed
positive torch-zoo veto keeps provably torch-native zoos (timm's ``tf_*`` = "ported
from TF weights" naming convention) out of the native tail entirely.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from menagerie.crawler.constants import (
    TORCH_NATIVE_ZOO_VETO,
    framework_from_markers,
)

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

#: Zoos adjudicated GENUINELY native-framework in the EN-1 pre-flip census over both
#: corpora (2026-08-24): every row still routed to the native tail by the unified
#: derivation belongs to one of these zoos. A native-routed row from any OTHER zoo is
#: a new misroute-class instance and must be adjudicated, never silently routed.
_ADJUDICATED_NATIVE_ZOOS = frozenset(
    {
        "discovered-tensorflow",
        "discovered-jax",
        "discovered-keras",
        "discovered-paddle",
        "paddleclas-configs",
        "paddleocr-configs",
        "paddleocr (web; paddlepaddle source-catalog)",
    }
)


@pytest.mark.smoke
def test_m10502_timm_tf_prefixed_name_stays_pytorch() -> None:
    """The production misroute: zoo timm is torch-only; ``tf_`` names are TF-ported weights."""

    assert framework_from_markers("timm", "tf_efficientnet_b5") == "pytorch"
    assert framework_from_markers("timm", "maxvit_base_tf_224") == "pytorch"
    assert framework_from_markers("effdet", "effdet_tf_efficientdet_d0") == "pytorch"


@pytest.mark.smoke
def test_m4714_escnn_quotientfourier_stays_pytorch() -> None:
    """The second production hit: ``tf`` inside "QuotientFourier" is not a marker."""

    assert framework_from_markers("escnn", "escnn_QuotientFourierELU") == "pytorch"


@pytest.mark.smoke
def test_tf_marker_matches_tokens_never_substrings() -> None:
    """``tf`` fires only as a delimited token, so word-internal hits are dead."""

    # Word-internal false positives from the pre-flip census, all provably torch.
    for zoo, name in (
        ("timm", "efficientformer_l1"),
        ("open-mmlab/mmsegmentation", "mmseg_fastfcn"),
        ("nerfstudio", "nerfstudio_splatfacto"),
        ("RobotFlamingo", "roboflamingo"),
        ("neuralop", "tfno"),
        ("nvidia-nemo", "nemo_sortformer_diarizer"),
        ("espnet (web; not installed)", "espnet_tfgridnet_separator"),
        ("arxiv(2411.19722 Google)", "JetFormer"),
        ("transformers:AutoModelForImageClassification", "swiftformer:image-classification"),
        ("diffusers ShapEPipeline (pip diffusers; INSTALLED)", "ShapE_MLPNeRSTFModel"),
    ):
        assert framework_from_markers(zoo, name) == "pytorch", (zoo, name)
    # Genuine delimited tokens still route native.
    assert framework_from_markers("google", "vision model tf") == "tensorflow"
    assert framework_from_markers("discovered-tf", "model") == "tensorflow"
    assert framework_from_markers("google", "tf_keras_efficientnet") == "tensorflow"


@pytest.mark.smoke
def test_torch_native_zoo_veto_outranks_name_markers() -> None:
    """A row from a provably torch-native zoo can never reach the native tail."""

    for zoo in ("timm", "effdet", "escnn", "nerfstudio", "github:NVlabs/stylegan"):
        assert framework_from_markers(zoo, "tf_anything_tensorflow_keras") == "pytorch", zoo
    # The veto is an exact zoo match: it can never eat a genuine native zoo.
    assert framework_from_markers("discovered-tensorflow", "movinet") == "tensorflow"
    assert framework_from_markers("discovered-jax", "vit") == "jax"
    for zoo in TORCH_NATIVE_ZOO_VETO:
        assert zoo == zoo.strip().lower(), f"veto entries are normalized: {zoo!r}"


@pytest.mark.smoke
def test_cli_has_no_second_framework_derivation() -> None:
    """EN-1's structural census: exactly one framework derivation is reachable.

    ``cli._framework`` was the divergent plan-side derivation; it is deleted, and no
    module-level function named ``_framework`` may reappear in cli.py.
    """

    from menagerie.crawler import cli

    assert not hasattr(cli, "_framework")
    tree = ast.parse(inspect.getsource(cli))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_framework" not in defined
    assert "framework_from_markers" in inspect.getsource(cli)


def _corpus_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


@pytest.mark.parametrize(
    "corpus",
    ("master_catalog.jsonl", "crawl_roster.jsonl"),
)
def test_native_routed_rows_come_only_from_adjudicated_native_zoos(corpus: str) -> None:
    """Whole-corpus reroute pin: the unified derivation's native set is closed.

    Every row the unified derivation routes native belongs to a zoo the EN-1 census
    adjudicated as genuinely native-framework. This is the permanent form of the
    pre-flip per-row adjudication: a torch-zoo row landing native again (a new alias,
    a veto regression, a re-forked derivation) fails here by zoo name.
    """

    offenders: list[tuple[str, str, str]] = []
    for row in _corpus_rows(_DATA_ROOT / corpus):
        zoo = str(row.get("zoo", ""))
        name = str(row.get("name", ""))
        derived = framework_from_markers(zoo, name)
        if derived != "pytorch" and zoo.strip().lower() not in _ADJUDICATED_NATIVE_ZOOS:
            offenders.append((name, zoo, derived))
    assert offenders == []


def test_named_production_misroutes_rederive_pytorch_in_the_corpora() -> None:
    """m10502 and m4714 exist in the corpora and rederive to the torch lane."""

    seen: dict[str, str] = {}
    for row in _corpus_rows(_DATA_ROOT / "master_catalog.jsonl"):
        name = str(row.get("name", ""))
        if name in {"tf_efficientnet_b5", "escnn_QuotientFourierELU"}:
            seen[name] = framework_from_markers(str(row.get("zoo", "")), name)
    assert seen == {
        "tf_efficientnet_b5": "pytorch",
        "escnn_QuotientFourierELU": "pytorch",
    }
