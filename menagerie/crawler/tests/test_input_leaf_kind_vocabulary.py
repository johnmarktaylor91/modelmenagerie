"""``input_contract`` leaf ``kind`` is closed because the executor's vocabulary is.

``worker._materialize_declarative_call`` materializes exactly two leaf kinds:
``tensor``, allocated from ``shape``/``dtype``/``distribution``, and
``constructed``, built through the declarative constructor grammar. Every other
value is refused there and nowhere earlier.

The schema, however, published ``kind`` as an open ``nonempty_string``. So a
proposal could name a perfectly descriptive kind, validate, pass the independent
checker, be admitted, and only then die in the worker -- after the single
authoring visit this catalog gives a model. The rung-3 census spent ``m7362``
(``timm`` ``mobileone_s3``, an entirely ordinary R1 image model) exactly that way,
four times across its train and eval forwards, on the sole defect that its image
leaf said ``kind: "standard-image-tensor"`` instead of ``kind: "tensor"``. The
refusal text -- "contains a non-tensor leaf" -- described the leaf as something it
was not.

These tests pin the closure with the REAL archived contract that failed, in the
failing direction:

- the authored value still refuses at execution, and now names itself;
- the SAME contract with the one word corrected materializes the real tensor;
- the schema now refuses the authored value up front, where a refusal is cheap;
- the ``constructed`` arm, which the census's DGL contract used correctly, keeps
  working -- the closure is a vocabulary fix, not a narrowing of expressiveness.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import AUTHOR_PROPOSAL_SCHEMA_VERSION_V3, RunMode
from menagerie.crawler.proposal import (
    ProposalValidationError,
    _validate_scalar_leaf_kinds,
    validate_author_proposal,
)
from menagerie.crawler.schema import PayloadValidationError, get_validator, validate_payload
from menagerie.crawler.tests.conftest import make_author_proposal
from menagerie.crawler.worker import WorkerRequest, _materialize_declarative_call

# The exact leaf ``m7362`` published, and the exact leaf it needed to publish.
_AUTHORED_LEAF: dict[str, Any] = {
    "path": "args[0]",
    "kind": "standard-image-tensor",
    "semantic_role": (
        "Batch of RGB images to classify, in NCHW layout; values are normalized image intensities."
    ),
    "shape": [1, 3, 224, 224],
    "dtype": "float32",
    "device_policy": "cpu",
    "distribution": "normal",
    "constraints": ["batch dimension is free and set to 1"],
    "source_evidence_ids": ["ev-timm-forward"],
}

_CONSTRUCTED_LEAF: dict[str, Any] = {
    "path": "args[0]",
    "kind": "constructed",
    "semantic_role": "homogeneous DGLGraph supplying the connectivity",
    "shape": [6, 6],
    "dtype": "int64",
    "device_policy": "cpu",
    "distribution": "constructor",
    "constraints": ["must be homogeneous"],
    "source_evidence_ids": ["ev-layer-ctor"],
    "constructor": {"module": "dgl", "symbol": "graph", "kwargs": {"data": [[0, 1]]}},
}


def _contract(leaf: dict[str, Any]) -> dict[str, Any]:
    """Wrap one leaf in a minimal complete declarative input contract.

    Parameters
    ----------
    leaf:
        Single positional ``input_leaf``.

    Returns
    -------
    dict[str, Any]
        Contract shaped as ``_materialize_declarative_call`` consumes it.
    """

    return {"args": [deepcopy(leaf)], "kwargs": [], "non_tensor_values": []}


def _request(contract: dict[str, Any], tmp_path: Path) -> WorkerRequest:
    """Build one worker request around a declarative contract.

    Parameters
    ----------
    contract:
        Input contract under test.
    tmp_path:
        Per-test scratch root.

    Returns
    -------
    WorkerRequest
        Request sufficient for declarative input materialization.
    """

    return WorkerRequest(
        stable_id="m7362",
        recipe={"distribution": "timm", "version": "1.0.28"},
        modality="image",
        input_spec=None,
        scratch_root=tmp_path,
        receipt_path=tmp_path / "receipt.json",
        work_id="work-m7362",
        input_contract=contract,
        device="cpu",
        framework="torch",
        mode=RunMode.EVAL,
    )


def test_the_authored_kind_still_refuses_and_now_names_itself(tmp_path: Path) -> None:
    """The executor's vocabulary is unchanged; only the diagnosis improves."""

    with pytest.raises(TypeError) as excinfo:
        _materialize_declarative_call(_request(_contract(_AUTHORED_LEAF), tmp_path))
    message = str(excinfo.value)
    assert "'args[0]'" in message
    assert "'standard-image-tensor'" in message
    assert "'tensor' or 'constructed'" in message
    # The old text asserted a falsehood about an ordinary image tensor.
    assert "contains a non-tensor leaf" not in message


def test_the_same_contract_with_the_one_word_corrected_materializes(tmp_path: Path) -> None:
    """The census model was one label away from a real forward, and nothing else."""

    torch = pytest.importorskip("torch")
    corrected = {**_AUTHORED_LEAF, "kind": "tensor"}
    args, kwargs, input_kind, _asset, _note = _materialize_declarative_call(
        _request(_contract(corrected), tmp_path)
    )
    assert kwargs == {}
    materialized = args[0]
    assert isinstance(materialized, torch.Tensor)
    assert tuple(materialized.shape) == (1, 3, 224, 224)
    assert materialized.dtype is torch.float32
    # The leaf DECLARES distribution "normal", and since DX-5's totality fix a
    # declared distribution is honored by its own arm instead of being silently
    # substituted with the modality asset (the pre-fix behavior this assertion
    # used to pin as "standard-image").
    assert input_kind == "standard-normal"


def test_the_schema_now_refuses_the_authored_kind_before_execution() -> None:
    """The refusal moves to proposal validation, where it costs no permanent record."""

    validator = get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
    baseline = make_author_proposal()
    assert list(validator.iter_errors(baseline)) == []
    assert baseline["proposed_facts"]["input_contract"]["args"][0]["kind"] == "tensor"

    refused = deepcopy(baseline)
    refused["proposed_facts"]["input_contract"]["args"][0]["kind"] = _AUTHORED_LEAF["kind"]
    messages = [error.message for error in validator.iter_errors(refused)]
    assert messages == ["'standard-image-tensor' is not one of ['tensor', 'constructed']"]


def test_the_closed_kinds_both_validate() -> None:
    """The closure names exactly the executor's two kinds, and refuses nothing else."""

    validator = get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
    proposal = make_author_proposal()
    contract = proposal["proposed_facts"]["input_contract"]
    contract["args"] = [deepcopy(_CONSTRUCTED_LEAF)]
    assert list(validator.iter_errors(proposal)) == []


def test_the_constructed_arm_is_untouched(tmp_path: Path) -> None:
    """Closing the vocabulary narrows labels, never the grammar's expressiveness."""

    pytest.importorskip("dgl")
    args, _kwargs, input_kind, _asset, _note = _materialize_declarative_call(
        _request(_contract(_CONSTRUCTED_LEAF), tmp_path)
    )
    assert input_kind == "standard-constructed-input"
    assert args


# The exact self-contradictory leaf shape the rung-11 census found on the
# superseded ``m9304``/``m9407``/``m9653``/``m9666``/``m9678`` input terminals:
# a literal constructor kwarg spelled as an input leaf with
# ``kind: "constructed"``, ``distribution: "categorical"``, and no constructor
# block. It validated against the historical open shape, passed the metadata
# gates, and died in the worker with a diagnostic that named ``constructed`` as
# inside the vocabulary while refusing it. The specimen below is ``m9666``'s
# ``kwargs.encoder_name`` leaf.
_LITERAL_KWARG_LEAF: dict[str, Any] = {
    "path": "kwargs.encoder_name",
    "kind": "constructed",
    "semantic_role": "constructor kwarg selecting the se_resnet50 encoder architecture",
    "shape": [],
    "dtype": "not-applicable",
    "device_policy": "not-applicable",
    "distribution": "categorical",
    "constraints": ['fixed value "se_resnet50" selecting the SENetEncoder registry entry'],
    "source_evidence_ids": ["ev-pan-signature"],
}

_LITERAL_KWARG_AS_MACHINE_VALUE: dict[str, Any] = {
    "path": "kwargs.encoder_name",
    "type": "str",
    "value": "se_resnet50",
    "semantic_role": "constructor kwarg selecting the se_resnet50 encoder architecture",
    "constraints": ['fixed value "se_resnet50" selecting the SENetEncoder registry entry'],
    "source_evidence_ids": ["ev-pan-signature"],
}


def test_the_contradictory_scalar_shape_refuses_at_schema_time() -> None:
    """kind='constructed' with a non-constructor distribution never reaches a gate."""

    validator = get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
    refused = make_author_proposal()
    refused["proposed_facts"]["input_contract"]["kwargs"] = [deepcopy(_LITERAL_KWARG_LEAF)]
    messages = [error.message for error in validator.iter_errors(refused)]
    assert messages == ["'categorical' is not one of ['constructor']"]
    with pytest.raises(PayloadValidationError, match=r"is not one of \['constructor'\]"):
        validate_payload(refused, AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)

    refused_args = make_author_proposal()
    refused_args["proposed_facts"]["input_contract"]["args"].append(
        {**deepcopy(_LITERAL_KWARG_LEAF), "path": "args[1]"}
    )
    assert [error.message for error in validator.iter_errors(refused_args)] == [
        "'categorical' is not one of ['constructor']"
    ]


def test_the_machine_value_respelling_validates() -> None:
    """The correct spelling for a literal constructor kwarg is a non_tensor_values row."""

    validator = get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
    proposal = make_author_proposal()
    contract = proposal["proposed_facts"]["input_contract"]
    contract["non_tensor_values"] = [
        *contract.get("non_tensor_values", []),
        deepcopy(_LITERAL_KWARG_AS_MACHINE_VALUE),
    ]
    assert list(validator.iter_errors(proposal)) == []


def test_the_proposal_validator_names_the_machine_value_respelling(tmp_path: Path) -> None:
    """The author-facing refusal prescribes non_tensor_values, not a schema enum."""

    refused = make_author_proposal()
    refused["proposed_facts"]["input_contract"]["kwargs"] = [deepcopy(_LITERAL_KWARG_LEAF)]
    with pytest.raises(ProposalValidationError) as excinfo:
        validate_author_proposal(refused, allowed_model_dir=tmp_path, source_manifest=[])
    message = str(excinfo.value)
    assert "'kwargs.encoder_name'" in message
    assert "'categorical'" in message
    assert "input_contract.non_tensor_values" in message
    # The refusal never claims 'constructed' sits outside the vocabulary.
    assert "is not one of" not in message


def test_the_scalar_seam_check_passes_coherent_shapes() -> None:
    """The seam check refuses exactly the contradiction and nothing else."""

    coherent = {
        "input_contract": {
            "args": [deepcopy(_CONSTRUCTED_LEAF), {**deepcopy(_AUTHORED_LEAF), "kind": "tensor"}],
            "kwargs": [],
            "non_tensor_values": [deepcopy(_LITERAL_KWARG_AS_MACHINE_VALUE)],
        }
    }
    _validate_scalar_leaf_kinds(coherent)
    _validate_scalar_leaf_kinds(None)
    _validate_scalar_leaf_kinds({"input_contract": "not-a-mapping"})
    with pytest.raises(ProposalValidationError, match="non_tensor_values"):
        _validate_scalar_leaf_kinds({"input_contract": {"kwargs": [deepcopy(_LITERAL_KWARG_LEAF)]}})


def test_the_worker_diagnostic_names_the_actual_contract(tmp_path: Path) -> None:
    """If the seam is ever reached, the diagnostic no longer contradicts itself."""

    with pytest.raises(TypeError) as excinfo:
        _materialize_declarative_call(
            _request(
                {"args": [], "kwargs": [deepcopy(_LITERAL_KWARG_LEAF)], "non_tensor_values": []},
                tmp_path,
            )
        )
    message = str(excinfo.value)
    assert "'kwargs.encoder_name'" in message
    assert "kind 'constructed' with distribution 'categorical'" in message
    assert "input_contract.non_tensor_values" in message
    # The old text claimed 'constructed' was inside the vocabulary while refusing it.
    assert "the executable vocabulary is" not in message


def test_the_machine_value_respelling_materializes(tmp_path: Path) -> None:
    """The prescribed respelling produces the exact literal kwarg at execution."""

    pytest.importorskip("torch")
    corrected = {
        "args": [{**deepcopy(_AUTHORED_LEAF), "kind": "tensor"}],
        "kwargs": [],
        "non_tensor_values": [deepcopy(_LITERAL_KWARG_AS_MACHINE_VALUE)],
    }
    args, kwargs, _input_kind, _asset, _note = _materialize_declarative_call(
        _request(corrected, tmp_path)
    )
    assert len(args) == 1
    assert kwargs == {"encoder_name": "se_resnet50"}
