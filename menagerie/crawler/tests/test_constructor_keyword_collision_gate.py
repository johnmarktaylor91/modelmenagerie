"""Constructor keywords in the forward contract are refused at CHECK time.

Leg 4 (2026-08-12), the campaign's first REAL executions: all four
``segmentation_models_pytorch`` models died at forward with ``TypeError:
SegmentationModel.forward() got an unexpected keyword argument 'encoder_name'``.
The author had filed the recipe's constructor kwargs a SECOND time as
``input_contract.non_tensor_values`` leaves with paths like
``kwargs.encoder_name`` (m9430; the leaf's own semantic_role said "constructor
keyword..."), the checker accepted the contract, and the worker -- which
assembles EVERY contract leaf into the forward call -- faithfully executed the
impossible call after the environment had already been bought.

The gate pinned here (``proposal._validate_constructor_keyword_collisions``)
closes the class where it is cheap: a ``non_tensor_values`` or ``kwargs`` leaf
whose terminal path field is also a declared constructor binding (a
``declared_choices[].field`` or a ``library_recipe.kwargs`` key) refuses at
proposal validation, naming BOTH locations and the correct spelling
(constructor keywords are construction facts; the input_contract describes only
the forward call). A genuine forward keyword -- a real ``forward(mask=...)``
tensor or a non-tensor flag no constructor choice names -- still passes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from menagerie.crawler.proposal import (
    ProposalValidationError,
    _validate_constructor_keyword_collisions,
)

pytestmark = pytest.mark.smoke


def _m9430_facts() -> dict[str, Any]:
    """The exact double-filing shape of the m9430 author proposal."""

    return {
        "implementation": {
            "library_recipe": {
                "distribution": "segmentation-models-pytorch",
                "module": "segmentation_models_pytorch",
                "symbol": "FPN",
                "version": "0.5.0",
                "kwargs": {
                    "encoder_name": "resnext101_32x16d",
                    "encoder_weights": None,
                },
                "pretrained_disable_fields": ["encoder_weights"],
            },
            "declared_choices": [
                {
                    "field": "encoder_name",
                    "value": "resnext101_32x16d",
                    "material": True,
                    "rationale": "roster row names the resnext101_32x16d encoder",
                    "source_status": "author-selected-variant",
                    "evidence_ids": ["ex-encoder-variant"],
                },
                {
                    "field": "encoder_weights",
                    "value": "null",
                    "material": True,
                    "rationale": "encoder_weights None selects random initialization",
                    "source_status": "author-disabled-pretrained",
                    "evidence_ids": ["ex-fpn-weights-doc"],
                },
            ],
        },
        "input_contract": {
            "builder_symbol": "segmentation_models_pytorch.FPN",
            "args": [
                {
                    "path": "args[0]",
                    "kind": "tensor",
                    "shape": [1, 3, 64, 64],
                    "dtype": "float32",
                    "distribution": "normal",
                    "device_policy": "cpu",
                    "semantic_role": "RGB image batch",
                    "constraints": ["divisible by output stride 32"],
                    "source_evidence_ids": ["ex-check-input"],
                }
            ],
            "kwargs": [],
            "non_tensor_values": [
                {
                    "path": "kwargs.encoder_name",
                    "type": "str",
                    "value": "resnext101_32x16d",
                    "semantic_role": (
                        "constructor keyword selecting the resnext101_32x16d encoder registry entry"
                    ),
                    "constraints": ["must name a registered encoder entry"],
                    "source_evidence_ids": ["ex-fpn-signature"],
                },
                {
                    "path": "kwargs.encoder_weights",
                    "type": "Optional[str]",
                    "value": None,
                    "semantic_role": "constructor keyword selecting weight init",
                    "constraints": ["must be None"],
                    "source_evidence_ids": ["ex-fpn-signature"],
                },
            ],
        },
    }


def test_the_m9430_shape_is_refused_with_both_locations_named() -> None:
    """The exact leg-4 double-filing refuses, naming both spellings."""

    with pytest.raises(ProposalValidationError) as excinfo:
        _validate_constructor_keyword_collisions(_m9430_facts())
    message = str(excinfo.value)
    assert "'kwargs.encoder_name'" in message
    assert "input_contract.non_tensor_values" in message
    assert "implementation.library_recipe.kwargs" in message
    assert "construction facts" in message
    assert "forward" in message


def test_a_declared_choices_only_collision_is_refused_too() -> None:
    """A binding declared only through declared_choices still collides."""

    facts = _m9430_facts()
    facts["implementation"]["library_recipe"]["kwargs"] = {}
    with pytest.raises(ProposalValidationError) as excinfo:
        _validate_constructor_keyword_collisions(facts)
    assert "implementation.declared_choices" in str(excinfo.value)


def test_a_kwargs_collection_leaf_colliding_with_a_binding_is_refused() -> None:
    """The gate covers tensor kwargs leaves, not only non_tensor_values."""

    facts = _m9430_facts()
    facts["input_contract"]["non_tensor_values"] = []
    facts["input_contract"]["kwargs"] = [
        {
            "path": "kwargs.encoder_name",
            "kind": "tensor",
            "shape": [1],
            "dtype": "int64",
            "distribution": "integer-range",
            "device_policy": "cpu",
            "semantic_role": "mislabelled constructor selector",
            "constraints": [],
            "source_evidence_ids": ["ex-fpn-signature"],
        }
    ]
    with pytest.raises(ProposalValidationError, match="encoder_name"):
        _validate_constructor_keyword_collisions(facts)


def test_a_legitimate_forward_keyword_still_passes() -> None:
    """A real forward kwarg no constructor choice names is untouched."""

    facts = _m9430_facts()
    facts["input_contract"]["non_tensor_values"] = [
        {
            "path": "kwargs.upsample_output",
            "type": "bool",
            "value": True,
            "semantic_role": "forward flag selecting full-resolution output",
            "constraints": [],
            "source_evidence_ids": ["ex-forward-signature"],
        }
    ]
    facts["input_contract"]["kwargs"] = [
        {
            "path": "kwargs.mask",
            "kind": "tensor",
            "shape": [1, 1, 64, 64],
            "dtype": "float32",
            "distribution": "normal",
            "device_policy": "cpu",
            "semantic_role": "attention mask consumed by forward",
            "constraints": [],
            "source_evidence_ids": ["ex-forward-signature"],
        }
    ]
    _validate_constructor_keyword_collisions(facts)


def test_positional_and_nested_paths_key_on_the_terminal_field() -> None:
    """args[0] has no terminal field; a nested terminal collision still refuses."""

    facts = _m9430_facts()
    facts["input_contract"]["non_tensor_values"] = [
        {
            "path": "args[1]",
            "type": "int",
            "value": 3,
            "semantic_role": "positional forward scalar",
            "constraints": [],
            "source_evidence_ids": ["ex-forward-signature"],
        }
    ]
    _validate_constructor_keyword_collisions(facts)
    facts["input_contract"]["non_tensor_values"] = [
        {
            "path": "kwargs.options.encoder_name",
            "type": "str",
            "value": "resnext101_32x16d",
            "semantic_role": "constructor keyword smuggled one level down",
            "constraints": [],
            "source_evidence_ids": ["ex-fpn-signature"],
        }
    ]
    with pytest.raises(ProposalValidationError, match="encoder_name"):
        _validate_constructor_keyword_collisions(facts)


def test_malformed_and_binding_free_shapes_fall_through_to_the_schema() -> None:
    """Shapes the schema owns are ignored here, never crashed on."""

    _validate_constructor_keyword_collisions(None)
    _validate_constructor_keyword_collisions({"input_contract": "not-a-mapping"})
    _validate_constructor_keyword_collisions({"input_contract": {}})
    facts = _m9430_facts()
    facts["implementation"] = {}
    _validate_constructor_keyword_collisions(facts)
    facts = _m9430_facts()
    del facts["implementation"]
    _validate_constructor_keyword_collisions(facts)
    facts = _m9430_facts()
    facts["input_contract"]["non_tensor_values"] = ["not-a-mapping", {"path": 7}]
    facts["input_contract"]["kwargs"] = "not-a-list"
    _validate_constructor_keyword_collisions(facts)


def test_deepcopy_isolation_of_the_shared_fixture() -> None:
    """The fixture builder returns fresh objects each call."""

    first = _m9430_facts()
    second = _m9430_facts()
    assert first == second
    assert first is not second
    assert deepcopy(first) == second
