"""Distribution dispatch totality over the closed schema enum (DX-5 / RE-5).

The OMP trio died because a DECLARED ``zeros``/``ones`` distribution fell
through to modality text bytes. The fix was symptomatic: ``categorical``,
``normal``, and ``uniform`` kept falling through the same hole -- the same bug
with a different member of the closed enum, no tripwire, and ``categorical``
is the natural authored spelling for token/type ids.

These pins are GENERATED from the model-common schema's ``distribution`` enum
at test time: adding an enum member without a materialization arm fails here
in the same change, and no declared distribution can ever be silently
substituted with modality assets or the random fallback again.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from menagerie.crawler.standard_inputs import (
    StandardInputError,
    materialize_standard_input,
)

pytestmark = pytest.mark.smoke

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "model-common.schema.json"


def _schema_distribution_enum() -> tuple[str, ...]:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    text = json.dumps(schema)
    assert '"distribution"' in text

    # The enum lives on the input-contract leaf definition.
    def walk(node: object) -> tuple[str, ...] | None:
        if isinstance(node, dict):
            distribution = node.get("distribution")
            if isinstance(distribution, dict) and isinstance(distribution.get("enum"), list):
                return tuple(distribution["enum"])
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        if isinstance(node, list):
            for value in node:
                found = walk(value)
                if found is not None:
                    return found
        return None

    enum = walk(schema)
    assert enum, "model-common schema no longer declares a distribution enum"
    return enum


_ENUM = _schema_distribution_enum()

#: Substituted kinds a DECLARED distribution must never produce.
_SUBSTITUTED_KINDS = frozenset(
    {
        "standard-image",
        "standard-text",
        "standard-audio",
        "standard-video",
        "standard-tabular",
        "random-fallback",
    }
)


@pytest.mark.parametrize("member", sorted(_ENUM))
def test_dispatch_is_total_over_the_schema_enum(member: str) -> None:
    """Every enum member is honored by its own arm (or typed-refused by design).

    A NEW enum member without a matching materialization arm fails this test
    in the change that adds it: both representative specs raise the
    "no materialization arm" refusal and no honored kind is produced.
    """

    if member == "constructor":
        # The constructed-leaf grammar: reaching the TENSOR materializer with
        # it is a contract breach and must refuse typed, never substitute.
        with pytest.raises(StandardInputError, match="constructed-leaf grammar"):
            materialize_standard_input(
                "language", {"shape": [2, 4], "dtype": "int64", "distribution": member}
            )
        return
    honored: list[str] = []
    for spec in (
        {"shape": [2, 4], "dtype": "float32", "distribution": member},
        {
            "shape": [2, 4],
            "dtype": "int64",
            "distribution": member,
            "constraints": ["values in [0, 8)"],
        },
    ):
        try:
            materialized = materialize_standard_input("language", spec, framework="numpy")
        except StandardInputError:
            continue
        assert materialized.input_kind == f"standard-{member}"
        assert materialized.input_kind not in _SUBSTITUTED_KINDS
        assert materialized.input_asset is None
        honored.append(materialized.input_kind)
    assert honored, (
        f"schema enum member {member!r} has no materialization arm -- a declared "
        "contract would be refused or substituted (DX-5 totality breach)"
    )


@pytest.mark.parametrize("member", sorted(set(_ENUM) | {"poisson-not-in-the-enum"}))
def test_declared_distribution_is_never_silently_substituted(member: str) -> None:
    """Declared distributions are honored or typed-refused; never asset bytes."""

    for dtype in ("float32", "int64", "bool"):
        spec = {
            "shape": [2, 6],
            "dtype": dtype,
            "distribution": member,
            "constraints": ["values in [0, 4)"],
        }
        try:
            materialized = materialize_standard_input("language", spec, framework="numpy")
        except StandardInputError:
            continue
        assert materialized.input_kind == f"standard-{member}"
        assert materialized.input_kind not in _SUBSTITUTED_KINDS


def test_categorical_samples_inside_the_declared_bounds() -> None:
    """The natural token/type-id spelling samples ids inside its legal range."""

    materialized = materialize_standard_input(
        "language",
        {
            "shape": [3, 16],
            "dtype": "int64",
            "distribution": "categorical",
            "constraints": ["indices selected in [0, 4)"],
        },
        framework="numpy",
    )
    assert materialized.input_kind == "standard-categorical"
    values = np.asarray(materialized.value)
    assert values.min() >= 0 and values.max() < 4
    # Deterministic under the seed.
    again = materialize_standard_input(
        "language",
        {
            "shape": [3, 16],
            "dtype": "int64",
            "distribution": "categorical",
            "constraints": ["indices selected in [0, 4)"],
        },
        framework="numpy",
    )
    assert np.array_equal(values, np.asarray(again.value))


def test_categorical_without_coherent_bounds_refuses_typed() -> None:
    """No declared bound means REFUSE -- inventing a range is the [0,16) class."""

    with pytest.raises(StandardInputError, match="coherent integer bound"):
        materialize_standard_input(
            "language",
            {"shape": [2, 4], "dtype": "int64", "distribution": "categorical"},
            framework="numpy",
        )
    with pytest.raises(StandardInputError, match="integer dtype"):
        materialize_standard_input(
            "language",
            {"shape": [2, 4], "dtype": "float32", "distribution": "categorical"},
            framework="numpy",
        )


def test_integer_range_with_unparseable_constraints_refuses_typed() -> None:
    """A declared integer-range that cannot parse no longer substitutes assets."""

    with pytest.raises(StandardInputError, match="coherent integer bound"):
        materialize_standard_input(
            "language",
            {
                "shape": [2, 4],
                "dtype": "int64",
                "distribution": "integer-range",
                "constraints": ["symbolic vocabulary size N"],
            },
            framework="numpy",
        )


def test_normal_is_a_real_standard_normal_sample() -> None:
    """The normal arm bypasses the byte-asset rescale (no silent /255)."""

    materialized = materialize_standard_input(
        "vision",
        {"shape": [4, 64], "dtype": "float32", "distribution": "normal"},
        framework="numpy",
    )
    values = np.asarray(materialized.value)
    assert materialized.input_kind == "standard-normal"
    # A /255-rescaled standard normal would have std ~0.004; a real one ~1.
    assert 0.7 < float(values.std()) < 1.3
    with pytest.raises(StandardInputError, match="float dtype"):
        materialize_standard_input(
            "vision",
            {"shape": [2, 4], "dtype": "int64", "distribution": "normal"},
            framework="numpy",
        )


def test_uniform_honors_float_and_bounded_integer_specs() -> None:
    """Uniform floats live in [0, 1); uniform integers require declared bounds."""

    floats = materialize_standard_input(
        "vision",
        {"shape": [4, 64], "dtype": "float32", "distribution": "uniform"},
        framework="numpy",
    )
    values = np.asarray(floats.value)
    assert floats.input_kind == "standard-uniform"
    assert values.min() >= 0.0 and values.max() < 1.0
    integers = materialize_standard_input(
        "language",
        {
            "shape": [2, 8],
            "dtype": "int64",
            "distribution": "uniform",
            "constraints": ["ids in [0, 5)"],
        },
        framework="numpy",
    )
    assert integers.input_kind == "standard-uniform"
    int_values = np.asarray(integers.value)
    assert int_values.min() >= 0 and int_values.max() < 5
    with pytest.raises(StandardInputError, match="coherent integer bound"):
        materialize_standard_input(
            "language",
            {"shape": [2, 8], "dtype": "int64", "distribution": "uniform"},
            framework="numpy",
        )


def test_undeclared_distribution_keeps_the_historical_modality_path() -> None:
    """No declared distribution -> the modality-asset path stays byte-identical."""

    text = materialize_standard_input("language", {"shape": [1, 16], "dtype": "int64"})
    assert text.input_kind == "standard-text"
    unknown = materialize_standard_input("unknown", {"shape": [2, 5], "dtype": "float32"})
    assert unknown.input_kind == "random-fallback"
