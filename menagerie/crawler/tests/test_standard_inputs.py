"""Tests for canonical modality-aware input materialization."""

from __future__ import annotations

import numpy as np

from menagerie.crawler.standard_inputs import materialize_standard_input


def test_standard_image_text_and_unknown_fallback_record_input_kind() -> None:
    """Known modalities use assets while unknown modalities record random fallback."""

    image = materialize_standard_input("vision", {"shape": [1, 3, 12, 10], "dtype": "float32"})
    text = materialize_standard_input("language", {"shape": [1, 16], "dtype": "int64"})
    unknown = materialize_standard_input("unknown", {"shape": [2, 5], "dtype": "float32"})

    assert image.input_kind == "standard-image"
    assert image.input_asset is not None
    assert tuple(getattr(image.value, "shape")) == (1, 3, 12, 10)
    assert text.input_kind == "standard-text"
    assert text.input_asset is not None
    assert tuple(getattr(text.value, "shape")) == (1, 16)
    assert unknown.input_kind == "random-fallback"
    assert unknown.input_asset is None
    assert "random fallback" in unknown.input_note


def test_declared_zero_and_one_distributions_override_language_asset() -> None:
    """Mask distributions stay constant instead of becoming text token ids."""

    token_type_ids = materialize_standard_input(
        "language",
        {
            "shape": [2, 8],
            "dtype": "int64",
            "distribution": "zeros",
            "constraints": ["indices selected in [0, 1] (type_vocab_size 2)"],
        },
        framework="numpy",
    )
    attention_mask = materialize_standard_input(
        "language",
        {
            "shape": [2, 8],
            "dtype": "float32",
            "distribution": "ones",
            "constraints": ["mask values selected in [0, 1]"],
        },
        framework="numpy",
    )

    assert token_type_ids.input_kind == "standard-zeros"
    assert token_type_ids.input_asset is None
    assert np.array_equal(token_type_ids.value, np.zeros((2, 8), dtype=np.int64))
    assert attention_mask.input_kind == "standard-ones"
    assert attention_mask.input_asset is None
    assert np.array_equal(attention_mask.value, np.ones((2, 8), dtype=np.float32))
