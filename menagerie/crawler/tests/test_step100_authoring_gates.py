"""Ranked authoring gates from the step-100 triage (ranks 3 and 4).

Rank 3, DEVICE-POLICY-PROSE (m11584, m1359, m9148, m8650): authors filled
``implementation.device_policy`` with placement narrative ("constructed on
cpu; caller moves the module with .to(device)"), the driver forwarded it
verbatim, and ``torch.as_tensor(device=...)`` killed input generation after
the single authoring visit. Gate: the value must parse as a literal torch
device at proposal validation, and the driver's own consumption re-parses it
typed (defense in depth).

Rank 4, BN-TRAIN-BATCH1 (m9304, m9610, m9617, m9636, m9653, m9666, m9678,
m9911): a batch-1 probe plus a declared train meaningful mode dies in the
train receipt at every BatchNorm ("Expected more than 1 value per channel
when training"). Gate: the mode DECLARATION is the signal -- batch dim 1 +
declared train refuses with a batch >= 2 prescription; no BatchNorm guessing
from authored facts. Both directions pinned: eval-only batch-1 stays legal,
train with batch >= 2 stays legal.
"""

from __future__ import annotations

from typing import Any

import pytest

from menagerie.crawler.proposal import (
    ProposalValidationError,
    _validate_device_policy,
    _validate_train_mode_probe_batch,
    parse_device_policy,
)

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# rank 3: DEVICE-POLICY-PROSE
# ---------------------------------------------------------------------------

#: The four verbatim prose values from the step-100 leg's diagnostics.
_PROSE_DEVICE_POLICIES = (
    "device-agnostic; build_model constructs on CPU and the caller moves it",
    "constructed on cpu; caller moves the module with .to(device)",
    "model built on cpu and movable; make_dummy_call places input tensors",
    "cpu (module constructed on host; move with .to(device) when needed)",
)


@pytest.mark.parametrize("prose", _PROSE_DEVICE_POLICIES)
def test_prose_device_policy_is_refused_naming_the_field(prose: str) -> None:
    """Every observed prose spelling refuses with the exact field named."""

    with pytest.raises(ProposalValidationError) as excinfo:
        _validate_device_policy({"device_policy": prose})
    message = str(excinfo.value)
    assert "implementation.device_policy" in message
    assert "LITERAL torch device string" in message


@pytest.mark.parametrize("literal", ["cpu", "cuda", "cuda:0", "mps", "meta"])
def test_literal_torch_devices_stay_legal(literal: str) -> None:
    """Real device literals parse and return unchanged."""

    assert parse_device_policy(literal) == literal
    _validate_device_policy({"device_policy": literal})


@pytest.mark.parametrize("value", [None, 0, "", " cpu", "cpu "])
def test_non_string_and_padded_device_policies_are_refused(value: object) -> None:
    """Only an exact literal string is a device; padding and non-strings refuse."""

    with pytest.raises(ProposalValidationError):
        parse_device_policy(value)


def test_driver_site_names_a_custom_field() -> None:
    """The defense-in-depth parser names whatever field its caller consumes."""

    with pytest.raises(ProposalValidationError, match="request.device"):
        parse_device_policy("constructed on cpu", field="request.device")


# ---------------------------------------------------------------------------
# rank 4: BN-TRAIN-BATCH1
# ---------------------------------------------------------------------------


def _facts(
    *,
    meaningful_modes: list[str],
    shape: list[Any],
    kind: str = "tensor",
    collection: str = "args",
) -> dict[str, Any]:
    """Minimal fact tree for the probe-batch gate."""

    leaf = {
        "path": f"{collection}[0]" if collection == "args" else "kwargs.x",
        "kind": kind,
        "shape": shape,
        "dtype": "float32",
        "distribution": "normal",
    }
    return {
        "modes": {"meaningful_modes": meaningful_modes},
        "input_contract": {"args": [], "kwargs": [], collection: [leaf]},
    }


def test_the_m9617_shape_is_refused_with_the_batch_prescription() -> None:
    """The namesake: [1,3,32,32] + declared [train, eval] refuses."""

    with pytest.raises(ProposalValidationError) as excinfo:
        _validate_train_mode_probe_batch(
            _facts(meaningful_modes=["train", "eval"], shape=[1, 3, 32, 32])
        )
    message = str(excinfo.value)
    assert "batch dimension 1" in message
    assert ">= 2" in message
    assert "'train'" in message or "train" in message


def test_a_kwargs_leaf_is_gated_too() -> None:
    """The batch axis is checked on every tensor leaf, not only args[0]."""

    with pytest.raises(ProposalValidationError, match="kwargs.x"):
        _validate_train_mode_probe_batch(
            _facts(meaningful_modes=["train"], shape=[1, 256, 16], collection="kwargs")
        )


def test_rank_2_mlp_probes_are_deliberately_spared() -> None:
    """Batch-1 train over a [1, F] MLP probe stays legal.

    The corpus's honest batch-1-train MLP class (the slice-F driver fixtures
    run (1, 2) probes under declared train) has no norm layer failing on the
    batch axis; every observed BN-TRAIN-BATCH1 incident was a rank >= 3
    [1, C, ...] probe. The gate keys on that observed class.
    """

    _validate_train_mode_probe_batch(_facts(meaningful_modes=["train", "eval"], shape=[1, 2]))


def test_eval_only_batch_1_stays_legal() -> None:
    """Direction 2: the same probe with eval as the only meaningful mode passes."""

    _validate_train_mode_probe_batch(_facts(meaningful_modes=["eval"], shape=[1, 3, 32, 32]))


def test_train_with_batch_2_stays_legal() -> None:
    """Direction 2: declared train with a batch >= 2 probe passes."""

    _validate_train_mode_probe_batch(
        _facts(meaningful_modes=["train", "eval"], shape=[2, 3, 32, 32])
    )


def test_rank_1_and_symbolic_leading_dimensions_are_untouched() -> None:
    """No batch axis, no gate: index vectors and symbolic dims stay legal."""

    _validate_train_mode_probe_batch(_facts(meaningful_modes=["train"], shape=[1]))
    _validate_train_mode_probe_batch(_facts(meaningful_modes=["train"], shape=["batch", 3, 32, 32]))


def test_constructed_leaves_are_untouched() -> None:
    """A constructed (graph) leaf carries no probe batch axis to gate."""

    _validate_train_mode_probe_batch(
        _facts(meaningful_modes=["train"], shape=[1, 3], kind="constructed")
    )
