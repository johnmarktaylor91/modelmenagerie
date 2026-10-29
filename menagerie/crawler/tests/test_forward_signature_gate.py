"""Pre-forward signature gate: every required forward() parameter must be bound.

m2336 (PyKEEN TransE, step-100 triage class FORWARD-KWARG-OMITTED): PyKEEN's
``ERModel.forward(h, r, t, *, mode)`` declares a required keyword-only ``mode``;
the authored contract bound three positional leaves and no kwargs, so the
forward call itself died with ``TypeError: forward() missing 1 required
keyword-only argument: 'mode'`` INSIDE the mode receipt. That burned the
receipt on a defect that is a static fact of the contract: the omission is
provable from ``inspect.signature`` and the materialized args/kwargs before
any forward runs. The worker now refuses with a typed ``RecipeError`` naming
the missing parameter BEFORE the first mode receipt starts, and fails open
when the forward callable is genuinely uninspectable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from menagerie.crawler.constants import RunMode
from menagerie.crawler.recipe import RecipeError
from menagerie.crawler.worker import (
    WorkerRequest,
    _refuse_unbound_forward_parameters,
    run_worker,
)

pytestmark = pytest.mark.smoke


class _KeywordOnlyForward(torch.nn.Module):
    """The m2336 PyKEEN shape: a required keyword-only forward parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, *, mode) -> torch.Tensor:
        del mode
        return self.linear(h + r + t)


def test_the_m2336_keyword_only_omission_is_refused_naming_the_parameter() -> None:
    """A contract binding only positionals is refused, naming ``mode``."""

    model = _KeywordOnlyForward()
    tensors = tuple(torch.zeros(1, 4) for _ in range(3))
    with pytest.raises(RecipeError) as caught:
        _refuse_unbound_forward_parameters(model, tensors, {})
    message = str(caught.value)
    assert "'mode'" in message
    assert "keyword-only" in message
    assert "forward()" in message


def test_a_contract_that_binds_the_keyword_only_parameter_passes() -> None:
    """Binding ``mode`` through kwargs (a non_tensor leaf) satisfies the gate."""

    model = _KeywordOnlyForward()
    tensors = tuple(torch.zeros(1, 4) for _ in range(3))
    _refuse_unbound_forward_parameters(model, tensors, {"mode": None})


def test_a_missing_required_positional_is_refused_too() -> None:
    """The gate covers required positionals, not just keyword-only parameters."""

    model = torch.nn.Transformer(d_model=8, nhead=2)
    with pytest.raises(RecipeError) as caught:
        _refuse_unbound_forward_parameters(model, (torch.zeros(2, 3, 8),), {})
    assert "'tgt'" in str(caught.value)


def test_variadic_and_uninspectable_forwards_fail_open() -> None:
    """``**kwargs`` absorbs anything; an uninspectable signature refuses nothing."""

    class _Variadic(torch.nn.Module):
        def forward(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return args, kwargs

    _refuse_unbound_forward_parameters(_Variadic(), (), {})

    class _Uninspectable:
        # builtins.max has no retrievable signature; the gate must fail open
        # rather than kill an honest model it cannot adjudicate.
        forward = max

    _refuse_unbound_forward_parameters(_Uninspectable(), (), {})


def test_the_worker_refuses_before_any_mode_receipt_burns(tmp_path: Path) -> None:
    """End to end: the refusal lands typed in the diagnostic with no per-mode receipt."""

    tensor_leaf = {
        "kind": "tensor",
        "semantic_role": "source sequence",
        "shape": [2, 3, 8],
        "dtype": "float32",
        "device_policy": "cpu",
        "distribution": "normal",
        "constraints": [],
        "source_evidence_ids": ["evidence-1"],
    }
    contract = {
        "builder_symbol": "make_dummy_call",
        "seed": 0,
        "semantic_description": "Source sequence only; the target leaf is omitted.",
        "source_basis": ["evidence-1"],
        "smallest_valid_probe_rationale": "Smallest batch that reaches forward.",
        "args": [{**tensor_leaf, "path": "args[0]"}],
        "kwargs": [],
        "non_tensor_values": [],
        "masks_state_and_control": [],
        "expected_output_semantics": "decoded sequence",
    }
    request = WorkerRequest(
        stable_id="m_forward_kwarg_omitted",
        recipe={
            "kind": "declarative-library",
            "recipe": {
                "distribution": "torch",
                "version": "test",
                "artifact_sha256": "sha256:" + "a" * 64,
                "module": "torch.nn",
                "symbol": "Transformer",
                "kwargs": {"d_model": 8, "nhead": 2},
                "pretrained_disable_fields": [],
            },
        },
        modality=None,
        input_spec=None,
        input_contract=contract,
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
    )

    receipt = run_worker(request)

    assert receipt["input_completed"] is True
    assert receipt["per_mode"] == {}
    error = receipt["error"]
    assert error["exception_type"] == "menagerie.crawler.recipe.RecipeError"
    assert "'tgt'" in error["message"]
