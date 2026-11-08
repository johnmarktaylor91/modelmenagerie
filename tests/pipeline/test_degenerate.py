"""``degenerate``: the discrete-step witness, the train-mode check and the init-change probe on
tiny models; the worker's use of them is in ``test_worker``."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from menagerie.pipeline import degenerate


def _forward(model: nn.Module) -> Any:
    def run(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        torch.manual_seed(0)
        return model(*args, **kwargs)

    return run


def _inputs() -> tuple[tuple[tuple[Any, ...], dict], tuple[tuple[Any, ...], dict]]:
    g = torch.Generator().manual_seed(1)
    return ((torch.randn(4, 6, generator=g),), {}), ((torch.randn(4, 6, generator=g) * 3 + 1,), {})


class Rounds(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc, self.norm, self.dec = nn.Linear(6, 3), nn.BatchNorm1d(3), nn.Linear(3, 2)

    def forward(self, x: Any) -> Any:
        y = self.norm(self.enc(x)) * 0.01
        return self.dec(y + (torch.rand_like(y) - 0.5) if self.training else torch.round(y))


class DeadThenThreshold(nn.Module):
    """Constant before its threshold: the threshold's input is the same for both inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 3)
        nn.init.zeros_(self.lin.weight)

    def forward(self, x: Any) -> Any:
        return (self.lin(x) > 0).float() + self.lin(x)


def test_witness_is_the_first_discrete_op_whose_input_differs_and_output_does_not() -> None:
    a = [("aten.gt", "x1", "m"), ("aten.round", "y1", "z")]
    second = [("aten.gt", "x2", "m2"), ("aten.round", "y2", "z")]
    assert degenerate.witness(a, second) == "aten.round"
    assert degenerate.witness(a, [("aten.gt", "x2", "m"), ("aten.round", "y2", "z")]) == "aten.gt"
    assert degenerate.witness(a, [("aten.gt", "x1", "m"), ("aten.round", "y1", "z")]) is None
    assert degenerate.witness(a, [("aten.max", "x2", "m"), ("aten.round", "y2", "z")]) is None


def test_a_rounding_collapse_needs_no_train_forward_and_keeps_its_buffers() -> None:
    model = Rounds().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    a, b = _inputs()
    assert degenerate.discrete_step(model, _forward(model), a, b) == ("aten.round", False)
    assert "train-mode" not in degenerate.discrete_note("aten.round", alive=False)
    assert not model.training
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "train-mode forwards restored"


def test_straight_through_rounding_has_a_witness_and_a_dead_train_mode() -> None:
    class StraightThrough(Rounds):
        def forward(self, x: Any) -> Any:
            y = self.norm(self.enc(x)) * 0.01
            return self.dec(torch.round(y) + (y - y.detach()))

    model = StraightThrough().eval()
    a, b = _inputs()
    assert degenerate.discrete_step(model, _forward(model), a, b) == ("aten.round", False)
    assert "aten.round" in degenerate.QUANTIZE and "aten.gt" not in degenerate.QUANTIZE


def test_a_net_constant_before_its_threshold_gets_no_note() -> None:
    model = DeadThenThreshold().eval()
    a, b = _inputs()
    assert degenerate.discrete_step(model, _forward(model), a, b) == (None, False)


def test_a_model_that_raises_gets_no_note_and_no_fix() -> None:
    class Raises(nn.Module):
        def forward(self, x: Any) -> Any:
            raise RuntimeError("boom")

    model, (a, b) = Raises().eval(), _inputs()
    assert degenerate.discrete_step(model, _forward(model), a, b) == (None, False)
    assert degenerate.init_fix(model, _forward(model), a, b) is None


def test_init_fix_resets_zero_weights_and_fills_fresh_batchnorm() -> None:
    head = nn.Sequential(nn.Linear(6, 3), nn.Linear(3, 2))
    nn.init.zeros_(head[1].weight)
    nn.init.zeros_(head[1].bias)
    a, b = _inputs()
    fix = degenerate.init_fix(head.eval(), _forward(head), a, b)
    assert fix == "1 all-zero weight layers reset to the framework default"

    class Saturates(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin, self.norm = nn.Linear(6, 3), nn.BatchNorm1d(3)

        def forward(self, x: Any) -> Any:
            return torch.tanh(self.norm(self.lin(x) + 1e4))

    model = Saturates().eval()
    fix = degenerate.init_fix(model, _forward(model), a, b)
    assert fix == "1 BatchNorm layers' statistics filled by one train-mode no-grad pass"
    assert not model.training and model.norm.momentum == 0.1
