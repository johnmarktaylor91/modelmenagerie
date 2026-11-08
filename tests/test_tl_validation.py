"""Menagerie's validator door picks the public route when TorchLens offers one, else the legacy
module, else refuses; it never reaches a private ``torchlens._*`` module."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("torchlens")

from menagerie import tl_validation  # noqa: E402


def _no_legacy() -> Any:
    raise AssertionError("the legacy validator must not be used on the public route")


def test_installed_route_hands_the_live_validation_trace_to_the_observer() -> None:
    """The legacy module at the pin, the public ``trace_observer`` route once TorchLens has it."""

    from menagerie import tl_compat

    seen: list[Any] = []
    prior = torch.get_num_threads()

    def observe(trace: Any) -> None:
        status = trace.validation_replay_status
        seen.append((status.state, tl_compat.num_ops(trace) > 0))

    torch.manual_seed(0)
    passed = tl_validation.validate_forward(torch.nn.Linear(3, 2).eval(), [torch.ones(1, 3)], {},
                                            seed=0, num_threads=1, observer=observe)
    assert passed is True
    assert seen == [("passed", True)]
    assert torch.get_num_threads() == prior
    assert tl_validation.failure_for(None) is None or not passed


def test_public_route_pins_threads_around_tl_validate_and_names_seed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    prior = torch.get_num_threads()

    def validate(model: Any, inputs: Any, input_kwargs: Any = None, *, scope: str,
                 seed: int | None = None, validate_metadata: bool = True,
                 trace_observer: Any = None) -> bool:
        calls.append({"scope": scope, "seed": seed, "threads": torch.get_num_threads(),
                      "kwargs": input_kwargs})
        trace_observer(SimpleNamespace(num_ops=3))
        return False

    monkeypatch.setattr(tl, "validate", validate)
    monkeypatch.setattr(tl_validation, "_legacy_validator", _no_legacy)
    seen: list[Any] = []
    passed = tl_validation.validate_forward(object(), [1], {"k": 2}, seed=7, num_threads=1,
                                            observer=seen.append)
    assert passed is False
    assert calls == [{"scope": "forward", "seed": 7, "threads": 1, "kwargs": {"k": 2}}]
    assert seen[0].num_ops == 3
    assert torch.get_num_threads() == prior


def test_public_route_passes_num_threads_and_random_seed_when_accepted(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def validate(model: Any, inputs: Any, input_kwargs: Any = None, *, scope: str,
                 random_seed: int | None = None, validate_metadata: bool = True,
                 num_threads: int | None = None, trace_observer: Any = None) -> bool:
        calls.append({"random_seed": random_seed, "num_threads": num_threads})
        return True

    monkeypatch.setattr(tl, "validate", validate)
    monkeypatch.setattr(tl_validation, "_legacy_validator", _no_legacy)
    assert tl_validation.validate_forward(object(), [1], seed=4, num_threads=2) is True
    assert calls == [{"random_seed": 4, "num_threads": 2}]


def test_no_public_observer_and_no_legacy_module_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(model: Any, inputs: Any, input_kwargs: Any = None, *, scope: str,
                 seed: int | None = None) -> bool:
        raise AssertionError("a verdict without its trace fields must not be taken")

    monkeypatch.setattr(tl, "validate", validate)
    monkeypatch.setitem(sys.modules, "torchlens.user_funcs", None)
    with pytest.raises(tl_validation.TorchLensBoundaryError, match="trace_observer"):
        tl_validation.validate_forward(object(), [1], seed=0)


def test_failure_check_reads_both_field_names() -> None:
    assert tl_validation.failure_check(None) is None
    assert tl_validation.failure_check(SimpleNamespace(check="retrace_mismatch")) == "retrace_mismatch"
    assert tl_validation.failure_check(SimpleNamespace(check_name="perturbation")) == "perturbation"


def test_save_options_reach_either_route_and_an_unaccepted_one_refuses(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def validate(model: Any, inputs: Any, input_kwargs: Any = None, *, scope: str,
                 seed: int | None = None, validate_metadata: bool = True,
                 trace_observer: Any = None, output_device: str = "same",
                 save_budget: Any = "auto") -> bool:
        calls.append({"output_device": output_device, "save_budget": save_budget})
        return True

    saves = {"output_device": "cpu", "save_budget": 1024}
    monkeypatch.setattr(tl, "validate", validate)
    monkeypatch.setattr(tl_validation, "_legacy_validator", _no_legacy)
    assert tl_validation.validate_forward(object(), [1], save_options=saves) is True
    assert tl_validation.validate_forward(object(), [1]) is True
    assert calls == [saves, {"output_device": "same", "save_budget": "auto"}]

    def narrow(model: Any, inputs: Any, input_kwargs: Any = None, *, scope: str,
               validate_metadata: bool = True, trace_observer: Any = None) -> bool:
        raise AssertionError("refused before the call")

    monkeypatch.setattr(tl, "validate", narrow)
    with pytest.raises(tl_validation.TorchLensBoundaryError, match="save_budget"):
        tl_validation.validate_forward(object(), [1], save_options=saves)

    legacy: list[dict[str, Any]] = []
    monkeypatch.setattr(tl, "validate", lambda *a, **k: True)  # no trace_observer: legacy route
    monkeypatch.setattr(tl_validation, "_legacy_validator",
                        lambda: lambda *a, **k: legacy.append(k) or True)
    assert tl_validation.validate_forward(object(), [1], {}, save_options=saves) is True
    assert legacy[0]["output_device"] == "cpu" and legacy[0]["save_budget"] == 1024
