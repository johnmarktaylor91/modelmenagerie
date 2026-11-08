"""What a degenerate random-weight eval forward means: a discrete output step, or an init the
init-change rule lets an adapter replace (decisions.md 2026-10-28 and 2026-10-29).

``worker`` types an eval output that is all zero, or the same for two distinct inputs,
``degenerate_forward:constant``. Two questions follow, both answered on the worker's own build
after that check fired, so a passing forward never pays for them:

- ``discrete_step``: is the collapse only a discrete output step? Lead's refinement (2026-10-29,
  10:05 AM PT): a forward whose eval output collapses only through quantization rounding, an NMS or
  score threshold, or an argmax, while its train-mode output is finite and changes with the input,
  is not degenerate and keeps a one-line note. The eval evidence is a witness: a discrete op (the
  ``DISCRETE`` aten ops and torchvision's NMS) whose input differs between the two inputs while its
  output is the same, so the signal was alive up to that step and collapsed there. A net that dies
  before any discrete step has no witness, and one that is also constant in train mode is not
  rescued either (a silent spiking net's threshold is a witness, but its train mode is silent too).
  A quantizer (``QUANTIZE``) needs only the witness: a straight-through codec rounds in train mode
  too, and the Lead's codec rule (2026-10-29, 10:44 AM PT) asks only that the input reach it.
- ``init_fix``: would an init change the rule allows make the forward non-degenerate? Zero weights
  (a zero-initialised output layer, residual zero-init) reset to the framework default, then fresh
  BatchNorm statistics filled by one train-mode no-grad pass (the BlendFace fix). The answer goes
  in the failure's message, so a retype or a 2b repair lane knows which change fits.

Both run with gradients off, never raise (a model that breaks on them answers ``None``) and leave
the model in eval mode. ``discrete_step`` restores every buffer its train-mode forwards changed;
``init_fix`` changes the model it is given, which the worker discards after a failure.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

# aten ops that turn a varying value into a discrete one: rounding and sign (learned codecs),
# comparisons (score thresholds), index reductions (argmax), bucketing (quantizers).
DISCRETE = frozenset({
    "round", "floor", "ceil", "trunc", "sign", "argmax", "argmin", "max", "min", "gt", "ge", "lt",
    "le", "bucketize", "searchsorted", "heaviside",
})
DISCRETE_OTHER = frozenset({("torchvision", "nms")})
# Quantizers: the Lead's codec rule (2026-10-29, 10:44 AM PT) asks only that the input reach the
# step, because a straight-through quantizer rounds in train mode too. Thresholds and argmax keep
# the train-mode test (a silent spiking net's threshold is reached by its input as well).
QUANTIZE = frozenset({"aten.round", "aten.floor", "aten.ceil", "aten.trunc", "aten.bucketize",
                      "aten.searchsorted"})

Forward = Callable[[tuple[Any, ...], dict[str, Any]], Any]
Call = tuple[str, str, str]  # op name, digest of its tensor inputs, digest of its tensor outputs


def _digest(value: Any) -> str:
    import torch
    from torch.utils._pytree import tree_leaves

    h = hashlib.blake2b(digest_size=16)
    for leaf in tree_leaves(value):
        if not isinstance(leaf, torch.Tensor):
            h.update(repr(leaf).encode())
            continue
        data = leaf.detach().cpu()
        h.update(f"{data.dtype}{tuple(data.shape)}".encode())
        if data.dtype == torch.bfloat16:
            data = data.float()
        if data.is_floating_point():
            data = data + 0.0  # -0.0 becomes 0.0: rounding a small negative latent is still zero
        try:
            h.update(data.contiguous().numpy().tobytes())
        except (TypeError, RuntimeError):  # quantized and other dtypes numpy cannot read
            h.update(repr(data).encode())
    return h.hexdigest()


def _discrete_calls(forward: Forward, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Call]:
    """The discrete ops one forward called, in order, with digests of their inputs and outputs."""

    from torch.utils._python_dispatch import TorchDispatchMode

    calls: list[Call] = []

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            name = func.overloadpacket.__name__.split(".")[-1]
            if (func.namespace == "aten" and name in DISCRETE) or (
                    (func.namespace, name) in DISCRETE_OTHER):
                calls.append((f"{func.namespace}.{name}", _digest((args, kwargs)), _digest(out)))
            return out

    with Record():
        forward(args, kwargs)
    return calls


def witness(first: Sequence[Call], second: Sequence[Call]) -> str | None:
    """The first discrete op whose input differs between two forwards while its output does not,
    comparing calls in order until the two forwards' op sequences part."""

    for (name, given, made), (other, given2, made2) in zip(first, second):
        if name != other:
            return None
        if given != given2 and made == made2:
            return name
    return None


def _buffers(model: Any) -> dict[str, Any]:
    return {name: buf.detach().clone() for name, buf in model.named_buffers()}


def _restore(model: Any, saved: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name in saved:
                buf.copy_(saved[name])


def _degenerate(out: Any, other: Any | None) -> bool:
    from menagerie.pipeline import worker

    return worker.nonfinite(out) is not None or worker.constant(out, other) is not None


def _train_alive(model: Any, forward: Forward, a: tuple[Any, dict], b: tuple[Any, dict]) -> bool:
    """Whether a train-mode forward is finite and differs between the two inputs; every buffer the
    train-mode forwards update (BatchNorm statistics) is put back, and the model left in eval."""

    saved = _buffers(model)
    try:
        model.train(True)
        first, second = forward(*a), forward(*b)
        return not _degenerate(first, None) and not _degenerate(second, None) and \
            not _degenerate(first, second)
    finally:
        model.train(False)
        _restore(model, saved)


def discrete_step(model: Any, forward: Forward, a: tuple[Any, dict],
                  b: tuple[Any, dict]) -> tuple[str | None, bool]:
    """The discrete op the eval collapse happens at (the witness, or ``None``) and whether the
    train-mode output is finite and changes with the input (``False`` without a witness, and
    untested at a ``QUANTIZE`` step, where the input reaching it is enough)."""

    import torch

    try:
        with torch.no_grad():
            step = witness(_discrete_calls(forward, *a), _discrete_calls(forward, *b))
    except Exception:  # noqa: BLE001 -- model code raises anything; no witness then
        return None, False
    if step is None or step in QUANTIZE:
        return step, False
    try:
        with torch.no_grad():
            return step, _train_alive(model, forward, a, b)
    except Exception:  # noqa: BLE001 -- a train-mode forward that raises is not alive
        return step, False


def discrete_note(step: str, *, alive: bool = True) -> str:
    """The one-line note of a forward kept because its collapse is only a discrete step: a
    threshold or argmax with a live train mode, or a quantizer its input reaches."""

    if alive:
        return (f"eval output collapses only at a discrete step ({step}); the train-mode output "
                "is finite and changes with the input (decisions.md 2026-10-29)")
    return (f"eval output collapses only at a quantization step ({step}); the values reaching it "
            "change with the input (decisions.md 2026-10-29, codec rule)")


def _zero_weights(model: Any) -> list[Any]:
    import torch

    return [m for m in model.modules() if isinstance(getattr(m, "weight", None), torch.nn.Parameter)
            and m.weight.numel() and not bool(m.weight.any()) and hasattr(m, "reset_parameters")]


def _batchnorms(model: Any) -> list[Any]:
    import torch

    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            and m.track_running_stats]


def _fill_batchnorm(model: Any, forward: Forward, a: tuple[Any, dict], norms: list[Any]) -> None:
    """One train-mode forward with each BatchNorm's momentum ``None``, so its statistics become
    exactly that batch's (BlendFace's fix), then eval again."""

    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
    try:
        model.train(True)
        forward(*a)
    finally:
        model.train(False)
        for m, momentum in zip(norms, momenta):
            m.momentum = momentum


def init_fix(model: Any, forward: Forward, a: tuple[Any, dict],
             b: tuple[Any, dict] | None) -> str | None:
    """Which init change the rule allows makes the eval forward finite and not constant: zero
    weights reset to the framework default, then fresh BatchNorm statistics; ``None`` when
    neither does (the model is changed either way)."""

    import torch

    def fixed() -> bool:
        first = forward(*a)
        return not _degenerate(first, forward(*b) if b is not None else None)

    try:
        with torch.no_grad():
            done = []
            zeros = _zero_weights(model)
            if zeros:
                torch.manual_seed(0)
                for m in zeros:
                    m.reset_parameters()
                done.append(f"{len(zeros)} all-zero weight layers reset to the framework default")
                if fixed():
                    return "; ".join(done)
            norms = _batchnorms(model)
            if norms:
                _fill_batchnorm(model, forward, a, norms)
                done.append(f"{len(norms)} BatchNorm layers' statistics filled by one train-mode "
                            "no-grad pass")
                if fixed():
                    return "; ".join(done)
    except Exception:  # noqa: BLE001 -- model code raises anything; no fix then
        return None
    return None
