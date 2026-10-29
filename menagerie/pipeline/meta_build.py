"""Structure-only builds on the meta device, with a sized CPU fallback.

A constructor that reads a value back from a tensor (segmentation_models_pytorch's MiT encoders
call ``.item()`` on ``torch.linspace`` for their drop-path schedule) cannot run on the meta device.
Such a model is sized by a probe build that keeps its parameters on meta and makes only small
factory tensors on the CPU; it is then built on the CPU when that fits the declared memory
(``MENAGERIE_MEMORY_GB``, ``host_budget``), and typed otherwise. The model built is always the
recipe's own; nothing is substituted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from menagerie import host_budget
from menagerie.pipeline.harvest_build import on_device
from menagerie.pipeline.records import RecordError

# Factory tensors at most this many elements are made on the CPU by the probe (schedules, scalars).
PROBE_CPU_NUMEL = 1 << 16
# A CPU build holds the fp32 weights plus initialisation temporaries and the process itself.
CPU_BUILD_FACTOR = 3.0


class MetaBuildRefused(RecordError):
    """A model that cannot build on meta and has no safe CPU build; ``cls`` is its give-up."""

    def __init__(self, cls: str, reason: str) -> None:
        super().__init__(f"{cls}: {reason}")
        self.cls, self.reason = cls, reason


def is_meta_error(exc: BaseException) -> bool:
    """Whether ``exc`` is the meta device refusing a data access (``.item()``, a copy out)."""

    return isinstance(exc, (RuntimeError, NotImplementedError)) and \
        "meta tensor" in str(exc).casefold()


def probe_mode() -> Any:
    import torch
    from torch.overrides import TorchFunctionMode
    from torch.utils._device import _device_constructors

    factories = _device_constructors()

    class SmallOnCpu(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
            kwargs = kwargs or {}
            made = func(*args, **kwargs)
            if (func in factories and isinstance(made, torch.Tensor) and
                    made.device.type == "meta" and made.numel() <= PROBE_CPU_NUMEL):
                return func(*args, **{**kwargs, "device": "cpu"})
            return made

    return SmallOnCpu()


def probe_gib(build: Callable[[], Any]) -> float:
    """The fp32 parameter size of ``build()``'s model, from a meta build with small CPU tensors."""

    import torch

    try:
        with torch.random.fork_rng(devices=[]), on_device("meta"), probe_mode():
            model = build()
    except Exception as exc:
        raise MetaBuildRefused("construct_error", "no meta build and no sized probe: "
                               f"{type(exc).__name__}: {str(exc)[:160]}") from exc
    if not isinstance(model, torch.nn.Module):
        raise MetaBuildRefused("construct_error", "the meta probe built no torch module")
    return sum(p.numel() for p in model.parameters()) * 4 / 2**30


def cpu_build_fits(gib: float) -> bool:
    """Whether a CPU build of ``gib`` fp32 weights fits this job's declared memory."""

    return gib * CPU_BUILD_FACTOR <= host_budget.declared_memory_gb()


def build_structure(build: Callable[[], Any]) -> Any:
    """``build()`` on meta; on a meta data access, on the CPU when sized safe, else refused."""

    import torch

    try:
        with on_device("meta"):
            return build()
    except Exception as exc:
        if not is_meta_error(exc):
            raise
    gib = probe_gib(build)
    if not cpu_build_fits(gib):
        raise MetaBuildRefused("memory_cap", f"needs a CPU build of {gib:.2f} GiB of weights, "
                               f"over {host_budget.MEMORY_ENV}={host_budget.declared_memory_gb()}"
                               f" / {CPU_BUILD_FACTOR:g}")
    # The caller's random state is untouched, as by the meta build it replaces.
    with torch.random.fork_rng(devices=[]), on_device("cpu"):
        return build()
