"""Menagerie's one door to TorchLens's forward validator (``docs/torchlens-boundary.md``).

Every validation Menagerie runs goes through ``validate_forward``. The worker needs more than the
verdict: an intra-op thread pin for its single-thread retry, and the validation trace itself (op
count, graph hash, ``validation_replay_status``), read by an observer before TorchLens frees it,
or ``None`` when TorchLens refused a precondition before building one.

Route, decided once per call from what the installed TorchLens offers:

1. Public ``tl.validate(scope="forward")`` when it accepts ``trace_observer`` (TorchLens 3.0.1
   does; 2.38.1 does not). ``num_threads`` goes as its keyword when
   accepted, else as a ``torch.set_num_threads`` pin around the call, which TorchLens confirmed is
   equivalent (the harness pins once at entry, before any forward).
2. Else the validator in ``torchlens.user_funcs``, the module that carried it through 2.37
   (the only private TorchLens call left in Menagerie's validation path).
3. Else ``TorchLensBoundaryError``: from 2.38 the validator is in a private module and the public
   door returns only a bool, so the worker's ledger fields cannot be read through public API.
   Menagerie never imports a ``torchlens._*`` module for it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

Observer = Callable[[Any], None]

PUBLIC_OBSERVER = "trace_observer"


class TorchLensBoundaryError(RuntimeError):
    """The installed TorchLens offers no public route to what Menagerie's validation reads."""


def validate_forward(model: Any, inputs: Any, input_kwargs: Mapping[str, Any] | None = None, *,
                     seed: int | None = None, num_threads: int | None = None,
                     observer: Observer | None = None, validate_metadata: bool = True,
                     save_options: Mapping[str, Any] | None = None) -> bool:
    """Run TorchLens's forward validation; ``observer`` sees the validation trace or ``None``.

    ``inputs`` and ``input_kwargs`` are passed as the caller built them; ``seed`` ``None`` leaves
    TorchLens's default. ``save_options`` (``output_device``, ``save_budget``: a GPU capture's
    saves in host memory under an absolute budget, ``worker.gpu_save_options``) go to the
    validator as keywords; empty or ``None`` passes none. The verdict is TorchLens's own bool.
    """

    import torchlens as tl

    params = inspect.signature(tl.validate).parameters
    if PUBLIC_OBSERVER in params:
        return _public(tl, params, model, inputs, input_kwargs, seed=seed,
                       num_threads=num_threads, observer=observer,
                       validate_metadata=validate_metadata, save_options=save_options or {})
    validator = _legacy_validator()
    # The 2.36 validator takes output_device= and save_budget= (docs/torchlens-boundary.md).
    keywords: dict[str, Any] = {"validate_metadata": validate_metadata, **(save_options or {})}
    if seed is not None:
        keywords["random_seed"] = seed
    if num_threads is not None:
        keywords["num_threads"] = num_threads
    if observer is not None:
        keywords["_trace_observer"] = observer
    if input_kwargs is None:
        return bool(validator(model, inputs, **keywords))
    return bool(validator(model, inputs, input_kwargs, **keywords))


def _public(tl: Any, params: Mapping[str, Any], model: Any, inputs: Any,
            input_kwargs: Mapping[str, Any] | None, *, seed: int | None,
            num_threads: int | None, observer: Observer | None,
            validate_metadata: bool, save_options: Mapping[str, Any]) -> bool:
    refused = sorted(set(save_options) - set(params))
    if refused:
        raise TorchLensBoundaryError(f"tl.validate takes no {', '.join(refused)}=")
    keywords: dict[str, Any] = {"scope": "forward", "validate_metadata": validate_metadata,
                                PUBLIC_OBSERVER: observer, **save_options}
    if seed is not None:
        keywords["seed" if "seed" in params else "random_seed"] = seed
    pin = num_threads is not None and "num_threads" not in params
    if num_threads is not None and not pin:
        keywords["num_threads"] = num_threads
    if not pin:
        return bool(tl.validate(model, inputs, input_kwargs, **keywords))
    import torch

    prior = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        return bool(tl.validate(model, inputs, input_kwargs, **keywords))
    finally:
        torch.set_num_threads(prior)


def _legacy_validator() -> Callable[..., Any]:
    """The validator from ``torchlens.user_funcs`` (TorchLens 2.37 and earlier)."""

    try:
        from torchlens.user_funcs import _validate_forward_pass_torch
    except ImportError as exc:
        import torchlens as tl

        raise TorchLensBoundaryError(
            f"TorchLens {tl.__version__}: tl.validate takes no {PUBLIC_OBSERVER}= and "
            "torchlens.user_funcs is gone, so the validation trace's op count, graph hash and "
            "replay status have no public route; Menagerie cannot validate at this pin "
            "(docs/torchlens-boundary.md)") from exc
    return _validate_forward_pass_torch


def failure_for(trace: Any) -> Any:
    """TorchLens's recorded failure for a validation trace, or for a refusal when ``None``."""

    import torchlens as tl

    return tl.validation.get_validation_failure(trace)


def last_failure() -> Any:
    """The failure TorchLens recorded for the last validation run, or ``None``."""

    import torchlens as tl

    return tl.validation.last_validation_failure()


def failure_check(failure: Any) -> str | None:
    """The check a failure names: ``check`` through 2.37, ``check_name`` from 2.38."""

    if failure is None:
        return None
    return getattr(failure, "check", None) or getattr(failure, "check_name", None)
