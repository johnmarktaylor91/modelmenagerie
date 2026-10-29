"""Shape-aware structural digests for menagerie migration parity.

Phase 2b of the menagerie redesign introduces two explicit hash entry points
for two different questions, following the TWO-HASH DISTINCTION in
``.research/menagerie-redesign/CONVERGENCE_DECISIONS.md``:

* ``structural_fingerprint`` is the migration-fidelity fingerprint. It is
  shape-aware because it returns TorchLens' raw-event shape hash, which includes
  output shapes and dtypes. Phase-2c uses it to prove a migrated recipe builds
  the same dimensional architecture.
* ``architecture_distinctness_hash`` is the architecture-count hash. It returns
  ``Trace.graph_shape_hash``, which is intentionally shape-blind and answers the
  coarser distinct-design counting question.

This module deliberately does not attempt the deferred full module-hyperparameter
digest. The raw-event output shape/dtype hash is the safety net needed to catch
the layer-width corruption called out by red-team item R5.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

import torch
import torchlens as tl
from torchlens.options import CaptureOptions


def _to_cpu(value: Any) -> Any:
    """Move tensors inside an input object to CPU.

    Parameters
    ----------
    value:
        Arbitrary example input object.

    Returns
    -------
    Any
        Input object with tensors moved to CPU where possible.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


# Fixed seed for the digest trace's capture (see ``_trace_for_digest``).
_DIGEST_TRACE_SEED = 0

# Versioned identity of HOW a digest trace is captured. Both persisted identity hashes
# (``graph_shape_hash`` / ``structural_fingerprint_hash``) are functions of the captured
# taken-path graph, and that graph can depend on the capture policy (a fixed forward
# seed changes which branch an RNG-steered forward takes). Any change to this policy
# -- the seed, ``inference_only``, RNG handling -- MUST bump the ``digest_capture_vN``
# prefix; trace-summary currency binds on this exact string, so every row captured
# under an older policy goes stale and is recomputed instead of staying authoritative.
DIGEST_CAPTURE_POLICY = (
    f"digest_capture_v1:inference_only;random_seed={_DIGEST_TRACE_SEED};host_rng_restored"
)


@contextmanager
def _restored_host_rng_state() -> Iterator[None]:
    """Restore the Python, NumPy and torch global RNG states on exit."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def _trace_for_digest(model: Any, example_input: Any) -> Any:
    """Trace a model on CPU with inference-only capture.

    Parameters
    ----------
    model:
        Model or callable accepted by ``torchlens.trace``.
    example_input:
        Example input accepted by ``torchlens.trace``.

    Returns
    -------
    Any
        TorchLens trace object.
    """

    if hasattr(model, "to"):
        model = model.to("cpu")
    if hasattr(model, "eval"):
        model.eval()
    # Some architectures draw from the global RNG INSIDE the forward with structural
    # consequences (3DSSD's furthest-point sampling seeds its first centroid with
    # ``torch.randint``, so ball-query padding and thus op shapes vary per call). A digest
    # is a function of (model, input), so the capture runs under a fixed ``random_seed``
    # (TorchLens seeds Python, NumPy and torch for the forward) and every engine's state is
    # restored afterwards so the caller's RNG streams are untouched.
    with _restored_host_rng_state():
        return tl.trace(
            model,
            _to_cpu(example_input),
            capture=CaptureOptions(inference_only=True, random_seed=_DIGEST_TRACE_SEED),
        )


def structural_fingerprint(model: Any, example_input: Any) -> str:
    """Return the shape-aware migration-fidelity fingerprint.

    Parameters
    ----------
    model:
        Model or callable to trace on CPU.
    example_input:
        Example input for the model.

    Returns
    -------
    str
        SHA-256 raw-event shape hash from the TorchLens trace.

    Raises
    ------
    RuntimeError
        If the trace does not expose a raw-event shape hash.
    """

    trace = _trace_for_digest(model, example_input)
    raw_event_hash = getattr(trace, "_raw_event_shape_hash", None)
    if not raw_event_hash:
        raise RuntimeError("TorchLens trace did not expose _raw_event_shape_hash")
    return str(raw_event_hash)


def architecture_distinctness_hash(model: Any, example_input: Any) -> str:
    """Return the shape-blind architecture distinctness hash.

    Parameters
    ----------
    model:
        Model or callable to trace on CPU.
    example_input:
        Example input for the model.

    Returns
    -------
    str
        ``Trace.graph_shape_hash`` for distinct-architecture counting.

    Raises
    ------
    RuntimeError
        If the trace does not expose ``graph_shape_hash``.
    """

    trace = _trace_for_digest(model, example_input)
    graph_hash = getattr(trace, "graph_shape_hash", None)
    if not graph_hash:
        raise RuntimeError("TorchLens trace did not expose graph_shape_hash")
    return str(graph_hash)
