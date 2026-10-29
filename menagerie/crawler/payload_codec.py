"""ONE canonical dtype vocabulary and ONE host-array conversion for every framework.

Two shipped defects live here, and both are the same shape: machinery written for torch
sitting on a backend-neutral path.

``BK-1`` (AREA_BACKENDS_fable, reproduced in the ledger). The award chain compared the
authored input contract against the worker-observed input signature with a normalizer that
stripped exactly one prefix -- ``str(...).lower().replace("torch.", "")``. The worker
recorded the observed dtype as ``str(tensor.dtype)`` verbatim, which is ``"torch.float32"``
for torch but ``"paddle.float32"`` for Paddle, ``"<dtype: 'float32'>"`` for TensorFlow and
``"mlx.core.float32"`` for MLX. Those never match ``"float32"``, so the contract conjunct
failed, and the supervised-failure fallback published ``failed:forward / mode-run`` with a
null ``mode_error`` -- an evidence-free model defect for a forward the work dir proves
completed. m7960 carries four such attempts.

``B3`` (AREA_BACKENDS_opus). ``modes._to_numpy`` required a ``.numpy()`` method, so
``jax.Array`` and ``mlx.core.array`` produced no value digest at all and every value-based
check silently degraded to "unverifiable" -- 100% for those frameworks, and already 6.1% of
succeeded attempts inside the *torch* lane, because ``.numpy()`` raises on ``bfloat16`` and
the ``RuntimeError`` was swallowed into a bare ``None``.

The remedy in both cases is a vocabulary and a codec, not another comparison site:

* :func:`canonical_dtype` maps every framework spelling into ONE closed enum, and
  :func:`canonical_dtype_text` is the total spelling used at capture time -- so the ledger
  itself becomes framework-neutral and every downstream comparer, including ones this
  module never sees, matches for every framework.
* :func:`to_host_array` converts a tensor of any framework to a host NumPy array through
  an explicit ladder, transports ``bfloat16`` as ``uint16`` plus a logical dtype the way
  TorchLens's own payload codecs do, and returns a TYPED reason when no host encoding
  exists instead of a bare ``None`` that reads identically to "not an array".

Nothing here imports a framework. Conversion sniffs only modules the process has ALREADY
imported, so the crawler's import-light paths stay import-light.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
try:  # python >= 3.11
    from enum import StrEnum
except ImportError:  # python 3.10: the oldest routed-env interpreter (legacy-torch
    # lock resolves py3.10). The worker closure must import there; the constants
    # shim is value-identical (str content == member value), differing only in
    # str()/format() rendering, which nothing identity-bearing uses (DX-3/RE-3).
    from menagerie.crawler.constants import StrEnum  # type: ignore[assignment]
from typing import Any, Mapping, Optional, cast

import numpy as np


class DtypeVocabularyError(ValueError):
    """Raised when a dtype spelling is outside the closed canonical vocabulary.

    It is a REFUSAL, never a default. The precedent is DEVICE-POLICY-PROSE: a producer
    string that steers machine behaviour must be refused typed rather than coerced into
    the one framework the code happens to know.
    """


class CanonicalDtype(StrEnum):
    """The closed, framework-neutral dtype vocabulary the ledger records.

    Members are the NumPy canonical names, which are also the spellings TorchLens's
    payload codecs use for logical dtypes, so a crawler record and a ``.tlspec`` bundle
    name the same thing. ``bfloat16`` has no NumPy dtype and is carried as a logical name
    over a ``uint16`` transport (see :class:`HostArray`).
    """

    BOOL = "bool"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"


#: Framework module prefixes stripped from an observed dtype spelling.
#:
#: Closed on purpose: an unrecognised prefix must survive into the record verbatim rather
#: than being guessed away, which is how ``replace("torch.", "")`` turned a Paddle dtype
#: into a mismatch instead of an unknown.
DTYPE_MODULE_PREFIXES: tuple[str, ...] = (
    "torch.",
    "paddle.",
    "mlx.core.",
    "mlx.",
    "jax.numpy.",
    "jaxlib.",
    "jnp.",
    "numpy.",
    "np.",
    "tensorflow.",
    "tf.",
    "tinygrad.dtype.",
    "tinygrad.",
    "dtypes.",
)

#: Closed non-canonical spellings, keyed by the framework that produces them.
#:
#: torch's short aliases are the trap: ``torch.float`` is float32 while ``np.dtype("float")``
#: is float64, so a NumPy fallback alone would silently change a recorded dtype.
DTYPE_ALIASES: Mapping[str, CanonicalDtype] = {
    # torch short aliases
    "half": CanonicalDtype.FLOAT16,
    "float": CanonicalDtype.FLOAT32,
    "double": CanonicalDtype.FLOAT64,
    "bfloat": CanonicalDtype.BFLOAT16,
    "bf16": CanonicalDtype.BFLOAT16,
    "long": CanonicalDtype.INT64,
    "int": CanonicalDtype.INT32,
    "short": CanonicalDtype.INT16,
    "char": CanonicalDtype.INT8,
    "byte": CanonicalDtype.UINT8,
    "cfloat": CanonicalDtype.COMPLEX64,
    "cdouble": CanonicalDtype.COMPLEX128,
    "complex_float": CanonicalDtype.COMPLEX64,
    "complex_double": CanonicalDtype.COMPLEX128,
    "boolean": CanonicalDtype.BOOL,
}
# Bare width codes (``i8``, ``f16``, ``u4``) are deliberately ABSENT. They are genuinely
# ambiguous -- NumPy reads the digits as BYTES (``i8`` is int64, ``f16`` is float128) while
# most frameworks read them as BITS (``i8`` is int8) -- and resolving an ambiguity by
# picking one convention is exactly the silent-default class this module removes. They
# refuse, and widening the vocabulary is a reviewed change.

#: ``str(tf.DType)`` renders as ``<dtype: 'float32'>``; nothing else in the pipeline does.
_TF_DTYPE_REPR = re.compile(r"^<dtype:\s*'([^']+)'>$")

#: NumPy char codes (``<f4``, ``>f8``, ``|b1``). An explicit byteorder character is
#: REQUIRED: it is what proves the spelling is NumPy's, whose digits mean bytes.
_NUMPY_CHAR_CODE = re.compile(r"^[<>=|][biufc]\d+$")

_BFLOAT16_TRANSPORT_DTYPE = CanonicalDtype.UINT16
"""bfloat16 has no NumPy dtype; it rides as raw ``uint16`` with its logical name kept."""


def _normalized_spelling(value: object) -> str:
    """Return one observed dtype rendered as lowercase text with no module prefix."""

    text = str(value).strip()
    match = _TF_DTYPE_REPR.match(text)
    if match is not None:
        text = match.group(1)
    text = text.strip().lower()
    for prefix in DTYPE_MODULE_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def canonical_dtype(value: object) -> CanonicalDtype:
    """Map any framework dtype spelling onto the closed canonical vocabulary.

    Parameters
    ----------
    value:
        A dtype object or its text, in any framework's spelling.

    Returns
    -------
    CanonicalDtype
        The one canonical member for this dtype.

    Raises
    ------
    DtypeVocabularyError
        When the spelling is outside the closed vocabulary. Refusing is deliberate:
        widening the vocabulary is a reviewed change, and a silent default is the class
        of bug this module exists to remove.
    """

    text = _normalized_spelling(value)
    if not text:
        raise DtypeVocabularyError("empty dtype spelling")
    try:
        return CanonicalDtype(text)
    except ValueError:
        pass
    alias = DTYPE_ALIASES.get(text)
    if alias is not None:
        return alias
    if _NUMPY_CHAR_CODE.match(text):
        try:
            resolved = np.dtype(text).name
        except TypeError as exc:
            raise DtypeVocabularyError(f"unsupported dtype spelling: {text!r}") from exc
        try:
            return CanonicalDtype(resolved)
        except ValueError as exc:
            raise DtypeVocabularyError(f"unsupported dtype spelling: {text!r}") from exc
    raise DtypeVocabularyError(f"unsupported dtype spelling: {text!r}")


def canonical_dtype_text(value: object) -> Optional[str]:
    """Return the canonical dtype text, or the ORIGINAL spelling when unmappable.

    This is the total function capture uses. Totality matters because a recorded dtype is
    descriptive metadata that must never block a capture; honesty matters more, so an
    unmappable spelling is preserved verbatim rather than coerced to a canonical member
    it is not. A verbatim survivor still fails :func:`dtype_matches`, which is the
    tripwire's correct behaviour: an unknown dtype is unknown, not equal.

    Parameters
    ----------
    value:
        A dtype object or its text, or ``None``.

    Returns
    -------
    str | None
        Canonical name, the original text, or ``None`` for an absent dtype.
    """

    if value is None:
        return None
    try:
        return canonical_dtype(value).value
    except DtypeVocabularyError:
        return str(value)


def dtype_matches(expected: object, observed: object) -> bool:
    """Compare an authored dtype against an observed dtype, framework-neutrally.

    Parameters
    ----------
    expected:
        Authored contract dtype, in any spelling.
    observed:
        Worker-observed dtype, in any spelling.

    Returns
    -------
    bool
        True only when both canonicalize to the same member, or when both are outside the
        vocabulary and textually identical. Two unknowns that merely look similar are NOT
        equal.
    """

    try:
        return canonical_dtype(expected) is canonical_dtype(observed)
    except DtypeVocabularyError:
        expected_text = canonical_dtype_text(expected)
        observed_text = canonical_dtype_text(observed)
        if expected_text is None or observed_text is None:
            return False
        return _normalized_spelling(expected_text) == _normalized_spelling(observed_text)


@dataclass(frozen=True)
class HostArray:
    """One framework tensor realized on the host, with its logical dtype retained.

    Parameters
    ----------
    array:
        The host NumPy array. For ``bfloat16`` this is the raw ``uint16`` transport, which
        is exactly how ``torchlens._io.payload_codec`` moves bf16, so byte digests stay
        stable and lossless.
    logical_dtype:
        Canonical name of the SOURCE dtype -- ``"bfloat16"`` even when the transport is
        ``uint16``.
    transport_dtype:
        Canonical name of ``array.dtype``.
    converter:
        The ladder rung that produced the array, for receipt provenance.
    """

    array: np.ndarray[Any, Any]
    logical_dtype: Optional[str]
    transport_dtype: str
    converter: str


@dataclass(frozen=True)
class HostArrayRefusal:
    """A tensor-like leaf with NO host encoding, and the typed reason why.

    A bare ``None`` was indistinguishable from "this leaf is not an array at all", so a
    whole framework's missing digests looked exactly like ordinary Python outputs and
    nobody was tracking the 6.1%. The reason string is
    ``"<framework-or-module>:<qualname>:<cause>"`` so a census can group it.
    """

    reason: str


def _module_root(value: object) -> str:
    """Return the top-level module name of a value's type."""

    return type(value).__module__.split(".", 1)[0]


def _qualified_type(value: object) -> str:
    """Return the module-qualified type name of a value."""

    return f"{type(value).__module__}.{type(value).__qualname__}"


def _is_tensor_like(value: object) -> bool:
    """Return whether a leaf claims both a shape and a dtype.

    The gate is deliberately conservative. ``np.asarray`` succeeds on ints, strings and
    nested lists, so an unguarded conversion would silently reclassify ordinary Python
    outputs as arrays and change every recorded signature.
    """

    return hasattr(value, "shape") and hasattr(value, "dtype")


def _bfloat16_transport(current: Any) -> Optional[np.ndarray[Any, Any]]:
    """Bitcast a bfloat16 tensor to ``uint16`` without importing a framework.

    Only frameworks already present in ``sys.modules`` are consulted: a bf16 tensor cannot
    exist unless its framework is imported, so this never triggers an import and never
    fires on a foreign object.
    """

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(current, getattr(torch, "Tensor", ())):
        try:
            return np.asarray(cast(Any, current).view(torch.uint16).numpy())
        except (TypeError, ValueError, RuntimeError):
            return None
    for module_name, caster in (("paddle", "astype"), ("mlx.core", "view")):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        try:
            viewer = getattr(current, caster, None)
            if not callable(viewer):
                continue
            candidate = viewer(getattr(module, "uint16"))
            numpy_method = getattr(candidate, "numpy", None)
            if callable(numpy_method):
                return np.asarray(numpy_method())
            return np.asarray(candidate)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
    return None


def to_host_array(value: object) -> Optional[HostArray] | HostArrayRefusal:
    """Realize any framework tensor as a host array, or say typed why not.

    Parameters
    ----------
    value:
        Candidate output or input leaf.

    Returns
    -------
    HostArray or HostArrayRefusal or None
        ``HostArray`` on success, ``HostArrayRefusal`` when the leaf IS tensor-like but has
        no host encoding, and ``None`` when the leaf is not array-like at all. The
        three-way return is the point: the old two-way ``ndarray | None`` conflated the
        last two.
    """

    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            return HostArrayRefusal(f"numpy:{_qualified_type(value)}:object-dtype")
        name = canonical_dtype_text(array.dtype)
        return HostArray(array, name, str(name), "numpy-direct")

    if not _is_tensor_like(value):
        return None

    logical = canonical_dtype_text(getattr(value, "dtype", None))
    current: Any = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        try:
            current = detach()
        except (TypeError, ValueError, RuntimeError):
            pass
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        try:
            current = cpu()
        except (TypeError, ValueError, RuntimeError):
            pass

    if logical == CanonicalDtype.BFLOAT16.value:
        transported = _bfloat16_transport(current)
        if transported is not None:
            return HostArray(
                transported,
                CanonicalDtype.BFLOAT16.value,
                _BFLOAT16_TRANSPORT_DTYPE.value,
                "bfloat16-as-uint16",
            )
        return HostArrayRefusal(f"{_module_root(value)}:{_qualified_type(value)}:bfloat16")

    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        try:
            array = np.asarray(numpy_method())
        except (TypeError, ValueError, RuntimeError):
            array = None
        if array is not None:
            if array.dtype.hasobject:
                return HostArrayRefusal(
                    f"{_module_root(value)}:{_qualified_type(value)}:object-dtype"
                )
            return HostArray(array, logical, str(canonical_dtype_text(array.dtype)), "numpy-method")

    # JAX and MLX expose no `.numpy()`; TorchLens converts both with `asarray`, and so do
    # their own documented host paths. `__array__`/`__dlpack__`/`__array_interface__` is
    # the positive identification that such a path exists -- membership proven, not
    # residual (catalog C-25).
    if any(
        hasattr(current, attribute)
        for attribute in ("__array__", "__dlpack__", "__array_interface__")
    ):
        try:
            array = np.asarray(current)
        except (TypeError, ValueError, RuntimeError, BufferError) as exc:
            return HostArrayRefusal(
                f"{_module_root(value)}:{_qualified_type(value)}:{type(exc).__name__}"
            )
        if array.dtype.hasobject:
            return HostArrayRefusal(f"{_module_root(value)}:{_qualified_type(value)}:object-dtype")
        return HostArray(array, logical, str(canonical_dtype_text(array.dtype)), "asarray")

    return HostArrayRefusal(f"{_module_root(value)}:{_qualified_type(value)}:no-host-encoding")


def host_array_or_reason(
    value: object,
) -> tuple[Optional[np.ndarray[Any, Any]], Optional[str]]:
    """Return ``(array, unencodable_reason)`` for one leaf.

    Exactly one element is ever non-``None`` for a tensor-like leaf; a non-array leaf
    returns ``(None, None)``. Callers that record receipts write the reason so a missing
    value digest names its cause instead of vanishing.

    Parameters
    ----------
    value:
        Candidate leaf.

    Returns
    -------
    tuple[numpy.ndarray | None, str | None]
        The host array, or the typed reason no host encoding exists.
    """

    result = to_host_array(value)
    if result is None:
        return None, None
    if isinstance(result, HostArrayRefusal):
        return None, result.reason
    return result.array, None
