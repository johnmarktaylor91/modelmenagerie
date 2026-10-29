"""Meaningful runtime-mode detection and train/eval divergence classification."""

from __future__ import annotations

import hashlib
import inspect
import struct
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from menagerie.crawler.constants import RunMode
from menagerie.crawler.payload_codec import canonical_dtype_text, host_array_or_reason


@dataclass(frozen=True)
class DivergenceResult:
    """Train/eval divergence classification and concise evidence.

    Parameters
    ----------
    classification:
        ``none``, ``statistical``, or ``structural``.
    evidence:
        Mechanical explanation based on output or graph signatures.
    """

    classification: str
    evidence: str


def _shape(value: object) -> Optional[list[int]]:
    """Return a concrete tensor/array shape when exposed.

    Parameters
    ----------
    value:
        Candidate output leaf.

    Returns
    -------
    list[int] | None
        Concrete shape or ``None`` for a non-array leaf.
    """

    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dimension) for dimension in shape]
    except (TypeError, ValueError):
        return None


def _dtype(value: object) -> Optional[str]:
    """Return the CANONICAL tensor/array dtype when exposed.

    This recorded exactly ``str(tensor.dtype)``, which is a different string per framework
    for the same dtype: ``"torch.float32"``, ``"paddle.float32"``,
    ``"<dtype: 'float32'>"``, ``"mlx.core.float32"``. The award chain then compared it to
    the authored contract with a normalizer that stripped only ``"torch."``, so every
    completed non-torch forward failed the contract conjunct and was published as an
    evidence-free ``failed:forward / mode-run`` -- reproduced in the ledger on m7960, whose
    work dir proves both forwards completed (AREA_BACKENDS_fable BK-1).

    Canonicalizing HERE, at capture, is the structural fix rather than a fix at one
    comparison site: the recorded fact becomes framework-neutral, so every comparer
    downstream -- including ones owned by other modules -- matches for every framework. An
    unmappable spelling is preserved verbatim rather than coerced, so nothing is laundered.

    Parameters
    ----------
    value:
        Candidate output leaf.

    Returns
    -------
    str | None
        Canonical dtype name, the original spelling when unmappable, or ``None``.
    """

    dtype = getattr(value, "dtype", None)
    return None if dtype is None else canonical_dtype_text(dtype)


def output_signature(value: object, path: str = "output") -> dict[str, Any]:
    """Describe an output pytree without retaining tensor payloads.

    Parameters
    ----------
    value:
        Arbitrary nested output.
    path:
        Root path used in leaf locations.

    Returns
    -------
    dict[str, Any]
        Deterministic tree and leaf signature.
    """

    leaves: list[dict[str, Any]] = []

    def visit(item: object, item_path: str) -> Any:
        """Visit one output node and append leaf metadata.

        Parameters
        ----------
        item:
            Current output node.
        item_path:
            Stable pytree path.

        Returns
        -------
        Any
            JSON-compatible tree descriptor.
        """

        if isinstance(item, Mapping):
            keys = sorted(item, key=str)
            return {str(key): visit(item[key], f"{item_path}.{key}") for key in keys}
        if isinstance(item, tuple):
            return {
                "tuple": [visit(child, f"{item_path}[{index}]") for index, child in enumerate(item)]
            }
        if isinstance(item, list):
            return {
                "list": [visit(child, f"{item_path}[{index}]") for index, child in enumerate(item)]
            }
        leaf = {
            "path": item_path,
            "kind": "tensor" if _shape(item) is not None else "python",
            "shape": _shape(item),
            "dtype": _dtype(item),
            "device": str(getattr(item, "device", "")) or None,
            "python_type": f"{type(item).__module__}.{type(item).__qualname__}",
        }
        leaves.append(leaf)
        return {"leaf": len(leaves) - 1}

    tree = visit(value, path)
    return {"tree": tree, "leaves": leaves}


def _to_numpy(value: object) -> Optional[np.ndarray[Any, Any]]:
    """Convert a tensor-like leaf to a detached host NumPy array.

    Delegates to the shared framework-neutral codec. The shipped ladder was
    ``detach -> cpu -> numpy() -> isinstance(np.ndarray)``, and it required a ``.numpy()``
    method: ``jax.Array`` and ``mlx.core.array`` expose none, so their value digests were
    silently ``None`` and every value-based check (train/eval divergence, cold-run
    determinism, family-variant equality) degraded to "unverifiable" for whole frameworks.
    The same hole was already live inside the TORCH lane, because ``.numpy()`` raises on
    ``bfloat16`` and the ``RuntimeError`` was swallowed into the same bare ``None``
    (AREA_BACKENDS_opus B3; 42 of 684 succeeded attempts already carry a null digest).

    Use :func:`unencodable_reason` when a receipt needs to say WHY a digest is absent: a
    reasoned absence and "not an array at all" are different facts and must not share a
    spelling.

    Parameters
    ----------
    value:
        Tensor-like leaf.

    Returns
    -------
    numpy.ndarray | None
        Host array, or ``None`` when no host encoding exists.
    """

    array, _reason = host_array_or_reason(value)
    return array


def unencodable_reason(value: object) -> Optional[str]:
    """Return the typed reason one tensor-like leaf has no host encoding.

    Parameters
    ----------
    value:
        Candidate output leaf.

    Returns
    -------
    str | None
        ``"<framework>:<qualname>:<cause>"`` when the leaf is tensor-like and unencodable;
        ``None`` when it converted, or when it is not array-like at all.
    """

    _array, reason = host_array_or_reason(value)
    return reason


def _flatten(value: object) -> list[object]:
    """Flatten an output pytree in signature order.

    Parameters
    ----------
    value:
        Arbitrary nested output.

    Returns
    -------
    list[object]
        Ordered leaf values.
    """

    if isinstance(value, Mapping):
        return [leaf for key in sorted(value, key=str) for leaf in _flatten(value[key])]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


class _UnstableOutputValue(TypeError):
    """Raised internally when an output has no process-independent encoding."""


def _frame(tag: bytes, payload: bytes) -> bytes:
    """Frame one stable output value without ambiguous concatenation.

    Parameters
    ----------
    tag:
        Closed semantic type tag.
    payload:
        Stable value bytes.

    Returns
    -------
    bytes
        Length-delimited type and payload bytes.
    """

    return struct.pack(">I", len(tag)) + tag + struct.pack(">Q", len(payload)) + payload


def _type_name(value: object) -> str:
    """Return the stable qualified Python type name for a structured value.

    Parameters
    ----------
    value:
        Value whose type participates in an encoding.

    Returns
    -------
    str
        Module-qualified type name.
    """

    return f"{type(value).__module__}.{type(value).__qualname__}"


def _stable_array_bytes(array: np.ndarray[Any, Any]) -> bytes:
    """Encode a non-object NumPy array in canonical little-endian C order.

    Parameters
    ----------
    array:
        Array converted from a supported tensor or supplied directly.

    Returns
    -------
    bytes
        Versioned dtype, shape, and exact value bytes.

    Raises
    ------
    _UnstableOutputValue
        If the array contains Python objects.
    """

    if array.dtype.hasobject:
        raise _UnstableOutputValue("object arrays have no stable output encoding")
    contiguous = np.ascontiguousarray(array)
    byteorder = contiguous.dtype.byteorder
    if byteorder == ">" or (byteorder == "=" and not np.little_endian):
        contiguous = contiguous.byteswap().view(contiguous.dtype.newbyteorder("<"))
    elif byteorder == "=":
        contiguous = contiguous.view(contiguous.dtype.newbyteorder("<"))
    dtype = contiguous.dtype.str.encode("ascii")
    shape = b"".join(struct.pack(">q", int(dimension)) for dimension in contiguous.shape)
    payload = (
        _frame(b"dtype", dtype)
        + _frame(b"rank", struct.pack(">I", contiguous.ndim))
        + _frame(b"shape", shape)
        + _frame(b"data", contiguous.tobytes(order="C"))
    )
    return _frame(b"array-v2", payload)


def _stable_output_bytes(value: object, active: set[int]) -> bytes:
    """Recursively encode one explicitly supported output value.

    Parameters
    ----------
    value:
        Output value or nested child.
    active:
        Object identities on the active recursion path.

    Returns
    -------
    bytes
        Process-independent type-tagged bytes.

    Raises
    ------
    _UnstableOutputValue
        If a leaf is unsupported, cyclic, or otherwise unstable.
    """

    array = _to_numpy(value)
    if array is not None:
        return _stable_array_bytes(array)
    if value is None:
        return _frame(b"none", b"")
    if isinstance(value, bool):
        return _frame(b"bool", b"1" if value else b"0")
    if isinstance(value, int):
        return _frame(b"int", str(value).encode("ascii"))
    if isinstance(value, float):
        return _frame(b"float64", struct.pack(">d", value))
    if isinstance(value, complex):
        return _frame(b"complex128", struct.pack(">dd", value.real, value.imag))
    if isinstance(value, str):
        return _frame(b"utf8", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _frame(b"bytes", value)

    identity = id(value)
    if identity in active:
        raise _UnstableOutputValue("cyclic outputs have no stable output encoding")
    active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            encoded_fields = b"".join(
                _frame(
                    field.name.encode("utf-8"),
                    _stable_output_bytes(getattr(value, field.name), active),
                )
                for field in fields(value)
            )
            return _frame(
                b"dataclass-v1",
                _frame(b"type", _type_name(value).encode("utf-8"))
                + _frame(b"fields", encoded_fields),
            )
        named_fields = getattr(type(value), "_fields", None)
        if (
            isinstance(value, tuple)
            and isinstance(named_fields, tuple)
            and all(isinstance(name, str) for name in named_fields)
        ):
            encoded_fields = b"".join(
                _frame(name.encode("utf-8"), _stable_output_bytes(getattr(value, name), active))
                for name in named_fields
            )
            return _frame(
                b"namedtuple-v1",
                _frame(b"type", _type_name(value).encode("utf-8"))
                + _frame(b"fields", encoded_fields),
            )
        if isinstance(value, Mapping):
            encoded_items: list[tuple[bytes, bytes]] = []
            for key, child in value.items():
                encoded_key = _stable_output_bytes(key, active)
                encoded_items.append((encoded_key, _stable_output_bytes(child, active)))
            encoded_items.sort(key=lambda item: item[0])
            if any(
                encoded_items[index - 1][0] == encoded_items[index][0]
                for index in range(1, len(encoded_items))
            ):
                raise _UnstableOutputValue("mapping keys collide under stable output encoding")
            payload = b"".join(
                _frame(b"key", key) + _frame(b"value", child) for key, child in encoded_items
            )
            return _frame(b"mapping-v1", payload)
        if isinstance(value, tuple):
            return _frame(
                b"tuple-v1",
                b"".join(_frame(b"item", _stable_output_bytes(child, active)) for child in value),
            )
        if isinstance(value, list):
            return _frame(
                b"list-v1",
                b"".join(_frame(b"item", _stable_output_bytes(child, active)) for child in value),
            )
    finally:
        active.remove(identity)
    raise _UnstableOutputValue(f"unsupported output leaf type: {_type_name(value)}")


def output_value_sha256(value: object) -> Optional[str]:
    """Hash exact output values using a versioned stable encoder.

    Parameters
    ----------
    value:
        Arbitrary nested output whose structure is separately captured by
        :func:`output_signature`.

    Returns
    -------
    str | None
        Domain-separated SHA-256, or ``None`` when any leaf lacks an explicitly
        supported process-independent representation.
    """

    try:
        encoded = _stable_output_bytes(value, set())
    except _UnstableOutputValue:
        return None
    digest = hashlib.sha256(b"menagerie-output-values-v2\0")
    digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def classify_observed_mode_receipts(
    train_receipt: Mapping[str, object], eval_receipt: Mapping[str, object]
) -> Optional[DivergenceResult]:
    """Classify independently captured train/eval receipt observations.

    Parameters
    ----------
    train_receipt, eval_receipt:
        Per-mode worker receipts carrying output structure and optional value digests.

    Returns
    -------
    DivergenceResult | None
        Mechanical classification, or ``None`` when equal structures lack the value
        digests required to distinguish ``none`` from ``statistical``.
    """

    train_signature = train_receipt.get("output_signature")
    eval_signature = eval_receipt.get("output_signature")
    if train_signature != eval_signature:
        return DivergenceResult(
            "structural", "train and eval output trees, types, dtypes, or shapes differ"
        )
    train_values = train_receipt.get("output_value_sha256")
    eval_values = eval_receipt.get("output_value_sha256")
    if not isinstance(train_values, str) or not isinstance(eval_values, str):
        return None
    if train_values == eval_values:
        return DivergenceResult("none", "train and eval output value digests match")
    return DivergenceResult(
        "statistical", "train and eval structures match but output value digests differ"
    )


def _leaf_equal(left: object, right: object) -> bool:
    """Compare two structurally equivalent output leaves.

    Parameters
    ----------
    left, right:
        Output leaf values.

    Returns
    -------
    bool
        True only when values are exactly equal, treating paired NaNs as equal.
    """

    left_array = _to_numpy(left)
    right_array = _to_numpy(right)
    if left_array is not None and right_array is not None:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    try:
        equality = left == right
        if isinstance(equality, (bool, np.bool_)):
            return bool(equality)
    except (TypeError, ValueError, RuntimeError):
        pass
    left_digest = output_value_sha256(left)
    right_digest = output_value_sha256(right)
    return left_digest is not None and left_digest == right_digest


def classify_train_eval_divergence(
    train_output: object,
    eval_output: object,
    *,
    train_graph_signature: Optional[object] = None,
    eval_graph_signature: Optional[object] = None,
) -> DivergenceResult:
    """Classify train/eval behavior from captured outputs and optional graphs.

    Parameters
    ----------
    train_output, eval_output:
        Captured per-mode outputs.
    train_graph_signature, eval_graph_signature:
        Optional operation-graph signatures from a framework-native observer.

    Returns
    -------
    DivergenceResult
        ``structural`` for tree/shape/graph changes, ``statistical`` for value-only
        changes, and ``none`` for equal outputs.
    """

    train_signature = output_signature(train_output)
    eval_signature = output_signature(eval_output)
    if train_signature != eval_signature:
        return DivergenceResult(
            "structural", "train and eval output trees, types, dtypes, or shapes differ"
        )
    if (
        train_graph_signature is not None
        and eval_graph_signature is not None
        and train_graph_signature != eval_graph_signature
    ):
        return DivergenceResult("structural", "train and eval operation graphs differ")
    train_leaves = _flatten(train_output)
    eval_leaves = _flatten(eval_output)
    if all(_leaf_equal(left, right) for left, right in zip(train_leaves, eval_leaves)):
        return DivergenceResult("none", "train and eval outputs are equal")
    return DivergenceResult(
        "statistical", "train and eval output signatures match but values differ"
    )


def _vetted_zero_rate_dropout_types() -> tuple[type, ...]:
    """Return the exact torch dropout classes whose semantics are vetted.

    Each listed class is, by its own shipped contract, the identity function in
    BOTH train and eval mode when ``p == 0.0``: no element is dropped and the
    ``1/(1-p)`` train-mode rescale is ``1``. The set is deliberately EXACT
    implementations, never a name or ancestry test -- inertness is a claim
    about forward SEMANTICS, and only these shipped forwards are known.

    Returns
    -------
    tuple[type, ...]
        Exact vetted classes, or empty when torch is unavailable (everything
        then fails closed to mode-sensitive).
    """

    try:
        from torch import nn
    except ImportError:
        return ()
    return (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )


def _dropout_family_is_provably_inert(module: object) -> bool:
    """Return whether a module IS a vetted dropout class at rate exactly zero.

    The rung-11 census (m10949/m10979) motivated the carve-out: timm ViTs
    instantiate ``torch.nn.Dropout(p=0.0)`` throughout, the value-blind name
    heuristic detected ``[train, eval]``, and the repair loop then demanded the
    author assert a divergence its own frozen ``train_eval_divergence: "none"``
    evidence contradicted.

    The wave-8 adversarial review then showed the round-1 ruling (name match
    plus a closed zero-rate attribute set) proved nothing about forward
    SEMANTICS: a custom ``SneakyDropout`` with ``p = 0.0`` whose forward
    branches on ``self.training`` was classified eval-only while its train and
    eval outputs genuinely differed, suppressing the train receipt tripwire.
    The module-semantics set is open, so inertness is now granted only to the
    EXACT vetted ``torch.nn`` dropout implementations -- ``type(module) is``
    identity, never ``isinstance`` (a subclass can override ``forward``) and
    never a class-name substring -- with the declared rate still required to
    be exactly zero. Everything else fails closed to mode-sensitive.

    Parameters
    ----------
    module:
        Child module whose type name matched the dropout family.

    Returns
    -------
    bool
        True only for an exact vetted torch dropout instance whose ``p`` is a
        real number equal to zero.
    """

    for vetted in _vetted_zero_rate_dropout_types():
        if type(module) is vetted:
            rate = getattr(module, "p", None)
            return (
                isinstance(rate, (int, float)) and not isinstance(rate, bool) and float(rate) == 0.0
            )
    return False


def _has_mode_sensitive_child(model: object) -> bool:
    """Detect known normalization/dropout mode-sensitive children.

    Dropout CANDIDACY and dropout INERTNESS are deliberately separate tests
    (wave-8 round-2 review, finding 1): a candidate is any module that
    subclasses a vetted torch dropout class -- ancestry via ``isinstance``,
    because a subclass may carry ANY name, and gating the check on a
    ``dropout`` name substring let a renamed ``torch.nn.Dropout`` subclass
    with a training-branching forward bypass detection entirely -- or any
    module whose type name contains ``dropout`` (custom non-torch classes).
    Inertness then stays exact-type-only: candidacy widens with ancestry,
    while the identity carve-out never does.

    Parameters
    ----------
    model:
        Native model object.

    Returns
    -------
    bool
        True when a child type names BatchNorm, or is a dropout candidate (a
        vetted-class descendant by ancestry, or dropout-named) that is not an
        exact vetted torch dropout implementation at rate exactly zero.
    """

    modules_method = getattr(model, "modules", None)
    if not callable(modules_method):
        return False
    try:
        modules: Sequence[object] = tuple(modules_method())
    except (TypeError, RuntimeError):
        return False
    vetted = _vetted_zero_rate_dropout_types()
    for module in modules:
        type_name = type(module).__name__.lower()
        if "batchnorm" in type_name:
            return True
        dropout_candidate = "dropout" in type_name or isinstance(module, vetted)
        if dropout_candidate and not _dropout_family_is_provably_inert(module):
            return True
    return False


def _forward_reads_training(model: object) -> bool:
    """Detect an explicit ``training`` branch in Python forward source.

    Parameters
    ----------
    model:
        Native model object.

    Returns
    -------
    bool
        True when inspectable forward source references training mode.
    """

    forward = getattr(model, "forward", None)
    if not callable(forward):
        return False
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        return False
    return ".training" in source or "training=" in source or "training =" in source


def detect_meaningful_modes(
    model: object, *, captured_outputs: Optional[Mapping[str, object]] = None
) -> tuple[RunMode, ...]:
    """Detect the minimal meaningful train/eval mode set.

    Models with BatchNorm, actively configured Dropout, an inspectable
    training branch, or already observed train/eval divergence require both
    modes. A model confidently lacking those features -- including one whose
    only dropout-family children are EXACT vetted ``torch.nn`` dropout
    implementations declaring rate ``0.0``, which are the identity in both
    modes by their shipped contract -- uses one eval-mode representative.
    Custom dropout-named classes and subclasses fail closed to both modes:
    their forward semantics are unproven.

    Parameters
    ----------
    model:
        Native model or transparent adapter.
    captured_outputs:
        Optional train/eval outputs from a probe.

    Returns
    -------
    tuple[RunMode, ...]
        Meaningful modes in train/eval order.
    """

    # This heuristic cannot see every native/custom branch; author declarations and checker vetting cover it.
    if captured_outputs is not None and {"train", "eval"}.issubset(captured_outputs):
        divergence = classify_train_eval_divergence(
            captured_outputs["train"], captured_outputs["eval"]
        )
        if divergence.classification != "none":
            return (RunMode.TRAIN, RunMode.EVAL)
        # DX-9/RE-9 (mode contract, adopted via section 6 C.12): captured
        # both-mode outputs are AUTHORITATIVE over the structural heuristic.
        # An observed no-divergence forward pair proves the modes are
        # equivalent on this contract -- a BatchNorm/Dropout child whose
        # effect never reaches the output cannot demand a second mode. The
        # heuristic remains the authority whenever no evidence was captured.
        return (RunMode.EVAL,)
    if _has_mode_sensitive_child(model) or _forward_reads_training(model):
        return (RunMode.TRAIN, RunMode.EVAL)
    return (RunMode.EVAL,)
