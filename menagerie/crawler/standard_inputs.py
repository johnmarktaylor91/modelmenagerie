"""Canonical modality-aware standard inputs with honest random fallback."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from menagerie.crawler.backend import (
    BackendProtocolUnimplementedError,
    Framework,
    FrameworkVocabularyError,
    normalize_framework,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.payload_codec import DtypeVocabularyError, canonical_dtype

#: NumPy is an input MATERIALIZATION target, not a model execution framework, so it is not a
#: member of the closed :class:`~menagerie.crawler.backend.Framework` vocabulary. Keeping it
#: out is what lets ``run_framework`` be closed to frameworks a model can actually execute
#: in, while an input contract may still ask for a raw array.
NUMPY_TARGETS: frozenset[str] = frozenset({"numpy", "np"})


ASSET_ROOT = Path(__file__).with_name("assets") / "standard"

#: Closed constraint grammar recovering a declared ``integer-range`` bound.
#: Three exact shapes cover every parseable spelling observed in the pilot
#: corpus; anything else refuses (``None``) and keeps the historical fallback.
#: The bracket rule requires an ``in``/``within``/``from`` anchor so shape
#: prose ("shape is [2, 4]") can never be misread as a value range.
_INTEGER_RANGE_BRACKET = re.compile(
    r"(?:\bin|\bwithin|\bfrom)\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*([\]\)])"
)
#: Comparison chains such as ``0 <= t < 1000``; a symbolic upper bound
#: (``0 <= value < vocab_size (131072)``) deliberately fails the digit match.
_INTEGER_RANGE_CHAIN = re.compile(r"(-?\d+)\s*<=\s*[A-Za-z_][\w .'\"-]*?\s*(<=?)\s*(-?\d+)")
#: A single legal constant, e.g. ``integer value 0 (num_relations=1)``.
_INTEGER_RANGE_CONSTANT = re.compile(r"^integer values?\s+(-?\d+)\b")


def parse_integer_range_constraints(
    constraints: Sequence[object],
) -> Optional[tuple[int, int]]:
    """Recover one exact ``[low, high)`` bound from declared leaf constraints.

    Parameters
    ----------
    constraints:
        The input-contract leaf's authored constraint strings.

    Returns
    -------
    tuple[int, int] | None
        Half-open ``(low, high)`` when EXACTLY one coherent, non-empty bound
        parses out of the closed grammar; ``None`` for anything unparseable,
        conflicting, or empty. ``None`` always means "keep the historical
        fallback", never an error: the grammar is a recovery aid for bounds
        the schema only carries as prose, not a validator.
    """

    candidates: set[tuple[int, int]] = set()
    for raw in constraints:
        if not isinstance(raw, str):
            continue
        for match in _INTEGER_RANGE_BRACKET.finditer(raw):
            low, high = int(match.group(1)), int(match.group(2))
            candidates.add((low, high + 1 if match.group(3) == "]" else high))
        for match in _INTEGER_RANGE_CHAIN.finditer(raw):
            low, high = int(match.group(1)), int(match.group(3))
            candidates.add((low, high + 1 if match.group(2) == "<=" else high))
        constant = _INTEGER_RANGE_CONSTANT.match(raw)
        if constant is not None:
            value = int(constant.group(1))
            candidates.add((value, value + 1))
    if len(candidates) != 1:
        return None
    low, high = next(iter(candidates))
    if low >= high:
        return None
    return (low, high)


class StandardInputError(ValueError):
    """Raised when an input specification cannot produce any valid tensor."""


@dataclass(frozen=True)
class InputSpec:
    """Concrete single-tensor input specification.

    Parameters
    ----------
    shape:
        Positive concrete tensor dimensions.
    dtype:
        Framework-neutral dtype token.
    """

    shape: tuple[int, ...]
    dtype: str = "float32"

    @classmethod
    def from_value(cls, value: Union["InputSpec", Mapping[str, Any]]) -> "InputSpec":
        """Normalize a typed input specification.

        Parameters
        ----------
        value:
            Existing spec or mapping with shape and dtype.

        Returns
        -------
        InputSpec
            Concrete normalized specification.
        """

        if isinstance(value, InputSpec):
            return value
        shape_value = value.get("shape")
        # An EMPTY shape is a scalar tensor -- a legal authored contract (the
        # schema's shape array has no minimum length; m10251 declares a
        # scalar diffusion timestep). Only a non-sequence is malformed.
        if not isinstance(shape_value, (list, tuple)):
            raise StandardInputError("input spec shape must be an integer sequence")
        try:
            shape = tuple(int(dimension) for dimension in shape_value)
        except (TypeError, ValueError) as exc:
            raise StandardInputError("input spec shape must contain concrete integers") from exc
        if any(dimension <= 0 for dimension in shape):
            raise StandardInputError("input spec dimensions must be positive")
        dtype = value.get("dtype", "float32")
        if not isinstance(dtype, str) or not dtype:
            raise StandardInputError("input spec dtype must be a non-empty string")
        return cls(shape=shape, dtype=dtype.lower().replace("torch.", ""))


@dataclass(frozen=True)
class MaterializedInput:
    """Tensor plus mechanically observed input provenance.

    Parameters
    ----------
    value:
        Framework-native tensor.
    input_kind:
        Canonical asset kind or ``random-fallback``.
    input_asset:
        Content-addressed asset identifier, or ``None`` for random fallback.
    input_note:
        Brief mechanical shaping/fallback note.
    spec:
        Concrete fulfilled input specification.
    """

    value: object
    input_kind: str
    input_asset: str | None
    input_note: str
    spec: InputSpec

    @property
    def args(self) -> tuple[object, ...]:
        """Return this single tensor as positional forward arguments.

        Returns
        -------
        tuple[object, ...]
            One positional input.
        """

        return (self.value,)

    @property
    def kwargs(self) -> dict[str, object]:
        """Return empty keyword arguments for the standard call.

        Returns
        -------
        dict[str, object]
            Empty keyword mapping.
        """

        return {}


def _asset_id(path: Path) -> str:
    """Return a content-addressed standard asset identifier.

    Parameters
    ----------
    path:
        Bundled asset path.

    Returns
    -------
    str
        Asset name and SHA-256 digest.
    """

    return f"standard:{path.name}:{hash_bytes(path.read_bytes())}"


def _load_ppm(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Load the bundled ASCII PPM image without an imaging dependency.

    Parameters
    ----------
    path:
        P3 PPM asset.

    Returns
    -------
    numpy.ndarray
        Height-by-width-by-three uint8 image.
    """

    tokens: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        content = line.partition("#")[0]
        tokens.extend(content.split())
    if len(tokens) < 4 or tokens[0] != "P3":
        raise StandardInputError(f"invalid standard PPM asset: {path}")
    width, height, maximum = (int(tokens[index]) for index in (1, 2, 3))
    if maximum != 255:
        raise StandardInputError("standard PPM must use an 8-bit maximum")
    pixels = np.asarray([int(token) for token in tokens[4:]], dtype=np.uint8)
    if pixels.size != width * height * 3:
        raise StandardInputError("standard PPM pixel count is inconsistent")
    return pixels.reshape(height, width, 3)


def _resize_image(image: np.ndarray[Any, Any], height: int, width: int) -> np.ndarray[Any, Any]:
    """Resize an image deterministically with nearest-neighbor sampling.

    Parameters
    ----------
    image:
        Source HWC image.
    height, width:
        Requested dimensions.

    Returns
    -------
    numpy.ndarray
        Resized HWC image.
    """

    rows = np.linspace(0, image.shape[0] - 1, height).round().astype(np.int64)
    columns = np.linspace(0, image.shape[1] - 1, width).round().astype(np.int64)
    return image[rows][:, columns]


def _image_array(spec: InputSpec, *, video: bool = False) -> np.ndarray[Any, Any]:
    """Shape the canonical image for an NCHW/NHWC vision or video spec.

    Parameters
    ----------
    spec:
        Requested tensor specification.
    video:
        Whether a five-dimensional video layout is allowed.

    Returns
    -------
    numpy.ndarray
        Standard image/video array.
    """

    shape = spec.shape
    if video and len(shape) == 5:
        if shape[1] in {1, 3, 4}:
            batch, channels, frames, height, width = shape
            frame_spec = InputSpec((batch, channels, height, width), spec.dtype)
            frame = _image_array(frame_spec)
            return np.repeat(frame[:, :, None, :, :], frames, axis=2)
        if shape[-1] in {1, 3, 4}:
            batch, frames, height, width, channels = shape
            frame_spec = InputSpec((batch, height, width, channels), spec.dtype)
            frame = _image_array(frame_spec)
            return np.repeat(frame[:, None, :, :, :], frames, axis=1)
        raise StandardInputError("video shape has no recognizable channel axis")
    if len(shape) == 4 and shape[1] in {1, 3, 4}:
        batch, channels, height, width = shape
        channel_first = True
    elif len(shape) == 4 and shape[-1] in {1, 3, 4}:
        batch, height, width, channels = shape
        channel_first = False
    elif len(shape) == 3 and shape[0] in {1, 3, 4}:
        channels, height, width = shape
        batch = 0
        channel_first = True
    elif len(shape) == 3 and shape[-1] in {1, 3, 4}:
        height, width, channels = shape
        batch = 0
        channel_first = False
    else:
        raise StandardInputError("vision shape has no recognizable image layout")
    image = _resize_image(_load_ppm(ASSET_ROOT / "image.ppm"), height, width)
    if channels == 1:
        image = image.astype(np.float32).mean(axis=2, keepdims=True)
    elif channels == 4:
        alpha = np.full((height, width, 1), 255, dtype=image.dtype)
        image = np.concatenate((image, alpha), axis=2)
    if channel_first:
        image = np.transpose(image, (2, 0, 1))
    if batch:
        image = np.repeat(image[None, ...], batch, axis=0)
    return image


def _text_array(spec: InputSpec) -> np.ndarray[Any, Any]:
    """Tokenize the fixed standard text into the requested shape.

    Parameters
    ----------
    spec:
        Requested tensor specification.

    Returns
    -------
    numpy.ndarray
        Deterministic UTF-8-byte token array.
    """

    text = (ASSET_ROOT / "text.txt").read_text(encoding="utf-8").strip()
    tokens = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64) + 1
    return np.resize(tokens, spec.shape)


def _csv_array(path: Path, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    """Resize numeric CSV asset values to a concrete tensor shape.

    Parameters
    ----------
    path:
        Bundled numeric CSV.
    shape:
        Requested tensor shape.

    Returns
    -------
    numpy.ndarray
        Resized floating array.
    """

    with path.open(newline="", encoding="utf-8") as handle:
        values = [float(value) for row in csv.reader(handle) for value in row]
    return np.resize(np.asarray(values, dtype=np.float32), shape)


def _random_array(spec: InputSpec, seed: int) -> np.ndarray[Any, Any]:
    """Generate a deterministic random fallback array.

    Parameters
    ----------
    spec:
        Requested tensor specification.
    seed:
        RNG seed.

    Returns
    -------
    numpy.ndarray
        Random array before framework conversion.
    """

    rng = np.random.default_rng(seed)
    dtype = spec.dtype
    if dtype in {"bool"}:
        return rng.integers(0, 2, size=spec.shape, dtype=np.int8).astype(np.bool_)
    if dtype.startswith("int") or dtype in {"long", "uint8"}:
        return rng.integers(0, 16, size=spec.shape, dtype=np.int64)
    return rng.standard_normal(spec.shape).astype(np.float32)


def _constant_array(spec: InputSpec, value: int) -> np.ndarray[Any, Any]:
    """Generate a constant array for a declared zero or one distribution.

    Parameters
    ----------
    spec:
        Requested tensor specification.
    value:
        Declared constant, either zero or one.

    Returns
    -------
    numpy.ndarray
        Constant array before framework conversion.
    """

    if spec.dtype == "bool":
        return np.full(spec.shape, bool(value), dtype=np.bool_)
    if spec.dtype.startswith("int") or spec.dtype in {"long", "uint8"}:
        return np.full(spec.shape, value, dtype=np.int64)
    return np.full(spec.shape, value, dtype=np.float32)


def _is_integer_dtype(spec: InputSpec) -> bool:
    """Return whether the declared dtype is one of the integer family tokens."""

    return spec.dtype.startswith("int") or spec.dtype in {"long", "uint8"}


def _is_float_dtype(spec: InputSpec) -> bool:
    """Return whether the declared dtype is one of the float family tokens."""

    return spec.dtype.startswith("float") or spec.dtype in {"bfloat16", "half", "double"}


def _convert_array(
    array: np.ndarray[Any, Any],
    spec: InputSpec,
    framework: str,
    device: str,
    *,
    normalize: bool = True,
) -> object:
    """Convert a NumPy array into the requested native framework tensor.

    Parameters
    ----------
    array:
        Source array.
    spec:
        Requested dtype.
    framework:
        Native execution framework.
    device:
        Native device string.
    normalize:
        Whether to apply the byte-asset [0, 1] rescale for float targets. The
        modality-asset path keeps it; DECLARED distributions must not -- a
        standard-normal sample with max 2.5 would silently divide by 255.

    Returns
    -------
    object
        Framework-native tensor.
    """

    normalized = framework.strip().lower()
    if _is_float_dtype(spec):
        if normalize:
            array = array.astype(np.float32) / (255.0 if array.max(initial=0) > 1.0 else 1.0)
        else:
            array = array.astype(np.float32)
    # NumPy is a passthrough TARGET, not a model execution framework, so it is handled
    # before the closed framework vocabulary rather than being smuggled into it.
    if normalized in NUMPY_TARGETS:
        return array
    try:
        resolved = normalize_framework(normalized)
    except FrameworkVocabularyError as exc:
        raise StandardInputError(f"unsupported input framework: {framework!r}") from exc
    # ONE dtype vocabulary, shared with the signature recorder and the contract comparison.
    # The torch arm used to carry its own alias map, and every other arm did
    # `getattr(tf, spec.dtype)` / `getattr(jnp, spec.dtype)` -- so an authored `"half"` or
    # `"long"` resolved for torch and raised an AttributeError for TensorFlow and JAX.
    try:
        canonical = canonical_dtype(spec.dtype)
    except DtypeVocabularyError as exc:
        raise StandardInputError(
            f"unsupported input dtype for {resolved.value}: {spec.dtype!r}"
        ) from exc
    if resolved is Framework.TORCH:
        import torch

        torch_dtype = getattr(torch, canonical.value, None)
        if torch_dtype is None:
            raise StandardInputError(f"unsupported PyTorch dtype: {spec.dtype!r}")
        return torch.as_tensor(array, dtype=torch_dtype, device=device)
    if resolved is Framework.TENSORFLOW:
        import tensorflow as tf

        return tf.convert_to_tensor(array, dtype=getattr(tf, canonical.value))
    if resolved is Framework.JAX:
        import jax.numpy as jnp

        return jnp.asarray(array, dtype=getattr(jnp, canonical.value))
    if resolved is Framework.PADDLE:
        import paddle

        return paddle.to_tensor(array, dtype=canonical.value, place=device)
    if resolved is Framework.MLX:
        import mlx.core as mlx_core

        return mlx_core.array(array, dtype=getattr(mlx_core, canonical.value))
    raise BackendProtocolUnimplementedError(
        f"{resolved.value} has no input-conversion arm in this worker"
    )


def _declared_bounds(input_spec: Union[InputSpec, Mapping[str, Any]]) -> Optional[tuple[int, int]]:
    """Return the single coherent ``[low, high)`` bound parsed from constraints.

    Index-valued leaves (graph edge indices, entity/user/item ids, token ids)
    declare bounds narrower than the historical ``[0, 16)`` integer fallback --
    which crashed every such model at its first real forward (m1115 edge_index
    in [0, 2], m2336 entity ids in [0, 1], m8650 user/item ids below 10).

    Parameters
    ----------
    input_spec:
        Original spec value, possibly carrying constraints.

    Returns
    -------
    tuple[int, int] | None
        Half-open sampling bound, or ``None`` when no single coherent bound
        was declared.
    """

    if isinstance(input_spec, InputSpec) or not isinstance(input_spec, Mapping):
        return None
    constraints = input_spec.get("constraints")
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        return None
    return parse_integer_range_constraints(constraints)


def _integer_sample(
    input_spec: Union[InputSpec, Mapping[str, Any]],
    spec: InputSpec,
    distribution: str,
    seed: int,
) -> tuple[np.ndarray[Any, Any], tuple[int, int]]:
    """Sample one bounded integer array for a declared index distribution.

    Parameters
    ----------
    input_spec:
        Original spec value carrying the authored constraints.
    spec:
        Normalized concrete specification (must be integer-family).
    distribution:
        The declared distribution, for the refusal text.
    seed:
        Deterministic sampling seed.

    Returns
    -------
    tuple
        Sampled array and its half-open bound.

    Raises
    ------
    StandardInputError
        If the dtype is not integer-family or no single coherent bound was
        declared -- sampling an UNDECLARED range is the silent-substitution
        class this dispatch exists to kill.
    """

    if not _is_integer_dtype(spec):
        raise StandardInputError(
            f"declared {distribution!r} distribution requires an integer dtype, got {spec.dtype!r}"
        )
    bounds = _declared_bounds(input_spec)
    if bounds is None:
        raise StandardInputError(
            f"declared {distribution!r} distribution requires exactly one coherent "
            "integer bound in the leaf's constraints (e.g. '[0, 32)'); refusing to "
            "invent a range"
        )
    low, high = bounds
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=spec.shape, dtype=np.int64), bounds


def _materialize_declared_distribution(
    distribution: str,
    input_spec: Union[InputSpec, Mapping[str, Any]],
    spec: InputSpec,
    *,
    framework: str,
    device: str,
    seed: int,
) -> MaterializedInput:
    """Honor one DECLARED distribution or refuse typed -- never substitute.

    Total over the model-common schema's closed ``distribution`` enum
    (``normal``, ``uniform``, ``integer-range``, ``zeros``, ``ones``,
    ``categorical``, ``constructor``) plus a typed refusal arm for any member
    this worker does not implement, so a future enum extension without a
    matching arm fails loudly instead of substituting modality bytes
    (DX-5/RE-5; the totality pin reads the enum from the schema).

    Parameters
    ----------
    distribution:
        Declared non-null distribution.
    input_spec:
        Original spec value carrying constraints.
    spec:
        Normalized concrete specification.
    framework, device:
        Native tensor target.
    seed:
        Deterministic sampling seed.

    Returns
    -------
    MaterializedInput
        Honored declared-distribution tensor.

    Raises
    ------
    StandardInputError
        Typed refusal for incoherent or unimplemented declarations.
    """

    if distribution in {"zeros", "ones"}:
        value = 0 if distribution == "zeros" else 1
        array = _constant_array(spec, value)
        return MaterializedInput(
            value=_convert_array(array, spec, framework, device, normalize=False),
            input_kind=f"standard-{distribution}",
            input_asset=None,
            input_note=f"declared {distribution} distribution fulfilled from the input contract",
            spec=spec,
        )
    if distribution in {"integer-range", "categorical"}:
        # ``categorical`` is the natural authored spelling for token/type/id
        # leaves; it samples exactly like a declared integer range. Before
        # this arm it silently fell through to modality assets -- the same bug
        # as the OMP trio with a different enum member (DX-5/RE-5).
        array, (low, high) = _integer_sample(input_spec, spec, distribution, seed)
        return MaterializedInput(
            value=_convert_array(array, spec, framework, device, normalize=False),
            input_kind=f"standard-{distribution}",
            input_asset=None,
            input_note=(f"declared {distribution} [{low}, {high}) sampled from the input contract"),
            spec=spec,
        )
    if distribution == "normal":
        if not _is_float_dtype(spec):
            raise StandardInputError(
                f"declared 'normal' distribution requires a float dtype, got {spec.dtype!r}"
            )
        rng = np.random.default_rng(seed)
        array = rng.standard_normal(spec.shape)
        return MaterializedInput(
            value=_convert_array(array, spec, framework, device, normalize=False),
            input_kind="standard-normal",
            input_asset=None,
            input_note="declared standard-normal distribution sampled from the input contract",
            spec=spec,
        )
    if distribution == "uniform":
        if _is_float_dtype(spec):
            rng = np.random.default_rng(seed)
            array = rng.random(spec.shape)
            return MaterializedInput(
                value=_convert_array(array, spec, framework, device, normalize=False),
                input_kind="standard-uniform",
                input_asset=None,
                input_note="declared uniform [0, 1) distribution sampled from the input contract",
                spec=spec,
            )
        array, (low, high) = _integer_sample(input_spec, spec, distribution, seed)
        return MaterializedInput(
            value=_convert_array(array, spec, framework, device, normalize=False),
            input_kind="standard-uniform",
            input_asset=None,
            input_note=(f"declared uniform integers [{low}, {high}) from the input contract"),
            spec=spec,
        )
    if distribution == "constructor":
        raise StandardInputError(
            "distribution 'constructor' is the constructed-leaf grammar; a tensor "
            "leaf carrying it is a contract breach upstream of materialization"
        )
    raise StandardInputError(
        f"declared distribution {distribution!r} has no materialization arm in this "
        "worker; refusing to substitute modality assets for a declared contract"
    )


def _normalized_modalities(modality: Union[str, Sequence[str], None]) -> tuple[str, ...]:
    """Normalize source-gated modality values for matching.

    Parameters
    ----------
    modality:
        One modality, several modalities, or null.

    Returns
    -------
    tuple[str, ...]
        Lowercase modality tokens.
    """

    if modality is None:
        return ()
    values = (modality,) if isinstance(modality, str) else tuple(modality)
    return tuple(str(value).strip().lower() for value in values)


def materialize_standard_input(
    modality: Union[str, Sequence[str], None],
    input_spec: Union[InputSpec, Mapping[str, Any]],
    *,
    framework: str = "pytorch",
    device: str = "cpu",
    seed: int = 0,
) -> MaterializedInput:
    """Build a canonical modality input or an honest random fallback.

    Parameters
    ----------
    modality:
        Source-gated ``external_metadata.modality`` value.
    input_spec:
        Concrete shape and dtype contract.
    framework, device:
        Native tensor target.
    seed:
        Deterministic fallback seed.

    Returns
    -------
    MaterializedInput
        Tensor plus ``input_kind``, asset hash, and shaping/fallback note.
    """

    spec = InputSpec.from_value(input_spec)
    distribution = input_spec.get("distribution") if isinstance(input_spec, Mapping) else None
    if distribution is not None:
        # A DECLARED distribution is a contract: it is honored by its own arm
        # or refused typed. It never falls through to the modality assets or
        # the random fallback -- that silent substitution is exactly the OMP
        # trio's failure shape (text bytes standing in for declared zeros),
        # and `categorical` fell through the same hole with token/type ids
        # (DX-5/RE-5). The dispatch below is TOTAL over the model-common
        # schema's closed enum; the totality pin is generated from the schema
        # at test time so a new member cannot be added without an arm.
        return _materialize_declared_distribution(
            str(distribution),
            input_spec,
            spec,
            framework=framework,
            device=device,
            seed=seed,
        )
    modalities = _normalized_modalities(modality)
    selected: tuple[str, Path] | None = None
    try:
        if any(value in {"vision", "image", "computer-vision"} for value in modalities):
            asset = ASSET_ROOT / "image.ppm"
            array = _image_array(spec)
            selected = ("standard-image", asset)
        elif any(value in {"language", "text", "nlp"} for value in modalities):
            asset = ASSET_ROOT / "text.txt"
            array = _text_array(spec)
            selected = ("standard-text", asset)
        elif any(value in {"audio", "speech"} for value in modalities):
            asset = ASSET_ROOT / "audio.csv"
            array = _csv_array(asset, spec.shape)
            selected = ("standard-audio", asset)
        elif "video" in modalities:
            asset = ASSET_ROOT / "image.ppm"
            array = _image_array(spec, video=True)
            selected = ("standard-video", asset)
        elif any(value in {"tabular", "recsys"} for value in modalities):
            asset = ASSET_ROOT / "tabular.csv"
            array = _csv_array(asset, spec.shape)
            selected = ("standard-tabular", asset)
        else:
            raise StandardInputError("no canonical asset for the declared modality")
        if array.shape != spec.shape:
            raise StandardInputError(
                f"standard asset produced {array.shape}, expected {spec.shape}"
            )
        value = _convert_array(array, spec, framework, device)
        kind, asset = selected
        return MaterializedInput(
            value=value,
            input_kind=kind,
            input_asset=_asset_id(asset),
            input_note=f"canonical {asset.name} shaped to {list(spec.shape)}",
            spec=spec,
        )
    except StandardInputError as exc:
        fallback = _convert_array(_random_array(spec, seed), spec, framework, device)
        return MaterializedInput(
            value=fallback,
            input_kind="random-fallback",
            input_asset=None,
            input_note=f"random fallback: {exc}",
            spec=spec,
        )
