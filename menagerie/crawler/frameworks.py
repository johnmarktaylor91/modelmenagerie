"""Transparent native-framework forward adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from menagerie.crawler.backend import (
    Framework,
    framework_capability,
    normalize_framework,
    origin_framework_token,
)


@runtime_checkable
class ForwardProtocol(Protocol):
    """Runtime protocol required by the execution worker."""

    def forward(self, *args: object, **kwargs: object) -> object:
        """Delegate one native forward call.

        Parameters
        ----------
        *args, **kwargs:
            Native call arguments.

        Returns
        -------
        object
            Unmodified native output.
        """

        ...


@dataclass(frozen=True)
class NativeCallMetadata:
    """Provenance for one transparent native call.

    Parameters
    ----------
    original_framework, run_framework:
        Source and actual execution frameworks.
    native_object_type:
        Qualified native object type.
    native_call_method:
        Method invoked by ``forward``.
    transparent_forward_adapter:
        Always true for this adapter implementation.
    """

    original_framework: str
    run_framework: str
    native_object_type: str
    native_call_method: str
    transparent_forward_adapter: bool = True

    def __post_init__(self) -> None:
        """Refuse a non-canonical spelling before it reaches a record or an identity.

        These two strings are published on the model record AND hashed into the execution
        identity, so a spelling rotation is a work-eating re-drive axis (B1's second kill:
        an author-model rotation re-emitting ``"PyTorch"`` where the cached authoring said
        ``"pytorch"`` re-generations an unchanged model). Normalizing at construction means
        only one spelling per framework can ever be written.
        """

        object.__setattr__(self, "run_framework", normalize_framework(self.run_framework).value)
        # The ORIGIN is a token, not a closed enum, and for the same reason the schema keeps
        # it open: an architecture's original framework is a world-fact wider than anything
        # this pipeline executes. Prose is still refused.
        object.__setattr__(
            self,
            "original_framework",
            origin_framework_token(self.original_framework),
        )


class NativeForwardAdapter:
    """Expose ``forward`` as a direct delegation to one native call method.

    Parameters
    ----------
    native_object:
        Framework-native model or bound callable container.
    original_framework:
        Framework in which the architecture is implemented.
    run_framework:
        Framework executing the call. Defaults to the original framework.
    call_method:
        Exact native method delegated to by ``forward``.
    """

    def __init__(
        self,
        native_object: object,
        *,
        original_framework: str,
        run_framework: str | None = None,
        call_method: str = "forward",
    ) -> None:
        """Initialize a transparent native adapter.

        Parameters
        ----------
        native_object:
            Framework-native object.
        original_framework, run_framework, call_method:
            Native call provenance.
        """

        native_call = getattr(native_object, call_method, None)
        if not callable(native_call):
            raise TypeError(
                f"native object {type(native_object).__qualname__} has no callable {call_method!r}"
            )
        self.native_object = native_object
        self._native_call = native_call
        self.training = bool(getattr(native_object, "training", False))
        self.metadata = NativeCallMetadata(
            original_framework=original_framework,
            run_framework=run_framework or original_framework,
            native_object_type=(
                f"{type(native_object).__module__}.{type(native_object).__qualname__}"
            ),
            native_call_method=call_method,
        )

    def forward(self, *args: object, **kwargs: object) -> object:
        """Delegate directly to the recorded native method.

        Parameters
        ----------
        *args, **kwargs:
            Native model inputs.

        Returns
        -------
        object
            Exact native output without tensor transformation.
        """

        return self._native_call(*args, **kwargs)

    def train(self, mode: bool = True) -> "NativeForwardAdapter":
        """Delegate native training-mode selection when supported.

        Parameters
        ----------
        mode:
            Desired training flag.

        Returns
        -------
        NativeForwardAdapter
            This adapter.
        """

        native_train = getattr(self.native_object, "train", None)
        if callable(native_train):
            native_train(mode)
        self.training = mode
        return self

    def eval(self) -> "NativeForwardAdapter":
        """Delegate native evaluation-mode selection when supported.

        Returns
        -------
        NativeForwardAdapter
            This adapter.
        """

        native_eval = getattr(self.native_object, "eval", None)
        if callable(native_eval):
            native_eval()
        else:
            native_train = getattr(self.native_object, "train", None)
            if callable(native_train):
                native_train(False)
        self.training = False
        return self

    def parameters(self, *args: object, **kwargs: object) -> Any:
        """Delegate parameter iteration when the native object exposes it.

        Parameters
        ----------
        *args, **kwargs:
            Native parameter iterator arguments.

        Returns
        -------
        Any
            Native parameter iterator.
        """

        native_parameters = getattr(self.native_object, "parameters", None)
        if not callable(native_parameters):
            return iter(())
        return native_parameters(*args, **kwargs)

    def modules(self) -> Any:
        """Delegate module traversal when supported.

        Returns
        -------
        Any
            Native module iterator, or an iterator containing this adapter.
        """

        native_modules = getattr(self.native_object, "modules", None)
        return native_modules() if callable(native_modules) else iter((self,))


class PyTorchForwardAdapter(NativeForwardAdapter):
    """Transparent adapter for a PyTorch ``forward`` method."""

    def __init__(self, native_object: object) -> None:
        """Wrap a PyTorch object.

        Parameters
        ----------
        native_object:
            Native PyTorch module.
        """

        super().__init__(native_object, original_framework="pytorch", call_method="forward")


class TensorFlowForwardAdapter(NativeForwardAdapter):
    """Transparent adapter for a TensorFlow/Keras ``__call__`` method."""

    def __init__(self, native_object: object) -> None:
        """Wrap a TensorFlow/Keras object.

        Parameters
        ----------
        native_object:
            Native TensorFlow/Keras model.
        """

        super().__init__(native_object, original_framework="tensorflow", call_method="__call__")


class JaxForwardAdapter(NativeForwardAdapter):
    """Transparent adapter for a bound JAX/Flax ``apply`` method."""

    def __init__(self, native_object: object) -> None:
        """Wrap a JAX/Flax object whose parameters are already bound.

        Parameters
        ----------
        native_object:
            Native object exposing ``apply``.
        """

        super().__init__(native_object, original_framework="jax", call_method="apply")


class PaddleForwardAdapter(NativeForwardAdapter):
    """Transparent adapter for a Paddle ``forward`` method."""

    def __init__(self, native_object: object) -> None:
        """Wrap a Paddle layer.

        Parameters
        ----------
        native_object:
            Native Paddle layer.
        """

        super().__init__(native_object, original_framework="paddle", call_method="forward")


#: Frameworks with a hand-written adapter subclass. Every other member of the closed enum
#: is served by the base adapter using its registry-declared call method, so a framework
#: can never be absent from this dispatch -- only more specifically served by it.
_ADAPTER_TYPES: Mapping[Framework, Callable[[object], NativeForwardAdapter]] = {
    Framework.TORCH: PyTorchForwardAdapter,
    Framework.TENSORFLOW: TensorFlowForwardAdapter,
    Framework.JAX: JaxForwardAdapter,
    Framework.PADDLE: PaddleForwardAdapter,
}


def adapt_native_model(native_object: object, framework: str) -> NativeForwardAdapter:
    """Create the transparent adapter for one closed-vocabulary framework.

    The adapter's call method comes from the framework capability registry rather than a
    local dictionary. That local dictionary was the fourth of the six drifting framework
    vocabularies (AREA_BACKENDS_opus B1): it knew ``torch/pytorch/tf/tensorflow/jax/flax/
    paddle/paddlepaddle``, and it knew neither MLX nor tinygrad -- two of the five TorchLens
    preview backends, on a fleet that is entirely Apple Silicon. Reading the registry makes
    a missing arm impossible: a framework added to the enum without a capability row fails
    at import, and the dispatch-arm census asserts totality.

    Parameters
    ----------
    native_object:
        Framework-native model object.
    framework:
        Any accepted spelling in :data:`menagerie.crawler.backend.FRAMEWORK_ALIASES`.

    Returns
    -------
    NativeForwardAdapter
        Framework-specific transparent adapter.

    Raises
    ------
    FrameworkVocabularyError
        For a spelling outside the closed vocabulary. Typed, because the shipped
        ``ValueError("unsupported native framework")`` was indistinguishable from an
        ordinary programming error at every call site.
    """

    resolved = normalize_framework(framework)
    capability = framework_capability(resolved)
    adapter_type = _ADAPTER_TYPES.get(resolved)
    if adapter_type is not None:
        return adapter_type(native_object)
    return NativeForwardAdapter(
        native_object,
        original_framework=resolved.value,
        call_method=capability.adapter_call_method,
    )
