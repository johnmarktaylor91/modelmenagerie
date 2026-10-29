"""ONE closed backend/framework authority for dispatch, identity, and lane routing.

``run_framework`` was free text. It was typed as ``nonempty_string`` in both
``model-common`` and ``author-proposal-v3``, author-gated, and then consumed verbatim by
six independent closed maps with six different key sets and three different failure modes
(AREA_BACKENDS_opus B1's table): silent no-op in ``worker._seed_frameworks``, typed raise
in ``standard_inputs._convert_array``, and -- worst -- the WRONG DEFAULT in
``constants.framework_from_markers``, which asserted "pytorch" for everything it did not
recognise. It was also hashed verbatim into the execution identity, making a spelling
rotation (``"pytorch"`` -> ``"PyTorch"``) a work-eating re-drive axis.

The catalog names this class C-22 and the process rule P-01 names why the instance sweep
failed: ``device_policy`` was found and fixed, and the identical defect one field over went
untouched. This module is the class fix:

* :class:`Framework` -- the closed vocabulary of frameworks a model may EXECUTE in, whose
  canonical spellings are the ones the pipeline and ``envs/registry.yml`` already use, so
  exactly one spelling per framework can reach an identity preimage.
* :class:`OriginFramework` -- the wider closed vocabulary a model may be IMPLEMENTED in.
  A framework with no TorchLens backend can be an origin and never a run framework, which
  is what makes the undisclosed-torch-reimplementation class unspellable.
* :data:`FRAMEWORK_CAPABILITIES` -- one record per framework naming every per-framework
  behaviour the pipeline needs, with EVERY field required. A framework cannot be
  half-added, which is the only way to stop this class re-opening for backend #7.
* :class:`FrameworkFact` -- framework as a fact with a BASIS. The roster-name guess is a
  prior, never an authority.
* :data:`PRODUCER_DISPATCH_INVENTORY` -- the closed C-22 census inventory: every
  producer-written string that reaches a dispatch or an identity, bound to the validator
  that closes it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

try:  # python >= 3.11
    from enum import StrEnum
except ImportError:  # python 3.10: the oldest routed-env interpreter (legacy-torch
    # lock resolves py3.10). The worker closure must import there; the constants
    # shim is value-identical (str content == member value), differing only in
    # str()/format() rendering, which nothing identity-bearing uses (DX-3/RE-3).
    from menagerie.crawler.constants import StrEnum  # type: ignore[assignment]
from importlib import import_module
from typing import Callable, Mapping, Optional

from menagerie.crawler.constants import (
    BACKEND_PARITY_DEFERRAL_STATUS_CODE,
    EnvironmentPhase,
    normalize_recipe_revision,
    normalize_recipe_type,
)

# `payload_codec` is deliberately NOT imported here. It needs NumPy, and `routing` imports
# this module, and the read-only rung-staging tools import `routing` -- a chain the
# import-light tripwire in test_rung_boundary_protocol enforces. The dtype validator is
# reached through a lazy import in `_canonical_dtype` below.


class FrameworkVocabularyError(ValueError):
    """Raised when a framework spelling is outside the closed vocabulary.

    Deliberately a refusal at the admission boundary rather than a normalization with a
    fallback. ``"torch 2.13"``, ``"pytorch (cpu)"`` and ``"PyTorch/2.x"`` are the live
    drift samples B1 reproduced; each of them previously seeded nothing, routed nowhere,
    and still entered the execution identity.
    """


class NativeLaneUndisclosedError(ValueError):
    """Raised when ``original_framework != run_framework`` with no typed disposition."""


class BackendProtocolUnimplementedError(ValueError):
    """Raised when a closed-vocabulary framework's named protocol has no implementation.

    Distinct from :class:`FrameworkVocabularyError`, which means "this string is not a
    framework". This one means "this IS a framework, its capability row names the protocol,
    and the code for it is not here" -- a machinery gap that must refuse loudly instead of
    falling off the end of an ``if``/``elif`` chain. Every silent no-op in B1's table was
    that missing ``else``.
    """


class Framework(StrEnum):
    """Closed vocabulary of frameworks a model may be EXECUTED in.

    Values are the spellings already written throughout the pipeline and
    ``envs/registry.yml`` (``"pytorch"``, not ``"torch"``), so adopting the enum renames
    nothing in the ledger and no record migration is owed. Membership is exactly the
    frameworks TorchLens can capture: the stable torch backend plus its five technical
    previews.
    """

    TORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    PADDLE = "paddle"
    MLX = "mlx"
    TINYGRAD = "tinygrad"


class OriginFramework(StrEnum):
    """Closed vocabulary of frameworks a model may be IMPLEMENTED in.

    Wider than :class:`Framework` because an architecture's original implementation can
    live in a framework nothing here can execute -- the roster carries mindspore (29),
    caffe (30), darknet (45), onnx (11), chainer (4), theano (1) and mxnet rows. Those are
    legitimate ORIGINS and never legitimate run frameworks, and separating the two
    vocabularies is what lets :func:`require_native_lane_disposition` refuse an
    undisclosed torch reimplementation.
    """

    TORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    PADDLE = "paddle"
    MLX = "mlx"
    TINYGRAD = "tinygrad"
    MXNET = "mxnet"
    MINDSPORE = "mindspore"
    CAFFE = "caffe"
    DARKNET = "darknet"
    ONNX = "onnx"
    CHAINER = "chainer"
    THEANO = "theano"


class LaneStatus(StrEnum):
    """Whether this campaign can execute a framework at all."""

    EXECUTABLE = "executable"
    NO_ENVIRONMENT_LANE = "no-environment-lane"


class FrameworkBasis(StrEnum):
    """Closed evidence bases for a framework claim, weakest first.

    ``MARKER`` is the roster row's zoo and name -- the weakest evidence available, and the
    only basis the pipeline had. It carries a measured 13% false-positive rate on the live
    roster (TRACKB_opus F6: 8 of 63 native-tail rows are provably PyTorch, and their own
    zoo says so), and its false-negative rate is unmeasurable from the roster at all. It
    is a PRIOR for staging, never an authority for execution.
    """

    MARKER = "marker"
    SOURCE_PROBE = "source-probe"
    AUTHOR_DECLARED_CHECKER_CONFIRMED = "author-declared-checker-confirmed"


#: The intent name for torch work, which is derived from declared PACKAGES rather than
#: being one fixed lane (``core``/``graph``/``audio``/``oddballs``/...). The sentinel keeps
#: the capability record total without pretending torch has a single intent.
TORCH_PACKAGE_DERIVED_INTENT = "package-derived"

#: Closed vocabulary for the disclosure owed when a model runs in a framework other than
#: the one it was implemented in (AREA_BACKENDS_opus B6, fix 2).
#:
#: 11 of 67 settled pilot heads carry ``original_framework != run_framework``. m10470 is
#: HF ``transformers`` T5 -- a TF-origin architecture executed through its exact maintained
#: torch library, which is legitimate and honest. Nothing in the record distinguished it
#: from a torch reimplementation of a native model, which the settled native-first decision
#: forbids and which is the 2026-07-01 1,029-row deletion class. This vocabulary is the
#: cheapest possible enforcement: the honest case NAMES itself, and the forbidden case has
#: no spelling.
NATIVE_LANE_DISPOSITIONS: frozenset[str] = frozenset(
    {
        "exact-maintained-library-in-run-framework",
        "native-lane-unavailable",
        BACKEND_PARITY_DEFERRAL_STATUS_CODE,
    }
)


@dataclass(frozen=True)
class FrameworkCapability:
    """Every per-framework behaviour the pipeline dispatches on, in ONE record.

    Every field is REQUIRED. That is the whole design: the six drifting maps existed
    because each new framework could be added to whichever map its author happened to be
    editing, and the census test over this record is what makes a half-added framework a
    test failure instead of a silent no-op at model 8,000.

    Parameters
    ----------
    framework:
        The canonical member this row describes.
    aliases:
        Every accepted spelling, normalized lowercase. Includes the canonical value.
    env_intent:
        Fixed environment intent, or :data:`TORCH_PACKAGE_DERIVED_INTENT` for torch, or
        ``None`` when no lane exists.
    lane_phase:
        Campaign phase the framework's work belongs to.
    lane_status:
        Whether this campaign can execute the framework at all.
    no_lane_reason, no_lane_disposition:
        Typed reason and terminal disposition when it cannot; empty/``None`` otherwise.
    torchlens_backend:
        TorchLens backend name, or ``None`` where none exists.
    seed_protocol, inference_context, mode_protocol, parameter_counter,
    host_array_converter, realization_barrier:
        Named per-framework protocols. Text, because their implementations live in the
        worker and codec modules; the census asserts presence so a framework cannot be
        added without someone deciding each one.
    dtype_prefixes:
        Module prefixes this framework's ``str(dtype)`` emits, cross-checked against the
        canonical dtype codec.
    weight_suffixes:
        Checkpoint file suffixes, used by checkpoint policy.
    adapter_call_method:
        Native method the transparent forward adapter delegates to.
    requires_device_profile:
        Whether execution needs an explicit device profile.
    """

    framework: Framework
    aliases: tuple[str, ...]
    env_intent: Optional[str]
    lane_phase: EnvironmentPhase
    lane_status: LaneStatus
    no_lane_reason: str
    no_lane_disposition: Optional[str]
    torchlens_backend: Optional[str]
    seed_protocol: str
    inference_context: str
    mode_protocol: str
    parameter_counter: str
    host_array_converter: str
    realization_barrier: str
    dtype_prefixes: tuple[str, ...]
    weight_suffixes: tuple[str, ...]
    adapter_call_method: str
    requires_device_profile: bool


FRAMEWORK_CAPABILITIES: Mapping[Framework, FrameworkCapability] = {
    Framework.TORCH: FrameworkCapability(
        framework=Framework.TORCH,
        aliases=("pytorch", "torch"),
        env_intent=TORCH_PACKAGE_DERIVED_INTENT,
        lane_phase=EnvironmentPhase.PYTORCH,
        lane_status=LaneStatus.EXECUTABLE,
        no_lane_reason="",
        no_lane_disposition=None,
        torchlens_backend="torch",
        seed_protocol="torch.manual_seed",
        inference_context="torch.inference_mode",
        mode_protocol="module.train/module.eval",
        parameter_counter="module.parameters-numel",
        host_array_converter="payload_codec.to_host_array",
        realization_barrier="eager",
        dtype_prefixes=("torch.",),
        weight_suffixes=(".pt", ".pth", ".bin", ".safetensors", ".ckpt"),
        adapter_call_method="forward",
        requires_device_profile=True,
    ),
    Framework.TENSORFLOW: FrameworkCapability(
        framework=Framework.TENSORFLOW,
        aliases=("tensorflow", "tf", "keras", "tf-keras"),
        env_intent="tf-keras-arm64",
        lane_phase=EnvironmentPhase.NATIVE_TAIL,
        lane_status=LaneStatus.EXECUTABLE,
        no_lane_reason="",
        no_lane_disposition=None,
        torchlens_backend="tf",
        seed_protocol="tf.random.set_seed",
        inference_context="none-required",
        mode_protocol="keras-training-kwarg",
        parameter_counter="keras-count-params",
        host_array_converter="payload_codec.to_host_array",
        realization_barrier="eager",
        dtype_prefixes=("tf.", "tensorflow."),
        weight_suffixes=(".h5", ".keras", ".pb", ".index", ".ckpt"),
        adapter_call_method="__call__",
        requires_device_profile=False,
    ),
    Framework.JAX: FrameworkCapability(
        framework=Framework.JAX,
        aliases=("jax", "flax"),
        env_intent="jax-flax-arm64",
        lane_phase=EnvironmentPhase.NATIVE_TAIL,
        lane_status=LaneStatus.EXECUTABLE,
        no_lane_reason="",
        no_lane_disposition=None,
        torchlens_backend="jax",
        seed_protocol="jax.random.PRNGKey-explicit",
        inference_context="none-required",
        mode_protocol="explicit-deterministic-argument",
        parameter_counter="pytree-leaf-size-sum",
        host_array_converter="payload_codec.to_host_array",
        realization_barrier="jax.block_until_ready",
        dtype_prefixes=("jax.numpy.", "jnp."),
        weight_suffixes=(".msgpack", ".flax", ".npz", ".safetensors"),
        adapter_call_method="apply",
        requires_device_profile=False,
    ),
    Framework.PADDLE: FrameworkCapability(
        framework=Framework.PADDLE,
        aliases=("paddle", "paddlepaddle"),
        env_intent="paddle-arm64",
        lane_phase=EnvironmentPhase.NATIVE_TAIL,
        lane_status=LaneStatus.EXECUTABLE,
        no_lane_reason="",
        no_lane_disposition=None,
        torchlens_backend="paddle",
        seed_protocol="paddle.seed",
        inference_context="paddle.no_grad",
        mode_protocol="layer.train/layer.eval",
        parameter_counter="layer.parameters-numel",
        host_array_converter="payload_codec.to_host_array",
        realization_barrier="eager",
        dtype_prefixes=("paddle.",),
        weight_suffixes=(".pdparams", ".pdmodel", ".pdiparams"),
        adapter_call_method="forward",
        requires_device_profile=True,
    ),
    Framework.MLX: FrameworkCapability(
        framework=Framework.MLX,
        aliases=("mlx",),
        env_intent=None,
        lane_phase=EnvironmentPhase.NATIVE_TAIL,
        lane_status=LaneStatus.NO_ENVIRONMENT_LANE,
        no_lane_reason="no mlx environment intent exists in envs/registry.yml",
        no_lane_disposition=BACKEND_PARITY_DEFERRAL_STATUS_CODE,
        torchlens_backend="mlx",
        seed_protocol="mlx.core.random.seed",
        inference_context="none-required",
        mode_protocol="module.train/module.eval",
        parameter_counter="module.parameters-tree-size",
        host_array_converter="payload_codec.to_host_array",
        realization_barrier="mlx.core.eval",
        dtype_prefixes=("mlx.core.",),
        weight_suffixes=(".safetensors", ".npz"),
        adapter_call_method="__call__",
        requires_device_profile=False,
    ),
    Framework.TINYGRAD: FrameworkCapability(
        framework=Framework.TINYGRAD,
        aliases=("tinygrad",),
        env_intent=None,
        lane_phase=EnvironmentPhase.NATIVE_TAIL,
        lane_status=LaneStatus.NO_ENVIRONMENT_LANE,
        no_lane_reason="no tinygrad environment intent exists in envs/registry.yml",
        no_lane_disposition=BACKEND_PARITY_DEFERRAL_STATUS_CODE,
        torchlens_backend="tinygrad",
        seed_protocol="tinygrad-global-seed",
        inference_context="tinygrad.Tensor.no_grad",
        mode_protocol="tinygrad.Tensor.training-flag",
        parameter_counter="state-dict-leaf-size-sum",
        host_array_converter="payload_codec.to_host_array",
        realization_barrier="tinygrad.Tensor.realize",
        dtype_prefixes=("tinygrad.dtype.", "dtypes."),
        weight_suffixes=(".safetensors", ".npz"),
        adapter_call_method="__call__",
        requires_device_profile=False,
    ),
}

#: Frameworks legal as ``run_framework``. Every capability row is one, today; the set is
#: explicit so a future non-executing member (a pure origin backend) cannot leak into a
#: dispatch by being added to the enum alone.
RUN_FRAMEWORKS: frozenset[Framework] = frozenset(FRAMEWORK_CAPABILITIES)

FRAMEWORK_ALIASES: Mapping[str, Framework] = {
    alias: capability.framework
    for capability in FRAMEWORK_CAPABILITIES.values()
    for alias in capability.aliases
}

ORIGIN_FRAMEWORK_ALIASES: Mapping[str, OriginFramework] = {
    **{alias: OriginFramework(framework.value) for alias, framework in FRAMEWORK_ALIASES.items()},
    **{origin.value: origin for origin in OriginFramework},
    "mxnet": OriginFramework.MXNET,
    "apache-mxnet": OriginFramework.MXNET,
    "mindspore": OriginFramework.MINDSPORE,
    "caffe": OriginFramework.CAFFE,
    "caffe2": OriginFramework.CAFFE,
    "darknet": OriginFramework.DARKNET,
    "onnx": OriginFramework.ONNX,
    "chainer": OriginFramework.CHAINER,
    "theano": OriginFramework.THEANO,
}

# The roster-marker scan deliberately does NOT live here. There is ONE framework
# derivation for roster rows -- `constants.framework_from_markers`, in the import-light
# module the read-only staging tools can reach without NumPy -- and a second copy here
# would be the seventh drifting map. Its alias table is also deliberately narrower than
# `FRAMEWORK_ALIASES` (no mlx/tinygrad); the reason is documented at that table.


def normalize_framework(value: object) -> Framework:
    """Resolve any accepted spelling to its canonical framework, or refuse typed.

    Parameters
    ----------
    value:
        Producer-supplied or derived framework text.

    Returns
    -------
    Framework
        The one canonical member.

    Raises
    ------
    FrameworkVocabularyError
        For empty text or any spelling outside the closed alias vocabulary.
    """

    text = str(value or "").strip().lower()
    if not text:
        raise FrameworkVocabularyError("framework must be a non-empty closed-vocabulary spelling")
    framework = FRAMEWORK_ALIASES.get(text)
    if framework is None:
        raise FrameworkVocabularyError(
            f"framework {str(value)!r} is outside the closed vocabulary {sorted(FRAMEWORK_ALIASES)}"
        )
    return framework


def normalize_origin_framework(value: object) -> OriginFramework:
    """Resolve any accepted origin spelling to its canonical member, or refuse typed.

    Parameters
    ----------
    value:
        Producer-supplied original-framework text.

    Returns
    -------
    OriginFramework
        The one canonical member.

    Raises
    ------
    FrameworkVocabularyError
        For empty text or any spelling outside the closed origin vocabulary.
    """

    text = str(value or "").strip().lower()
    if not text:
        raise FrameworkVocabularyError(
            "original framework must be a non-empty closed-vocabulary spelling"
        )
    origin = ORIGIN_FRAMEWORK_ALIASES.get(text)
    if origin is None:
        raise FrameworkVocabularyError(
            f"original framework {str(value)!r} is outside the closed vocabulary "
            f"{sorted(ORIGIN_FRAMEWORK_ALIASES)}"
        )
    return origin


#: Grammar for ``implementation.original_framework``: one lowercase token, never prose.
#:
#: An ENUM would be wrong here and closing it would destroy honest facts. An architecture's
#: original framework is a world-fact, and the world is wider than anything TorchLens can
#: execute -- the live pilot records ``"Brian"`` (a spiking-network simulator), and MATLAB,
#: Lua Torch and CNTK origins are all real and all legitimate. What must be impossible is
#: PROSE, because this field is compared against the closed run framework to decide whether
#: a native-lane disclosure is owed, and ``"originally TF1, ported"`` silently reads as
#: "different from pytorch" for accidental reasons rather than declared ones.
ORIGIN_FRAMEWORK_TOKEN_PATTERN = r"^[a-z0-9][a-z0-9._+-]*$"


def origin_framework_token(value: object) -> str:
    """Return a validated lowercase origin-framework token, refusing prose typed.

    Parameters
    ----------
    value:
        Producer-written ``implementation.original_framework``.

    Returns
    -------
    str
        The token unchanged. Canonical members of :class:`OriginFramework` round-trip; an
        unlisted-but-real origin (``"brian"``, ``"matlab"``) is accepted as itself so an
        honest fact stays recordable.

    Raises
    ------
    FrameworkVocabularyError
        For a non-string, empty text, or anything that is not a single lowercase token.
    """

    if not isinstance(value, str) or re.fullmatch(ORIGIN_FRAMEWORK_TOKEN_PATTERN, value) is None:
        raise FrameworkVocabularyError(
            f"original_framework {value!r} must be one lowercase token matching "
            f"{ORIGIN_FRAMEWORK_TOKEN_PATTERN}, never prose"
        )
    return value


def framework_capability(framework: Framework | str) -> FrameworkCapability:
    """Return the total capability record for one framework.

    Parameters
    ----------
    framework:
        Canonical member or any accepted spelling.

    Returns
    -------
    FrameworkCapability
        The framework's complete behaviour record.
    """

    resolved = framework if isinstance(framework, Framework) else normalize_framework(framework)
    return FRAMEWORK_CAPABILITIES[resolved]


@dataclass(frozen=True)
class FrameworkFact:
    """A framework claim with the evidence that supports it.

    Parameters
    ----------
    framework:
        Canonical framework claimed.
    basis:
        Closed evidence basis.
    evidence:
        Literal, attributable evidence -- the markers matched, the import root probed, or
        the checker-confirmed author declaration. Never empty.
    """

    framework: Framework
    basis: FrameworkBasis
    evidence: str

    def __post_init__(self) -> None:
        """Require literal evidence; a basis label with nothing behind it is not a fact."""

        if not self.evidence.strip():
            raise ValueError("a framework fact must carry literal evidence")

    @property
    def is_execution_authority(self) -> bool:
        """Return whether this basis may decide what actually runs.

        A roster-name marker may stage and order work. It may not decide execution: it is
        the weakest evidence available, it never reads the source the author lane already
        fetches, and its measured false-positive rate is 13%.
        """

        return self.basis is not FrameworkBasis.MARKER


@dataclass(frozen=True)
class FrameworkReconciliation:
    """Whether a declared run framework and a routed environment intent agree."""

    declared: Framework
    routed_intent: str
    intent_framework: Optional[Framework]
    agrees: bool
    disposition: Optional[str]
    detail: str


#: Environment intents that serve torch work, mirrored from ``envs/registry.yml``.
#:
#: Duplicated here deliberately and cross-checked by the dispatch-arm census against the
#: registry file, because backend.py must stay import-light (no YAML/JSON read at import)
#: while still being able to answer "does this intent serve the framework the author
#: declared" -- the comparison nothing in the pipeline made.
TORCH_ENVIRONMENT_INTENTS: frozenset[str] = frozenset(
    {
        "core",
        "graph",
        "audio",
        "speech",
        "mmlab",
        "tabular-ts",
        "recsys",
        "detect-misc",
        "legacy-torch",
        "oddballs",
    }
)

INTENT_FRAMEWORKS: Mapping[str, Framework] = {
    **{intent: Framework.TORCH for intent in TORCH_ENVIRONMENT_INTENTS},
    **{
        capability.env_intent: capability.framework
        for capability in FRAMEWORK_CAPABILITIES.values()
        if capability.env_intent is not None
        and capability.env_intent != TORCH_PACKAGE_DERIVED_INTENT
    },
}


def reconcile_execution_framework(
    *, declared_run_framework: object, routed_intent: str
) -> FrameworkReconciliation:
    """Refuse a model whose declared framework the routed lane cannot serve.

    Nothing in the shipped pipeline compared these two. The lane came from a substring
    match on the roster row's name; the run framework came from the author after reading
    the source; and a disagreement meant the model executed in an environment built for a
    different framework, burned the build, and failed against the MODEL rather than
    against the routing.

    Parameters
    ----------
    declared_run_framework:
        The author-declared (or driver-derived) run framework.
    routed_intent:
        Environment intent the router selected.

    Returns
    -------
    FrameworkReconciliation
        Structured verdict; ``agrees`` is False with a typed disposition on disagreement.
    """

    declared = normalize_framework(declared_run_framework)
    intent_framework = INTENT_FRAMEWORKS.get(routed_intent)
    if intent_framework is None:
        return FrameworkReconciliation(
            declared=declared,
            routed_intent=routed_intent,
            intent_framework=None,
            agrees=False,
            disposition=BACKEND_PARITY_DEFERRAL_STATUS_CODE,
            detail=f"intent {routed_intent!r} declares no framework",
        )
    if intent_framework is declared:
        return FrameworkReconciliation(
            declared=declared,
            routed_intent=routed_intent,
            intent_framework=intent_framework,
            agrees=True,
            disposition=None,
            detail=f"intent {routed_intent!r} serves {declared.value}",
        )
    return FrameworkReconciliation(
        declared=declared,
        routed_intent=routed_intent,
        intent_framework=intent_framework,
        agrees=False,
        disposition=BACKEND_PARITY_DEFERRAL_STATUS_CODE,
        detail=(
            f"declared run framework {declared.value} disagrees with intent "
            f"{routed_intent!r}, which serves {intent_framework.value}"
        ),
    )


def require_native_lane_disposition(
    *,
    original_framework: object,
    run_framework: object,
    disposition: Optional[str],
) -> Optional[str]:
    """Require a typed disclosure whenever a model runs outside its origin framework.

    Parameters
    ----------
    original_framework:
        Framework the architecture is implemented in.
    run_framework:
        Framework the model executes in.
    disposition:
        Member of :data:`NATIVE_LANE_DISPOSITIONS`, or ``None``.

    Returns
    -------
    str | None
        The accepted disposition, or ``None`` when the frameworks agree and none is owed.

    Raises
    ------
    NativeLaneUndisclosedError
        When the frameworks differ and the disposition is absent or outside the closed set.
    """

    origin = origin_framework_token(original_framework)
    run = normalize_framework(run_framework)
    if origin == run.value:
        return None
    if disposition not in NATIVE_LANE_DISPOSITIONS:
        raise NativeLaneUndisclosedError(
            f"{origin} architecture running as {run.value} requires a "
            f"native_lane_disposition from {sorted(NATIVE_LANE_DISPOSITIONS)}; got "
            f"{disposition!r}"
        )
    return disposition


@dataclass(frozen=True)
class ProducerDispatchField:
    """One producer-written string that reaches a dispatch or an identity.

    Parameters
    ----------
    field:
        Schema leaf name as producers write it.
    schema_paths:
        Exact ``(schema_version, normalized_path)`` placements that reach a sink. Paths, not
        just names, because the SAME name can be a dispatch key in one block and a
        world-fact claim in another: ``implementation.run_framework`` steers execution while
        ``external_metadata.original_framework`` records where an architecture came from
        (the pilot carries ``"Brian"``), and closing the latter would make an honest fact
        unrecordable.
    schema_closure, closure_note:
        Whether the schema itself closes the field, and -- when it cannot -- the disclosed
        reason. See :data:`CODE_ONLY_CLOSURE_FIELDS`.
    sinks:
        Dispatch/identity sinks that consume it.
    validator:
        Dotted path of the function that closes it.
    refusal:
        Dotted path of the exception type raised for a value outside the closed vocabulary.
        A path rather than a type so this inventory stays at the bottom of the import graph:
        two of the validators live in heavy modules (``proposal`` pulls in metadata, recipe
        and the fetcher) and importing them here would drag the whole chain into every
        routing decision.
    canonical_sample, free_text_sample:
        An accepted value and a live drift sample the validator must refuse. The drift
        samples are real: they come from the shipped ledger, the golden transcripts, and
        the step-100 terminal triage.
    finding:
        The audit row this closure discharges.
    """

    field: str
    schema_paths: tuple[tuple[str, str], ...]
    schema_closure: "SchemaClosure"
    closure_note: str
    sinks: tuple[str, ...]
    validator: str
    refusal: str
    canonical_sample: str
    free_text_sample: str
    finding: str

    @property
    def refusal_type(self) -> type[Exception]:
        """Resolve and return the exception type the validator raises for free text."""

        module_path, _, name = self.refusal.rpartition(".")
        resolved = getattr(import_module(module_path), name)
        if not (isinstance(resolved, type) and issubclass(resolved, Exception)):
            raise TypeError(f"{self.refusal} is not an exception type")
        return resolved


_MODEL_V3 = "menagerie.crawler.model.v3"
_PROPOSAL_V3 = "menagerie.crawler.author-proposal.v3"

DISPATCH_AND_IDENTITY_SINKS: frozenset[str] = frozenset(
    {
        # identity preimages
        "derive_execution_identity",
        "compute_execution_identity",
        # framework/backend dispatch
        "route_model",
        "adapt_native_model",
        "normalize_framework",
        "normalize_origin_framework",
        "framework_capability",
        # value dispatch on a producer string
        "parse_device_policy",
        "canonical_dtype",
        "_convert_array",
        "_seed_frameworks",
    }
)
"""Closed set of sinks that turn a string into BEHAVIOUR or into an IDENTITY.

A sink is named here because passing a producer string into it has a mechanical consequence
nobody can review afterwards: a dispatch selects code, and an identity preimage decides
whether cached work is reused or re-driven. The producer-field census scans the WHOLE
package for calls to these names, so a NEW call site of an existing sink is caught without
touching this set; adding a sink is a reviewed change.
"""

CODE_ONLY_CLOSURE_FIELDS: frozenset[str] = frozenset({"device_policy", "dtype"})
"""The two dispatch fields whose closure lives in CODE, not in the schema, and why.

Both are disclosed rather than quietly excused, and the set is set-compared by the census,
so a third field cannot join it without a reviewed change. Neither exemption grants
anything: each field still has an executable refusal at the point it steers behaviour, and
the census proves that refusal bites. What the exemption buys is only that immutable
history stays loadable -- canonical records and golden refusal transcripts written before
the refusal existed carry prose in these fields, and a schema enum would make the ledger
unreadable rather than making the pipeline safer.
"""


class SchemaClosure(StrEnum):
    """Where a producer field's closed vocabulary is enforced."""

    CLOSED = "closed"
    CODE_ONLY = "code-only"


PRODUCER_DISPATCH_INVENTORY: tuple[ProducerDispatchField, ...] = (
    ProducerDispatchField(
        field="run_framework",
        schema_paths=(
            (_MODEL_V3, "$.implementation.run_framework"),
            (_PROPOSAL_V3, "$.proposed_facts.implementation.run_framework"),
        ),
        schema_closure=SchemaClosure.CLOSED,
        closure_note="",
        sinks=(
            "derive_execution_identity",
            "route_model",
            "adapt_native_model",
            "_convert_array",
            "_seed_frameworks",
        ),
        validator="menagerie.crawler.backend.normalize_framework",
        refusal="menagerie.crawler.backend.FrameworkVocabularyError",
        canonical_sample="pytorch",
        free_text_sample="PyTorch/2.x",
        finding="AREA_BACKENDS_opus B1 / catalog C-22",
    ),
    ProducerDispatchField(
        field="original_framework",
        schema_paths=(
            (_MODEL_V3, "$.implementation.original_framework"),
            (_PROPOSAL_V3, "$.proposed_facts.implementation.original_framework"),
        ),
        schema_closure=SchemaClosure.CLOSED,
        closure_note="",
        sinks=("adapt_native_model", "normalize_origin_framework"),
        validator="menagerie.crawler.backend.origin_framework_token",
        refusal="menagerie.crawler.backend.FrameworkVocabularyError",
        canonical_sample="tensorflow",
        free_text_sample="originally TF1, ported",
        finding="AREA_BACKENDS_opus B6 (native-lane disclosure)",
    ),
    ProducerDispatchField(
        field="recipe_type",
        schema_paths=(
            (_MODEL_V3, "$.implementation.recipe_type"),
            (_PROPOSAL_V3, "$.proposed_facts.implementation.recipe_type"),
        ),
        schema_closure=SchemaClosure.CLOSED,
        closure_note="",
        sinks=("derive_execution_identity",),
        validator="menagerie.crawler.constants.normalize_recipe_type",
        refusal="menagerie.crawler.constants.ProducerVocabularyError",
        canonical_sample="declarative-library",
        free_text_sample="declarative library recipe",
        finding="found by this census (C-22 sweep)",
    ),
    ProducerDispatchField(
        field="recipe_revision",
        schema_paths=(
            (_MODEL_V3, "$.implementation.recipe_revision"),
            (_PROPOSAL_V3, "$.proposed_facts.implementation.recipe_revision"),
        ),
        schema_closure=SchemaClosure.CLOSED,
        closure_note="",
        sinks=("derive_execution_identity",),
        validator="menagerie.crawler.constants.normalize_recipe_revision",
        refusal="menagerie.crawler.constants.ProducerVocabularyError",
        canonical_sample="sha256:" + "0" * 64,
        free_text_sample="revision 2",
        finding="found by this census (C-22 sweep)",
    ),
    ProducerDispatchField(
        field="device_policy",
        schema_paths=(
            (_MODEL_V3, "$.implementation.device_policy"),
            (_MODEL_V3, "$.input_contract.args[].device_policy"),
            (_MODEL_V3, "$.input_contract.kwargs[].device_policy"),
            (_PROPOSAL_V3, "$.proposed_facts.implementation.device_policy"),
            (_PROPOSAL_V3, "$.proposed_facts.input_contract.args[].device_policy"),
            (_PROPOSAL_V3, "$.proposed_facts.input_contract.kwargs[].device_policy"),
        ),
        schema_closure=SchemaClosure.CODE_ONLY,
        closure_note=(
            "Immutable canonical records and the golden refusal transcripts predate the "
            "runtime refusal and carry authored prose here ('moved to the device argument "
            "of make_dummy_call(seed, device)', 'cuda recommended: ...'). A schema enum "
            "would make that history unloadable without making the pipeline safer, so the "
            "closure is proposal.parse_device_policy at the acceptance gate and again at "
            "the driver's own consumption. Narrowing the schema belongs to the proposal-gate "
            "owner together with a records migration."
        ),
        sinks=("derive_execution_identity", "parse_device_policy"),
        validator="menagerie.crawler.proposal.parse_device_policy",
        refusal="menagerie.crawler.proposal.ProposalValidationError",
        canonical_sample="cpu",
        free_text_sample="constructed on cpu; caller moves the module with .to(device)",
        finding="step-100 DEVICE-POLICY-PROSE, rank 3 (4 of the first 22 terminals)",
    ),
    ProducerDispatchField(
        field="dtype",
        schema_paths=(
            (_MODEL_V3, "$.input_contract.args[].dtype"),
            (_MODEL_V3, "$.input_contract.kwargs[].dtype"),
            (_PROPOSAL_V3, "$.proposed_facts.input_contract.args[].dtype"),
            (_PROPOSAL_V3, "$.proposed_facts.input_contract.kwargs[].dtype"),
        ),
        schema_closure=SchemaClosure.CODE_ONLY,
        closure_note=(
            "The same field carries a NON-dispatch placeholder for kind='constructed' "
            "leaves, which materialize no tensor at all: live records spell those 'object' "
            "and 'not-applicable' (the m538 graph-object shape). An enum of real dtypes "
            "would refuse the honest placeholder, so the closure is payload_codec."
            "canonical_dtype at the tensor-materialization dispatch, where a dtype "
            "actually steers behaviour."
        ),
        sinks=("_convert_array", "canonical_dtype"),
        validator="menagerie.crawler.payload_codec.canonical_dtype",
        refusal="menagerie.crawler.payload_codec.DtypeVocabularyError",
        canonical_sample="float32",
        free_text_sample="a standard float image tensor",
        finding="AREA_BACKENDS_fable BK-1",
    ),
)


def _canonical_dtype(value: object) -> object:
    """Call the canonical dtype vocabulary through a lazy import.

    ``payload_codec`` needs NumPy; this module must import without it (see the note beside
    the imports).
    """

    from menagerie.crawler.payload_codec import canonical_dtype

    return canonical_dtype(value)


def _parse_device_policy(value: object) -> object:
    """Call the device-policy grammar through a lazy import.

    ``proposal`` pulls in metadata, recipe and the fetcher, so it must never be imported at
    THIS module's import time -- ``routing`` imports this module on every routing decision,
    and ``constants``/``backend`` sit at the bottom of the import graph deliberately.
    """

    from menagerie.crawler.proposal import parse_device_policy

    return parse_device_policy(value)


_PRODUCER_DISPATCH_VALIDATORS: Mapping[str, Callable[[object], object]] = {
    "run_framework": normalize_framework,
    "original_framework": origin_framework_token,
    "dtype": _canonical_dtype,
    "device_policy": _parse_device_policy,
    "recipe_type": normalize_recipe_type,
    "recipe_revision": normalize_recipe_revision,
}


def producer_dispatch_validator(field: str) -> Callable[[object], object]:
    """Return the live validator that closes one inventoried producer field.

    Parameters
    ----------
    field:
        Inventoried schema leaf name.

    Returns
    -------
    Callable[[object], object]
        The validator, so the census can prove the claimed closure is executable rather
        than documentary.
    """

    return _PRODUCER_DISPATCH_VALIDATORS[field]
