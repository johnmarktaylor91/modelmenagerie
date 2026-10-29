"""Closed declarative R1 recipes and audited typed adapter loading."""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import inspect
import os
import re
import sys
import threading
from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from menagerie.crawler.identity import canonical_json_bytes, compute_recipe_revision, hash_bytes
from menagerie.crawler.package_namespace import (
    canonical_distribution_name,
    inventory_row_provides_distribution,
)


class RecipeError(ValueError):
    """Raised when a recipe is opaque, executable text, or violates its typed contract."""


BuildModel = Callable[[], object]
MakeDummyCall = Callable[[int, str], tuple[tuple[object, ...], dict[str, object]]]

CONSTRUCT_NODE_KEY = "__construct__"
"""Marker key that turns a declarative kwargs mapping into a construct node."""

MAX_CONSTRUCT_DEPTH = 4
"""Maximum construct-node nesting depth admitted by the declarative grammar."""

MAX_CONSTRUCT_NODES = 16
"""Maximum total construct nodes admitted by one declarative recipe."""

MAX_POST_CONSTRUCT_CALLS = 8
"""Maximum bounded post-construction configuration calls per recipe."""

_GENERIC_CONTAINER_SYMBOLS = frozenset(
    {"Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict", "Module"}
)

CONSTRUCT_MODULE_ALLOWLIST = frozenset({"torch", "torch.nn", "torch.nn.utils.rnn"})
"""Exact auxiliary modules a construct node may name besides the pinned distribution.

A construct node's ``module`` is imported and its ``symbol`` is called, so the set
of reachable callables *is* the grammar's real bound. Without this the grammar is
a bounded syntax over an unbounded callable namespace: ``subprocess.run`` parses,
imports, and executes before any downstream type error is reached.

The bound is therefore "the artifact the recipe pins, plus a named auxiliary
allowlist". Membership is EXACT module names, not prefixes, and is derived from
the shapes that actually occur: ``torch`` (``torch.zeros``), ``torch.nn``
(``TransformerEncoderLayer``), and ``torch.nn.utils.rnn``
(``pack_padded_sequence``). Exactness keeps the auxiliary surface minimal --
``torch.hub``, ``torch.jit``, and ``torch.utils.collect_env`` all expose
general-purpose code/state loaders and none of them are on this list.

Extend this deliberately, one module at a time, with a test that names the real
recipe shape requiring it. Widening it to a top-level package prefix would hand
every construct node the whole of torch again.
"""


def _is_generic_container_reference(module: str, symbol: str) -> bool:
    """Return whether a declared module/symbol names a generic torch container.

    Parameters
    ----------
    module, symbol:
        Declared import module and constructor attribute.

    Returns
    -------
    bool
        True when the reference names a composition container rather than a
        published architecture entrypoint.
    """

    return symbol in _GENERIC_CONTAINER_SYMBOLS and (
        module == "torch.nn" or module.startswith("torch.nn.")
    )


@lru_cache(maxsize=None)
def _distribution_top_level_packages(distribution: str) -> frozenset[str]:
    """Return the top-level import packages the named distribution installs.

    This inverts the interpreter's package-to-distribution map so a distribution
    whose import package differs from its distribution name (``opencv-python``
    ships ``cv2``) still resolves without importing anything.

    Parameters
    ----------
    distribution:
        Declared pinned distribution name.

    Returns
    -------
    frozenset[str]
        Top-level package names attributed to the distribution, empty when the
        distribution is not installed in this interpreter.
    """

    wanted = canonical_distribution_name(distribution)
    return frozenset(
        package
        for package, names in _packages_distributions_snapshot().items()
        if any(canonical_distribution_name(name) == wanted for name in names)
    )


def assert_construct_namespace(module: str, *, distribution: str, context: str) -> None:
    """Refuse a construct-node module outside the pinned distribution and allowlist.

    This runs BEFORE the module is imported. Import order matters: importing an
    author-named module already executes that module's top-level code, so a
    post-import check would be a check applied after the thing it exists to
    prevent. The declared string is the only artifact available early enough.

    Two namespaces are admitted. The pinned distribution's own packages are
    admitted because that distribution is exactly what the recipe pins, versions,
    and digests -- running its code is the point of the R1 rung. Everything else
    must be an exact member of :data:`CONSTRUCT_MODULE_ALLOWLIST`.

    Parameters
    ----------
    module:
        Declared dotted module name of the construct node.
    distribution:
        Declared pinned distribution of the enclosing recipe.
    context:
        Location used in refusal messages.

    Raises
    ------
    RecipeError
        If the module belongs to neither admitted namespace.
    """

    if module in CONSTRUCT_MODULE_ALLOWLIST:
        return
    top_level = module.split(".")[0]
    if top_level not in sys.stdlib_module_names and canonical_distribution_name(
        top_level
    ) == canonical_distribution_name(distribution):
        # The name-equality fallback exists for distributions whose metadata is
        # not visible to this interpreter. The standard library is excluded from
        # it: stdlib modules import without any installed distribution, so
        # declaring ``distribution: "subprocess"`` must not admit ``subprocess``.
        return
    if top_level in _distribution_top_level_packages(distribution):
        return
    raise RecipeError(
        f"construct node at {context} names module {module!r}, which is outside the "
        f"pinned distribution {distribution!r} and the allowlisted modules "
        f"{sorted(CONSTRUCT_MODULE_ALLOWLIST)!r}"
    )


def assert_recipe_root_namespace(
    module: str, *, distribution: str, require_installed: bool
) -> None:
    """Refuse a recipe root module outside the pinned distribution's namespace.

    This runs BEFORE the root module is imported, for the same reason as
    :func:`assert_construct_namespace`: importing an author-named module already
    executes that module's top-level code, so a post-import refusal arrives
    after the thing it exists to prevent. ``subprocess.run`` as a recipe root
    parsed, imported, and executed its side effect before the returned-object
    provenance tripwire ever saw the result.

    The root rule is STRICTER than the construct-node rule: it has no auxiliary
    allowlist. The root names the model constructor itself, and
    :func:`assert_model_provenance` already requires the constructed class to be
    defined by the pinned distribution, so a root outside that distribution is
    at best a guaranteed later failure and at worst an arbitrary import. The
    allowlist entries (``torch``, ``torch.nn``, ``torch.nn.utils.rnn``) admit
    kwargs helpers, not models; when the pinned distribution *is* ``torch``
    they are admitted here as its own packages, and under any other pin they
    would only admit recipes destined to fail provenance.

    Parameters
    ----------
    module:
        Declared dotted root module name.
    distribution:
        Declared pinned distribution of the recipe.
    require_installed:
        When true (the pre-import load-time check), membership must come from
        the interpreter's installed-package metadata; the name-equality
        fallback for not-installed distributions is refused because an
        uninstalled distribution cannot legitimately supply an importable
        module -- everything ``import_module`` could still reach (the standard
        library, ``sys.modules`` injections, ``sys.path`` strays) is exactly
        the surface this check closes.

    Raises
    ------
    RecipeError
        If the module belongs to the standard library or is not a package of
        the pinned distribution.
    """

    top_level = module.split(".")[0]
    if top_level in sys.stdlib_module_names:
        raise RecipeError(
            f"declarative recipe root names module {module!r} from the Python "
            f"standard library; the standard library can never be the pinned "
            f"distribution {distribution!r}"
        )
    if top_level in _distribution_top_level_packages(distribution):
        return
    if not require_installed and canonical_distribution_name(
        top_level
    ) == canonical_distribution_name(distribution):
        return
    raise RecipeError(
        f"declarative recipe root names module {module!r}, which is not a package "
        f"of the pinned distribution {distribution!r}"
        + (" installed in this interpreter" if require_installed else "")
    )


def _reject_container_instance(value: object, description: str) -> None:
    """Refuse a materialized object whose type is a generic torch container.

    The root model and every construct-node result are materialized the same way,
    so they get the same instance check. Checking only the root leaves the exact
    asymmetry composition laundering needs: a returned-object check on the root
    and a constructor-symbol-only check on everything the root is built from.

    Parameters
    ----------
    value:
        Materialized object.
    description:
        Leading refusal clause naming what constructed the object.

    Raises
    ------
    RecipeError
        If the object's type is one of the generic torch containers.
    """

    value_type = type(value)
    module_name = str(getattr(value_type, "__module__", "") or "")
    if module_name.split(".")[0] != "torch":
        return
    import torch

    containers: tuple[object, ...] = (
        torch.nn.Sequential,
        torch.nn.ModuleList,
        torch.nn.ModuleDict,
        torch.nn.ParameterList,
        torch.nn.ParameterDict,
        torch.nn.Module,
    )
    if any(value_type is container for container in containers):
        raise RecipeError(
            f"{description} {module_name}.{value_type.__qualname__}; "
            "a composed container cannot claim R1"
        )


def _reject_container_constructor(constructor: object, context: str) -> None:
    """Refuse a resolved constructor whose identity is a generic torch container.

    Name-based parse rules are defeated by re-export laundering, so this check
    compares the resolved object identity against the actual torch container
    classes whenever the resolved constructor originates from torch.

    Parameters
    ----------
    constructor:
        Resolved callable constructor.
    context:
        Human-readable reference used in the refusal.

    Raises
    ------
    RecipeError
        If the constructor is one of the generic torch.nn containers.
    """

    module_name = str(getattr(constructor, "__module__", "") or "")
    if module_name.split(".")[0] != "torch":
        return
    import torch

    containers: tuple[object, ...] = (
        torch.nn.Sequential,
        torch.nn.ModuleList,
        torch.nn.ModuleDict,
        torch.nn.ParameterList,
        torch.nn.ParameterDict,
        torch.nn.Module,
    )
    if any(constructor is container for container in containers):
        raise RecipeError(
            f"generic torch.nn containers are refused in declarative R1 recipes: {context}"
        )


def _validate_construct_node_spec(
    spec: Any, context: str, *, distribution: str
) -> Mapping[str, Any]:
    """Validate the exact three-field construct-node payload.

    Parameters
    ----------
    spec:
        Candidate ``{"module", "symbol", "kwargs"}`` payload.
    context:
        Location used in refusal messages.
    distribution:
        Pinned distribution bounding the construct node's callable namespace.
        Keyword-only and required so no call site can resolve a construct node
        against an unbounded namespace by omission.

    Returns
    -------
    Mapping[str, Any]
        The validated construct payload.

    Raises
    ------
    RecipeError
        If the payload deviates from the closed construct-node shape or names a
        module outside the bounded namespace.
    """

    if not isinstance(spec, Mapping) or set(spec) != {"module", "symbol", "kwargs"}:
        raise RecipeError(
            f"construct node at {context} must declare exactly module, symbol, and kwargs"
        )
    module = spec["module"]
    symbol = spec["symbol"]
    if not isinstance(module, str) or not all(part.isidentifier() for part in module.split(".")):
        raise RecipeError(f"construct node at {context} module must be a dotted identifier")
    if not isinstance(symbol, str) or not symbol.isidentifier():
        raise RecipeError(f"construct node at {context} symbol must be a direct identifier")
    if _is_generic_container_reference(module, symbol):
        raise RecipeError(
            f"generic torch.nn containers are refused in declarative R1 recipes: {module}.{symbol}"
        )
    assert_construct_namespace(module, distribution=distribution, context=context)
    kwargs = spec["kwargs"]
    if not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
        raise RecipeError(f"construct node at {context} kwargs must be a string-keyed mapping")
    return spec


def _walk_construct_values(
    value: Any, *, context: str, depth: int, counter: list[int], distribution: str
) -> None:
    """Recursively validate construct nodes inside declarative JSON values.

    Parameters
    ----------
    value:
        JSON-compatible declarative value.
    context:
        Location used in refusal messages.
    depth:
        Number of enclosing construct nodes.
    counter:
        One-slot mutable total construct-node count.
    distribution:
        Pinned distribution bounding every construct node's callable namespace.

    Raises
    ------
    RecipeError
        If a construct node is malformed, too deep, too numerous, or names a
        module outside the bounded namespace.
    """

    if isinstance(value, Mapping):
        if CONSTRUCT_NODE_KEY in value:
            if set(value) != {CONSTRUCT_NODE_KEY}:
                raise RecipeError(
                    f"construct node at {context} must carry only the {CONSTRUCT_NODE_KEY!r} key"
                )
            if depth + 1 > MAX_CONSTRUCT_DEPTH:
                raise RecipeError(
                    f"construct graph exceeds the maximum depth of {MAX_CONSTRUCT_DEPTH}"
                )
            counter[0] += 1
            if counter[0] > MAX_CONSTRUCT_NODES:
                raise RecipeError(
                    f"construct graph exceeds the maximum of {MAX_CONSTRUCT_NODES} nodes"
                )
            spec = _validate_construct_node_spec(
                value[CONSTRUCT_NODE_KEY], context, distribution=distribution
            )
            for name, child in spec["kwargs"].items():
                _walk_construct_values(
                    child,
                    context=f"{context}.{name}",
                    depth=depth + 1,
                    counter=counter,
                    distribution=distribution,
                )
            return
        for name, child in value.items():
            _walk_construct_values(
                child,
                context=f"{context}.{name}",
                depth=depth,
                counter=counter,
                distribution=distribution,
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_construct_values(
                child,
                context=f"{context}[{index}]",
                depth=depth,
                counter=counter,
                distribution=distribution,
            )


def _reject_construct_nodes(value: Any, context: str) -> None:
    """Refuse construct nodes anywhere inside a plain-JSON-only value.

    Parameters
    ----------
    value:
        JSON-compatible value that must stay free of construct nodes.
    context:
        Location used in refusal messages.

    Raises
    ------
    RecipeError
        If any nested mapping carries the construct marker key.
    """

    if isinstance(value, Mapping):
        if CONSTRUCT_NODE_KEY in value:
            raise RecipeError(f"{context} must be plain JSON without construct nodes")
        for name, child in value.items():
            _reject_construct_nodes(child, f"{context}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_construct_nodes(child, f"{context}[{index}]")


@lru_cache(maxsize=1)
def _packages_distributions_snapshot() -> Mapping[str, tuple[str, ...]]:
    """Return the interpreter's frozen top-level-package to distribution map.

    Returns
    -------
    Mapping[str, tuple[str, ...]]
        Immutable copy of ``importlib.metadata.packages_distributions()``. The
        environment cannot change inside one worker process, so one snapshot is
        both correct and cheap.
    """

    return {
        name: tuple(values) for name, values in importlib.metadata.packages_distributions().items()
    }


def assert_model_provenance(model: object, distribution: str) -> None:
    """Fail closed unless the constructed model's class is defined by the pinned distribution.

    Parse-time grammar rules bound what a declarative recipe can *express*; this
    runtime tripwire bounds what is actually *constructed*. It closes composition
    laundering: a model assembled from generic primitives, or resolved through a
    factory into another package's class, cannot carry the R1 label of the
    declared distribution.

    Parameters
    ----------
    model:
        Constructed native model object.
    distribution:
        Declared pinned distribution name.

    Raises
    ------
    RecipeError
        If the model type is a generic container, its defining module cannot be
        attributed to any installed distribution, or the attribution does not
        include the declared distribution.
    """

    model_type = type(model)
    module_name = str(getattr(model_type, "__module__", "") or "")
    qualified = f"{module_name}.{model_type.__qualname__}"
    top_level = module_name.split(".")[0]
    _reject_container_instance(model, "declarative recipe constructed a generic container")
    if not top_level or top_level in {"builtins", "__main__"}:
        raise RecipeError(
            f"constructed model type {qualified} has no importable defining module "
            "and cannot claim R1"
        )
    wanted = canonical_distribution_name(distribution)
    attributed = _packages_distributions_snapshot().get(top_level, ())
    if not any(canonical_distribution_name(name) == wanted for name in attributed):
        described = sorted(attributed) if attributed else "no installed distribution"
        raise RecipeError(
            f"constructed model type {qualified} is not defined by the pinned "
            f"distribution {distribution!r}: attributed to {described}"
        )


def _is_disabling_pretrained_value(value: Any) -> bool:
    """Return whether a declarative value explicitly disables pretrained assets.

    Parameters
    ----------
    value:
        JSON-compatible constructor value.

    Returns
    -------
    bool
        True only for conventional explicit opt-out values.
    """

    return (
        value is None
        or value is False
        or (isinstance(value, str) and value.strip().lower() in {"", "none", "random"})
    )


PRETRAINED_PARAMETER_NAMES = frozenset(
    {
        "checkpoint",
        "checkpoint_path",
        "ckpt",
        "ckpt_path",
        "encoder_weights",
        "init_checkpoint",
        "init_ckpt",
        "load_pretrained",
        "load_weights",
        "pretrained",
        "pretrained_backbone",
        "pretrained_cfg",
        "pretrained_cfg_overlay",
        "pretrained_model",
        "pretrained_model_name_or_path",
        "pretrained_path",
        "pretrained_weights",
        "use_pretrained",
        "weights",
        "weights_backbone",
        "weights_path",
        "weights_url",
    }
)
"""Constructor keywords KNOWN to load pretrained assets when left enabled.

This set is a machine FLOOR, not a definition. The author still has to declare
every pretrained keyword its pinned constructor exposes, including spellings
absent from here; what this set adds is a check the author cannot talk its way
past, because the driver derives it from the real signature rather than from the
declaration.

Membership is by EXACT name on purpose. A prefix or substring rule would refuse
``pretrained_window_sizes`` -- a Swin geometry list that loads nothing -- and a
rule that manufactures a refusal out of a spelling is a wall, not a tripwire.
"""


def _is_construct_node(value: Any) -> bool:
    """Return whether a declarative value is a build-time construct node.

    Parameters
    ----------
    value:
        JSON-compatible declarative value.

    Returns
    -------
    bool
        True for the single-key ``__construct__`` mapping form.
    """

    return isinstance(value, Mapping) and set(value) == {CONSTRUCT_NODE_KEY}


def validate_pretrained_disable_fields(
    kwargs: Mapping[str, Any], disable_fields: Sequence[str]
) -> None:
    """Validate explicit pretrained opt-outs against constructor keyword values.

    Parameters
    ----------
    kwargs:
        Complete declarative constructor keyword mapping.
    disable_fields:
        Fields claimed to disable pretrained assets.

    Raises
    ------
    RecipeError
        If a field is duplicated, absent, or does not carry a disabling value.
    """

    if len(set(disable_fields)) != len(disable_fields):
        raise RecipeError("pretrained_disable_fields must not contain duplicates")
    for field in disable_fields:
        if field not in kwargs:
            raise RecipeError(
                f"pretrained disable field {field!r} is absent from constructor kwargs"
            )
        if not _is_disabling_pretrained_value(kwargs[field]):
            raise RecipeError(
                f"pretrained disable field {field!r} does not carry a disabling value"
            )


def enabled_pretrained_kwargs(kwargs: Mapping[str, Any]) -> tuple[str, ...]:
    """Return declared kwargs naming a known pretrained keyword without disabling it.

    Parameters
    ----------
    kwargs:
        Complete declarative constructor keyword mapping.

    Returns
    -------
    tuple[str, ...]
        Sorted keyword names left enabled.
    """

    return tuple(
        sorted(
            name
            for name, value in kwargs.items()
            if name in PRETRAINED_PARAMETER_NAMES and not _is_disabling_pretrained_value(value)
        )
    )


def validate_pretrained_disposition(
    kwargs: Mapping[str, Any], disable_fields: Sequence[str], *, fields_absent: bool
) -> None:
    """Require an R1 recipe to state, positively, how it disables pretrained assets.

    The safety requirement is that a capture never downloads weights the catalog
    does not describe. The requirement it was implemented as -- "name at least one
    field" -- is neither necessary nor sufficient for that. It is not necessary:
    ``MiniMaxForCausalLM(config)`` exposes no such keyword, so no value satisfies
    the rule and every ``XForCausalLM``-shaped R1 becomes a permanent dead record.
    It is not sufficient either: naming one harmless disabled field satisfied the
    check while a second, genuinely enabling keyword sat untouched in the same
    ``kwargs``.

    So the declaration is now a positive assertion with two honest spellings, and
    silence is still refused because it is exactly what "I did not think about
    it" looks like:

    * a non-empty ``pretrained_disable_fields``, validated exactly as before, or
    * ``pretrained_fields_absent`` -- "I read the pinned signature and it exposes
      no such keyword", a CLAIM the driver re-derives from the real constructor in
      :func:`load_declarative_recipe` rather than believing.

    On top of either spelling, a kwarg that names a known pretrained keyword and
    carries an enabling value is refused outright, and one that carries a
    disabling value but is missing from ``disable_fields`` is refused until it is
    declared there. Both clauses need no interpreter; the first closes the
    sufficiency hole above, the second keeps the disable list a complete account
    of the constructor's known pretrained surface.

    Parameters
    ----------
    kwargs:
        Complete declarative constructor keyword mapping.
    disable_fields:
        Fields claimed to disable pretrained assets.
    fields_absent:
        Author assertion that the pinned constructor exposes no pretrained
        keyword at all.

    Raises
    ------
    RecipeError
        If the disposition is silent, self-contradictory, leaves a known
        pretrained keyword enabled, or leaves a known pretrained keyword in
        ``kwargs`` undeclared in ``disable_fields``.
    """

    if fields_absent and disable_fields:
        raise RecipeError(
            "pretrained_fields_absent contradicts a non-empty pretrained_disable_fields"
        )
    if not fields_absent and not disable_fields:
        raise RecipeError(
            "R1_LIBRARY must declare its pretrained disposition: either name every "
            "pretrained keyword of the pinned constructor in pretrained_disable_fields, "
            "or assert pretrained_fields_absent when that constructor exposes none"
        )
    validate_pretrained_disable_fields(kwargs, disable_fields)
    enabled = enabled_pretrained_kwargs(kwargs)
    if enabled:
        raise RecipeError(
            f"declarative kwargs leave known pretrained keywords enabled: {list(enabled)!r}"
        )
    # Declaration completeness, on top of value safety: a kwargs key with a
    # known pretrained-asset name must itself appear in the disable list, even
    # when it already carries a disabling value. The disable list is the
    # author's account of the constructor's pretrained surface, and a known
    # keyword riding through it unlisted -- whatever its value today -- leaves
    # that account incomplete for every later reader.
    unlisted = sorted(
        name
        for name in kwargs
        if name in PRETRAINED_PARAMETER_NAMES and name not in set(disable_fields)
    )
    if unlisted:
        raise RecipeError(
            "kwargs carry pretrained-capable keys that pretrained_disable_fields does "
            f"not declare: {unlisted!r}"
        )


def constructor_pretrained_parameters(constructor: Any) -> tuple[str, ...]:
    """Return the known pretrained keywords the pinned constructor really exposes.

    Parameters
    ----------
    constructor:
        Resolved library constructor.

    Returns
    -------
    tuple[str, ...]
        Sorted exposed keyword names.

    Raises
    ------
    RecipeError
        If the signature cannot be inspected, so no claim about it is verifiable.
    """

    try:
        parameters = inspect.signature(constructor).parameters
    except (TypeError, ValueError) as exc:
        raise RecipeError(
            "cannot verify the pretrained disposition against constructor signature"
        ) from exc
    return tuple(
        sorted(
            name
            for name, parameter in parameters.items()
            if name in PRETRAINED_PARAMETER_NAMES
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        )
    )


def assert_constructor_pretrained_assets_disabled(
    constructor: Any, kwargs: Mapping[str, Any], *, fields_absent: bool = False
) -> None:
    """Refuse a pinned constructor that would still resolve a pretrained asset.

    This is the layer the author cannot author around. It reads the REAL
    signature, in the routed environment, immediately before construction, and
    checks the effective value of every known pretrained keyword -- the declared
    ``kwargs`` value when the recipe supplies one, the parameter DEFAULT when it
    does not. An enabling default nobody overrode (``encoder_weights="imagenet"``)
    is the exact shape the old declaration check waved through.

    Parameters
    ----------
    constructor:
        Resolved library constructor.
    kwargs:
        Complete declarative constructor keyword mapping.
    fields_absent:
        Author assertion that this constructor exposes no pretrained keyword.

    Raises
    ------
    RecipeError
        If a known pretrained keyword resolves to an enabling value, or if the
        absence assertion is contradicted by the signature.
    """

    exposed = constructor_pretrained_parameters(constructor)
    if fields_absent and exposed:
        raise RecipeError(
            "pretrained_fields_absent is contradicted by the pinned constructor signature, "
            f"which exposes {list(exposed)!r}"
        )
    parameters = inspect.signature(constructor).parameters
    enabled: list[str] = []
    for name in exposed:
        if name in kwargs:
            value = kwargs[name]
            if _is_construct_node(value) or not _is_disabling_pretrained_value(value):
                enabled.append(name)
            continue
        default = parameters[name].default
        if default is inspect.Parameter.empty or not _is_disabling_pretrained_value(default):
            enabled.append(name)
    if enabled:
        raise RecipeError(
            "pinned constructor would resolve pretrained assets: "
            f"{enabled!r} are neither disabled in kwargs nor disabled by default"
        )


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def resolve_environment_artifact_digest(
    packages: Sequence[Mapping[str, Any]],
    *,
    distribution: str,
    version: str,
) -> Optional[str]:
    """Derive the installed-distribution artifact digest from an exact inventory.

    The author stage has no package inventory, no environment identity, and no
    interpreter, so it cannot know this digest; the routed environment's exact
    resolved export does. This resolves the one row that PROVIDES
    ``distribution`` -- under either namespace spelling, see
    :mod:`menagerie.crawler.package_namespace` -- and returns its recorded
    artifact digest.

    An inventory that exists but provides no such row is NOT a null result. It
    means the routed environment does not install the artifact the recipe pins,
    so the R1 claim has nothing behind it and no digest can honestly be recorded.
    That is a typed refusal: a permanent R1 row whose machine-derived identity is
    silently absent is exactly the reading an instrument produces when it was
    structurally unable to measure anything.

    Parameters
    ----------
    packages:
        Exact ``name``/``version``/``sha256`` rows from the routed intent's
        resolved export or the materialized prefix inventory.
    distribution, version:
        Declared recipe distribution and version, in the Python distribution
        namespace.

    Returns
    -------
    str | None
        The canonical ``sha256:``-prefixed artifact digest. ``None`` ONLY when
        the caller supplied no inventory at all -- an unlocked or unrouted
        target, where the machine genuinely holds no instrument. It is never a
        fabricated digest and never a stand-in for an unsatisfied lookup.

    Raises
    ------
    RecipeError
        If a nonempty inventory provides no matching row, provides more than
        one, records a noncanonical digest, or records a version that
        contradicts the recipe.
    """

    if not packages:
        return None
    matches = [
        row
        for row in packages
        if isinstance(row, Mapping)
        and isinstance(row.get("name"), str)
        and inventory_row_provides_distribution(str(row["name"]), distribution)
    ]
    if not matches:
        raise RecipeError(
            f"routed environment does not install distribution {distribution!r}: no "
            "package row carries that name and none declares it as a provision"
        )
    digests = {str(row.get("sha256")) for row in matches}
    versions = {str(row.get("version")) for row in matches}
    if len(digests) != 1 or len(versions) != 1:
        raise RecipeError(f"environment inventory names distribution {distribution!r} ambiguously")
    # Name the inventory row too when the two namespaces spell it differently, so
    # a mismatch is debuggable without reading the provision registry.
    row_name = str(matches[0]["name"])
    located = (
        repr(distribution)
        if canonical_distribution_name(row_name) == canonical_distribution_name(distribution)
        else f"{distribution!r} (inventory package {row_name!r})"
    )
    digest = digests.pop()
    if _HASH_PATTERN.fullmatch(digest) is None:
        raise RecipeError(f"environment inventory digest for {located} is not a canonical sha256")
    resolved_version = versions.pop()
    if resolved_version != str(version).strip():
        raise RecipeError(
            f"recipe declares {located} version {version!r} but the routed "
            f"environment installs {resolved_version!r}"
        )
    return digest


def bind_library_artifact_digest(
    implementation: dict[str, Any],
    packages: Sequence[Mapping[str, Any]],
) -> bool:
    """Fill the machine-derived R1 artifact digest, refusing a conflicting claim.

    The digest identifies the installed distribution, which only the machine can
    derive, so this never trusts an author value over its own derivation and
    never silently overwrites one either: a conflicting supplied digest is a
    typed refusal, so the check cannot be left structurally dead.

    The digest is required whenever it is derivable. A routed environment that
    exposes an inventory must yield one, so the only surviving null is the case
    where no inventory exists at all -- an unlocked or unrouted target, which
    cannot materialize an environment and therefore cannot reach a run record
    either. In that case a supplied digest is refused rather than trusted:
    there is nothing to check it against.

    Parameters
    ----------
    implementation:
        Mutable proposal implementation block.
    packages:
        Exact package rows for the routed environment.

    Returns
    -------
    bool
        Whether the recipe bytes changed and dependent identities must rebind.

    Raises
    ------
    RecipeError
        If a supplied digest conflicts with the derived one, is malformed, is
        unverifiable, the routed environment does not install the pinned
        distribution, or the inventory itself is ambiguous.
    """

    if implementation.get("recipe_type") != "declarative-library":
        return False
    recipe = implementation.get("library_recipe")
    if not isinstance(recipe, dict):
        return False
    distribution = recipe.get("distribution")
    version = recipe.get("version")
    if not isinstance(distribution, str) or not isinstance(version, str):
        raise RecipeError("declarative recipe lacks a distribution and version to resolve")
    supplied = recipe.get("artifact_sha256")
    if supplied is not None and (
        not isinstance(supplied, str) or _HASH_PATTERN.fullmatch(supplied) is None
    ):
        raise RecipeError("supplied artifact_sha256 is not a canonical sha256 digest")
    derived = resolve_environment_artifact_digest(
        packages, distribution=distribution, version=version
    )
    if supplied is not None and derived is not None and supplied != derived:
        raise RecipeError(
            "supplied artifact_sha256 conflicts with the routed environment: "
            f"supplied {supplied}, derived {derived}"
        )
    if derived is None:
        if supplied is not None:
            raise RecipeError(
                "supplied artifact_sha256 cannot be verified: the routed environment "
                "exposes no package inventory"
            )
        if "artifact_sha256" in recipe:
            return False
        recipe["artifact_sha256"] = None
        return True
    if supplied is not None:
        return False
    recipe["artifact_sha256"] = derived
    return True


@dataclass(frozen=True)
class PostConstructCall:
    """One bounded declarative post-construction configuration call.

    Parameters
    ----------
    method:
        Public method name invoked on the constructed model.
    args, kwargs:
        Plain-JSON call arguments; construct nodes are refused here.
    """

    method: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = dataclass_field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> "PostConstructCall":
        """Validate and construct one closed post-construction call.

        Parameters
        ----------
        value:
            Candidate call mapping.
        context:
            Location used in refusal messages.

        Returns
        -------
        PostConstructCall
            Strict configuration call.
        """

        if not isinstance(value, Mapping):
            raise RecipeError(f"{context} must be an object")
        unknown = set(value) - {"method", "args", "kwargs"}
        if unknown:
            raise RecipeError(f"{context} has unknown fields: {sorted(unknown)!r}")
        method = value.get("method")
        if not isinstance(method, str) or not method.isidentifier():
            raise RecipeError(f"{context} method must be a direct identifier")
        if method.startswith("_"):
            raise RecipeError(f"{context} method must be a public name, not {method!r}")
        args = value.get("args", [])
        if not isinstance(args, (list, tuple)):
            raise RecipeError(f"{context} args must be a list")
        kwargs = value.get("kwargs", {})
        if not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
            raise RecipeError(f"{context} kwargs must be a string-keyed mapping")
        try:
            canonical_json_bytes({"args": list(args), "kwargs": dict(kwargs)})
        except (TypeError, ValueError) as exc:
            raise RecipeError(f"{context} arguments must be JSON-compatible") from exc
        _reject_construct_nodes(list(args), f"{context}.args")
        _reject_construct_nodes(dict(kwargs), f"{context}.kwargs")
        return cls(method=method, args=tuple(args), kwargs=dict(kwargs))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible call mapping.

        Returns
        -------
        dict[str, Any]
            Post-construction call payload.
        """

        return {"method": self.method, "args": list(self.args), "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class DeclarativeRecipe:
    """Closed R1 library constructor description.

    Parameters
    ----------
    distribution, version:
        Exact package distribution and version.
    module, symbol:
        Importable module and direct constructor attribute.
    kwargs:
        JSON-compatible constructor keyword arguments. A mapping value carrying
        the single ``__construct__`` key is a construct node resolved from a
        declared module/symbol at build time; everything else is literal JSON.
    artifact_sha256:
        Optional exact installed-artifact hash.
    pretrained_disable_fields:
        Keyword names explicitly set to disable pretrained assets.
    pretrained_fields_absent:
        Positive assertion that the pinned constructor exposes no pretrained
        keyword at all, so there is nothing to disable. It is the only honest
        spelling for a constructor such as ``MiniMaxForCausalLM(config)`` and is
        re-derived from the real signature at load time, never believed.
    post_construct:
        Bounded declarative configuration calls applied after construction.
    entrypoint:
        Optional public non-``forward`` call method delegated through the
        crawler-owned transparent adapter. ``None`` means native ``forward``.
    """

    distribution: str
    version: str
    module: str
    symbol: str
    kwargs: Mapping[str, Any]
    artifact_sha256: Optional[str] = None
    pretrained_disable_fields: tuple[str, ...] = ()
    pretrained_fields_absent: bool = False
    post_construct: tuple[PostConstructCall, ...] = ()
    entrypoint: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DeclarativeRecipe":
        """Validate and construct a closed declarative recipe.

        Parameters
        ----------
        value:
            Candidate recipe mapping.

        Returns
        -------
        DeclarativeRecipe
            Strict declarative recipe.
        """

        allowed = {
            "distribution",
            "version",
            "artifact_sha256",
            "module",
            "symbol",
            "kwargs",
            "pretrained_disable_fields",
            "pretrained_fields_absent",
            "post_construct",
            "entrypoint",
        }
        unknown = set(value) - allowed
        if unknown:
            raise RecipeError(f"unknown declarative recipe fields: {sorted(unknown)!r}")
        required = {"distribution", "version", "module", "symbol", "kwargs"}
        missing = required - set(value)
        if missing:
            raise RecipeError(f"missing declarative recipe fields: {sorted(missing)!r}")
        strings = {name: value[name] for name in required - {"kwargs"}}
        if any(not isinstance(item, str) or not item for item in strings.values()):
            raise RecipeError("distribution, version, module, and symbol must be non-empty strings")
        module = str(value["module"])
        symbol = str(value["symbol"])
        if not all(part.isidentifier() for part in module.split(".")) or not symbol.isidentifier():
            raise RecipeError("module and symbol must be direct Python identifiers")
        assert_recipe_root_namespace(
            module, distribution=str(value["distribution"]), require_installed=False
        )
        if _is_generic_container_reference(module, symbol):
            raise RecipeError(
                "generic torch.nn containers are refused in declarative R1 recipes: "
                f"{module}.{symbol}"
            )
        kwargs = value["kwargs"]
        if not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
            raise RecipeError("kwargs must be a string-keyed mapping")
        try:
            canonical_json_bytes(kwargs)
        except (TypeError, ValueError) as exc:
            raise RecipeError(
                "kwargs must contain only JSON-compatible declarative values"
            ) from exc
        node_counter = [0]
        for name, child in kwargs.items():
            _walk_construct_values(
                child,
                context=f"kwargs.{name}",
                depth=0,
                counter=node_counter,
                distribution=str(value["distribution"]),
            )
        entrypoint = value.get("entrypoint")
        if entrypoint is not None:
            if not isinstance(entrypoint, str) or not entrypoint.isidentifier():
                raise RecipeError("entrypoint must be a direct method identifier or null")
            if entrypoint.startswith("_"):
                raise RecipeError(f"entrypoint must be a public method, not {entrypoint!r}")
        raw_post_construct = value.get("post_construct", [])
        if not isinstance(raw_post_construct, (list, tuple)):
            raise RecipeError("post_construct must be a list of configuration calls")
        if len(raw_post_construct) > MAX_POST_CONSTRUCT_CALLS:
            raise RecipeError(
                f"post_construct exceeds the maximum of {MAX_POST_CONSTRUCT_CALLS} calls"
            )
        post_construct = tuple(
            PostConstructCall.from_mapping(item, f"post_construct[{index}]")
            for index, item in enumerate(raw_post_construct)
        )
        disable_fields = value.get("pretrained_disable_fields", [])
        if not isinstance(disable_fields, (list, tuple)) or not all(
            isinstance(field, str) and field for field in disable_fields
        ):
            raise RecipeError("pretrained_disable_fields must be a string sequence")
        fields_absent = value.get("pretrained_fields_absent", False)
        if not isinstance(fields_absent, bool):
            raise RecipeError("pretrained_fields_absent must be a boolean")
        if fields_absent and disable_fields:
            raise RecipeError(
                "pretrained_fields_absent contradicts a non-empty pretrained_disable_fields"
            )
        validate_pretrained_disable_fields(kwargs, disable_fields)
        artifact = value.get("artifact_sha256")
        if artifact is not None and not isinstance(artifact, str):
            raise RecipeError("artifact_sha256 must be a string or null")
        return cls(
            distribution=str(value["distribution"]),
            version=str(value["version"]),
            module=module,
            symbol=symbol,
            kwargs=dict(kwargs),
            artifact_sha256=artifact,
            pretrained_disable_fields=tuple(disable_fields),
            pretrained_fields_absent=fields_absent,
            post_construct=post_construct,
            entrypoint=entrypoint,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible recipe mapping.

        The grammar-extension fields are emitted only when they deviate from
        their defaults so that every pre-extension recipe keeps its exact
        historical recipe-revision hash.

        Returns
        -------
        dict[str, Any]
            Declarative recipe payload.
        """

        payload: dict[str, Any] = {
            "distribution": self.distribution,
            "version": self.version,
            "artifact_sha256": self.artifact_sha256,
            "module": self.module,
            "symbol": self.symbol,
            "kwargs": dict(self.kwargs),
            "pretrained_disable_fields": list(self.pretrained_disable_fields),
        }
        if self.pretrained_fields_absent:
            payload["pretrained_fields_absent"] = True
        if self.post_construct:
            payload["post_construct"] = [call.to_dict() for call in self.post_construct]
        if self.entrypoint is not None:
            payload["entrypoint"] = self.entrypoint
        return payload


@dataclass(frozen=True)
class LoadedRecipe:
    """Executable constructor obtained from a safe recipe form.

    Parameters
    ----------
    kind:
        ``declarative-library`` or ``typed-adapter``.
    build_model:
        Zero-argument random-initialized constructor.
    make_dummy_call:
        Typed adapter input builder, absent for declarative R1 recipes.
    recipe_revision:
        Source-bound recipe hash when a source identity was supplied.
    module:
        Loaded typed module, absent for declarative recipes.
    adapter_sha256:
        Digest observed from the one exact adapter byte string executed by the loader.
    entrypoint:
        Declared public non-``forward`` call method for declarative recipes;
        the executor delegates through the crawler-owned transparent adapter.
    """

    kind: str
    build_model: BuildModel
    make_dummy_call: Optional[MakeDummyCall]
    recipe_revision: str
    module: Optional[ModuleType]
    adapter_sha256: Optional[str] = None
    entrypoint: Optional[str] = None


def reject_opaque_recipe(value: Mapping[str, Any]) -> None:
    """Reject every legacy expression, statement, eval, or exec recipe form.

    Parameters
    ----------
    value:
        Candidate recipe mapping.

    Raises
    ------
    RecipeError
        If the mapping contains an opaque executable recipe marker.
    """

    forbidden_keys = {"code", "expression", "statement", "eval", "exec", "callable_string"}
    offending = forbidden_keys.intersection(value)
    recipe_type = str(value.get("type", ""))
    if offending or recipe_type in {"statement", "expression", "exec-string", "eval-string"}:
        details = sorted(offending) or [recipe_type]
        raise RecipeError(f"opaque executable recipes are forbidden: {details!r}")


def _materialize_construct_value(value: Any, context: str, *, distribution: str) -> Any:
    """Resolve construct nodes inside one declarative value at build time.

    Parameters
    ----------
    value:
        Validated JSON-compatible declarative value.
    context:
        Location used in refusal messages.
    distribution:
        Pinned distribution bounding every construct node's callable namespace.

    Returns
    -------
    Any
        The literal value, or the object built by the declared construct graph.

    Raises
    ------
    RecipeError
        If a construct node names a module outside the bounded namespace, cannot
        be resolved to a callable non-container symbol, or builds a generic
        container instance.
    """

    if isinstance(value, Mapping):
        if CONSTRUCT_NODE_KEY in value:
            spec = _validate_construct_node_spec(
                value.get(CONSTRUCT_NODE_KEY), context, distribution=distribution
            )
            constructed_module = importlib.import_module(str(spec["module"]))
            constructed_symbol = getattr(constructed_module, str(spec["symbol"]), None)
            if constructed_symbol is None or not callable(constructed_symbol):
                raise RecipeError(
                    f"construct node at {context}: {spec['module']}.{spec['symbol']} "
                    "is not a callable constructor"
                )
            _reject_container_constructor(constructed_symbol, f"{spec['module']}.{spec['symbol']}")
            materialized = {
                name: _materialize_construct_value(
                    child, f"{context}.{name}", distribution=distribution
                )
                for name, child in spec["kwargs"].items()
            }
            constructed = constructed_symbol(**materialized)
            _reject_container_instance(
                constructed, f"construct node at {context} constructed a generic container"
            )
            return constructed
        return {
            name: _materialize_construct_value(
                child, f"{context}.{name}", distribution=distribution
            )
            for name, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _materialize_construct_value(child, f"{context}[{index}]", distribution=distribution)
            for index, child in enumerate(value)
        ]
    return value


def resolve_input_constructor(spec: Mapping[str, Any], *, distribution: str) -> object:
    """Materialize one declared constructed input leaf without authored code.

    This implements the input-leaf ``distribution: "constructor"`` reservation:
    the value is built by importing a declared module, resolving a declared
    symbol, and calling it with JSON-only (possibly construct-node) kwargs.

    Parameters
    ----------
    spec:
        Exact ``{"module", "symbol", "kwargs"}`` input constructor payload.
    distribution:
        Pinned distribution of the recipe this input is being built for; bounds
        the constructor's callable namespace exactly as it bounds a recipe's
        construct nodes. A constructed input leaf runs the same import-then-call
        machinery, so leaving it unbounded would reopen the hole on the input
        side alone.

    Returns
    -------
    object
        The constructed input value.

    Raises
    ------
    RecipeError
        If the payload deviates from the closed grammar, names a module outside
        the bounded namespace, resolves to or builds a generic container, or
        produces ``None``.
    """

    validated = _validate_construct_node_spec(
        spec, "input_contract constructor", distribution=distribution
    )
    counter = [0]
    for name, child in validated["kwargs"].items():
        _walk_construct_values(
            child,
            context=f"input constructor kwargs.{name}",
            depth=1,
            counter=counter,
            distribution=distribution,
        )
    try:
        canonical_json_bytes(dict(validated["kwargs"]))
    except (TypeError, ValueError) as exc:
        raise RecipeError(
            "input constructor kwargs must contain only JSON-compatible values"
        ) from exc
    constructed_module = importlib.import_module(str(validated["module"]))
    constructed_symbol = getattr(constructed_module, str(validated["symbol"]), None)
    if constructed_symbol is None or not callable(constructed_symbol):
        raise RecipeError(
            f"input constructor {validated['module']}.{validated['symbol']} "
            "is not a callable constructor"
        )
    _reject_container_constructor(
        constructed_symbol, f"{validated['module']}.{validated['symbol']}"
    )
    value = constructed_symbol(
        **{
            name: _materialize_construct_value(
                child, f"input constructor kwargs.{name}", distribution=distribution
            )
            for name, child in validated["kwargs"].items()
        }
    )
    if value is None:
        raise RecipeError(
            f"input constructor {validated['module']}.{validated['symbol']} produced None"
        )
    _reject_container_instance(
        value,
        f"input constructor {validated['module']}.{validated['symbol']} "
        "constructed a generic container",
    )
    return value


def load_declarative_recipe(
    value: Union[DeclarativeRecipe, Mapping[str, Any]], *, source_identity: str = "unbound"
) -> LoadedRecipe:
    """Load a closed R1 recipe without evaluating source text.

    Parameters
    ----------
    value:
        Validated recipe object or mapping.
    source_identity:
        Exact source identity to bind into the recipe revision.

    Returns
    -------
    LoadedRecipe
        Direct import/getattr constructor wrapper.
    """

    if isinstance(value, Mapping):
        reject_opaque_recipe(value)
        recipe = DeclarativeRecipe.from_mapping(value)
    else:
        recipe = value
    assert_recipe_root_namespace(
        recipe.module, distribution=recipe.distribution, require_installed=True
    )
    module = importlib.import_module(recipe.module)
    constructor = getattr(module, recipe.symbol, None)
    if constructor is None or not callable(constructor):
        raise RecipeError(f"{recipe.module}.{recipe.symbol} is not a callable constructor")
    _reject_container_constructor(constructor, f"{recipe.module}.{recipe.symbol}")
    if recipe.pretrained_disable_fields:
        try:
            parameters = inspect.signature(constructor).parameters
        except (TypeError, ValueError) as exc:
            raise RecipeError(
                "cannot verify pretrained disable fields against constructor signature"
            ) from exc
        # A disable field satisfied through the entrypoint's ``**kwargs`` is a
        # REAL disable, not a fabrication: timm entrypoints such as
        # ``dla60(pretrained=False, **kwargs)`` forward ``pretrained_cfg`` to
        # ``build_model_with_cfg``, which honors it at runtime (rung-11 census
        # defect class 3 terminalized 11 proven-runnable timm models on exactly
        # this refusal; m4334 ran verified in rung 9b). The field's VALUE is
        # still enforced by :func:`validate_pretrained_disable_fields`, and the
        # machine floor in :func:`assert_constructor_pretrained_assets_disabled`
        # still reads the real signature -- acceptance here loosens nothing
        # about what may be constructed. A field the constructor can neither
        # name explicitly nor absorb through ``**kwargs`` stays refused: it is
        # genuinely un-forwardable and the declaration is fiction.
        accepts_var_keyword = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        unsupported = [
            field for field in recipe.pretrained_disable_fields if field not in parameters
        ]
        if unsupported and not accepts_var_keyword:
            raise RecipeError(
                "pretrained disable fields are not explicit constructor parameters "
                f"and the constructor accepts no **kwargs to forward them: {unsupported!r}"
            )
    # The declaration above is the author's account of the constructor. This is
    # the machine's, read from the real signature in the routed environment
    # immediately before construction, and it runs whether or not the recipe
    # declared anything: an enabling DEFAULT nobody overrode downloads weights
    # just as surely as an enabling kwarg, and no declaration check can see it.
    assert_constructor_pretrained_assets_disabled(
        constructor, recipe.kwargs, fields_absent=recipe.pretrained_fields_absent
    )

    def build_model() -> object:
        """Invoke the direct library constructor with declarative kwargs.

        Construct-node kwargs are resolved from their declared modules -- bounded
        to the pinned distribution and :data:`CONSTRUCT_MODULE_ALLOWLIST`, and
        refused if they build a generic container -- bounded post-construction
        configuration calls are applied, and the runtime provenance tripwire
        refuses any constructed model whose class is not defined by the pinned
        distribution.

        Returns
        -------
        object
            Constructed random-initialized model.
        """

        materialized = {
            name: _materialize_construct_value(
                child, f"kwargs.{name}", distribution=recipe.distribution
            )
            for name, child in recipe.kwargs.items()
        }
        model = constructor(**materialized)
        for index, call in enumerate(recipe.post_construct):
            method = getattr(model, call.method, None)
            if not callable(method):
                raise RecipeError(
                    f"post_construct[{index}] method {call.method!r} is not callable "
                    "on the constructed model"
                )
            method(*call.args, **dict(call.kwargs))
        assert_model_provenance(model, recipe.distribution)
        return model

    revision = compute_recipe_revision(recipe.to_dict(), source_identity)
    return LoadedRecipe(
        "declarative-library", build_model, None, revision, None, entrypoint=recipe.entrypoint
    )


def _check_adapter_source(source: str, path: Path) -> ast.Module:
    """Statically reject opaque execution and TorchLens imports in an adapter.

    Parameters
    ----------
    source:
        Exact Python adapter source.
    path:
        Source path used in diagnostics.

    Returns
    -------
    ast.Module
        Parsed module syntax tree.
    """

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RecipeError(f"typed adapter has invalid syntax: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "torchlens" or alias.name.startswith("torchlens.")
                for alias in node.names
            ):
                raise RecipeError("typed adapters must not import TorchLens")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torchlens" or str(node.module).startswith("torchlens."):
                raise RecipeError("typed adapters must not import TorchLens")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile"}:
                raise RecipeError(f"typed adapters must not call {node.func.id}()")
    return tree


def _validate_typed_function(
    function: Callable[..., Any], name: str, expected_parameters: tuple[str, ...]
) -> None:
    """Validate one typed adapter entry point.

    Parameters
    ----------
    function:
        Candidate function.
    name:
        Required public symbol name.
    expected_parameters:
        Exact positional parameter names.
    """

    if not inspect.isfunction(function):
        raise RecipeError(f"{name} must be a Python function")
    signature = inspect.signature(function)
    if tuple(signature.parameters) != expected_parameters:
        raise RecipeError(f"{name} must have parameters {expected_parameters!r}")
    annotations = function.__annotations__
    if "return" not in annotations or any(
        parameter not in annotations for parameter in expected_parameters
    ):
        raise RecipeError(f"{name} must carry parameter and return type annotations")


def _module_origins(module: object) -> tuple[str, ...]:
    """Return every filesystem origin a loaded module claims.

    Parameters
    ----------
    module:
        ``sys.modules`` entry.

    Returns
    -------
    tuple[str, ...]
        ``__file__`` plus every ``__path__`` entry; empty when the module
        exposes no determinable filesystem origin.
    """

    candidates: list[str] = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        candidates.append(module_file)
    raw_path = getattr(module, "__path__", None)
    try:
        # ``__path__`` is not always a real list: ``torch.classes`` exposes a
        # non-iterable ``_ClassNamespace`` (observed live on the m8189 replay),
        # and iterating it raises ``TypeError``.
        entries = [] if raw_path is None else list(raw_path)
    except TypeError:
        entries = []
    candidates.extend(entry for entry in entries if isinstance(entry, str))
    return tuple(candidates)


def _origins_inside(origins: Sequence[str], root: Path) -> bool:
    """Return whether EVERY claimed origin sits under the manifest root.

    ALL-inside on purpose: a namespace package spanning the root and a foreign
    ``sys.path`` entry is exactly the split-provenance shape the verification
    exists to refuse, so a single outside entry disqualifies the module.

    Parameters
    ----------
    origins:
        Claimed filesystem origins; an empty sequence is never inside.
    root:
        Resolved staged code-manifest root.
    """

    if not origins:
        return False
    for origin in origins:
        try:
            if not Path(origin).resolve().is_relative_to(root):
                return False
        except (OSError, ValueError):
            return False
    return True


def _manifest_owned_names(
    manifest_root: Path,
    adapter_path: Path,
    member_paths: Optional[Sequence[Path]],
) -> frozenset[str]:
    """Return the top-level import names the staged code manifest OWNS.

    Ownership drives both halves of the collision defense: an owned name
    already bound in ``sys.modules`` is displaced for the import window, and
    an owned name that resolves to bytes outside the manifest root is refused.

    Parameters
    ----------
    manifest_root:
        Staged code-manifest import root.
    adapter_path:
        Exact staged adapter file.
    member_paths:
        Declared staged manifest member paths. ``None`` falls back to the
        root's top-level directory listing -- a convenience for direct
        (test/tooling) callers; the production ``load_recipe`` path always
        supplies the declared members, so ownership follows the manifest, not
        whatever sits on disk.
    """

    try:
        root = manifest_root.resolve()
    except OSError:
        root = manifest_root
    candidates: list[Path] = [adapter_path]
    if member_paths is not None:
        candidates.extend(member_paths)
    else:
        try:
            candidates.extend(
                entry
                for entry in manifest_root.iterdir()
                if entry.is_dir() or entry.suffix == ".py"
            )
        except OSError:
            pass
    names: set[str] = set()
    for candidate in candidates:
        candidate_path = Path(candidate)
        try:
            relative = candidate_path.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        if not relative.parts:
            continue
        if len(relative.parts) > 1:
            names.add(relative.parts[0])
        elif relative.suffix == ".py":
            names.add(relative.stem)
        elif candidate_path.is_dir():
            # A top-level package directory (``vendored/``, ``recbole/``) owns
            # its import name even when no member below it was enumerated.
            names.add(relative.parts[0])
    return frozenset(names)


#: Process-wide serialization of the displace/import/collect/restore
#: transaction. The window mutates global interpreter state (``sys.modules``,
#: ``sys.meta_path``), so two windows interleaving corrupt each other's
#: displacement bookkeeping (Sol wave-8 R2 finding 4: A-enter/B-enter/A-exit/
#: B-exit left A's manifest module in global ``sys.modules``).
_MANIFEST_WINDOW_LOCK = threading.Lock()

#: The context currently holding :data:`_MANIFEST_WINDOW_LOCK`, if any.
_ACTIVE_MANIFEST_WINDOW: Optional["_ManifestImportContext"] = None


class _VerifiedMemberLoader(importlib.machinery.SourceFileLoader):
    """Load exactly the digest-verified staged bytes of one declared member.

    Delegation, not opaque execution: the stdlib source loader owns
    compilation and module exec (the worker-source static boundary bans
    direct ``compile``/``exec`` calls in execution sources, and that tripwire
    is correct); this subclass only pins WHICH BYTES the stdlib may see.
    ``get_data`` serves the declared member path alone, re-hashing the bytes
    against the declared digest on every read -- the loader then compiles
    exactly the returned object, so there is no gap between verification and
    execution. Every other path (bytecode-cache probes included) is refused,
    and ``set_data`` never writes a cache into staged custody. Forgeable
    post-hoc metadata (``__file__``, spec origin) plays no part in what
    executes.
    """

    def __init__(self, fullname: str, member_path: Path, expected_sha256: Optional[str]) -> None:
        """Bind one module name to its declared staged path and digest.

        Parameters
        ----------
        fullname:
            Dotted module name this loader serves.
        member_path:
            Exact declared staged member file.
        expected_sha256:
            Declared canonical digest; ``None`` only for direct (test/tooling)
            callers that supplied no declaration, where the verified path
            itself is the authority.
        """

        super().__init__(fullname, str(member_path))
        self._expected = expected_sha256

    def get_data(self, path: str) -> bytes:
        """Return the declared member's re-verified bytes; refuse other paths.

        Raises
        ------
        OSError
            For any path other than the declared member -- notably the
            bytecode-cache probe, which makes the stdlib fall through to the
            source path instead of executing an unverified cache.
        RecipeError
            If the declared member is unreadable or its bytes no longer
            reproduce the declared digest.
        """

        if str(path) != self.path:
            raise OSError(f"only the declared manifest member may be read, not {path!r}")
        try:
            data = Path(self.path).read_bytes()
        except OSError as exc:
            raise RecipeError(
                f"staged manifest member for module {self.name!r} is unreadable: {exc}"
            ) from exc
        observed = hash_bytes(data)
        if self._expected is not None and observed != self._expected:
            raise RecipeError(
                f"staged manifest member for module {self.name!r} drifted from "
                f"its declared digest: declared {self._expected}, observed {observed}"
            )
        return data

    def set_data(self, path: object, data: object, *, _mode: int = 0o666) -> None:
        """Never write bytecode caches into staged manifest custody."""

        del path, data, _mode


class _ManifestMemberFinder(importlib.abc.MetaPathFinder):
    """``sys.meta_path`` authority for manifest-owned names during a window."""

    def __init__(self, context: "_ManifestImportContext") -> None:
        """Bind the finder to its owning import window."""

        self._context = context

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        """Resolve manifest-owned names from the declared members only."""

        del path, target
        return self._context._find_owned_spec(fullname)


def _manifest_module_registry(
    resolved_root: Path, members: Sequence[tuple[Path, Optional[str]]]
) -> tuple[dict[str, tuple[Path, Optional[str], bool]], dict[str, Path]]:
    """Map declared manifest members to the module names they may satisfy.

    Parameters
    ----------
    resolved_root:
        Resolved staged code-manifest root.
    members:
        Declared ``(staged path, declared digest)`` pairs.

    Returns
    -------
    tuple[dict, dict]
        ``fullname -> (path, digest, is_package)`` for loadable members, and
        ``fullname -> directory`` namespace-package fallbacks for ancestor
        packages that declare no ``__init__.py`` member.
    """

    modules: dict[str, tuple[Path, Optional[str], bool]] = {}
    namespaces: dict[str, Path] = {}
    for member_path, digest in members:
        try:
            relative = Path(member_path).resolve().relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        parts = relative.parts
        if not parts:
            continue
        for index in range(1, len(parts)):
            name = ".".join(parts[:index])
            namespaces.setdefault(name, resolved_root.joinpath(*parts[:index]))
        if relative.suffix != ".py":
            continue
        if relative.name == "__init__.py":
            if len(parts) >= 2:
                modules[".".join(parts[:-1])] = (Path(member_path), digest, True)
        else:
            modules[".".join([*parts[:-1], relative.stem])] = (
                Path(member_path),
                digest,
                False,
            )
    for name in modules:
        namespaces.pop(name, None)
    return modules, namespaces


class _ManifestImportContext:
    """One adapter's verified import window over its digest-verified closure.

    The defects closed here (Sol wave-8 findings 3 and R2-4, plus the lazy
    functionality edge) are all the same class -- code/receipt substitution
    through ambient interpreter state -- so the window follows one principle:
    CONTROL resolution, never infer it.

    * **Displacement**: every manifest-OWNED top-level name is displaced out
      of ``sys.modules`` for the window and restored after it, so a
      worker-startup module with a generic name never silently satisfies an
      adapter import.
    * **Owned resolution authority**: a manifest-owned finder sits FIRST on
      ``sys.meta_path`` for the window and binds each declared member name to
      its digest-verified staged path; its loader re-hashes and executes
      exactly those bytes. An owned name with no declared member behind it is
      a typed :class:`RecipeError` -- the finder never defers an owned name
      to ambient import machinery, so a foreign ``sys.meta_path`` hook that
      forges ``__file__``/spec origin (Sol R2 finding 4) is simply never
      consulted. The post-window origin sweep remains as a redundant assert,
      not the authority.
    * **Serialization**: the whole displace/import/collect/restore
      transaction holds a process-wide lock. Reentry by the SAME context
      (lazy ``build_model``/``make_dummy_call`` windows) nests freely; a
      DIFFERENT context opening on the same thread mid-window is a typed
      refusal, and another thread blocks until the window closes.

    Outside every window the manifest's modules live ONLY in this context's
    private cache, so a later adapter in the same process can never resolve a
    same-named sibling from a previous model's closure.
    """

    def __init__(
        self,
        manifest_root: Path,
        owned_names: frozenset[str],
        adapter_module_name: str,
        members: Sequence[tuple[Path, Optional[str]]],
    ) -> None:
        """Bind the window to one root, ownership set, and declared members.

        Parameters
        ----------
        manifest_root:
            Staged code-manifest import root.
        owned_names:
            Top-level import names the manifest owns.
        adapter_module_name:
            Synthetic unique adapter module name; exempt from collection.
        members:
            Declared ``(staged path, declared digest)`` pairs backing the
            owned names.
        """

        self._root_text = str(manifest_root)
        try:
            self._resolved_root = manifest_root.resolve()
        except OSError:
            self._resolved_root = manifest_root
        self._owned = owned_names
        self._adapter_module_name = adapter_module_name
        self._members, self._namespaces = _manifest_module_registry(self._resolved_root, members)
        self._finder = _ManifestMemberFinder(self)
        self._cache: dict[str, ModuleType] = {}
        self._displaced: dict[str, ModuleType] = {}
        self._preexisting: frozenset[str] = frozenset()
        self._depth = 0
        self._owner_thread: Optional[int] = None

    def _is_owned_key(self, name: str) -> bool:
        """Return whether a ``sys.modules`` key falls under an owned name."""

        return name.split(".")[0] in self._owned

    def _find_owned_spec(self, fullname: str) -> Optional[importlib.machinery.ModuleSpec]:
        """Resolve one import request against the declared manifest members.

        Parameters
        ----------
        fullname:
            Dotted module name the import system is resolving.

        Returns
        -------
        importlib.machinery.ModuleSpec | None
            A spec bound to a verified member loader or namespace directory,
            or ``None`` for names the manifest does not own.

        Raises
        ------
        RecipeError
            If the manifest owns the name but declares no member behind it:
            deferring to ambient machinery would hand an owned name to
            whatever finder or path entry answers first.
        """

        if fullname == self._adapter_module_name:
            return None
        if not self._is_owned_key(fullname):
            return None
        entry = self._members.get(fullname)
        if entry is not None:
            member_path, digest, is_package = entry
            loader = _VerifiedMemberLoader(fullname, member_path, digest)
            spec = importlib.machinery.ModuleSpec(fullname, loader, origin=str(member_path))
            spec.has_location = True
            if is_package:
                spec.submodule_search_locations = [str(member_path.parent)]
            return spec
        namespace_dir = self._namespaces.get(fullname)
        if namespace_dir is not None:
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = [str(namespace_dir)]
            return spec
        raise RecipeError(
            f"manifest-owned module {fullname!r} is not a declared member of the "
            "staged code manifest"
        )

    def __enter__(self) -> "_ManifestImportContext":
        """Open the window under the process-wide transaction lock."""

        global _ACTIVE_MANIFEST_WINDOW
        if self._depth == 0:
            holder = _ACTIVE_MANIFEST_WINDOW
            if (
                holder is not None
                and holder is not self
                and holder._owner_thread == threading.get_ident()
            ):
                # Blocking on the lock here would deadlock this thread against
                # itself; interleaved windows are a coherence violation either
                # way (Sol R2 finding 4's A/B probe), so refuse typed.
                raise RecipeError(
                    "another adapter's manifest import window is already open on "
                    "this thread; adapter import windows do not interleave"
                )
            _MANIFEST_WINDOW_LOCK.acquire()
            _ACTIVE_MANIFEST_WINDOW = self
            self._owner_thread = threading.get_ident()
            try:
                self._displaced = {
                    name: sys.modules.pop(name)
                    for name in [key for key in sys.modules if self._is_owned_key(key)]
                }
                sys.modules.update(self._cache)
                self._preexisting = frozenset(sys.modules)
                sys.meta_path.insert(0, self._finder)
            except BaseException:
                _ACTIVE_MANIFEST_WINDOW = None
                self._owner_thread = None
                _MANIFEST_WINDOW_LOCK.release()
                raise
        self._depth += 1
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the window: collect, verify, restore, unlock, fail closed."""

        global _ACTIVE_MANIFEST_WINDOW
        self._depth -= 1
        if self._depth:
            return
        violation: Optional[RecipeError] = None
        try:
            try:
                sys.meta_path.remove(self._finder)
            except ValueError:
                pass
            violation = self._collect_window_modules()
            for name, module in self._displaced.items():
                sys.modules.setdefault(name, module)
            self._displaced = {}
        finally:
            _ACTIVE_MANIFEST_WINDOW = None
            self._owner_thread = None
            _MANIFEST_WINDOW_LOCK.release()
        if violation is not None and exc_type is None:
            raise violation

    def _collect_window_modules(self) -> Optional[RecipeError]:
        """Move manifest modules to the private cache; verify owned bindings.

        Returns
        -------
        RecipeError | None
            The first owned-name provenance violation, after every offending
            binding has been removed from ``sys.modules``.
        """

        violation: Optional[RecipeError] = None
        for name in [
            key
            for key in sys.modules
            # New bindings, every owned name, and every cache reinstallation:
            # a cached non-owned module reinstalled at enter is preexisting by
            # snapshot, so without the cache clause it would leak past exit.
            if key not in self._preexisting or self._is_owned_key(key) or key in self._cache
        ]:
            if name == self._adapter_module_name:
                continue
            module = sys.modules.get(name)
            if module is None:
                continue
            origins = _module_origins(module)
            inside = _origins_inside(origins, self._resolved_root)
            if self._is_owned_key(name):
                sys.modules.pop(name, None)
                if inside:
                    self._cache[name] = module
                elif violation is None:
                    described = list(origins) if origins else "no determinable origin"
                    violation = RecipeError(
                        f"manifest-owned module {name!r} resolved outside the staged "
                        f"code manifest root {self._root_text!r}: {described}"
                    )
            elif inside:
                self._cache[name] = module
                sys.modules.pop(name, None)
        return violation


def _bind_adapter_import_context(
    function: Callable[..., Any], context: "_ManifestImportContext"
) -> Callable[..., Any]:
    """Wrap one adapter entry point so lazy imports resolve its manifest.

    Parameters
    ----------
    function:
        Validated adapter ``build_model`` or ``make_dummy_call``.
    context:
        The adapter's reusable manifest import window.

    Returns
    -------
    Callable[..., Any]
        Signature-preserving wrapper executing inside the window.
    """

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with context:
            return function(*args, **kwargs)

    return wrapped


def _manifest_import_root(adapter_path: Path, manifest: Sequence[Any]) -> Path:
    """Return the one directory that makes a staged code manifest importable.

    Parameters
    ----------
    adapter_path:
        Exact staged adapter file.
    manifest:
        Staged ``code_manifest`` member rows carrying ``path`` fields.

    Returns
    -------
    pathlib.Path
        Deepest common parent of the adapter and every manifest member --
        the adapter's own directory for flat closures (``pit.py`` beside
        ``pit_adapter.py``) and the custody root for package trees
        (``recbole/`` beside the adapter).
    """

    directories = [adapter_path.parent]
    for member in manifest:
        if not isinstance(member, Mapping):
            continue
        member_path = member.get("path")
        if isinstance(member_path, str) and member_path:
            directories.append(Path(member_path).parent)
    try:
        return Path(os.path.commonpath([str(directory) for directory in directories]))
    except ValueError:
        # Mixed absolute/relative rows cannot share a common path; the
        # adapter's own directory is the only root that is always coherent.
        return adapter_path.parent


def load_typed_adapter(
    path: Path,
    *,
    source_identity: str = "unbound",
    expected_recipe_revision: Optional[str] = None,
    expected_adapter_sha256: Optional[str] = None,
    manifest_root: Optional[Path] = None,
    manifest_member_paths: Optional[Sequence[Path]] = None,
    manifest_member_digests: Optional[Mapping[str, str]] = None,
) -> LoadedRecipe:
    """Statically check and import a typed R2--R4 adapter module.

    Parameters
    ----------
    path:
        Exact adapter Python file.
    source_identity:
        Exact source identity to bind into the recipe revision.
    expected_recipe_revision:
        Parent-authorized revision that the observed bytes must reproduce before import.
    expected_adapter_sha256:
        Optional direct digest of the parent-authorized adapter bytes.
    manifest_root:
        Staged code-manifest directory made importable through the adapter's
        :class:`_ManifestImportContext` window -- during module exec AND
        around every later ``build_model``/``make_dummy_call`` invocation, so
        lazy in-function imports resolve too. Manifest-owned names already in
        ``sys.modules`` are displaced for the window and restored after it
        (never silently shadowed), and owned resolutions are served ONLY by
        the window's own ``sys.meta_path`` finder, which loads the declared
        members' digest-verified bytes. Without the root the bound closure
        died ``ModuleNotFoundError`` at import (rung-11 census, R2 lane
        0-for-5). This widens no read authorization and no hash policy: the
        worker's execution policy still refuses reads outside the allowlisted
        manifest member paths.
    manifest_member_paths:
        Declared staged manifest member paths; they define exactly which
        top-level import names the manifest OWNS and which module names its
        finder may satisfy. When absent, the members fall back to the root's
        recursive listing (direct callers only; the production path always
        supplies the declaration).
    manifest_member_digests:
        Declared canonical member digests keyed by the exact string of each
        member path, re-verified by the window's loader immediately before
        each member executes.

    Returns
    -------
    LoadedRecipe
        Typed constructor and dummy-call functions.
    """

    source_bytes = path.read_bytes()
    observed_adapter_sha256 = hash_bytes(source_bytes)
    revision = compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": path.name},
        source_identity,
        adapter_bytes=source_bytes,
    )
    if expected_adapter_sha256 is not None and observed_adapter_sha256 != expected_adapter_sha256:
        raise RecipeError(
            "typed adapter digest mismatch: "
            f"expected {expected_adapter_sha256}, observed {observed_adapter_sha256}"
        )
    if expected_recipe_revision is not None and revision != expected_recipe_revision:
        raise RecipeError(
            "typed adapter recipe revision mismatch: "
            f"expected {expected_recipe_revision}, observed {revision}"
        )
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecipeError("typed adapter must be UTF-8 Python source") from exc
    _check_adapter_source(source, path)
    module_name = f"menagerie_crawler_adapter_{observed_adapter_sha256[7:23]}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    # ``FunctionType(code, module.__dict__)()`` never inserts ``__builtins__``
    # into the globals dict (the function object caches its builtins instead),
    # so any C-level lazy import issued FROM an adapter function frame --
    # ``datetime.timetuple()`` importing ``time`` is the live m5445 case --
    # dies ``KeyError: '__builtins__'``. Real module dicts always carry it.
    module.__dict__.setdefault("__builtins__", builtins)
    sys.modules[module_name] = module
    import_context: Optional[_ManifestImportContext] = None
    if manifest_root is not None:
        digests = dict(manifest_member_digests or {})
        if manifest_member_paths is not None:
            members: list[tuple[Path, Optional[str]]] = [
                (Path(member), digests.get(str(member))) for member in manifest_member_paths
            ]
        else:
            # Direct-caller convenience: no declaration, so the members are
            # the root's own recursive source listing and the verified path
            # is the authority (production always declares).
            try:
                members = [(member, None) for member in sorted(manifest_root.rglob("*.py"))]
            except OSError:
                members = []
        import_context = _ManifestImportContext(
            manifest_root,
            _manifest_owned_names(manifest_root, path, manifest_member_paths),
            module_name,
            members,
        )

    def _execute_adapter_module() -> None:
        """Compile and execute the adapter bytes into the synthetic module."""

        loader = importlib.machinery.SourceFileLoader(module_name, str(path))
        code = loader.source_to_code(source_bytes, str(path))
        FunctionType(code, module.__dict__)()

    try:
        if import_context is not None:
            with import_context:
                _execute_adapter_module()
        else:
            _execute_adapter_module()
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    build_model = getattr(module, "build_model", None)
    make_dummy_call = getattr(module, "make_dummy_call", None)
    if not callable(build_model) or not callable(make_dummy_call):
        raise RecipeError("typed adapter must define build_model and make_dummy_call")
    _validate_typed_function(build_model, "build_model", ())
    _validate_typed_function(make_dummy_call, "make_dummy_call", ("seed", "device"))
    if import_context is not None:
        # Lazy imports inside the adapter's entry points must resolve manifest
        # siblings too (Sol wave-8 functionality edge): re-enter the SAME
        # window around every call, with cached manifest modules reinstalled
        # so module-level state persists and nothing re-executes.
        build_model = _bind_adapter_import_context(build_model, import_context)
        make_dummy_call = _bind_adapter_import_context(make_dummy_call, import_context)
    return LoadedRecipe(
        "typed-adapter",
        build_model,
        make_dummy_call,
        revision,
        module,
        observed_adapter_sha256,
    )


def load_recipe(
    value: Mapping[str, Any],
    *,
    source_identity: str = "unbound",
    expected_recipe_revision: Optional[str] = None,
) -> LoadedRecipe:
    """Load one explicitly tagged closed recipe form.

    Parameters
    ----------
    value:
        Mapping with kind ``declarative-library`` or ``typed-adapter``.
    source_identity:
        Source identity bound into the recipe revision.
    expected_recipe_revision:
        Parent-authorized revision required before a typed adapter is imported.

    Returns
    -------
    LoadedRecipe
        Executable typed recipe contract.
    """

    reject_opaque_recipe(value)
    kind = value.get("kind") or value.get("recipe_type")
    if kind == "declarative-library":
        payload = value.get("recipe", value)
        if not isinstance(payload, Mapping):
            raise RecipeError("declarative recipe payload must be an object")
        if payload is value:
            payload = {
                key: item for key, item in value.items() if key not in {"kind", "recipe_type"}
            }
        loaded = load_declarative_recipe(payload, source_identity=source_identity)
        if (
            expected_recipe_revision is not None
            and loaded.recipe_revision != expected_recipe_revision
        ):
            raise RecipeError(
                "declarative recipe revision mismatch: "
                f"expected {expected_recipe_revision}, observed {loaded.recipe_revision}"
            )
        return loaded
    if kind == "typed-adapter":
        path_value = value.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise RecipeError("typed-adapter recipe requires a path")
        digest_value = value.get("adapter_sha256")
        if digest_value is not None and not isinstance(digest_value, str):
            raise RecipeError("typed-adapter adapter_sha256 must be a string when supplied")
        # The mapping-level pin is the ADAPTER-flavor revision the parent bound
        # at admission (``gates.revalidate_recipe_revision_pin``) -- the exact
        # value :func:`load_typed_adapter` recomputes from the staged bytes. The
        # ``expected_recipe_revision`` PARAMETER historically received the
        # accepted-FACTS revision, a structurally different hash that mismatched
        # on every typed adapter and pushed the worker into a permissive retry
        # whose real failure masked the refusal (rung-11 census, defect class
        # 4). A pin riding in the mapping cannot be dropped by a caller passing
        # ``expected_recipe_revision=None``, so the loader check stays armed.
        pinned_revision = value.get("recipe_revision")
        if pinned_revision is not None and not isinstance(pinned_revision, str):
            raise RecipeError("typed-adapter recipe_revision must be a string when supplied")
        manifest_value = value.get("code_manifest")
        manifest_root: Optional[Path] = None
        manifest_member_paths: Optional[list[Path]] = None
        manifest_member_digests: Optional[dict[str, str]] = None
        if isinstance(manifest_value, list) and manifest_value:
            manifest_root = _manifest_import_root(Path(path_value), manifest_value)
            manifest_member_paths = []
            manifest_member_digests = {}
            for member in manifest_value:
                if not isinstance(member, Mapping):
                    continue
                member_path = member.get("path")
                if not isinstance(member_path, str) or not member_path:
                    continue
                manifest_member_paths.append(Path(member_path))
                member_digest = member.get("sha256")
                if isinstance(member_digest, str) and member_digest:
                    manifest_member_digests[str(Path(member_path))] = member_digest
        return load_typed_adapter(
            Path(path_value),
            source_identity=source_identity,
            expected_recipe_revision=(
                pinned_revision if pinned_revision is not None else expected_recipe_revision
            ),
            expected_adapter_sha256=digest_value,
            manifest_root=manifest_root,
            manifest_member_paths=manifest_member_paths,
            manifest_member_digests=manifest_member_digests,
        )
    raise RecipeError(f"unsupported recipe kind: {kind!r}")
