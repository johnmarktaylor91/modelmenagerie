"""C-21 (torch-lane arm) -- a framework dispatch may not fall through to torch.

``FAILURE_PATTERN_CATALOG`` C-21: machinery written for one backend silently mis-serves
the others and destroys completed work at the last gate. The crawler had SIX independent
closed framework maps with six different key sets and three different failure modes --
silent no-op, typed raise, and wrong default (AREA_BACKENDS_opus B1's table). The wrong
default is the dangerous one: ``framework_from_markers`` returned ``"pytorch"`` for
everything it did not recognise, so an unknown framework was not refused, it was
*asserted to be torch*.

The remedy is structural: ONE registry
(:data:`menagerie.crawler.backend.FRAMEWORK_CAPABILITIES`) that is TOTAL over the closed
:class:`~menagerie.crawler.backend.Framework` enum, with every capability field required,
so a framework cannot be half-added. This module is the census that keeps it total:

* every ``Framework`` member has a capability row, and every row field is populated;
* every alias resolves, round-trips, and no alias collides across frameworks;
* every dispatch surface (adapter construction, env routing, input conversion, dtype
  vocabulary) is exhaustive over the enum -- a missing arm fails here rather than
  defaulting to torch;
* a framework with no environment lane is REFUSED with a typed disposition instead of
  being routed into the torch lane (B6's ~120 no-lane roster rows).
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields

import pytest

from menagerie.crawler.backend import (
    FRAMEWORK_ALIASES,
    FRAMEWORK_CAPABILITIES,
    ORIGIN_FRAMEWORK_ALIASES,
    RUN_FRAMEWORKS,
    Framework,
    FrameworkVocabularyError,
    LaneStatus,
    OriginFramework,
    framework_capability,
    normalize_framework,
    normalize_origin_framework,
)
import menagerie.crawler.routing as routing_module
from menagerie.crawler.constants import BACKEND_PARITY_DEFERRAL_STATUS_CODE, EnvironmentPhase
from menagerie.crawler.frameworks import adapt_native_model
from menagerie.crawler.routing import ModelRequirements, RoutingError, route_model


@pytest.mark.smoke
def test_every_framework_has_a_complete_capability_row() -> None:
    """A framework cannot be half-added: every member, every field."""

    assert set(FRAMEWORK_CAPABILITIES) == set(Framework)
    for framework in Framework:
        capability = FRAMEWORK_CAPABILITIES[framework]
        assert capability.framework is framework
        # The lane fields are conditional by construction -- an executable framework has an
        # intent and no refusal reason, a no-lane framework has the reverse -- so they are
        # asserted as a pair in `test_no_lane_frameworks_are_typed_not_defaulted_to_torch`
        # rather than for mere presence here. Every OTHER field is unconditional.
        conditional = {"env_intent", "torchlens_backend", "no_lane_reason", "no_lane_disposition"}
        for field in dataclass_fields(capability):
            if field.name in conditional:
                continue
            value = getattr(capability, field.name)
            assert value is not None and value != "" and value != (), (
                f"{framework.value} capability field {field.name} is unpopulated"
            )
        assert set(conditional) < {field.name for field in dataclass_fields(capability)}


@pytest.mark.smoke
def test_no_lane_frameworks_are_typed_not_defaulted_to_torch() -> None:
    """The absence of a lane is a typed disposition, never a torch route."""

    for framework in Framework:
        capability = FRAMEWORK_CAPABILITIES[framework]
        if capability.lane_status is LaneStatus.EXECUTABLE:
            assert capability.env_intent, f"{framework.value} claims a lane with no intent"
            assert capability.no_lane_reason == ""
            assert capability.no_lane_disposition is None
            continue
        assert capability.env_intent is None
        assert capability.no_lane_disposition == BACKEND_PARITY_DEFERRAL_STATUS_CODE
        assert capability.no_lane_reason, f"{framework.value} has no typed no-lane reason"


@pytest.mark.smoke
def test_aliases_are_closed_total_and_collision_free() -> None:
    """Every accepted spelling resolves to exactly one canonical framework."""

    for alias, framework in FRAMEWORK_ALIASES.items():
        assert alias == alias.strip().lower(), f"alias {alias!r} is not normalized"
        assert normalize_framework(alias) is framework
        assert normalize_framework(alias.upper()) is framework
        assert normalize_framework(f"  {alias} ") is framework
    # Canonical spellings are themselves accepted aliases, so a record round-trips.
    for framework in Framework:
        assert normalize_framework(framework.value) is framework
        for alias in FRAMEWORK_CAPABILITIES[framework].aliases:
            assert FRAMEWORK_ALIASES[alias] is framework
    for alias, origin in ORIGIN_FRAMEWORK_ALIASES.items():
        assert normalize_origin_framework(alias) is origin
    # Every run framework is also a legal origin framework; the reverse is not true.
    for framework in RUN_FRAMEWORKS:
        assert normalize_origin_framework(framework.value) is OriginFramework(framework.value)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "prose",
    [
        "torch 2.13",
        "pytorch (cpu)",
        "PyTorch/2.x",
        "the model runs in pytorch",
        "tf1 session code",
        "",
        "   ",
    ],
)
def test_free_text_is_refused_not_silently_defaulted(prose: str) -> None:
    """B1's live drift samples are refused typed, never seeded as torch."""

    with pytest.raises(FrameworkVocabularyError):
        normalize_framework(prose)


@pytest.mark.smoke
def test_adapter_dispatch_is_exhaustive_over_the_enum() -> None:
    """Adapter construction has an arm for every framework, or refuses typed."""

    class _Native:
        def forward(self, *args: object, **kwargs: object) -> object:
            return None

        def __call__(self, *args: object, **kwargs: object) -> object:
            return None

        def apply(self, *args: object, **kwargs: object) -> object:
            return None

    for framework in RUN_FRAMEWORKS:
        capability = framework_capability(framework)
        adapter = adapt_native_model(_Native(), framework.value)
        assert adapter.metadata.native_call_method == capability.adapter_call_method
        assert adapter.metadata.run_framework == framework.value
    with pytest.raises(FrameworkVocabularyError):
        adapt_native_model(_Native(), "keras-3-functional")


@pytest.mark.smoke
def test_env_routing_is_exhaustive_over_the_enum() -> None:
    """Every executable framework routes; no framework silently lands in the torch lane."""

    for framework in RUN_FRAMEWORKS:
        capability = framework_capability(framework)
        requirements = ModelRequirements(stable_id="m1", framework=framework.value)
        if capability.lane_status is not LaneStatus.EXECUTABLE:
            with pytest.raises(RoutingError) as caught:
                route_model(requirements)
            assert BACKEND_PARITY_DEFERRAL_STATUS_CODE in str(caught.value)
            continue
        route = route_model(requirements)
        assert route.phase is capability.lane_phase
        if framework is Framework.TORCH:
            assert route.phase is EnvironmentPhase.PYTORCH
        else:
            assert route.intent == capability.env_intent


@pytest.mark.smoke
def test_the_intent_framework_map_agrees_with_the_environment_registry() -> None:
    """The duplicated intent->framework map may not drift from registry.yml.

    ``backend.INTENT_FRAMEWORKS`` is what lets the pipeline ask "does this lane serve the
    framework the author declared" -- the comparison nothing made. It mirrors
    ``envs/registry.yml`` because backend.py must stay import-light, so the mirror needs a
    tripwire or it becomes a seventh drifting map.
    """

    import json
    from pathlib import Path

    from menagerie.crawler.backend import INTENT_FRAMEWORKS

    registry = json.loads(
        (Path(routing_module.__file__).parent / "envs" / "registry.yml").read_text(encoding="utf-8")
    )
    declared = {name: intent["framework"] for name, intent in registry["intents"].items()}
    assert set(declared) == set(INTENT_FRAMEWORKS), (
        "the intent inventory drifted from envs/registry.yml: "
        f"{sorted(set(declared) ^ set(INTENT_FRAMEWORKS))}"
    )
    for intent, framework_value in declared.items():
        assert INTENT_FRAMEWORKS[intent] is Framework(framework_value)
    # And the canonical spellings mirrored into import-light constants must be identical.
    from menagerie.crawler.constants import CANONICAL_FRAMEWORK_VALUES

    assert set(CANONICAL_FRAMEWORK_VALUES) == {framework.value for framework in Framework}


@pytest.mark.smoke
def test_the_marker_prior_never_asserts_a_framework_fact() -> None:
    """The roster-name guess is a PRIOR; only a source/author basis is an authority."""

    from menagerie.crawler.backend import FrameworkBasis, FrameworkFact

    prior = FrameworkFact(
        framework=Framework.TORCH,
        basis=FrameworkBasis.MARKER,
        evidence="zoo+name markers",
    )
    assert not prior.is_execution_authority
    probed = FrameworkFact(
        framework=Framework.TORCH,
        basis=FrameworkBasis.SOURCE_PROBE,
        evidence="import torch in src/model.py",
    )
    assert probed.is_execution_authority
    # Evidence is mandatory: a basis claim with nothing behind it is not a fact.
    with pytest.raises(ValueError):
        FrameworkFact(framework=Framework.TORCH, basis=FrameworkBasis.SOURCE_PROBE, evidence="  ")
