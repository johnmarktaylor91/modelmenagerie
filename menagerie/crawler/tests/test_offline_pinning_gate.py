"""OFFLINE-PINNING gate: known phone-home libraries must have their switch set.

m10759 (ultralytics_yolov8n_det, step-100 triage row 9): the workload attempted
DNS resolution inside the sandbox (mDNSResponder, sandbox-denied) because
ultralytics' online check was never disabled -- ``is_online()`` probes public
DNS unless its documented ``YOLO_OFFLINE`` switch is set. The safe execution
environment already pinned HF hub, wandb, and comet offline, but each switch
was an inline literal: the ultralytics gap was invisible until a model burned a
``policy / network-attempt`` terminal on it.

The closed list now lives in ONE constant
(:data:`menagerie.crawler.constants.PHONE_HOME_OFFLINE_SWITCHES`);
``build_safe_environment`` consumes it verbatim, and proposal validation
refuses a recipe naming a listed distribution whose documented switch the
execution environment does not guarantee -- naming the switch -- so a future
list entry whose switch is not yet wired refuses at authoring instead of
burning attempts at forward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler import proposal as proposal_module
from menagerie.crawler.constants import PHONE_HOME_OFFLINE_SWITCHES
from menagerie.crawler.policy import EXECUTION_OFFLINE_ENV, build_safe_environment
from menagerie.crawler.proposal import (
    ProposalValidationError,
    _validate_offline_pinning,
    validate_author_proposal,
)
from menagerie.crawler.tests.conftest import make_author_proposal

pytestmark = pytest.mark.smoke


def _facts(distribution: str) -> dict[str, Any]:
    """Return minimal proposed facts naming ``distribution`` in the recipe."""

    return {
        "implementation": {
            "recipe_type": "declarative-library",
            "library_recipe": {
                "distribution": distribution,
                "version": "8.4.117",
                "module": "ultralytics.nn.tasks",
                "symbol": "DetectionModel",
                "kwargs": {},
                "pretrained_disable_fields": [],
            },
        }
    }


def test_the_closed_list_starts_with_the_step100_phone_home_libraries() -> None:
    """The ONE constant names the four libraries the triage called out."""

    assert {"ultralytics", "wandb", "huggingface-hub", "comet-ml"} <= set(
        PHONE_HOME_OFFLINE_SWITCHES
    )
    assert PHONE_HOME_OFFLINE_SWITCHES["ultralytics"]["YOLO_OFFLINE"] == "true"


def test_the_execution_environment_guarantees_every_documented_switch(
    tmp_path: Path,
) -> None:
    """The m10759 fix: every closed-list switch reaches the sandboxed child env."""

    safe = build_safe_environment(tmp_path / "scratch")
    for switches in PHONE_HOME_OFFLINE_SWITCHES.values():
        for name, value in switches.items():
            assert safe.get(name) == value
    assert safe["YOLO_OFFLINE"] == "true"


def test_a_guaranteed_phone_home_recipe_passes_the_gate() -> None:
    """ultralytics/wandb/hf-hub/comet recipes pass because the env discharges them."""

    for distribution in ("ultralytics", "wandb", "huggingface_hub", "comet-ml"):
        _validate_offline_pinning(_facts(distribution))


def test_an_unguaranteed_switch_is_refused_naming_the_switch() -> None:
    """A listed library whose documented switch is not set refuses, naming it."""

    with pytest.raises(ProposalValidationError) as caught:
        _validate_offline_pinning(
            _facts("ultralytics"), execution_environment={"HF_HUB_OFFLINE": "1"}
        )
    message = str(caught.value)
    assert "YOLO_OFFLINE" in message
    assert "ultralytics" in message


def test_a_wrong_switch_value_is_refused_too() -> None:
    """The documented value is part of the contract: WANDB_MODE=online is not offline."""

    environment = {name: value for name, value in EXECUTION_OFFLINE_ENV.items()}
    environment["WANDB_MODE"] = "online"
    with pytest.raises(ProposalValidationError, match="WANDB_MODE"):
        _validate_offline_pinning(_facts("wandb"), execution_environment=environment)


def test_a_distribution_off_the_closed_list_passes() -> None:
    """The list is closed: unlisted distributions are not this gate's business."""

    _validate_offline_pinning(_facts("timm"), execution_environment={})
    _validate_offline_pinning(None, execution_environment={})
    _validate_offline_pinning({"implementation": "not-a-mapping"}, execution_environment={})


def test_the_gate_is_wired_into_validate_author_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full validator refuses pre-schema when the guarantee is missing."""

    stripped = {
        name: value for name, value in EXECUTION_OFFLINE_ENV.items() if name != "YOLO_OFFLINE"
    }
    monkeypatch.setattr(proposal_module, "EXECUTION_OFFLINE_ENV", stripped)
    proposal = make_author_proposal()
    proposal["proposed_facts"]["implementation"] = _facts("ultralytics")["implementation"]
    with pytest.raises(ProposalValidationError, match="YOLO_OFFLINE"):
        validate_author_proposal(proposal, allowed_model_dir=tmp_path, source_manifest=[])
