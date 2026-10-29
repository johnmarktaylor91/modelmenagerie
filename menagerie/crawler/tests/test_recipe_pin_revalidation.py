"""Admission-time recipe-revision pin revalidation (rung-11 defect class 4)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler.driver_receipts import _worker_request
from menagerie.crawler.gates import revalidate_recipe_revision_pin
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes, stable_hash
from menagerie.crawler.recipe import RecipeError

_SOURCE_IDENTITY = "sha256:" + "a" * 64
_ADAPTER_BYTES = b"def build_model():\n    return None\n"
_ADAPTER_NAME = "pit_adapter.py"


def _typed_proposal(adapter_bytes: bytes = _ADAPTER_BYTES) -> dict[str, Any]:
    """Build a minimal coherent typed-adapter proposal for the hook."""

    implementation: dict[str, Any] = {
        "recipe_type": "typed-adapter",
        "code_path": _ADAPTER_NAME,
        "code_sha256": hash_bytes(adapter_bytes),
        "library_recipe": None,
    }
    input_contract = {"args": [], "kwargs": [], "seed": 0}
    facts = {
        "implementation": implementation,
        "input_contract": input_contract,
        "modes": {"meaningful_modes": ["eval"]},
    }
    pin = compute_recipe_revision(
        {
            "implementation": dict(implementation),
            "input_contract": input_contract,
            "modes": {"meaningful_modes": ["eval"]},
        },
        _SOURCE_IDENTITY,
    )
    implementation["recipe_revision"] = pin
    return {
        "stable_id": "m_example",
        "source_identity": _SOURCE_IDENTITY,
        "recipe_revision": pin,
        "proposed_facts": facts,
    }


def _declarative_proposal() -> dict[str, Any]:
    """Build a minimal coherent declarative proposal for the hook."""

    implementation: dict[str, Any] = {
        "recipe_type": "declarative-library",
        "library_recipe": {
            "distribution": "torch",
            "version": "2.0",
            "module": "torch.nn",
            "symbol": "Linear",
            "kwargs": {"in_features": 4, "out_features": 2},
            "pretrained_disable_fields": [],
        },
    }
    input_contract = {"args": [], "kwargs": [], "seed": 0}
    facts = {
        "implementation": implementation,
        "input_contract": input_contract,
        "modes": {"meaningful_modes": ["eval"]},
    }
    pin = compute_recipe_revision(
        {
            "implementation": dict(implementation),
            "input_contract": input_contract,
            "modes": {"meaningful_modes": ["eval"]},
        },
        _SOURCE_IDENTITY,
    )
    implementation["recipe_revision"] = pin
    return {
        "stable_id": "m_example",
        "source_identity": _SOURCE_IDENTITY,
        "recipe_revision": pin,
        "proposed_facts": facts,
    }


@pytest.mark.smoke
def test_coherent_typed_pin_returns_the_loader_flavor_revision() -> None:
    """A coherent typed proposal yields the exact revision the loader recomputes."""

    proposal = _typed_proposal()
    derived = revalidate_recipe_revision_pin(
        proposal, adapter_bytes=_ADAPTER_BYTES, adapter_path_name=_ADAPTER_NAME
    )
    assert derived == compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": _ADAPTER_NAME},
        _SOURCE_IDENTITY,
        adapter_bytes=_ADAPTER_BYTES,
    )


@pytest.mark.smoke
def test_stale_declared_pin_is_refused_at_admission() -> None:
    """A pin copied from a prior attempt (re-frozen sources) refuses typed.

    This is the rung-8 m538 class the rung-11 census asked to catch at
    admission: the declared ``recipe_revision`` does not reproduce from the
    proposal's OWN components.
    """

    proposal = _typed_proposal()
    proposal["recipe_revision"] = "sha256:" + "0" * 64
    with pytest.raises(RecipeError, match="stale for the proposal's own components"):
        revalidate_recipe_revision_pin(
            proposal, adapter_bytes=_ADAPTER_BYTES, adapter_path_name=_ADAPTER_NAME
        )


@pytest.mark.smoke
def test_stale_embedded_pin_is_refused_at_admission() -> None:
    """The embedded implementation.recipe_revision must reproduce too."""

    proposal = _typed_proposal()
    proposal["proposed_facts"]["implementation"]["recipe_revision"] = "sha256:" + "1" * 64
    with pytest.raises(RecipeError, match="stale for the proposal's own components"):
        revalidate_recipe_revision_pin(
            proposal, adapter_bytes=_ADAPTER_BYTES, adapter_path_name=_ADAPTER_NAME
        )


@pytest.mark.smoke
def test_drifted_staged_adapter_bytes_are_refused() -> None:
    """Staged bytes that no longer reproduce the accepted digest refuse."""

    proposal = _typed_proposal()
    with pytest.raises(RecipeError, match="staged adapter bytes do not reproduce"):
        revalidate_recipe_revision_pin(
            proposal,
            adapter_bytes=b"tampered = True\n",
            adapter_path_name=_ADAPTER_NAME,
        )


@pytest.mark.smoke
def test_typed_revalidation_requires_the_staged_bytes() -> None:
    """A typed proposal without its staged bytes cannot be revalidated."""

    with pytest.raises(RecipeError, match="requires the staged adapter bytes"):
        revalidate_recipe_revision_pin(_typed_proposal())


@pytest.mark.smoke
def test_declarative_revalidation_accepts_and_returns_none() -> None:
    """A coherent declarative proposal revalidates with no adapter revision."""

    assert revalidate_recipe_revision_pin(_declarative_proposal()) is None


@pytest.mark.smoke
def test_declarative_revalidation_refuses_adapter_bytes() -> None:
    """Declarative recipes stage no adapter bytes; supplying them is a fault."""

    with pytest.raises(RecipeError, match="stage no adapter bytes"):
        revalidate_recipe_revision_pin(
            _declarative_proposal(),
            adapter_bytes=_ADAPTER_BYTES,
            adapter_path_name=_ADAPTER_NAME,
        )


@pytest.mark.smoke
def test_stale_declarative_pin_is_refused_at_admission() -> None:
    """The staleness tripwire covers the declarative lane identically."""

    proposal = _declarative_proposal()
    proposal["recipe_revision"] = "sha256:" + "2" * 64
    with pytest.raises(RecipeError, match="stale for the proposal's own components"):
        revalidate_recipe_revision_pin(proposal)


def _worker_request_fixture(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, Any]]:
    """Stage a minimal typed-adapter artifact accepted enough to build a request."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    adapter = model_dir / _ADAPTER_NAME
    adapter.write_bytes(_ADAPTER_BYTES)
    implementation: dict[str, Any] = {
        "recipe_type": "typed-adapter",
        "code_path": _ADAPTER_NAME,
        "code_sha256": hash_bytes(_ADAPTER_BYTES),
        "code_manifest": [{"path": _ADAPTER_NAME, "sha256": hash_bytes(_ADAPTER_BYTES)}],
        "library_recipe": None,
        "device_policy": "cpu",
        "run_framework": "pytorch",
    }
    input_contract: dict[str, Any] = {
        "builder_symbol": "build_model",
        "non_tensor_values": [],
        "args": [],
        "kwargs": [],
        "seed": 0,
    }
    facts = {
        "implementation": implementation,
        "input_contract": input_contract,
        "modes": {"meaningful_modes": ["eval"]},
        "external_metadata": {"modality": ["vision"]},
    }
    pin = compute_recipe_revision(
        {
            "implementation": dict(implementation),
            "input_contract": input_contract,
            "modes": {"meaningful_modes": ["eval"]},
        },
        _SOURCE_IDENTITY,
    )
    implementation["recipe_revision"] = pin
    proposal = {
        "stable_id": "m_example",
        "work_id": "work-m_example",
        "source_identity": _SOURCE_IDENTITY,
        "recipe_revision": pin,
        "proposed_facts": facts,
    }
    artifact = SimpleNamespace(proposal=proposal, model_dir=model_dir)
    manifest = SimpleNamespace(
        manifest_id=stable_hash("manifest"), code_manifest_identity=stable_hash("code")
    )
    return artifact, {
        "scratch_root": tmp_path / "scratch",
        "receipt_path": tmp_path / "scratch" / "result" / "receipt.json",
        "execution_identity": stable_hash("execution"),
        "execution_manifest": manifest,
    }


@pytest.mark.smoke
def test_worker_request_binds_the_loader_flavor_pin(tmp_path: Path) -> None:
    """The typed-adapter request carries the admission-derived adapter revision."""

    artifact, extras = _worker_request_fixture(tmp_path)
    request = _worker_request(
        artifact,  # type: ignore[arg-type]
        extras["scratch_root"],
        extras["receipt_path"],
        extras["execution_identity"],
        extras["execution_manifest"],
        0,
        "eval",
    )
    assert request["recipe"]["recipe_revision"] == compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": _ADAPTER_NAME},
        _SOURCE_IDENTITY,
        adapter_bytes=_ADAPTER_BYTES,
    )


@pytest.mark.smoke
def test_worker_request_refuses_a_stale_pin_before_any_worker_spawns(
    tmp_path: Path,
) -> None:
    """Admission refuses the stale pin typed; no worker ever observes it."""

    artifact, extras = _worker_request_fixture(tmp_path)
    artifact.proposal["recipe_revision"] = "sha256:" + "3" * 64
    with pytest.raises(RecipeError, match="stale for the proposal's own components"):
        _worker_request(
            artifact,  # type: ignore[arg-type]
            extras["scratch_root"],
            extras["receipt_path"],
            extras["execution_identity"],
            extras["execution_manifest"],
            0,
            "eval",
        )


@pytest.mark.smoke
def test_worker_request_refuses_drifted_staged_bytes(tmp_path: Path) -> None:
    """Post-acceptance staged-byte drift refuses typed at request build."""

    artifact, extras = _worker_request_fixture(tmp_path)
    (artifact.model_dir / _ADAPTER_NAME).write_bytes(b"tampered = True\n")
    with pytest.raises(RecipeError, match="staged adapter bytes do not reproduce"):
        _worker_request(
            artifact,  # type: ignore[arg-type]
            extras["scratch_root"],
            extras["receipt_path"],
            extras["execution_identity"],
            extras["execution_manifest"],
            0,
            "eval",
        )
