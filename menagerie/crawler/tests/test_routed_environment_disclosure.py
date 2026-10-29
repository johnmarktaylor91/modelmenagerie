"""The R1 recipe pin is resolved against an inventory the author is now shown.

``recipe.resolve_environment_artifact_digest`` refuses a declarative recipe whose
``distribution`` no routed package row provides, and refuses one whose ``version``
contradicts the installed row. Both refusals are correct: a permanent R1 record
that pins an artifact the environment does not install has nothing behind it, and
one that pins a version the environment does not install describes different code
than the code that will run.

What was NOT correct is that the author had no channel to the inventory those two
rules test against. The envelope carried the routed intent's name nowhere and its
package rows nowhere, so ``library_recipe.distribution``/``version`` were authored
blind and a fully correct author could still be refused -- an unsatisfiable rule
rather than a check. The rung-3 census failed four models on the first form
(``segmentation-models-pytorch`` twice, ``segmentation_models_pytorch`` twice) and
two on the second (``transformers`` ``4.57.1`` against an installed ``5.14.1``),
each a permanent dead record for a model the authoring stage visits exactly once.

These tests pin the repair in the failing direction, against the REAL committed
resolved export rather than a hand-built fixture:

- the resolution itself is unchanged and still refuses every one of those cases;
- the two ``segmentation*models*pytorch`` spellings were ALREADY equivalent under
  the namespace bridge, so the census's "two spellings" was one defect and not a
  normalization gap -- normalization is now the exact PEP 503 rule anyway, which
  cannot alias two distinct distributions;
- the envelope now discloses the routed intent and its exact inventory, in the
  distribution namespace the recipe is written in, WITHOUT the machine-derived
  artifact digest;
- an unlocked route is disclosed as "no inventory held", not as an empty one.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorResultMalformedError,
    _validate_routed_distribution,
    build_author_envelope,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import AUTHOR_PROMPT_NAME, CHECKER_PROMPT_NAME
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.driver_admission import CommandAuthorLane
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.package_namespace import (
    canonical_distribution_name,
    inventory_disclosure,
    inventory_row_provides_distribution,
)
from menagerie.crawler.recipe import RecipeError, bind_library_artifact_digest

_LOCK_DIR = Path(__file__).resolve().parents[1] / "envs" / "locks"
_RESOLVED_EXPORT = _LOCK_DIR / "round19-osx-arm64.resolved.json"


def _real_inventory() -> tuple[Mapping[str, Any], ...]:
    """Return the committed release lock's exact resolved-export package rows.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Real ``name``/``version``/``sha256`` rows, the same shape
        ``driver_admission._routed_environment_packages`` hands the resolver.
    """

    packages = json.loads(_RESOLVED_EXPORT.read_text(encoding="utf-8"))["packages"]
    return tuple(row for row in packages if isinstance(row, Mapping))


def _bind(distribution: str, version: str) -> str:
    """Resolve one declarative recipe against the real inventory.

    Parameters
    ----------
    distribution, version:
        Declared recipe pin, in the Python distribution namespace.

    Returns
    -------
    str
        The bound machine-derived artifact digest.
    """

    implementation: dict[str, Any] = {
        "recipe_type": "declarative-library",
        "library_recipe": {"distribution": distribution, "version": version},
    }
    bind_library_artifact_digest(implementation, _real_inventory())
    return str(implementation["library_recipe"]["artifact_sha256"])


# -- the resolution is unchanged and still refuses ------------------------------


@pytest.mark.parametrize(
    "distribution",
    ["segmentation-models-pytorch", "segmentation_models_pytorch", "dgl"],
)
def test_a_distribution_the_environment_does_not_install_is_still_refused(
    distribution: str,
) -> None:
    """The tripwire stays armed: an absent distribution has no honest digest."""

    with pytest.raises(RecipeError, match="does not install distribution"):
        _bind(distribution, "0.5.0")


def test_the_two_census_spellings_were_already_one_case() -> None:
    """Both spellings normalize identically, so the census showed one defect twice.

    The rung-3 diagnostics carried the refusal under both ``-`` and ``_``
    spellings, which reads like a normalization gap. It is not: the bridge has
    always case-folded and mapped ``_``/``.`` to ``-``, so the two names were the
    same lookup and failed for the same real reason -- the routed environment
    genuinely installs no such package.
    """

    assert canonical_distribution_name("segmentation_models_pytorch") == (
        canonical_distribution_name("segmentation-models-pytorch")
    )
    # PEP 503 collapses RUNS too, which the previous rule did not.
    assert canonical_distribution_name("Segmentation__Models.Pytorch") == (
        "segmentation-models-pytorch"
    )


def test_a_version_the_environment_does_not_install_is_still_refused() -> None:
    """A recipe pinned to un-installed code still refuses, and names both versions."""

    with pytest.raises(RecipeError, match="but the routed environment installs '2.5.1'"):
        _bind("torch", "2.11.0")


def test_the_correct_pin_against_the_same_real_inventory_resolves() -> None:
    """The satisfiable case passes, across the divergent-name bridge."""

    digest = _bind("torch", "2.5.1")
    assert digest.startswith("sha256:")
    row = next(entry for entry in _real_inventory() if entry["name"] == "pytorch")
    assert digest == row["sha256"]


# -- the disclosure that makes the rule satisfiable -----------------------------


def test_the_disclosure_publishes_versions_and_provisions_but_never_the_digest() -> None:
    """The author learns what to pin; it never learns the digest it must not supply."""

    rows = inventory_disclosure(_real_inventory())
    assert rows, "the committed release lock is not empty"
    assert all(set(row) == {"name", "version", "provides"} for row in rows)
    serialized = json.dumps(rows)
    for entry in _real_inventory():
        assert str(entry["sha256"]) not in serialized

    by_name = {str(row["name"]): row for row in rows}
    # The two namespaces diverge exactly here, and the disclosure spells both.
    assert by_name["pytorch"]["version"] == "2.5.1"
    assert by_name["pytorch"]["provides"] == ["torch"]
    # A package whose spellings agree provides only itself.
    assert by_name["numpy"]["provides"] == ["numpy"]
    # Nothing in this environment provides what the census models needed.
    provided = {name for row in rows for name in row["provides"]}
    assert "segmentation-models-pytorch" not in provided
    assert "dgl" not in provided


def test_the_disclosure_is_stable_across_row_order() -> None:
    """A rebuilt envelope discloses the same bytes for the same environment."""

    forward = inventory_disclosure(_real_inventory())
    reversed_rows = inventory_disclosure(tuple(reversed(_real_inventory())))
    assert forward == reversed_rows


def test_malformed_rows_are_dropped_rather_than_disclosed_as_facts() -> None:
    """A row without an exact name and version states nothing and is not published."""

    malformed: list[Any] = [
        {"name": "timm", "version": "1.0.28", "sha256": "sha256:" + "a" * 64},
        {"name": "broken"},
        {"version": "1.0"},
        {"name": 7, "version": "1.0"},
        "not-a-row",
    ]
    rows = inventory_disclosure(malformed)
    assert [row["name"] for row in rows] == ["timm"]


# -- the envelope carries it ----------------------------------------------------


def _prompt_hash(name: str) -> str:
    """Hash one exact shipped prompt's bytes.

    Parameters
    ----------
    name:
        Prompt basename without its ``.txt`` suffix.

    Returns
    -------
    str
        Canonical prompt identity.
    """

    path = Path(__file__).resolve().parents[1] / "prompts" / f"{name}.txt"
    return hash_bytes(path.read_bytes())


def _context(stable_id: str) -> AuthorityContext:
    """Return a frozen authority context whose author prompt matches shipped bytes.

    Parameters
    ----------
    stable_id:
        Model identity the envelope will bind.

    Returns
    -------
    AuthorityContext
        Context accepted by the production envelope builder.
    """

    author_fields = {
        "provider": "anthropic",
        "model": "claude-sonnet",
        "version": "current",
        "prompt_sha256": _prompt_hash(AUTHOR_PROMPT_NAME),
    }
    checker_fields = {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "version": "current",
        "prompt_sha256": _prompt_hash(CHECKER_PROMPT_NAME),
    }
    return AuthorityContext(
        active_intake_snapshot_id="intake-1",
        active_intake_snapshot_sha256="sha256:" + "1" * 64,
        intake_by_stable_id={stable_id: {"stable_id": stable_id, "variant": "base"}},
        family_bindings={},
        author_prompt_identity=author_fields["prompt_sha256"],
        author_model_identity=stable_hash(author_fields),
        author_schema_identity="sha256:" + "6" * 64,
        author_dispatcher_identity="sha256:" + "2" * 64,
        author_model_fields=author_fields,
        checker_prompt_identity=checker_fields["prompt_sha256"],
        checker_model_identity=stable_hash(checker_fields),
        checker_schema_identity="sha256:" + "9" * 64,
        checker_model_fields=checker_fields,
        environment_generations={},
        reducer_policy_identity="sha256:" + "a" * 64,
        runner_policy_identity="sha256:" + "b" * 64,
        terminal_policy_identity="sha256:" + "c" * 64,
        publication_policy_identity="sha256:" + "d" * 64,
    )


def _envelope(tmp_path: Path, **routed: Any) -> dict[str, Any]:
    """Build one real author envelope through the production builder.

    Parameters
    ----------
    tmp_path:
        Per-test sandbox for the model dir and result path.
    routed:
        Routed-environment keyword arguments under test.

    Returns
    -------
    dict[str, Any]
        Hash-bound author envelope.
    """

    return build_author_envelope(
        context=_context("m_example"),
        work_id="work-m_example",
        stable_id="m_example",
        campaign_id="c1-mech",
        created_at="2026-08-04T00:00:00Z",
        untrusted_hints={},
        source_manifest={"sources": []},
        allowed_model_dir=tmp_path,
        output_path=tmp_path / "out" / "result.json",
        **routed,
    )


def test_the_envelope_discloses_the_routed_inventory(tmp_path: Path) -> None:
    """The fact the R1 pin is resolved against now reaches the author that writes it."""

    envelope = _envelope(
        tmp_path,
        routed_environment_intent="core",
        routed_environment_packages=_real_inventory(),
    )
    routed = envelope["routed_environment"]
    assert routed["intent"] == "core"
    assert routed["inventory_disclosed"] is True
    assert routed["packages"] == [dict(row) for row in inventory_disclosure(_real_inventory())]
    assert routed["legal_distributions"] == sorted(
        {name for row in routed["packages"] for name in row["provides"]}
    )
    # The envelope self-hash still binds the body it now carries.
    assert envelope["envelope_sha256"] == stable_hash(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )


def test_the_driver_binds_its_exact_registry_into_the_author_lane() -> None:
    """The production lane no longer builds every request from an unset registry."""

    registry = object()
    lane = CommandAuthorLane(("true",))
    CrawlerDriver(
        paths=SimpleNamespace(),
        config=SimpleNamespace(),
        dependencies=SimpleNamespace(author=lane),
        registry=registry,
    )
    assert lane.registry is registry


def test_an_unlocked_route_is_disclosed_as_no_inventory_held(tmp_path: Path) -> None:
    """An unlocked target says so, instead of claiming an environment installs nothing."""

    envelope = _envelope(tmp_path, routed_environment_intent="graph")
    routed = envelope["routed_environment"]
    assert routed == {
        "intent": "graph",
        "inventory_disclosed": False,
        "legal_distributions": [],
        "packages": [],
    }


def test_the_disclosure_never_leaks_the_machine_derived_digest(tmp_path: Path) -> None:
    """No artifact digest is reachable from the envelope the author reads."""

    envelope = _envelope(
        tmp_path,
        routed_environment_intent="core",
        routed_environment_packages=_real_inventory(),
    )
    serialized = json.dumps(envelope["routed_environment"], sort_keys=True)
    for row in _real_inventory():
        assert str(row["sha256"]) not in serialized


def test_the_disclosure_is_a_copy_and_cannot_alias_the_registry(tmp_path: Path) -> None:
    """Mutating the built envelope cannot reach back into the routed inventory."""

    inventory = [dict(row) for row in _real_inventory()]
    envelope = _envelope(
        tmp_path,
        routed_environment_intent="core",
        routed_environment_packages=inventory,
    )
    before = deepcopy(inventory)
    envelope["routed_environment"]["packages"][0]["version"] = "0.0.0"
    assert inventory == before


@pytest.mark.parametrize(
    ("stable_id", "legal_distribution"),
    [
        ("m8245", "timm"),
        ("m8867", "timm"),
        ("m9577", "segmentation-models-pytorch"),
        ("m9666", "segmentation-models-pytorch"),
        ("m9819", "segmentation-models-pytorch"),
    ],
)
def test_rung9b_pypi_recipes_are_refused_with_their_legal_repair(
    stable_id: str, legal_distribution: str
) -> None:
    """The five frozen rung-9b guesses cannot bypass the disclosed vocabulary."""

    proposal: dict[str, Any] = {
        "proposed_facts": {"implementation": {"library_recipe": {"distribution": "pypi"}}}
    }
    envelope = {
        "routed_environment": {
            "intent": "core",
            "inventory_disclosed": True,
            "legal_distributions": [legal_distribution],
            "packages": [
                {
                    "name": legal_distribution,
                    "version": "1.0.28" if legal_distribution == "timm" else "0.5.0",
                    "provides": [legal_distribution],
                }
            ],
        }
    }
    with pytest.raises(
        AuthorResultMalformedError,
        match=r"Package indexes such as 'pypi' and 'conda' are not distributions",
    ) as caught:
        _validate_routed_distribution(proposal, envelope)
    assert stable_id
    assert legal_distribution in str(caught.value)

    proposal["proposed_facts"]["implementation"]["library_recipe"]["distribution"] = (
        legal_distribution
    )
    _validate_routed_distribution(proposal, envelope)


# -- LIB-VERSION-DRIFT (m8189): the version half of the same pre-publication gate


def _versioned_envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a disclosed routed envelope carrying exactly ``rows``.

    Parameters
    ----------
    rows:
        Disclosure-shaped ``name``/``version``/``provides`` package rows.
    """

    return {
        "routed_environment": {
            "intent": "core",
            "inventory_disclosed": True,
            "legal_distributions": sorted({name for row in rows for name in row["provides"]}),
            "packages": rows,
        }
    }


def _timm_proposal(version: str) -> dict[str, Any]:
    """Return the m8189 shape: a timm recipe pinned to ``version``."""

    return {
        "proposed_facts": {
            "implementation": {"library_recipe": {"distribution": "timm", "version": version}}
        }
    }


def test_the_m8189_version_drift_is_refused_naming_both_versions() -> None:
    """A pin against the paper-era version dies pre-publication, not at forward."""

    envelope = _versioned_envelope([{"name": "timm", "version": "1.0.28", "provides": ["timm"]}])
    with pytest.raises(AuthorResultMalformedError) as caught:
        _validate_routed_distribution(_timm_proposal("0.6.13"), envelope)
    message = str(caught.value)
    assert "'0.6.13'" in message
    assert "'1.0.28'" in message
    assert "library_recipe.version" in message


def test_the_installed_version_pin_passes_the_same_gate() -> None:
    """Pinning the disclosed installed version is exactly what the gate demands."""

    envelope = _versioned_envelope([{"name": "timm", "version": "1.0.28", "provides": ["timm"]}])
    _validate_routed_distribution(_timm_proposal("1.0.28"), envelope)


def test_an_ambiguous_inventory_version_is_left_to_the_binding_authority() -> None:
    """Two rows providing one distribution stay the resolver's ambiguity, not ours."""

    envelope = _versioned_envelope(
        [
            {"name": "timm", "version": "1.0.28", "provides": ["timm"]},
            {"name": "timm", "version": "0.9.0", "provides": ["timm"]},
        ]
    )
    _validate_routed_distribution(_timm_proposal("0.6.13"), envelope)


def test_the_version_gate_never_disagrees_with_the_binding_refusal() -> None:
    """Every pre-publication version refusal reproduces on the real inventory bind."""

    inventory = _real_inventory()
    torch_rows = [
        row
        for row in inventory
        if inventory_row_provides_distribution(str(row.get("name")), "torch")
    ]
    assert torch_rows, "release inventory must install torch"
    installed = str(torch_rows[0]["version"])
    drifted = installed + ".post999"
    envelope = _versioned_envelope([dict(row) for row in inventory_disclosure(tuple(torch_rows))])
    proposal = {
        "proposed_facts": {
            "implementation": {"library_recipe": {"distribution": "torch", "version": drifted}}
        }
    }
    with pytest.raises(AuthorResultMalformedError):
        _validate_routed_distribution(proposal, envelope)
    with pytest.raises(RecipeError, match="environment installs"):
        _bind("torch", drifted)


# ---------------------------------------------------------------------------
# Item 13 (F51 DECIDED_PLAN; author R-3, section 4): stage 1 carries the SAME
# routed_environment object stage 2 gets, plus a host-capacity hint, so the
# author pins the installed version up front (432 of 573 supplement requests
# were a forced installed-version re-fetch) and can size to the host (6 of 52
# R6b members died `needs-more-memory` on tiny configs).
# ---------------------------------------------------------------------------


def test_the_stage1_routed_block_is_byte_identical_to_the_stage2_envelopes(
    tmp_path: Path,
) -> None:
    """The helper stage 1 uses produces the exact object stage 2 embeds."""

    from menagerie.crawler.author_dispatch import routed_environment_block

    inventory = _real_inventory()
    stage2 = _envelope(
        tmp_path,
        routed_environment_intent="core",
        routed_environment_packages=inventory,
    )["routed_environment"]
    stage1 = routed_environment_block("core", inventory)
    assert stage1 == stage2


def test_an_unlocked_route_block_is_no_inventory_held() -> None:
    """An unlocked/unrouted target discloses "no inventory held", never an empty one."""

    from menagerie.crawler.author_dispatch import routed_environment_block

    assert routed_environment_block("graph", ()) == {
        "intent": "graph",
        "inventory_disclosed": False,
        "legal_distributions": [],
        "packages": [],
    }


def test_host_capacity_block_is_ram_and_a_null_accelerator() -> None:
    """The host-capacity hint is exactly RAM plus a null accelerator (CPU-only crawl)."""

    from menagerie.crawler.author_dispatch import host_capacity_block

    capacity = host_capacity_block()
    assert set(capacity) == {"ram_gib", "accelerator"}
    assert capacity["accelerator"] is None
    assert capacity["ram_gib"] is None or (
        isinstance(capacity["ram_gib"], float) and capacity["ram_gib"] > 0.0
    )
